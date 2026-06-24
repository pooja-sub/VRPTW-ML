import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data
except Exception:
    MessagePassing = None
    Data = None


@dataclass
class PruningConfig:
    enabled: bool = True
    model_path: Optional[str] = None
    threshold: float = 0.5
    keep_ratio: float = 0.25
    top_k_per_node: int = 8


class GraphSAGELayer(MessagePassing if MessagePassing is not None else object):
    def __init__(self, in_channels, out_channels):
        if MessagePassing is None:
            raise ImportError("torch_geometric is required for GraphSAGELayer")
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        neigh = self.propagate(edge_index, x=x)
        return self.lin_self(x) + self.lin_neigh(neigh)

    def message(self, x_j):
        return x_j


class EdgeScoringGNN(nn.Module if nn is not None else object):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        if nn is None:
            raise ImportError("PyTorch is required for EdgeScoringGNN")
        super().__init__()
        self.input = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(max(1, num_layers)))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.input(x))
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_mlp(edge_input).squeeze(-1)


def _node_features(params, dual_pi: Optional[Sequence[float]] = None):
    node_count = params.nbclients + 1
    features = []
    for node in range(node_count):
        dual_value = 0.0
        if dual_pi is not None and 1 <= node < params.nbclients:
            dual_value = float(dual_pi[node - 1])
        features.append([
            float(params.posx[node]),
            float(params.posy[node]),
            float(params.d[node]),
            float(params.a[node]),
            float(params.b[node]),
            float(params.s[node]),
            dual_value,
        ])
    return np.asarray(features, dtype=float)


def _edge_features(params, arc: Tuple[int, int], dual_pi: Optional[Sequence[float]] = None):
    i, j = arc
    reduced_cost_hint = float(params.dist[i][j])
    if dual_pi is not None and 1 <= j < params.nbclients:
        reduced_cost_hint -= float(dual_pi[j - 1])
    return [
        float(params.dist[i][j]),
        float(params.ttime[i][j]),
        reduced_cost_hint,
        float(params.a[i]),
        float(params.b[i]),
        float(params.a[j]),
        float(params.b[j]),
    ]


def _expanded_forward_path(node) -> Sequence[int]:
    if not isinstance(node, tuple) or not node:
        return []
    destination = node[0]
    if destination == 0:
        return [0]
    predecessors = [value for value in node[1:] if isinstance(value, (int, np.integer)) and value > 0]
    return list(reversed(predecessors)) + [destination]


def _simulate_elapsed_time(params, forward_path: Sequence[int]):
    if not forward_path:
        return 0.0
    if len(forward_path) == 1:
        return float(params.a[forward_path[0]])

    current_time = float(params.a[forward_path[0]])
    for previous, current in zip(forward_path[:-1], forward_path[1:]):
        current_time += float(params.s[previous]) + float(params.ttime[previous][current])
        if current_time < float(params.a[current]):
            current_time = float(params.a[current])
    return current_time


def _expanded_node_features(params, node, node_data, dual_pi: Optional[Sequence[float]] = None):
    forward_path = _expanded_forward_path(node)
    destination = forward_path[-1] if forward_path else 0

    accumulated_demand = 0.0
    for customer in forward_path:
        if 1 <= customer < params.nbclients:
            accumulated_demand += float(params.d[customer])

    elapsed_time = _simulate_elapsed_time(params, forward_path) if forward_path else 0.0
    remaining_capacity = float(params.capacity) - accumulated_demand
    time_slack = float(params.b[destination]) - elapsed_time if 0 <= destination < len(params.b) else 0.0

    if 0 <= destination < len(params.posx):
        posx = float(params.posx[destination])
        posy = float(params.posy[destination])
        demand = float(params.d[destination])
    else:
        posx = 0.0
        posy = 0.0
        demand = 0.0

    return [
        posx,
        posy,
        demand,
        accumulated_demand,
        elapsed_time,
        remaining_capacity,
        time_slack,
    ]


def _expanded_edge_features(params, source_node, target_node, edge_data, dual_pi: Optional[Sequence[float]] = None):
    source_path = _expanded_forward_path(source_node)
    target_path = _expanded_forward_path(target_node)

    source_city = source_path[-1] if source_path else 0
    target_city = target_path[-1] if target_path else 0

    parent_arc = edge_data.get("parent_arc")
    if parent_arc and isinstance(parent_arc, tuple) and len(parent_arc) == 2:
        source_city, target_city = parent_arc

    distance = float(params.dist[source_city][target_city])
    travel_time = float(params.ttime[source_city][target_city])
    direct_rc = distance
    if dual_pi is not None and 1 <= target_city < params.nbclients:
        direct_rc -= float(dual_pi[target_city - 1])

    source_accumulated = 0.0
    source_elapsed = 0.0
    source_remaining = float(params.capacity)
    if source_path:
        source_accumulated = sum(
            float(params.d[customer]) for customer in source_path if 1 <= customer < params.nbclients
        )
        source_elapsed = _simulate_elapsed_time(params, source_path)
        source_remaining = float(params.capacity) - source_accumulated

    demand_after = source_accumulated + (float(params.d[target_city]) if 1 <= target_city < params.nbclients else 0.0)
    can_fit_demand = 1.0 if demand_after <= float(params.capacity) + 1e-9 else 0.0

    arrival_time = source_elapsed + float(params.s[source_city]) + travel_time
    if arrival_time < float(params.a[target_city]):
        arrival_time = float(params.a[target_city])
    time_feasible = 1.0 if arrival_time <= float(params.b[target_city]) + 1e-9 else 0.0
    slack_at_target = max(0.0, float(params.b[target_city]) - arrival_time)

    if dual_pi is not None and 1 <= target_city < params.nbclients:
        max_dual = max(1e-9, max(abs(float(value)) for value in dual_pi))
        dual_contribution = abs(float(dual_pi[target_city - 1])) / max_dual
    else:
        dual_contribution = 0.0

    is_mandatory = 1.0 if source_city == 0 or target_city == params.nbclients or edge_data.get("edge_type") == "sink_label_match" else 0.0

    return [
        distance,
        travel_time,
        direct_rc,
        can_fit_demand,
        time_feasible,
        slack_at_target,
        is_mandatory,
    ]


class GuidedArcPruner:
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        node_dim: int = 7,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        self.config = config or PruningConfig()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._model = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        self._loaded = True

        if torch is None or nn is None or Data is None:
            return

        model_path = self.config.model_path
        if not model_path:
            here = os.path.dirname(__file__)
            candidates = [
                os.path.join(here, "MLModels", "gnn.pt"),
                os.path.join(here, "MLModels", "gnn.pth"),
                os.path.join(here, "MLModels", "gnn.joblib"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if not model_path or not os.path.exists(model_path):
            return

        model = EdgeScoringGNN(self.node_dim, self.edge_dim, self.hidden_dim, self.num_layers)
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        self._model = model.eval()

    def _heuristic_scores(self, params, arcs: Sequence[Tuple[int, int]], dual_pi=None):
        scores = []
        for i, j in arcs:
            reduced_cost = float(params.dist[i][j])
            if dual_pi is not None and 1 <= j < params.nbclients:
                reduced_cost -= float(dual_pi[j - 1])
            tw_slack = max(0.0, float(params.b[j] - params.a[i]))
            demand_penalty = float(params.d[i] + params.d[j]) / max(1.0, float(params.capacity))
            score = 1.0 / (1.0 + np.exp(0.05 * (reduced_cost - tw_slack)))
            score = 0.65 * score + 0.35 * (1.0 - min(1.0, demand_penalty))
            scores.append(float(score))
        return np.asarray(scores, dtype=float)

    def score_arcs(self, params, arcs: Iterable[Tuple[int, int]], dual_pi=None):
        arcs = list(dict.fromkeys(arcs))
        if not arcs:
            return [], np.asarray([], dtype=float)

        self._load_model()
        if self._model is None or torch is None or Data is None:
            return arcs, self._heuristic_scores(params, arcs, dual_pi=dual_pi)

        node_x = torch.tensor(_node_features(params, dual_pi=dual_pi), dtype=torch.float32)
        edge_index = torch.tensor(np.asarray(arcs, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(
            np.asarray([_edge_features(params, arc, dual_pi=dual_pi) for arc in arcs], dtype=float),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = self._model(node_x, edge_index, edge_attr)
            scores = torch.sigmoid(logits).cpu().numpy()
        return arcs, scores

    def prune_arcs(
        self,
        params,
        arcs: Iterable[Tuple[int, int]],
        dual_pi=None,
        threshold: Optional[float] = None,
        keep_ratio: Optional[float] = None,
        top_k_per_node: Optional[int] = None,
    ):
        if not self.config.enabled:
            return set(arcs)

        arcs = list(dict.fromkeys(arcs))
        if not arcs:
            return set()

        threshold = self.config.threshold if threshold is None else threshold
        keep_ratio = self.config.keep_ratio if keep_ratio is None else keep_ratio
        top_k_per_node = self.config.top_k_per_node if top_k_per_node is None else top_k_per_node

        scored_arcs, scores = self.score_arcs(params, arcs, dual_pi=dual_pi)
        keep = set()

        # Always preserve the depot arcs and any explicitly mandatory arcs.
        nb_last = params.nbclients
        for arc, score in zip(scored_arcs, scores):
            i, j = arc
            if i == 0 or j == nb_last:
                keep.add(arc)

        if not scored_arcs:
            return keep

        if keep_ratio is not None and keep_ratio > 0:
            keep_target = max(len(keep), int(np.ceil(len(scored_arcs) * float(keep_ratio))))
            ranked = sorted(zip(scored_arcs, scores), key=lambda item: item[1], reverse=True)
            for arc, score in ranked[:keep_target]:
                if score >= threshold:
                    keep.add(arc)
                else:
                    keep.add(arc)
        else:
            for arc, score in zip(scored_arcs, scores):
                if score >= threshold:
                    keep.add(arc)

        if top_k_per_node is not None and top_k_per_node > 0:
            grouped = {}
            for arc, score in zip(scored_arcs, scores):
                grouped.setdefault(arc[0], []).append((arc, score))
            for _, ranked in grouped.items():
                ranked.sort(key=lambda item: item[1], reverse=True)
                for arc, _ in ranked[:top_k_per_node]:
                    keep.add(arc)

        if not keep:
            ranked = sorted(zip(scored_arcs, scores), key=lambda item: item[1], reverse=True)
            for arc, _ in ranked[: max(1, len(ranked) // 4)]:
                keep.add(arc)

        # Preserve feasibility by retaining all depot arcs.
        for k in range(1, nb_last):
            keep.add((0, k))
            keep.add((k, nb_last))

        return keep

    def score_expanded_graph(self, params, expanded_graph, dual_pi=None):
        if expanded_graph is None:
            return [], np.asarray([], dtype=float)

        self._load_model()

        node_list = list(expanded_graph.nodes())
        edge_items = list(expanded_graph.edges(keys=True, data=True))
        if not node_list or not edge_items:
            return edge_items, np.asarray([], dtype=float)

        if self._model is None or torch is None or Data is None:
            return edge_items, self._heuristic_expanded_scores(params, edge_items, dual_pi=dual_pi)

        node_index = {node: idx for idx, node in enumerate(node_list)}
        node_x = torch.tensor(
            np.asarray([_expanded_node_features(params, node, expanded_graph.nodes[node], dual_pi=dual_pi) for node in node_list], dtype=float),
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            np.asarray([[node_index[u], node_index[v]] for u, v, _, _ in edge_items], dtype=np.int64).T,
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            np.asarray([
                _expanded_edge_features(params, u, v, data, dual_pi=dual_pi)
                for u, v, _, data in edge_items
            ], dtype=float),
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits = self._model(node_x, edge_index, edge_attr)
            scores = torch.sigmoid(logits).cpu().numpy()
        return edge_items, scores

    def _heuristic_expanded_scores(self, params, edge_items, dual_pi=None):
        scores = []
        for u, v, _, data in edge_items:
            distance, travel_time, direct_rc, can_fit_demand, time_feasible, slack_at_target, is_mandatory = _expanded_edge_features(
                params, u, v, data, dual_pi=dual_pi
            )
            score = 1.0 / (1.0 + np.exp(0.08 * direct_rc))
            score = 0.35 * score + 0.25 * can_fit_demand + 0.25 * time_feasible
            score += 0.15 * min(1.0, slack_at_target / max(1.0, float(params.b[v[0]] if isinstance(v, tuple) and v else 1.0)))
            if is_mandatory:
                score = max(score, 0.95)
            scores.append(float(min(1.0, max(0.0, score))))
        return np.asarray(scores, dtype=float)

    def prune_expanded_graph(
        self,
        params,
        expanded_graph,
        dual_pi=None,
        threshold: Optional[float] = None,
        keep_ratio: Optional[float] = None,
        top_k_per_node: Optional[int] = None,
    ):
        if not self.config.enabled or expanded_graph is None:
            return expanded_graph.copy() if expanded_graph is not None else None

        threshold = self.config.threshold if threshold is None else threshold
        keep_ratio = self.config.keep_ratio if keep_ratio is None else keep_ratio
        top_k_per_node = self.config.top_k_per_node if top_k_per_node is None else top_k_per_node

        edge_items, scores = self.score_expanded_graph(params, expanded_graph, dual_pi=dual_pi)
        if not edge_items:
            return expanded_graph.copy()

        keep = set()
        sink = params.nbclients

        for (u, v, key, data), score in zip(edge_items, scores):
            source_city = u[0] if isinstance(u, tuple) and u else None
            target_city = v[0] if isinstance(v, tuple) and v else None
            if source_city == 0 or target_city == sink or data.get("edge_type") == "sink_label_match":
                keep.add((u, v, key))

        ranked = sorted(zip(edge_items, scores), key=lambda item: item[1], reverse=True)

        if keep_ratio is not None and keep_ratio > 0:
            keep_target = max(len(keep), int(np.ceil(len(edge_items) * float(keep_ratio))))
            for (u, v, key, _), _score in ranked[:keep_target]:
                keep.add((u, v, key))
        else:
            for (u, v, key, _), score in zip(edge_items, scores):
                if score >= threshold:
                    keep.add((u, v, key))

        if top_k_per_node is not None and top_k_per_node > 0:
            grouped = {}
            for (u, v, key, _), score in zip(edge_items, scores):
                grouped.setdefault(u, []).append(((u, v, key), score))
            for _, ranked_group in grouped.items():
                ranked_group.sort(key=lambda item: item[1], reverse=True)
                for (u, v, key), _score in ranked_group[:top_k_per_node]:
                    keep.add((u, v, key))

        if not keep:
            (u, v, key, _), _score = ranked[0]
            keep.add((u, v, key))

        pruned_graph = expanded_graph.copy()
        removable = [(u, v, key) for (u, v, key, _) in edge_items if (u, v, key) not in keep]
        if removable:
            pruned_graph.remove_edges_from(removable)

        return pruned_graph
