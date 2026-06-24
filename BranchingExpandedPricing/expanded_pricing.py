import contextlib
import io
import os
import sys
import time
from collections import defaultdict

import networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Common.route import Route

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_GRAPH_TRANSFORM_DIR = os.path.join(_ROOT_DIR, "graphTransform")
if _GRAPH_TRANSFORM_DIR not in sys.path:
    sys.path.insert(0, _GRAPH_TRANSFORM_DIR)

from graphTransformation import graphTransformation_digraph_depot  # noqa: E402  # type: ignore[import-not-found]


class _Label:
    """Label state for pricing on an expanded graph."""

    __slots__ = ("node", "time", "demand", "rc", "prev", "dominated")

    def __init__(self, node, time, demand, rc, prev=None):
        self.node = node
        self.time = time
        self.demand = demand
        self.rc = rc
        self.prev = prev
        self.dominated = False

    def dominates(self, other):
        return (
            self.rc <= other.rc + 1e-9
            and self.time <= other.time + 1e-9
            and self.demand <= other.demand + 1e-9
        )


def _reconstruct_customers(label):
    """Reconstruct customer sequence from label back-pointers (no depots)."""
    path = []
    cur = label
    while cur is not None:
        path.append(cur.node[0])
        cur = cur.prev
    path.reverse()
    return path


class ExpandedGraphPricing:
    """
    Pricing backend using precomputed expanded graphs.

    It precomputes G^m once per ColumnGeneration instance (one BnP node) and
    then reuses the chosen graph across pricing iterations while duals change.
    """

    def __init__(self, user_param, max_m=None, quiet_graph_logs=True, quiet_spprc_logs=True):
        self.quiet_graph_logs = quiet_graph_logs
        self.quiet_spprc_logs = quiet_spprc_logs

        self.instance_path = getattr(user_param, "instance_path", "")
        if not self.instance_path:
            raise ValueError(
                "ParamsVRP.instance_path is missing. Initialize ParamsVRP via init_params(path)."
            )

        # max_m is a safety cap only; default is customer_count (effectively unbounded).
        # Precomputation stops as soon as the first acyclic (DAG) expanded graph is found.
        customer_count = max(1, user_param.nbclients - 1)
        self.max_m = customer_count if max_m is None else max(1, min(max_m, customer_count))
        self.graph_cache = []
        self.selected_idx = None
        self._user_param = user_param  # keep reference so _precompute_graphs can use it

        self._precompute_graphs(user_param=user_param)

    def _build_expanded_graph(self, m, user_param=None):
        t0 = time.time()
        # Build original directed graph from ParamsVRP
        G_original = nx.DiGraph()
        n = user_param.nbclients
        
        # Add all nodes (depot 0 through customers and end depot)
        for i in range(n + 1):
            G_original.add_node(i)
        
        # Add edges from feasible arcs
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j and user_param.dist[i][j] < user_param.verybig - 1e-6:
                    G_original.add_edge(i, j, cost=user_param.cost[i][j])
        
        # Use depot-based graph transformation
        if self.quiet_graph_logs:
            with contextlib.redirect_stdout(io.StringIO()):
                expanded_graph = graphTransformation_digraph_depot(G_original, m=m)
        else:
            expanded_graph = graphTransformation_digraph_depot(G_original, m=m)
        
        # Return None for gt since we're not using GraphTransformation class anymore
        return None, expanded_graph, time.time() - t0

    @staticmethod
    def _is_dag(expanded_graph):
        simple = nx.DiGraph(expanded_graph)
        return nx.is_directed_acyclic_graph(simple)

    def _expanded_graph_has_negative_cycle(self, expanded_graph, user_param, dual_pi=None):
        """
        Check whether expanded_graph contains a negative cycle under the
        current reduced costs defined by `user_param` and `dual_pi`.

        Returns True if a negative cycle exists, False otherwise.
        """
        p = user_param
        # Build a simple DiGraph of minimum-weight edges among parallel edges
        Gw = nx.DiGraph()

        last_customer = p.nbclients - 1

        for u, v, data in expanded_graph.edges(data=True):
            # Determine base cost for this expanded edge
            parent = data.get("parent_arc") or data.get("parent")
            if parent and isinstance(parent, tuple) and len(parent) == 2:
                i, j = parent
                weight = p.cost[i][j]
            else:
                # Dummy or sink edges treated as zero-cost (no duals applied)
                weight = 0.0

            # Keep minimum weight among parallel edges
            if Gw.has_edge(u, v):
                if weight < Gw[u][v]["weight"]:
                    Gw[u][v]["weight"] = weight
            else:
                Gw.add_edge(u, v, weight=weight)

        try:
            return nx.negative_edge_cycle(Gw, weight="weight")
        except Exception:
            # Conservatively assume no negative cycle if something unexpected happens
            return False

    def _precompute_graphs(self, user_param=None):
        print(f"[ExpandedPricing] Precomputing expanded graphs until first acyclic G^m (safety cap: m={self.max_m})")

        for m in range(1, self.max_m + 1):
            gt, exp_graph, build_sec = self._build_expanded_graph(m, user_param=user_param)
            is_dag = self._is_dag(exp_graph)
            self.graph_cache.append({
                "m": m,
                "gt": gt,
                "graph": exp_graph,
                "is_dag": is_dag,
                "build_sec": build_sec,
            })
            print(
                f"  [m={m}] nodes={exp_graph.number_of_nodes()}, edges={exp_graph.number_of_edges()}, "
                f"dag={'YES' if is_dag else 'NO'}, build={build_sec:.2f}s"
            )

            if is_dag:
                self.selected_idx = len(self.graph_cache) - 1
                print(f"[ExpandedPricing] First acyclic expanded graph found at m={m}. Stopping precomputation.")
                return

        # Reached the safety cap without finding a DAG — use the last graph built.
        self.selected_idx = len(self.graph_cache) - 1
        print(
            f"[ExpandedPricing] No acyclic expanded graph found up to m={self.max_m} (safety cap); "
            f"using m={self.graph_cache[self.selected_idx]['m']}"
        )

    def _ensure_graph_built(self, target_m):
        """Build and cache missing graphs up to target_m (inclusive), on demand."""
        if target_m < 1:
            target_m = 1
        if target_m > self.max_m:
            target_m = self.max_m

        current_max = len(self.graph_cache)
        if target_m <= current_max:
            return

        for m in range(current_max + 1, target_m + 1):
            gt, exp_graph, build_sec = self._build_expanded_graph(m, user_param=self._user_param)
            is_dag = self._is_dag(exp_graph)
            self.graph_cache.append({
                "m": m,
                "gt": gt,
                "graph": exp_graph,
                "is_dag": is_dag,
                "build_sec": build_sec,
            })
            print(
                f"  [m={m}] nodes={exp_graph.number_of_nodes()}, edges={exp_graph.number_of_edges()}, "
                f"dag={'YES' if is_dag else 'NO'}, build={build_sec:.2f}s [on-demand]"
            )

    def current_m(self):
        if self.selected_idx is None or not self.graph_cache:
            return None
        return self.graph_cache[self.selected_idx]["m"]

    def current_is_dag(self):
        """Whether the currently selected expanded graph is acyclic."""
        if self.selected_idx is None or not self.graph_cache:
            return False
        return bool(self.graph_cache[self.selected_idx].get("is_dag", False))

    def can_advance_m(self):
        """Whether we can move to a larger m (and build it if missing)."""
        m_now = self.current_m()
        if m_now is None:
            return False
        return m_now < self.max_m

    def advance_to_next_m(self):
        """Select next m. Build it on demand if it was not precomputed yet."""
        m_now = self.current_m()
        if m_now is None or m_now >= self.max_m:
            return False

        m_next = m_now + 1
        self._ensure_graph_built(m_next)
        self.selected_idx = m_next - 1  # cache is 0-based, m starts at 1
        print(f"[ExpandedPricing] Escalated selected expanded graph to m={m_next}")
        return True

    def _spprc_on_expanded(self, user_param, expanded_graph, dual_pi=None, max_routes=20):
        """Label-setting SPPRC on a precomputed expanded graph using current node costs/duals."""
        p = user_param
        n = p.nbclients
        last_customer = n - 1
        depot_start = 0
        depot_end = n + 1  # End depot is n+1 in depot-based transformation

        if dual_pi is None:
            dual_pi = [0.0] * last_customer
        if len(dual_pi) != last_customer:
            raise ValueError(
                f"dual_pi must have length {last_customer}; got {len(dual_pi)}"
            )

        out_edges = defaultdict(list)
        for u, v, data in expanded_graph.edges(data=True):
            out_edges[u].append((v, data))

        def arc_rc(from_city, to_city):
            # Reduced cost on original arc from_city->to_city using current duals
            return p.cost[from_city][to_city]

        def arrive(current_time, from_city, to_city):
            t = current_time + p.s[from_city] + p.ttime[from_city][to_city]
            if t < p.a[to_city]:
                t = p.a[to_city]
            if t > p.b[to_city]:
                return None
            return t

        node_labels = defaultdict(list)
        pending = []

        # Find origin candidates: tuple states starting with depot (or just the depot itself)
        origin_candidates = []
        for node in expanded_graph.nodes():
            if isinstance(node, tuple) and len(node) > 0 and node[0] == depot_start:
                origin_candidates.append(node)

        if not origin_candidates:
            return []

        # Prefer the canonical origin state (0, 0, 0, ...), but fall back to any origin node
        origin_node = None
        for node in origin_candidates:
            if all(x == depot_start for x in node):  # All zeros
                origin_node = node
                break
        if origin_node is None:
            origin_node = origin_candidates[0]

        origin_lbl = _Label(node=origin_node, time=p.a[depot_start], demand=0.0, rc=0.0, prev=None)
        node_labels[origin_node].append(origin_lbl)
        pending.append(origin_lbl)

        pending.sort(key=lambda lb: lb.rc)
        completed_routes = []

        while pending and len(completed_routes) < max_routes:
            current = pending.pop(0)
            if current.dominated:
                continue

            # Extract current city: handle both tuple states and singleton nodes
            if isinstance(current.node, tuple):
                city_cur = current.node[0]
                is_sink = city_cur == depot_end
            else:
                city_cur = current.node
                is_sink = city_cur == depot_end
            
            if is_sink:
                if current.rc < -1e-7:
                    completed_routes.append(current)
                continue

            for neighbour, _ in out_edges[current.node]:
                # Extract next city: handle both tuple states and singleton nodes
                if isinstance(neighbour, tuple):
                    city_next = neighbour[0]
                    is_next_sink = city_next == depot_end
                else:
                    city_next = neighbour
                    is_next_sink = city_next == depot_end

                if is_next_sink:
                    sink_lbl = _Label(
                        node=neighbour,
                        time=current.time,
                        demand=current.demand,
                        rc=current.rc,
                        prev=current,
                    )
                    if sink_lbl.rc < -1e-7:
                        completed_routes.append(sink_lbl)
                    continue

                # Respect current branch-node arc eliminations.
                if p.dist[city_cur][city_next] >= p.verybig - 1e-6:
                    continue

                t_next = arrive(current.time, city_cur, city_next)
                if t_next is None:
                    continue

                dem_next = current.demand + p.d[city_next]
                if dem_next > p.capacity:
                    continue

                rc_next = current.rc + arc_rc(city_cur, city_next)
                new_lbl = _Label(
                    node=neighbour,
                    time=t_next,
                    demand=dem_next,
                    rc=rc_next,
                    prev=current,
                )

                dominated = False
                survivors = []
                for existing in node_labels[neighbour]:
                    if existing.dominated:
                        continue
                    if existing.dominates(new_lbl):
                        dominated = True
                        survivors.append(existing)
                        break
                    if new_lbl.dominates(existing):
                        existing.dominated = True
                    else:
                        survivors.append(existing)

                if dominated:
                    continue

                node_labels[neighbour] = survivors + [new_lbl]
                pending.append(new_lbl)
                pending.sort(key=lambda lb: lb.rc)

        neg_routes = []
        for sink_lbl in completed_routes:
            sequence = _reconstruct_customers(sink_lbl.prev)
            customers = [node for node in sequence if 1 <= node <= last_customer]
            neg_routes.append({
                "customers": customers,
                "rc": sink_lbl.rc,
                "time": sink_lbl.time,
                "demand": sink_lbl.demand,
            })

        neg_routes.sort(key=lambda r: r["rc"])
        return neg_routes

    def price(self, user_param, dual_pi, max_routes=20, expanded_graph=None):
        """Return Route objects generated from expanded-graph pricing for this BnP node."""
        if self.selected_idx is None:
            return []

        working_graph = expanded_graph if expanded_graph is not None else self.graph_cache[self.selected_idx]["graph"]

        # Ensure the selected expanded graph does not contain a negative
        # reduced-cost cycle under the current duals. If it does, try to
        # advance m (build larger expanded graph) until no negative cycle
        # remains or we hit the cap. This keeps the per-iteration work to
        # checking cached graphs rather than rebuilding repeatedly.
        while True:
            selected = self.graph_cache[self.selected_idx]
            m = selected["m"]
            expanded_graph = working_graph

            has_neg = False
            try:
                has_neg = self._expanded_graph_has_negative_cycle(expanded_graph, user_param, dual_pi)
            except Exception:
                has_neg = False

            if has_neg:
                if self.can_advance_m():
                    # Build/select the next larger expanded graph and re-check
                    self.advance_to_next_m()
                    continue
                else:
                    if not self.quiet_spprc_logs:
                        print(f"[ExpandedPricing] Warning: negative cycle detected in expanded graph m={m} and cannot advance m further.")
                    break
            else:
                break

        if not self.quiet_spprc_logs:
            graph_tag = "caller-pruned" if working_graph is not self.graph_cache[self.selected_idx]["graph"] else f"cached m={self.graph_cache[self.selected_idx]['m']}"
            print(f"[ExpandedPricing] Running expanded-graph SPPRC on {graph_tag}")

        priced = self._spprc_on_expanded(
            user_param=user_param,
            expanded_graph=working_graph,
            dual_pi=dual_pi,
            max_routes=max_routes,
        )

        routes = []
        for item in priced:
            customers = item["customers"]
            full_path = [0] + customers + [user_param.nbclients]
            routes.append(Route(path=full_path, cost=item["rc"], Q=0.0))
        return routes
