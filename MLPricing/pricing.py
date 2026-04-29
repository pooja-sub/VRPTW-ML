"""
ML-based pricing module implementing Algorithm 2 (reduced-graph pricing with ML arc relevance).

This module exposes:
- build_reduced_graph(params, feature_csv=None, predict_fn=None)
- run_ml_pricing_iteration(params, pi, sp_solver, A_r, useReducedG, eta_min=1, eta_max=5)

Notes:
- `predict_fn` should accept a dict of feature values and return 0/1.
- `sp_solver` should be the SPPRC class (importable); this function will call
    `sp_solver.shortestPath(params, new_routes, params.nbclients - 1)` after masking disallowed arcs.
- The CSV is expected in the same format as your file: header includes `from` and `to`.

This file only creates/returns routes. It does not modify the RMP; the caller should
add the generated routes to the master problem.
"""
from typing import Set, Tuple, Optional, Callable
import os
import csv
import copy

# default predict function placeholder - user can pass their own
try:
    # try to import user-supplied wrapper (created earlier)
    from MLPricing.model_wrapper import predict as default_predict
except Exception:
    def default_predict(features):
        return 0


def build_reduced_graph(params, feature_csv: Optional[str] = None, predict_fn: Optional[Callable] = None):
    """Read arc-feature CSV and return set of arcs predicted relevant A_r.

    Returns: set of (from,to) tuples (integers)
    """
    predict = predict_fn or default_predict

    # determine file path
    if feature_csv is None:
        # try to locate in DatasetFeatures matching dataset name like earlier logic
        root = os.path.dirname(os.path.dirname(__file__))
        df_dir = os.path.join(root, 'DatasetFeatures')
        if not os.path.isdir(df_dir):
            return set()
        name = (params.datasetName or '').lower()
        chosen = None
        for fname in os.listdir(df_dir):
            lf = fname.lower()
            if 'arc' in lf and name in lf and lf.endswith('.csv'):
                chosen = os.path.join(df_dir, fname)
                break
        if chosen is None:
            # fallback: first arc csv
            for fname in os.listdir(df_dir):
                lf = fname.lower()
                if 'arc' in lf and lf.endswith('.csv'):
                    chosen = os.path.join(df_dir, fname)
                    break
        feature_csv = chosen

    arcs = set()
    if not feature_csv or not os.path.isfile(feature_csv):
        return arcs

    # Read CSV and predict for each arc
    with open(feature_csv, 'r', newline='', encoding='utf-8') as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            try:
                f = int(row.get('from', row.get('From', '')).strip())
                t = int(row.get('to', row.get('To', '')).strip())
            except Exception:
                continue
            # skip depot arcs (assume depot index 0 or last index not present in file per user)
            nb_last = params.nbclients
            if f == 0 or t == 0 or f == nb_last or t == nb_last:
                continue
            # call predict on the feature dict (pass the row as-is)
            try:
                y = predict(row)
            except Exception:
                y = 0
            if int(y) == 1:
                arcs.add((f, t))
    # Always include depot->customer and customer->depot arcs so reduced graph
    # preserves feasibility (start/end depot arcs are required by SPPRC).
    try:
        nb_last = params.nbclients
        for k in range(1, nb_last):
            arcs.add((0, k))
            arcs.add((k, nb_last))
    except Exception:
        # if params is malformed, just ignore and return whatever was predicted
        pass

    # optional short log: number of arcs in reduced-graph
    try:
        # avoid importing logging here to keep behavior minimal; print is acceptable
        print(f"[MLPricing] build_reduced_graph: using feature_csv={feature_csv}, predicted_arcs={len(arcs)}")
    except Exception:
        pass

    return arcs


def _mask_graph_by_arcs(params, allowed_arcs: Set[Tuple[int, int]]):
    """Temporarily set params.dist for disallowed arcs to a very big value.
    Returns a deep copy of the original dist matrix so it can be restored later.
    """
    orig = copy.deepcopy(params.dist)
    nb = params.nbclients + 2
    verybig = params.verybig
    for i in range(nb):
        for j in range(nb):
            if (i, j) not in allowed_arcs:
                params.dist[i][j] = verybig
    return orig


def _restore_graph(params, orig_dist):
    params.dist = orig_dist


def run_ml_pricing_iteration(params, pi, sp_solver, A_r: Set[Tuple[int,int]], useReducedG: bool,
                             eta_min: int = 5, eta_max: int = 10):
    """Run one ML-augmented pricing iteration.

    - params: ParamsVRP instance
    - pi: current dual prices (list)
    - sp_solver: SPPRC class (module imported as SPPRC)
    - A_r: reduced arc set (set of (from,to))
    - useReducedG: whether to use reduced graph this iteration
    - eta_min/eta_max: thresholds for switching graphs

    Returns: (new_routes_list, useReducedG_after, num_generated)
    """
    # Update reduced cost (same as original code does)
    for i in range(1, params.nbclients):
        for j in range(params.nbclients + 2):
            params.cost[i][j] = params.dist[i][j] - pi[i - 1]

    # decide active arcs
    orig_dist = None
    if useReducedG and A_r:
        allowed_arcs = set()
        # include arcs predicted relevant
        for (a, b) in A_r:
            allowed_arcs.add((a, b))
        # Always include depot arcs (start depot -> customers) and (customers -> end depot)
        nb_last = params.nbclients
        for k in range(1, nb_last):
            allowed_arcs.add((0, k))
            allowed_arcs.add((k, nb_last))

        # Mask the graph to allowed arcs
        orig_dist = _mask_graph_by_arcs(params, allowed_arcs)

    # solve SPPRC on the (possibly masked) graph
    new_routes = []
    try:
        sp_solver.shortestPath(params, new_routes, params.nbclients - 1)
    finally:
        if orig_dist is not None:
            _restore_graph(params, orig_dist)

    # count generated columns
    gen_count = len(new_routes) if new_routes else 0

    # switching logic (internal): switch between reduced and full graph depending on generation
    useReducedG_after = useReducedG
    if useReducedG and gen_count < eta_min:
        useReducedG_after = False
    elif (not useReducedG) and gen_count >= eta_max and A_r:
        useReducedG_after = True

    return new_routes, useReducedG_after, gen_count
