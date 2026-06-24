import networkx as nx
from pathlib import Path
from graphTransformation import graphTransformation_digraph


def build_weighted_test_graph():
    """
    Build a larger directed graph with designed negative-cost cycles.

    The graph contains multiple SCCs and both negative/positive cycles,
    including length-2 bidirectional cycles.
    """
    G = nx.DiGraph()

    # Negative cycle A (length 3): 1 -> 2 -> 3 -> 1  with total -2
    # Weights: -4 + (-3) + 5 = -2
    # Negative cycle B (length 4): 4 -> 5 -> 6 -> 7 -> 4 with total -1
    # Weights: -2 + (-2) + (-2) + 5 = -1
    # Negative cycle D (length 2): 11 <-> 12 with total -2
    # Weights: -1 + (-1) = -2
    # Positive cycle C (length 3): 8 -> 9 -> 10 -> 8 with total +3
    # Weights: 1 + 1 + 1 = +3
    edges_with_w = [
        (1, 2, -4.0),
        (2, 3, -3.0),
        (3, 1, 5.0),
        (4, 5, -2.0),
        (5, 6, -2.0),
        (6, 7, -2.0),
        (7, 4, 5.0),
        (8, 9, 1.0),
        (9, 10, 1.0),
        (10, 8, 1.0),
        # Cross links to make the graph larger and richer.
        (3, 4, 2.0),
        (7, 8, 2.0),
        (2, 6, 3.0),
        (5, 9, 4.0),
        (10, 11, 2.0),
        # Length-2 negative cycle: 11 <-> 12 with total -2
        (11, 12, -1.0),
        (12, 11, -1.0),
        (6, 11, 1.0),
        (12, 2, 3.0),
        (1, 8, 6.0),
        (9, 4, 2.0),
    ]

    for u, v, w in edges_with_w:
        G.add_edge(u, v, weight=w)

    return G


def build_weighted_test_graph_2():
    """Alternative graph with multiple interacting negative cycles, including length-2."""
    G = nx.DiGraph()
    edges_with_w = [
        # Negative cycle A: 1->2->3->1 = -1
        (1, 2, -2.0),
        (2, 3, -2.0),
        (3, 1, 3.0),
        # Negative cycle B: 4->5->6->4 = -2
        (4, 5, -3.0),
        (5, 6, -2.0),
        (6, 4, 3.0),
        # Negative cycle C: 7->8->9->10->7 = -1
        (7, 8, -2.0),
        (8, 9, -1.0),
        (9, 10, -1.0),
        (10, 7, 3.0),
        # Length-2 negative cycle: 11 <-> 12
        (11, 12, -2.0),
        (12, 11, -1.0),
        # Bridges / alternate loops
        (3, 4, 1.0),
        (6, 7, 2.0),
        (10, 11, 1.0),
        (12, 2, 2.0),
        (9, 5, 2.0),
        (2, 8, 4.0),
        (11, 1, 2.0),
    ]
    for u, v, w in edges_with_w:
        G.add_edge(u, v, weight=w)
    return G


def build_weighted_test_graph_3():
    """Larger sparse graph with one short and one long negative cycle, plus length-2."""
    G = nx.DiGraph()
    edges_with_w = [
        # Short negative cycle: 1->2->1 = -1
        (1, 2, -3.0),
        (2, 1, 2.0),
        # Long negative cycle: 3->4->5->6->7->3 = -2
        (3, 4, -1.0),
        (4, 5, -1.0),
        (5, 6, -1.0),
        (6, 7, -1.0),
        (7, 3, 2.0),
        # Length-2 negative cycle: 11 <-> 12
        (11, 12, -1.0),
        (12, 11, -1.0),
        # Positive cycle: 8->9->10->8 = +3
        (8, 9, 1.0),
        (9, 10, 1.0),
        (10, 8, 1.0),
        # Connectors
        (2, 3, 1.0),
        (7, 8, 2.0),
        (10, 11, 1.0),
        (6, 12, 2.0),
        (13, 11, 2.0),
        (9, 4, 3.0),
    ]
    for u, v, w in edges_with_w:
        G.add_edge(u, v, weight=w)
    return G


def build_weighted_expanded_graph(G_original, m):
    """
    Run graphTransformation.graphTransformation_digraph and assign costs to expanded arcs.

    - regular arc cost = parent original arc weight
    - dummy arc cost = 0
    """
    G_expanded = graphTransformation_digraph(G_original, m=m)
    G_weighted = nx.DiGraph()

    for u, v, data in G_expanded.edges(data=True):
        if data.get("type") == "regular":
            i, j = data["parent"]
            w = G_original[i][j]["weight"]
        else:
            w = 0.0

        # Keep the best (lowest-cost) parallel projection.
        if G_weighted.has_edge(u, v):
            if w < G_weighted[u][v]["weight"]:
                G_weighted[u][v]["weight"] = w
        else:
            G_weighted.add_edge(u, v, weight=w, edge_type=data.get("type"))

    for n in G_expanded.nodes():
        if n not in G_weighted:
            G_weighted.add_node(n)

    return G_expanded, G_weighted


def cycle_cost(G_weighted, cycle_nodes):
    total = 0.0
    k = len(cycle_nodes)
    for i in range(k):
        u = cycle_nodes[i]
        v = cycle_nodes[(i + 1) % k]
        total += G_weighted[u][v]["weight"]
    return total


def cycle_to_node_walk(cycle_nodes):
    """
    Convert an expanded-graph cycle to a node walk in the original graph.

    For tuple states, use effective-tail semantics from transformation:
    start at the last non-X predecessor of the first tuple (if any), then
    append tuple destinations: [tail(first), first[0], second[0], ...].
    which gives a closed walk segment suitable for repeated-node detection.
    For non-tuple nodes, fallback to [n0, n1, ..., n0].
    """
    if not cycle_nodes:
        return []

    def effective_tail_label(state):
        # Match graphTransformation compatibility: for (i, a, b, X, X), tail is b.
        # True origin states (i, X, X, ...) keep tail X.
        if not isinstance(state, tuple) or len(state) <= 1:
            return None
        for val in reversed(state[1:]):
            if val != "X":
                return val
        return "X"

    all_tuples = all(isinstance(n, tuple) and len(n) > 0 for n in cycle_nodes)
    if all_tuples:
        start = effective_tail_label(cycle_nodes[0])
        if start in (None, "X"):
            # Fallback keeps the walk meaningful when the state has no real predecessor.
            start = cycle_nodes[0][0]
        walk = [start]
        walk.extend(n[0] for n in cycle_nodes)
        return walk

    return list(cycle_nodes) + [cycle_nodes[0]]


def smallest_revisit_subcycle(node_walk):
    """
    Return the shortest closed subpath induced by a repeated node in node_walk.

    Example: [1, 2, 3, 1, 2] -> [1, 2, 3, 1]
    """
    if not node_walk:
        return None

    last_seen = {}
    best_len = None
    best_path = None

    for idx, node in enumerate(node_walk):
        if node in last_seen:
            start = last_seen[node]
            cycle_path = node_walk[start : idx + 1]
            clen = idx - start
            if best_len is None or clen < best_len:
                best_len = clen
                best_path = cycle_path
        last_seen[node] = idx

    return best_path


def original_path_cost(G_original, closed_path):
    """Cost of a closed node path in the original weighted graph."""
    if not closed_path or len(closed_path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(closed_path) - 1):
        u = closed_path[i]
        v = closed_path[i + 1]
        if not G_original.has_edge(u, v):
            return None
        total += G_original[u][v]["weight"]
    return total


def list_cycles_with_cost(G_weighted, max_cycles=20000):
    """Enumerate cycles with their total costs (bounded by max_cycles)."""
    rows = []
    count = 0
    truncated = False

    for cyc in nx.simple_cycles(G_weighted):
        count += 1
        if count > max_cycles:
            truncated = True
            break
        rows.append((cyc, cycle_cost(G_weighted, cyc)))

    rows.sort(key=lambda x: (x[1], len(x[0])))
    return rows, count, truncated


def sample_negative_cycles(G_weighted, max_negative=20, max_explored=50000):
    """
    Enumerate directed cycles and keep only negative-cost cycles.
    Prioritizes shorter cycles (esp. length-2).

    Returns:
        neg_cycles: list[(cycle_nodes, cost)]
        explored: number of cycles examined
        truncated: whether enumeration was cut by max_explored
    """
    neg_cycles = []
    explored = 0
    truncated = False

    for cyc in nx.simple_cycles(G_weighted):
        explored += 1
        if explored > max_explored:
            truncated = True
            break

        c = cycle_cost(G_weighted, cyc)
        if c < -1e-9:
            neg_cycles.append((cyc, c))
            if len(neg_cycles) >= max_negative:
                break

    # Sort by cycle length first (shorter first), then by most negative cost
    neg_cycles.sort(key=lambda x: (len(x[0]), x[1]))
    return neg_cycles, explored, truncated


def find_blocking_violation(G_original, G_weighted, m, max_explored=200000):
    """
    Search for a counterexample to the blocking claim for a given m.

    A violation is a negative cycle in the expanded graph that reconstructs to a
    valid original closed walk of length <= m+1.

    Returns:
        (violation_dict_or_none, explored_count, truncated)
    """
    explored = 0

    for cyc in nx.simple_cycles(G_weighted):
        explored += 1
        if explored > max_explored:
            return None, explored, True

        c = cycle_cost(G_weighted, cyc)
        if c >= -1e-9:
            continue

        orig_path = reconstruct_original_path_from_expanded(cyc, m)
        is_valid, orig_cost = is_valid_cycle_in_original(G_original, orig_path)
        if not is_valid:
            continue

        orig_len = len(orig_path) - 1
        if orig_len <= m + 1:
            return {
                "expanded_cycle": cyc,
                "expanded_cost": c,
                "original_path": orig_path,
                "original_cost": orig_cost,
                "original_len": orig_len,
            }, explored, False

    return None, explored, False


def format_expanded_cycle(cycle_nodes):
    """Format a cycle from the expanded graph as a readable closed walk."""
    if not cycle_nodes:
        return ""

    closed = list(cycle_nodes) + [cycle_nodes[0]]
    return " -> ".join(map(str, closed))


def reconstruct_original_path_from_expanded(cycle_nodes, m):
    """
    Reconstruct the original graph path from an expanded cycle.
    
    Each expanded state (dest, pred_1, ..., pred_m) represents an arc pred_1 -> dest.
    For a sequence of states, we chain them to get the full path.
    
    Example for m=1:
        (2, 1) -> (3, 2) -> (1, 3) represents arcs:
        1->2, 2->3, 3->1, so path is [1, 2, 3, 1]
    """
    if not cycle_nodes:
        return []
    
    path = []
    
    # Start with the first state's shape to get the initial nodes
    first_state = cycle_nodes[0]
    if isinstance(first_state, tuple) and len(first_state) > 1:
        # For m=1: (dest, pred) -> start from pred
        # For m=2: (dest, pred_1, pred_2) -> start from pred_2, pred_1, dest
        # In general, read backwards to get the full path up to dest
        path.append(first_state[-1])  # Add the furthest predecessor
        for i in range(len(first_state) - 2, 0, -1):
            if first_state[i] != "X":
                path.append(first_state[i])
        path.append(first_state[0])  # Add destination
    else:
        # Singleton node
        path.append(first_state)
    
    # For subsequent states, just append the destination (since predecessors shift)
    for i in range(1, len(cycle_nodes)):
        state = cycle_nodes[i]
        if isinstance(state, tuple):
            path.append(state[0])
        else:
            path.append(state)
    
    return path


def is_valid_cycle_in_original(G_original, original_path):
    """Check if a path forms a valid closed cycle in the original graph."""
    if not original_path or len(original_path) < 2:
        return False, 0.0
    
    # Check if the path closes (first == last)
    if original_path[0] != original_path[-1]:
        return False, 0.0
    
    # Check if all edges exist and compute cost
    total_cost = 0.0
    for i in range(len(original_path) - 1):
        u = original_path[i]
        v = original_path[i + 1]
        if not G_original.has_edge(u, v):
            return False, 0.0
        total_cost += G_original[u][v]["weight"]
    
    return True, total_cost


def _analyze_single_graph(G, m_values, out):
    """Write full analysis for one weighted graph to output stream."""

    out.write("=" * 90 + "\n")
    out.write("NEGATIVE-CYCLE EXPERIMENT ON EXPANDED GRAPHS\n")
    out.write("=" * 90 + "\n")
    out.write(f"Original weighted graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    has_neg_base = nx.negative_edge_cycle(G, weight="weight")
    out.write(f"Original graph has negative cycle: {has_neg_base}\n")

    base_cycles, base_count, base_truncated = list_cycles_with_cost(G, max_cycles=5000)
    out.write(f"Original graph cycles found: {len(base_cycles)}{' (truncated)' if base_truncated else ''}\n")
    out.write("Original graph cycle details:\n")
    if not base_cycles:
        out.write("  (none)\n")
    else:
        for idx, (cyc, c) in enumerate(base_cycles, start=1):
            cyc_path = cyc + [cyc[0]]
            kind = "NEG" if c < -1e-9 else ("ZERO" if abs(c) <= 1e-9 else "POS")
            out.write(
                f"  {idx:02d}. type={kind:4s}, len={len(cyc):2d}, "
                f"cost={c:7.2f}, cycle={' -> '.join(map(str, cyc_path))}\n"
            )

    out.write("=" * 90 + "\n")

    # Ground truth in original graph: shortest negative simple cycle lengths.
    base_neg_lengths = sorted(len(cyc) for cyc, c in base_cycles if c < -1e-9)
    if base_neg_lengths:
        out.write(f"Original negative cycle lengths (simple): {base_neg_lengths}\n")
    else:
        out.write("Original negative cycle lengths (simple): none\n")
    out.write("=" * 90 + "\n")

    for m in m_values:
        G_exp, Gw = build_weighted_expanded_graph(G, m)

        has_neg = nx.negative_edge_cycle(Gw, weight="weight") if Gw.number_of_edges() else False
        neg_samples, explored, truncated = sample_negative_cycles(Gw, max_negative=20, max_explored=30000)

        out.write(f"\n[m={m}] expanded nodes={G_exp.number_of_nodes()}, edges={G_exp.number_of_edges()}\n")
        out.write(f"  has_negative_cycle={has_neg}\n")
        out.write(f"  explored_cycles={explored}{' (truncated)' if truncated else ''}\n")
        out.write(f"  sampled_negative_cycles={len(neg_samples)}\n")

        short_neg_in_original = any(c < -1e-9 and len(cyc) <= (m + 1) for cyc, c in base_cycles)
        out.write(
            f"  original_has_negative_cycle_len<=m+1 ({m+1}) = {short_neg_in_original}\n"
        )

        violation, v_explored, v_truncated = find_blocking_violation(
            G, Gw, m, max_explored=200000
        )
        out.write(
            f"  blocking_check_explored={v_explored}{' (truncated)' if v_truncated else ''}\n"
        )

        if violation is None and not v_truncated:
            out.write("  blocking_verdict=PASS (no violating negative cycle found)\n")
        elif violation is None and v_truncated:
            out.write("  blocking_verdict=INCONCLUSIVE (search truncated)\n")
        else:
            out.write("  blocking_verdict=FAIL (counterexample found)\n")
            out.write(
                f"    violating_original_len={violation['original_len']} <= {m+1}, "
                f"expanded_cost={violation['expanded_cost']:.2f}, "
                f"original_cost={violation['original_cost']:.2f}\n"
            )
            out.write(
                f"    violating_expanded_cycle={format_expanded_cycle(violation['expanded_cycle'])}\n"
            )
            out.write(
                f"    violating_original_path={' -> '.join(map(str, violation['original_path']))}\n"
            )

        for idx, (cyc, c) in enumerate(neg_samples, start=1):
            # Reconstruct the original path from expanded cycle
            orig_path = reconstruct_original_path_from_expanded(cyc, m)
            is_valid, orig_cost = is_valid_cycle_in_original(G, orig_path)
            
            orig_path_str = " -> ".join(map(str, orig_path)) if orig_path else "(invalid)"
            orig_cost_str = f"{orig_cost:7.2f}" if is_valid else "   N/A"
            
            out.write(
                f"    {idx:02d}. len={len(cyc):2d}, cost={c:7.2f}, "
                f"expanded_cycle={format_expanded_cycle(cyc)}\n"
            )
            out.write(
                f"        reconstructed_original_path={orig_path_str}, "
                f"valid={is_valid}, orig_cost={orig_cost_str}\n"
            )


def analyze_m_range(m_values):
    """Backward-compatible single-graph analysis to stdout."""
    G = build_weighted_test_graph()
    _analyze_single_graph(G, m_values, out=__import__("sys").stdout)


def analyze_multiple_graphs_to_file(output_path, m_values):
    """Run analysis on multiple graphs and save all details to a text file."""
    graphs = [
        ("Graph_A", build_weighted_test_graph()),
        ("Graph_B", build_weighted_test_graph_2()),
        ("Graph_C", build_weighted_test_graph_3()),
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("FULL NEGATIVE-CYCLE EXPERIMENT REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"m_values={list(m_values)}\n")
        f.write("=" * 100 + "\n\n")

        for name, G in graphs:
            f.write(f"\n{'#' * 100}\n")
            f.write(f"DATASET: {name}\n")
            f.write(f"{'#' * 100}\n")
            _analyze_single_graph(G, m_values, out=f)
            f.write("\n")

    print(f"[Report saved to: {output_path}]")


if __name__ == "__main__":
    # m is number of predecessors (so tuple length is m+1 in org.py)
    m_values = [1, 2, 3, 4, 5, 6, 7, 8]
    report_file = Path(__file__).with_name("org_negative_cycle_report.txt")
    analyze_multiple_graphs_to_file(report_file, m_values)
