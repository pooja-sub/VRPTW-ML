import networkx as nx
from collections import deque
import matplotlib.pyplot as plt


def graphTransformation(G_original, m=1):
    # m = number of predecessors to track.
    # Therefore tuple/state length is m + 1 (destination + m predecessors).
    tuple_len = m + 1

    G = G_original.copy()
    dummy = "X"

    # -------------------------
    # Phase I: Add dummy node
    # -------------------------
    if dummy not in G:
        G.add_node(dummy)

    for node in list(G.nodes()):
        G.add_edge(dummy, node)

    G_rev = G.reverse()

    def has_repeated_non_dummy_nodes(path_nodes):
        non_dummy = [n for n in path_nodes if n != dummy]
        return len(non_dummy) != len(set(non_dummy))

    def is_feasible_path(path_nodes):
        # path_nodes is [current, pred_1, pred_2, ...], so each step needs pred -> current in G.
        for idx in range(len(path_nodes) - 1):
            current_node = path_nodes[idx]
            predecessor = path_nodes[idx + 1]
            if not G.has_edge(predecessor, current_node):
                return False
        return True

    # -------------------------
    # Phase I: Enumerate M_i
    # -------------------------
    M = {}

    for target in G.nodes():
        sequences = set()

        queue = deque()
        queue.append((target, [target]))

        while queue:
            current, path = queue.popleft()

            if len(path) > tuple_len:
                continue

            if not is_feasible_path(path):
                continue

            seq = path[:]

            # pad with X at END
            if len(seq) < tuple_len:
                seq = seq + [dummy] * (tuple_len - len(seq))

            seq = tuple(seq)

            # Keep padded states, including the all-dummy tuple.
            sequences.add(seq)

            for pred in G_rev.neighbors(current):
                if len(path) < tuple_len:
                    next_path = path + [pred]
                    queue.append((pred, next_path))

        M[target] = sequences

    print("\n=== M_i sets ===")
    for node in sorted(M.keys(), key=str):
        mi_sorted = sorted(M[node], key=str)
        print(f"M_{node} ({len(mi_sorted)} states):")
        for state in mi_sorted:
            print(f"  {state}")

    # -------------------------
    # Phase II: Build expanded graph
    # -------------------------
    G_expanded = nx.DiGraph()

    # Add sequence nodes
    for i in M:
        for seq in M[i]:
            G_expanded.add_node(seq)

    # Add singleton nodes
    for i in G.nodes():
        if i != dummy:
            G_expanded.add_node(i)

    # -------------------------
    # Add REGULAR arcs
    # -------------------------
    def states_are_compatible(k, l):
        # Regular arc definition:
        # first m elements of k equal last m elements of l.
        return k[:m] == l[1:]

    def creates_cycle_on_transition(k, l):
        # Reject transitions where the next node already exists in the current
        # memory state.
        next_node = l[0]
        return next_node in (x for x in k if x != dummy)

    for (i, j) in G_original.edges():

        for k in M[i]:
            for l in M[j]:
                if not states_are_compatible(k, l):
                    continue

                if creates_cycle_on_transition(k, l):
                    continue

                G_expanded.add_edge(
                    k, l,
                    type="regular",
                    parent=(i, j)
                )

    # -------------------------
    # Add DUMMY arcs (FINAL CORRECT)
    # -------------------------
    for i in G.nodes():
        if i == dummy:
            continue

        for k in M[i]:
            G_expanded.add_edge(
                k, i,
                type="dummy"
            )

    # Add all-dummy tuple transitions to each origin singleton node.
    all_dummy_state = tuple([dummy] * tuple_len)
    if all_dummy_state in G_expanded:
        for i in G.nodes():
            if i != dummy:
                G_expanded.add_edge(
                    all_dummy_state, i,
                    type="dummy"
                )

    return G_expanded


def print_cycles(G_expanded, m, max_to_print=None):
    """Print directed cycles in the expanded graph for a given m."""
    cycles = list(nx.simple_cycles(G_expanded))

    print(f"\n=== Cycles for m={m} ===")
    print(f"Total cycles: {len(cycles)}")

    if not cycles:
        return

    # Keep output readable if cycle count grows.
    limit = len(cycles) if max_to_print is None else min(max_to_print, len(cycles))
    for idx, cyc in enumerate(cycles[:limit], start=1):
        # Repeat first node at end for readability.
        cyc_path = cyc + [cyc[0]]
        print(f"Cycle {idx}: {' -> '.join(map(str, cyc_path))}")

    if limit < len(cycles):
        print(f"... {len(cycles) - limit} more cycles not shown")


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    G = nx.DiGraph()

    edges = [
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 2),
        (3, 4),
        (3, 5),
        (4, 5),
    ]

    G.add_edges_from(edges)

    # Try higher m values here.
    m_values = [1, 2]

    for m in m_values:
        G_expanded = graphTransformation(G, m=m)

        print(f"\n\n########## RESULTS FOR m={m} ##########")
        print("\n=== Nodes ===")
        for n in sorted(G_expanded.nodes(), key=str):
            print(n)

        print("\n=== Edges ===")
        for u, v, d in G_expanded.edges(data=True):
            print(f"{u} -> {v}, {d}")

        print("\nTotal nodes:", G_expanded.number_of_nodes())
        print("Total edges:", G_expanded.number_of_edges())

        # Print all cycles for this simple graph.
        print_cycles(G_expanded, m=m)

        # Draw one figure per m.
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G_expanded, seed=2)
        nx.draw(G_expanded, pos, with_labels=True, node_size=900, font_size=8)
        plt.title(f"Expanded Graph for m={m}")

    # plt.show()