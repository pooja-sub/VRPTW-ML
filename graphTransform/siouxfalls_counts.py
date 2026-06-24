import csv
import networkx as nx

from graphTransformation import graphTransformation_digraph


def build_graph_from_csv(csv_path: str) -> nx.DiGraph:
    g = nx.DiGraph()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["A"])
            b = int(row["B"])
            g.add_edge(a, b)
    return g


def main() -> None:
    csv_path = "SiouxFalls_net.csv"
    g = build_graph_from_csv(csv_path)

    print(f"Original graph: nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")

    for m in (1, 2, 3):
        g_expanded = graphTransformation_digraph(g, m=m)
        print(
            f"m={m}: expanded_nodes={g_expanded.number_of_nodes()}, "
            f"expanded_edges={g_expanded.number_of_edges()}"
        )


if __name__ == "__main__":
    main()
