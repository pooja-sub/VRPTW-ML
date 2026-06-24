import networkx as nx
import numpy as np
import sys
from collections import deque
import matplotlib.pyplot as plt
import time

# Add Branching folder to path to import ParamsVRP
sys.path.insert(0, '../Branching')
from paramsVRP import ParamsVRP

# Graph Transformation function
def graphTransformation_digraph(G_original, m=1):
    """
    Args:
        G_original: networkx.DiGraph
        m: number of predecessors to track (tuple length is m+1)

    Returns:
        networkx.DiGraph expanded graph with edge attributes:
        - type: "regular" or "dummy"
        - parent: (i, j) for regular edges
    """
    if not isinstance(G_original, nx.DiGraph):
        raise TypeError("G_original must be a networkx.DiGraph")

    tuple_len = m + 1
    dummy = "X"

    G = G_original.copy()
    if dummy not in G:
        G.add_node(dummy)

    for node in list(G.nodes()):
        G.add_edge(dummy, node)

    G_rev = G.reverse()

    def is_feasible_path(path_nodes):
        for idx in range(len(path_nodes) - 1):
            current_node = path_nodes[idx]
            predecessor = path_nodes[idx + 1]
            if not G.has_edge(predecessor, current_node):
                return False
        return True

    M = {}
    for target in G.nodes():
        sequences = set()
        queue = deque([(target, [target])])

        while queue:
            current, path = queue.popleft()

            if len(path) > tuple_len:
                continue
            if not is_feasible_path(path):
                continue

            seq = path[:]
            if len(seq) < tuple_len:
                seq = seq + [dummy] * (tuple_len - len(seq))
            seq = tuple(seq)

            sequences.add(seq)

            if len(path) < tuple_len:
                for pred in G_rev.neighbors(current):
                    next_path = path + [pred]
                    queue.append((pred, next_path))

        M[target] = sequences

    G_expanded = nx.DiGraph()

    for i in M:
        for seq in M[i]:
            G_expanded.add_node(seq)

    for i in G.nodes():
        if i != dummy:
            G_expanded.add_node(i)

    def states_are_compatible(k, l):
        # Regular arc definition:
        # first m elements of k equal last m elements of l.
        return k[:m] == l[1:]

    def creates_cycle_on_transition(k, l):
        # Match the log-style transition rule: reject only when the next node
        # already appears in the current state memory.
        next_node = l[0]
        return next_node in (x for x in k if x != dummy)

    for i, j in G_original.edges():

        for k in M[i]:
            for l in M[j]:
                if states_are_compatible(k, l):
                    if creates_cycle_on_transition(k, l):
                        continue
                    G_expanded.add_edge(k, l, type="regular", parent=(i, j))

    for i in G.nodes():
        if i == dummy:
            continue
        for k in M[i]:
            G_expanded.add_edge(k, i, type="dummy")

    # Add all-dummy tuple transitions to each origin singleton node.
    all_dummy_state = tuple([dummy] * tuple_len)
    if all_dummy_state in G_expanded:
        for i in G.nodes():
            if i != dummy:
                G_expanded.add_edge(all_dummy_state, i, type="dummy")

    return G_expanded


def graphTransformation_digraph_depot(G_original, m=1):
    """
    Modified graph transformation using depot node 0 instead of dummy node.
    - Depot 0 is start point only
    - End depot is NOT part of any state tuple
    - All customer states connect directly to end depot
    
    Args:
        G_original: networkx.DiGraph with edge weights (cost attribute)
        m: number of predecessors to track (tuple length is m+1)

    Returns:
        networkx.DiGraph expanded graph with edge attributes:
        - cost: edge cost
        - parent: (i, j) for regular edges
    """
    if not isinstance(G_original, nx.DiGraph):
        raise TypeError("G_original must be a networkx.DiGraph")

    tuple_len = m + 1
    depot = 0  # Start depot
    end_depot = max(G_original.nodes())   # End depot - determined from graph

    # Build graph with only customer-to-customer edges (no depot edges)
    G = nx.DiGraph()
    for node in G_original.nodes():
        G.add_node(node)
    
    for i, j in G_original.edges():
        # Only add edges between non-depot nodes 
        if i != depot and i != end_depot and j != depot and j != end_depot:
            G.add_edge(i, j)

    G_rev = G.reverse()

    def is_feasible_path(path_nodes):
        for idx in range(len(path_nodes) - 1):
            current_node = path_nodes[idx]
            predecessor = path_nodes[idx + 1]
            if not G.has_edge(predecessor, current_node):
                return False
        return True

    M = {}
    for target in G.nodes():
        if target == depot or target == end_depot:
            continue  # Skip both depots in enumeration
            
        sequences = set()
        queue = deque([(target, [target])])

        while queue:
            current, path = queue.popleft()

            if len(path) > tuple_len:
                continue
            if not is_feasible_path(path):
                continue

            seq = path[:]
            if len(seq) < tuple_len:
                seq = seq + [depot] * (tuple_len - len(seq))
            seq = tuple(seq)

            sequences.add(seq)

            if len(path) < tuple_len:
                for pred in G_rev.neighbors(current):
                    if pred == depot or pred == end_depot:  # Don't include depots in intermediate paths
                        continue
                    next_path = path + [pred]
                    queue.append((pred, next_path))

        M[target] = sequences

    G_expanded = nx.DiGraph()

    # Add all tuple states as nodes
    for i in M:
        for seq in M[i]:
            G_expanded.add_node(seq)

    # Add singleton customer nodes (not tuples)
    for i in G.nodes():
        if i != depot and i != end_depot:
            G_expanded.add_node(i)
    
    # Add end depot node
    G_expanded.add_node(end_depot)

    def states_are_compatible(k, l):
        # first m elements of k equal last m elements of l
        return k[:m] == l[1:]

    def creates_cycle_on_transition(k, l):
        # reject if next node already appears in current state memory (excluding depot)
        next_node = l[0]
        return next_node in (x for x in k if x != depot)

    # Add regular arcs between tuple states (only for customer-customer edges)
    for i, j in G_original.edges():
        # Skip any edges involving depots
        if i == depot or i == end_depot or j == depot or j == end_depot:
            continue
            
        for k in M[i]:
            for l in M[j]:
                if states_are_compatible(k, l):
                    if creates_cycle_on_transition(k, l):
                        continue
                    edge_data = {"type": "regular", "parent": (i, j)}
                    if G_original.has_edge(i, j) and "cost" in G_original[i][j]:
                        edge_data["cost"] = G_original[i][j]["cost"]
                    G_expanded.add_edge(k, l, **edge_data)

    # Add arcs from tuple states to singleton customer nodes
    for i in G.nodes():
        if i == depot or i == end_depot:
            continue
        for k in M[i]:
            G_expanded.add_edge(k, i, type="transition")

    # Add return arcs from singleton customer nodes to end depot
    for i in G.nodes():
        if i != depot and i != end_depot:
            if G_original.has_edge(depot, i):
                edge_data = G_original[depot][i]
                if "cost" in edge_data:
                    cost = edge_data["cost"]
            
            G_expanded.add_edge(i, end_depot, type="return", cost=cost)

    # Add origin arcs from all-depot state to customer origin states
    all_depot_state = tuple([depot] * tuple_len)
    if all_depot_state not in G_expanded:
        G_expanded.add_node(all_depot_state)
        
    for i in G.nodes():
        if i != depot and i != end_depot:
            # Create origin state (customer, 0, 0, ..., 0)
            origin_state = tuple([i] + [depot] * (tuple_len - 1))
            
            # Get cost from depot to customer
            cost = 1.0  # default cost
            if G_original.has_edge(depot, i):
                edge_data = G_original[depot][i]
                if "cost" in edge_data:
                    cost = edge_data["cost"]
            G_expanded.add_edge(all_depot_state, origin_state, type="origin", cost=cost)

    return G_expanded


# class GraphTransformation:
    
#     def __init__(self, instance_path, m=1, verbose=False, params=None):
#         """
#         Initialize graph transformation for a Solomon instance.
        
#         Args:
#             instance_path: Path to Solomon instance file
#             m: Maximum number of predecessors to track (default: 1)
#             verbose: Whether to print detailed intermediate path listings
#             params: Optional pre-initialised (and pre-preprocessed) ParamsVRP object.
#                     When supplied, instance_path is ignored and no file is re-read.
#         """
#         if params is not None:
#             self.params = params
#         else:
#             self.params = ParamsVRP()
#             self.params.init_params(instance_path)
        
#         # Create base graph from Solomon instance
#         self.graph = None
#         self.M = {}  # M[i] = set of feasible vectors of visited nodes for destination i
#         self.metrics_file = None  # File handle for logging metrics
#         self.all_shortest_paths = None  # Cache for all-pairs shortest paths
#         self.m = m  # Maximum number of predecessors
#         self.verbose = verbose
        
#         # Phase 2 edge tracking (before and after pruning)
#         self.phase2_edges_before_pruning = 0
#         self.phase2_edges_after_pruning = 0
        
#     def create_undirected_graph(self):
#         """
#         Create undirected graph from Solomon instance.
#         Nodes: 0 (depot), 1..n (customers)
#         Edges: exist if arc is feasible based on capacity and time window
#         """
#         G = nx.Graph()
        
#         # Add all nodes (depot + customers)
#         for i in range(self.params.nbclients + 1):
#             G.add_node(i, 
#                       x=self.params.posx[i],
#                       y=self.params.posy[i],
#                       demand=self.params.d[i],
#                       earliest=self.params.a[i],
#                       latest=self.params.b[i],
#                       service_time=self.params.s[i])
        
#         # Add feasible edges
#         for i in range(self.params.nbclients + 1):
#             for j in range(i + 1, self.params.nbclients + 1):
#                 if self.params.dist[i][j] < self.params.verybig - 1e-6:
#                     G.add_edge(i, j, 
#                               distance=self.params.dist[i][j],
#                               travel_time=self.params.ttime[i][j])
        
#         self.graph = G
#         print(f"[Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges]")
#         return G
    
#     def remove_infeasible_nodes(self):
#         """
#         Remove nodes that are not reachable from depot or cannot reach depot 
#         within time window and capacity constraints.
#         """
#         if self.graph is None:
#             self.create_undirected_graph()
        
#         nodes_to_remove = []
#         depot = 0
        
#         for node in self.graph.nodes():
#             if node == depot:
#                 continue
            
#             # Check if node is reachable from depot
#             try:
#                 nx.shortest_path(self.graph, depot, node)
#             except nx.NetworkXNoPath:
#                 nodes_to_remove.append(node)
#                 continue
            
#             # Check if node can reach back to depot
#             try:
#                 nx.shortest_path(self.graph, node, depot)
#             except nx.NetworkXNoPath:
#                 nodes_to_remove.append(node)
        
#         self.graph.remove_nodes_from(nodes_to_remove)
#         if nodes_to_remove:
#             print(f"[Removed {len(nodes_to_remove)} infeasible nodes: {nodes_to_remove}]")
#         else:
#             print(f"[No infeasible nodes to remove]")
    
#     def _log_metrics(self, message):
#         """
#         Log metrics to both console and file.
        
#         Args:
#             message: Message to log
#         """
#         print(message)
#         if self.metrics_file:
#             self.metrics_file.write(message + '\n')
    
#     def _compute_all_shortest_paths(self):
#         """
#         Compute all-pairs shortest paths using NetworkX's optimized algorithm.
#         This is called once and cached for reuse across all destinations.
        
#         Returns:
#             all_shortest_paths: Dict[source][target] = distance
#         """
#         if self.all_shortest_paths is not None:
#             return self.all_shortest_paths
        
#         print("[Optimized: Computing all-pairs shortest paths (using NetworkX)...]")
#         start_time = time.time()
        
#         # Use NetworkX's optimized all-pairs shortest path length
#         # This uses BFS for unweighted graphs (most efficient)
#         shortest_paths = {}
#         for source, lengths in nx.all_pairs_shortest_path_length(self.graph):
#             shortest_paths[source] = dict(lengths)
        
#         self.all_shortest_paths = shortest_paths
#         elapsed = time.time() - start_time
#         print(f"[All-pairs shortest paths computed in {elapsed:.3f}s]")
        
#         return shortest_paths
    
#     def bfs_nodes_reaching_i(self, destination):
#         """
#         Get distance labels from all nodes to destination using cached all-pairs paths.
#         Optimized version that leverages pre-computed shortest paths.
        
#         Returns distance labels and reachable nodes for a destination.
#         """
#         if self.all_shortest_paths is None:
#             self._compute_all_shortest_paths()
        
#         distance_labels = {}
#         reachable_nodes = {destination}
        
#         # Extract distances to this destination from cache
#         for source in self.graph.nodes():
#             if destination in self.all_shortest_paths[source]:
#                 distance_labels[source] = self.all_shortest_paths[source][destination]
#                 if distance_labels[source] < float('inf'):
#                     reachable_nodes.add(source)
#             else:
#                 distance_labels[source] = float('inf')
        
#         return distance_labels, reachable_nodes
    
#     def enumerate_phase1(self):
#         """
#         Phase 1: Enumerate feasible tuple states using bounded reverse-path expansion.

#         - State tuple format: (destination, pred_1, ..., pred_m)
#         - Enumerate reverse paths with BFS up to tuple length m+1
#         - Pad shorter tuples with dummy node (X)
#         - Enforce no repeated non-dummy nodes
#         - Keep only states that are VRPTW-feasible on the exact implied forward order

#         Returns:
#             M: dict mapping destination i -> set of feasible tuple states
#         """
#         print("\n[Phase 1: Starting path enumeration]")
#         phase1_start = time.time()
        
#         if self.graph is None:
#             self.create_undirected_graph()
        
#         self.remove_infeasible_nodes()

#         M = {}
#         n = self.params.nbclients  # End depot index in this dataset format
#         m = self.m
#         depot = 0
#         end_depot = self.params.nbclients
#         dummy_node = -1
#         tuple_len = m + 1

#         verybig_cutoff = self.params.verybig - 1e-6
#         available_nodes = sorted(self.graph.nodes())

#         # Precompute predecessor lists for fast reverse expansion.
#         predecessors = {}
#         for node in available_nodes:
#             preds = []
#             for pred in available_nodes:
#                 if pred == node:
#                     continue
#                 if self.params.dist[pred][node] < verybig_cutoff:
#                     preds.append(pred)
#             predecessors[node] = preds
        
#         print(f"[Using m={m} predecessors for path enumeration]")
        
#         for i in range(1, n + 1):
#             print(f"\n[Enumerating M_{i}]")

#             queue = deque([(i, [i])])
#             visited_prefixes = {tuple([i])}
#             M_i_feasible = set()

#             while queue:
#                 current, reverse_path = queue.popleft()

#                 if len(reverse_path) > tuple_len:
#                     continue

#                 # Keep only simple states on non-dummy nodes.
#                 if len(reverse_path) != len(set(reverse_path)):
#                     continue

#                 # Filter by exact forward-order VRPTW feasibility early.
#                 path_tuple = tuple(reverse_path)
#                 if self._filter_infeasible_paths_cartesian(i, {path_tuple}, dummy_node, m):
#                     padded = list(reverse_path)
#                     if len(padded) < tuple_len:
#                         padded.extend([dummy_node] * (tuple_len - len(padded)))
#                     M_i_feasible.add(tuple(padded))

#                 if len(reverse_path) == tuple_len:
#                     continue

#                 for pred in predecessors.get(current, []):
#                     if pred in reverse_path:
#                         continue
#                     next_path = reverse_path + [pred]
#                     key = tuple(next_path)
#                     if key in visited_prefixes:
#                         continue
#                     visited_prefixes.add(key)
#                     queue.append((pred, next_path))
            
#             # Add connector nodes (i, 0, 0, ..., 0) for customer destinations only.
#             # These are used to connect the single source (0, -1, ..., -1)
#             # to customer states in the expanded graph.
#             if 1 <= i < end_depot:
#                 connector_node = (i,) + (depot,) * m
#                 M_i_feasible.add(connector_node)
            
#             M[i] = M_i_feasible
#             print(f"  M_{i} feasible paths after filtering: {len(M_i_feasible)}")
        
#         # Add the single source node (0, -1, ..., -1) explicitly.
#         depot_origin = (depot,) + (dummy_node,) * m
#         M[depot] = {depot_origin}
#         print(f"\n[Single origin node added: {depot_origin}]")
        
#         # Display all feasible paths from Cartesian products
#         if self.verbose:
#             print("\n" + "="*70)
#             print("PHASE 1: COMPLETE CARTESIAN PRODUCT OUTPUT")
#             print("="*70)
#         total_paths = 0
#         for i in sorted(M.keys()):
#             total_paths += len(M[i])
#             if self.verbose:
#                 print(f"\nM_{i}: {len(M[i])} feasible paths")
#                 if M[i]:
#                     print("  Paths (in reverse order - destination first):")
#                     for path_tuple in sorted(M[i])[:20]:
#                         formatted_path = " -> ".join(str(n) for n in path_tuple)
#                         print(f"    ({formatted_path})")
#                     if len(M[i]) > 20:
#                         print(f"    ... and {len(M[i]) - 20} more paths")
        
#         print(f"\n{'='*70}")
#         print(f"TOTAL FEASIBLE PATHS (across all destinations): {total_paths}")
#         print(f"{'='*70}\n")
        
#         phase1_elapsed = time.time() - phase1_start
#         print(f"[Phase 1 completed in {phase1_elapsed:.3f} seconds]")
        
#         self.M = M
#         return M
    
#     def _filter_infeasible_paths_cartesian(self, destination, path_tuples, dummy_node, m):
#         """
#         Filter infeasible path tuples from the Cartesian product.
        
#         Each tuple represents: (dest, node_at_dist_1, node_at_dist_2, ..., node_at_dist_m)
#         Example for m=1: (3, 7) means destination 3 with predecessor 7
#         Path in forward order: depot -> 7 -> 3
        
#         Dummy nodes (X) are ignored.
        
#         A path is feasible if:
#         1. Total demand ≤ vehicle capacity
#         2. Time windows satisfied for all nodes
        
#         Args:
#             destination: Target node i
#             path_tuples: Set of path tuples from Cartesian product
#             dummy_node: Dummy node identifier for padding
#             m: Number of predecessors
            
#         Returns:
#             feasible_paths: Set of feasible path tuples
#         """
#         feasible_paths = set()
#         depot = 0
#         capacity = self.params.capacity
#         invalid_count = 0
        
#         for path_tuple in path_tuples:
#             # Tuple format: (dest, pred_at_dist_1, pred_at_dist_2, ..., pred_at_dist_m)
#             dest_node = path_tuple[0]
            
#             # Get predecessors (all elements except first = destination)
#             all_predecessors = list(path_tuple[1:])
            
#             # Filter out dummies and keep only real nodes
#             actual_predecessors = [n for n in all_predecessors if n != dummy_node and n >= 0]

#             # Skip tuples that repeat a node (including repeating the destination)
#             if len(actual_predecessors) != len(set(actual_predecessors)) or dest_node in actual_predecessors:
#                 invalid_count += 1
#                 continue
            
#             # Reverse to get forward order (farthest from destination first)
#             forward_path = list(reversed(actual_predecessors))
            
#             # All nodes in the path
#             all_path_nodes = actual_predecessors + [dest_node]
            
#             # Skip if only destination (no predecessors)
#             if not actual_predecessors:
#                 if self.params.dist[depot][dest_node] < self.params.verybig - 1e-6:
#                     feasible_paths.add(path_tuple)
#                 continue
            
#             # Check 1: Total demand
#             total_demand = sum(self.params.d[n] for n in all_path_nodes)
#             if total_demand > capacity:
#                 invalid_count += 1
#                 continue
            
#             # Check 2: Time window feasibility
#             if self._is_path_to_destination_feasible_cartesian(dest_node, forward_path):
#                 feasible_paths.add(path_tuple)
#             else:
#                 invalid_count += 1
        
#         if invalid_count > 0:
#             print(f"    [{invalid_count} paths filtered as infeasible (capacity/time window violations)]")
        
#         return feasible_paths
    
#     def _is_path_to_destination_feasible_cartesian(self, destination, intermediate_nodes):
#         """
#         Check feasibility of a path using time windows.
        
#         Path: depot -> intermediate_nodes... -> destination
#         intermediate_nodes is a list of nodes in forward order
        
#         Args:
#             destination: Target node i
#             intermediate_nodes: List of nodes in forward order (empty list if just depot->dest)
            
#         Returns:
#             True if path satisfies time windows, False otherwise
#         """
#         depot = 0
#         current_time = self.params.a[depot] + self.params.s[depot]  # Start at depot earliest time + service time
#         current_node = depot
        
#         # Visit all intermediate nodes
#         path_sequence = list(intermediate_nodes) + [destination]
        
#         for next_node in path_sequence:
#             # Check arc exists
#             if self.params.dist[current_node][next_node] >= self.params.verybig - 1e-6:
#                 return False
            
#             # Calculate arrival time
#             travel_time = self.params.ttime[current_node][next_node]
#             arrival_time = current_time + travel_time
            
#             # Check time window lower bound
#             if arrival_time < self.params.a[next_node]:
#                 arrival_time = self.params.a[next_node]
            
#             # Check time window upper bound
#             if arrival_time > self.params.b[next_node]:
#                 return False
            
#             # Update current time after service
#             current_time = arrival_time + self.params.s[next_node]
#             current_node = next_node
        
#         return True
    
#     # ===========================
#     # PHASE 2: EXPANDED GRAPH
#     # ===========================
    
#     def enumerate_phase2(self):
#         """
#         Phase 2: Create expanded network G = (N, A).
        
#         Nodes represent feasible partial paths (customers in reverse order).
#         For example, a node (3,4,1) means the path: depot -> 1 -> 4 -> 3 -> destination
        
#         Steps:
#         1. Create node set N from all M_i sets (Phase 1 output)
#         2. Prune infeasible nodes (capacity & time window check)
#         3. Create edges between compatible nodes
#         4. Prune infeasible edges (combined path feasibility check)
        
#         Returns:
#             expanded_graph: NetworkX graph with path nodes and edges
#         """
#         if not self.M:
#             raise ValueError("Phase 1 must be completed first. Call enumerate_phase1()")
        
#         print("\n" + "="*60)
#         print("PHASE 2: EXPANDED GRAPH CREATION")
#         print("="*60)
        
#         phase2_start = time.time()
        
#         # Create expanded graph
#         expanded_graph = nx.MultiDiGraph()  # Allow multiple edges between same nodes
        
#         # Step 1: Create nodes from all M_i sets
#         # Each M_i contains frozensets (paths), so we collect all unique paths
#         all_paths_by_destination = {}  # {destination: set of paths (frozensets)}
#         all_paths_set = set()  # All unique paths across all destinations
        
#         for dest, paths in self.M.items():
#             all_paths_by_destination[dest] = paths
#             all_paths_set.update(paths)
        
#         print(f"\n[Created {len(all_paths_set)} unique path nodes from Phase 1]")
        
#         # Step 2: Add nodes to graph
#         feasible_paths = set()
#         for path in all_paths_set:
#             if self._is_node_feasible(path):
#                 feasible_paths.add(path)
#                 expanded_graph.add_node(path, 
#                                        path=path,
#                                        demand_sum=self._get_path_demand(path),
#                                        time_cost=self._get_path_time_cost(path))
        
#         print(f"[After node pruning: {len(feasible_paths)} feasible nodes]")
        
#         # Step 2b: Add ONLY the sink destination node (end_depot,).
#         sink_node = (self.params.nbclients,)
#         expanded_graph.add_node(sink_node, node_type='destination', customer=self.params.nbclients)
#         print(f"[Added single destination node: {sink_node}]")
#         print(f"[No direct tuple->sink dummy arcs are added; only compatibility-rule arcs are kept]")
        
#         # Step 3 & 4: Create edges between compatible paths
#         # org.py compatibility rule: for parent arc (i,j), states k∈M_i and l∈M_j
#         # are compatible iff k[0] == l[-1], and arc (i,j) is not already embedded
#         # in either state tuple as an adjacent pair.
        
#         edge_count = 0
#         potential_edge_count = 0  # Count edges before pruning
#         feasible_paths_list = list(feasible_paths)
        
#         # Build destination -> states mapping.
#         dest_to_paths = {}
#         for path_tuple in feasible_paths_list:
#             dest = path_tuple[0]  # First element is destination
#             if dest not in dest_to_paths:
#                 dest_to_paths[dest] = []
#             dest_to_paths[dest].append(path_tuple)

#         # Cache forward paths and tuple-adjacent arcs; index states by exact last label.
#         forward_path_cache = {}
#         tuple_adjacent_arc_cache = {}
#         l_index_by_dest_and_last_label = {}
#         for dest, paths in dest_to_paths.items():
#             index_map = {}
#             for path_tuple in paths:
#                 forward_path_cache[path_tuple] = self._tuple_to_forward_path(path_tuple)
#                 tuple_adjacent_arc_cache[path_tuple] = set(
#                     (path_tuple[t], path_tuple[t + 1]) for t in range(len(path_tuple) - 1)
#                 )
#                 last_label = path_tuple[-1]
#                 index_map.setdefault(last_label, []).append(path_tuple)
#             l_index_by_dest_and_last_label[dest] = index_map

#         # Arc coverage tracking: for every original arc (i,j),
#         # count compatible pairs before and after feasibility pruning.
#         arc_coverage = {}  # { (i,j): {'before': int, 'after': int} }

#         verybig_cutoff = self.params.verybig - 1e-6

#         # Iterate only over feasible original arcs whose endpoint destinations exist.
#         original_arcs = []
#         for i in dest_to_paths:
#             for j in dest_to_paths:
#                 if i == j:
#                     continue
#                 if self.params.dist[i][j] < verybig_cutoff:
#                     original_arcs.append((i, j))
#                     arc_coverage[(i, j)] = {'before': 0, 'after': 0}

#         for i, j in original_arcs:
#             candidate_l_paths = l_index_by_dest_and_last_label.get(j, {}).get(i, [])
#             if not candidate_l_paths:
#                 continue

#             for path_k in dest_to_paths[i]:
#                 k_forward = forward_path_cache[path_k]
#                 if not k_forward:
#                     continue

#                 for path_l in candidate_l_paths:
#                     # org.py rule: skip if parent arc already appears in either state tuple.
#                     if (i, j) in tuple_adjacent_arc_cache[path_k] or (i, j) in tuple_adjacent_arc_cache[path_l]:
#                         continue

#                     l_forward = forward_path_cache[path_l]
#                     if not l_forward:
#                         continue

#                     if k_forward == [0]:
#                         merged_nodes = l_forward[1:]
#                     else:
#                         merged_nodes = k_forward + l_forward[1:]

#                     potential_edge_count += 1
#                     arc_coverage[(i, j)]['before'] += 1

#                     if self._is_merged_path_feasible(merged_nodes):
#                         expanded_graph.add_edge(path_k, path_l,
#                                                parent_arc=(i, j),
#                                                merged_path=merged_nodes,
#                                                distance=self.params.dist[i][j],
#                                                travel_time=self.params.ttime[i][j])
#                         edge_count += 1
#                         arc_coverage[(i, j)]['after'] += 1

#         # Special-case sink compatibility:
#         # Treat sink node (end_depot,) as having label "end_depot" so every
#         # tuple node (end_depot, ...) can connect to it by label matching.
#         # This does NOT add generic tuple->sink arcs from other destinations.
#         sink_dest = self.params.nbclients
#         sink_compat_edges = 0
#         for path_k in dest_to_paths.get(sink_dest, []):
#             if path_k == sink_node:
#                 continue
#             expanded_graph.add_edge(path_k, sink_node,
#                                    parent_arc=(sink_dest, sink_dest),
#                                    edge_type='sink_label_match',
#                                    distance=0.0,
#                                    travel_time=0.0)
#             edge_count += 1
#             potential_edge_count += 1
#             sink_compat_edges += 1

#         if sink_compat_edges > 0:
#             print(f"[Added {sink_compat_edges} label-matched edges: (end_depot,...) -> ({sink_dest},)]")

#         # Aggressive Phase 2 pruning: keep only tuple states that can still
#         # reach the sink node through currently feasible edges.
#         removed_nodes, removed_edges = self._prune_nodes_not_reaching_sink(expanded_graph, sink_node)
#         if removed_nodes > 0 or removed_edges > 0:
#             print(
#                 f"[Aggressive Phase 2 pruning] Removed {removed_nodes} nodes and {removed_edges} edges "
#                 f"(cannot reach sink {sink_node})"
#             )

#         # Refresh edge count after aggressive reachability pruning.
#         edge_count = expanded_graph.number_of_edges()

#         print(f"[Edge creation: {edge_count} edges created after pruning, {potential_edge_count} potential edges before pruning]")
#         print(f"[Edge pruning ratio: {(potential_edge_count - edge_count) / potential_edge_count * 100:.2f}% edges pruned]" if potential_edge_count > 0 else "[No edges to prune]")
#         print(f"[Final expanded graph: {expanded_graph.number_of_nodes()} nodes, {expanded_graph.number_of_edges()} edges]")

#         # === ARC COVERAGE REPORT ===
#         print(f"\n{'='*70}")
#         print(f"ARC COVERAGE REPORT (original arcs with Phase 1 nodes on both ends)")
#         print(f"{'='*70}")
#         print(f"{'Arc (i->j)':<14} {'Before Pruning':>15} {'After Pruning':>14} {'Pruned':>8} {'Status'}")
#         print(f"{'-'*70}")
#         arcs_with_no_compatible = []
#         arcs_with_compatible = []
#         for arc in sorted(arc_coverage.keys()):
#             before = arc_coverage[arc]['before']
#             after = arc_coverage[arc]['after']
#             pruned = before - after
#             if before == 0:
#                 arcs_with_no_compatible.append(arc)
#                 status = "NO COMPATIBLE PAIRS"
#             elif after == 0:
#                 status = "ALL PRUNED"
#                 arcs_with_compatible.append(arc)
#             else:
#                 status = "OK"
#                 arcs_with_compatible.append(arc)
#             print(f"  ({arc[0]:2d},{arc[1]:2d})       {before:>15}  {after:>14}  {pruned:>8}   {status}")
#         print(f"{'-'*70}")
#         print(f"Arcs with at least one compatible pair (before pruning): {len(arcs_with_compatible)}")
#         print(f"Arcs with NO compatible pairs at all:                    {len(arcs_with_no_compatible)}")
#         if arcs_with_no_compatible:
#             print(f"  Arcs not covered: {arcs_with_no_compatible}")
#         print(f"{'='*70}\n")
        
#         # Store for metrics reporting
#         self.phase2_edges_before_pruning = potential_edge_count
#         self.phase2_edges_after_pruning = edge_count
        
#         phase2_elapsed = time.time() - phase2_start
#         print(f"[Phase 2 completed in {phase2_elapsed:.3f} seconds]")
        
#         self.expanded_graph = expanded_graph
#         return expanded_graph

#     def print_phase2_graph(self, output_file=None):
#         """
#         Print all nodes and arcs of the Phase 2 expanded graph after pruning.
#         Categorises nodes as: single origin node (0,-1,...,-1), single destination node (end_depot,),
#         connector nodes (i,0,...,0) for customers, and other tuple nodes.
#         Categorises arcs as: dummy arcs (any tuple -> (end_depot,)) and real Phase 2 arcs.

#         Args:
#             output_file: If given, write to this file path. Otherwise print to stdout.
#         """
#         if not hasattr(self, 'expanded_graph') or self.expanded_graph is None:
#             print("Phase 2 has not been run yet. Call enumerate_phase2() first.")
#             return

#         G = self.expanded_graph
#         lines = []

#         sink = self.params.nbclients
#         dest_nodes = sorted([n for n in G.nodes() if isinstance(n, tuple) and len(n) == 1 and n[0] == sink])
#         origin_nodes = sorted([n for n in G.nodes() if isinstance(n, tuple) and len(n) > 1
#                     and n[0] == 0 and all(x == -1 for x in n[1:])])
#         connector_nodes = sorted([n for n in G.nodes() if isinstance(n, tuple) and len(n) > 1
#                       and n[0] != 0 and all(x == 0 for x in n[1:])])
#         other_nodes = sorted([n for n in G.nodes() if isinstance(n, tuple) and len(n) > 1
#                       and n not in origin_nodes and n not in connector_nodes])

#         lines.append("=" * 70)
#         lines.append(f"PHASE 2 EXPANDED GRAPH  (m={self.m})")
#         lines.append("=" * 70)

#         lines.append(f"\n--- DESTINATION NODE (sink={sink})  [{len(dest_nodes)} total] ---")
#         for n in dest_nodes:
#             lines.append(f"  {n}")

#         lines.append(f"\n--- ORIGIN NODE (0,-1,...,-1)  [{len(origin_nodes)} total] ---")
#         for n in origin_nodes:
#             lines.append(f"  {n}")

#         lines.append(f"\n--- CONNECTOR NODES (i,0,...,0)  [{len(connector_nodes)} total] ---")
#         for n in connector_nodes:
#             lines.append(f"  {n}")

#         lines.append(f"\n--- OTHER TUPLE NODES (i, pred1, ...)  [{len(other_nodes)} total] ---")
#         for n in other_nodes:
#             lines.append(f"  {n}")

#         dummy_arcs = []
#         real_arcs  = []
#         for u, v, d in G.edges(data=True):
#             if d.get('edge_type') == 'dummy':
#                 dummy_arcs.append((u, v))
#             else:
#                 real_arcs.append((u, v, d.get('parent_arc')))

#         lines.append(f"\n--- DUMMY ARCS  tuple(any dest) -> ({sink},)  cost=0  [{len(dummy_arcs)} total] ---")
#         for u, v in sorted(dummy_arcs):
#             lines.append(f"  {u}  ->  {v}")

#         lines.append(f"\n--- REAL PHASE 2 ARCS  tuple -> tuple  [{len(real_arcs)} total] ---")
#         for u, v, arc in sorted(real_arcs):
#             lines.append(f"  {u}  ->  {v}   [parent arc {arc}]")

#         lines.append("\n" + "=" * 70)
#         lines.append(f"SUMMARY")
#         lines.append(f"  Destination nodes : {len(dest_nodes)}")
#         lines.append(f"  Origin nodes      : {len(origin_nodes)}")
#         lines.append(f"  Connector nodes   : {len(connector_nodes)}")
#         lines.append(f"  Other tuple nodes : {len(other_nodes)}")
#         lines.append(f"  Dummy arcs        : {len(dummy_arcs)}")
#         lines.append(f"  Real arcs         : {len(real_arcs)}")
#         lines.append("=" * 70)

#         text = "\n".join(lines) + "\n"

#         if output_file:
#             import os
#             os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
#             with open(output_file, "w", encoding="utf-8") as f:
#                 f.write(text)
#             print(f"[Phase 2 graph written to: {output_file}]")
#         else:
#             print(text)

#     def _forward_order(self, path_set):
#         """Convert a frozenset path to forward order (starting from depot)."""
#         # For now, just sort numerically to get a consistent order
#         return sorted(path_set)

#     def _tuple_to_forward_path(self, path_tuple):
#         """
#         Convert tuple node format to exact forward path sequence.

#         Tuple format: (destination, pred_1, pred_2, ..., pred_m)
#         Forward path: [pred_m, ..., pred_2, pred_1, destination]
#         Dummy predecessors (-1) and depot-padding predecessors (0) are ignored.
#         """
#         destination = path_tuple[0]
#         # Connector tuples use 0 as padding (i, 0, ..., 0). These are not
#         # traversed customer visits and must not create artificial 0->0 arcs.
#         predecessors = [node for node in path_tuple[1:] if node > 0]
#         return list(reversed(predecessors)) + [destination]
    
    
#     def _is_node_feasible(self, path):
#         """
#         Check if a tuple node is feasible in Phase 2.

#         We keep only tuple states whose exact forward path is feasible with
#         respect to arc existence, time windows and capacity.
        
#         Args:
#             path: tuple state (destination, pred_1, ..., pred_m)
            
#         Returns:
#             True if the tuple node is feasible, False otherwise.
#         """
#         if not isinstance(path, tuple) or len(path) == 0:
#             return False

#         depot = 0
#         sink = self.params.nbclients

#         # Keep only the canonical single origin state for depot destination.
#         if path[0] == depot:
#             return all(x == -1 for x in path[1:])

#         forward_path = self._tuple_to_forward_path(path)
#         if not forward_path:
#             return False

#         # No repeated customer/sink nodes on a single tuple state.
#         visited = [n for n in forward_path if n > 0]
#         if len(visited) != len(set(visited)):
#             return False

#         # Basic destination sanity.
#         if path[0] not in range(1, sink + 1):
#             return False

#         return self._is_merged_path_feasible(forward_path)

#     def _prune_nodes_not_reaching_sink(self, expanded_graph, sink_node):
#         """
#         Remove nodes/arcs that cannot reach sink_node in the current expanded graph.

#         Returns:
#             (removed_node_count, removed_edge_count)
#         """
#         if sink_node not in expanded_graph:
#             return 0, 0

#         before_nodes = expanded_graph.number_of_nodes()
#         before_edges = expanded_graph.number_of_edges()

#         keep_nodes = nx.ancestors(expanded_graph, sink_node)
#         keep_nodes.add(sink_node)

#         to_remove = [n for n in expanded_graph.nodes() if n not in keep_nodes]
#         if to_remove:
#             expanded_graph.remove_nodes_from(to_remove)

#         after_nodes = expanded_graph.number_of_nodes()
#         after_edges = expanded_graph.number_of_edges()
#         return before_nodes - after_nodes, before_edges - after_edges
    
    
    
#     def _is_merged_path_feasible(self, path_nodes):
#         """
#         Check feasibility of a merged partial path.
        
#         The path_nodes are visited in exact order and include customers/end nodes
#         from the merged tuple path. We check:
#         1. Total capacity is not exceeded
#         2. The exact sequence satisfies arc existence and time windows
        
#         Args:
#             path_nodes: Ordered list of nodes in exact merged path order
            
#         Returns:
#             True if feasible, False otherwise
#         """
#         if not path_nodes:
#             return True
        
#         # Check 1: Total demand (ignore depot nodes)
#         total_demand = sum(self.params.d[n] for n in path_nodes if n != 0 and n != self.params.nbclients)
#         if total_demand > self.params.capacity:
#             return False

#         # Check 2: Time window feasibility on the exact given order
#         depot = 0
#         current_time = self.params.a[depot]
#         current_node = depot

#         for next_node in path_nodes:
#             if self.params.dist[current_node][next_node] >= self.params.verybig - 1e-6:
#                 return False

#             arrival_time = current_time + self.params.ttime[current_node][next_node]
#             if arrival_time < self.params.a[next_node]:
#                 arrival_time = self.params.a[next_node]
#             if arrival_time > self.params.b[next_node]:
#                 return False

#             current_time = arrival_time + self.params.s[next_node]
#             current_node = next_node
        
#         return True
    
    
#     def _get_path_demand(self, path):
#         """Get total demand of a path (frozenset)."""
#         return sum(self.params.d[node] for node in path)
    
#     def _get_path_time_cost(self, path):
#         """
#         Get total time cost of a path from depot.
#         Uses greedy nearest-neighbor ordering.
        
#         Args:
#             path: frozenset of nodes
            
#         Returns:
#             Total time from depot through all nodes (cumulative end time)
#         """
#         if not path:
#             return 0
        
#         depot = 0
#         current_time = self.params.a[depot]
#         current_node = depot
#         visited = {depot}
#         remaining = set(path)
        
#         while remaining:
#             nearest_node = None
#             best_arrival_time = float('inf')
            
#             for next_node in list(remaining):  # Convert to list to avoid mutation error
#                 if self.params.dist[current_node][next_node] >= self.params.verybig - 1e-6:
#                     continue
                
#                 arrival_time = current_time + self.params.ttime[current_node][next_node]
#                 arrival_time = max(arrival_time, self.params.a[next_node])
                
#                 if arrival_time <= self.params.b[next_node] and arrival_time < best_arrival_time:
#                     best_arrival_time = arrival_time
#                     nearest_node = next_node
            
#             if nearest_node is None:
#                 return float('inf')
            
#             current_time = best_arrival_time + self.params.s[nearest_node]
#             current_node = nearest_node
#             visited.add(nearest_node)
#             remaining.discard(nearest_node)
        
#         return current_time
    
    
#     def compare_graphs(self):
#         """
#         Compare sparsity between original graph and expanded graph.
        
#         Metrics:
#         - Number of nodes
#         - Number of edges
#         - Graph density
#         - Average degree
#         """
#         if self.graph is None or self.expanded_graph is None:
#             self._log_metrics("[Error] Both Phase 1 and Phase 2 must be completed")
#             return
        
#         self._log_metrics("\n" + "="*60)
#         self._log_metrics("GRAPH SPARSITY COMPARISON")
#         self._log_metrics("="*60)
        
#         # Phase 1 Graph Statistics
#         g1_nodes = self.graph.number_of_nodes()
#         g1_edges = self.graph.number_of_edges()
#         g1_density = nx.density(self.graph)
#         g1_avg_degree = 2 * g1_edges / g1_nodes if g1_nodes > 0 else 0
        
#         self._log_metrics(f"\nPhase 1 Graph (Original Network):")
#         self._log_metrics(f"  Nodes: {g1_nodes}")
#         self._log_metrics(f"  Edges: {g1_edges}")
#         self._log_metrics(f"  Density: {g1_density:.6f}")
#         self._log_metrics(f"  Average Degree: {g1_avg_degree:.2f}")
        
#         # Phase 2 Graph Statistics
#         g2_nodes = self.expanded_graph.number_of_nodes()
#         g2_edges = self.expanded_graph.number_of_edges()
#         g2_density = nx.density(self.expanded_graph)
#         g2_avg_degree = g2_edges / g2_nodes if g2_nodes > 0 else 0
        
#         self._log_metrics(f"\nPhase 2 Graph (Expanded Network):")
#         self._log_metrics(f"  Nodes: {g2_nodes}")
#         self._log_metrics(f"  Edges (before pruning): {getattr(self, 'phase2_edges_before_pruning', 'N/A')}")
#         self._log_metrics(f"  Edges (after pruning): {g2_edges}")
#         if hasattr(self, 'phase2_edges_before_pruning') and self.phase2_edges_before_pruning > 0:
#             pruning_ratio = (self.phase2_edges_before_pruning - g2_edges) / self.phase2_edges_before_pruning * 100
#             self._log_metrics(f"  Edge pruning ratio: {pruning_ratio:.2f}%")
#         self._log_metrics(f"  Density: {g2_density:.6f}")
#         self._log_metrics(f"  Average Degree: {g2_avg_degree:.2f}")
        
#         # Comparison
#         self._log_metrics(f"\nComparison:")
#         self._log_metrics(f"  Node ratio (Phase2/Phase1): {g2_nodes/g1_nodes:.2f}x")
#         self._log_metrics(f"  Edge ratio (Phase2/Phase1): {g2_edges/g1_edges:.2f}x" if g1_edges > 0 else "  Edge ratio: inf")
#         self._log_metrics(f"  Density ratio (Phase2/Phase1): {g2_density/g1_density:.4f}" if g1_density > 0 else "  Density ratio: inf")
    
#     def compute_degree_stats(self, graph_name, graph):
#         """
#         Compute and return minimum and maximum degree for a graph.
        
#         Args:
#             graph_name: Name of the graph (for printing)
#             graph: NetworkX graph object
            
#         Returns:
#             tuple: (min_degree, max_degree, avg_degree)
#         """
#         if graph.number_of_nodes() == 0:
#             self._log_metrics(f"[{graph_name}] No nodes in graph")
#             return 0, 0, 0
        
#         degrees = [degree for node, degree in graph.degree()]
        
#         min_degree = min(degrees) if degrees else 0
#         max_degree = max(degrees) if degrees else 0
#         avg_degree = sum(degrees) / len(degrees) if degrees else 0
        
#         self._log_metrics(f"\n[{graph_name}]")
#         self._log_metrics(f"  Minimum degree: {min_degree}")
#         self._log_metrics(f"  Maximum degree: {max_degree}")
#         self._log_metrics(f"  Average degree: {avg_degree:.2f}")
        
#         # Degree distribution
#         degree_counts = {}
#         for d in degrees:
#             degree_counts[d] = degree_counts.get(d, 0) + 1
#         self._log_metrics(f"  Degree distribution: {sorted(degree_counts.items())}")
        
#         return min_degree, max_degree, avg_degree
    
#     def print_summary(self):
#         """Print summary of Phase 1 enumeration."""
#         print("\n" + "="*60)
#         print("PHASE 1 ENUMERATION SUMMARY")
#         print("="*60)
        
#         total_paths = sum(len(paths) for paths in self.M.values())
#         print(f"\nTotal feasible paths: {total_paths}")
        
#         for node in sorted(self.M.keys()):
#             num_paths = len(self.M[node])
#             if num_paths > 0:
#                 print(f"\nM_{node}: {num_paths} feasible paths")
#                 for path in sorted(self.M[node]):
#                     if len(path) == 0:
#                         print(f"  - (empty path - direct depot to {node})")
#                     else:
#                         path_str = " -> ".join(str(n) for n in sorted(path))
#                         print(f"  - {path_str}")
