import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from branchBound import BranchAndBound
from Common.paramsVRP import ParamsVRP
from Common.route import Route
from Common.logcapture import start_logging
import time
import heapq


def savings_heuristic_vrptw(user_param):
    """
    Savings heuristic for VRPTW based on Clarke and Wright (1964).
    Extended to handle time windows.
    
    :param user_param: ParamsVRP object with problem data
    :return: List of initial routes
    """
    print(f"[Initializing routes with Savings Heuristic]")

    def is_route_feasible(path):
        # Check capacity and time windows along the full path (includes end depot).
        total_demand = sum(user_param.d[c] for c in path[1:-1])
        if total_demand > user_param.capacity:
            return False

        current_time = user_param.a[0]
        for idx in range(1, len(path)):
            prev_customer = path[idx - 1]
            customer = path[idx]
            current_time += user_param.s[prev_customer] + user_param.ttime[prev_customer][customer]
            if current_time < user_param.a[customer]:
                current_time = user_param.a[customer]
            if current_time > user_param.b[customer]:
                return False
        return True
    
    # Step 1: Start with n routes (one customer per vehicle)
    routes = []
    for i in range(1, user_param.nbclients):
        # Check if route depot -> i -> depot is feasible
        if (user_param.dist[0][i] < user_param.verybig - 1e-6 and 
            user_param.dist[i][user_param.nbclients] < user_param.verybig - 1e-6):
            
            # Check time window feasibility
            arrival_time = user_param.a[0] + user_param.ttime[0][i]
            if arrival_time < user_param.a[i]:
                arrival_time = user_param.a[i]
            
            departure_time = arrival_time + user_param.s[i]
            time_to_depot = departure_time + user_param.ttime[i][user_param.nbclients]
            
            if arrival_time <= user_param.b[i] and time_to_depot <= user_param.b[user_param.nbclients]:
                route_cost = user_param.dist[0][i] + user_param.dist[i][user_param.nbclients]
                route = Route(path=[0, i, user_param.nbclients], cost=route_cost, Q=1.0)
                routes.append(route)
    
    print(f"  [Initial single-customer routes: {len(routes)}]")
    
    # Step 2: Calculate savings for all pairs of routes
    # Savings s_ij = d_i0 + d_0j - d_ij (distance saved by combining routes)
    savings_list = []

    def push_savings(r1_idx, r2_idx, order_flag):
        # order_flag: 0 means r1 before r2, 1 means r2 before r1
        if order_flag == 0:
            route1 = routes[r1_idx]
            route2 = routes[r2_idx]
        else:
            route1 = routes[r2_idx]
            route2 = routes[r1_idx]

        i = route1.path[-2]
        j = route2.path[1]
        if user_param.dist[i][j] < user_param.verybig - 1e-6:
            saving = (user_param.dist[i][user_param.nbclients] +
                     user_param.dist[0][j] - user_param.dist[i][j])
            heapq.heappush(savings_list, (-saving, r1_idx, r2_idx, i, j, order_flag))

    for r1_idx in range(len(routes)):
        for r2_idx in range(r1_idx + 1, len(routes)):
            push_savings(r1_idx, r2_idx, 0)
            push_savings(r1_idx, r2_idx, 1)

    print(f"  [Calculated {len(savings_list)} potential merges]")
    
    # Step 3: Iteratively merge routes based on savings
    merged_count = 0
    route_active = [True] * len(routes)  # Track which routes are still active
    initial_route_count = len(routes)
    
    while savings_list:
        neg_saving, r1_idx, r2_idx, i, j, order_flag = heapq.heappop(savings_list)
        saving = -neg_saving
        
        # Check if both routes are still active
        if not route_active[r1_idx] or not route_active[r2_idx]:
            continue

        if order_flag == 0:
            route1 = routes[r1_idx]
            route2 = routes[r2_idx]
        else:
            route1 = routes[r2_idx]
            route2 = routes[r1_idx]

        # Check if i is last customer in route1 and j is first customer in route2
        if route1.path[-2] != i or route2.path[1] != j:
            continue

        # Create merged route
        merged_path = route1.path[:-1] + route2.path[1:]  # [0, ..., i, j, ..., nbclients]

        if is_route_feasible(merged_path):
            # Merge routes
            merged_cost = sum(user_param.dist[merged_path[k]][merged_path[k+1]] 
                            for k in range(len(merged_path) - 1))
            merged_route = Route(path=merged_path, cost=merged_cost, Q=1.0)
            
            # Add merged route and mark originals as inactive
            routes.append(merged_route)
            route_active.append(True)
            route_active[r1_idx] = False
            route_active[r2_idx] = False
            
            merged_count += 1

            # Add savings for the new route against all active routes
            new_idx = len(routes) - 1
            for other_idx in range(len(routes) - 1):
                if not route_active[other_idx]:
                    continue
                if other_idx == new_idx:
                    continue
                push_savings(new_idx, other_idx, 0)
                push_savings(new_idx, other_idx, 1)
    
    # Filter out inactive routes
    final_routes = [routes[i] for i in range(len(routes)) if route_active[i]]
    
    print(f"  [Savings heuristic complete: {merged_count} merges performed]")
    print(f"  [Final routes: {len(final_routes)} (reduced from {initial_route_count})]")
    
    return final_routes


def solomon_insertion_heuristic(user_param, alpha1=1.0, alpha2=0.0, mu=1.0, lambda_param=2.0):
    """
    Solomon's I1 insertion heuristic for VRPTW (Solomon 1987).
    Builds routes by iteratively inserting customers at positions that minimize
    a weighted combination of distance and time window criteria.
    
    :param user_param: ParamsVRP object with problem data
    :param alpha1: Weight for distance criterion (default: 1.0)
    :param alpha2: Weight for time criterion (default: 0.0) 
    :param mu: Weight for distance to depot in seed selection (default: 1.0)
    :param lambda_param: Weight for best feasible insertion criterion (default: 2.0)
    :return: List of initial routes
    """
    print(f"[Initializing routes with Solomon's Insertion Heuristic]")
    print(f"  [Parameters: alpha1={alpha1}, alpha2={alpha2}, mu={mu}, lambda={lambda_param}]")
    
    depot = 0
    end_depot = user_param.nbclients
    
    # Track unrouted customers
    unrouted = set(range(1, user_param.nbclients))
    routes = []
    
    def calculate_time_at_customer(route_path, position):
        """Calculate arrival time at customer at given position in route."""
        current_time = user_param.a[depot]
        for idx in range(1, position + 1):
            prev = route_path[idx - 1]
            curr = route_path[idx]
            current_time += user_param.s[prev] +user_param.ttime[prev][curr]
            if current_time < user_param.a[curr]:
                current_time = user_param.a[curr]
        return current_time
    
    def check_insertion_feasibility(route_path, customer, insert_pos):
        """Check if inserting customer at insert_pos is feasible."""
        # Create temporary path with customer inserted
        temp_path = route_path[:insert_pos] + [customer] + route_path[insert_pos:]
        
        # Check capacity
        total_demand = sum(user_param.d[c] for c in temp_path[1:-1])
        if total_demand > user_param.capacity:
            return False, None, None
        
        # Check time windows for all customers from insertion point onwards
        current_time = user_param.a[depot]
        for idx in range(1, len(temp_path)):
            prev = temp_path[idx - 1]
            curr = temp_path[idx]
            current_time += user_param.s[prev] + user_param.ttime[prev][curr]
            
            if current_time < user_param.a[curr]:
                current_time = user_param.a[curr]
            
            if current_time > user_param.b[curr]:
                return False, None, None
        
        return True, temp_path, current_time
    
    def calculate_insertion_cost(route_path, customer, insert_pos):
        """
        Calculate insertion cost for inserting customer at insert_pos.
        Uses Solomon's c11 criterion (generalized cost).
        """
        i = route_path[insert_pos - 1]
        j = route_path[insert_pos]
        u = customer
        
        # c1(i,u,j): Distance-based cost
        # = d(i,u) + d(u,j) - d(i,j)
        c1 = (user_param.dist[i][u] + user_param.dist[u][j] - 
              user_param.dist[i][j])
        
        # c2(i,u,j): Time-based cost  
        # = b(u) - (t(i) + t(i,u)) where t(i) is arrival time at i
        # This measures how late we can arrive at u
        time_at_i = calculate_time_at_customer(route_path, insert_pos - 1)
        arrival_at_u = time_at_i + user_param.s[i] + user_param.ttime[i][u]
        if arrival_at_u < user_param.a[u]:
            arrival_at_u = user_param.a[u]
        
        c2 = user_param.b[u] - arrival_at_u
        
        # Combined cost: c(i,u,j) = alpha1 * c1(i,u,j) - alpha2 * c2(i,u,j)
        # Note: negative sign for c2 because larger time slack is better
        cost = alpha1 * c1 - alpha2 * c2
        
        return cost
    
    def find_best_insertion(routes, customer):
        """Find the best route and position to insert customer."""
        best_cost = float('inf')
        best_route_idx = None
        best_position = None
        best_path = None
        
        for r_idx, route in enumerate(routes):
            route_path = route.path
            
            # Try inserting at each position (between depot and end_depot)
            for pos in range(1, len(route_path)):
                # Check feasibility
                feasible, temp_path, _ = check_insertion_feasibility(route_path, customer, pos)
                
                if feasible:
                    # Calculate insertion cost
                    ins_cost = calculate_insertion_cost(route_path, customer, pos)
                    
                    if ins_cost < best_cost:
                        best_cost = ins_cost
                        best_route_idx = r_idx
                        best_position = pos
                        best_path = temp_path
        
        return best_route_idx, best_position, best_cost, best_path
    
    def select_seed_customer(unrouted_set):
        """Select seed customer for new route (farthest from depot)."""
        max_dist = -1
        seed = None
        
        for customer in unrouted_set:
            # Use Solomon's criterion: mu * distance to depot + lambda * time to latest arrival
            dist_to_depot = user_param.dist[depot][customer]
            
            if dist_to_depot < user_param.verybig - 1e-6:
                criterion = mu * dist_to_depot + lambda_param * user_param.b[customer]
                
                if criterion > max_dist:
                    max_dist = criterion
                    seed = customer
        
        return seed
    
    def create_route_with_seed(seed):
        """Create a new route with seed customer."""
        route_path = [depot, seed, end_depot]
        
        # Check basic feasibility
        if (user_param.dist[depot][seed] >= user_param.verybig - 1e-6 or
            user_param.dist[seed][end_depot] >= user_param.verybig - 1e-6):
            return None
        
        # Check capacity
        if user_param.d[seed] > user_param.capacity:
            return None
        
        # Check time windows
        arrival_time = user_param.a[depot] + user_param.s[depot] + user_param.ttime[depot][seed]
        if arrival_time < user_param.a[seed]:
            arrival_time = user_param.a[seed]
        
        if arrival_time > user_param.b[seed]:
            return None
        
        time_to_end_depot = arrival_time + user_param.s[seed] + user_param.ttime[seed][end_depot]
        
        if time_to_end_depot > user_param.b[end_depot]:
            return None
        
        # Calculate route cost
        route_cost = user_param.dist[depot][seed] + user_param.dist[seed][end_depot]
        
        return Route(path=route_path, cost=route_cost, Q=1.0)
    
    # Main construction loop
    iteration = 0
    
    while unrouted:
        iteration += 1
        
        if not routes:
            # Initialize first route with seed customer
            seed = select_seed_customer(unrouted)
            if seed is None:
                print(f"  [Warning: No feasible seed customer found]")
                break
            
            route = create_route_with_seed(seed)
            if route is None:
                print(f"  [Warning: Cannot create route with seed {seed}]")
                unrouted.remove(seed)
                continue
            
            routes.append(route)
            unrouted.remove(seed)
            print(f"  [Iteration {iteration}: Created route with seed customer {seed}]")
            continue
        
        # Find best insertion among all unrouted customers
        best_overall_customer = None
        best_overall_route_idx = None
        best_overall_position = None
        best_overall_cost = float('inf')
        best_overall_path = None
        
        for customer in unrouted:
            route_idx, position, cost, path = find_best_insertion(routes, customer)
            
            if route_idx is not None and cost < best_overall_cost:
                best_overall_customer = customer
                best_overall_route_idx = route_idx
                best_overall_position = position
                best_overall_cost = cost
                best_overall_path = path
        
        if best_overall_customer is not None:
            # Insert the customer into the best route
            old_path = routes[best_overall_route_idx].path
            routes[best_overall_route_idx].path = best_overall_path
            
            # Recalculate route cost
            new_cost = sum(user_param.dist[best_overall_path[k]][best_overall_path[k+1]]
                          for k in range(len(best_overall_path) - 1))
            routes[best_overall_route_idx].cost = new_cost
            
            unrouted.remove(best_overall_customer)
            
            if iteration % 10 == 0 or len(unrouted) < 5:
                print(f"  [Iteration {iteration}: Inserted customer {best_overall_customer} "
                      f"at position {best_overall_position} in route {best_overall_route_idx}, "
                      f"remaining: {len(unrouted)}]")
        else:
            # No feasible insertion found, create new route with seed
            seed = select_seed_customer(unrouted)
            if seed is None:
                print(f"  [Warning: No feasible seed customer found for remaining {len(unrouted)} customers]")
                break
            
            route = create_route_with_seed(seed)
            if route is None:
                print(f"  [Warning: Cannot create route with seed {seed}]")
                unrouted.remove(seed)
                continue
            
            routes.append(route)
            unrouted.remove(seed)
            print(f"  [Iteration {iteration}: Created new route with seed customer {seed}, "
                  f"total routes: {len(routes)}]")
    
    print(f"  [Solomon's insertion heuristic complete]")
    print(f"  [Final routes: {len(routes)}, Unrouted customers: {len(unrouted)}]")
    
    if unrouted:
        print(f"  [Warning: Could not route customers: {unrouted}]")
    
    return routes


def main(datasetPath, 
         SHOWFIG,
         enable_column_deletion = False,     # Enable column deletion acceleration
         deletion_threshold = float('inf'),            # Delete routes not in basis for this many BB nodes
         enable_early_stop_pricing = False, # Enable early stopping in SPPRC (disabled by default)
         gap_threshold = None,              # Optional optimality gap to stop early
         heuristic = 'savings',             # Initial heuristic: 'savings' or 'solomon'
         time_limit = None,                 # Time limit in seconds (None for no limit)
         use_expanded_pricing = True,       # Use expanded-state pricing backend
         expanded_max_m = None,             # Optional cap on expanded graph resource states
         expanded_max_routes = 20,          # Max routes generated per expanded pricing iteration
         reduced_graph_min_columns = 5,    # Reduced->full expanded fallback threshold
         use_gnn_pruning = True,            # Apply GNN soft pruning on the expanded graph
         gnn_model_path = None,             # Optional path to a trained GNN checkpoint
         gnn_threshold = 0.55,              # Arc score threshold for pruning
         gnn_keep_ratio = 0.35,             # Fraction of arcs to keep by score
         gnn_top_k_per_node = 6):           # Always retain top-k outgoing arcs per node
    """
    Main function to solve VRPTW using Branch-and-Price with acceleration techniques.
    
    Initial heuristics:
    1. Savings Heuristic (Clarke-Wright 1964): Merges routes based on distance savings
    2. Solomon's Insertion (Solomon 1987): Sequential insertion based on distance/time criteria
    
    :param datasetPath: Path to VRPTW instance file
    :param SHOWFIG: Whether to show visualization
    :param enable_column_deletion: Enable/disable column deletion
    :param deletion_threshold: Number of BB nodes before deleting inactive routes
    :param enable_early_stop_pricing: Enable/disable early stopping in pricing subproblem
    :param gap_threshold: Optional gap threshold override to terminate early (defaults to ParamsVRP.gap)
    :param heuristic: Initial heuristic to use ('savings' or 'solomon')
    :param time_limit: Time limit in seconds (None for no limit)
    :param use_expanded_pricing: Enable expanded-state pricing backend
    :param expanded_max_m: Optional cap on expanded resource states
    :param expanded_max_routes: Max number of routes generated by pricing per iteration
    :param reduced_graph_min_columns: Fallback threshold from reduced expanded graph to full
    :param use_gnn_pruning: Enable GNN-guided pruning on the expanded pricing graph
    :param gnn_model_path: Optional checkpoint for the GNN scorer
    :param gnn_threshold: Probability threshold for keeping an arc
    :param gnn_keep_ratio: Fraction of arcs kept by ranking if threshold is too strict
    :param gnn_top_k_per_node: Retain at most this many outgoing arcs per node
    """
    # Initialize branch and price algorithm with acceleration parameters
    bp = BranchAndBound(enable_column_deletion=enable_column_deletion,
                       deletion_threshold=deletion_threshold,
                       enable_early_stop=enable_early_stop_pricing,
                       gap_threshold=gap_threshold,
                       time_limit=time_limit,
                       use_expanded_pricing=use_expanded_pricing,
                       expanded_max_m=expanded_max_m,
                       expanded_max_routes=expanded_max_routes,
                       reduced_graph_min_columns=reduced_graph_min_columns,
                       use_gnn_pruning=use_gnn_pruning,
                       gnn_model_path=gnn_model_path,
                       gnn_threshold=gnn_threshold,
                       gnn_keep_ratio=gnn_keep_ratio,
                       gnn_top_k_per_node=gnn_top_k_per_node)

    # Initialize problem instance
    user_param = ParamsVRP()
    user_param.init_params(datasetPath)
    
    # Display configuration
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Initial Heuristic:       {heuristic.upper()}")
    print(f"Column Deletion:         {'ENABLED' if enable_column_deletion else 'DISABLED'}")
    if enable_column_deletion:
        print(f"  - Deletion Threshold:  {deletion_threshold} BB nodes")
    print(f"Early Stop Pricing:      {'ENABLED' if enable_early_stop_pricing else 'DISABLED'}")
    print(f"Expanded Pricing:        {'ENABLED' if use_expanded_pricing else 'DISABLED'}")
    if use_expanded_pricing:
        print(f"  - max_m:              {expanded_max_m}")
        print(f"  - max routes/iter:    {expanded_max_routes}")
        print(f"  - fallback threshold: {reduced_graph_min_columns}")
    print(f"GNN Pruning:             {'ENABLED' if use_gnn_pruning else 'DISABLED'}")
    if use_gnn_pruning:
        print(f"  - checkpoint:         {gnn_model_path}")
        print(f"  - threshold:          {gnn_threshold}")
        print(f"  - keep ratio:         {gnn_keep_ratio}")
        print(f"  - top-k per node:     {gnn_top_k_per_node}")
    if time_limit is not None:
        print(f"Time Limit:              {time_limit:.2f} seconds")
    print("="*70 + "\n")

    # Initialize routes using selected heuristic
    if heuristic.lower() == 'solomon':
        init_routes = solomon_insertion_heuristic(user_param)
    else:
        init_routes = savings_heuristic_vrptw(user_param)
    
    best_routes = []

    # Start timing
    start_time = time.time()

    # Execute branch and bound algorithm
    bp.bb_node(user_param, init_routes, None, best_routes, 0)

    # End timing
    end_time = time.time()
    sol_time = end_time - start_time

    # Calculate optimal cost
    opt_cost = 0
    print("\n" + "="*70)
    print("SOLUTION SUMMARY")
    print("="*70)
    print(f"Best Upper Bound = {bp.upperbound:.4f}")
    print(f"Best Lower Bound = {bp.lowerbound:.4f}")
    print(f"Final Gap = {((bp.upperbound - bp.lowerbound) / bp.upperbound * 100) if bp.upperbound > 0 else 0:.2f}%")
    print(f"Total Time = {sol_time:.2f} seconds")
    print(f"BB Nodes Processed = {bp.bb_node_count}")
    
    if best_routes:
        print(f"\nRoutes found: {len(best_routes)}")
        for i, route in enumerate(best_routes):
            print(f"  Route {i+1}: {route.get_path()} (Cost: {route.get_cost():.2f})")
            opt_cost += route.get_cost()
        print(f"\nTotal Route Cost = {opt_cost:.2f}")
    else:
        print("\nNo feasible solution found within time limit.")
    
    print("="*70)

if __name__ == "__main__":

    # Start logging to a timestamped file (also tee to console)
    start_logging()

    # Single dataset processing
    # Choose heuristic: 'savings' (Clarke-Wright) or 'solomon' (Solomon's I1 insertion)
    main(datasetPath="C:\\Users\\CiSTUP\\Downloads\\CVRTPW_Dataset\\solomon_25_customer_instances\\r201.txt", 
         SHOWFIG=True,
         heuristic='savings',
         time_limit=1000)  # Set to 300 seconds (5 minutes). Change to 'solomon' to test Solomon's insertion heuristic. Set time_limit=None for no limit.