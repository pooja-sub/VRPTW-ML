from branchBound import BranchAndBound
from columnGen import ColumnGeneration
from paramsVRP import ParamsVRP
from route import Route
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from solVisualization import solVis
from logcapture import start_logging
from datetime import datetime
import heapq


def savings_heuristic_vrptw(user_param, max_waiting_time=50):
    """
    Savings heuristic for VRPTW based on Clarke and Wright (1964).
    Extended to handle time windows and route orientation.
    
    :param user_param: ParamsVRP object with problem data
    :param max_waiting_time: Maximum allowed waiting time when joining customers
    :return: List of initial routes
    """
    print(f"[Initializing routes with Savings Heuristic]")
    
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
            
            departure_time = arrival_time
            time_to_depot = departure_time + user_param.ttime[i][user_param.nbclients]
            
            if arrival_time <= user_param.b[i] and time_to_depot <= user_param.b[user_param.nbclients]:
                route_cost = user_param.dist[0][i] + user_param.dist[i][user_param.nbclients]
                route = Route(path=[0, i, user_param.nbclients], cost=route_cost, Q=1.0)
                routes.append(route)
    
    print(f"  [Initial single-customer routes: {len(routes)}]")
    
    # Step 2: Calculate savings for all pairs of routes
    # Savings s_ij = d_i0 + d_0j - d_ij (distance saved by combining routes)
    savings_list = []
    
    for r1_idx in range(len(routes)):
        for r2_idx in range(r1_idx + 1, len(routes)):
            route1 = routes[r1_idx]
            route2 = routes[r2_idx]
            
            # Get end customers (before depot)
            i = route1.path[-2]  # Last customer of route1
            j = route2.path[1]   # First customer of route2
            
            # Calculate savings: s_ij = d_i,depot + d_depot,j - d_ij
            if user_param.dist[i][j] < user_param.verybig - 1e-6:
                saving = (user_param.dist[i][user_param.nbclients] + 
                         user_param.dist[0][j] - user_param.dist[i][j])
                
                # Use negative saving for max-heap (Python's heapq is min-heap)
                heapq.heappush(savings_list, (-saving, r1_idx, r2_idx, i, j))
    
    print(f"  [Calculated {len(savings_list)} potential merges]")
    
    # Step 3: Iteratively merge routes based on savings
    merged_count = 0
    route_active = [True] * len(routes)  # Track which routes are still active
    
    while savings_list:
        neg_saving, r1_idx, r2_idx, i, j = heapq.heappop(savings_list)
        saving = -neg_saving
        
        # Check if both routes are still active
        if not route_active[r1_idx] or not route_active[r2_idx]:
            continue
        
        route1 = routes[r1_idx]
        route2 = routes[r2_idx]
        
        # Check if i is last customer in route1 and j is first customer in route2
        if route1.path[-2] != i or route2.path[1] != j:
            continue
        
        # Create merged route
        merged_path = route1.path[:-1] + route2.path[1:]  # [0, ..., i, j, ..., nbclients]
        
        # Check capacity constraint
        total_demand = sum(user_param.d[customer] for customer in merged_path[1:-1])
        if total_demand > user_param.capacity:
            continue
        
        # Check time window feasibility with waiting time limit
        feasible = True
        current_time = user_param.a[0]
        waiting_at_j = 0
        
        for idx in range(1, len(merged_path) - 1):
            prev_customer = merged_path[idx - 1]
            customer = merged_path[idx]
            
            # Arrival time at customer
            current_time = current_time + user_param.ttime[prev_customer][customer]
            
            # Check if we arrive before time window opens (waiting time)
            if current_time < user_param.a[customer]:
                waiting_time = user_param.a[customer] - current_time
                if customer == j:
                    waiting_at_j = waiting_time
                current_time = user_param.a[customer]
            
            # Check if we arrive after time window closes
            if current_time > user_param.b[customer]:
                feasible = False
                break
        
        # Check return to depot
        if feasible:
            return_time = current_time + user_param.ttime[merged_path[-2]][user_param.nbclients]
            if return_time > user_param.b[user_param.nbclients]:
                feasible = False
        
        # Check waiting time limit at junction point
        if feasible and waiting_at_j > max_waiting_time:
            feasible = False
        
        if feasible:
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
    
    # Filter out inactive routes
    final_routes = [routes[i] for i in range(len(routes)) if route_active[i]]
    
    print(f"  [Savings heuristic complete: {merged_count} merges performed]")
    print(f"  [Final routes: {len(final_routes)} (reduced from {len(routes) - merged_count})]")
    
    return final_routes


def main(datasetPath = "C:\\Users\\CiSTUP\\Downloads\\VRTPW_Dataset\\solomon_100_customer_instances\\c101.txt", 
         SHOWFIG = False,
         enable_column_deletion = True,     # Enable column deletion acceleration
         deletion_threshold = 20,            # Delete routes not in basis for this many BB nodes
         enable_early_stop_pricing = False): # Enable early stopping in SPPRC (disabled by default)
    """
    Main function to solve VRPTW using Branch-and-Price with acceleration techniques.
    
    Acceleration techniques:
    1. Column Deletion (Larsen 2004): Remove routes not in basis for deletion_threshold nodes
       - Reported 2.5x speedup on 27 instances
    2. Early Stopping in SPPRC (Larsen 1999): Stop pricing when first negative cost route found
       - Dramatic running time reductions, especially for large time windows
    
    :param datasetPath: Path to VRPTW instance file
    :param SHOWFIG: Whether to show visualization
    :param enable_column_deletion: Enable/disable column deletion
    :param deletion_threshold: Number of BB nodes before deleting inactive routes
    :param enable_early_stop_pricing: Enable/disable early stopping in pricing subproblem
    """
    # Initialize branch and price algorithm with acceleration parameters
    bp = BranchAndBound(enable_column_deletion=enable_column_deletion,
                       deletion_threshold=deletion_threshold,
                       enable_early_stop=enable_early_stop_pricing)

    # Initialize problem instance
    user_param = ParamsVRP()
    user_param.init_params(datasetPath)
    dataset_name = user_param.datasetName
    
    # Display acceleration settings
    print("\n" + "="*70)
    print("ACCELERATION TECHNIQUES CONFIGURATION")
    print("="*70)
    print(f"Column Deletion:         {'ENABLED' if enable_column_deletion else 'DISABLED'}")
    if enable_column_deletion:
        print(f"  - Deletion Threshold:  {deletion_threshold} BB nodes")
    print(f"Early Stop Pricing:      {'ENABLED' if enable_early_stop_pricing else 'DISABLED'}")
    print("="*70 + "\n")

    # Initialize routes using Savings Heuristic
    init_routes = savings_heuristic_vrptw(user_param, max_waiting_time=50)
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
    print("\nSolution >>>")
    for route in best_routes:
        print(route.get_path())
        opt_cost += route.get_cost()

    print(f"\nBest Cost = {opt_cost}")
    print(f"Total Time = {sol_time:.2f} seconds")
    print(f"Total BB Nodes Processed = {bp.bb_node_count}")
    print(f"Final Lower Bound = {bp.lowerbound:.4f}")
    print(f"Final Upper Bound = {bp.upperbound:.4f}")
    print(f"Final Gap = {((bp.upperbound - bp.lowerbound) / bp.upperbound * 100):.2f}%")

    # Visualize solution
    # solVis(user_param, best_routes, sol_time, opt_cost, dataset_name, SHOWFIG)


def BatchMain(folder_path="C:\\Users\\CiSTUP\\Downloads\\VRTPW_Dataset\\solomon_50_customer_instances", banned_datasets=()):

    datasetBatch = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Remove unwanted datasets
    datasetBan = [os.path.join(folder_path, f"{ds}.txt") for ds in banned_datasets]
    datasetBatch = [ds for ds in datasetBatch if ds not in datasetBan]

    # Call main function for each dataset
    for dataset in datasetBatch:
        print(f"Processing dataset: {dataset}")
        main(dataset)

if __name__ == "__main__":

    # Start logging to a timestamped file (also tee to console)
    start_logging()

    # Single dataset processing
    main(datasetPath="C:\\Users\\CiSTUP\\Downloads\\VRTPW_Dataset\\solomon_100_customer_instances\\c102.txt", SHOWFIG=True)

    # Batch processing datasets
    #BatchMain(folder_path="F:/absolutePythonProject/universalPythonProject/BP-VRPTW/dataset", banned_datasets=["c110_1", "c101"])


