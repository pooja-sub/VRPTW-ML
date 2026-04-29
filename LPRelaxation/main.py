from lp_relaxation import LPRelaxationSolver
from columnGen import ColumnGeneration
from paramsVRP import ParamsVRP
from route import Route
import time
import matplotlib.pyplot as plt
import numpy as np
import os
# from solVisualization import solVis
from logcapture import start_logging


def main(datasetPath, SHOWFIG = False):
    # Initialize LP relaxation solver
    lp_solver = LPRelaxationSolver()

    # Initialize problem instance
    user_param = ParamsVRP()
    user_param.init_params(datasetPath)
    dataset_name = user_param.datasetName

    # Initialize initial routes (one customer per route)
    init_routes = []
    end_depot = user_param.nbclients
    for customer_id in range(1, user_param.nbclients):
        route_cost = user_param.dist[0][customer_id] + user_param.dist[customer_id][end_depot]
        route = Route(path=[0, customer_id, end_depot], cost=route_cost, Q=1.0)
        init_routes.append(route)

    # Start timing
    start_time = time.time()

    # Solve LP relaxation to optimality
    lp_value, final_routes = lp_solver.solve_lp_relaxation(
        user_param,
        init_routes,
        dataset_path=datasetPath,
    )

    # End timing
    end_time = time.time()
    sol_time = end_time - start_time

    # Calculate total cost from final routes
    total_cost = 0
    print("\n" + "="*70)
    print("LP RELAXATION SOLUTION")
    print("="*70)
    for route in final_routes:
        if route.get_Q() > 1e-6:  # Only include routes with positive flow
            print(f"Route: {route.get_path()}, Cost: {route.get_cost():.4f}, Flow: {route.get_Q():.6f}")
            total_cost += route.get_cost() * route.get_Q()

    print(f"\nLP Relaxation Bound = {lp_value:.4f}")
    print(f"Total Time = {sol_time:.2f} seconds")
    print("="*70 + "\n")


def BatchMain(folder_path="C:/Users/CiSTUP/Downloads/VRTPW_Dataset/solomon_50_customer_instances", banned_datasets=()):

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
    #main(datasetPath="C:/Users/CiSTUP/Downloads/VRTPW_Dataset/solomon_50_customer_instances/c101.txt", SHOWFIG=True)

    # Batch processing datasets
    BatchMain(banned_datasets=['c101','c102'])



