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


def main(datasetPath = "C:\\Users\\CiSTUP\\Downloads\\VRTPW_Dataset\\solomon_100_customer_instances\\c101.txt", SHOWFIG = False):
    # Initialize branch and price algorithm
    bp = BranchAndBound()

    # Initialize problem instance
    user_param = ParamsVRP()
    user_param.init_params(datasetPath)
    dataset_name = user_param.datasetName

    # Initialize initial routes and best routes lists (only for feasible arcs)
    init_routes = []
    for i in range(1, user_param.nbclients):
        # Only create route if both arcs are feasible (not eliminated by preprocessing)
        if (user_param.dist[0][i] < user_param.verybig - 1e-6 and 
            user_param.dist[i][user_param.nbclients] < user_param.verybig - 1e-6):
            route_cost = user_param.dist[0][i] + user_param.dist[i][user_param.nbclients]
            route = Route(path=[0, i, user_param.nbclients], cost=route_cost, Q=1.0)
            init_routes.append(route)
    best_routes = []
    print(f"[Initial feasible routes created: {len(init_routes)} out of {user_param.nbclients - 1} customers]")

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


