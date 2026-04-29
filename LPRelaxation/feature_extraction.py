import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
import math
import re
import networkx as nx

from utilities import readInstance, parse_instance

# -------------------- Step 2: Compute arc features -------------------- #

def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def generate_arc_features(df):
    n = len(df)
    arc_features = {}
    #cust_num = df['CUST NO.'].values
    coords = df[['XCOORD.', 'YCOORD.']].values
    demand = df['DEMAND'].values
    ready_time = df['READY TIME'].values
    due_date = df['DUE DATE'].values
    service_time = df['SERVICE TIME'].values

    # Precompute degrees and consumption stats
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    
    for i, j in product(range(n), repeat=2):
        # Keep depot<->customer arcs; only exclude self-loops.
        if i == j:
            continue

        # Cost
        cost = euclidean_distance(*coords[i], *coords[j])

        # Time resource = service time at i + travel time
        travel_time = cost
        time_consumption = service_time[i] + travel_time

        # Capacity resource = demand at j
        capacity_consumption = demand[j]

        # Save arc consumption
        out_arcs[i].append((j, time_consumption, capacity_consumption))
        in_arcs[j].append((i, time_consumption, capacity_consumption))

        arc_features[(i, j)] = {
            'cost': cost,
            'time_consumption': time_consumption,
            'capacity_consumption': capacity_consumption,
            'deg_out_i': 0,  # to fill later
            'deg_in_j': 0,
            'time_stats_out_i': None,
            'time_stats_in_j': None,
            'cap_stats_out_i': None,
            'cap_stats_in_j': None,
            'time_bounds_i': (ready_time[i], due_date[i]),
            'time_bounds_j': (ready_time[j], due_date[j]),
        }


    # Fill degree and stats
    for i in range(n):
        out_time = [x[1] for x in out_arcs[i]]
        out_cap = [x[2] for x in out_arcs[i]]
        in_time = [x[1] for x in in_arcs[i]]
        in_cap = [x[2] for x in in_arcs[i]]

        for j in range(n):
            if i == j:
                continue
            
            if (i, j) in arc_features:
                arc_features[(i, j)]['deg_out_i'] = len(out_arcs[i])
                arc_features[(i, j)]['deg_in_j'] = len(in_arcs[j])

                arc_features[(i, j)]['time_stats_out_i'] = (
                    np.min(out_time), np.max(out_time), np.mean(out_time)
                )
                arc_features[(i, j)]['time_stats_in_j'] = (
                    np.min(in_time), np.max(in_time), np.mean(in_time)
                )

                arc_features[(i, j)]['cap_stats_out_i'] = (
                    np.min(out_cap), np.max(out_cap), np.mean(out_cap)
                )
                arc_features[(i, j)]['cap_stats_in_j'] = (
                    np.min(in_cap), np.max(in_cap), np.mean(in_cap)
                )

    return arc_features


# -------------------- Step 3: Export to DataFrame for ML -------------------- #

def features_to_dataframe(arc_features, used_arcs=None):
    """
    Convert arc features to DataFrame and optionally label with arc usage.
    
    :param arc_features: Dictionary of arc features from generate_arc_features()
    :param used_arcs: Optional set of (from, to) tuples representing arcs in RMP solution.
                      If provided, adds 'label' column (1 if arc used, 0 otherwise).
    :return: pandas DataFrame with arc features and optional label
    """
    rows = []
    for (i, j), feat in arc_features.items():
        row = {
            'from': i, 'to': j,
            'cost': feat['cost'],
            'time_consumption': feat['time_consumption'],
            'capacity_consumption': feat['capacity_consumption'],
            # 'deg_out_i': feat['deg_out_i'],
            # 'deg_in_j': feat['deg_in_j'],
            'time_min_out_i': feat['time_stats_out_i'][0],
            'time_max_out_i': feat['time_stats_out_i'][1],
            'time_avg_out_i': feat['time_stats_out_i'][2],
            'time_min_in_j': feat['time_stats_in_j'][0],
            'time_max_in_j': feat['time_stats_in_j'][1],
            'time_avg_in_j': feat['time_stats_in_j'][2],
            'cap_min_out_i': feat['cap_stats_out_i'][0],
            'cap_max_out_i': feat['cap_stats_out_i'][1],
            'cap_avg_out_i': feat['cap_stats_out_i'][2],
            'cap_min_in_j': feat['cap_stats_in_j'][0],
            'cap_max_in_j': feat['cap_stats_in_j'][1],
            'cap_avg_in_j': feat['cap_stats_in_j'][2],
            'ready_i': feat['time_bounds_i'][0],
            'due_i': feat['time_bounds_i'][1],
            'ready_j': feat['time_bounds_j'][0],
            'due_j': feat['time_bounds_j'][1]
        }
        
        # Add label if used_arcs provided (for ML training data).
        # Label all non-self arcs, including depot<->customer arcs.
        if used_arcs is not None and i != j:
            row['label'] = 1 if (i, j) in used_arcs else 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # If labels were added, keep only rows where label is defined.
    if used_arcs is not None and 'label' in df.columns:
        df = df[df['label'].notna()].reset_index(drop=True)
    
    return df


# -------------------- Example usage -------------------- #

if __name__ == "__main__":
    n, INSTANCE_NAME, INSTANCE_FILENAME = readInstance()
    name, vehicle_number, vehicle_capacity, df = parse_instance(INSTANCE_FILENAME)
    arc_features = generate_arc_features(df)

    df_features = features_to_dataframe(arc_features)
    df_features = df_features.round(2)


    df_features.to_csv(f"../DatasetFeatures/50-customer-instances/{name}_arc_features.csv", index=False)
