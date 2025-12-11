import numpy as np
from sys import exit
import os
import pandas as pd
import re

# def update_arc_feature_labels(instance_name, arcs_added):
#     file_name = f"DatasetFeatures_CVRP/{instance_name}_arc_features.csv"
#     df = pd.read_csv(file_name)
#     # Only label arcs where from != 0 and to != 0
#     def label_row(row):
#         f, t = int(row['from']), int(row['to'])
#         if f == 0 or t == 0:
#             return 0
#         return 1 if (f, t) in arcs_added else 0
#     df['label'] = df.apply(label_row, axis=1)
#     df.to_csv(file_name, index=False)


def readInstance():
    # Take in input instance name
    print("Instance name: [R|C|RC][N_N_N]")
    INSTANCE_NAME = None
    try:
        INSTANCE_NAME = input()
    except Exception as e:
        print("Invalid instance. Exit."); exit(1)

    if INSTANCE_NAME.endswith(".txt"):
        INSTANCE_NAME = INSTANCE_NAME[:-4]
    
    # Change this if you want to use another dataset path
    INSTANCE_FILENAME = os.path.join("../dataset","50-customer-instances", INSTANCE_NAME+".txt")
    #INSTANCE_FILENAME = os.path.join("../DummyData", INSTANCE_NAME+".TXT")
    if not INSTANCE_NAME or not os.path.exists(INSTANCE_FILENAME):
        print("Instance does not exists. Exit."); exit(1)

    # change it to 400 for 400-customer instances
    return 100, INSTANCE_NAME, INSTANCE_FILENAME

def parse_instance(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    name = lines[0] 
    vehicle_data = lines[2:4]
    customer_data = lines[5:]
    
    vehicle_number = int(vehicle_data[1].split()[0])
    vehicle_capacity = int(vehicle_data[1].split()[-1])

    columns = re.split(r'\s{2,}', customer_data[0])

    data = [list(map(float, line.split())) for line in customer_data[1:]]
    df = pd.DataFrame(data, columns=columns)
    return name, vehicle_number, vehicle_capacity, df


def readData(filename):
    _, vehicleNumber, capacity, df = parse_instance(filename)
    data = df.to_dict('records')

    data.append(data[0]) # The depot is represented by two identical

    # change this to change the depot index for 200 and 400 customers
    #                       
    data[-1]["CUST-NO."] = 101

    x = []; y = []; q = []; r = []; d = []; s = []
    for customer in data:
        x.append(int(customer["XCOORD."]))
        y.append(int(customer["YCOORD."]))
        q.append(int(customer["DEMAND"]))
        r.append(int(customer["READY TIME"]))
        d.append(int(customer["DUE DATE"]))
        s.append(int(customer["SERVICE TIME"]))

    return vehicleNumber, capacity, x, y, q, r, d, s


def createDistanceMatrix(x, y):
    n = len(x)
    d = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p1 = np.array([x[i], y[i]])
            p2 = np.array([x[j], y[j]])
            d[i,j] = d[j,i] = int(round(np.linalg.norm(p1-p2)))
    return d


def addRoutesToMaster(routes, mat, costs, d):
    for i, route in enumerate(routes):
        route = np.array(route)
        
        # vectorized arc cost (no loops!)
        costs[i] = np.sum(d[route[:-1], route[1:]])

        # update incidence matrix
        np.add.at(mat[:, i], route[1:-1] - 1, 1)

