import numpy as np
from sys import exit
import os
import pandas as pd
import re

def readInstance(instance_path):
    """
    Extract instance name and customer count from instance path.
    
    Args:
        instance_path: Full path to instance file or folder containing it
        
    Returns:
        (num_customers, instance_name, instance_filename)
    """
    # Normalize path
    if instance_path.endswith(".txt"):
        INSTANCE_FILENAME = instance_path
        INSTANCE_NAME = os.path.basename(instance_path)[:-4]
    else:
        # Assume it's a folder, find the .txt file
        if os.path.isdir(instance_path):
            txt_files = [f for f in os.listdir(instance_path) if f.endswith(".txt")]
            if not txt_files:
                print(f"No .txt files found in {instance_path}. Exit."); exit(1)
            INSTANCE_FILENAME = os.path.join(instance_path, txt_files[0])
            INSTANCE_NAME = txt_files[0][:-4]
        else:
            print(f"Instance path does not exist: {instance_path}. Exit."); exit(1)
    
    if not os.path.exists(INSTANCE_FILENAME):
        print(f"Instance file does not exist: {INSTANCE_FILENAME}. Exit."); exit(1)
    
    # Extract customer count from path (e.g., "50-customer-instances" -> 50)
    path_str = instance_path.lower()
    match = re.search(r'(\d+)-customer', path_str)
    if match:
        num_customers = int(match.group(1))
    else:
        # Fallback: try to extract from parent folder or default to 100
        num_customers = 100
    
    return num_customers, INSTANCE_NAME, INSTANCE_FILENAME

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
