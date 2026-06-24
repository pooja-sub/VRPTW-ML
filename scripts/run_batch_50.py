import sys
import os
import time

# Ensure workspace root is on path
sys.path.insert(0, os.path.abspath('.'))

import subprocess

DATASET_DIR = r"C:\Users\CiSTUP\Downloads\CVRTPW_Dataset\solomon_50_customer_instances"

def run_all():
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.txt')])
    print(f"Found {len(files)} files in {DATASET_DIR}")

    for idx, fname in enumerate(files, start=1):
        path = os.path.join(DATASET_DIR, fname)
        print(f"\n=== ({idx}/{len(files)}) Running: {fname} ===")
        # Launch the BranchingExpandedPricing script as a subprocess with env override
        env = os.environ.copy()
        env['BNP_DATASET_PATH'] = path
        try:
            subprocess.check_call([sys.executable, os.path.join('BranchingExpandedPricing','main.py')], env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {fname}: exit {e.returncode}")
        # small pause between runs
        time.sleep(0.5)

if __name__ == '__main__':
    run_all()
