import sys
import os
import time
import subprocess

DATASET_DIR = r"C:\Users\CiSTUP\Downloads\CVRTPW_Dataset\solomon_50_customer_instances"

def run_all():
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.txt')])
    print(f"Found {len(files)} files in {DATASET_DIR}")
    print("Running naive LP relaxation (Branching folder) on 50-customer instances\n")

    results = []
    for idx, fname in enumerate(files, start=1):
        path = os.path.join(DATASET_DIR, fname)
        print(f"\n=== ({idx}/{len(files)}) Running: {fname} ===")
        # Launch the Branching script as a subprocess with env override
        env = os.environ.copy()
        env['BNP_DATASET_PATH'] = path
        try:
            subprocess.check_call([sys.executable, os.path.join('Branching', 'main.py')], env=env)
            results.append((fname, 'SUCCESS'))
        except subprocess.CalledProcessError as e:
            print(f"Error running {fname}: exit {e.returncode}")
            results.append((fname, f'FAILED (exit {e.returncode})'))
        time.sleep(0.5)
    
    print("\n" + "="*70)
    print("BATCH RUN SUMMARY")
    print("="*70)
    for fname, status in results:
        print(f"  {fname}: {status}")
    successful = sum(1 for _, s in results if s == 'SUCCESS')
    print(f"\nTotal: {successful}/{len(results)} completed successfully")
    print("="*70)

if __name__ == '__main__':
    run_all()
