"""
Batch Data Collection Runner

Automates collecting training data from multiple instance sizes and classes.
Supports running sequentially or organizing output by instance size.

Usage:
    python batch_collect_training_data.py --strategy standard --output_dir ./training_data
    python batch_collect_training_data.py --strategy quick --instances c101 r101 rc101
    python batch_collect_training_data.py --strategy comprehensive --num_workers 2
"""

import sys
import os
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = r"C:\Users\CiSTUP\Desktop\A-Branch-and-Price-Algorithm-for-VRPTW"
DATASET_BASE = r"C:\Users\CiSTUP\Downloads\CVRTPW_Dataset"
MLGNN_DIR = os.path.join(ROOT_DIR, "MLPricingGNN")


class BatchCollector:
    """Manages batch data collection across multiple instance sets."""
    
    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = self.output_dir / "collection_log.json"
        self.stats = {
            "start_time": time.time(),
            "collections": []
        }
    
    def run_collection(self, instances_dir: str, instance_names: str, 
                       max_instances: int, output_csv: str, 
                       time_limit: int = 300) -> bool:
        """Run a single collection job."""
        
        cmd = [
            sys.executable,
            os.path.join(MLGNN_DIR, "collect_training_data.py"),
            "--instances_dir", instances_dir,
            "--max", str(max_instances),
            "--output", output_csv,
        ]
        
        print(f"\n{'='*70}")
        print(f"Running: {instance_names}")
        print(f"Instances dir: {instances_dir}")
        print(f"Max instances: {max_instances}")
        print(f"Output: {output_csv}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd=MLGNN_DIR, capture_output=False)
            elapsed = time.time() - start_time
            
            success = result.returncode == 0
            
            self.stats["collections"].append({
                "name": instance_names,
                "success": success,
                "elapsed_seconds": elapsed,
                "output_file": output_csv,
                "timestamp": time.time()
            })
            
            if success:
                print(f"\n✓ {instance_names} completed in {elapsed:.1f}s")
            else:
                print(f"\n✗ {instance_names} failed")
            
            return success
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return False
    
    def collect_quick(self, output_dir: str = None):
        """Quick collection: few instances of each major type."""
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        jobs = [
            {
                "name": "solomon_25_c_series",
                "dir": os.path.join(DATASET_BASE, "solomon_25_customer_instances"),
                "max": 3,
            },
            {
                "name": "solomon_25_r_series",
                "dir": os.path.join(DATASET_BASE, "solomon_25_customer_instances"),
                "max": 3,
            },
        ]
        
        for job in jobs:
            output_csv = os.path.join(output_dir, f"{job['name']}.csv")
            self.run_collection(
                instances_dir=job['dir'],
                instance_names=job['name'],
                max_instances=job['max'],
                output_csv=output_csv
            )
            
            # Brief pause between jobs
            time.sleep(5)
    
    def collect_standard(self, output_dir: str = None):
        """Standard collection: balanced set from 25 and 50 customer instances."""
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        jobs = [
            {
                "name": "solomon_25_customers",
                "dir": os.path.join(DATASET_BASE, "solomon_25_customer_instances"),
                "max": 15,
            },
            {
                "name": "solomon_50_customers",
                "dir": os.path.join(DATASET_BASE, "solomon_50_customer_instances"),
                "max": 10,
            },
        ]
        
        for job in jobs:
            output_csv = os.path.join(output_dir, f"{job['name']}.csv")
            self.run_collection(
                instances_dir=job['dir'],
                instance_names=job['name'],
                max_instances=job['max'],
                output_csv=output_csv
            )
            time.sleep(5)
    
    def collect_comprehensive(self, output_dir: str = None):
        """Comprehensive collection: all available sizes."""
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        jobs = [
            {
                "name": "solomon_25_customers",
                "dir": os.path.join(DATASET_BASE, "solomon_25_customer_instances"),
                "max": 30,
            },
            {
                "name": "solomon_50_customers",
                "dir": os.path.join(DATASET_BASE, "solomon_50_customer_instances"),
                "max": 20,
            },
            {
                "name": "solomon_100_customers",
                "dir": os.path.join(DATASET_BASE, "solomon_100_customer_instances"),
                "max": 10,
            },
            {
                "name": "homberger_200_customers",
                "dir": os.path.join(DATASET_BASE, "homberger_200_customer_instances"),
                "max": 5,
            },
        ]
        
        for job in jobs:
            if not os.path.isdir(job['dir']):
                print(f"⚠ Directory not found: {job['dir']}")
                continue
            
            output_csv = os.path.join(output_dir, f"{job['name']}.csv")
            self.run_collection(
                instances_dir=job['dir'],
                instance_names=job['name'],
                max_instances=job['max'],
                output_csv=output_csv
            )
            time.sleep(5)
    
    def collect_specific(self, instances: List[str], output_dir: str = None):
        """Collect from specific instance files."""
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        # Find instances
        instance_paths = []
        for instance_name in instances:
            found = False
            for dataset_dir in [
                os.path.join(DATASET_BASE, "solomon_25_customer_instances"),
                os.path.join(DATASET_BASE, "solomon_50_customer_instances"),
                os.path.join(DATASET_BASE, "solomon_100_customer_instances"),
            ]:
                instance_file = os.path.join(dataset_dir, f"{instance_name}.txt")
                if os.path.exists(instance_file):
                    instance_paths.append(instance_file)
                    found = True
                    break
            
            if not found:
                print(f"⚠ Instance not found: {instance_name}")
        
        if not instance_paths:
            print("No instances found!")
            return
        
        output_csv = os.path.join(output_dir, "specific_instances.csv")
        
        # Run with the specific paths
        cmd = [
            sys.executable,
            os.path.join(MLGNN_DIR, "collect_training_data.py"),
            "--instances",
            *instance_paths,
            "--output", output_csv,
        ]
        
        print(f"\nCollecting data from {len(instance_paths)} specific instances...")
        print(f"Output: {output_csv}\n")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=MLGNN_DIR, capture_output=False)
        elapsed = time.time() - start_time
        
        self.stats["collections"].append({
            "name": "specific_instances",
            "count": len(instance_paths),
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed,
            "output_file": output_csv,
        })
    
    def merge_csvs(self, output_dir: str = None):
        """Merge all CSV files into one."""
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        import glob
        import pandas as pd
        
        csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith("_merged.csv")]
        
        if not csv_files:
            print("No CSV files found to merge!")
            return
        
        print(f"\nMerging {len(csv_files)} CSV files...")
        
        dfs = []
        for csv_file in sorted(csv_files):
            print(f"  Reading {os.path.basename(csv_file)}...")
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                print(f"    -> {len(df)} rows")
            except Exception as e:
                print(f"    ERROR: {e}")
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_file = os.path.join(output_dir, "combined_training_data.csv")
            merged_df.to_csv(merged_file, index=False)
            print(f"\nMerged data: {len(merged_df)} total rows")
            print(f"Saved to: {merged_file}")
            
            # Stats
            print(f"\nData statistics:")
            print(f"  Instances: {merged_df['instance'].nunique()}")
            print(f"  Edges: {len(merged_df)}")
            print(f"  Label distribution: {dict(merged_df['label'].value_counts())}")
    
    def save_stats(self):
        """Save collection statistics."""
        self.stats["end_time"] = time.time()
        self.stats["total_seconds"] = self.stats["end_time"] - self.stats["start_time"]
        
        with open(self.log_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Collection Statistics")
        print(f"{'='*70}")
        print(f"Total time: {self.stats['total_seconds']:.1f}s")
        print(f"Collections: {len(self.stats['collections'])}")
        for coll in self.stats['collections']:
            status = "✓" if coll.get('success') else "✗"
            print(f"  {status} {coll['name']}: {coll['elapsed_seconds']:.1f}s")
        print(f"\nLog: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch collection of GNN training data"
    )
    parser.add_argument(
        "--strategy",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Collection strategy"
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        help="Specific instances to collect (e.g., c101 r101 rc101)"
    )
    parser.add_argument(
        "--output_dir",
        default="./training_data",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all CSV files after collection"
    )
    
    args = parser.parse_args()
    
    # Verify paths
    if not os.path.isdir(MLGNN_DIR):
        print(f"ERROR: MLPricingGNN directory not found: {MLGNN_DIR}")
        return
    
    if not os.path.isdir(DATASET_BASE):
        print(f"ERROR: Dataset directory not found: {DATASET_BASE}")
        return
    
    collector = BatchCollector(output_dir=args.output_dir)
    
    print(f"{'='*70}")
    print(f"GNN Training Data Collection")
    print(f"{'='*70}")
    print(f"Strategy: {args.strategy}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    if args.instances:
        collector.collect_specific(args.instances, output_dir=args.output_dir)
    elif args.strategy == "quick":
        collector.collect_quick(output_dir=args.output_dir)
    elif args.strategy == "standard":
        collector.collect_standard(output_dir=args.output_dir)
    elif args.strategy == "comprehensive":
        collector.collect_comprehensive(output_dir=args.output_dir)
    
    if args.merge:
        collector.merge_csvs(output_dir=args.output_dir)
    
    collector.save_stats()
    
    print(f"\nDone! Check {args.output_dir} for CSV files.")


if __name__ == "__main__":
    main()
