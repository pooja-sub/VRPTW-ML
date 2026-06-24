"""
GNN Training Data Analysis and Preparation

Analyzes collected training data and prepares it for GNN model training.
- Checks data quality and completeness
- Normalizes features
- Handles class imbalance
- Generates train/val/test splits
- Creates PyG graph data

Usage:
    python prepare_gnn_data.py --data training_data.csv --output prepared_data.pt
    python prepare_gnn_data.py --data combined_training_data.csv --analyze
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


class GNNDataAnalyzer:
    """Analyzes and prepares GNN training data."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and validate data."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} rows")
    
    def analyze(self):
        """Print data analysis."""
        print(f"\n{'='*70}")
        print("Data Analysis")
        print(f"{'='*70}\n")
        
        # Basic stats
        print("### Basic Statistics ###")
        print(f"Total rows: {len(self.df):,}")
        print(f"Instances: {self.df['instance'].nunique()}")
        print(f"CG iterations: {self.df['cg_iteration'].nunique()}")
        print(f"Features: {len(self.df.columns)}")
        
        # Missing values
        print("\n### Missing Values ###")
        missing = self.df.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
        else:
            print("✓ No missing values")
        
        # Label distribution
        print("\n### Label Distribution ###")
        label_counts = self.df['label'].value_counts()
        print(f"Label 0 (not in route): {label_counts[0]:,} ({100*label_counts[0]/len(self.df):.1f}%)")
        print(f"Label 1 (in route): {label_counts[1]:,} ({100*label_counts[1]/len(self.df):.1f}%)")
        
        # Feature statistics
        print("\n### Feature Statistics ###")
        feature_cols = [
            'edge_distance', 'edge_travel_time', 'direct_reduced_cost',
            'accumulated_demand', 'elapsed_time', 'remaining_capacity', 'time_slack'
        ]
        
        for col in feature_cols:
            if col in self.df.columns:
                stats = self.df[col].describe()
                print(f"{col}:")
                print(f"  mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                print(f"  min={stats['min']:.4f}, max={stats['max']:.4f}")
        
        # Instance-level stats
        print("\n### Per-Instance Statistics ###")
        instance_stats = self.df.groupby('instance').agg({
            'label': 'count',
            'cg_iteration': 'max',
            'rmp_objective': 'mean'
        }).rename(columns={'label': 'edges', 'cg_iteration': 'max_iter'})
        
        print(instance_stats.head(10))
        print(f"... and {len(instance_stats) - 10} more instances")
        
        # Data quality
        print("\n### Data Quality Checks ###")
        
        # Check for binary features
        binary_features = [
            'can_fit_demand', 'time_window_feasible', 'is_mandatory_arc', 'label'
        ]
        for feat in binary_features:
            if feat in self.df.columns:
                unique = self.df[feat].unique()
                if set(unique).issubset({0, 1}):
                    print(f"✓ {feat}: binary")
                else:
                    print(f"✗ {feat}: not binary! Values: {unique}")
        
        # Check for negative values where not expected
        non_negative = [
            'edge_distance', 'edge_travel_time', 'accumulated_demand',
            'elapsed_time', 'remaining_capacity'
        ]
        for feat in non_negative:
            if feat in self.df.columns:
                if (self.df[feat] >= 0).all():
                    print(f"✓ {feat}: all non-negative")
                else:
                    neg_count = (self.df[feat] < 0).sum()
                    print(f"⚠ {feat}: {neg_count} negative values")
        
        print(f"\n{'='*70}\n")
    
    def normalize_features(self, output_csv: str = None):
        """Normalize features to [0, 1] or [-1, 1] ranges."""
        print("Normalizing features...")
        
        df_norm = self.df.copy()
        
        # Features to normalize
        normalize_specs = {
            'edge_distance': (0, 1),
            'edge_travel_time': (0, 1),
            'direct_reduced_cost': (-1, 1),
            'accumulated_demand': (0, 1),
            'elapsed_time': (0, 1),
            'remaining_capacity': (0, 1),
            'time_slack': (0, 1),
            'avg_dual_magnitude': (-1, 1),
            'max_dual_magnitude': (-1, 1),
        }
        
        for feat, (min_val, max_val) in normalize_specs.items():
            if feat in df_norm.columns:
                data = df_norm[feat]
                if data.std() > 0:
                    # Standardize first
                    data_std = (data - data.mean()) / data.std()
                    # Then scale to range
                    if min_val == 0:
                        data_scaled = (data_std - data_std.min()) / (data_std.max() - data_std.min())
                    else:
                        # Scale to [-1, 1]
                        data_scaled = 2 * (data_std - data_std.min()) / (data_std.max() - data_std.min()) - 1
                    df_norm[f'{feat}_norm'] = data_scaled
                else:
                    df_norm[f'{feat}_norm'] = 0.0
        
        if output_csv:
            df_norm.to_csv(output_csv, index=False)
            print(f"Saved normalized data to {output_csv}")
        
        return df_norm
    
    def split_by_instance(self, train_ratio: float = 0.7, 
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15):
        """Split data by instance into train/val/test."""
        
        instances = self.df['instance'].unique()
        n_instances = len(instances)
        
        n_train = max(1, int(n_instances * train_ratio))
        n_val = max(1, int(n_instances * val_ratio))
        n_test = n_instances - n_train - n_val
        
        np.random.seed(42)
        np.random.shuffle(instances)
        
        train_inst = instances[:n_train]
        val_inst = instances[n_train:n_train + n_val]
        test_inst = instances[n_train + n_val:]
        
        train_df = self.df[self.df['instance'].isin(train_inst)]
        val_df = self.df[self.df['instance'].isin(val_inst)]
        test_df = self.df[self.df['instance'].isin(test_inst)]
        
        print(f"\nInstance-based split:")
        print(f"Train: {len(train_inst)} instances, {len(train_df):,} rows")
        print(f"Val:   {len(val_inst)} instances, {len(val_df):,} rows")
        print(f"Test:  {len(test_inst)} instances, {len(test_df):,} rows")
        
        return train_df, val_df, test_df
    
    def handle_imbalance(self, target_ratio: float = 0.5):
        """Balance classes if needed."""
        
        label_counts = self.df['label'].value_counts()
        
        if len(label_counts) < 2:
            print("⚠ Only one class present, cannot balance")
            return self.df
        
        n0, n1 = label_counts.get(0, 0), label_counts.get(1, 0)
        ratio = min(n0, n1) / max(n0, n1)
        
        print(f"\nClass balance: {n0:,} (0) vs {n1:,} (1), ratio={ratio:.2%}")
        
        if ratio < target_ratio:
            print(f"Imbalance detected! Consider using weighted loss or oversampling.")
            # Optionally oversample minority class
            if n0 < n1:
                df0 = self.df[self.df['label'] == 0]
                df1 = self.df[self.df['label'] == 1]
                df0_oversample = df0.sample(n=len(df1), replace=True, random_state=42)
                df_balanced = pd.concat([df0_oversample, df1], ignore_index=True)
                print(f"After oversampling: {len(df_balanced)} rows")
                return df_balanced
        
        return self.df
    
    def export_for_pytorch_geometric(self, output_file: str):
        """Export data in PyTorch Geometric compatible format."""
        try:
            import torch
            from torch_geometric.data import Data, Dataset
        except ImportError:
            print("ERROR: PyTorch Geometric not installed")
            return
        
        print(f"Exporting to PyG format...")
        
        # Feature columns
        feature_cols = [
            'edge_distance', 'edge_travel_time', 'direct_reduced_cost',
            'accumulated_demand', 'elapsed_time', 'remaining_capacity',
            'time_slack', 'avg_dual_magnitude', 'max_dual_magnitude'
        ]
        
        # Normalize
        X = self.df[feature_cols].values
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X = (X - X_mean) / X_std
        
        # Labels
        y = self.df['label'].values
        
        # Create tensor data
        x = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Save
        torch.save({
            'x': x,
            'y': y,
            'feature_names': feature_cols,
            'n_features': len(feature_cols),
            'n_samples': len(self.df),
        }, output_file)
        
        print(f"✓ Saved to {output_file}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(self.df)}")
        print(f"  Classes: 2 (0: not in route, 1: in route)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and prepare GNN training data"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Input CSV file with training data"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print data analysis"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features"
    )
    parser.add_argument(
        "--output",
        help="Output file for processed data"
    )
    parser.add_argument(
        "--export_pyg",
        help="Export to PyTorch Geometric format"
    )
    
    args = parser.parse_args()
    
    # Verify input
    if not os.path.exists(args.data):
        print(f"ERROR: File not found: {args.data}")
        return
    
    analyzer = GNNDataAnalyzer(args.data)
    
    if args.analyze:
        analyzer.analyze()
    
    if args.normalize:
        df_norm = analyzer.normalize_features(output_csv=args.output)
    
    if args.export_pyg:
        analyzer.export_for_pytorch_geometric(args.export_pyg)


if __name__ == "__main__":
    main()
