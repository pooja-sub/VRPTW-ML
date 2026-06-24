"""
GNN TRAINING DATA COLLECTION - Quick Start Guide
================================================

The code is set up to collect labeled training data for your GNN model.
Follow these simple steps:

STEP 1: Quick Collection (Recommended for Testing)
---------------------------------------------------
Run this command to collect data from 3-5 instances of each type:

    cd C:\Users\CiSTUP\Desktop\A-Branch-and-Price-Algorithm-for-VRPTW\MLPricingGNN
    python collect_training_data.py --instances_dir "C:\Users\CiSTUP\Downloads\CVRTPW_Dataset\solomon_25_customer_instances" --max 5 --output c101_data.csv

This will:
  - Load instances from the directory
  - Run B&P at root node only
  - Collect labeled edges (1=in negative routes, 0=not)
  - Extract 18 features per edge
  - Save to CSV

Output: c101_data.csv with columns:
  - Edge features: distance, travel_time, reduced_cost, demand_feasibility, etc.
  - Context: RMP objective, dual values, route count
  - Label: 1/0 indicating if edge was in optimal pricing solution


STEP 2: Batch Collection (Multiple Instance Sets)
--------------------------------------------------
Run this for comprehensive data collection across all instance sizes:

    python batch_collect_training_data.py --strategy quick --output_dir ./training_data

Strategies:
  - quick: 3 instances each (25 customers, R/C series) = ~30 min
  - standard: 5-10 instances each (25/50 customers, R/C series) = 1-2 hours
  - comprehensive: 20+ instances including 100+ customers = 4+ hours

Output: ./training_data/ with:
  - solomon_25_data.csv
  - solomon_50_data.csv
  - combined_training_data.csv (all merged)
  - collection_stats.json


STEP 3: Prepare Data for GNN Training
------------------------------------
Once you have training CSVs, prepare them for PyTorch:

    python prepare_gnn_data.py --data combined_training_data.csv --analyze --normalize

This will:
  - Analyze feature distributions
  - Normalize features to [0,1]
  - Check class balance (positive/negative examples)
  - Split into train/val/test by instance
  - Export as PyTorch geometric format


COMPLETE WORKFLOW EXAMPLE:
==========================

# 1. Collect data from one instance type (fast test)
python collect_training_data.py --instances_dir "path/to/instances" --max 5 --output test_data.csv

# 2. Analyze the data
python prepare_gnn_data.py --data test_data.csv --analyze

# 3. Collect larger dataset
python batch_collect_training_data.py --strategy standard --output_dir ./training_data

# 4. Prepare full dataset for training
python prepare_gnn_data.py --data ./training_data/combined_training_data.csv --normalize --split

# 5. Train your GNN model
# (Use the train/val/test PyG tensors exported in step 4)


FEATURES COLLECTED (18 total):
==============================
Node/Edge Features:
  - edge_distance: Direct arc distance
  - edge_travel_time: Travel time on arc
  - direct_reduced_cost: Current reduced cost
  - can_fit_demand: Capacity constraint satisfied (0/1)
  - time_window_feasible: Time feasibility (0/1)
  - slack_at_target: Time slack at destination
  - is_mandatory_arc: Depot connection (0/1)

Context Features (Global):
  - rmp_objective: Current RMP objective value
  - avg_dual_magnitude: Mean absolute dual price
  - max_dual_magnitude: Max dual price
  - num_routes_in_rmp: Number of columns in master problem
  - accumulated_demand: Cumulative demand on path
  - elapsed_time: Cumulative time on path
  - remaining_capacity: Available vehicle capacity

Instance Info:
  - instance: Instance name
  - cg_iteration: Column generation iteration
  - label: 1 if edge in optimal route, 0 otherwise


TROUBLESHOOTING:
================

Q: I don't see a CSV being created?
A: Check that:
   - Instance directory path exists and has .txt files
   - --max value is >= 1
   - Write permissions in output directory
   - No import errors (try: python -c "from Common.paramsVRP import ParamsVRP")

Q: Data collection is very slow?
A: This is normal - exact pricing is NP-hard:
   - Start with --max 2-3 instances to test
   - Increase time_limit parameter if needed
   - Use standard instances (C series) before R/RC series

Q: How much data do I need?
A: For GNN training:
   - Minimum: 10K-50K examples (achievable in 30 min-1 hour)
   - Good: 100K-500K examples (2-6 hours)
   - Comprehensive: 1M+ examples (8-24 hours)

Q: Do I need the RF model to collect data?
A: No! Data collection uses exact pricing (SPPRC), not the RF pruning.
   The RF model is only used during the B&P solve for pruning.


FILES YOU'LL USE:
=================

✓ collect_training_data.py - Main data collection script (KEEP)
✗ collect_gnn_training_data.py - DUPLICATE (can remove)
✓ batch_collect_training_data.py - Batch runner (KEEP)
✓ prepare_gnn_data.py - Data preparation (KEEP)

✓ model_wrapper.py - For RF model loading (KEEP)
✓ gnn_pruning.py - For GNN inference (KEEP)
✓ pricing.py - Pricing logic (KEEP)


NEXT STEPS:
===========
1. Run: python collect_training_data.py --instances_dir [...] --max 3 --output test.csv
2. Check test.csv - should have 18 columns + 1 label
3. Run: python prepare_gnn_data.py --data test.csv --analyze
4. View statistics in output
5. Scale up to larger batch collection
6. Train your GNN model on the PyG tensors
7. Load model in gnn_pruning.py for inference
"""
