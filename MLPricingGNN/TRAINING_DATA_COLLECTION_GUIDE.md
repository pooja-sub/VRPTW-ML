## GNN Training Data Collection - Summary

### Data Collection Pipeline Overview

Your three-layer pricing system:
1. **Stage 1 (RF Pruning)**: Random Forest model prunes full graph → reduced graph
2. **Stage 2 (Expanded Graph)**: Build expanded state-space for exact pricing  
3. **Stage 3 (GNN Pruning)**: GNN model prunes expanded graph → further reduced graph
4. **Stage 4 (Exact Pricing)**: SPPRC solves reduced expanded graph exactly

### To Collect GNN Training Data

**Step 1: Collect labeled edge examples**
```bash
cd C:\Users\CiSTUP\Desktop\A-Branch-and-Price-Algorithm-for-VRPTW\MLPricingGNN

# Quick test (3 instances, ~5 minutes)
python collect_training_data.py --instances_dir "C:\Users\CiSTUP\Downloads\CVRTPW_Dataset\solomon_25_customer_instances" --max 3 --output test_data.csv

# Standard collection (5 instances, ~30 minutes)
python collect_training_data.py --max 5 --output training_data.csv

# Large dataset (10+ instances, 1-2 hours)
python batch_collect_training_data.py --strategy standard --output_dir ./training_data
```

**Step 2: Analyze and prepare data**
```bash
# View statistics
python prepare_gnn_data.py --data training_data.csv --analyze

# Normalize and split for training
python prepare_gnn_data.py --data training_data.csv --normalize --split
```

**Step 3: Train your GNN**
- Use PyG tensors exported from `prepare_gnn_data.py`
- Save model to `MLPricingGNN/MLModels/gnn_model.pt` or similar
- GNN model will auto-load in `gnn_pruning.py` for inference

### What Gets Collected

For each edge in optimal pricing routes:
- **18 Features**: distance, travel time, reduced cost, capacity/time feasibility, dual values, etc.
- **Label**: 1 if edge in optimal route, 0 otherwise
- **Context**: Instance name, iteration, RMP objective

Output: CSV with ~15-50K rows per instance (varies with instance size)

---

## REDUNDANT CODE TO REMOVE

I found **duplicate/redundant files** you can delete:

### DELETE THESE (Recommended)
```
✗ MLPricingGNN/collect_gnn_training_data.py
  → Same functionality as collect_training_data.py
  → Older version with more complex arg parsing
  → Use collect_training_data.py instead (cleaner)
  
✗ MLPricingGNN/create_dummy_model.py
  → Was for testing, now you have real RF model
  → If you don't use it, safe to remove
```

### KEEP THESE (Essential)
```
✓ collect_training_data.py - Main data collector
✓ batch_collect_training_data.py - Batch runner for multiple instances
✓ prepare_gnn_data.py - Data preparation & analysis
✓ model_wrapper.py - RF model loader
✓ gnn_pruning.py - GNN inference engine
✓ pricing.py - RF + GNN pruning orchestration
✓ expanded_pricing.py - Expanded graph builder
✓ branchBound.py, columnGen.py, main.py - Core B&P solver
```

### Command to Remove Redundant Files
```bash
cd C:\Users\CiSTUP\Desktop\A-Branch-and-Price-Algorithm-for-VRPTW\MLPricingGNN
rm collect_gnn_training_data.py  # Windows: del collect_gnn_training_data.py
```

---

## Quick Reference: Data Collection Commands

```bash
# 1. Mini test (3 instances, 1-2 MB data)
python collect_training_data.py --max 3 --output test.csv

# 2. Quick batch (3 instances each of C/R/RC, ~50 MB)
python batch_collect_training_data.py --strategy quick --output_dir ./data

# 3. Standard (5-10 each, ~200-300 MB)
python batch_collect_training_data.py --strategy standard --output_dir ./data

# 4. Comprehensive (20+, ~500 MB - 1 GB, requires several hours)
python batch_collect_training_data.py --strategy comprehensive --output_dir ./data

# 5. Analyze data quality
python prepare_gnn_data.py --data test.csv --analyze

# 6. Prepare for training
python prepare_gnn_data.py --data test.csv --normalize --split --output ./pytorch_data
```

---

## Data Collection Estimates

| Strategy | Instances | Time | Data Size | Rows |
|----------|-----------|------|-----------|------|
| Quick | 3-5 | 5-15 min | 5-15 MB | 10K-30K |
| Standard | 10-15 | 30 min - 2 hr | 50-150 MB | 100K-300K |
| Comprehensive | 30+ | 4-12 hours | 300 MB - 1 GB | 500K-1.5M |

**Recommendation**: Start with "Quick", then scale to "Standard" once you've trained a baseline GNN.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError: No module 'Common' | Already fixed - sys.path updated |
| No CSV created | Check directory permissions, path exists, --max >= 1 |
| Very slow collection | Normal - exact pricing is NP-hard. Try --max 2-3 first |
| Out of memory | Increase time_limit or reduce max_instances |
| "ParamsVRP already has 27 items" | Already fixed - nbclients+1 instead of +2 |

---

## Files Changed/Created This Session

**Fixed**:
- ✓ Branching/ → Added sys.path, fixed nbclients+2 bugs
- ✓ BranchingExpandedPricing/ → Uses Common modules
- ✓ MLPricing/ → Added sys.path, fixed array bounds, created model_wrapper.py
- ✓ MLPricingGNN/ → Added sys.path, fixed array bounds, verified two-stage pruning

**Created**:
- ✓ GNN_DATA_COLLECTION_QUICKSTART.md (this file's companion guide)

**Recommendations**:
- Delete: collect_gnn_training_data.py, create_dummy_model.py (if unused)
- Keep: collect_training_data.py, batch_collect_training_data.py, prepare_gnn_data.py

---

## Next: Training Your GNN

Once you have CSV data:

1. **Exploratory Analysis**
   ```bash
   python prepare_gnn_data.py --data combined_training_data.csv --analyze
   ```
   → Check class balance, feature ranges, instance distribution

2. **Prepare PyG Format**
   ```bash
   python prepare_gnn_data.py --data combined_training_data.csv --normalize --split --output_format pytorch_geometric
   ```
   → Outputs train/val/test tensors for PyTorch

3. **Define & Train GNN**
   - Use PyTorch Geometric for graph neural networks
   - Input: Graph with node/edge/global features
   - Output: Binary edge classification (relevant / not relevant)
   - Save: `MLPricingGNN/MLModels/gnn_model.pt`

4. **Load in Solver**
   - `gnn_pruning.py` automatically loads model from MLModels/
   - Pass to `apply_gnn_pruning_to_expanded_graph()` in pricing.py
   - Inference happens during column generation

---
