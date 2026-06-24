# GNN Training Data Features for VRPTW Expanded-State Scoring

## Overview
The GNN predicts: **Should this expanded-state edge be kept for pricing at the current dual state?**

The scorer operates on the **expanded graph** where nodes are tuple states built from the pricing backend, not the original customer graph. Each node represents a reachable partial-route state, so the graph is sparse and easier to model than the original complete arc set.

---

## Feature Categories

### 1. NODE FEATURES (Expanded-State Level)
These characterize each tuple-state node in the expanded graph.

#### Inference-Time Node Vector Used by `MLPricingGNN/gnn_pruning.py`
- `x_coord`: Float - coordinate of the node's destination customer
- `y_coord`: Float - coordinate of the node's destination customer
- `demand`: Float - demand of the node's destination customer
- `accumulated_demand`: Float - total demand accumulated along the expanded state
- `elapsed_time`: Float - arrival time at the destination customer in this state
- `remaining_capacity`: Float = `capacity - accumulated_demand`
- `time_slack`: Float = `b[destination] - elapsed_time`

---

### 2. EDGE FEATURES (Expanded-Edge Level)
These characterize each candidate edge between expanded states.

#### Inference-Time Edge Vector Used by `MLPricingGNN/gnn_pruning.py`
- `distance`: Float - travel distance on the underlying parent arc
- `travel_time`: Float - travel time on the underlying parent arc
- `direct_reduced_cost`: Float = `distance - dual_pi[target]`
- `can_fit_demand`: Binary - whether the target state's demand remains feasible
- `time_feasible`: Binary - whether the target state's time window remains feasible
- `slack_at_target`: Float - time slack after arriving at the target state
- `is_mandatory`: Binary - keeps depot and sink connectivity protected during pruning

---

### 3. GLOBAL/CONTEXT FEATURES (Problem State)
These describe the overall solve state during an RMP solve.

#### LP State
- `cg_iteration`: Int - current column generation iteration number
- `rmp_objective`: Float - current RMP objective value
- `gap`: Float = `(ub - lb) / ub` - optimality gap
- `num_routes_in_rmp`: Int - number of columns currently in RMP

#### Problem Statistics
- `avg_dual_magnitude`: Float - mean absolute value of all duals
- `max_dual_magnitude`: Float - maximum absolute dual value
- `num_customers`: Int - problem size
- `vehicle_capacity`: Float - shared across all vehicles
- `time_horizon`: Int - [0, max_time_window_end]

---

### 4. LABEL: TARGET VARIABLE

#### Training Label (Supervised)
For each expanded-state edge, determine:
- **Label = 1**: Edge is kept by the exact pricing solver when generating a negative reduced-cost route
- **Label = 0**: Edge is not needed for the best negative reduced-cost routes at that dual state

#### How to Collect:
```
FOR each solved root-node CG iteration:
  Extract duals π from the RMP
  Build the expanded graph for the instance

  FOR each expanded-state edge:
    score the edge with the GNN input tensors
    run exact pricing on the pruned expanded graph
    label the edge using whether it survives on a negative reduced-cost route
```

---

## Feature Engineering Summary

### Static Features (don't change per CG iteration)
- Customer coordinates, demand, time windows, service time
- Problem dimensions (capacity, time horizon)
- Graph topology (degrees, shortest paths if precomputed)

### Dynamic Features (change per CG iteration)
- Dual values π from RMP
- Reduced costs derived from duals
- Arc feasibility flags
- Elapsed time / accumulated demand (per state)

### Derived Features (computed on-the-fly)
- `slack = time_window_end - arrival_time`
- `remaining_capacity = capacity - accumulated_demand`
- `reduced_cost_ratio`, `cost_per_time`
- Dual contributions relative to problem scale

---

## Practical Data Collection

### Format: CSV Per CG Iteration

```
instance,cg_iter,node_dest,accumulated_demand,elapsed_time,remaining_capacity,time_slack,
edge_distance,edge_travel_time,direct_rc,can_fit,time_feas,slack_to,is_mandatory,
cg_objective,avg_dual,num_routes_rmp,label
```

### Example Row
```
r201.txt,5,7,180,450,220,35,
25.3,18,12.5,1,1,35,0,
1523.4,3.2,18,1
```

### Scale
- **Small instance (20-30 customers):** ~500K-1M arcs per CG iteration × 10-50 iterations = 5M-50M training samples
- **Medium instance (50-100 customers):** 10M-100M+ samples per root LP
- **Multiple instances:** Combine data from 10-20 instances for diverse training set

---

## Feature Normalization Tips

1. **Distance features:** Normalize by problem scale (e.g., divide by max_distance)
2. **Time features:** Normalize by time_horizon
3. **Demand:** Normalize by capacity
4. **Dual values:** Keep them in the edge reduced-cost term or normalize them consistently per CG iteration
5. **Binary/categorical:** Keep them as 0/1 floats

---

## PyG Message Passing Setup

Your `EdgeScoringGNN` will use:

```python
# Node embedding inputs
x = [
  x_coord, y_coord,           # location of the destination customer
  demand,                     # customer demand
  accumulated_demand,         # partial-route resource state
  elapsed_time,               # partial-route resource state
  remaining_capacity,         # residual vehicle capacity
  time_slack,                 # deadline slack at this state
]  # Shape: [num_nodes_in_expanded_graph, node_dim=7]

# Edge attribute inputs
edge_attr = [
  distance, travel_time,      # parent arc properties
  direct_rc,                  # current reduced-cost signal
  can_fit_demand, time_feasible,  # feasibility flags
  slack_at_j,                 # time margin at the target state
  is_mandatory,               # keep depot/sink connectivity stable
]  # Shape: [num_edges, edge_dim=7]

# Message passing
# h_i, h_j -> MLP -> edge_score -> sigmoid -> P(label=1)

# Label
y = [0 or 1 for each edge]  # Shape: [num_edges]
```

---

## Training Recommendations

1. **Imbalance:** Expect ~70-80% class-0 (arcs not in negative routes), 20-30% class-1
   - Use weighted loss or class weights to handle imbalance
   
2. **Temporal dynamics:** Duals change each CG iteration
   - Option A: Train on data from multiple CG iterations mixed
   - Option B: Train separate model per iteration (harder)
   
3. **Generalization:** Train on multiple instances
   - Normalize features across instances
   - Validate on held-out instances (not just held-out CG iterations)

4. **Hard negatives:** Arcs close to the decision boundary are most important
   - Oversample arcs with scores near 0.5

---

## Example Training Data Collection Script Pseudocode

```python
def collect_gnn_training_data(instance_paths, output_csv):
    """
    Collect features for training GNN arc scorer.
    Run on root node of B&P for each instance.
    """
    rows = []
    
    for instance_path in instance_paths:
        params = ParamsVRP()
        params.init_params(instance_path)
        
        # Run root B&P with CG
        bp = BranchAndBound(use_expanded_pricing=True)
        cg = ColumnGeneration(params, ...)
        
        iteration = 0
        while cg_not_converged:
            cg.solve_rmp()  # RMP solve
            duals = cg.extract_duals()
            
            # For each feasible arc in expanded graph
            for arc in expanded_graph.edges:
                i, j = arc
                
                # Extract node features
                node_feat_i = extract_node_features(i, params, duals)
                node_feat_j = extract_node_features(j, params, duals)
                
                # Extract edge features
                edge_feat = extract_edge_features(
                    i, j, params, duals, node_feat_i, node_feat_j
                )
                
                # Solve SPPRC to get label
                routes = cg.solve_pricing()
                label = 1 if arc_in_negative_route(arc, routes) else 0
                
                # Collect context
                context = {
                    'instance': instance_path,
                    'cg_iteration': iteration,
                    'cg_objective': cg.current_obj,
                    'avg_dual': np.mean(np.abs(duals)),
                    'num_routes': len(cg.routes),
                }
                
                rows.append({
                    **node_feat_i,
                    **node_feat_j,
                    **edge_feat,
                    **context,
                    'label': label,
                })
            
            iteration += 1
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Collected {len(rows)} training examples")
```

---

## Summary Checklist

**Node Features (7-10 features per node):**
- [x] Customer ID, coordinates, demand, TW, service time
- [x] Accumulated demand, elapsed time, time slack
- [x] Dual price, reduced cost contribution
- [x] Remaining capacity

**Edge Features (7-9 features per edge):**
- [x] Distance, travel time, direct reduced cost
- [x] Feasibility flags (capacity, time)
- [x] Time slack at destination
- [x] Reduced cost ratio, cost efficiency
- [x] Dual contribution

**Context/Global Features (for normalization):**
- [x] CG iteration, RMP objective, gap
- [x] Problem scale (num customers, capacity, horizon)
- [x] Dual statistics (mean, max)

**Label:**
- [x] Arc appears in negative reduced-cost route? (Yes/No = 1/0)

**Scale:**
- [x] Target: 5M-100M+ examples depending on instance size
- [x] Multiple instances for generalization
- [x] Normalize across instances
