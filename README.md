# VRPTW-ML: Branch-and-Price with Machine Learning Guided Pricing

## Overview

This repository implements a **Branch-and-Price algorithm** for the **Vehicle Routing Problem with Time Windows (VRPTW)**, enhanced with a **Machine Learning (ML)-guided pricing strategy**.

The implementation consists of three main modules:

- `LPRelaxation/` – Solves the LP relaxation and generates labeled data for ML training  
- `Branching/` – Implements the full Branch-and-Price framework  
- `MLPricing/` – Integrates a trained ML model to prune unpromising arcs before solving the pricing problem  

The goal is to accelerate column generation by reducing the search space of the pricing subproblem.

---

## Vehicle Routing Problem with Time Windows (VRPTW)

Given:
- A depot
- A set of customers with demand and time windows
- A homogeneous fleet of vehicles with capacity constraints

Objective:
- Minimize total routing cost
- Serve each customer exactly once
- Respect vehicle capacity and time window constraints

---

## Module Description

### LPRelaxation/

This module:

- Solves the LP relaxation of the Restricted Master Problem (RMP)
- Runs column generation without branching
- Solves the pricing subproblem iteratively
- Generates feasible columns (routes)

### Data Collection for ML

During column generation, this module also:

- Tracks arcs that appear in columns added to the RMP
- Labels arcs as:
  - **Positive**: if they appear in generated columns
  - **Negative**: otherwise
- Extracts arc-level features
- Stores labeled data for supervised ML training

This enables learning which arcs are likely to belong to useful columns.

---

### Branching/

This module implements the complete **Branch-and-Price framework**, including:

- Initialization of the Restricted Master Problem
- Column generation
- Branching on fractional variables
- Node exploration in the search tree
- Solving pricing subproblems at each node

This serves as the baseline exact method.

---

### MLPricing/

This module enhances the pricing step using a trained ML model.

Workflow:

1. Load trained arc-classification model
2. Predict promising/unpromising arcs
3. Prune arcs predicted as unlikely to appear in useful columns
4. Solve the reduced pricing subproblem
5. Continue Branch-and-Price with pruned network, reverting to the original network when needed

---

## Execution Flow

Each module contains a `main.py` file.

You must pass `datasetPath` as a command-line argument when running the script.

### General Format

```bash
python main.py --datasetPath <path_to_dataset> --SHOWFIG <True/False>
```

---

## Dependencies

Install using:

```bash
pip install -r requirements.txt
```

---

## References

This implementation builds upon classical column generation and recent
machine learning–assisted pricing techniques, including:

1. **Morabit, M., Desaulniers, G., & Lodi, A. (2021)**.
   *Machine-Learning-Based Column Selection for Column Generation.*  
   Transportation Science, 2021

2. **Morabit, M., Desaulniers, G., & Lodi, A. (2023)**.  
  *Machine-Learning-Based Arc Selection for Constrained Shortest Path Problems in Column Generation.*  
  *INFORMS Journal on Optimization*, 5(2), 191–210. :contentReference[oaicite:0]{index=0}
