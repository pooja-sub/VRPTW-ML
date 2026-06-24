"""
Simplified GNN Training Data Collection - Focused on Feature Extraction

This collects labeled training data by:
1. Running exact pricing on root node of B&P for each instance
2. Recording which expanded-state edges appear in optimal negative routes
3. Extracting features for those edges
"""

import sys
import os
import csv
import time
import numpy as np
from typing import List, Dict, Tuple, Set

# Setup paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    # Prefer shared Common implementations to avoid running package main scripts
    from Common.paramsVRP import ParamsVRP
    from Common.route import Route
    from Branching.main import savings_heuristic_vrptw
    from Common.SPPRC import SPPRC
except Exception:
    # Fallback to local modules if Common/Branching not available on path
    try:
        from paramsVRP import ParamsVRP
        from route import Route
        from main import savings_heuristic_vrptw
        from SPPRC import SPPRC
    except Exception:
        raise

import gurobipy as gp
from gurobipy import GRB


class SimpleGNNDataCollector:
    """Collects GNN training data at the root node of B&P."""
    
    def __init__(self, output_csv: str):
        self.output_csv = output_csv
        self.data_rows = []
        self._init_csv()
    
    def _init_csv(self):
        """Create CSV with headers."""
        headers = [
            "instance",
            "cg_iteration",
            "node_dest_customer",
            "accumulated_demand",
            "elapsed_time",
            "remaining_capacity",
            "time_slack",
            "edge_distance",
            "edge_travel_time",
            "direct_reduced_cost",
            "can_fit_demand",
            "time_window_feasible",
            "slack_at_target",
            "is_mandatory_arc",
            "rmp_objective",
            "avg_dual_magnitude",
            "max_dual_magnitude",
            "num_routes_in_rmp",
            "label"
        ]
        
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def add_row(self, row: Dict):
        """Add a data row."""
        self.data_rows.append(row)
        
        # Write immediately to avoid memory buildup
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def collect_instance(self, instance_path: str, instance_name: str, init_mode: str = 'savings') -> bool:
        """Run data collection on a single instance."""
        print(f"\n{'='*70}")
        print(f"Instance: {instance_name}")
        print(f"{'='*70}")
        
        try:
            # Load params
            params = ParamsVRP()
            params.init_params(instance_path)
            print(f"Loaded: {params.nbclients} customers, capacity={params.capacity}")
            
            # Generate initial routes
            print("Generating initial routes... mode=", init_mode)
            if init_mode == 'single':
                # Create single-customer routes depot->i->depot for all customers
                initial_routes = []
                for i in range(1, params.nbclients):
                    if params.dist[0][i] < params.verybig - 1e-6 and params.dist[i][params.nbclients] < params.verybig - 1e-6:
                        route = Route(path=[0, i, params.nbclients])
                        initial_routes.append(route)
            else:
                initial_routes = savings_heuristic_vrptw(params)
            print(f"Initial routes: {len(initial_routes)}")
            
            # Run root node CG loop and collect data
            self._run_root_cg_collection(params, initial_routes, instance_name)
            
            return True
            
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_root_cg_collection(self, params: ParamsVRP, initial_routes: List[Route],
                                instance_name: str):
        """Run CG at root node, collecting data for each iteration."""
        
        print("\nSetting up RMP...")
        
        # Setup Gurobi RMP
        model = gp.Model("RootCG")
        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)
        
        routes = []
        for route in initial_routes:
            cost = sum(params.dist[route.path[i]][route.path[i + 1]] 
                      for i in range(len(route.path) - 1))
            route.set_cost(cost)
            routes.append(route)
        
        # Variables
        y = model.addVars(len(routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)
        
        # Constraints: each customer served exactly once
        model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(routes) 
                        if customer in route.path[1:-1]) >= 1
             for customer in range(1, params.nbclients)),
            "serve"
        )
        
        # Objective
        model.setObjective(
            gp.quicksum(y[i] * routes[i].cost for i in range(len(routes))),
            GRB.MINIMIZE
        )
        
        model.update()
        known_routes = {tuple(r.path) for r in routes}
        
        # CG iterations
        max_iter = 50
        iteration = 0
        
        while iteration < max_iter:
            print(f"\n[CG Iteration {iteration}]")
            
            # Solve RMP
            model.optimize()
            if model.status != GRB.OPTIMAL:
                print(f"  RMP status: {model.status}, stopping")
                break
            
            rmp_obj = model.objVal
            print(f"  RMP Obj: {rmp_obj:.2f}")
            
            # Extract duals
            constraints = model.getConstrs()
            duals = np.array([c.Pi for c in constraints if "serve" in c.ConstrName])
            
            dual_stats = {
                'mean': float(np.mean(np.abs(duals))) if len(duals) > 0 else 0,
                'max': float(np.max(np.abs(duals))) if len(duals) > 0 else 0,
            }
            
            print(f"  Duals: mean={dual_stats['mean']:.4f}, max={dual_stats['max']:.4f}")
            
            # Run pricing with exact solver (Common.SPPRC)
            print(f"  Running exact pricing...")
            pricing_solver = SPPRC(userParam=params)
            # Call shortestPath to generate negative reduced-cost routes
            new_routes = pricing_solver.shortestPath(params, [], 20, early_stop=True, lambda_pricing=False, dual_pi=duals, max_columns=20)
            
            if not new_routes:
                print(f"  No negative reduced-cost routes found. OPTIMAL at iteration {iteration}.")
                break
            
            print(f"  Found {len(new_routes)} negative routes")
            
            # Collect data: for each route found, extract features
            # This is a simplified version - we record that these edges were used
            self._record_routes_data(
                params, new_routes, duals,
                instance_name, iteration, rmp_obj,
                dual_stats['mean'], dual_stats['max'], len(routes),
                label=1  # These edges are in negative routes
            )
            
            # Add new routes to model
            added = 0
            for route in new_routes:
                route_tuple = tuple(route.path)
                if route_tuple not in known_routes:
                    known_routes.add(route_tuple)
                    cost = sum(params.dist[route.path[i]][route.path[i+1]] 
                              for i in range(len(route.path)-1))
                    route.set_cost(cost)
                    routes.append(route)
                    
                    # Add to model
                    y_var = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{len(routes)-1}", lb=0.0)
                    for customer in route.path[1:-1]:
                        # Find constraint for this customer
                        for c in constraints:
                            if c.ConstrName == f"serve[{customer}]":
                                c.RHS += y_var
                    added += 1
            
            model.update()
            print(f"  Added {added} new routes to model")
            iteration += 1
        
        print(f"\nCG completed at iteration {iteration}")
    
    def _record_routes_data(self, params: ParamsVRP, routes: List[Route], duals: np.ndarray,
                           instance_name: str, cg_iter: int, rmp_obj: float,
                           avg_dual_mag: float, max_dual_mag: float, num_routes: int,
                           label: int):
        """
        Record features for edges in the given routes.
        Simplified version: record one row per route representing its key edge.
        """
        
        for route_idx, route in enumerate(routes):
            path = route.path
            
            # Record each edge in the route
            for edge_idx in range(len(path) - 1):
                from_node = path[edge_idx]
                to_node = path[edge_idx + 1]
                
                # Skip depot-to-depot
                if from_node == 0 and to_node == params.nbclients:
                    continue
                
                # Calculate features
                distance = params.dist[from_node][to_node]
                travel_time = params.ttime[from_node][to_node]
                
                # For scoring: reduced cost of the arc
                if to_node < params.nbclients:
                    direct_rc = distance - duals[to_node - 1]
                else:
                    direct_rc = distance
                
                # Demand feasibility
                can_fit = 1  # Assuming route is feasible
                
                # Time window feasibility
                time_feasible = 1  # Assuming route is feasible
                
                # Slack at target
                slack = 0  # Simplified
                
                # Check if mandatory
                is_mandatory = 1 if (from_node == 0 or to_node == params.nbclients) else 0
                
                # Node destination info
                if to_node < params.nbclients:
                    node_dest = to_node
                else:
                    node_dest = -1  # Depot
                
                row = {
                    "instance": instance_name,
                    "cg_iteration": cg_iter,
                    "node_dest_customer": node_dest,
                    "accumulated_demand": 0,  # Would need to track through path
                    "elapsed_time": 0,  # Would need to track through path
                    "remaining_capacity": params.capacity,  # Would need to track
                    "time_slack": slack,
                    "edge_distance": distance,
                    "edge_travel_time": travel_time,
                    "direct_reduced_cost": direct_rc,
                    "can_fit_demand": can_fit,
                    "time_window_feasible": time_feasible,
                    "slack_at_target": slack,
                    "is_mandatory_arc": is_mandatory,
                    "rmp_objective": rmp_obj,
                    "avg_dual_magnitude": avg_dual_mag,
                    "max_dual_magnitude": max_dual_mag,
                    "num_routes_in_rmp": num_routes,
                    "label": label
                }
                
                self.add_row(row)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect GNN training data from VRPTW instances"
    )
    parser.add_argument(
        "--instances_dir",
        default=r"C:\Users\CiSTUP\Downloads\CVRTPW_Dataset\solomon_25_customer_instances",
        help="Directory with instance files"
    )
    parser.add_argument(
        "--output",
        default="gnn_training_data.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=5,
        help="Max instances to process"
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        help="Specific instances to process"
    )
    parser.add_argument(
        "--init",
        choices=["savings", "single"],
        default="savings",
        help="Initial route construction: 'savings' or 'single' (depot->i->depot)"
    )
    
    args = parser.parse_args()
    
    # Determine instance list
    instances_to_process = []
    
    if args.instances:
        instances_to_process = args.instances
    else:
        # Load from directory
        if os.path.isdir(args.instances_dir):
            all_files = sorted([
                os.path.join(args.instances_dir, f)
                for f in os.listdir(args.instances_dir)
                if f.endswith('.txt')
            ])
            instances_to_process = all_files[:args.max]
        else:
            print(f"Directory not found: {args.instances_dir}")
            return
    
    print(f"Processing {len(instances_to_process)} instances")
    print(f"Output: {args.output}\n")
    
    collector = SimpleGNNDataCollector(args.output)
    
    successful = 0
    for i, instance_path in enumerate(instances_to_process, 1):
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        print(f"\n[{i}/{len(instances_to_process)}]", end=" ")
        
        if collector.collect_instance(instance_path, instance_name, init_mode=args.init):
            successful += 1
    
    print(f"\n{'='*70}")
    print(f"Data collection complete!")
    print(f"Processed: {successful}/{len(instances_to_process)} instances")
    print(f"Data saved to: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
