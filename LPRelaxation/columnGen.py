import gurobipy as gp
from gurobipy import GRB
import os
import sys

# Ensure repo root is importable when running from LPRelaxation/.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Branching.ESPPRC import ESPPRC
from paramsVRP import ParamsVRP
from route import Route
import numpy as np
import time

class ColumnGeneration:
    def __init__(self, user_param):
        self.paramsVRP = user_param
        self.routes = []
        
        # Timing statistics
        self.rmp_time = 0.0  # Cumulative time spent on RMP (Restricted Master Problem)
        self.pp_time = 0.0   # Cumulative time spent on PP (Pricing Problem)

    
    def compute_route_reduced_cost(self, route, pi):
        """
        Compute the reduced cost of a route.
        
        Reduced cost = route_cost - sum(pi[customer] for customers in route)
        
        :param route: Route object
        :param pi: Dual values (shadow prices)
        :return: Reduced cost of the route
        """
        reduced_cost = route.cost
        for customer in route.path[1:-1]:  # Exclude depots
            if 1 <= customer < self.paramsVRP.nbclients:
                reduced_cost -= pi[customer - 1]
        return reduced_cost

    def collect_arc_labels(self):
        """
        Collect all arcs used in the final RMP solution.
        
        Returns a set of (from, to) tuples representing used arcs.
        Includes all arcs in routes with positive flow (Q > 0).
        
        :return: Set of (from, to) arc tuples used in solution
        """
        used_arcs = set()
        
        # Extract arcs from all routes in solution
        for i, route in enumerate(self.routes):
            if route.Q > 1e-6:  # Route is in solution (Q > 0)
                # Route path is [start_depot, customer1, customer2, ..., end_depot]
                path = route.path
                for j in range(len(path) - 1):
                    arc = (path[j], path[j + 1])
                    used_arcs.add(arc)
        
        return used_arcs
    
    def collect_all_generated_arc_labels(self):
        """
        Collect all arcs that appear in ANY generated route (regardless of Q value).
        
        This labels arcs based on whether they were generated during column generation,
        not based on whether they're in the final solution. Good for ML training data.
        
        :return: Set of (from, to) arc tuples from all generated routes
        """
        all_arcs = set()
        
        # Extract arcs from ALL routes (not just those with Q > 0)
        for route in self.routes:
            path = route.path
            for j in range(len(path) - 1):
                arc = (path[j], path[j + 1])
                all_arcs.add(arc)
        
        return all_arcs

    def compute_col_gen(self, initial_routes):
        """
        Execute the column generation algorithm until optimality (no negative reduced cost routes).
        
        This is a pure LP relaxation - solves the master problem to optimality through iterative
        column generation without any branch-and-bound logic.
        
        :param initial_routes: Initial route list
        :return: Optimal objective value and final routes
        """
        # Start timing for column generation
        col_gen_start_time = time.time()
        
        # Initialize Gurobi model
        model = gp.Model("LP Relaxation - Column Generation")
        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)

        # Add initial routes
        for route in initial_routes:
            cost = sum(self.paramsVRP.dist[route.path[i]][route.path[i + 1]] for i in range(len(route.path) - 1))
            route.set_cost(cost)
            self.routes.append(route)

        # Create variables and objective function
        y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

        # Add constraints: each customer must be served exactly once
        # Note: nbclients includes depot in the line count, so actual customers are 1 to nbclients-1
        constraints = model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
             for client in range(1, self.paramsVRP.nbclients)),
            "ClientService"
        )

        model.update()

        # Set objective function
        model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)

        # Column generation main loop
        iteration = 0
        
        while True:
            # Solve current RMP (Restricted Master Problem)
            rmp_start = time.time()
            model.optimize()
            rmp_end = time.time()
            self.rmp_time += (rmp_end - rmp_start)
            elapsed_total = time.time() - col_gen_start_time
            
            if model.status == GRB.OPTIMAL:
                print(f"[LP-Relaxation Iteration {iteration}] RMP Optimal: Objective = {model.objVal:.4f} | Elapsed: {elapsed_total:.2f}s")
            elif model.status == GRB.INFEASIBLE:
                elapsed_total = time.time() - col_gen_start_time
                print(f"[LP-Relaxation Iteration {iteration}] RMP is INFEASIBLE - cannot serve all customers | Elapsed: {elapsed_total:.2f}s")
                return self.paramsVRP.verybig, self.routes
            elif model.status == GRB.UNBOUNDED:
                elapsed_total = time.time() - col_gen_start_time
                print(f"[LP-Relaxation Iteration {iteration}] RMP is UNBOUNDED | Elapsed: {elapsed_total:.2f}s")
                return -self.paramsVRP.verybig, self.routes
            else:
                elapsed_total = time.time() - col_gen_start_time
                print(f"[LP-Relaxation Iteration {iteration}] RMP not solved properly. Status = {model.status} | Elapsed: {elapsed_total:.2f}s")
                return self.paramsVRP.verybig, self.routes

            # Get dual prices (shadow prices on customer service constraints)
            pi = [constr.Pi for constr in constraints.values()]

            # Update reduced cost matrix for pricing problem
            # Reduced cost for arc (i,j) = distance[i][j] - dual[j] (if j is customer)
            # Reset cost matrix to base distances first
            self.paramsVRP.cost[:, :] = self.paramsVRP.dist[:, :]
            for j in range(1, self.paramsVRP.nbclients):  # customers only
                self.paramsVRP.cost[:, j] = self.paramsVRP.dist[:, j] - pi[j - 1]

            # Solve pricing problem (SPPRC)
            sp = ESPPRC(self.paramsVRP)
            new_routes = []
            
            pp_start = time.time()
            sp.shortestPath(self.paramsVRP, new_routes, self.paramsVRP.nbclients - 1)
            pp_end = time.time()
            self.pp_time += (pp_end - pp_start)
            elapsed_total = time.time() - col_gen_start_time
            
            if new_routes:
                min_cost = min([r.cost for r in new_routes])
                pp_time = pp_end - pp_start
                print(f"[LP-Relaxation Iteration {iteration}] Pricing: Generated {len(new_routes)} columns, min cost = {min_cost:.4f} | PP time: {pp_time:.2f}s | Total elapsed: {elapsed_total:.2f}s")
            else:
                elapsed_total = time.time() - col_gen_start_time
                print(f"[LP-Relaxation Iteration {iteration}] Pricing: No negative reduced cost routes found - OPTIMALITY ACHIEVED | Total elapsed: {elapsed_total:.2f}s")
                break

            # Add new routes to the model
            for new_route in new_routes:
                cost = sum(self.paramsVRP.dist[new_route.path[i]][new_route.path[i + 1]] for i in range(len(new_route.path) - 1))
                new_route.set_cost(cost)
                self.routes.append(new_route)

            # Rebuild model with all routes (including new ones)
            vars_to_remove = model.getVars()
            for var in vars_to_remove:
                model.remove(var)
            constrs_to_remove = model.getConstrs()
            for constr in constrs_to_remove:
                model.remove(constr)

            # Recreate variables and constraints
            y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)
            constraints = model.addConstrs(
                (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
                 for client in range(1, self.paramsVRP.nbclients)),
                "ClientService"
            )

            model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)
            model.update()

            iteration += 1

        # Set final variable values and output solution
        print(f"\n[LP-Relaxation] Solution:")
        print(f"{'Route':<6} {'Cost':<10} {'Flow':<10} {'Path':<30}")
        print("-" * 60)
        
        for i, route in enumerate(self.routes):
            route.set_Q(y[i].x)
            if route.Q > 1e-6:
                path_str = " -> ".join(map(str, route.path))
                print(f"{i:<6} {route.cost:<10.4f} {route.Q:<10.6f} {path_str:<30}")

        optimal_value = model.objVal
        elapsed_total = time.time() - col_gen_start_time
        print(f"\n[LP-Relaxation] Final Objective Value: {optimal_value:.4f}")
        print(f"[LP-Relaxation] Total Iterations: {iteration}")
        print(f"[LP-Relaxation] RMP Time: {self.rmp_time:.2f}s | PP Time: {self.pp_time:.2f}s | Total Elapsed: {elapsed_total:.2f}s")
        
        return optimal_value, self.routes

