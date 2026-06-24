import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gurobipy as gp
from gurobipy import GRB
from Common.paramsVRP import ParamsVRP
from Common.route import Route
from Common.SPPRC import SPPRC
from expanded_pricing import ExpandedGraphPricing
import numpy as np
import time

# try to import the ML pricing module (local MLPricing.pricing)
try:
    from pricing import build_reduced_graph, run_ml_pricing_iteration, apply_gnn_pruning_to_expanded_graph
except Exception:
    build_reduced_graph = None
    run_ml_pricing_iteration = None
    apply_gnn_pruning_to_expanded_graph = None

class ColumnGeneration:
    def __init__(self, user_param, use_expanded_pricing=False, expanded_max_m=None, expanded_max_routes=20,
                 shared_expanded_pricing_reduced=None, shared_expanded_pricing_full=None,
                 reduced_graph_min_columns=5):
        self.paramsVRP = user_param
        self.routes = []
        self.use_expanded_pricing = use_expanded_pricing
        self.expanded_max_m = expanded_max_m
        self.expanded_max_routes = expanded_max_routes
        self.expanded_pricing_reduced = shared_expanded_pricing_reduced
        self.expanded_pricing_full = shared_expanded_pricing_full
        self.reduced_graph_min_columns = max(0, int(reduced_graph_min_columns))
        self._use_reduced_expanded = bool(self.expanded_pricing_reduced)

    def _ensure_expanded_pricing(self):
        if self.expanded_pricing_full is not None:
            return
        self.expanded_pricing_full = ExpandedGraphPricing(
            user_param=self.paramsVRP,
            max_m=self.expanded_max_m,
            quiet_graph_logs=True,
            quiet_spprc_logs=True,
        )

    def compute_col_gen(self, initial_routes):
        """
        Execute the column generation algorithm.

        :param initial_routes: Initial route list
        :return: Optimal objective value
        """
        #try:
        # Initialize Gurobi model
        model = gp.Model("Column Generation")

        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)

        # Add initial routes
        for route in initial_routes:
            cost = sum(self.paramsVRP.dist[route.path[i]][route.path[i + 1]] for i in range(len(route.path) - 1))
            route.set_cost(cost)
            self.routes.append(route)

        # Track path signatures to avoid adding identical columns forever.
        known_route_paths = {tuple(route.path) for route in self.routes}

        # Create variables and objective function
        y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

        # Add constraints: each customer must be served once
        constraints = model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
             for client in range(1, self.paramsVRP.nbclients)),
            "ClientService"
        )

        model.update()
        #print(constraints)

        # Set objective function
        model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)

        # Column generation main loop
        iteration = 0
        total_start = time.time()
        rmp_total = 0.0
        pp_total = 0.0

        if self.use_expanded_pricing:
            self._ensure_expanded_pricing()

        while True:
            # Solve current model (RMP)
            rmp_start = time.time()
            model.optimize()
            rmp_time = time.time() - rmp_start
            rmp_total += rmp_time
            if model.status == GRB.OPTIMAL:
                print(f"[-----Column Generation -----] Iteration {iteration}: Objective = {model.objVal}  RMP_time = {rmp_time:.3f}s")
            elif model.status == GRB.INFEASIBLE:
                print(f"[-----Column Generation -----] Iteration {iteration}: Model is infeasible.  RMP_time = {rmp_time:.3f}s")
            elif model.status == GRB.UNBOUNDED:
                print(f"[-----Column Generation -----] Iteration {iteration}: Model is unbounded.  RMP_time = {rmp_time:.3f}s")
            else:
                print(f"[-----Column Generation -----] Iteration {iteration}: Model not solved. Status = {model.status}  RMP_time = {rmp_time:.3f}s")

            objectiveFunc = model.getObjective()
            '''
            print(f"Model Objective Function: {objectiveFunc}")
            constraints_ = model.getConstrs()
            for i, constr in enumerate(constraints_):
                print(f"Constraint {i}: {constr.ConstrName} with Linear Expression: {model.getRow(constr)} {constr.Sense} {constr.RHS}")
            '''
            #print(f"y = {y}")

            # Get dual prices
            pi = [constr.Pi for constr in constraints.values()]
            #print(f"Iteration {iteration}: Objective = {model.objVal}, Pi = {pi}")

            # Update SPPRC cost matrix
            for i in range(1, self.paramsVRP.nbclients):
                for j in range(self.paramsVRP.nbclients + 1):
                    self.paramsVRP.cost[i][j] = self.paramsVRP.dist[i][j] - pi[i - 1]
                    if self.paramsVRP.cost[i][j] < 0:
                        #print(f"Negative cost found: {self.paramsVRP.cost[i][j]} at {i}, {j}")
                        pass


            # Solve pricing problem
            new_routes = []
            pricing_mode = "spprc"

            if self.use_expanded_pricing:
                active_backend = self.expanded_pricing_full
                pricing_mode = "full"
                if self._use_reduced_expanded and self.expanded_pricing_reduced is not None:
                    active_backend = self.expanded_pricing_reduced
                    pricing_mode = "reduced"

                pp_start = time.time()
                max_routes = max(1, min(self.expanded_max_routes, self.paramsVRP.nbclients - 1))
                guided_graph = None
                if apply_gnn_pruning_to_expanded_graph is not None:
                    try:
                        guided_graph = apply_gnn_pruning_to_expanded_graph(
                            active_backend,
                            self.paramsVRP,
                            dual_pi=pi,
                        )
                    except Exception as exc:
                        print(f"[MLPricing][GNN] Expanded-graph pruning error: {exc}; using cached graph")
                        guided_graph = None

                new_routes = active_backend.price(
                    user_param=self.paramsVRP,
                    dual_pi=pi,
                    max_routes=max_routes,
                    expanded_graph=guided_graph,
                )
                if (pricing_mode == "reduced"
                        and len(new_routes) < self.reduced_graph_min_columns
                        and self.expanded_pricing_full is not None):
                    print(
                        f"[MLPricing][PP] Reduced expanded pricing generated {len(new_routes)} columns "
                        f"(< {self.reduced_graph_min_columns}); switching to full expanded graph"
                    )
                    self._use_reduced_expanded = False
                    new_routes = self.expanded_pricing_full.price(
                        user_param=self.paramsVRP,
                        dual_pi=pi,
                        max_routes=max_routes,
                    )
                    pricing_mode = "full-fallback"
                pp_time = time.time() - pp_start
                pp_total += pp_time
            else:
                sp = SPPRC(self.paramsVRP)

                # If MLPricing available, run ML-based pricing; otherwise fallback to SPPRC
                if build_reduced_graph and run_ml_pricing_iteration:
                    # lazy build of A_r (RF-based arc set)
                    if not hasattr(self, '_A_r') or self._A_r is None:
                        try:
                            self._A_r = build_reduced_graph(self.paramsVRP)
                        except Exception as e:
                            print(f"Failed to build reduced graph: {e}")
                            self._A_r = set()
                        # Enable reduced-graph pricing when we have a non-empty ML arc set.
                        # run_ml_pricing_iteration can switch back to the full graph if the
                        # reduced network does not generate enough columns.
                        self._useReducedG = bool(self._A_r)

                    try:
                        pp_start = time.time()
                        new_routes, self._useReducedG, gen_count = run_ml_pricing_iteration(self.paramsVRP, pi, SPPRC, self._A_r, self._useReducedG)
                        pp_time = time.time() - pp_start
                        pp_total += pp_time
                    except Exception as e:
                        print(f"ML pricing error: {e}; falling back to SPPRC")
                        pp_start = time.time()
                        new_routes = []
                        sp.shortestPath(self.paramsVRP, new_routes, self.paramsVRP.nbclients - 1)
                        pp_time = time.time() - pp_start
                        pp_total += pp_time
                else:
                    pp_start = time.time()
                    sp.shortestPath(self.paramsVRP, new_routes, self.paramsVRP.nbclients - 1)
                    pp_time = time.time() - pp_start
                    pp_total += pp_time

            print(new_routes)
            # Log pricing summary for this iteration
            try:
                gen_count_val = len(new_routes) if new_routes else 0
            except Exception:
                gen_count_val = 0
            use_reduced = getattr(self, '_useReducedG', False)
            if self.use_expanded_pricing:
                use_reduced_expanded = getattr(self, '_use_reduced_expanded', False)
                print(
                    f"[MLPricing][PP] Iteration {iteration}: PP_time = {pp_time:.3f}s  "
                    f"gen_count = {gen_count_val}  mode = {pricing_mode}  "
                    f"useReducedExpanded = {use_reduced_expanded}"
                )
            else:
                print(f"[MLPricing][PP] Iteration {iteration}: PP_time = {pp_time:.3f}s  gen_count = {gen_count_val}  useReducedG = {use_reduced}")

            # Check if there are new negative cost paths
            if not new_routes:
                print("[-]No new negative cost paths found.")
                # Check model status
                if model.status == GRB.OPTIMAL:
                    print(f"[-----ColumnGeneration-----]Iteration {iteration}: Objective = {model.objVal}")
                elif model.status == GRB.INFEASIBLE:
                    print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is infeasible.")
                elif model.status == GRB.UNBOUNDED:
                    print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is unbounded.")
                else:
                    print(
                        f"[-----ColumnGeneration-----]Iteration {iteration}: Model not solved. Status = {model.status}")
                break

            # Keep only genuinely new columns by path.
            unique_new_routes = []
            for cand in new_routes:
                key = tuple(cand.path)
                if key in known_route_paths:
                    continue
                known_route_paths.add(key)
                unique_new_routes.append(cand)

            if not unique_new_routes:
                print("[-]Pricing returned only duplicate routes; stopping column generation.")
                break

            # Add new routes to the model
            for new_route in unique_new_routes:
                cost = sum(self.paramsVRP.dist[new_route.path[i]][new_route.path[i + 1]] for i in range(len(new_route.path) - 1))
                new_route.set_cost(cost)
                self.routes.append(new_route)

                # Remove all variables from the model
                vars_to_remove = model.getVars()
                for var in vars_to_remove:
                    model.remove(var)
                constrs_to_remove = model.getConstrs()
                for constr in constrs_to_remove:
                    model.remove(constr)

                # Create variables and objective function
                y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

                # Add constraint: each customer must be served once
                constraints = model.addConstrs(
                    (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
                     for client in range(1, self.paramsVRP.nbclients)),
                    "ClientService"
                )

                model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)

                model.update()

            iteration += 1
            '''
            # Check model status
            if model.status == GRB.OPTIMAL:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Objective = {model.objVal}")
            elif model.status == GRB.INFEASIBLE:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is infeasible.")
            elif model.status == GRB.UNBOUNDED:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is unbounded.")
            else:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model not solved. Status = {model.status}")
            '''

        # final timings
        total_time = time.time() - total_start
        print(f"[MLPricing] Column generation finished. Total_time = {total_time:.3f}s  RMP_total = {rmp_total:.3f}s  PP_total = {pp_total:.3f}s")

        # Output routes (silent here) — caller prints the final solution
        for i, route in enumerate(self.routes):
            route.set_Q(y[i].x)
            if route.Q > 0:
                print(f"Route {i}: Cost = {route.cost}, Q = {route.Q}, Path = {route.path}")

        return model.objVal, self.routes

        '''
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
        except Exception as e:
            print(f"Error in compute_col_gen: {e}")
        '''
