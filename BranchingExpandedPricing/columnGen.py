import gurobipy as gp
from gurobipy import GRB
from paramsVRP import ParamsVRP
# from route import Route
from ESPPRC import ESPPRC
from expanded_pricing import ExpandedGraphPricing
import numpy as np
import time

class ColumnGeneration:
    def __init__(self, user_param, enable_column_deletion, deletion_threshold, min_nonbasic_columns,
                 lambda_pricing, lambda_factor, num_columns_to_keep,
                 enable_node_elimination, node_elimination_threshold,
                 use_expanded_pricing=True, expanded_max_m=None, expanded_max_routes=20,
                 shared_expanded_pricing=None):
        self.paramsVRP = user_param
        self.routes = []
        self.fixed_routes = set()  # Track indices of permanently eliminated routes
        
        # Column deletion parameters
        self.enable_column_deletion = enable_column_deletion
        self.deletion_threshold = deletion_threshold  # Number of iterations without being in basis
        self.min_nonbasic_columns = min_nonbasic_columns  # Minimum nonbasic columns to keep
        self.route_last_basis_iteration = {}  # Maps route index to last iteration where it was in basis
        self.current_iteration = 0  # Current column generation iteration count
        
        # Multi-column selection parameters
        self.lambda_pricing = lambda_pricing  # Apply lambda pricing rule (ratio-based)
        self.lambda_factor = lambda_factor  # Multiplier for lambda pricing
        self.num_columns_to_keep = num_columns_to_keep  # Number of columns to keep per iteration
        
        # Node elimination parameters
        self.enable_node_elimination = enable_node_elimination  # Enable/disable node elimination strategy
        self.node_elimination_threshold = node_elimination_threshold  # Multiplier for identifying high negative duals
        self.eliminated_customers = set()  # Track eliminated customers per pricing iteration

        # Optional expanded-graph pricing backend
        self.use_expanded_pricing = use_expanded_pricing
        self.expanded_max_m = expanded_max_m
        self.expanded_max_routes = expanded_max_routes
        # Optional shared backend built once at BranchAndBound level and reused across BnP nodes.
        self.expanded_pricing = shared_expanded_pricing
        
        # Timing statistics
        self.rmp_time = 0.0  # Cumulative time spent on RMP (Restricted Master Problem)
        self.pp_time = 0.0   # Cumulative time spent on PP (Pricing Problem)

    def _ensure_expanded_pricing(self):
        """Build expanded-graph cache once per ColumnGeneration object (i.e., per BnP node)."""
        if self.expanded_pricing is not None:
            return
        self.expanded_pricing = ExpandedGraphPricing(
            user_param=self.paramsVRP,
            max_m=self.expanded_max_m,
            quiet_graph_logs=True,
            quiet_spprc_logs=True,
        )

    def _route_key(self, route):
        """Hashable route identity based on full path."""
        return tuple(route.path)

    def _append_unique_route(self, route):
        """Add route only if no existing route has the same path."""
        key = self._route_key(route)
        for existing in self.routes:
            if self._route_key(existing) == key:
                return False
        self.routes.append(route)
        return True

    def _append(self, route):
        """Backward-compatible alias used by existing code paths."""
        return self._append_unique_route(route)

    def _rebuild_rmp(self, model):
        """Rebuild the RMP from current route pool and return (y, constraints)."""
        vars_to_remove = model.getVars()
        for var in vars_to_remove:
            model.remove(var)
        constrs_to_remove = model.getConstrs()
        for constr in constrs_to_remove:
            model.remove(constr)

        y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)
        constraints = model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
             for client in range(1, self.paramsVRP.nbclients)),
            "ClientService"
        )
        model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)
        model.update()
        return y, constraints


    def fix_variables_by_reduced_cost(self, model, y, lower_bound, upper_bound, constraints):
        """
        Permanently eliminate path variables whose reduced cost exceeds the optimality gap.
        
        If reduced_cost(route) > (upper_bound - lower_bound), the route cannot be part 
        of an optimal solution and is permanently removed from the RMP.
        
        :param model: Gurobi model
        :param y: Decision variables
        :param lower_bound: Current lower bound (LB)
        :param upper_bound: Current upper bound (UB)
        :param constraints: Model constraints
        :return: Number of routes fixed (eliminated)
        """
        if upper_bound >= self.paramsVRP.verybig - 1:
            # No valid upper bound yet, skip fixing
            return 0
        
        optimality_gap = upper_bound - lower_bound
        if optimality_gap < 0:
            return 0
        
        fixed_count = 0
        routes_to_keep = []
        
        # Get dual values
        pi = [constr.Pi for constr in constraints.values()]
        
        for i, route in enumerate(self.routes):
            if i in self.fixed_routes:
                # Already fixed, skip
                continue
            
            # Calculate reduced cost for this route
            # Reduced cost = route_cost - sum(pi[customer] for customers in route)
            reduced_cost = route.cost
            for customer in route.path[1:-1]:  # Exclude depots
                if 1 <= customer < self.paramsVRP.nbclients:
                    reduced_cost -= pi[customer - 1]
            
            # If reduced cost > gap, this route can never be in optimal solution
            if reduced_cost > optimality_gap + 1e-6:
                self.fixed_routes.add(i)
                fixed_count += 1
                print(f"  [Fixed route {i}] Reduced cost {reduced_cost:.4f} > Gap {optimality_gap:.4f}")
            else:
                routes_to_keep.append(i)
        
        if fixed_count > 0:
            print(f"[Variable Fixing] Eliminated {fixed_count} routes permanently. Remaining: {len(routes_to_keep)}")
        
        return fixed_count

    def delete_inactive_columns(self, model, y, constraints):
        """
        Delete columns (routes) that haven't been in the basis for deletion_threshold iterations.
        Keeps at least min_nonbasic_columns nonbasic routes.
        
        Variables not in basic solution for more than threshold iterations are removed from RMP.
        
        :param model: Gurobi model
        :param y: Decision variables
        :param constraints: Model constraints
        :return: Number of routes deleted
        """
        if not self.enable_column_deletion:
            return 0
        
        # Get basis status for all variables
        rmp_start = time.time()
        model.optimize()
        rmp_end = time.time()
        self.rmp_time += (rmp_end - rmp_start)
        
        if model.status != GRB.OPTIMAL:
            return 0
        
        # Update last basis iteration for routes currently in basis
        nonbasic_routes = []
        for i, route in enumerate(self.routes):
            if i in self.fixed_routes:
                continue
            
            if i < len(y):
                var = y[i]
                # Check if variable is basic (vBasis == 0 means basic)
                if var.VBasis == 0 or var.X > 1e-6:
                    self.route_last_basis_iteration[i] = self.current_iteration
                else:
                    nonbasic_routes.append(i)
        
        # Identify routes to delete (not in basis for > threshold iterations)
        routes_to_delete = []
        for i in nonbasic_routes:
            if i not in self.route_last_basis_iteration:
                self.route_last_basis_iteration[i] = self.current_iteration - 1
            
            iterations_inactive = self.current_iteration - self.route_last_basis_iteration[i]
            if iterations_inactive > self.deletion_threshold:
                routes_to_delete.append(i)
        
        # Keep at least min_nonbasic_columns nonbasic routes
        if len(nonbasic_routes) - len(routes_to_delete) < self.min_nonbasic_columns:
            # Sort by inactivity and keep the most recently used
            routes_to_delete.sort(key=lambda i: self.current_iteration - self.route_last_basis_iteration[i], reverse=True)
            num_to_keep = max(0, len(nonbasic_routes) - self.min_nonbasic_columns)
            routes_to_delete = routes_to_delete[:num_to_keep]
        
        if routes_to_delete:
            print(f"[Column Deletion] Deleting {len(routes_to_delete)} inactive routes at iteration {self.current_iteration}")
            for i in routes_to_delete:
                self.fixed_routes.add(i)
        
        return len(routes_to_delete)
    
    def eliminate_nodes_by_dual_values(self, pi):
        """
        Node elimination strategy: eliminate customers with very high negative dual values.
        
        A customer with a very low (highly negative) dual value π[c] << 0 makes it harder to achieve 
        negative reduced cost when including that customer in a route, since -π[c] becomes a large 
        positive contribution to the reduced cost.
        
        This method identifies customers whose dual values are extreme outliers on the negative side
        (more negative than Q1 - threshold * IQR, using interquartile range method).
        
        :param pi: List of dual values for customers (π[0] corresponds to customer 1)
        :return: Set of eliminated customer indices (1-indexed)
        """
        self.eliminated_customers = set()
        
        if not self.enable_node_elimination or len(pi) == 0:
            return self.eliminated_customers
        
        # Sort dual values to compute quartiles
        sorted_duals = sorted(pi)
        n = len(pi)
        
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        q1 = sorted_duals[n // 4]
        q3 = sorted_duals[3 * n // 4]
        iqr = q3 - q1  # Interquartile range
        
        # Outlier threshold: values below Q1 - threshold * IQR are considered extreme negative outliers
        # Standard outlier detection uses threshold=1.5, but we make it configurable
        dual_threshold = q1 - self.node_elimination_threshold * iqr
        
        # Identify and eliminate customers with dual values below threshold (extreme negative)
        for customer_idx, dual_value in enumerate(pi):
            if dual_value < dual_threshold:
                # Convert to 1-indexed customer number
                customer_id = customer_idx + 1
                self.eliminated_customers.add(customer_id)
        
        if self.eliminated_customers:
            min_dual = min(pi)
            max_dual = max(pi)
            print(f"  [Node Elimination] Eliminated {len(self.eliminated_customers)} customers with extreme negative duals")
            print(f"    Threshold: {dual_threshold:.4f} (Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f})")
            print(f"    Dual range: [{min_dual:.4f}, {max_dual:.4f}]")
            print(f"    Eliminated customers: {sorted(list(self.eliminated_customers)[:10])}{'...' if len(self.eliminated_customers) > 10 else ''}")
        
        return self.eliminated_customers
    
    def get_timing_stats(self):
        """
        Return timing statistics for RMP and PP.
        
        :return: Dictionary with 'rmp_time' and 'pp_time' keys
        """
        return {'rmp_time': self.rmp_time, 'pp_time': self.pp_time}
    
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

    def compute_col_gen(self, initial_routes, lower_bound=-1e10, upper_bound=1e10, fix_interval=10, bb_node_count=0, early_stop_pricing=False,
                        vehicle_lower_bound=None, vehicle_upper_bound=None):
        """
        Execute the column generation algorithm.

        :param initial_routes: Initial route list
        :param lower_bound: Current lower bound from branch-and-bound
        :param upper_bound: Current upper bound from branch-and-bound
        :param fix_interval: Number of iterations between variable fixing attempts
        :param bb_node_count: Current branch-and-bound node count (unused with iteration-based deletion)
        :param early_stop_pricing: Enable early stopping in SPPRC when negative cost route found
        :return: Optimal objective value and routes
        """
        #try:
        # Reset iteration counter for this column generation call
        self.current_iteration = 0
        
        # Initialize Gurobi model
        model = gp.Model("Column Generation")

        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)

        # Add initial routes
        for route in initial_routes:
            cost = sum(self.paramsVRP.dist[route.path[i]][route.path[i + 1]] for i in range(len(route.path) - 1))
            route.set_cost(cost)
            self._append(route)

        # Create variables and objective function
        y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

        # Add constraints: each customer must be served once
        constraints = model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
             for client in range(1, self.paramsVRP.nbclients)),
            "ClientService"
        )

        # Optional vehicle count bounds (for vehicle-count branching)
        # vehicle_constr_lower = None
        # vehicle_constr_upper = None
        # if vehicle_lower_bound is not None:
        #     vehicle_constr_lower = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) >= vehicle_lower_bound,
        #                                            name="VehicleCountLower")
        # if vehicle_upper_bound is not None:
        #     vehicle_constr_upper = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) <= vehicle_upper_bound,
        #                                            name="VehicleCountUpper")

        model.update()
        #print(constraints)

        # Set objective function
        model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)

        # Column generation main loop
        iteration = 0
        col_gen_start_time = time.time()

        if self.use_expanded_pricing:
            self._ensure_expanded_pricing()

        while True:
            elapsed_total = time.time() - col_gen_start_time
            # Solve current model (RMP)
            rmp_start = time.time()
            model.optimize()
            rmp_end = time.time()
            self.rmp_time += (rmp_end - rmp_start)
            
            if model.status == GRB.OPTIMAL:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Objective = {model.objVal} | Elapsed: {elapsed_total:.2f}s")
                
                objectiveFunc = model.getObjective()
                '''
                print(f"Model Objective Function: {objectiveFunc}")
                constraints_ = model.getConstrs()
                for i, constr in enumerate(constraints_):
                    print(f"Constraint {i}: {constr.ConstrName} with Linear Expression: {model.getRow(constr)} {constr.Sense} {constr.RHS}")
                '''
                #print(f"y = {y}")

                # Get dual prices (Pi is only available when model is OPTIMAL)
                pi = [constr.Pi for constr in constraints.values()]
            elif model.status == GRB.INFEASIBLE:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is infeasible. | Elapsed: {elapsed_total:.2f}s")
                # Prune this node by infeasibility - no dual values available
                return float('inf'), []
            elif model.status == GRB.UNBOUNDED:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is unbounded. | Elapsed: {elapsed_total:.2f}s")
                return float('inf'), []
            else:
                print(
                    f"[-----ColumnGeneration-----]Iteration {iteration}: Model not solved. Status = {model.status} | Elapsed: {elapsed_total:.2f}s")
                return float('inf'), []
            #print(f"Iteration {iteration}: Objective = {model.objVal}, Pi = {pi}")
            
            # Apply column deletion if enabled
            if self.enable_column_deletion and iteration > 0 and iteration % fix_interval == 0:
                deleted_count = self.delete_inactive_columns(model, y, constraints)
                
                if deleted_count > 0:
                    # Rebuild model without deleted routes
                    active_routes = [route for i, route in enumerate(self.routes) if i not in self.fixed_routes]
                    
                    if len(active_routes) == 0:
                        print("[ERROR] All routes have been deleted. Cannot continue.")
                        break
                    
                    # Track old indices to new indices for basis tracking
                    old_to_new = {}
                    new_idx = 0
                    for old_idx in range(len(self.routes)):
                        if old_idx not in self.fixed_routes:
                            old_to_new[old_idx] = new_idx
                            new_idx += 1
                    
                    # Update route_last_basis_iteration with new indices
                    new_route_last_basis_iteration = {}
                    for old_idx, iteration in self.route_last_basis_iteration.items():
                        if old_idx in old_to_new:
                            new_route_last_basis_iteration[old_to_new[old_idx]] = iteration
                    self.route_last_basis_iteration = new_route_last_basis_iteration
                    
                    # Update routes and reset fixed_routes
                    self.routes = active_routes
                    self.fixed_routes.clear()
                    
                    # Rebuild model
                    vars_to_remove = model.getVars()
                    for var in vars_to_remove:
                        model.remove(var)
                    constrs_to_remove = model.getConstrs()
                    for constr in constrs_to_remove:
                        model.remove(constr)
                    
                    y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)
                    constraints = model.addConstrs(
                        (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
                         for client in range(1, self.paramsVRP.nbclients)),
                        "ClientService"
                    )
                    # if vehicle_lower_bound is not None:
                    #     vehicle_constr_lower = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) >= vehicle_lower_bound,
                    #                                            name="VehicleCountLower")
                    # if vehicle_upper_bound is not None:
                    #     vehicle_constr_upper = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) <= vehicle_upper_bound,
                    #                                            name="VehicleCountUpper")
                    model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)
                    model.update()
                    continue
            
            # DISABLED: Apply variable fixing at regular intervals
            # if iteration > 0 and iteration % fix_interval == 0:
            #     print(f"[Iteration {iteration}] Applying variable fixing by reduced cost...")
            #     fixed_count = self.fix_variables_by_reduced_cost(model, y, lower_bound, upper_bound, constraints)
            #     
            #     # If routes were fixed, rebuild the model
            #     if fixed_count > 0:
            #         # Get active routes (not fixed)
            #         active_routes = [route for i, route in enumerate(self.routes) if i not in self.fixed_routes]
            #         
            #         if len(active_routes) == 0:
            #             print("[ERROR] All routes have been fixed. Cannot continue.")
            #             break
            #         
            #         # Rebuild model with only active routes
            #         self.routes = active_routes
            #         self.fixed_routes.clear()  # Reset indices after rebuilding
            #         
            #         # Remove all variables and constraints
            #         vars_to_remove = model.getVars()
            #         for var in vars_to_remove:
            #             model.remove(var)
            #         constrs_to_remove = model.getConstrs()
            #         for constr in constrs_to_remove:
            #             model.remove(constr)
            #         
            #         # Recreate with active routes only
            #         y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)
            #         constraints = model.addConstrs(
            #             (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
            #              for client in range(1, self.paramsVRP.nbclients)),
            #             "ClientService"
            #         )
            #         # if vehicle_lower_bound is not None:
            #         #     vehicle_constr_lower = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) >= vehicle_lower_bound,
            #         #                                            name="VehicleCountLower")
            #         # if vehicle_upper_bound is not None:
            #         #     vehicle_constr_upper = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) <= vehicle_upper_bound,
            #         #                                            name="VehicleCountUpper")
            #     model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)
            #     model.update()
                    
                    # Re-optimize after fixing
                    continue

            # Update SPPRC cost matrix
            for i in range(1, self.paramsVRP.nbclients):
                for j in range(self.paramsVRP.nbclients + 2):
                    self.paramsVRP.cost[i][j] = self.paramsVRP.dist[i][j] - pi[i - 1]
                    # if self.paramsVRP.cost[i][j] < 0:
                    #     #print(f"Negative cost found: {self.paramsVRP.cost[i][j]} at {i}, {j}")
                    #     pass

            # Apply node elimination strategy if enabled
            eliminated_nodes = set()
            if self.enable_node_elimination:
                eliminated_nodes = self.eliminate_nodes_by_dual_values(pi)

            new_routes = []

            # Solve pricing problem (PP)
            pp_start = time.time()
            if self.use_expanded_pricing:
                print("[ColumnGen] Calling expanded-graph pricing backend")
                max_routes = max(1, min(self.expanded_max_routes, self.num_columns_to_keep))
                new_routes = self.expanded_pricing.price(
                    user_param=self.paramsVRP,
                    dual_pi=pi,
                    max_routes=max_routes,
                )
            else:
                # Determine if this is the final pricing (proving optimality)
                # is_final_pricing = (iteration > 0)  # After first iteration, we're refining
                sp = ESPPRC(self.paramsVRP)
                # Enable early stopping except when we need to prove optimality
                print(f"[ColumnGen] Calling SPPRC: early_stop_pricing={early_stop_pricing}, lambda_pricing={self.lambda_pricing}")
                sp.shortestPath(self.paramsVRP, new_routes, self.paramsVRP.nbclients - 1,
                              early_stop=early_stop_pricing,
                              lambda_pricing=self.lambda_pricing,
                              lambda_factor=self.lambda_factor,
                              dual_pi=pi,
                              max_columns=self.num_columns_to_keep,
                              eliminated_customers=eliminated_nodes,
                              min_columns_early_stop=10)  # Generate at least 10 columns when early stopping
            pp_end = time.time()
            self.pp_time += (pp_end - pp_start)
            
            elapsed_total = time.time() - col_gen_start_time
            print(f"[Pricing] Generated {len(new_routes)} columns: {[r.cost for r in new_routes]} | PP time: {(pp_end - pp_start):.2f}s | Total elapsed: {elapsed_total:.2f}s")

            # Check if there are new negative cost paths
            if not new_routes:
                print("[-]No new negative cost paths found.")
                # Check model status
                if model.status == GRB.OPTIMAL:
                    elapsed_total = time.time() - col_gen_start_time
                    print(f"[-----ColumnGeneration-----]Iteration {iteration}: Objective = {model.objVal} | Total elapsed: {elapsed_total:.2f}s")
                elif model.status == GRB.INFEASIBLE:
                    elapsed_total = time.time() - col_gen_start_time
                    print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is infeasible. | Total elapsed: {elapsed_total:.2f}s")
                elif model.status == GRB.UNBOUNDED:
                    elapsed_total = time.time() - col_gen_start_time
                    print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is unbounded. | Total elapsed: {elapsed_total:.2f}s")
                else:
                    elapsed_total = time.time() - col_gen_start_time
                    print(
                        f"[-----ColumnGeneration-----]Iteration {iteration}: Model not solved. Status = {model.status} | Total elapsed: {elapsed_total:.2f}s")
                break

            # Add new routes to the pool, skipping duplicate paths.
            added_new_route = False
            for new_route in new_routes:
                cost = sum(self.paramsVRP.dist[new_route.path[i]][new_route.path[i + 1]] for i in range(len(new_route.path) - 1))
                new_route.set_cost(cost)
                if self._append_unique_route(new_route):
                    added_new_route = True

            if not added_new_route:
                print("[-]Pricing returned only duplicate routes; no new columns added.")
                break

            y, constraints = self._rebuild_rmp(model)

            # Increment iteration counter
            self.current_iteration += 1
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

        # Output final results
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
