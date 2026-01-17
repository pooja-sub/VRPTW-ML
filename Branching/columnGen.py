import gurobipy as gp
from gurobipy import GRB
from paramsVRP import ParamsVRP
from route import Route
from SPPRC import SPPRC
import numpy as np

class ColumnGeneration:
    def __init__(self, user_param, enable_column_deletion=True, deletion_threshold=20, min_nonbasic_columns=10,
                 lambda_pricing=True, lambda_factor=2.0, num_columns_to_keep=10):
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
        model.optimize()
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
                self.route_last_basis_iteration[i] = self.current_iteration
            
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
            self.routes.append(route)

        # Create variables and objective function
        y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

        # Add constraints: each customer must be served once
        constraints = model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
             for client in range(1, self.paramsVRP.nbclients)),
            "ClientService"
        )

        # Optional vehicle count bounds (for vehicle-count branching)
        vehicle_constr_lower = None
        vehicle_constr_upper = None
        if vehicle_lower_bound is not None:
            vehicle_constr_lower = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) >= vehicle_lower_bound,
                                                   name="VehicleCountLower")
        if vehicle_upper_bound is not None:
            vehicle_constr_upper = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) <= vehicle_upper_bound,
                                                   name="VehicleCountUpper")

        model.update()
        #print(constraints)

        # Set objective function
        model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)

        # Column generation main loop
        iteration = 0
        while True:
            # Solve current model
            model.optimize()
            if model.status == GRB.OPTIMAL:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Objective = {model.objVal}")
            elif model.status == GRB.INFEASIBLE:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is infeasible.")
            elif model.status == GRB.UNBOUNDED:
                print(f"[-----ColumnGeneration-----]Iteration {iteration}: Model is unbounded.")
            else:
                print(
                    f"[-----ColumnGeneration-----]Iteration {iteration}: Model not solved. Status = {model.status}")

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
                    
                    # Update route_last_basis_node with new indices
                    new_route_last_basis_node = {}
                    for old_idx, node in self.route_last_basis_node.items():
                        if old_idx in old_to_new:
                            new_route_last_basis_node[old_to_new[old_idx]] = node
                    self.route_last_basis_node = new_route_last_basis_node
                    
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
                    if vehicle_lower_bound is not None:
                        vehicle_constr_lower = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) >= vehicle_lower_bound,
                                                               name="VehicleCountLower")
                    if vehicle_upper_bound is not None:
                        vehicle_constr_upper = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) <= vehicle_upper_bound,
                                                               name="VehicleCountUpper")
                    model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)
                    model.update()
                    continue
            
            # Apply variable fixing at regular intervals
            if iteration > 0 and iteration % fix_interval == 0:
                print(f"[Iteration {iteration}] Applying variable fixing by reduced cost...")
                fixed_count = self.fix_variables_by_reduced_cost(model, y, lower_bound, upper_bound, constraints)
                
                # If routes were fixed, rebuild the model
                if fixed_count > 0:
                    # Get active routes (not fixed)
                    active_routes = [route for i, route in enumerate(self.routes) if i not in self.fixed_routes]
                    
                    if len(active_routes) == 0:
                        print("[ERROR] All routes have been fixed. Cannot continue.")
                        break
                    
                    # Rebuild model with only active routes
                    self.routes = active_routes
                    self.fixed_routes.clear()  # Reset indices after rebuilding
                    
                    # Remove all variables and constraints
                    vars_to_remove = model.getVars()
                    for var in vars_to_remove:
                        model.remove(var)
                    constrs_to_remove = model.getConstrs()
                    for constr in constrs_to_remove:
                        model.remove(constr)
                    
                    # Recreate with active routes only
                    y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)
                    constraints = model.addConstrs(
                        (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
                         for client in range(1, self.paramsVRP.nbclients)),
                        "ClientService"
                    )
                    if vehicle_lower_bound is not None:
                        vehicle_constr_lower = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) >= vehicle_lower_bound,
                                                               name="VehicleCountLower")
                    if vehicle_upper_bound is not None:
                        vehicle_constr_upper = model.addConstr(gp.quicksum(y[i] for i in range(len(self.routes))) <= vehicle_upper_bound,
                                                               name="VehicleCountUpper")
                    model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)
                    model.update()
                    
                    # Re-optimize after fixing
                    continue

            # Update SPPRC cost matrix
            for i in range(1, self.paramsVRP.nbclients):
                for j in range(self.paramsVRP.nbclients + 2):
                    self.paramsVRP.cost[i][j] = self.paramsVRP.dist[i][j] - pi[i - 1]
                    if self.paramsVRP.cost[i][j] < 0:
                        #print(f"Negative cost found: {self.paramsVRP.cost[i][j]} at {i}, {j}")
                        pass


            # Solve SPPRC to get new columns
            # Determine if this is the final pricing (proving optimality)
            is_final_pricing = (iteration > 0)  # After first iteration, we're refining
            
            sp = SPPRC(self.paramsVRP)
            new_routes = []
            # Enable early stopping except when we need to prove optimality
            print(f"[ColumnGen] Calling SPPRC: early_stop_pricing={early_stop_pricing}, lambda_pricing={self.lambda_pricing}")
            sp.shortestPath(self.paramsVRP, new_routes, self.paramsVRP.nbclients - 1, 
                          early_stop=early_stop_pricing,
                          lambda_pricing=self.lambda_pricing,
                          lambda_factor=self.lambda_factor,
                          dual_pi=pi,
                          max_columns=self.num_columns_to_keep)
            print(f"[Pricing] Generated {len(new_routes)} columns: {[r.cost for r in new_routes]}")

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

            # Add new routes to the model
            for new_route in new_routes:
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
