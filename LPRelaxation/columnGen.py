import gurobipy as gp
from gurobipy import GRB
from paramsVRP import ParamsVRP
from route import Route
from SPPRC import SPPRC
import numpy as np
import os
import csv

class ColumnGeneration:
    def __init__(self, user_param):
        self.paramsVRP = user_param
        self.routes = []
        self._feature_file = None

    def compute_col_gen(self, initial_routes):
        """
        Execute column generation algorithm.

        :param initial_routes: Initial list of routes
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

        # Create variables and objective function
        y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

        # Add constraints: each customer must be served exactly once
        constraints = model.addConstrs(
            (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
             for client in range(1, self.paramsVRP.nbclients - 1)),
            "ClientService"
        )

        model.update()
        #print(constraints)

        # Set objective function
        model.setObjective(gp.quicksum(y[i] * self.routes[i].cost for i in range(len(self.routes))), GRB.MINIMIZE)

        # Column generation main loop
        iteration = 0
        while True:
            # Solve the current model
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

            # Update SPPRC cost matrix
            for i in range(1, self.paramsVRP.nbclients - 1):
                for j in range(self.paramsVRP.nbclients):
                    self.paramsVRP.cost[i][j] = self.paramsVRP.dist[i][j] - pi[i - 1]
                    if self.paramsVRP.cost[i][j] < 0:
                        #print(f"Negative cost found: {self.paramsVRP.cost[i][j]} at {i}, {j}")
                        pass


            # Solve SPPRC to get new columns
            sp = SPPRC(self.paramsVRP)
            new_routes = []
            sp.shortestPath(self.paramsVRP, new_routes, self.paramsVRP.nbclients - 2)
            print(new_routes)

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
                # Mark arcs present in this new route in the dataset feature CSV (if available)
                try:
                    # find feature file lazily
                    if self._feature_file is None:
                        self._feature_file = self._find_feature_file()
                    if self._feature_file:
                        arcs = []
                        prevcity = None
                        for city in new_route.path:
                            if prevcity is not None:
                                arcs.append((prevcity, city))
                            prevcity = city
                        # filter arcs that involve depot (0) or the end-depot (nbclients-1)
                        nb_last = self.paramsVRP.nbclients - 1
                        arcs_to_mark = set()
                        for a, b in arcs:
                            if a == 0 or b == 0 or a == nb_last or b == nb_last:
                                continue
                            arcs_to_mark.add((int(a), int(b)))
                        if arcs_to_mark:
                            self._mark_arcs_in_feature(self._feature_file, arcs_to_mark)
                except Exception as e:
                    print(f"Warning: failed to update feature file: {e}")

                # Get all variables in the model
                vars_to_remove = model.getVars()
                for var in vars_to_remove:
                    model.remove(var)
                constrs_to_remove = model.getConstrs()
                for constr in constrs_to_remove:
                    model.remove(constr)


                # Create variables and objective function
                y = model.addVars(len(self.routes), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

                # Add constraints: each customer must be served exactly once
                constraints = model.addConstrs(
                    (gp.quicksum(y[i] for i, route in enumerate(self.routes) if client in route.path[1:-1]) >= 1
                     for client in range(1, self.paramsVRP.nbclients - 1)),
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

        # Output final results
        for i, route in enumerate(self.routes):
            route.set_Q(y[i].x)
            if route.Q > 0:
                print(f"Route {i}: Cost = {route.cost}, Q = {route.Q}, Path = {route.path}")

        return model.objVal, self.routes

    def _find_feature_file(self):
        """Try to locate an arc-features CSV for the current dataset inside DatasetFeatures/.
        Returns full path or None if not found.
        """
        try:
            root = os.path.dirname(os.path.dirname(__file__))
            df_dir = os.path.join(root, 'DatasetFeatures','50-customer-instances')
            if not os.path.isdir(df_dir):
                return None
            name = (self.paramsVRP.datasetName or '').lower()
            for fname in os.listdir(df_dir):
                lf = fname.lower()
                if 'arc' in lf and 'feature' in lf and lf.endswith('.csv') and name in lf:
                    return os.path.join(df_dir, fname)
                # fallback: match dataset name and 'arc' in filename
                if 'arc' in lf and name in lf and lf.endswith('.csv'):
                    return os.path.join(df_dir, fname)
            return None
        except Exception:
            return None

    def _mark_arcs_in_feature(self, feature_file, arcs_to_mark):
        """Mark arcs in the CSV by setting/creating a 'label' column to '1' for arcs in arcs_to_mark.

        arcs_to_mark: set of (from,to) tuples (integers)
        """
        try:
            tmp_file = feature_file + '.tmp'
            with open(feature_file, 'r', newline='', encoding='utf-8') as rf:
                reader = csv.DictReader(rf)
                fieldnames = reader.fieldnames[:] if reader.fieldnames else []
                if 'label' not in fieldnames:
                    fieldnames = fieldnames + ['label']

                rows = []
                for row in reader:
                    try:
                        f = int(row.get('from', row.get('From', '')).strip()) if row.get('from', None) else None
                        t = int(row.get('to', row.get('To', '')).strip()) if row.get('to', None) else None
                    except Exception:
                        f = None
                        t = None
                    if f is not None and t is not None and (f, t) in arcs_to_mark:
                        row['label'] = '1'
                    else:
                        # ensure label exists (keep existing or set 0)
                        if 'label' not in row or row.get('label', '') == '':
                            row['label'] = row.get('label', '0')
                    rows.append(row)

            # write back
            with open(tmp_file, 'w', newline='', encoding='utf-8') as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            # replace original
            os.replace(tmp_file, feature_file)
        except Exception as e:
            print(f"Error updating feature file {feature_file}: {e}")

        '''
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
        except Exception as e:
            print(f"Error in compute_col_gen: {e}")
        '''
