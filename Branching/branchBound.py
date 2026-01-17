import gurobipy as gp
from gurobipy import GRB
from paramsVRP import ParamsVRP
from route import Route
from columnGen import ColumnGeneration
import numpy as np
import copy

class BranchAndBound:
    def __init__(self, enable_column_deletion=True, deletion_threshold=30, enable_early_stop=False):
        self.lowerbound = -1e10
        self.upperbound = 1e10
        self.bb_node_count = 0  # Track number of BB nodes processed
        
        # Acceleration technique flags
        self.enable_column_deletion = enable_column_deletion
        self.deletion_threshold = deletion_threshold
        self.enable_early_stop = enable_early_stop

    class TreeBB:
        def __init__(self, father=None, branch_from=-1, branch_to=-1, branch_value=-1,
                     vehicle_lower_bound=None, vehicle_upper_bound=None):
            self.father = father
            self.son0 = None
            self.branch_from = branch_from
            self.branch_to = branch_to
            self.branch_value = branch_value
            self.lowest_value = -1e10
            self.toplevel = False
            self.vehicle_lower_bound = vehicle_lower_bound
            self.vehicle_upper_bound = vehicle_upper_bound

    def edges_based_on_branching(self, user_param, branching, recur):
        if branching.father is not None:  # Stop before root node
            if branching.branch_value == 0:  # Forbid this edge
                user_param.dist[branching.branch_from][branching.branch_to] = user_param.verybig
            else:  # Impose this edge
                if branching.branch_from != 0:  # Not from depot
                    user_param.dist[branching.branch_from][:] = user_param.verybig
                    user_param.dist[branching.branch_from][branching.branch_to] = user_param.dist_base[branching.branch_from][branching.branch_to]
                if branching.branch_to != user_param.nbclients:  # Not to depot
                    user_param.dist[:, branching.branch_to] = user_param.verybig
                    user_param.dist[branching.branch_from][branching.branch_to] = user_param.dist_base[branching.branch_from][branching.branch_to]
                user_param.dist[branching.branch_to][branching.branch_from] = user_param.verybig  # Forbid reverse edge

            if recur:
                self.edges_based_on_branching(user_param, branching.father, recur)

    def vehicle_count_branch(self, routes):
        """
        Branching rule on total number of vehicles (Desrochers, Desrosiers, Solomon 1992).

        Returns bounds (floor, ceil) of the fractional vehicle count if non-integer.
        Caller can pass these as vehicle_lower_bound / vehicle_upper_bound to compute_col_gen.
        """
        total = sum(route.get_Q() for route in routes)
        if total < 0:
            return None
        frac = abs(total - round(total))
        if frac < 1e-6:
            return None
        lower = int(np.floor(total))
        upper = int(np.ceil(total))
        return lower, upper

    def bb_node(self, user_param, routes, branching, best_routes, depth):
        # Increment BB node counter
        self.bb_node_count += 1
        
        if not branching is None:
            print(f"[bb_node initiated] Node={self.bb_node_count} | Depth = {depth} | routes = {routes}")
        # Check if we need to solve this node
        if (self.upperbound - self.lowerbound) / self.upperbound < user_param.gap:
            print(f'[bb_node terminated] GAP SATISFIED')
            return True

        # Initialize root node
        if branching is None:
            branching = self.TreeBB()
            branching.toplevel = True
            print(f"[ROOT node initiated] Node={self.bb_node_count} | Depth = {depth} | routes = {routes}")

        # Display local info
        print(f"\nEdge from {branching.branch_from} to {branching.branch_to}: {'forbid' if branching.branch_value == 0 else 'set'}")
        #print(f"Memory: {gp.getMemUsage()} MB")

        # Compute solution using Column Generation (pass bounds for variable fixing)
        column_gen = ColumnGeneration(user_param, 
                         enable_column_deletion=self.enable_column_deletion,
                         deletion_threshold=self.deletion_threshold,
                         lambda_pricing=False,
                         num_columns_to_keep=30)
        cg_obj, routes = column_gen.compute_col_gen(routes, self.lowerbound, self.upperbound, 
                                fix_interval=10, bb_node_count=self.bb_node_count,
                                early_stop_pricing=self.enable_early_stop,
                                vehicle_lower_bound=branching.vehicle_lower_bound,
                                vehicle_upper_bound=branching.vehicle_upper_bound)

        # Check feasibility
        if cg_obj > 2 * user_param.maxlength or cg_obj < -1e-6:
            print(f"RELAX INFEASIBLE | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Routes: {len(routes)}")
            return True

        branching.lowest_value = cg_obj

        # Update global lower bound
        if branching.father and branching.father.son0 and branching.father.toplevel:
            self.lowerbound = min(branching.lowest_value, branching.father.son0.lowest_value)
            branching.toplevel = True
        elif branching.father is None:  # Root node
            self.lowerbound = cg_obj

        if branching.lowest_value > self.upperbound:
            print(f"CUT | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Local CG cost: {cg_obj} | Routes: {len(routes)}")
            return True

        # Hierarchical branching: vehicles first, then edges
        vehicle_bounds = self.vehicle_count_branch(routes)
        if vehicle_bounds:
            floor_v, ceil_v = vehicle_bounds
            parent_lower = branching.vehicle_lower_bound
            parent_upper = branching.vehicle_upper_bound

            # Branch 1: total vehicles <= floor_v
            child1_lower = parent_lower
            child1_upper = floor_v if parent_upper is None else min(parent_upper, floor_v)

            # Branch 2: total vehicles >= ceil_v
            child2_lower = ceil_v if parent_lower is None else max(parent_lower, ceil_v)
            child2_upper = parent_upper

            print(f"VEHICLE BRANCH | total={sum(r.get_Q() for r in routes):.4f} | floor={floor_v} | ceil={ceil_v} | Depth: {depth}")

            # Explore child with upper bound (<= floor)
            if child1_lower is None or child1_upper is None or child1_lower <= child1_upper:
                newnode1 = self.TreeBB(branching, -1, -1, -1, child1_lower, child1_upper)
                if not self.bb_node(user_param, routes, newnode1, best_routes, depth + 1):
                    return False
                branching.son0 = newnode1
            else:
                print(f"  [Pruned] Infeasible vehicle upper bound: {child1_lower}>{child1_upper}")

            # Explore child with lower bound (>= ceil)
            if child2_lower is None or child2_upper is None or child2_lower <= child2_upper:
                newnode2 = self.TreeBB(branching, -1, -1, -1, child2_lower, child2_upper)
                if not self.bb_node(user_param, routes, newnode2, best_routes, depth + 1):
                    return False
                if branching.son0 is None:
                    branching.son0 = newnode2
                branching.lowest_value = min(branching.lowest_value, newnode2.lowest_value)
            else:
                print(f"  [Pruned] Infeasible vehicle lower bound: {child2_lower}>{child2_upper}")

            if branching.son0 is not None and branching.lowest_value > branching.son0.lowest_value:
                branching.lowest_value = branching.son0.lowest_value
            return True

        # Check integer feasibility and find branching variable on edges
        feasible = True
        best_edge = (-1, -1)
        best_obj = -1.0
        best_val = 0

        # Convert path variables to edge variables
        user_param.edges = np.zeros((user_param.nbclients + 2, user_param.nbclients + 2))
        for route in routes:
            if route.get_Q() > 1e-6:
                path = route.get_path()
                prevcity = 0
                for city in path[1:]:
                    user_param.edges[prevcity][city] += route.get_Q()
                    prevcity = city

        # Find fractional edge
        for i in range(user_param.nbclients + 2):
            for j in range(user_param.nbclients + 2):
                coef = user_param.edges[i][j]
                if coef > 1e-6 and (coef < 0.9999999999 or coef > 1.0000000001):
                    feasible = False
                    change = min(coef, abs(1.0 - coef)) * routes[i].get_cost()
                    if change > best_obj:
                        best_edge = (i, j)
                        best_obj = change
                        best_val = 0 if abs(1.0 - coef) > coef else 1

        if feasible:
            if branching.lowest_value < self.upperbound:
                self.upperbound = branching.lowest_value
                best_routes.clear()
                for route in routes:
                    if route.get_Q() > 1e-6:
                        best_routes.append(copy.deepcopy(route))
                print(f"OPT | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Local CG cost: {cg_obj} | Routes: {len(routes)}")
            else:
                print(f"FEAS | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Local CG cost: {cg_obj} | Routes: {len(routes)}")
            return True
        else:
            print(f"INTEG INFEAS | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Local CG cost: {cg_obj} | Routes: {len(routes)}")

        # Branching
        newnode1 = self.TreeBB(branching, best_edge[0], best_edge[1], best_val,
                       branching.vehicle_lower_bound, branching.vehicle_upper_bound)
        self.edges_based_on_branching(user_param, newnode1, False)
        node_routes1 = [route for route in routes if best_edge not in zip(route.get_path()[:-1], route.get_path()[1:])]
        if not self.bb_node(user_param, node_routes1, newnode1, best_routes, depth + 1):
            return False

        branching.son0 = newnode1

        newnode2 = self.TreeBB(branching, best_edge[0], best_edge[1], 1 - best_val,
                       branching.vehicle_lower_bound, branching.vehicle_upper_bound)
        user_param.dist = copy.deepcopy(user_param.dist_base)
        self.edges_based_on_branching(user_param, newnode2, True)
        node_routes2 = [route for route in routes if all(user_param.dist[prevcity][city] < user_param.verybig - 1e-6 for prevcity, city in zip(route.get_path()[:-1], route.get_path()[1:]))]
        if not self.bb_node(user_param, node_routes2, newnode2, best_routes, depth + 1):
            return False

        branching.lowest_value = min(newnode1.lowest_value, newnode2.lowest_value)
        return True