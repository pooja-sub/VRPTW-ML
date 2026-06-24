import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gurobipy as gp
from gurobipy import GRB
from Common.paramsVRP import ParamsVRP
from Common.route import Route
from columnGen import ColumnGeneration
from expanded_pricing import ExpandedGraphPricing
import numpy as np
import copy

try:
    from pricing import build_reduced_graph
except Exception:
    try:
        from MLPricing.pricing import build_reduced_graph
    except Exception:
        build_reduced_graph = None

class BranchAndBound:
    def __init__(self, enable_column_deletion=True, deletion_threshold=30, enable_early_stop=False, gap_threshold=None,
                 enable_node_elimination=False, node_elimination_threshold=1.5, min_nonbasic_columns=100,
                 lambda_factor=2.0, time_limit=None,
                 use_expanded_pricing=True, expanded_max_m=None, expanded_max_routes=20,
                 reduced_graph_min_columns=5):
        self.lowerbound = -1e10
        self.upperbound = 1e10
        self.bb_node_count = 0  # Track number of BB nodes processed
        self.gap_threshold = gap_threshold  # Optional override for stopping criterion
        
        # Acceleration technique flags
        self.enable_column_deletion = enable_column_deletion
        self.deletion_threshold = deletion_threshold
        self.enable_early_stop = enable_early_stop
        self.enable_node_elimination = enable_node_elimination
        self.node_elimination_threshold = node_elimination_threshold
        self.min_nonbasic_columns = min_nonbasic_columns
        self.lambda_factor = lambda_factor
        self.use_expanded_pricing = use_expanded_pricing
        self.expanded_max_m = expanded_max_m
        self.expanded_max_routes = expanded_max_routes
        self.reduced_graph_min_columns = max(0, int(reduced_graph_min_columns))
        self.shared_expanded_pricing_full = None
        self.shared_expanded_pricing_reduced = None
        self.shared_expanded_pricing_key = None
        
        # Timing statistics and limit
        self.total_rmp_time = 0.0  # Cumulative RMP time across all nodes
        self.total_pp_time = 0.0   # Cumulative PP time across all nodes
        self.total_cg_time = 0.0   # Cumulative CG wall time across all nodes
        self.root_rmp_time = 0.0
        self.root_pp_time = 0.0
        self.root_cg_time = 0.0
        self.root_cg_iterations = 0
        self.time_limit = time_limit  # Optional time limit in seconds
        self.start_time = None  # Initialized when bb_node is called

    def _get_shared_expanded_pricing_pair(self, user_param):
        """Build full/reduced expanded-graph caches once and reuse across BnP nodes."""
        if not self.use_expanded_pricing:
            return None, None

        cache_key = (
            getattr(user_param, "instance_path", ""),
            int(self.expanded_max_m) if self.expanded_max_m is not None else None,
        )
        if self.shared_expanded_pricing_key == cache_key and self.shared_expanded_pricing_full is not None:
            return self.shared_expanded_pricing_reduced, self.shared_expanded_pricing_full

        base_param = copy.deepcopy(user_param)
        base_param.dist = copy.deepcopy(user_param.dist_base)

        reduced_param = None
        if build_reduced_graph is not None:
            try:
                allowed_arcs = build_reduced_graph(base_param)
            except Exception as exc:
                print(f"[MLPricing] Reduced expanded graph unavailable: {exc}")
                allowed_arcs = set()

            if allowed_arcs:
                reduced_param = copy.deepcopy(base_param)
                verybig = reduced_param.verybig
                node_count = reduced_param.nbclients + 1
                for i in range(node_count):
                    for j in range(node_count):
                        if (i, j) not in allowed_arcs:
                            reduced_param.dist[i][j] = verybig
                print(f"[MLPricing] Built reduced base graph for expanded pricing with {len(allowed_arcs)} allowed arcs")

        print("[MLPricing] Building shared FULL expanded-graph cache")
        self.shared_expanded_pricing_full = ExpandedGraphPricing(
            user_param=base_param,
            max_m=self.expanded_max_m,
            quiet_graph_logs=True,
            quiet_spprc_logs=True,
        )

        self.shared_expanded_pricing_reduced = None
        if reduced_param is not None:
            print("[MLPricing] Building shared REDUCED expanded-graph cache")
            self.shared_expanded_pricing_reduced = ExpandedGraphPricing(
                user_param=reduced_param,
                max_m=self.expanded_max_m,
                quiet_graph_logs=True,
                quiet_spprc_logs=True,
            )

        self.shared_expanded_pricing_key = cache_key
        return self.shared_expanded_pricing_reduced, self.shared_expanded_pricing_full

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

    # def _deduplicate_routes_by_path(self, routes):
    #     """
    #     Keep only one copy of each route path.
    #     """
    #     unique_routes = []
    #     seen_paths = set()
    #     for route in routes:
    #         route_key = route.path_key()
    #         if route_key in seen_paths:
    #             continue
    #         seen_paths.add(route_key)
    #         unique_routes.append(route)
    #     return unique_routes

    def edges_based_on_branching(self, user_param, branching, recur):
        if branching.father is not None:  # Stop before root node
            # Validate edge indices
            if branching.branch_from == -1 or branching.branch_to == -1:
                print(f"  [ERROR] Invalid branching edge ({branching.branch_from}, {branching.branch_to})")
                return
            
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

    # def vehicle_count_branch(self, routes):
    #     """
    #     Branching rule on total number of vehicles (Desrochers, Desrosiers, Solomon 1992).
    #
    #     Returns bounds (floor, ceil) of the fractional vehicle count if non-integer.
    #     Caller can pass these as vehicle_lower_bound / vehicle_upper_bound to compute_col_gen.
    #     """
    #     total = sum(route.get_Q() for route in routes)
    #     if total < 0:
    #         return None
    #     frac = abs(total - round(total))
    #     if frac < 1e-6:
    #         return None
    #     lower = int(np.floor(total))
    #     upper = int(np.ceil(total))
    #     return lower, upper

    def bb_node(self, user_param, routes, branching, best_routes, depth):
        import time
        
        # Increment BB node counter
        self.bb_node_count += 1
        
        # Initialize start time on first call (root node)
        if self.start_time is None and branching is None:
            self.start_time = time.time()
        
        # Check time limit
        if self.time_limit is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit:
                print(f'[TIME LIMIT EXCEEDED] Elapsed: {elapsed:.2f}s > Limit: {self.time_limit:.2f}s')
                return True
        
        if not branching is None:
            print(f"[bb_node initiated] Node={self.bb_node_count} | Depth = {depth} | routes = {routes}")
        # Check if we need to solve this node
        gap_limit = self.gap_threshold if self.gap_threshold is not None else user_param.gap
        if (self.upperbound - self.lowerbound) / self.upperbound < gap_limit:
            print(f'[bb_node terminated] GAP SATISFIED (threshold={gap_limit})')
            return True

        # Initialize root node
        if branching is None:
            branching = self.TreeBB()
            branching.toplevel = True
            print(f"[ROOT node initiated] Node={self.bb_node_count} | Depth = {depth} | routes = {routes}")

        # Display local info
        print(f"\nEdge from {branching.branch_from} to {branching.branch_to}: {'forbid' if branching.branch_value == 0 else 'set'}")

        # Compute solution using the local ColumnGeneration implementation.
        # The MLPricing version currently exposes a simplified API.
        reduced_backend, full_backend = self._get_shared_expanded_pricing_pair(user_param)
        column_gen = ColumnGeneration(
            user_param,
            use_expanded_pricing=self.use_expanded_pricing,
            expanded_max_m=self.expanded_max_m,
            expanded_max_routes=self.expanded_max_routes,
            shared_expanded_pricing_reduced=reduced_backend,
            shared_expanded_pricing_full=full_backend,
            reduced_graph_min_columns=self.reduced_graph_min_columns,
        )
        cg_obj, routes, cg_stats = column_gen.compute_col_gen(routes, return_stats=True)
        self.total_rmp_time += float(cg_stats.get("rmp_total", 0.0))
        self.total_pp_time += float(cg_stats.get("pp_total", 0.0))
        self.total_cg_time += float(cg_stats.get("total_time", 0.0))
        if depth == 0:
            self.root_rmp_time = float(cg_stats.get("rmp_total", 0.0))
            self.root_pp_time = float(cg_stats.get("pp_total", 0.0))
            self.root_cg_time = float(cg_stats.get("total_time", 0.0))
            self.root_cg_iterations = int(cg_stats.get("iterations", 0))

        # If the RMP at this node did not solve to optimality (e.g., infeasible), prune this node
        rmp_status = cg_stats.get("rmp_final_status", GRB.OPTIMAL)
        if rmp_status != GRB.OPTIMAL:
            print(f"[bb_node] Pruning node due to non-optimal RMP status: {rmp_status}")
            # Mark this node as pruned/unusable so parent won't take its lowest_value
            branching.lowest_value = float('inf')
            return True

        # # Check feasibility
        # if cg_obj > 2 * user_param.maxlength or cg_obj < -1e-6:
        #     print(f"RELAX INFEASIBLE | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Routes: {len(routes)}")
        #     return True

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

        # Hierarchical branching: vehicles first, then edges (DISABLED FOR NOW)
        # vehicle_bounds = self.vehicle_count_branch(routes)
        # if vehicle_bounds:
        #     floor_v, ceil_v = vehicle_bounds
        #     parent_lower = branching.vehicle_lower_bound
        #     parent_upper = branching.vehicle_upper_bound
        #
        #     # Branch 1: total vehicles <= floor_v
        #     child1_lower = parent_lower
        #     child1_upper = floor_v if parent_upper is None else min(parent_upper, floor_v)
        #
        #     # Branch 2: total vehicles >= ceil_v
        #     child2_lower = ceil_v if parent_lower is None else max(parent_lower, ceil_v)
        #     child2_upper = parent_upper
        #
        #     print(f"VEHICLE BRANCH | total={sum(r.get_Q() for r in routes):.4f} | floor={floor_v} | ceil={ceil_v} | Depth: {depth}")
        #
        #     # Explore child with upper bound (<= floor)
        #     if child1_lower is None or child1_upper is None or child1_lower <= child1_upper:
        #         newnode1 = self.TreeBB(branching, -1, -1, -1, child1_lower, child1_upper)
        #         if not self.bb_node(user_param, routes, newnode1, best_routes, depth + 1):
        #             return False
        #         branching.son0 = newnode1
        #     else:
        #         print(f"  [Pruned] Infeasible vehicle upper bound: {child1_lower}>{child1_upper}")
        #
        #     # Explore child with lower bound (>= ceil)
        #     if child2_lower is None or child2_upper is None or child2_lower <= child2_upper:
        #         newnode2 = self.TreeBB(branching, -1, -1, -1, child2_lower, child2_upper)
        #         if not self.bb_node(user_param, routes, newnode2, best_routes, depth + 1):
        #             return False
        #         if branching.son0 is None:
        #             branching.son0 = newnode2
        #         branching.lowest_value = min(branching.lowest_value, newnode2.lowest_value)
        #     else:
        #         print(f"  [Pruned] Infeasible vehicle lower bound: {child2_lower}>{child2_upper}")
        #
        #     if branching.son0 is not None and branching.lowest_value > branching.son0.lowest_value:
        #         branching.lowest_value = branching.son0.lowest_value
        #     return True

        # Check integer feasibility and find branching variable on edges
        feasible = True
        best_edge = (-1, -1)
        best_obj = -1.0
        best_val = 0

        # Convert path variables to edge variables
        user_param.edges = np.zeros((user_param.nbclients + 1, user_param.nbclients + 1))
        for route in routes:
            if route.get_Q() > 1e-6:
                path = route.get_path()
                prevcity = 0
                for city in path[1:]:
                    user_param.edges[prevcity][city] += route.get_Q()
                    prevcity = city

        # Find fractional edge
        fractional_edges_count = 0
        for i in range(user_param.nbclients + 1):
            for j in range(user_param.nbclients + 1):
                coef = user_param.edges[i][j]
                if coef > 1e-6 and (coef < 0.9999999999 or coef > 1.0000000001):
                    fractional_edges_count += 1
                    # Skip eliminated edges
                    if user_param.dist[i][j] >= user_param.verybig - 1e-6:
                        print(f"  [Debug] Skipping eliminated fractional edge ({i}, {j}) with coef={coef:.4f}")
                        continue
                    feasible = False
                    change = min(coef, abs(1.0 - coef)) * routes[i].get_cost()  # Approximate cost impact of branching on this edge
                    if change > best_obj:
                        best_edge = (i, j)
                        best_obj = change
                        best_val = 0 if abs(1.0 - coef) > coef else 1
        
        if fractional_edges_count > 0:
            print(f"  [Debug] Found {fractional_edges_count} fractional edges total")

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

        # Check if a valid fractional edge was found
        if best_edge == (-1, -1):
            print(f"  [ERROR] No fractional edge found for branching, but solution is not integer feasible!")
            return False

        # Branching
        print(f"  [Branching] On edge {best_edge}, value: {best_val}, objective change: {best_obj:.4f}")
        newnode1 = self.TreeBB(branching, best_edge[0], best_edge[1], best_val,
                       branching.vehicle_lower_bound, branching.vehicle_upper_bound)
        # Reset dist and apply all ancestor constraints (including this node)
        user_param.dist = copy.deepcopy(user_param.dist_base)
        self.edges_based_on_branching(user_param, newnode1, True)
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