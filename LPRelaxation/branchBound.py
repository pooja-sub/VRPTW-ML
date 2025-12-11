import gurobipy as gp
from gurobipy import GRB
from paramsVRP import ParamsVRP
from route import Route
from columnGen import ColumnGeneration
import numpy as np
import copy

class BranchAndBound:
    def __init__(self):
        self.lowerbound = -1e10
        self.upperbound = 1e10

    class TreeBB:
        def __init__(self, father=None, branch_from=-1, branch_to=-1, branch_value=-1):
            self.father = father
            self.son0 = None
            self.branch_from = branch_from
            self.branch_to = branch_to
            self.branch_value = branch_value
            self.lowest_value = -1e10
            self.toplevel = False

    def edges_based_on_branching(self, user_param, branching, recur):
        if branching.father is not None:  # Stop before root node
            if branching.branch_value == 0:  # Forbid this edge
                user_param.dist[branching.branch_from][branching.branch_to] = user_param.verybig
            else:  # Impose this edge
                if branching.branch_from != 0:  # Not from depot
                    user_param.dist[branching.branch_from][:] = user_param.verybig
                    user_param.dist[branching.branch_from][branching.branch_to] = user_param.dist_base[branching.branch_from][branching.branch_to]
                if branching.branch_to != user_param.nbclients + 1:  # Not to depot
                    user_param.dist[:, branching.branch_to] = user_param.verybig
                    user_param.dist[branching.branch_from][branching.branch_to] = user_param.dist_base[branching.branch_from][branching.branch_to]
                user_param.dist[branching.branch_to][branching.branch_from] = user_param.verybig  # Forbid reverse edge

            if recur:
                self.edges_based_on_branching(user_param, branching.father, recur)

    def bb_node(self, user_param, routes, branching, best_routes, depth):
        if not branching is None:
            print(f"[bb_node initiated] Depth = {depth} | routes = {routes}")
        # Check if we need to solve this node
        if (self.upperbound - self.lowerbound) / self.upperbound < user_param.gap:
            print(f'[bb_node terminated] GAP SATISFIED')
            return True

        # Initialize root node
        if branching is None:
            branching = self.TreeBB()
            branching.toplevel = True
            print(f"[ROOT node initiated] Depth = {depth} | routes = {routes}")

        # Display local info
        print(f"\nEdge from {branching.branch_from} to {branching.branch_to}: {'forbid' if branching.branch_value == 0 else 'set'}")
        #print(f"Memory: {gp.getMemUsage()} MB")

        # Compute solution using Column Generation
        column_gen = ColumnGeneration(user_param)
        cg_obj, routes = column_gen.compute_col_gen(routes)

        # Check feasibility
        if cg_obj > 2 * user_param.maxlength or cg_obj < -1e-6:
            print(f"RELAX INFEASIBLE | Lower bound: {self.lowerbound} | Upper bound: {self.upperbound} | Gap: {(self.upperbound - self.lowerbound) / self.upperbound} | Depth: {depth} | Routes: {len(routes)}")
            return True

        branching.lowest_value = cg_obj
        # LP-relaxation only: update lower bound and return current (possibly fractional) routes
        if branching.father and branching.father.son0 and branching.father.toplevel:
            self.lowerbound = min(branching.lowest_value, branching.father.son0.lowest_value)
            branching.toplevel = True
        elif branching.father is None:  # Root node
            self.lowerbound = cg_obj

        # Collect current routes (these may be fractional) and stop further branching
        best_routes.clear()
        for route in routes:
            if route.get_Q() > 1e-6:
                best_routes.append(copy.deepcopy(route))

        print(f"LP RELAXATION ONLY | Lower bound: {self.lowerbound} | Local CG cost: {cg_obj} | Routes: {len(routes)}")
        return True