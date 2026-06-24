import heapq
from functools import total_ordering
from sortedcontainers import SortedSet
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from route import Route
from paramsVRP import ParamsVRP

'''
 Shortest Path with Resource Constraints (SPPRC) - NON-ELEMENTARY VERSION
 
 This version does NOT enforce elementarity (allows visiting customers multiple times).
 Useful for comparison and performance analysis against ESPPRC.
 
 Differences from ESPPRC:
 1. No vertex_visited tracking (allows cycles)
 2. Simpler label state: (city, cost, ttime, demand)
 3. Simpler dominance: pure Pareto dominance on (cost, ttime, demand)
 4. Much smaller state space (no combinatorial explosion from visited subsets)
 
 Structure:
 labels: List of all labels created
 U: Unprocessed labels (priority queue)
 P: Processed labels ending at depot with negative cost
 city2labels: For each city, list of label indices at that city
'''


class SPPRC:

    def __init__(self, userParam=None):
        self.paramsVRP = ParamsVRP() if userParam is None else userParam
        self.labels = []

    @total_ordering
    class label:
        def __init__(self, city, index_prev_label, cost, ttime, demand, dominated, parent):
            self.city = city  # int
            self.index_prev_label = index_prev_label  # int
            self.cost = cost  # double
            self.ttime = ttime  # float
            self.demand = demand  # double
            self.dominated = dominated  # boolean
            self.parent = parent

        def __lt__(self, other):
            """Ordering for SortedSet: compare by cost first, then ttime, then demand."""
            if self.cost - other.cost < -1e-7:
                return True
            elif self.cost - other.cost > 1e-7:
                return False
            else:
                if self.city == other.city:
                    if self.ttime - other.ttime < -1e-7:
                        return True
                    elif self.ttime - other.ttime > 1e-7:
                        return False
                    else:
                        if self.demand - other.demand < -1e-7:
                            return True
                        elif self.demand - other.demand > 1e-7:
                            return False
                        else:
                            return False
                elif self.city > other.city:
                    return False
                else:
                    return True

        def __eq__(self, other):
            """Equality: exact match on all attributes."""
            if abs(self.cost - other.cost) > 1e-7:
                return False
            if self.city != other.city:
                return False
            if abs(self.ttime - other.ttime) > 1e-7:
                return False
            if abs(self.demand - other.demand) > 1e-7:
                return False
            return True

    def shortestPath(self, userParamArg, routes, nbroute, early_stop=True, 
                     lambda_pricing=True, lambda_factor=2.0, dual_pi=None, max_columns=None,
                     eliminated_customers=None, min_columns_early_stop=10):
        '''
        Label-setting algorithm for SPPRC
        Allows revisiting customers; no cycle prevention.
        
        Simpler state space than ESPPRC: only (city, cost, ttime, demand).
        
        :param early_stop: If True, stop after finding min_columns_early_stop negative cost routes
        :param lambda_pricing: If True, apply lambda pricing rule (ratio-based)
        :param lambda_factor: Multiplier for lambda pricing rule (default 2.0)
        :param dual_pi: List of dual values π for customers
        :param max_columns: Maximum number of columns to keep
        :param eliminated_customers: Set of customer IDs (1-indexed) to eliminate
        :param min_columns_early_stop: Minimum columns before early stopping (default 10)
        '''
        if eliminated_customers is None:
            eliminated_customers = set()
        
        print(f"[---SPPRC.shortestPath called---] early_stop={early_stop}, lambda_pricing={lambda_pricing}, min_columns_early_stop={min_columns_early_stop}")
        self.paramsVRP = userParamArg
        node_count = self.paramsVRP.dist.shape[0]
        if max_columns is None:
            max_columns = 2 * nbroute

        # Initialize unprocessed (U) and processed (P) label sets
        U = SortedSet(key=lambda x: x)
        P = SortedSet(key=lambda x: x)

        # Initialize labels array and starting label at depot
        self.labels = []
        self.labels.append(self.label(0, -1, 0.0, 0, 0, False, self))  # Depot label
        U.add(0)

        # For each city, track indices of labels at that city (for dominance)
        checkDom = [0] * node_count
        city2labels = [[] for _ in range(node_count)]
        city2labels[0].append(0)

        nbsol = 0
        maxsol = max_columns
        min_cost = None  # Track minimum cost for lambda pricing

        while U and nbsol < maxsol:
            current_idx = U.pop(0)  # Process label with lowest cost
            current = self.labels[current_idx]

            # Dominance check: eliminate dominated labels at current city
            cleaning = []
            for i in range(checkDom[current.city], len(city2labels[current.city])):
                for j in range(i):
                    l1, l2 = city2labels[current.city][i], city2labels[current.city][j]
                    la1, la2 = self.labels[l1], self.labels[l2]
                    
                    if not la1.dominated and not la2.dominated and l1 != l2:
                        # Check if la2 is dominated by la1
                        if la1.cost <= la2.cost and la1.ttime <= la2.ttime and la1.demand <= la2.demand:
                            U.discard(l2)
                            self.labels[l2].dominated = True
                            cleaning.append(l2)

                        # Check if la1 is dominated by la2
                        elif la2.cost <= la1.cost and la2.ttime <= la1.ttime and la2.demand <= la1.demand:
                            U.discard(l1)
                            self.labels[l1].dominated = True
                            cleaning.append(l1)
                            j = len(city2labels[current.city])

            for c in cleaning:
                city2labels[current.city].remove(c)
            cleaning = None

            # Update checkDom counter
            checkDom[current.city] = len(city2labels[current.city])

            # Expand labels from current (if not dominated)
            if not current.dominated:
                if current.city == self.paramsVRP.nbclients:  # Reached end depot
                    if current.cost < -1e-7:  # Negative cost route candidate
                        P.add(current_idx)
                        nbsol = sum(1 for labi in P if not self.labels[labi].dominated)
                        
                        if min_cost is None or current.cost < min_cost:
                            min_cost = current.cost
                        
                        # Early stopping
                        if early_stop and nbsol >= min_columns_early_stop:
                            print(f"  [Early Stop] Found {nbsol} negative cost routes (>= {min_columns_early_stop}), stopping.")
                            break
                else:  # Extend to other customers
                    for i in range(node_count):
                        # Skip eliminated customers
                        if i in eliminated_customers:
                            continue
                        
                        # Check if edge exists (not forbidden)
                        if self.paramsVRP.dist[current.city][i] < self.paramsVRP.verybig - 1e-6:
                            # Calculate new label attributes
                            tt = current.ttime + self.paramsVRP.s[current.city] + self.paramsVRP.ttime[current.city][i]
                            if tt < self.paramsVRP.a[i]:
                                tt = self.paramsVRP.a[i]
                            
                            d = current.demand + self.paramsVRP.d[i]

                            # Check resource feasibility (time window and capacity)
                            if tt <= self.paramsVRP.b[i] and d <= self.paramsVRP.capacity:
                                idx = len(self.labels)
                                
                                # Calculate cost (using cost matrix)
                                new_cost = current.cost + self.paramsVRP.cost[current.city][i]
                                
                                # Create new label (NO vertex_visited checking - allows cycles!)
                                self.labels.append(self.label(i, current_idx, new_cost, tt, d, False, self))
                                
                                if idx not in U:
                                    U.add(idx)
                                    city2labels[i].append(idx)

        # Filter and convert labels to routes
        valid_labels = []
        for lab_idx in P:
            if not self.labels[lab_idx].dominated and self.labels[lab_idx].cost < -1e-4:
                valid_labels.append((lab_idx, self.labels[lab_idx].cost))
        
        # Sort by cost
        valid_labels.sort(key=lambda x: x[1])
        
        # Apply lambda pricing if enabled
        if lambda_pricing and valid_labels:
            ratios = []
            for lab_idx, cost in valid_labels:
                # Reconstruct path for dual computation
                path = []
                path_idx = lab_idx
                while path_idx >= 0:
                    path.append(self.labels[path_idx].city)
                    path_idx = self.labels[path_idx].index_prev_label
                path.reverse()

                # Compute π^T a_x
                pi_T_ax = 0.0
                if dual_pi is not None:
                    for city in path[1:-1]:  # Exclude depots
                        if 1 <= city < self.paramsVRP.nbclients:
                            pi_T_ax += dual_pi[city - 1]

                if pi_T_ax > 1e-9:
                    ratio = (-cost) / pi_T_ax
                    ratios.append((lab_idx, cost, ratio))

            if ratios:
                ratios.sort(key=lambda x: x[2])
                min_ratio = ratios[0][2]
                lambda_threshold = lambda_factor * min_ratio

                valid_labels = [(lab_idx, cost) for lab_idx, cost, ratio in ratios
                                if ratio <= lambda_threshold]

                print(f"  [Lambda Pricing (Ratio) - Bixby et al., 1992]")
                print(f"    l(π) = min((-c_x) / π^T a_x) = {min_ratio:.4f}")
                print(f"    λ · l(π) = {lambda_factor} × {min_ratio:.4f} = {lambda_threshold:.4f}")
                print(f"    Kept {len(valid_labels)} columns under ratio threshold")
        
        # Convert labels to routes
        i = 0
        for lab_idx, cost in valid_labels:
            if i >= nbroute:
                break
            
            route = Route()
            route.set_cost(self.labels[lab_idx].cost)
            route.add_city(self.labels[lab_idx].city)
            path = self.labels[lab_idx].index_prev_label
            while path >= 0:
                route.add_city(self.labels[path].city)
                path = self.labels[path].index_prev_label
            route.switch_path()
            routes.append(route)
            i += 1
        
        if routes:
            print(f"  [Columns Generated] {len(routes)} columns returned, costs: {[r.cost for r in routes]}")

        return valid_labels
