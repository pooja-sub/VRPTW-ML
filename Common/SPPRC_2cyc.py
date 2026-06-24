import heapq
from functools import total_ordering
from sortedcontainers import SortedSet
from route import Route
from paramsVRP import ParamsVRP

'''
 Shortest Path with Resource Constraints - 2-Cycle Elimination (SPPRC-2-cyc)
 
 Extends SPPRC non-elementary with enhanced dominance rule for 2-cycle elimination.
 
 Key Idea (Kohl 1995, Larsen 1999):
 - Keep the Pareto-best path P1 at a city
 - Keep a second-best path P2 from a DIFFERENT predecessor node
 - Eliminate any path Q if both P1 and P2 are better in time and have different predecessors
 
 Prevents 2-cycles like: v -> pred(v) -> v -> pred(v) -> ...
 
 Label state now includes:
 - city, cost, ttime, demand, dominated
 - index_prev_label (to track predecessor node)
 - pred_node: the second-to-last node (predecessor of current city in the path)
'''


class SPPRC_2Cyc:

    def __init__(self, userParam=None):
        self.paramsVRP = ParamsVRP() if userParam is None else userParam
        self.labels = []

    @total_ordering
    class label:
        def __init__(self, city, index_prev_label, cost, ttime, demand, dominated, pred_node, parent):
            self.city = city  # int - current city
            self.index_prev_label = index_prev_label  # int - previous label index
            self.cost = cost  # double
            self.ttime = ttime  # float
            self.demand = demand  # double
            self.dominated = dominated  # boolean
            self.pred_node = pred_node  # int - predecessor node (second-to-last in path)
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

    def _get_predecessor_node(self, label_idx):
        """Extract the predecessor node (second-to-last node in path) from a label."""
        if label_idx < 0:
            return -1
        label = self.labels[label_idx]
        if label.index_prev_label < 0:
            return -1  # Starting label (depot), no predecessor
        return label.city

    def _compute_predecessor_node(self, prev_label_idx, current_city, parent_spprc):
        """Compute predecessor node: the node we came from to reach current_city."""
        if prev_label_idx < 0:
            return -1  # First label, no predecessor
        # Predecessor is the city of the previous label
        return parent_spprc.labels[prev_label_idx].city

    def shortestPath(self, userParamArg, routes, nbroute, early_stop=True, 
                     lambda_pricing=True, lambda_factor=2.0, dual_pi=None, max_columns=None,
                     eliminated_customers=None, min_columns_early_stop=10):
        '''
        Label-setting algorithm for SPPRC with 2-cycle elimination.
        
        Enhanced dominance:
        - Standard SPPRC: eliminate Q if dominated by P on all resources
        - 2-cycle: also eliminate Q if P1 (best) and P2 (second-best from different pred)
                    are both better than Q in time, preventing 2-cycles
        
        :param early_stop: If True, stop after finding min_columns_early_stop negative cost routes
        :param lambda_pricing: If True, apply lambda pricing rule
        :param lambda_factor: Multiplier for lambda pricing rule (default 2.0)
        :param dual_pi: List of dual values π for customers
        :param max_columns: Maximum number of columns to keep
        :param eliminated_customers: Set of customer IDs (1-indexed) to eliminate
        :param min_columns_early_stop: Minimum columns before early stopping (default 10)
        '''
        if eliminated_customers is None:
            eliminated_customers = set()
        
        print(f"[---SPPRC_2Cyc.shortestPath called---] early_stop={early_stop}, lambda_pricing={lambda_pricing}")
        self.paramsVRP = userParamArg
        if max_columns is None:
            max_columns = 2 * nbroute

        # Initialize unprocessed (U) and processed (P) label sets
        U = SortedSet(key=lambda x: x)
        P = SortedSet(key=lambda x: x)

        # Initialize labels array and starting label at depot
        self.labels = []
        self.labels.append(self.label(0, -1, 0.0, 0, 0, False, -1, self))  # Depot label
        U.add(0)

        # For each city, track indices of labels at that city (for dominance)
        checkDom = [0] * (self.paramsVRP.nbclients + 1)
        city2labels = [[] for _ in range(self.paramsVRP.nbclients + 1)]
        city2labels[0].append(0)

        nbsol = 0
        maxsol = max_columns
        min_cost = None

        while U and nbsol < maxsol:
            current_idx = U.pop(0)
            current = self.labels[current_idx]

            # Enhanced dominance check with 2-cycle elimination
            cleaning = []
            labels_at_city = city2labels[current.city][checkDom[current.city]:]
            
            for i in range(len(labels_at_city)):
                for j in range(i):
                    l1_idx = labels_at_city[i]
                    l2_idx = labels_at_city[j]
                    la1 = self.labels[l1_idx]
                    la2 = self.labels[l2_idx]
                    
                    if not la1.dominated and not la2.dominated and l1_idx != l2_idx:
                        # ===== STANDARD SPPRC DOMINANCE =====
                        # Check if la2 is dominated by la1 (pure Pareto)
                        if (la1.cost <= la2.cost and la1.ttime <= la2.ttime and 
                            la1.demand <= la2.demand):
                            U.discard(l2_idx)
                            self.labels[l2_idx].dominated = True
                            cleaning.append(l2_idx)

                        # Check if la1 is dominated by la2 (pure Pareto)
                        elif (la2.cost <= la1.cost and la2.ttime <= la1.ttime and 
                              la2.demand <= la1.demand):
                            U.discard(l1_idx)
                            self.labels[l1_idx].dominated = True
                            cleaning.append(l1_idx)

            # ===== 2-CYCLE ELIMINATION RULE (Kohl 1995, Larsen 1999) =====
            # For labels with different predecessors:
            # If P1 (best) and P2 (second-best from different pred) both have time <= Q's time,
            # then Q can be eliminated because {P1, P2} can extend to all successors of Q
            
            if len(labels_at_city) >= 3:
                # Group remaining labels by predecessor node
                pred_to_labels = {}
                for label_idx in labels_at_city:
                    if not self.labels[label_idx].dominated:
                        pred = self.labels[label_idx].pred_node
                        if pred not in pred_to_labels:
                            pred_to_labels[pred] = []
                        pred_to_labels[pred].append(label_idx)
                
                # If we have labels from at least 2 different predecessors
                if len(pred_to_labels) >= 2:
                    # Find P1 and P2: best and second-best from different predecessors
                    all_active = [idx for idx in labels_at_city if not self.labels[idx].dominated]
                    
                    if len(all_active) >= 2:
                        # Sort by time (primary), then cost
                        all_active.sort(key=lambda x: (self.labels[x].ttime, self.labels[x].cost))
                        
                        P1_idx = all_active[0]
                        P1 = self.labels[P1_idx]
                        
                        # Find P2: second-best from DIFFERENT predecessor
                        P2_idx = None
                        for idx in all_active[1:]:
                            if self.labels[idx].pred_node != P1.pred_node:
                                P2_idx = idx
                                break
                        
                        if P2_idx is not None:
                            P2 = self.labels[P2_idx]
                            
                            # Now eliminate any Q where:
                            # - v(P1) = v(P2) = v(Q) (all at same city - guaranteed)
                            # - T(P1) <= T(Q) and T(P2) <= T(Q) (both better in time)
                            # - pred(P1) != pred(P2) (already verified)
                            
                            for q_idx in all_active:
                                if q_idx != P1_idx and q_idx != P2_idx:
                                    Q = self.labels[q_idx]
                                    if (P1.ttime <= Q.ttime and P2.ttime <= Q.ttime and
                                        not Q.dominated):
                                        # Eliminate Q: P1 and P2 together cover its extension potential
                                        U.discard(q_idx)
                                        self.labels[q_idx].dominated = True
                                        cleaning.append(q_idx)
                                        print(f"    [2-Cyc Elimination] Removed label {q_idx} at city {Q.city} "
                                              f"(dominated by {P1_idx} and {P2_idx} with different predecessors)")

            for c in cleaning:
                if c in city2labels[current.city]:
                    city2labels[current.city].remove(c)
            cleaning = None

            # Update checkDom counter
            checkDom[current.city] = len(city2labels[current.city])

            # Expand labels from current
            if not current.dominated:
                if current.city == self.paramsVRP.nbclients:  # Reached end depot
                    if current.cost < -1e-7:
                        P.add(current_idx)
                        nbsol = sum(1 for labi in P if not self.labels[labi].dominated)
                        
                        if min_cost is None or current.cost < min_cost:
                            min_cost = current.cost
                        
                        if early_stop and nbsol >= min_columns_early_stop:
                            print(f"  [Early Stop] Found {nbsol} negative cost routes (>= {min_columns_early_stop})")
                            break
                else:  # Extend to other cities
                    for i in range(self.paramsVRP.nbclients + 1):
                        if i in eliminated_customers:
                            continue
                        
                        if self.paramsVRP.dist[current.city][i] < self.paramsVRP.verybig - 1e-6:
                            # Calculate new label attributes
                            tt = current.ttime + self.paramsVRP.s[current.city] + self.paramsVRP.ttime[current.city][i]
                            if tt < self.paramsVRP.a[i]:
                                tt = self.paramsVRP.a[i]
                            
                            d = current.demand + self.paramsVRP.d[i]

                            if tt <= self.paramsVRP.b[i] and d <= self.paramsVRP.capacity:
                                idx = len(self.labels)
                                new_cost = current.cost + self.paramsVRP.cost[current.city][i]
                                
                                # Compute predecessor node for new label
                                pred_node = current.city
                                
                                # Create new label with predecessor tracking
                                self.labels.append(self.label(i, current_idx, new_cost, tt, d, False, pred_node, self))
                                
                                if idx not in U:
                                    U.add(idx)
                                    city2labels[i].append(idx)

        # Filter and convert labels to routes
        valid_labels = []
        for lab_idx in P:
            if not self.labels[lab_idx].dominated and self.labels[lab_idx].cost < -1e-4:
                valid_labels.append((lab_idx, self.labels[lab_idx].cost))
        
        valid_labels.sort(key=lambda x: x[1])
        
        # Apply lambda pricing if enabled
        if lambda_pricing and valid_labels:
            ratios = []
            for lab_idx, cost in valid_labels:
                path = []
                path_idx = lab_idx
                while path_idx >= 0:
                    path.append(self.labels[path_idx].city)
                    path_idx = self.labels[path_idx].index_prev_label
                path.reverse()

                pi_T_ax = 0.0
                if dual_pi is not None:
                    for city in path[1:-1]:
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

                print(f"  [Lambda Pricing] l(π) = {min_ratio:.4f}, threshold = {lambda_threshold:.4f}, "
                      f"kept {len(valid_labels)} columns")
        
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
            print(f"  [Columns Generated] {len(routes)} columns returned")

        return valid_labels
