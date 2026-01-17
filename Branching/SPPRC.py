import heapq
from functools import cmp_to_key
import numpy as np
from route import Route
from paramsVRP import ParamsVRP
from sortedcontainers import SortedSet
from functools import total_ordering

'''
 shortest path with resource constraints
 inspired by Irnish and Desaulniers, "SHORTEST PATH PROBLEMS WITH RESOURCE CONSTRAINTS"
 for educational demonstration only - (nearly) no code optimization

 four main lists will be used:
 labels: array (ArraList) => one dimensional unbounded vector
        list of all labels created along the feasible paths (i.e. paths satisfying the resource constraints)

 U: sorted list (TreeSet) => one dimensional unbounded vector
        sorted list containing the indices of the unprocessed labels (paths that can be extended to obtain a longer feasible path)

 P: sorted list (TreeSet) => one dimensional unbounded vector
        sorted list containing the indices of the processed labels ending at the depot with a negative cost

 city2labels: matrix (array of ArrayList) => nbclients x unbounded
        for each city, the list of (indices of the) labels attached to this city/vertex
        before processing a label at vertex i, we compare pairwise all labels at the same vertex to remove the dominated ones
'''


class SPPRC:

    def __init__(self, userParam=None):
        self.paramsVRP = ParamsVRP() if userParam is None else userParam
        self.labels = []
        #self.myLabelComparator = self.MyLabelComparator(self)

    @total_ordering
    class label:
        def __init__(self, city, index_prev_label, cost, ttime, demand, dominated, vertex_visited, parent):
            self.city = city  # int
            self.index_prev_label = index_prev_label  # int
            self.cost = cost  # double
            self.ttime = ttime  # float
            self.demand = demand  # double
            self.dominated = dominated  # boolean
            self.vertex_visited = vertex_visited  # boolean list
            self.parent = parent

        def updateLabel(self, a1, a2, a3, a4, a5, a6, a7):
            self.city = a1  # current vertex
            self.index_prev_label = a2  # previous label in the same path
            # (i.e. previous vertex in the same path with the state of the resources)
            self.cost = a3  # first resource: cost (e.g. distance or strict travel time)
            self.ttime = a4  # second resource: travel time along the path (including wait time and service time)
            self.demand = a5  # third resource: demand,i.e. total quantity delivered to the clients encountered on this path
            self.dominated = a6  # is this label dominated by another one? i.e. if dominated, forget this path.
            self.vertex_visited = a7

        def __lt__(self, other):
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
                            i = 0
                            while i < self.parent.paramsVRP.nbclients:
                                if self.vertex_visited[i] != other.vertex_visited[i]:
                                    if self.vertex_visited[i]:
                                        return True
                                    else:
                                        return False
                                i += 1
                            return False
                elif self.city > other.city:
                    return False
                else:
                    return True


        def __eq__(self, other):
            if self.cost - other.cost < -1e-7 or self.cost - other.cost > 1e-7:
                return False
            if self.city < other.city or self.city > other.city:
                return False
            if self.ttime - other.ttime < -1e-7 or self.ttime - other.ttime > 1e-7:
                return False
            if self.demand - other.demand < -1e-7 or self.demand - other.demand > 1e-7:
                return False

            if self.cost - other.cost > -1e-7 and self.cost - other.cost < 1e-7:
                if self.city == other.city:
                    if self.ttime - other.ttime > -1e-7 and self.ttime - other.ttime < 1e-7:
                        if self.demand - other.demand > -1e-7 and self.demand - other.demand < 1e-7:
                            i = 0
                            while i < self.parent.paramsVRP.nbclients:
                                if self.vertex_visited[i] != other.vertex_visited[i]:
                                    if self.vertex_visited[i]:
                                        return False
                                    else:
                                        return False
                                i += 1
                            return True



    '''
    class MyLabelComparator:

        def __init__(self, parent):
            self.parent = parent

        def compare(self, a, b):
            A = self.parent.labels[a]
            B = self.parent.labels[b]

            try:
                if A.cost - B.cost < 1e-7:
                    return -1
                elif A.cost - B.cost > 1e-7:
                    return 1
                else:
                    if A.city == B.city:
                        if A.ttime - B.ttime < -1e-7:
                            return -1
                        elif A.ttime - B.ttime > 1e-7:
                            return 1
                        else:
                            if A.demand - B.demand < -1e-7:
                                return -1
                            elif A.demand - B.demand > 1e-7:
                                return 1
                            else:
                                i = 0
                                while i < self.parent.paramsVRP.nbclients:
                                    if A.vertex_visited[i] != B.vertex_visited[i]:
                                        if A.vertex_visited[i]:
                                            return -1
                                        else:
                                            return 1
                                    i += 1
                                return 0
                    elif A.city > B.city:
                        return 1
                    else:
                        return -1
            except Exception as e:
                print("Error in MyLabelComparator.compare: ", e)
        '''


    def shortestPath(self, userParamArg, routes, nbroute, early_stop=True, 
                     lambda_pricing=True, lambda_factor=2.0, dual_pi=None, max_columns=None):
        '''
        This function implements the label-setting algorithm from Irnish and Desaulniers
        with multi-column selection using filtering rules.

        NOTE - this code implements 3 techniques
        1. dominance
        2. time-window strengthening (implemented in parametervehicle routing)
        3. early elimination of customers (speedup 3) -> Feillet et al. (2004), p. 495, Section 4.4
        
          FILTERING RULES:
          1. Lambda Pricing Rule (Bixby et al., 1992) - Ratio only:
              - l(π) = min{c_x / (π^T a_x) | π^T a_x > 0}
              - Keep columns with c_x / (π^T a_x) ≤ λ · l(π)
              - Good for set partitioning problems
              - Keeps richer columns (more non-zero entries) in addition to most negative
        
        :param early_stop: If True, stop as soon as one negative cost route is found.
        :param lambda_pricing: If True, apply lambda pricing rule (ratio-based)
        :param lambda_factor: Multiplier for lambda pricing rule (default 2.0)
        :param dual_pi: List of dual values π for customers to compute π^T a_x
        :param max_columns: Maximum number of columns to keep.
        '''
        print(f"[---SPPRC.shortestPath called---] early_stop={early_stop}, lambda_pricing={lambda_pricing}")
        self.paramsVRP = userParamArg
        if max_columns is None:
            max_columns = 2 * nbroute

        # Initialize unprocessed labels list (U) and processed labels list (P)
        U = SortedSet(key=lambda x: x)
        P = SortedSet(key=lambda x: x)

        # Initialize labels array        labels = []
        # for depot 0
        cust = [False] * (self.paramsVRP.nbclients + 2)
        cust[0] = True
        self.labels.append(self.label(0, -1, 0.0, 0, 0, False, cust, self))  # First label: start from depot (client 0)
        U.add(0)

        # For each city, an array with the index of the corresponding labels (for dominance)
        checkDom = [0] * (self.paramsVRP.nbclients + 2) # Number of labels checked for dominance at each customer node
        city2labels = [[] for _ in range(self.paramsVRP.nbclients + 2)]
        city2labels[0].append(0)
        #print("checkDom", checkDom)
        #print("city2labels:", city2labels)

        nbsol = 0
        maxsol = max_columns
        min_cost = None  # Track minimum cost for lambda pricing

        while U and nbsol < maxsol:
            #print("U:", U)
            current_idx = 0
            current_idx = U.pop(0)  # Process one label => get the index AND remove it from U
            current = self.labels[current_idx]

            # Check for dominance
            cleaning = []
            for i in range(checkDom[current.city], len(city2labels[current.city])):
                for j in range(i):
                    l1, l2 = city2labels[current.city][i], city2labels[current.city][j]
                    la1, la2 = self.labels[l1], self.labels[l2]
                    if not la1.dominated and not la2.dominated and l1 != l2:

                        # Q1: Check if label 2 is dominated
                        pathdom = True
                        for k in range(1, self.paramsVRP.nbclients + 2):
                            if not pathdom:
                                break
                            pathdom = pathdom and (not la1.vertex_visited[k] or la2.vertex_visited[k])
                        if pathdom and la1.cost <= la2.cost and la1.ttime <= la2.ttime and la1.demand <= la2.demand:
                            #print(f'U:{U}')
                            #print(f'[l2({l2}).dominated=True]labels:{[label.dominated for label in self.labels]}')
                            U.discard(l2)
                            self.labels[l2].dominated = True
                            cleaning.append(l2)
                            pathdom = False

                        pathdom = True
                        # Q2: Check if label 1 is dominated
                        for k in range(1, self.paramsVRP.nbclients + 2):
                            pathdom = pathdom and (not la2.vertex_visited[k] or la1.vertex_visited[k])
                        if pathdom and la2.cost <= la1.cost and la2.ttime <= la1.ttime and la2.demand <= la1.demand:
                            #print(f'U:{U}')
                            #print(f'[l1({l1}).dominated=True]labels:{[label.dominated for label in self.labels]}')
                            U.discard(l1)
                            self.labels[l1].dominated = True
                            cleaning.append(l1)
                            j = len(city2labels[current.city])

            for c in cleaning:
                city2labels[current.city].remove(c)
            cleaning = None

            # Update CheckDom: all labels in city2labels have been checked for dominance
            checkDom[current.city] = len(city2labels[current.city])
            #print(f'U:{U}, checkDom:{checkDom}')

            # Expand REF
            if not current.dominated:
                #print(f'[current_idx]:{current_idx} is not dominated')
                if current.city == self.paramsVRP.nbclients:  # Shortest path candidate to the depot!
                    if current.cost < -1e-7:  # SP candidate for the column generation
                        P.add(current_idx)
                        #print(f'[current_idx ADDED]:{current_idx}')
                        nbsol = sum(1 for labi in P if not self.labels[labi].dominated)
                        
                        # Track minimum cost for lambda pricing
                        if min_cost is None or current.cost < min_cost:
                            min_cost = current.cost
                        
                        # Early stopping: if enabled and we found a negative cost route, stop
                        if early_stop and nbsol >= 1:
                            print(f"  [Early Stop] Found negative cost route, stopping SPPRC early")
                            break
                else:  # If not the depot, we can consider extensions of the path
                    for i in range(self.paramsVRP.nbclients + 2):
                        if not current.vertex_visited[i] and self.paramsVRP.dist[current.city][i] < self.paramsVRP.verybig - 1e-6:
                            # ttime already includes service time at current.city
                            tt = current.ttime + self.paramsVRP.ttime[current.city][i]
                            if tt < self.paramsVRP.a[i]:
                                tt = self.paramsVRP.a[i]
                            d = current.demand + self.paramsVRP.d[i]

                            if tt <= self.paramsVRP.b[i] and d <= self.paramsVRP.capacity:
                                idx = len(self.labels)
                                newcust = current.vertex_visited[:]
                                newcust[i] = True

                                # Speedup: third technique - Feillet 2004 as mentioned in Laporte's paper
                                for j in range(1, self.paramsVRP.nbclients):
                                    if not newcust[j]:
                                        # ttime[i][j] already includes service time at i
                                        tt2 = tt + self.paramsVRP.ttime[i][j]
                                        d2 = d + self.paramsVRP.d[j]
                                        if tt2 > self.paramsVRP.b[j] or d2 > self.paramsVRP.capacity:
                                            newcust[j] = True

                                self.labels.append(self.label(i, current_idx, current.cost + self.paramsVRP.cost[current.city][i], tt, d, False, newcust, self))
                                if idx not in U:
                                    U.add(idx)
                                    city2labels[i].append(idx)
                                else:
                                    self.labels[idx].dominated = True

        # Filtering: find the path from depot to the destination
        # Apply filtering rules before converting labels to routes
        valid_labels = []
        
        for lab_idx in P:
            if not self.labels[lab_idx].dominated and self.labels[lab_idx].cost < -1e-4:
                valid_labels.append((lab_idx, self.labels[lab_idx].cost))
        
        # Sort by cost (most negative first)
        valid_labels.sort(key=lambda x: x[1])
        
        # Apply lambda pricing rule if enabled (ratio-only)
        # Lambda pricing rule (Bixby et al., 1992):
        # Define ratio r_x = (-c_x) / (π^T a_x) for columns with c_x < 0 and π^T a_x > 0
        # l(π) = min_{x∈X} r_x
        # Keep columns with r_x ≤ λ · l(π)
        if lambda_pricing and valid_labels:
            ratios = []
            for lab_idx, cost in valid_labels:
                # Reconstruct the route path to compute π^T a_x from duals
                path = []
                path_idx = lab_idx
                while path_idx >= 0:
                    path.append(self.labels[path_idx].city)
                    path_idx = self.labels[path_idx].index_prev_label
                path.reverse()

                # Compute π^T a_x using provided duals
                pi_T_ax = 0.0
                if dual_pi is not None:
                    for city in path[1:-1]:  # Exclude depots
                        if 1 <= city < self.paramsVRP.nbclients:
                            pi_T_ax += dual_pi[city - 1]

                # Only consider columns with π^T a_x > 0 to avoid division by zero
                if pi_T_ax > 1e-9:
                    # cost is reduced cost and negative for improving columns; use -cost for positive ratio
                    ratio = (-cost) / pi_T_ax
                    ratios.append((lab_idx, cost, ratio))

            if ratios:
                ratios.sort(key=lambda x: x[2])
                min_ratio = ratios[0][2]  # l(π), now positive
                lambda_threshold = lambda_factor * min_ratio

                # Keep columns with ratio ≤ threshold
                valid_labels = [(lab_idx, cost) for lab_idx, cost, ratio in ratios
                                if ratio <= lambda_threshold]

                print(f"  [Lambda Pricing (Ratio) - Bixby et al., 1992]")
                print(f"    l(π) = min((-c_x) / π^T a_x) = {min_ratio:.4f}")
                print(f"    λ · l(π) = {lambda_factor} × {min_ratio:.4f} = {lambda_threshold:.4f}")
                print(f"    Kept {len(valid_labels)} columns under ratio threshold")
        
        # Convert labels to routes
        i = 0
        checkDom = None
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

        # Return label indices and costs used for ratio filtering (not used by caller currently)
        return valid_labels
