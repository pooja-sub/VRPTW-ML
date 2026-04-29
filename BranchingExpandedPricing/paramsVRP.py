import numpy as np

# this class contains the inputs and methods to read the inputs
# for the Branch and Price CVRP with TW

class ParamsVRP:
    def __init__(self, nbclients=100, capacity=0, mvehic=0, speed=1.0, service_in_tw=False):
        """
        Initialize parameter class for storing and processing Vehicle Routing Problem (VRP) parameters.

        :param nbclients: Number of customers (excluding start and end depot)
        :param capacity: Vehicle capacity
        :param mvehic: Number of vehicles
        :param speed: Vehicle speed
        :param service_in_tw: Whether to consider service time within time window
        """
        self.datasetName = ""
        self.instance_path = ""
        self.verbose = False
        self.rndseed = 0
        self.nbclients = nbclients  # Number of customers
        self.capacity = capacity  # Vehicle capacity
        self.mvehic = mvehic  # Number of vehicles
        self.speed = speed  # Vehicle speed
        self.service_in_tw = service_in_tw  # Whether to consider service time within time window

        self.verybig = 1e10  # A very large number representing infinity
        self.gap = 1e-6  # Tolerance error in optimization process

        self.citieslab = None  # City labels
        self.posx = None  # City x coordinates
        self.posy = None  # City y coordinates
        self.d = None  # City demand
        self.a = None  # Time window start time
        self.b = None  # Time window end time
        self.s = None  # Service time

        self.dist_base = None  # Original distance matrix
        self.dist = None  # Updated distance matrix
        self.ttime = None  # Travel time matrix
        self.cost = None  # Cost matrix
        self.edges = None  # Edge weight matrix

    def init_params(self, input_path):
        try:
            self.instance_path = input_path
            with open(input_path, 'r') as file:
                lines = file.readlines()

            # Check if file content meets expectations
            if len(lines) < 10:  # At least 10 lines of data are required
                raise ValueError(f"File {input_path} has insufficient lines, cannot read required data.")

            self.datasetName = lines[0].strip()
            print(f'----------[Reading dataset: {self.datasetName}]----------')

            self.mvehic = int(lines[4].strip().split()[0])
            self.capacity = int(lines[4].strip().split()[1])
            self.nbclients = len(lines) - 9
            print(f'Number of vehicles: {self.mvehic}')
            print(f'Capacity: {self.capacity}')
            print(f'Number of customers: {self.nbclients}')

            # Initialize other data structures (nbclients+2 to accommodate start depot at 0, customers at 1 to nbclients, end depot at nbclients+1)
            self.citieslab = [None] * (self.nbclients + 2)
            self.posx = np.zeros(self.nbclients + 2)
            self.posy = np.zeros(self.nbclients + 2)
            self.d = np.zeros(self.nbclients + 2)
            self.a = np.zeros(self.nbclients + 2, dtype=int)
            self.b = np.zeros(self.nbclients + 2, dtype=int)
            self.s = np.zeros(self.nbclients + 2, dtype=int)
            self.dist_base = np.zeros((self.nbclients + 2, self.nbclients + 2))
            self.dist = np.zeros((self.nbclients + 2, self.nbclients + 2))
            self.ttime = np.zeros((self.nbclients + 2, self.nbclients + 2))
            self.cost = np.zeros((self.nbclients + 2, self.nbclients + 2))
            self.edges = np.zeros((self.nbclients + 2, self.nbclients + 2))

            # Read depot and customer data (depot at index 0, customers at indices 1 to nbclients)
            for i in range(0, self.nbclients):
                data = lines[9 + i].split()

                self.citieslab[i] = int(data[0])  # Customer ID
                self.posx[i] = float(data[1])  # x coordinate
                self.posy[i] = float(data[2])  # y coordinate
                self.d[i] = float(data[3])  # Demand
                self.a[i] = int(data[4])  # Time window start time
                self.b[i] = int(data[5])  # Time window end time
                self.s[i] = int(data[6])  # Service time
                if self.service_in_tw and i > 0:  # Only adjust time window for customers, not depot
                    self.b[i] -= self.s[i]  # If service time is within time window, adjust time window end time
                if i == 0:
                    print(f"Depot {i}: {self.citieslab[i]} {self.posx[i]} {self.posy[i]} {self.d[i]} {self.a[i]} {self.b[i]} {self.s[i]}")
                else:
                    print(f"Customer {i}: {self.citieslab[i]} {self.posx[i]} {self.posy[i]} {self.d[i]} {self.a[i]} {self.b[i]} {self.s[i]}")

            # Copy depot information to end depot (index nbclients)
            self.citieslab[self.nbclients] = self.nbclients
            self.posx[self.nbclients] = self.posx[0]
            self.posy[self.nbclients] = self.posy[0]
            self.d[self.nbclients] = 0.0
            self.a[self.nbclients] = self.a[0]
            self.b[self.nbclients] = self.b[0]
            self.s[self.nbclients] = 0
            print(f'End depot {self.citieslab[self.nbclients]}: {self.citieslab[self.nbclients]} '
                        f'{self.posx[self.nbclients]} {self.posy[self.nbclients]} {self.d[self.nbclients]} '
                        f'{self.a[self.nbclients]} {self.b[self.nbclients]} {self.s[self.nbclients]}')

            # Calculate distance matrix
            for i in range(self.nbclients + 2):
                for j in range(self.nbclients + 2):
                    # truncate to get the same results as in Solomon
                    self.dist_base[i, j] = np.round(
                        10 * np.sqrt((self.posx[i] - self.posx[j]) ** 2 + (self.posy[i] - self.posy[j]) ** 2)) / 10.0

            # Set depot to customer and vice versa distance as infinity and diagonal to infinity
            for i in range(self.nbclients + 2):
                self.dist_base[i, 0] = self.verybig  # Can't return to start depot
                self.dist_base[self.nbclients, i] = self.verybig  # Can't leave end depot
                self.dist_base[i, i] = self.verybig  # No self-loops

            for i in range(self.nbclients + 2):
                for j in range(self.nbclients + 2):
                    self.dist[i, j] = self.dist_base[i, j]

            # Calculate travel time matrix (including service time at origin)
            for i in range(self.nbclients + 2):
                for j in range(self.nbclients + 2):
                    self.ttime[i, j] = self.dist_base[i, j] / self.speed # + self.s[i]

            # Other edge costs are given in column generation
            for j in range(self.nbclients + 2):
                self.cost[0][j] = self.dist[0][j]
                self.cost[j][self.nbclients] = self.dist[j][self.nbclients]

            # Apply preprocessing to reduce arcs
            self.preprocess_arcs()
            
            # Apply cyclic time window strengthening
            self.strengthen_time_windows_cyclic()

            print(f"----------[ParamsVRP initialization complete]----------")


        except FileNotFoundError:
            print(f"Error: File {input_path} not found.")
        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"Error in init_params: {e}")

    def preprocess_arcs(self):
        """
        Apply preprocessing techniques to reduce the number of arcs in the network.
        
        1. Strengthens time windows: [ai, bi] -> [max(a0 + t0i, ai), min(bn+1 - ti,n+1, bi)]
        2. Eliminates arcs (i,j) based on:
           - Capacity constraint: di + dj > q (demand infeasibility)
           - Time window constraint: ai + si + tij > bj (time window infeasibility)
        """
        print(f"[Starting preprocessing: time window strengthening and arc elimination]")
        
        # Step 1: Strengthen time windows for all customers
        print(f"[Strengthening time windows]")
        strengthened_count = 0
        for i in range(1, self.nbclients):  # Only customers (not depots)
            original_ai = self.a[i]
            original_bi = self.b[i]
            
            # Strengthen lower bound: ai = max(a0 + t0i, ai)
            earliest_from_depot = self.a[0] + self.ttime[0][i]
            new_ai = max(earliest_from_depot, self.a[i])
            
            # Strengthen upper bound: bi = min(bn+1 - ti,n+1, bi)
            # Note: ttime already includes service time at i
            latest_to_depot = self.b[self.nbclients] - self.ttime[i][self.nbclients]
            new_bi = min(latest_to_depot, self.b[i])
            
            # Update time windows
            if new_ai != original_ai or new_bi != original_bi:
                self.a[i] = int(new_ai)
                self.b[i] = int(new_bi)
                strengthened_count += 1
        
        print(f"[Time windows strengthened for {strengthened_count} customers]")
        
        # Step 2: Eliminate infeasible arcs
        print(f"[Eliminating infeasible arcs]")
        eliminated_count = 0
        total_arcs = 0
        
        # Check all arcs between customers (not including depots in the checks)
        for i in range(self.nbclients + 2):
            for j in range(self.nbclients + 2):
                # Skip if already eliminated or is a depot self-loop
                if self.dist_base[i][j] >= self.verybig - 1e-6:
                    continue
                
                total_arcs += 1
                eliminated = False
                
                # Skip depot-to-depot and self-loops (already handled)
                if i == j or i == self.nbclients or j == 0:
                    continue
                
                # 1. Capacity constraint: di + dj > q
                if self.d[i] + self.d[j] > self.capacity:
                    self.dist[i][j] = self.verybig
                    self.dist_base[i][j] = self.verybig
                    eliminated = True
                    eliminated_count += 1
                
                # 2. Time window constraint: ai + tij > bj
                if not eliminated:
                    earliest_arrival_j = self.a[i] + self.s[i] + self.ttime[i][j]
                    if earliest_arrival_j > self.b[j]:
                        self.dist[i][j] = self.verybig
                        self.dist_base[i][j] = self.verybig
                        eliminated = True
                        eliminated_count += 1
        
        print(f"[Arc preprocessing complete] Eliminated {eliminated_count} out of {total_arcs} arcs")
        print(f"[Remaining arcs: {total_arcs - eliminated_count}]")
        
        # Calculate and display arc degree statistics
        self.calculate_arc_degree_statistics()

    def calculate_arc_degree_statistics(self):
        """
        Calculate and display arc degree statistics after preprocessing.
        
        The degree of a node is the number of feasible arcs incident to it.
        This includes both incoming and outgoing arcs.
        
        Displays:
        - Average degree across all nodes
        - Minimum degree among all nodes
        - Maximum degree among all nodes
        """
        # Use only the actual nodes (start depot + customers + end depot)
        # There is one extra padded row/column in the matrices because of historical +2 sizing;
        # we ignore that dummy node here to avoid inflating degrees.
        node_count = self.nbclients + 1  # indices 0..self.nbclients

        # Count incoming and outgoing arcs for each real node
        degree = np.zeros(node_count)
        
        for i in range(node_count):
            for j in range(i + 1, node_count):
                if self.dist[i][j] < self.verybig - 1e-6:
                    degree[i] += 1
                    degree[j] += 1
        
        # Exclude depots (0 and nbclients) from statistics to focus on customer nodes
        customer_degrees = degree[1:self.nbclients]
        
        if len(customer_degrees) > 0:
            avg_degree = np.mean(customer_degrees)
            min_degree = np.min(customer_degrees)
            max_degree = np.max(customer_degrees)
            
            print(f"[Arc Degree Statistics (for customer nodes)]")
            print(f"  Average degree: {avg_degree:.2f}")
            print(f"  Minimum degree: {int(min_degree)}")
            print(f"  Maximum degree: {int(max_degree)}")
        else:
            print(f"[Arc Degree Statistics] No customer nodes to analyze")

    def strengthen_time_windows_cyclic(self):
        """
        Apply cyclic time window strengthening using four rules.
        Based on Desrochers, Desrosiers and Solomon (1992).
        """

        print("[Starting cyclic time window strengthening]")

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changed = False
            print(f"  [Iteration {iteration}]")

            # Save windows at start of cycle
            a_old = self.a.copy()
            b_old = self.b.copy()

            # =========================
            # Rule 1: from predecessors
            # =========================
            a1 = a_old.copy()
            for l in range(1, self.nbclients):
                min_arrival = self.verybig
                for i in range(self.nbclients + 2):
                    if i != l and self.dist[i][l] < self.verybig - 1e-6:
                        min_arrival = min(min_arrival, a_old[i] + self.ttime[i][l])

                if min_arrival < self.verybig:
                    a1[l] = max(a_old[l], min(b_old[l], min_arrival))

            # =========================
            # Rule 2: to successors
            # =========================
            a2 = a1.copy()
            for l in range(1, self.nbclients):
                min_arrival = self.verybig
                for j in range(self.nbclients + 2):
                    if j != l and self.dist[l][j] < self.verybig - 1e-6:
                        min_arrival = min(min_arrival, a1[j] - self.ttime[l][j])

                if min_arrival < self.verybig:
                    a2[l] = max(a1[l], min(b_old[l], min_arrival))

            # =========================
            # Rule 3: from predecessors
            # =========================
            b3 = b_old.copy()
            for l in range(1, self.nbclients):
                max_departure = a2[l]
                for i in range(self.nbclients + 2):
                    if i != l and self.dist[i][l] < self.verybig - 1e-6:
                        max_departure = max(max_departure, b_old[i] + self.ttime[i][l])

                b3[l] = min(b_old[l], max_departure)

            # =========================
            # Rule 4: to successors
            # =========================
            b4 = b3.copy()
            for l in range(1, self.nbclients):
                max_departure = a2[l]
                for j in range(self.nbclients + 2):
                    if j != l and self.dist[l][j] < self.verybig - 1e-6:
                        max_departure = max(max_departure, b3[j] - self.ttime[l][j])

                b4[l] = min(b3[l], max_departure)

            # =========================
            # Convergence check
            # =========================
            if not np.array_equal(a2, self.a) or not np.array_equal(b4, self.b):
                changed = True
                self.a = a2
                self.b = b4

            if not changed:
                break

        print(f"[Cyclic time window strengthening complete] Iterations: {iteration}")


    def __str__(self):
        """
        Print parameter information.
        """
        return (f"ParamsVRP(\n"
                f"  nbclients={self.nbclients},\n"
                f"  capacity={self.capacity},\n"
                f"  mvehic={self.mvehic},\n"
                f"  speed={self.speed},\n"
                f"  service_in_tw={self.service_in_tw},\n"
                f"  verybig={self.verybig},\n"
                f"  gap={self.gap},\n"
                f")")