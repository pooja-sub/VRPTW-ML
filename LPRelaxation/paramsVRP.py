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
        self.verbose = False
        self.rndseed = 0
        self.nbclients = nbclients  # Number of customers
        self.capacity = capacity  # Vehicle capacity
        self.mvehic = mvehic  # Number of vehicles
        self.speed = speed  # Vehicle speed
        self.service_in_tw = service_in_tw  # Whether to consider service time within time window

        self.verybig = 1e10  # A very large number representing infinity
        self.gap = 1e-6  # Tolerance error in optimization process
        self.maxlength = 0.0  # Maximum path length

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
        self.wval = None  # Auxiliary variable

    def init_params(self, input_path):
        try:
            with open(input_path, 'r') as file:
                lines = file.readlines()

            # Check if file content meets expectations
            if len(lines) < 10:  # At least 10 lines of data are required
                raise ValueError(f"File {input_path} has insufficient lines, cannot read required data.")

            # Print first few lines for debugging
            '''
            print("File content (first 10 lines):")
            for i in range(min(10, len(lines))):
                print(f"Line {i}: {lines[i].strip()}")
            '''

            self.datasetName = lines[0].strip()
            print(f'----------[Reading dataset: {self.datasetName}]----------')

            self.mvehic = int(lines[4].strip().split()[0])
            self.capacity = int(lines[4].strip().split()[1])
            self.nbclients = len(lines) - 9
            print(f'Number of vehicles: {self.mvehic}')
            print(f'Capacity: {self.capacity}')
            print(f'Number of customers: {self.nbclients}')

            # Initialize other data structures
            self.citieslab = [None] * self.nbclients
            self.posx = np.zeros(self.nbclients)
            self.posy = np.zeros(self.nbclients)
            self.d = np.zeros(self.nbclients)
            self.a = np.zeros(self.nbclients, dtype=int)
            self.b = np.zeros(self.nbclients, dtype=int)
            self.s = np.zeros(self.nbclients, dtype=int)
            self.dist_base = np.zeros((self.nbclients, self.nbclients))
            self.dist = np.zeros((self.nbclients, self.nbclients))
            self.ttime = np.zeros((self.nbclients, self.nbclients))
            self.cost = np.zeros((self.nbclients, self.nbclients))
            self.edges = np.zeros((self.nbclients, self.nbclients))
            self.wval = np.zeros(self.nbclients)

            # Read customer data
            for i in range(0, self.nbclients-1):
                data = lines[9 + i].split()

                self.citieslab[i] = int(data[0])  # Customer ID
                self.posx[i] = float(data[1])  # x coordinate
                self.posy[i] = float(data[2])  # y coordinate
                self.d[i] = float(data[3])  # Demand
                self.a[i] = int(data[4])  # Time window start time
                self.b[i] = int(data[5])  # Time window end time
                self.s[i] = int(data[6])  # Service time
                if self.service_in_tw:
                    self.b[i] -= self.s[i]  # If service time is within time window, adjust time window end time
                print(f"Customer {i}: {self.citieslab[i]} {self.posx[i]} {self.posy[i]} {self.d[i]} {self.a[i]} {self.b[i]} {self.s[i]}")

            # Copy depot information to end depot
            self.citieslab[self.nbclients - 1] = self.nbclients - 1
            self.posx[self.nbclients - 1] = self.posx[0]
            self.posy[self.nbclients - 1] = self.posy[0]
            self.d[self.nbclients - 1] = 0.0
            self.a[self.nbclients - 1] = self.a[0]
            self.b[self.nbclients - 1] = self.b[0]
            self.s[self.nbclients - 1] = 0
            print(f'End depot {self.citieslab[self.nbclients - 1]}: {self.citieslab[self.nbclients - 1]} '
                        f'{self.posx[self.nbclients - 1]} {self.posy[self.nbclients - 1]} {self.d[self.nbclients - 1]} '
                        f'{self.a[self.nbclients - 1]} {self.b[self.nbclients - 1]} {self.s[self.nbclients - 1]}')

            # Calculate distance matrix
            for i in range(self.nbclients):
                max = 0
                for j in range(self.nbclients):
                    # truncate to get the same results as in Solomon
                    self.dist_base[i, j] = np.round(
                        10 * np.sqrt((self.posx[i] - self.posx[j]) ** 2 + (self.posy[i] - self.posy[j]) ** 2)) / 10.0
                    if max < self.dist_base[i, j]:
                        max = self.dist_base[i, j]
                self.maxlength += max

            # Set depot to depot distance as infinity
            for i in range(self.nbclients):
                self.dist_base[i, 0] = self.verybig
                self.dist_base[self.nbclients - 1, i] = self.verybig
                self.dist_base[i, i] = self.verybig

            for i in range(self.nbclients):
                for j in range(self.nbclients):
                    self.dist[i, j] = self.dist_base[i, j]

            # Calculate travel time matrix
            for i in range(self.nbclients):
                for j in range(self.nbclients):
                    self.ttime[i, j] = self.dist_base[i, j] / self.speed

            # Other edge costs are given in column generation
            for j in range(self.nbclients):
                self.cost[0][j] = self.dist[0][j]
                self.cost[j][self.nbclients - 1] = self.dist[j][self.nbclients - 1]

            #print(f"[Distance matrix dist initialized]\n{self.dist}")
            #print(f"[Time matrix ttime initialized]\n{self.ttime}")
            #print(f"[Cost matrix cost initialized]\n{self.cost}")

            for i in range(1, self.nbclients):
                self.wval[i] = 0.0
            print(f"[Auxiliary variable wval initialized]\n{self.wval}")

            print(f"----------[ParamsVRP initialization complete]----------")


        except FileNotFoundError:
            print(f"Error: File {input_path} not found.")
        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"Error in init_params: {e}")

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
                f"  maxlength={self.maxlength}\n"
                f")")