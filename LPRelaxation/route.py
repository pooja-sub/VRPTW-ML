import copy

class Route:
    def __init__(self, path=None, cost=0.0, Q=0.0):
        """
        Initialize the Route class.

        :param path: Path list, default is None, represents an empty path
        :param cost: Path cost, default is 0.0
        :param Q: Other resources of the path (e.g., flow), default is 0.0
        """
        self.path = path if path is not None else []
        self.cost = cost
        self.Q = Q

    def clone(self):
        """
        Deep copy the current route object.
        """
        return copy.deepcopy(self)

    def remove_city(self, city):
        """
        Remove a specified city from the path.

        :param city: City number to remove
        """
        if city in self.path:
            self.path.remove(city)

    def add_city(self, city, after_city=None):
        """
        Add a city to the path.

        :param city: City number to add
        :param after_city: After which city to add (optional)
        """
        if after_city is None:
            self.path.append(city)
        else:
            index = self.path.index(after_city)
            self.path.insert(index + 1, city)

    def set_cost(self, cost):
        """
        Set the path cost.

        :param cost: New cost value
        """
        self.cost = cost

    def get_cost(self):
        """
        Get the path cost.
        """
        return self.cost

    def set_Q(self, Q):
        """
        Set other resources of the path (e.g., flow).

        :param Q: New resource value
        """
        self.Q = Q

    def get_Q(self):
        """
        Get other resources of the path (e.g., flow).
        """
        return self.Q

    def get_path(self):
        """
        Get the path list.
        """
        return self.path

    def switch_path(self):
        """
        Reverse the path.
        """
        self.path = self.path[::-1]

    def __str__(self):
        """
        Return the string representation of the path.
        """
        return f"Route(cost={self.cost}, Q={self.Q}, path={self.path})"

    def __repr__(self):
        """
        Return the short string representation of the path.
        """
        return self.__str__()