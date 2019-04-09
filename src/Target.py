from GamePosition import GamePosition

class Target(GamePosition):
    def __init__(self, x, y, value, visited=False):
        """
        Represents an area that provides value if visited by an agent
        :param x: the x coordinate of the game position
        :param y: the y coordinate of the game position
        :param value: the value that is received for visiting this position
        :param visited: flag that determines if the target has been visited yet
        """
        super().__init__(x, y)
        self.value     = value
        self.visited   = visited


    def visit(self):
        """
        Sets the visited to flag to true upon agent visiting this location 
        """
        self.visited = True


    def get_value(self):
        """
        :return: the value of the target
        """
        return(self.value)


    def get_opacity(self):
        """
        :return: the opacity of the target when plotting
        """
        if (not self.visited):  return(1.0)
        else:                   return(0.15)
