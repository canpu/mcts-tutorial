class GamePosition():
    def __init__(self, x, y):
        """
        Represents a location within the environment 
        :param x: the x coordinate of the position
        :param y: the y coordinate of the position
        """
        self.x = x
        self.y = y


    def __eq__(self, other):
        """
        Defines equality for a GamePosition object
        """
        if isinstance(other, GamePosition): 
            if (self.get_xy() == other.get_xy()):
                return(True)
        else: 
            return (False)


    def __hash__(self):
        """
        Defines the hash function for the GamePosition object
        """
        return(hash(self.get_x()) + hash(self.get_y()))


    def get_x(self):
        """
        :return: the x coordinate of the position
        """
        return(self.x)  


    def get_y(self):
        """
        :return: the y coordinate of the position
        """
        return(self.y)


    def get_xy(self):
        """
        :return: the position of the object
        """
        return(self.x, self.y)  


    def get_pos_right(self):
        """
        :return: the position to the right of the current position
        """
        return(GamePosition(self.x+1, self.y))


    def get_pos_left(self):
        """
        :return: the position to the left of the current position
        """
        return(GamePosition(self.x-1, self.y))  


    def get_pos_above(self):
        """
        :return: the position above the current position
        """
        return(GamePosition(self.x, self.y+1)) 


    def get_pos_below(self):
        """
        :return: the position below the current position
        """
        return(GamePosition(self.x, self.y-1))


    def get_bbox(self):
        """
        :return: the x coordinate of the position
        """
        return([(self.x,   self.y), 
                (self.x+1, self.y),
                (self.x+1, self.y+1),
                (self.x,   self.y+1)])


    def get_centroid(self):
        """
        :return: the centroid of the game position
        """
        return(self.x+0.5, self.y+0.5)  
