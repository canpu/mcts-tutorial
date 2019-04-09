import numpy as np
import random

class AUV():
    def __init__(self, position, time_remaining):
        """
        Represents an AUV object navigating through the environment
        :param position: the initial position of the AUV
        :param time_remaining: the time remaining until the end of the "game" 
        """
        self.position       = position
        self.orientation    = 0
        self.path           = [position]
        self.total_value    = 0


    def get_position(self):
        """
        :return: the position of the AUV
        """
        return(self.position)


    def get_time_remaining(self):
        """
        :return: the time_remaining of the AUV
        """
        return(self.time_remaining)        


    def get_path(self):
        """
        :return: the path of the AUV
        """
        return([position for position in self.path])


    def get_orientation(self):
        """
        :return: the orientation of the AUV
        """
        return(self.orientation)


    def get_arrow_params(self):
        """
        :return: the parameters associated with drawing an arrow with mpl
            + base of arrow drawn at (x, y), head of arrow drawn at (x+dx, y+dy)
        """
        dx = np.cos(self.orientation)
        dy = np.sin(self.orientation)
        x  = self.position.get_x() + 0.5 - 0.5*dx
        y  = self.position.get_y() + 0.5 - 0.5*dy
        return(x,y,dx,dy)


    def get_rectangle_params(self, index):
        """
        :param index: the index of the orientation list that must be plotted
        :return: the parameters associated with drawing an rectangle with mpl
        """
        unit_len = 0.5
        point1 = self.path[index-1].get_centroid()
        point2 = self.path[index].get_centroid()
        xdata  = (point2[0], point1[0])
        ydata  = (point2[1], point1[1])
        return(xdata, ydata)


    def get_possible_positions(self, environment):
        """
        :return: a list of possible actions from the current state
        """
        possible_positions = []
        if (self.position.get_pos_right() not in environment.get_obstacles()):
            possible_positions.append(self.position.get_pos_right())
        if (self.position.get_pos_left() not in environment.get_obstacles()):
            possible_positions.append(self.position.get_pos_left())
        if (self.position.get_pos_above() not in environment.get_obstacles()):
            possible_positions.append(self.position.get_pos_above())
        if (self.position.get_pos_below() not in environment.get_obstacles()):
            possible_positions.append(self.position.get_pos_below())
        return(possible_positions)


    def move_to_position(self, new_position, environment):
        """
        moves the AUV from current location to a new position
        """
        # update the vehicle orientation
        if (new_position == self.position.get_pos_right()):
            self.orientation = 0
        elif (new_position == self.position.get_pos_left()):
            self.orientation = np.pi
        elif (new_position == self.position.get_pos_above()):
            self.orientation = np.pi/2
        elif (new_position == self.position.get_pos_below()):
            self.orientation = -np.pi/2

        # update the vehicle position and path
        self.position = new_position 
        self.path.append(new_position)

        # retrieve the reward if the new position is a target
        return(environment.visit_target(new_position))


    def take_random_step(self, environment):
        """
        takes a random step from the given position, biased towards new targets
        :param environment: the current state of the environment 
        """
        possible_positions = self.get_possible_positions(environment)
        if len(possible_positions) > 0:
            return(self.move_to_position(random.choice(possible_positions), 
                                         environment))
        else:
            return(0)

