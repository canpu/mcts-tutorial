import math
import random
from GamePosition import GamePosition
from copy import deepcopy
from state import AbstractAction


class AUV:
    def __init__(self, position: GamePosition, time_remaining):
        """ Create an AUV object navigating through the environment
        :param position: the initial position of the AUV
        :param time_remaining: the time remaining until the end of the "game" 
        """
        self._position = position
        self._orientation = 0
        self._path = [position]
        self._reward = 0

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def position(self) -> GamePosition:
        return self._position

    @property
    def time_remaining(self) -> int:
        return self.time_remaining

    @property
    def path(self) -> list:
        return deepcopy(self._path)

    @property
    def orientation(self):
        return self._orientation

    @property
    def arrow_params(self) -> (float, float, float, float):
        """ The parameters associated with drawing an arrow with matplotlib
            Base of arrow drawn at (x, y), head of arrow drawn at (x+dx, y+dy)
        """
        dx = math.cos(self._orientation)
        dy = math.sin(self._orientation)
        x = self._position.get_x() + 0.5 - 0.5 * dx
        y = self._position.get_y() + 0.5 - 0.5 * dy
        return(x,y,dx,dy)

    def get_rectangle_params(self, index) -> ((float, float), (float, float)):
        """
        :param index: the index of the orientation list that must be plotted
        :return: the parameters associated with drawing an rectangle with mpl
        """
        point1 = self.path[index-1].get_centroid()
        point2 = self.path[index].get_centroid()
        xdata = (point2[0], point1[0])
        ydata = (point2[1], point1[1])
        return xdata, ydata

    def get_possible_positions(self, environment):
        """
        :return: a list of possible actions from the current state
        """
        possible_positions = []
        if (self._position.get_pos_right() not in environment.get_obstacles()):
            possible_positions.append(self._position.get_pos_right())
        if (self._position.get_pos_left() not in environment.get_obstacles()):
            possible_positions.append(self._position.get_pos_left())
        if (self._position.get_pos_above() not in environment.get_obstacles()):
            possible_positions.append(self._position.get_pos_above())
        if (self._position.get_pos_below() not in environment.get_obstacles()):
            possible_positions.append(self._position.get_pos_below())
        return possible_positions

    def move_to(self, new_position, environment):
        """ The AUV moves to a new position and collects the reward
        """
        # update the vehicle orientation
        if new_position == self._position.get_pos_right():
            self._orientation = 0
        elif new_position == self._position.get_pos_left():
            self._orientation = math.pi
        elif new_position == self._position.get_pos_above():
            self._orientation = math.pi / 2
        elif new_position == self._position.get_pos_below():
            self._orientation = -math.pi / 2

        # update the vehicle position and path
        self._position = new_position
        self._path.append(new_position)

        # retrieve the reward if the new position is a target
        return environment.visit_target(new_position)

    def take_random_step(self, environment):
        """ Takes a random step from the given position,
            biased towards new targets
        :param environment: the current state of the environment 
        """
        possible_positions = self.get_possible_positions(environment)
        if len(possible_positions) > 0:
            return self.move_to(random.choice(possible_positions),
                                environment)
        else:
            return 0

