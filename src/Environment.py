import math
import random

from GamePosition import GamePosition
from Target       import Target


class Environment():
    def __init__(self, environment_bbox, obstacle_set=set([]), 
                 target_set=set([]), default_value=1, border=True):
        """
        Represents the environment that the agents are operating in
        :param environment_bbox: the bounding box [x_min, x_max, y_min, y_max]
        :param obstacle_set: the set of obstacles that must be avoided
        :param target_set: the set of targets in the environment
        :param default_value: the default value of non-obstacle spaces
        :param border: flag that creates explicit border of obstacles 
        :requires: obstacle_set and target_set are disjoint sets
            + obstacles take precedence over targets 
        """
        self.x_min        = environment_bbox[0]
        self.x_max        = environment_bbox[1]
        self.y_min        = environment_bbox[2]
        self.y_max        = environment_bbox[3]
        self.bbox         = environment_bbox
        self.x_range      = self.x_max+1 - self.x_min
        self.y_range      = self.y_max+1 - self.y_min
        self.obstacle_set = obstacle_set
        self.target_set   = target_set
        self.default_val  = default_value
        self.border       = border
        self.final_time = 10

        # generate the border of obstacles as necessary 
        if border: 
            self.gen_obstacle_border()

        # remove target regions that were defined within obstacles
        for position in obstacle_set:
            if position in target_set:
                self.target_set.remove(position)

        # compute the maximum value in the environment (used for plotting)
        self.max_value = self.get_max_value()


    def get_obstacles(self):
        """
        :return: the set of obstacles in the environment
        """
        return(self.obstacle_set)


    def get_targets(self):
        """
        :return: the set of targets in the environment
        """
        return(self.target_set)


    def get_max_value(self):
        """
        :return: the maximum value out of all the targets in the environment
        """
        if (len(self.target_set) == 0):
            return(1)
        else:
            return(max(max([t.get_value() for t in self.target_set]), 1))


    def add_obstacle(self, obstacle):
        """
        adds a new obstacle to the set of obstacles in the environment
        :param obstacle: the obstacle to be added to the environment
        """
        self.obstacle_set.add(obstacle)


    def rem_obstacle(self, obstacle):
        """
        removes an obstacle from the environment 
        :param obstacle: the obstacle to be removed from the environment
        """
        if obstacle in self.obstacle_set:
            self.obstacle_set.remove(obstacle)


    def add_target(self, target):
        """
        adds a new target to the set of targets in the environment
        :param target: the target to be added to the environment
        """
        self.max_value = max(self.max_value, target.get_value())
        if (target in self.target_set):
            self.target_set.remove(target)
            self.target_set.add(target)
        else:
            self.target_set.add(target)


    def rem_target(self, target):
        """
        removes an target from the environment 
        :param target: the target to be removed from the environment
        """
        if target in self.target_set:
            self.target_set.remove(target)


    def visit_target(self, position):
        """
        sets the target located at position to visited  
        :param position: the position of the target that is visited
        :return: the value received for visiting the target
        """
        for target in self.target_set:
            if (position == target):
                if (not target.visited):
                    target.visit()
                    return(target.get_value())
        return(0)


    def gen_obstacle_border(self):
        """
        generates a border of obstacles given the environment bounding box
        """
        # add left and right border obstacles 
        for row in range(1, self.y_range):
            self.add_obstacle(GamePosition(self.x_min, self.y_min + row))
            self.add_obstacle(GamePosition(self.x_max, self.y_min + row))

        # add top and bottom border obstacles
        for col in range(self.x_range):
            self.add_obstacle(GamePosition(self.x_min + col, self.y_max))
            self.add_obstacle(GamePosition(self.x_min + col, self.y_min))


    def gen_random_environment(self, obstacle_coverage=0.2, target_coverage=0.8, 
                               value_range=[1,3], border=True):
        """
        generates a random environment obstacles and targets
        :param obstacle_coverage: the desired percent coverage for obstacles
        :param target_coverage: the desired percent coverage for targets 
        :param value_range: the range of possible target values
        :param border: flag that creates explicit border of obstacles 
        """
        # disregard the previously set obstacles and targets
        self.obstacle_set = set([])
        self.target_set   = set([])

        # generate a border of obstacles if necessary
        if border: 
            self.gen_obstacle_border()

        # randomly add obstacles 
        num_samples = math.floor(self.x_range*self.y_range*obstacle_coverage)
        for i in range(num_samples):
            rand_x = random.randrange(self.x_min, self.x_max+1)
            rand_y = random.randrange(self.y_min, self.y_max+1)
            self.add_obstacle(GamePosition(rand_x, rand_y))

        # randomly add target values
        for i in range(1, self.x_range):
            for j in range(1, self.y_range):
                if GamePosition(i,j) not in self.obstacle_set:
                    if random.random() < target_coverage:
                        rand_val = random.randrange(value_range[0], 
                                                    value_range[1]+1)
                        self.add_target(Target(i,j,rand_val))

        # adjust the maximum value target                 
        self.max_value = self.get_max_value()

