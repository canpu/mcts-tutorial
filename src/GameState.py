from state import AbstractState
import math
import random
from GamePosition import GamePosition
from Target import Target
from copy import deepcopy


def _assert_position(position: tuple) -> None:
    """ Assert the input is a tuple with two integer elements
    """
    assert len(position) == 2
    assert isinstance(position[0], int) and position[0] > 0 and isinstance(
        position[1], int) and position[1] > 0


class AUV:
    def __init__(self, position):
        """
        Represents an AUV object navigating through the environment
        :param position: the initial position of the AUV
        """
        self.position = position
        self.orientation = 0
        self.path = [position]
        self.total_value = 0

    def get_position(self):
        """
        :return: the position of the AUV
        """
        return self.position

    def get_orientation(self):
        """
        :return: the orientation of the AUV
        """
        return self.orientation

    def get_path(self):
        """
        :return: the path of the AUV
        """
        return [position for position in self.path]

    def get_orientation_arrow_params(self):
        """ Get the parameters associated with drawing an arrow with
            matplotlib + base of arrow drawn at (x, y), head of arrow drawn
            at (x+dx, y+dy)
        :return: x, y, dx, dy
        """
        x, y = self.position.x, self.position.y
        dx, dy = 0, 0
        if self.orientation == 0:
            x = self.position.get_x()
            y = self.position.get_y() + 0.5
            dx = 1
            dy = 0
        elif self.orientation == 90:
            x = self.position.get_x() + 0.5
            y = self.position.get_y()
            dx = 0
            dy = 1
        elif self.orientation == 180:
            x = self.position.get_x()
            y = self.position.get_y() + 0.5
            dx = -1
            dy = 0
        elif self.orientation == 270:
            x = self.position.get_x() + 0.5
            y = self.position.get_y()
            dx = 0
            dy = -1
        return x, y, dx, dy

    # def take_random_step(self, environment):
    #     """
    #     takes a random step from the given position, biased towards new targets
    #     :param environment: the current state of the environment
    #     """
    #     possible_positions = self.get_possible_positions(environment)
    #     if len(possible_positions) > 0:
    #         return(self.move_to_position(random.choice(possible_positions),
    #                                      environment))
    #     else:
    #         return(0)

class Environment:
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
        self.x_min = environment_bbox[0]
        self.x_max = environment_bbox[1]
        self.y_min = environment_bbox[2]
        self.y_max = environment_bbox[3]
        self.size = (self.x_max - self.x_min, self.y_max - self.y_min)
        self.x_range = self.x_max + 1 - self.x_min
        self.y_range = self.y_max + 1 - self.y_min
        self.obstacle_set = obstacle_set
        self.target_set = target_set
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
        return (self.obstacle_set)

    @property
    def targets(self):
        """
        :return: the set of targets in the environment
        """
        return (self.target_set)

    def get_max_value(self):
        """
        :return: the maximum value out of all the targets in the environment
        """
        if (len(self.target_set) == 0):
            return (1)
        else:
            return (max(max([t.get_value() for t in self.target_set]), 1))

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
                               value_range=[1, 3], border=True):
        """
        generates a random environment obstacles and targets
        :param obstacle_coverage: the desired percent coverage for obstacles
        :param target_coverage: the desired percent coverage for targets
        :param value_range: the range of possible target values
        :param border: flag that creates explicit border of obstacles
        """
        # disregard the previously set obstacles and targets
        self.obstacle_set = set([])
        self.target_set = set([])

        # generate a border of obstacles if necessary
        if border:
            self.gen_obstacle_border()

        # randomly add obstacles
        num_samples = int(self.x_range * self.y_range * obstacle_coverage)
        for i in range(num_samples):
            rand_x = random.randrange(self.x_min, self.x_max + 1)
            rand_y = random.randrange(self.y_min, self.y_max + 1)
            self.add_obstacle(GamePosition(rand_x, rand_y))

        # randomly add target values
        for i in range(1, self.x_range):
            for j in range(1, self.y_range):
                if GamePosition(i, j) not in self.obstacle_set:
                    if random.random() < target_coverage:
                        rand_val = random.randrange(value_range[0],
                                                    value_range[1] + 1)
                        self.add_target(Target(i, j, rand_val))

        # adjust the maximum value target
        self.max_value = self.get_max_value()


class GameState():
    def __init__(self, auvs: list, environment: Environment, time: int = 0):
        """ Create a state of the game where the AUVs explore the environment to
            collect rewards
        :param auvs: the list of AUV objects operating in environment
        :param environment: the Environment object that encodes obstacles
        :param time: the time elapsed from the start of the game
        """
        assert time >= 0
        self.auvs = auvs
        self.environment = environment
        self.time = time
        self.cur_agent_index = 0

    @property
    def reward(self) -> float:
        return sum(auv.total_value for auv in self.auvs)

    @property
    def feasible_next_states(self) -> list:
        auv = self.auvs[self.cur_agent_index]
        i, j = auv.position
        possible_new_states = []
        new_state = deepcopy(self)
        if i > self.environment.x_min:
            new_state.auvs[self.cur_agent_index].move_to_position(
                GamePosition(i-1, j))
        if i < self.environment.x_max:
            new_state.auvs[self.cur_agent_index].move_to_position(
                GamePosition(i+1, j))
        if j > self.environment.y_min:
            new_state.auvs[self.cur_agent_index].move_to_position(
                GamePosition(i, j-1))
        if j < self.environment.y_max:
            new_state.auvs[self.cur_agent_index].move_to_position(
                GamePosition(i, j+1))
        new_state.cur_agent_index += 1
        new_state.cur_agent_index %= len(self.auvs)
        possible_new_states.append(new_state)
        return possible_new_states

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

    @property
    def is_terminal(self) -> bool:
        if self.time <= 0:
            return True
        for target in self.environment.target_set:
            if not target.visited:
                return False
        return True

    # def gen_random_walk(self):
    #     """
    #     generates a random walk with a bias towards targets that are not visited
    #     """
    #     while self.time < self.environment.final_time:
    #         for auv in self.auvs:
    #             auv.take_random_step(self.environment)
    #         self.time += 1
