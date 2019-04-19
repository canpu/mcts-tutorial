from state import *
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle, Circle
import numpy as np
from copy import deepcopy


class MazeAction(AbstractAction):
    def __init__(self, agent_index: int, position: (int, int)):
        self._agent_index = agent_index
        self._position = position

    @property
    def agent_index(self) -> int:
        return self._agent_index

    @property
    def position(self) -> (int, int):
        return self._position

    def __eq__(self, other) -> bool:
        return (self.__class__ == other.__class__ and
                self.agent_index == other.agent_index and
                self.position == other.position)

    def __hash__(self) -> int:
        return hash(tuple([self._agent_index] + list(self._position)))

    def __str__(self) -> str:
        return "Action: Agent {0} moves to {1}".format(self.agent_index,
                                                       self.position)


class MazeEnvironment:
    def __init__(self, xlim: (int, int) = (0, 1), ylim: (int, int) = (0, 1),
                 obstacles: set = None, targets: dict = None,
                 is_border_obstacle_filled: bool = True):
        """ Create an Environment object in which the AUV problem is defined
        """
        self._x_min, self._x_max = xlim
        self._y_min, self._y_max = ylim
        self._obstacles = obstacles if obstacles else set()
        self._targets = targets if targets else {}

        # Make border obstacles if specified
        if is_border_obstacle_filled:
            for j in range(self.y_min, self.y_max + 1):
                self.add_obstacle((self.x_min, j)).add_obstacle((self.x_max, j))
            for i in range(self.x_min, self.x_max + 1):
                self.add_obstacle((i, self.y_min)).add_obstacle((i, self.y_max))

        # Remove target regions that were defined within obstacles
        for position in self._obstacles:
            if position in self._targets.keys():
                del self._targets[position]

    @property
    def x_min(self) -> int:
        return self._x_min

    @property
    def y_min(self) -> int:
        return self._y_min

    @property
    def x_max(self) -> int:
        return self._x_max

    @property
    def y_max(self) -> int:
        return self._y_max

    @property
    def max_reward(self) -> float:
        if self._targets:
            return max(self._targets.values())
        else:
            return 0.0

    @property
    def x_range(self) -> int:
        return self._x_max + 1 - self._x_min

    @property
    def y_range(self) -> int:
        return self._y_max + 1 - self._y_min

    @property
    def obstacles(self) -> set:
        return deepcopy(self._obstacles)

    @property
    def rewards(self) -> dict:
        return deepcopy(self._targets)

    def add_obstacle(self, obstacle_position: (int, int)) -> "MazeEnvironment":
        if obstacle_position not in self._targets:
            self._obstacles.add(obstacle_position)
        return self

    def remove_obstacle(self, obstacle_position: (int, int)) \
            -> "MazeEnvironment":
        if obstacle_position in self._obstacles:
            self._obstacles.remove(obstacle_position)
            return self

    def add_reward(self, position: (int, int), reward: float) \
            -> "MazeEnvironment":
        if position not in self._obstacles:
            self._targets[position] = reward
        return self

    def remove_reward(self, position: (int, int)) -> "MazeEnvironment":
        if position in self._targets:
            del self._targets[position]
        return self

    def __eq__(self, other) -> bool:
        return (self.__class__ == other.__class and self.x_max == other.x_max
                and self.x_min == other.x_min and self.y_min == other.y_min and
                self.y_max == other.y_max and
                self._obstacles == other.obstacles and
                self._targets == other.rewards)


def gen_random_environment(xlim: (int, int) = (0, 10),
                           ylim: (int, int) = (0, 10),
                           obstacle_coverage: float = 0.2,
                           target_coverage: float = 0.2,
                           reward_range: (float, float) = (1.0, 3.0),
                           is_border_obstacle_filled: bool = True):
    if not (obstacle_coverage >= 0 and target_coverage >= 0 and
            obstacle_coverage + target_coverage <= 1):
        raise ValueError("The probability is not valid")

    env = MazeEnvironment(xlim, ylim,
                          is_border_obstacle_filled=is_border_obstacle_filled)

    # Generate obstacles and targets by the given probability
    if obstacle_coverage > 0 or target_coverage > 0:
        for i in range(ylim[0], ylim[1] + 1):
            for j in range(xlim[0], xlim[1] + 1):
                if (i, j) not in env.obstacles:
                    r = random.random()
                    if r <= obstacle_coverage:
                        env.add_obstacle((i, j))
                    elif r <= obstacle_coverage + target_coverage:
                        env.add_reward((i, j),
                                       random.randrange(reward_range[0],
                                                        reward_range[1]))
    return env


class MazeState(AbstractState):
    def __init__(self, environment: MazeEnvironment, time_remains: int = 10):
        """ Create a state of the AUV reward-collection game
        """
        if time_remains < 0:
            raise ValueError("The remaining time cannot be negative")
        self._paths = []
        self._environment = environment
        self._time_remains = time_remains
        self._turn = 0  # The index of which agent should move next

    def __copy__(self) -> "MazeState":
        """ Deep copy does not apply to the Environment object because
            it is supposed to be static
        """
        copy = MazeState(self._environment)
        copy._time_remains = self._time_remains
        copy._paths = deepcopy(self._paths)
        copy._turn = self._turn
        return copy

    def is_in_range(self, position: (int, int)) -> bool:
        return (self._environment.x_min <= position[0] <=
                self._environment.x_max and self._environment.y_min
                <= position[1] <= self._environment.y_max)

    def add_agent(self, position: (int, int)) -> "MazeState":
        if (self.is_in_range(position) and
                position not in self._environment.obstacles):
            self._paths.append([position])
        else:
            raise ValueError("The given position is invalid")
        return self

    @property
    def paths(self) -> list:
        return self._paths

    @property
    def environment(self) -> MazeEnvironment:
        return self._environment

    @property
    def visited(self) -> set:
        visited = []
        for path in self._paths:
            visited += path
        return set(visited)

    @property
    def reward(self) -> float:
        reward = 0.0
        for target_position in self._environment.rewards:
            if target_position in self.visited:
                reward += self._environment.rewards[target_position]
        return reward

    def switch_agent(self) -> "MazeState":
        """ After the movement of one agent, it would be the turn of the next
            agent
        """
        self._turn = (self._turn + 1) % len(self._paths)
        if self._turn == 0:  # When all agents have taken a turn of actions
            self._time_remains -= 1
        return self

    @property
    def time_remains(self) -> int:
        return self._time_remains

    @property
    def is_terminal(self) -> bool:
        """ A state is terminal if and only if the time runs out
        """
        return self._time_remains <= 0

    @property
    def turn(self) -> int:
        return self._turn

    def execute_action(self, action: MazeAction) -> "MazeState":
        """ Make a copy of the current state, execute the action and return the
            new state
        :param action: The action
        :return: A copy of the new state
        """
        new_state = self.__copy__()
        new_state.paths[new_state.turn].append(action.position)
        new_state.switch_agent()
        return new_state

    @property
    def possible_actions(self) -> list:
        i, j = self._paths[self._turn][-1]
        actions = [MazeAction(self._turn, (i + 1, j)),
                   MazeAction(self._turn, (i - 1, j)),
                   MazeAction(self._turn, (i, j + 1)),
                   MazeAction(self._turn, (i, j - 1))]
        return [action for action in actions if
                self.is_in_range(action.position) and
                action.position not in self._environment.obstacles]

    def visualize(self, file_name=None, fig_size: (float, float) = (6.5, 6.5),
                  size_auv_path: float = 0.8, size_max_radius: float = 0.3,
                  size_min_radius: float = 0.1,
                  tick_size: float = 14, grid_width: float = 0.25,
                  size_arrow_h_width: float = 0.4,
                  size_arrow_h_length: float = 0.3,
                  size_arrow_width: float = 0.4,
                  color_obstacle: str = 'firebrick',
                  color_target: str = 'deepskyblue',
                  color_auv: str = 'darkorange',
                  color_auv_path: str = 'peachpuff',
                  visited_reward_opacity: float = 0.15) -> Figure:

        if (fig_size[0] <= 0 or fig_size[1] <= 0 or size_auv_path <= 0 or
                size_max_radius <= 0 or size_arrow_h_width <= 0 or
                size_arrow_h_length <= 0 or size_arrow_width <= 0 or
                tick_size <= 0 or grid_width <= 0):
            raise ValueError("Size must be positive")
        max_reward = self._environment.max_reward
        title_font = {'fontname': 'Sans Serif', 'size': '16', 'color': 'black',
                      'weight': 'bold'}
        z = {'auv_path': 1, 'target': 2, 'obstacle': 3, 'auv': 5}

        # Initialize the figure
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        plt.hlines(y=range(self._environment.y_min, self._environment.y_max + 1)
                   , xmin=self._environment.x_min, xmax=self._environment.x_max,
                   color='k', linewidth=grid_width, zorder=0)
        plt.vlines(x=range(self._environment.x_min, self._environment.x_max + 1)
                   , ymin=self._environment.y_min, ymax=self._environment.y_max,
                   color='k', linewidth=grid_width, zorder=0)

        # Plot obstacles
        for i, j in self._environment.obstacles:
            ax.add_patch(Rectangle(xy=(i, j), width=1, height=1,
                                   color=color_obstacle, zorder=z['obstacle']))

        # Plot rewards
        for position, reward in self._environment.rewards.items():
            target_radius = ((reward / max_reward)
                             * (size_max_radius - size_min_radius)
                             + size_min_radius)
            centroid = (position[0] + 0.5, position[1] + 0.5)
            ax.add_patch(Circle(xy=centroid, radius=target_radius,
                                color=color_target, zorder=z['target'],
                                alpha=(visited_reward_opacity
                                       if position in self.visited else 1.0)))

        # Plot agents
        for path in self._paths:
            x, y = path[-1]
            dx, dy = 0, 1
            if len(path) >= 2:
                x_p, y_p = path[-2]
                if x == x_p + 1 and y == y_p:
                    dx, dy = 1, 0
                elif x == x_p - 1 and y == y_p:
                    dx, dy = -1, 0
                elif x == x_p and y == y_p - 1:
                    dx, dy = 0, -1
            x += 0.5 * float(1 - dx)
            y += 0.5 * float(1 - dy)
            ax.add_patch(FancyArrow(x=x, y=y, dx=dx, dy=dy, fc=color_auv,
                                    width=size_arrow_width,
                                    head_width=size_arrow_h_width,
                                    head_length=size_arrow_h_length,
                                    zorder=z['auv'], length_includes_head=True))

            # plot trajectories
            for i in range(1, len(path)):
                x, y = path[i]
                x_p, y_p = path[i - 1]
                ax.add_line(Line2D(xdata=(x + 0.5, x_p + 0.5),
                                   ydata=(y + 0.5, y_p + 0.5),
                                   linewidth=size_auv_path * 10,
                                   color=color_auv_path, zorder=z['auv_path']))



        # Plotting
        plt.title('AUV Trajectory \n Accumulated Reward: ' + str(self.reward),
                  title_font)
        plt.xlabel('x', title_font)
        plt.ylabel('y', title_font)
        x_ticks = np.arange(self._environment.x_min, self._environment.x_max + 1
                            , 1)
        y_ticks = np.arange(self._environment.y_min, self._environment.y_max + 1
                            , 1)
        plt.xticks(x_ticks + 0.5, x_ticks.astype(int))
        plt.yticks(y_ticks + 0.5, y_ticks.astype(int))
        ax.tick_params(labelsize=tick_size)
        ax.grid(False)
        ax.axis('equal')
        ax.set_xlim(self._environment.x_min - 0.5,
                    self._environment.x_max + 1.5)
        ax.set_ylim(self._environment.y_min - 0.5,
                    self._environment.y_max + 1.5)

        # Save and display
        plt.show()
        if file_name is not None:
            plt.savefig(file_name)

        return fig
