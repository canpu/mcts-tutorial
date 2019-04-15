from state import AbstractState
import random
from GamePosition import GamePosition
from Target import Target
from copy import deepcopy
from Environment import *
import matplotlib.pyplot    as plt
from matplotlib.figure import Figure
from matplotlib.lines       import Line2D
from matplotlib.patches     import FancyArrow, Polygon, Circle
import numpy as np


def _assert_position(position: tuple) -> None:
    """ Assert the input is a tuple with two integer elements
    """
    assert len(position) == 2
    assert isinstance(position[0], int) and position[0] > 0 and isinstance(
        position[1], int) and position[1] > 0


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
        return sum(auv.reward for auv in self.auvs)

    @property
    def feasible_next_states(self) -> list:
        auv = self.auvs[self.cur_agent_index]
        i, j = auv.position
        possible_new_states = []
        new_state = deepcopy(self)
        if i > self.environment.x_min:
            new_state.auvs[self.cur_agent_index].move_to(
                GamePosition(i-1, j))
        if i < self.environment.x_max:
            new_state.auvs[self.cur_agent_index].move_to(
                GamePosition(i+1, j))
        if j > self.environment.y_min:
            new_state.auvs[self.cur_agent_index].move_to(
                GamePosition(i, j-1))
        if j < self.environment.y_max:
            new_state.auvs[self.cur_agent_index].move_to(
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
        x = self.position.get_x() + 0.5 - 0.5*dx
        y = self.position.get_y() + 0.5 - 0.5*dy
        return x,y,dx,dy

    @property
    def is_terminal(self) -> bool:
        """ A state is terminal if and only if the time runs out or all rewards
            have been collected
        """
        if self.time <= 0:
            return True
        for target in self.environment.target_set:
            if not target.visited:
                return False
        return True

    def gen_random_walk(self) -> "GameState":
        """ Generates a random walk (until final time reached) with a bias
            towards targets that are not visited
        """
        while self.time < self.environment.final_time:
            for auv in self.auvs:
                auv.take_random_step(self.environment)
            self.time += 1
        return self

    def visualize(self, file_name=None, fig_size: (float, float) = (6.5, 6.5),
                  size_auv_path: float = 0.8, size_max_radius: float = 0.3,
                  size_arrow_h_width: float = 0.4,
                  size_arrow_h_length: float = 0.3,
                  size_arrow_width: float = 0.4,
                  color_obstacle: str = 'firebrick',
                  color_target: str = 'deepskyblue',
                  color_auv: str = 'darkorange',
                  color_auv_path: str = 'peachpuff') -> Figure:

        if (fig_size[0] <= 0 or fig_size[1] <= 0 or size_auv_path <= 0 or
                size_max_radius <= 0 or size_arrow_h_width <= 0 or
                size_arrow_h_length <= 0 or size_arrow_width <= 0):
            raise ValueError("Size must be positive")
        reference_length = max(self.environment.x_range,
                               self.environment.y_range)
        max_value = self.environment.max_value
        size_buffer = math.floor(0.1 * reference_length)
        title_font = {'fontname': 'Sans Serif', 'size': '16', 'color': 'black',
                      'weight': 'bold'}
        
        # Initialize the figure
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        # depth order for plotting
        z = {'auv_path': 1, 'target': 2, 'obstacle': 3, 'auv': 5}

        # Plot obstacles
        for obstacle in self.environment.get_obstacles():
            ax.add_patch(Polygon(xy=np.array(obstacle.get_bbox()),
                                 color=color_obstacle, zorder=z['obstacle']))

        # Plot targets
        for target in self.environment.get_targets():
            target_opacity = target.get_opacity()
            target_radius = (target.get_value() / max_value) * size_max_radius
            ax.add_patch(Circle(xy=target.get_centroid(), radius=target_radius,
                                color=color_target, alpha=target_opacity,
                                zorder=z['target']))

        # Plot auvs
        for auv in self.auvs:
            x, y, dx, dy = auv.arrow_params
            ax.add_patch(FancyArrow(x=x, y=y, dx=dx, dy=dy, fc=color_auv,
                                    width=size_arrow_width,
                                    head_width=size_arrow_h_width,
                                    head_length=size_arrow_h_length,
                                    zorder=z['auv'], length_includes_head=True))
            # plot trajectories
            for i in range(1, len(auv.path)):
                xdata, ydata = auv.get_rectangle_params(i)
                ax.add_line(Line2D(xdata=xdata, ydata=ydata,
                                   linewidth=size_auv_path * 10,
                                   color=color_auv_path, zorder=z['auv_path']))

        # Plotting
        plt.title('AUV Trajectory \n Accumulated Reward: ' + str(self.reward),
                  title_font)
        plt.xlabel('x', title_font)
        plt.ylabel('y', title_font)
        plt.xticks(np.arange(self.environment.x_min, self.environment.x_max + 2,
                             1.0),
                   ['' for _ in np.arange(self.environment.x_min,
                                          self.environment.x_max + 2, 1.0)])
        plt.yticks(np.arange(self.environment.y_min, self.environment.y_max + 2
                             , 1.0),
                   ['' for _ in np.arange(self.environment.y_min,
                                          self.environment.y_max + 2, 1.0)])
        ax.grid()
        ax.axis('equal')
        ax.set_xlim(self.environment.x_min - size_buffer,
                    self.environment.x_max + size_buffer + 1)
        ax.set_ylim(self.environment.y_min - size_buffer,
                    self.environment.y_max + size_buffer + 1)

        # Save and display
        plt.show()
        if file_name is not None:
            plt.savefig(file_name)

        return fig
