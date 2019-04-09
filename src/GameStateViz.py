import numpy as np
import math
import matplotlib
import matplotlib.pyplot    as plt
from matplotlib.lines       import Line2D
from matplotlib.patches     import FancyArrow, Polygon, Circle
from matplotlib.collections import PatchCollection


class GameStateViz():
    def __init__(self, game_state):
        """
        Object that can create visualizations of the game state
        :param game_state: the Game State to be visualized
        """
        self.game_state          = game_state

        # plotting parameters
        self.color_obstacle      = 'firebrick'
        self.color_target        = 'deepskyblue'
        self.color_auv           = 'darkorange'
        self.color_auv_path      = 'peachpuff'
        self.reference_length    = max(self.game_state.environment.x_range, 
                                       self.game_state.environment.y_range)
        self.max_value           = self.game_state.environment.max_value
        self.size_edge           = 3
        self.size_buffer         = math.floor(0.1 * self.reference_length)
        self.size_arrow_width    = 0.4
        self.size_arrow_h_width  = 0.4
        self.size_arrow_h_length = 0.3
        self.size_auv_path       = 0.8
        self.size_max_radius     = 0.3
        self.figsize             = (6.5,6.5)
        self.title_font          = {'fontname':'Arial',
                                    'size':'16',     
                                    'color':'black', 
                                    'weight':'bold'}


    def plot(self, filename=None):
        """
        plots the AUV(s) trajectory, environment obstacles, and target regions
        :param filename: name of the file to be saved after plot is generated
            + if filename is not specified, plot is displayed
            + otherwise the file is saved to the given filename
        """
        # initialize the figure
        fig = plt.figure(figsize=self.figsize)
        ax  = fig.add_subplot(111)

        # extract the min and max information from the environment
        x_min = self.game_state.environment.x_min
        x_max = self.game_state.environment.x_max
        y_min = self.game_state.environment.y_min
        y_max = self.game_state.environment.y_max

        # depth order for plotting
        z_auv_path = 1
        z_target   = 2
        z_obstacle = 3
        z_auv      = 5

        # add the obstacles 
        for obstacle in self.game_state.environment.get_obstacles():
            ax.add_patch(Polygon(xy     = np.array(obstacle.get_bbox()), 
                                 color  = self.color_obstacle, 
                                 zorder = z_obstacle))

        # add the targets 
        for target in self.game_state.environment.get_targets():
            target_opacity = target.get_opacity()
            target_radius  = (target.get_value() / self.max_value) * \
                              self.size_max_radius
            ax.add_patch(Circle(xy      = np.array(target.get_centroid()),
                                radius  = target_radius,
                                color   = self.color_target, 
                                alpha   = target_opacity,
                                zorder  = z_obstacle))

        # add the auvs
        for auv in self.game_state.auvs:
            (x, y, dx, dy) = auv.get_arrow_params()
            ax.add_patch(FancyArrow(x           = x,
                                    y           = y,
                                    dx          = dx,
                                    dy          = dy,
                                    fc          = self.color_auv,
                                    width       = self.size_arrow_width,
                                    head_width  = self.size_arrow_h_width ,
                                    head_length = self.size_arrow_h_length,
                                    zorder      = z_auv,
                                    length_includes_head = True))

            # add the auv paths
            for i in range(1,len(auv.get_path())):
                (xdata, ydata) = auv.get_rectangle_params(i)
                ax.add_line(Line2D(xdata        = xdata,
                                   ydata        = ydata,
                                   linewidth    = self.size_auv_path*10,
                                   color        = self.color_auv_path,
                                   zorder       = z_auv_path))

        # set title and other plot styles, then generate plot 
        plt.title('AUV Trajectory \n Accumulated Reward: ' + \
                   str(self.game_state.reward), self.title_font)
        plt.xlabel('local x [m]', self.title_font)
        plt.ylabel('local y [m]', self.title_font)
        plt.xticks(np.arange(x_min, x_max+2, 1.0), 
                   ['' for i in np.arange(x_min, x_max+2, 1.0)])
        plt.yticks(np.arange(y_min, y_max+2, 1.0), 
                   ['' for i in np.arange(y_min, y_max+2, 1.0)])
        ax.grid()
        ax.axis('equal')
        ax.set_xlim(x_min - self.size_buffer, x_max + self.size_buffer + 1)
        ax.set_ylim(y_min - self.size_buffer, y_max + self.size_buffer + 1)

        # save or display the file
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
