from pacman import *
from mcts import *


def line(start: (int, int), increment: (int, int), length: int) -> set:
    return set([(start[0] + k * increment[0], start[1] + k * increment[1]) for
                k in range(length + 1)])


def circle(center: (int, int), radius: int = 1):
    return set([(i, j)
                for i in range(center[0] - radius, center[0] + radius + 1)
                for j in range(center[1] - radius, center[1] + radius + 1)
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2])


def pacman_example():
    obstacles = {(10, 10), (11, 9), (11, 10), (12, 11), (13, 10), (12, 7),
                 (15, 7), (15, 9), (15, 10), (15, 12), (15, 13), (15, 14),
                 (16, 5), (16, 8), (16, 9), (16, 10), (3, 9), (3, 7), (2, 7),
                 (2, 8), (2, 9), (18, 2), (18, 3), (17, 3), (16, 3), (18, 6),
                 (19, 6), (16, 2), (12, 18), (12, 19), (18, 12), (19, 12),
                 (18, 9), (16, 16), (16, 17), (16, 19), (17, 16), (18, 19),
                 (10, 16), (10, 17), (9, 18), (13, 15), (2, 12), (7, 11),
                 (2, 2), (2, 4), (3, 3), (3, 2), (3, 5)}
    targets = {(17, 2): 10, (4, 18): 10, (2, 3): 10}
    obstacles = obstacles.union(line((5, 9), (1, 0), 4)) \
        .union(line((9, 9), (0, -1), 4)).union(line((5, 7), (0, -1), 5)) \
        .union(line((7, 5), (0, -1), 3)).union(line((9, 3), (1, 0), 4)) \
        .union(line((11, 4), (0, 1), 2)).union(line((9, 3), (1, 0), 4)) \
        .union(circle((5, 15), 3)).union(line((10, 13), (1, 0), 3))
    env = PacmanEnvironment(xlim=(0, 20), ylim=(0, 20), obstacles=obstacles,
                            targets=targets, is_border_obstacle_filled=True)
    state = PacmanState(environment=env, time_remains=25)
    state.add_agent((7, 7)).add_agent((13, 8)).add_agent((12, 12))
    return state


if __name__ == "__main__":

    # Initialize a random environment
    initial_state = pacman_example()

    # agents perform a random walk
    initial_state.visualize(fig_size=(15, 15))


