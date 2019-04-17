from maze_unfinished import *


def line(start: (int, int), increment: (int, int), length: int) -> set:
    return set([(start[0] + k * increment[0], start[1] + k * increment[1]) for
                k in range(length + 1)])


def circle(center: (int, int), radius: int = 1):
    return set([(i, j)
                for i in range(center[0] - radius, center[0] + radius + 1)
                for j in range(center[1] - radius, center[1] + radius + 1)
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2])


def gen_obstacles():
    obstacles = {(10, 10), (11, 9), (11, 10), (12, 11), (13, 10), (12, 7),
                 (15, 7), (15, 9), (15, 10), (15, 12), (15, 13), (15, 14),
                 (16, 5), (16, 8), (16, 9), (16, 10), (3, 9), (3, 7), (2, 7),
                 (2, 8), (2, 9), (18, 2), (18, 3), (17, 3), (16, 3), (18, 6),
                 (19, 6), (16, 2), (12, 18), (12, 19), (18, 12), (19, 12),
                 (18, 9), (16, 16), (16, 17), (16, 19), (17, 16), (18, 19),
                 (10, 16), (10, 17), (9, 18), (13, 15), (2, 12), (7, 11),
                 (2, 2), (2, 4), (3, 3), (3, 2), (3, 5)}
    obstacles = obstacles.union(line((5, 9), (1, 0), 4)) \
        .union(line((9, 9), (0, -1), 4)).union(line((5, 7), (0, -1), 5)) \
        .union(line((7, 5), (0, -1), 3)).union(line((9, 3), (1, 0), 4)) \
        .union(line((11, 4), (0, 1), 2)).union(line((9, 3), (1, 0), 4)) \
        .union(circle((5, 15), 3)).union(line((10, 13), (1, 0), 3))
    return obstacles


def maze_example_1():
    obstacles = gen_obstacles()
    targets = {(6, 3): 3, (4, 8): 3, (2, 3): 3, (14, 4): 3, (15, 8): 3,
               (17, 3): 3, (9, 12): 3, (14, 14): 3, (8, 16): 3, (11, 11): 1,
               (11, 13): 1, (12, 6): 1, (10, 7): 1, (8, 6): 1, (8, 8): 1,
               (16, 12): 1, (11, 16): 1, (18, 4): 1}
    env = MazeEnvironment(xlim=(0, 20), ylim=(0, 20), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    state = MazeState(environment=env, time_remains=15)
    state.add_agent((7, 7)).add_agent((13, 8)).add_agent((12, 12))
    return state


def maze_example_2():
    obstacles = gen_obstacles()
    targets = {(17, 2): 10, (4, 19): 10, (2, 3): 10,
               (9, 11): 1, (13, 12): 1, (14, 13): 1, (16, 15): 1,
               (18, 14): 1, (19, 17): 1, (17, 19): 4, (8, 17): 1, (3, 12): 1,
               (10, 15): 1, (14, 10): 1, (14, 11): 1, (17, 11): 1, (14, 4): 1,
               (19, 5): 1, (8, 6): 1, (8, 5): 1, (4, 8): 4, (1, 6): 1,
               (15, 8): 1, (6, 1): 1, (3, 4): 1, (6, 12): 1, (11, 2): 4,
               (19, 11): 4}
    env = MazeEnvironment(xlim=(0, 20), ylim=(0, 20), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    state = UnfinishedMazeState(environment=env, time_remains=15)
    state.add_agent((7, 7)).add_agent((13, 8)).add_agent((12, 12))
    return state


def simulate(initial_state: UnfinishedMazeState, rand_seed: int = 0)\
        -> UnfinishedMazeState:
    mcts = MonteCarloSearchTree(initial_state, max_tree_depth=15,
                                samples=1000)
    random.seed(rand_seed)
    state = initial_state.__copy__()
    time = 0
    while not state.is_terminal:
        actions = mcts.search_for_actions(search_depth=3)
        time += 1
        print("Time step {0}".format(time))
        for i in range(len(state.paths)):
            action = actions[i]
            print(action)
            state.take_action(action)
            mcts.update_root(action)
            if state.is_terminal:
                break
    return state
