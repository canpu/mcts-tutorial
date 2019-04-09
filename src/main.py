from AUV import AUV
from GamePosition import GamePosition
from GameState    import GameState
from GameStateViz import GameStateViz
from Environment  import Environment


if __name__ == "__main__":
    # initialize a random environment 
    bbox = [1, 11, 1, 11]
    environment = Environment(bbox)
    environment.gen_random_environment()

    # initialize the AUV
    time_steps = 10
    start_pos1 = GamePosition(2,2)
    start_pos2 = GamePosition(10,10)
    auv1       = AUV(start_pos1, time_steps)
    auv2       = AUV(start_pos2, time_steps)
    environment.rem_obstacle(start_pos1)
    environment.rem_obstacle(start_pos2)
    environment.rem_target(start_pos1)
    environment.rem_target(start_pos2)

    # initialize the game state given the environment
    initial_state = GameState(auvs=[auv1, auv2], environment=environment)

    # agents perform a random walk
    initial_state.gen_random_walk()

    # visualize the game state 
    viz = GameStateViz(initial_state)
    viz.plot()
