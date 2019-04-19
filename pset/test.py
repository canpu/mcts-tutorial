import numpy as np
from nose.tools import assert_equal, ok_
from IPython.display import display, HTML, clear_output
from maze_unfinished import MazeEnvironment, MazeAction

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!") 


def test_select(select):
    """if select method is implemented correctly"""
    # TODO 
    return True


def test_expand(expand):
    """if expand method is implemented correctly"""
    # TODO 
    return True


def test_default_rollout_policy(default_rollout_policy):
    """if default_rollout_policy method is implemented correctly"""
    # TODO 
    return True


def test_backpropagate(backpropagate):
    """if backpropagate method is implemented correctly"""
    # TODO 
    return True


def test_reward(maze_class):
    """if reward method is implemented correctly"""
    obstacles = {(2, 2), (3, 3), (4, 3), (4, 4)}
    targets = {(1, 2): 3, (2, 4): 4, (3, 1): 2, (4, 2): 1}
    env = MazeEnvironment(xlim=(0, 4), ylim=(0, 4), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    state = maze_class(environment=env, time_remains=15)
    state.add_agent((2, 3), (3, 2))
    assert_equal(state.reward, 0.0)
    state.paths[0].append((2, 4))
    assert_equal(state.reward, 4.0)
    state.paths[1].append((3, 1))
    assert_equal(state.reward, 6.0)
    state.paths[0].append((1, 2))
    assert_equal(state.reward, 9.0)
    state.paths[1].append((4, 2))
    assert_equal(state.reward, 10.0)
    return True


def test_is_terminal(maze_class):
    """if is_terminal method is implemented correctly"""
    obstacles = {(2, 2), (3, 3), (4, 3), (4, 4)}
    targets = {(1, 2): 3, (2, 4): 4, (3, 1): 2, (4, 2): 1}
    env = MazeEnvironment(xlim=(0, 4), ylim=(0, 4), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    for time in range(-10, 11):
        state = maze_class(environment=env, time_remains=time)
        assert_equal(state.is_terminal, time > 0)
    return True


def test_possible_actions(maze_class):
    """if possible_actions method is implemented correctly"""
    obstacles = {(2, 2), (3, 3), (4, 3), (4, 4)}
    targets = {(1, 2): 3, (2, 4): 4, (3, 1): 2, (4, 2): 1}
    env = MazeEnvironment(xlim=(0, 4), ylim=(0, 4), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    state = maze_class(environment=env, time_remains=15)
    state.add_agent((2, 3), (3, 2))
    assert_equal(set(state.possible_actions),
                 {MazeAction(0, (1, 3)), MazeAction(0, (2, 4))})
    state.switch_agent()
    assert_equal(set(state.possible_actions),
                 {MazeAction(1, (3, 1)), MazeAction(0, (4, 2))})
    state.switch_agent()
    state.paths[0].append((2, 4))
    assert_equal(set(state.possible_actions),
                 {MazeAction(0, (1, 4)), MazeAction(0, (3, 4)),
                  MazeAction(0, (2, 3))})
    state.switch_agent()
    state.paths[1].append((4, 2))
    assert_equal(set(state.possible_actions),
                 {MazeAction(1, (3, 2)), MazeAction(1, (4, 1))})
    return True


def test_take_action(maze_class):
    """if execute_action method is implemented correctly"""
    obstacles = {(2, 2), (3, 3), (4, 3), (4, 4)}
    targets = {(1, 2): 3, (2, 4): 4, (3, 1): 2, (4, 2): 1}
    env = MazeEnvironment(xlim=(0, 4), ylim=(0, 4), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    state = maze_class(environment=env, time_remains=15)
    state.add_agent((2, 3), (3, 2))
    state.execute_action(MazeAction(0, (2, 4)))
    assert_equal(state.paths[0], [(2, 3), (2, 4)])
    assert_equal(state.time_remains, 15)
    assert_equal(state.turn, 1)
    state.execute_action(MazeAction(1, (4, 2)))
    assert_equal(state.paths[1], [(3, 2), (4, 2)])
    assert_equal(state.time_remains, 14)
    assert_equal(state.turn, 0)
    state.execute_action(MazeAction(0, (1, 4)))
    assert_equal(state.paths[0], [(2, 3), (2, 4), (1, 4)])
    assert_equal(state.time_remains, 14)
    assert_equal(state.turn, 1)
    state.execute_action(MazeAction(1, (4, 1)))
    assert_equal(state.paths[1], [(3, 2), (4, 2), (4, 1)])
    assert_equal(state.time_remains, 13)
    assert_equal(state.turn, 0)
    return True
