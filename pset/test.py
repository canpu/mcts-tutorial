import math
import random
import numpy as np
from nose.tools import assert_equal, ok_
from IPython.display import display, HTML, clear_output

from state  import AbstractState  as State
from state  import AbstractAction as Action
from mcts   import Node
from mcts   import MonteCarloSearchTree
from gomoku import *


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
    """return true iff select method is implemented correctly
    """
    # initialize a blank Gomoku board
    gomoku_init_state = GomokuState(use_default_heuristics=True, reward_player=0)
    gomoku_init_node  = Node(gomoku_init_state)

    # black makes first move
    black_move_1    = gomoku_init_node.add_child(GomokuAction(0,(4,4)))    

    # consider all moves for white 
    white_move_1    = black_move_1.add_child(GomokuAction(1,(3,4)))
    white_move_2    = black_move_1.add_child(GomokuAction(1,(3,5)))
    white_move_3    = black_move_1.add_child(GomokuAction(1,(4,5)))
    white_move_4    = black_move_1.add_child(GomokuAction(1,(5,5)))
    white_move_5    = black_move_1.add_child(GomokuAction(1,(5,4)))
    white_move_6    = black_move_1.add_child(GomokuAction(1,(5,3)))
    white_move_7    = black_move_1.add_child(GomokuAction(1,(4,3)))
    white_move_8    = black_move_1.add_child(GomokuAction(1,(3,3)))

    # set the reward and sample count for black moves
    black_move_1.num_samples, white_move_1.tot_reward = (13,35)

    # set the reward and sample count for white moves
    white_move_1.num_samples, white_move_1.tot_reward = (1,  1)
    white_move_2.num_samples, white_move_2.tot_reward = (2,  4)
    white_move_3.num_samples, white_move_3.tot_reward = (5, 25)
    white_move_4.num_samples, white_move_4.tot_reward = (1,  3)
    white_move_5.num_samples, white_move_5.tot_reward = (1,  1)
    white_move_6.num_samples, white_move_6.tot_reward = (1,  1)
    white_move_7.num_samples, white_move_7.tot_reward = (1,  0)
    white_move_8.num_samples, white_move_8.tot_reward = (1,  0)

    # check that the correct action was selected
    assert_equal(select(black_move_1, 0.00)[0], GomokuAction(1,(4,5)), "wrong action selected!")
    assert_equal(select(black_move_1, 1.50)[0], GomokuAction(1,(4,5)), "wrong action selected!")
    assert_equal(select(black_move_1, 1.60)[0], GomokuAction(1,(5,5)), "wrong action selected!")
    assert_equal(select(black_move_1, 10.0)[0], GomokuAction(1,(5,5)), "wrong action selected!")

    # check that the correct node was selected
    assert_equal(select(black_move_1, 0.00)[1], white_move_3, "wrong child node selected!")
    assert_equal(select(black_move_1, 1.50)[1], white_move_3, "wrong child node selected!")
    assert_equal(select(black_move_1, 1.60)[1], white_move_4, "wrong child node selected!")
    assert_equal(select(black_move_1, 10.0)[1], white_move_4, "wrong child node selected!")

    return True


def test_expand(expand):
    """return true iff expand method is implemented correctly
    """
    # initialize a blank Gomoku board
    gomoku_init_state = GomokuState(use_default_heuristics=True, reward_player=0)
    gomoku_init_node  = Node(gomoku_init_state)

    # black makes first move
    init_node     = Node(gomoku_init_state)
    black_node    = init_node.add_child(GomokuAction(0,(4,4)))
    black_actions = list(black_node.unused_edges)
    num_edges     = len(black_actions)
    num_samples   = 500
    deviation     = .20
    white_nodes   = list([black_node.add_child(action) for 
                          action in black_actions])

    # count the results of calling `expand` many times
    frequency_dict  = {}
    for i in range(num_edges * num_samples):
        init_node  = Node(gomoku_init_state)
        black_node = init_node.add_child(GomokuAction(0,(4,4)))
        white_node = expand(black_node)
        if (white_node not in white_nodes):
            print(white_node)
            raise ValueError("returned a Node that is not associated with an untried action!")
        if str(white_node) in frequency_dict: 
            frequency_dict[str(white_node)] += 1
        else:        
            frequency_dict[str(white_node)]  = 1

    # check that expand is behaving via random selection
    for value in frequency_dict.values():
        if (abs(value - num_samples) > 
            num_samples * deviation):
            raise ValueError("possible actions are not being sampled uniformly at randomly!") 
   
    # check that exception is raised
    init_node       = Node(gomoku_init_state)
    black_node      = init_node.add_child(GomokuAction(0,(4,4)))
    white_move_1    = black_node.add_child(GomokuAction(1,(3,4)))
    white_move_2    = black_node.add_child(GomokuAction(1,(3,5)))
    white_move_3    = black_node.add_child(GomokuAction(1,(4,5)))
    white_move_4    = black_node.add_child(GomokuAction(1,(5,5)))
    white_move_5    = black_node.add_child(GomokuAction(1,(5,4)))
    white_move_6    = black_node.add_child(GomokuAction(1,(5,3)))
    white_move_7    = black_node.add_child(GomokuAction(1,(4,3)))
    white_move_8    = black_node.add_child(GomokuAction(1,(3,3)))
    try:
        expand(black_node)
    except Exception:
        pass
    else:
       raise Exception("Should throw an exception for trying to expand a node that has already been expanded")
    return True


def test_default_rollout_policy(default_rollout_policy):
    """return true iff default_rollout_policy method is implemented correctly
    """
    gomoku_init_state = GomokuState(use_default_heuristics=True, reward_player=0)
    frequency_dict    = {}
    for i in range(100):
        val = default_rollout_policy(gomoku_init_state)
        if val in frequency_dict:
            frequency_dict[val] += 1
        else:
            frequency_dict[val] = 1
    lower_bound = 40
    upper_bound = 70
    # check to see if the frequency is within a reasonable bound 
    if not(frequency_dict[1] > lower_bound and 
           frequency_dict[1] < upper_bound):
        raise ValueError("not performing random rollout policy correctly!")
    return True


def test_backpropagate(backpropagate):
    """return true iff backpropagate method is implemented correctly
    """
    init_state = GomokuState(use_default_heuristics=True, reward_player=0)
    init_node  = Node(init_state)

    # assemble all of the moves 
    black_move_0    = init_node.add_child(GomokuAction(0,(4,4)))
    white_move_1    = black_move_0.add_child(GomokuAction(1,(5,4)))     
    black_move_2    = white_move_1.add_child(GomokuAction(1,(6,4)))         
    black_move_3    = white_move_1.add_child(GomokuAction(1,(5,5)))   
    black_move_4    = white_move_1.add_child(GomokuAction(1,(5,3)))
    white_move_5    = black_move_2.add_child(GomokuAction(1,(6,5)))   
    white_move_6    = black_move_2.add_child(GomokuAction(1,(7,4)))   
   
    # assign values to the "terminal" moves and back-propagate
    backpropagate(black_move_2,  5)
    backpropagate(black_move_3, -1)
    backpropagate(black_move_4,  3)
    backpropagate(white_move_5, -4)
    backpropagate(white_move_6,  3)

    # check the values of the nodes in regards to num_samples and tot_reward
    assert_equal(black_move_0.num_samples, 5, "wrong number of samples!")
    assert_equal(white_move_1.num_samples, 5, "wrong number of samples!")
    assert_equal(black_move_2.num_samples, 3, "wrong number of samples!")
    assert_equal(black_move_3.num_samples, 1, "wrong number of samples!")
    assert_equal(black_move_4.num_samples, 1, "wrong number of samples!")
    assert_equal(white_move_5.num_samples, 1, "wrong number of samples!")
    assert_equal(white_move_6.num_samples, 1, "wrong number of samples!")
    assert_equal(black_move_0.tot_reward, 6, "wrong total reward!")
    assert_equal(white_move_1.tot_reward, 6, "wrong total reward!")
    assert_equal(black_move_2.tot_reward, 4, "wrong total reward!")
    assert_equal(black_move_3.tot_reward, -1, "wrong total reward!")
    assert_equal(black_move_4.tot_reward, 3, "wrong total reward!")
    assert_equal(white_move_5.tot_reward, -4, "wrong total reward!")
    assert_equal(white_move_6.tot_reward, 3, "wrong total reward!")

    return True


def test_reward(MazeState):
    """if reward method is implemented correctly"""
    # TODO 
    return True


def test_is_terminal(MazeState):
    """if is_terminal method is implemented correctly"""
    # TODO 
    return True


def test_possible_actions(MazeState):
    """if possible_actions method is implemented correctly"""
    # TODO 
    return True


def test_take_action(MazeState):
    """if take_action method is implemented correctly"""
    # TODO 
    return True
    