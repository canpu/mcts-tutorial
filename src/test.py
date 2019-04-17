import numpy as np
from nose.tools import assert_equal, ok_
from IPython.display import display, HTML, clear_output

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


def test_simulate(simulate):
    """if simulate method is implemented correctly"""
    # TODO 
    return True


def test_backpropagate(backpropagate):
    """if backpropagate method is implemented correctly"""
    # TODO 
    return True


def test_reward(reward):
    """if reward method is implemented correctly"""
    # TODO 
    return True


def test_is_terminal(is_terminal):
    """if is_terminal method is implemented correctly"""
    # TODO 
    return True


def test_possible_actions(possible_actions):
    """if possible_actions method is implemented correctly"""
    # TODO 
    return True


def test_take_action(take_action):
    """if take_action method is implemented correctly"""
    # TODO 
    return True