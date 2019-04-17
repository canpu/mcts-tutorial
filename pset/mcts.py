import math
import random
from node  import Node
from state import AbstractState  as State
from state import AbstractAction as Action


def default_rollout_policy(state: State) -> float:
    """ The default policy for simulation is to randomly (uniform distribution)
        select an action to update the state and repeat the simulation until
        a terminal state is reached
    :param state: The starting state
    :return: The reward at the terminal node
    """
    while not state.is_terminal:
        action = random.choice(state.possible_actions)
        state = state.execute_action(action)
    return state.reward


def execute_round(root: Node, rollout_policy=default_rollout_policy,
                  max_tree_depth: int = 15) -> None:
    """ Perform selection, expansion, simulation and backpropagation with
        one sample
    """
    cur = root
    while cur.is_expanded and cur.depth < max_tree_depth:
        act, cur = select(cur, exploration_const=1.0)
    simulation_node = expand(cur) if max_tree_depth > cur.depth else cur
    reward = rollout_policy(simulation_node.state)
    backpropagate(simulation_node, reward)


class MonteCarloSearchTree:
    def __init__(self, initial_state: State, samples: int = 1000,
                 exploration_const: float = 1.0, max_tree_depth: int = 10,
                 rollout_policy=default_rollout_policy):
        """ Create a MonteCarloSearchTree object
        :param initial_state: The initial state
        :param samples: The number of samples to generate to obtain the best
                action
        :param exploration_const: The constant on the second term of UCB
        :param max_tree_depth: The maximal allowable number of nodes in the tree
        :param rollout_policy: The simulation function
        :type: A function that takes a state as input, perform simulation until
                a terminal state, and returns the reward of the final state
        """
        if samples <= 0 or max_tree_depth <= 1:
            raise ValueError("The number of samples must be positive")
        self._max_samples = samples
        self._exploration_const = exploration_const
        self._rollout = rollout_policy
        self._root = Node(initial_state)
        self._max_tree_depth = max_tree_depth

    def search_for_actions(self, search_depth: int = 1) -> list:
        """ With given initial state, obtain the best actions to take by MCTS
        :param search_depth: How many steps of actions are wanted
        :return: The best actions
        :rtype: A list of AbstractAction objects
        """
        for _ in range(self._max_samples):
            execute_round(self._root, max_tree_depth=self._max_tree_depth)
        actions = []
        cur_node = self._root
        for _ in range(search_depth):
            if cur_node.is_terminal:
                break
            else:
                action, child = select(cur_node, exploration_const=0.0)
                cur_node = child
                actions.append(action)
        return actions

    def update_root(self, action: Action) -> "MonteCarloSearchTree":
        """ Update the root node to reflect the new state after an action is
            taken
        :param action: The action that brings a new state
        """
        if action in self._root.children:
            new_root = self._root.children[action]
        else:
            new_root = self._root.add_child(action)
        self._root.remove_child(new_root)
        self._root = new_root
        return self
