import random
from math import inf, log, sqrt
from state import *


class Node:
    def __init__(self, state: AbstractState, parent: 'Node' = None):
        self._state = state
        self.is_expanded = False
        self._parent = parent
        self._children = set()

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and self._state == other._state and self._parent == other._parent

    def clear_parent(self) -> 'Node':
        """ Set the parent of the node to None
        """
        self._parent = None
        return self

    @property
    def is_terminal_node(self) -> bool:
        return self._state.is_terminal

    @property
    def parent(self) -> 'Node':
        return self._parent

    @property
    def children(self) -> set:
        return self._children

    @property
    def state(self) -> AbstractState:
        return self._state

    def add_child(self, node: 'Node') -> 'Node':
        """ Add a child node to the current node
        :param node: The child node
        :type: A Node object
        """
        node._parent = self
        self._children.add(node)
        return self


def default_rollout_policy(child_state_list: list, cur_node: AbstractState) \
        -> list:
    """ Return an array with uniform probability distribution
    :param child_state_list: A list of possible next states
    :type: A list of AbstractState objects
    :param cur_node: The current state
    :type: An AbstractState object
    :return: A list of floats
    """
    return [1.0 / len(child_state_list)] * len(child_state_list)


class MonteCarloSearchTree:
    def __init__(self, initial_state: AbstractState, depth_limit: int = inf,
                 rollout_policy=default_rollout_policy):
        """ Create a MonteCarloSearchTree object with specified depth limit
            and rollout policy
        :param depth_limit: The maximal allowable depths of the search tree
        :type: A positive integer
        :param rollout_policy: The policy that assigns probability to each
                possible state
        :type: A function that takes a list of new states and the current state
                as input and returns a list of float probabilities with the
                same length as the possible new states
        """
        if depth_limit is not None:
            assert depth_limit > 0
        self._rollout = rollout_policy
        self._depth_limit = depth_limit if depth_limit else inf
        self._root = Node(initial_state)
        self._reward = {self._root: 0.0}
        self._num_samples = {self._root: 0}

    def sample(self, node: Node, max_depth: int = inf, num_samples: int = 1000) \
            -> 'MonteCarloSearchTree':
        """ Perform Monte Carlo sampling starting from a node and update the
            number of samples and reward for corresponding nodes
        :param node: The starting node
        :param max_depth: The maximal allowable depth for a single sample
        :param num_samples: The number of Monte Carlo samples
        """
        assert max_depth > 0 and num_samples > 0
        for _ in range(num_samples):
            depth = 0
            cur_node = node
            while depth < max_depth and not cur_node.is_terminal_node:
                possible_next_states = cur_node.state.feasible_next_states
                p = self._rollout(possible_next_states, cur_node.state)
                next_state = random.choice(possible_next_states, p=p)
                new_node = Node(next_state, parent=cur_node)
                if new_node not in cur_node.children:
                    cur_node.add_child(new_node)
                    self._reward[new_node] = 0.0
                    self._num_samples[new_node] = 0
                cur_node = new_node
                depth += 1
            self._back_propagate(cur_node)
        return self

    def _back_propagate(self, node: Node) -> 'MonteCarloSearchTree':
        """ Propagate from leaf to root and update the number of samples, and
            the rewards of each node along the way
        :param node: The start node of back propagation
        :type: A Node object
        """
        reward = node.state.reward
        while node is not None:
            self._num_samples[node] += 1
            self._reward[node] += reward
            node = node.parent
        return self

    def get_best_next_node(self, node: Node) -> Node:
        """ After performing sampling, obtain the best next possible state from
            a starting node
            When no possible action is available to change the state, return the
            node itself
        :param node: The starting node, from which action is taken to obtain
                the next state
        """
        children = node.children
        if children:
            return max(children, key=lambda n: (
                self._reward[n] / self._num_samples[n] + sqrt(2.0 * log(
                    self._num_samples[n]) / self._num_samples[n])))
        else:
            return node

    def execute_best_next_action(self) -> 'MonteCarloSearchTree':
        """ Perform sampling from the current state and choose the best action for the next state
        """
        next_node = self.get_best_next_node(self._root)
        self._root = next_node
        self._root.clear_parent()
        return self

    def simulate_all_history(self) -> AbstractState:
        """ Simulate the entire problem until the termination criteria is met
        :return: The final state
        """
        while not self._root.is_terminal_node:
            self.execute_best_next_action()
        return self._root.state
