import math
import random
from state import AbstractState as State, AbstractAction as Action


class Node(object):
    def __init__(self, state: State):
        """ Create a Node object with given state
        """
        self._state = state
        self._parent = None
        self._untried_edges = self._state.possible_actions
        self.children = {}  # {AbstractAction: AbstractState}
        self.tot_reward = 0
        self.num_samples = 0

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_terminal(self):
        return self._state.is_terminal

    @property
    def parent(self) -> "Node":
        return self._parent

    @property
    def depth(self) -> int:
        """ The depth (distance to the root, whose parent is None, plus 1) of
            the node
        """
        depth = 0
        node = self
        while node:
            depth += 1
            node = node.parent
        return depth

    @property
    def unused_edges(self) -> list:
        return self._untried_edges

    @property
    def is_expanded(self) -> bool:
        """ Whether all possible actions have been tried
        """
        return len(self._untried_edges) == 0

    def add_child(self, action: Action) -> "Node":
        """ Add a child node and set the parent of the child node and return the
            child node
        :param action: The action that would lead the current node to the child
            node
        :return: The child node
        """
        child = Node(self._state.execute_action(action))
        if action in self._untried_edges:
            self._untried_edges.remove(action)
        self.children[action] = child
        child._parent = self
        return child

    def remove_child(self, child: "Node") -> "Node":
        """ Remove a child node from this node
        :param child: The child node
        """
        act = None
        for action, node in self.children.items():
            if node == child:
                child._parent = None
                act = action
        if act:
            del self.children[act]
        else:
            raise ValueError("The node does not have the given child node")
        return self

    def __eq__(self, other: "Node") -> bool:
        return (self.__class__ == other.__class__ and
                self._state == other._state and self._parent == other._parent)

    def __str__(self) -> str:
        return str(self._state)


def select(node: Node, exploration_const: float = 1.0) -> (Action, Node):
    """ Select the best child node based on UCB; if there are multiple
        child nodes with the max UCB, randomly select one
    :param node: The parent node
    :param exploration_const: The exploration constant in UCB formula
    :return: The action and the corresponding best child node
    """
    max_val = -math.inf
    max_actions = []
    for action, child in node.children.items():
        # print(len(node.children))
        node_val = (child.tot_reward / child.num_samples
                    + exploration_const *
                    math.sqrt(2.0 * math.log(node.num_samples) /
                              child.num_samples))
        if node_val > max_val:
            max_val = node_val
            max_actions = [action]
        elif node_val == max_val:
            max_actions.append(action)
    max_action = random.choice(max_actions)
    return max_action, node.children[max_action]


def expand(node: Node) -> Node:
    """ Randomly select an untried action and create a child node based on it
        Return the new child node
    :param node: The parent node
    :return: The child node
    """
    if node.is_expanded:
        raise Exception("Should not expand a node that has already"
                        " been expanded")
    action = random.choice(node.unused_edges)
    return node.add_child(action)


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


def backpropagate(node: Node, reward: float = 0.0) -> None:
    """ Propagate the reward and sample count from the specified node
        back all the way to the root node (the node with no parent)
    :param node: The node where the reward is evaluated
    :param reward: The reward at the node
    """
    while node is not None:
        node.num_samples += 1
        node.tot_reward += reward
        node = node.parent


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
        count = 0
        for _ in range(self._max_samples):
            execute_round(self._root, max_tree_depth=self._max_tree_depth)
            count += 1
            print(count)
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
