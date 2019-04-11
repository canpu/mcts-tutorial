import math
import random
from state import AbstractState as State, AbstractAction as Action


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


class Node(object):
    def __init__(self, state: State):
        """ Create a Node object with given state
        """
        self._state = state
        self._parent = None
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
    def is_expanded(self) -> bool:
        """ Whether all possible actions have been added to child {edge: node}
        """
        return len(self._state.possible_actions) == len(self.children.items())

    def add_child(self, action: Action, node: "Node") -> "Node":
        """ Add a child node and set the parent of the child node
        :param action: The action that would lead the current node to the child
            node
        :param node: The child node
        """
        self.children[action] = node
        node._parent = self
        return self

    def __del__(self) -> None:
        """ Delete a node and all its child nodes
        """
        if self._parent:
            act = None
            for action, child_node in self._parent.children.items():
                if child_node == self:
                    act = action
                    break
            del self._parent.children[act]
        del self

    def __eq__(self, other: "Node") -> bool:
        return (self.__class__ == other.__class__ and
                self._state == other._state and self._parent == other._parent)


def back_propagate(node: Node, reward: float = 0.0) -> None:
    """ Propagate the reward and sample count from the specified node
        back all the way to the root node (the node with no parent)
    :param node: The node where the reward is evaluated
    :param reward: The reward at the node
    """
    while node is not None:
        node.num_samples += 1
        node.tot_reward += reward
        node = node.parent


class MonteCarloSearchTree:
    def __init__(self, samples: int = 1000, exploration_const: float = 1.0,
                 rollout_policy=default_rollout_policy):
        """ Create a MonteCarloSearchTree object
        :param samples: The number of samples to generate to obtain the best
                action
        :param exploration_const: The constant on the second term of UCB
        :param rollout_policy: The simulation function
        :type: A function that takes a state as input, perform simulation until
                a terminal state, and returns the reward of the final state
        """
        if samples <= 0:
            raise ValueError("The number of samples must be positive")
        self.max_samples = samples
        self.exploration_const = exploration_const
        self.rollout = rollout_policy
        self.root = None

    @staticmethod
    def select(node: Node, exploration_const: float = 1.0) -> Node:
        """ Select the best child node based on UCB; if there are multiple
            child nodes with the max UCB, randomly select_and_expand one
        :param node: The parent node
        :param exploration_const: The exploration constant
        :return: The best child node
        """
        max_val = -math.inf
        max_nodes = []
        for child in node.children.values():
            node_val = (child.tot_reward / child.num_samples
                        + exploration_const *
                        math.sqrt(2.0 * math.log(node.num_samples) /
                                  child.num_samples))
            if node_val > max_val:
                max_val = node_val
                max_nodes = [child]
            elif node_val == max_val:
                max_nodes.append(child)
        return random.choice(max_nodes)

    @staticmethod
    def expand(node: Node) -> Node:
        """ Add a new child node to the given node
        :param node: The parent node
        :return: The child node
        """
        if node.is_expanded:
            raise Exception("Should not expand a node that has already"
                            " been expanded")
        possible_actions = node.state.possible_actions
        for action in possible_actions:
            if action not in node.children.keys():
                child_node = Node(node.state.execute_action(action))
                node.add_child(action, child_node)
                return child_node

    @staticmethod
    def select_and_expand(node: Node, exploration_const: float) -> Node:
        """ If a node is expanded, then choose the best child node by UCB;
            if a node is not expanded, then choose a child node that is not
            built into the tree yet
        :param node: The parent node
        :param exploration_const: The exploration constant
        :return: The selected child node
        """
        while not node.is_terminal:
            if node.is_expanded:
                node = MonteCarloSearchTree.select(node, exploration_const)
            else:
                return MonteCarloSearchTree.expand(node)
        return node

    def execute_round(self) -> None:
        """ Perform selection, expansion, simulation and backpropagation with
            one sample
        """
        node = self.select_and_expand(self.root, self.exploration_const)
        reward = self.rollout(node.state)
        back_propagate(node, reward)

    def search_for_action(self, state: State) -> Action:
        """ With given initial state, obtain the best action to take by MCTS
        :param state: The initial state
        :return: The best action
        """
        self.root = Node(state)
        for i in range(self.max_samples):
            self.execute_round()
        best_child = self.select(self.root, 0)
        for action, node in self.root.children.items():
            if node is best_child:
                return action
