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
