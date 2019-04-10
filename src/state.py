class AbstractAction:
    pass


class AbstractState:
    @property
    def reward(self) -> float:
        raise NotImplementedError("The method not implemented")

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError("The method not implemented")

    @property
    def possible_actions(self) -> list:
        """ The possible actions to take at this state
            Make sure that the returned list is not empty unless the state is
            a terminal state
        :return: A list of possible states
        """
        raise NotImplementedError("The method not implemented")

    def execute_action(self, action: AbstractAction) -> "AbstractState":
        raise NotImplementedError("The method not implemented")
