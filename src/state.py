class AbstractState:
    @property
    def feasible_next_states(self) -> list:
        raise NotImplementedError("The method not implemented")

    @property
    def reward(self) -> float:
        raise NotImplementedError("The method not implemented")

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError("The method not implemented")
