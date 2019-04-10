from copy import deepcopy
from state import AbstractState, AbstractAction


class GomokuAction(AbstractAction):
    def __init__(self, player: int, position: tuple):
        """ Create a GomokuAction object identified by the player and
            location to place the piece
        :param player: The player to place the piece
        :param position: The location to place the piece
        """
        if player != 0 and player != 1:
            raise ValueError("The player is not valid")
        if not GomokuState.is_in_board(position):
            raise ValueError("The position is not valid")
        self.player = player
        self.position = position

    def __eq__(self, other) -> bool:
        return (self.__class__ == other.__class__ and
                self.player == other.player and self.position ==
                other.position)

    def __hash__(self) -> int:
        return hash((self.player, self.position[0], self.position[1]))

    def __str__(self) -> str:
        return ("Action: (player {0} position {1})".format(self.player,
                                                           self.position))


class GomokuState(AbstractState):
    board_size = 9
    winning_length = 4

    @staticmethod
    def is_in_board(position: tuple):
        return (len(position) == 2 and isinstance(position[0], int) and
                isinstance(position[1], int) and
                0 <= position[0] < GomokuState.board_size and
                0 <= position[1] < GomokuState.board_size)

    def __init__(self, reward_player: int = 0):
        """ Create a GomokuState object
        :param reward_player: The player based on whom reward is calculated
        """
        if reward_player != 0 and reward_player != 1:
            raise ValueError("The player index must be 0 or 1")
        self._history = [[], []]
        self._player = 0
        self._reward_player = reward_player

    def __eq__(self, other: "GomokuState") -> bool:
        return (self.__class__ == other.__class__ and
                self._history == other._history)

    def __hash__(self) -> tuple:
        return tuple(self._history[0] + self._history[1])

    def __copy__(self) -> "GomokuState":
        new_state = GomokuState()
        new_state._player = self._player
        new_state._history = deepcopy(self._history)
        return new_state

    def switch_side(self) -> None:
        """ Switch reward side from white to black or from black to white
        """
        self._player = 1 - self._player

    @property
    def player(self) -> int:
        return self._player

    @property
    def possible_actions(self) -> list:
        """ Get all positions that have not been occupied
        :return: A list of actions
        """
        return [GomokuAction(self._player, (i, j)) for i in range(
                GomokuState.board_size) for j in range(
                GomokuState.board_size) if
                (i, j) not in self._history[0] and
                (i, j) not in self._history[1]]

    @staticmethod
    def _max_line_seg_len(pts: set, direction: tuple, start: tuple) -> int:
        """ Get the maximal length of line segments formed by pts along the
            given direction, from a starting point
        :param pts: The points
        :param direction: The direction of increment
        :param start: The starting point
        :return: The maximal length of line segment along the direction
        """
        max_len = 0
        cur_len = 0
        pt = start
        while (0 <= pt[0] < GomokuState.board_size and 0 <= pt[1] <
            GomokuState.board_size):
            if pt in pts:
                cur_len += 1
                max_len = max(max_len, cur_len)
            else:
                cur_len = 0
            pt = (pt[0] + direction[0], pt[1] + direction[1])
        return max_len

    def if_player_wins(self, player: int) -> bool:
        """ If the given player has won the game
        :param player: The specified player's index
        :return: Whether the player has won
        """
        if player != 0 and player != 1:
            raise ValueError("The specified player does not exist")
        pts = set(self._history[player])
        if len(pts) < GomokuState.winning_length:
            return False
        for j in range(GomokuState.board_size):
            if (GomokuState._max_line_seg_len(pts, (1, 0), (0, j)) >=
                    GomokuState.winning_length):
                return True
            if (GomokuState._max_line_seg_len(pts, (1, 1), (0, j)) >=
                    GomokuState.winning_length):
                return True
            if (GomokuState._max_line_seg_len(
                    pts, (-1, 1), (GomokuState.board_size, j)) >=
                    GomokuState.winning_length):
                return True
        for i in range(GomokuState.board_size):
            if (GomokuState._max_line_seg_len(pts, (0, 1), (i, 0)) >=
                    GomokuState.winning_length):
                return True
            if (GomokuState._max_line_seg_len(pts, (1, 1), (i, 0)) >=
                    GomokuState.winning_length):
                return True
            if (GomokuState._max_line_seg_len(pts, (-1, 1), (i, 0)) >=
                    GomokuState.winning_length):
                return True
        return False

    @property
    def black_reward(self) -> float:
        if self.if_player_wins(0):
            return 1.0
        elif self.if_player_wins(1):
            return -1.0
        else:
            return 0.0

    @property
    def white_reward(self) -> float:
        return -self.black_reward

    @property
    def reward(self) -> float:
        if self._reward_player == 0:
            return self.black_reward
        else:
            return self.white_reward

    @property
    def is_terminal(self) -> bool:
        """ Either black wins or white wins, the game terminates
            If all positions are occupied, the game terminates
        """
        return (self.reward != 0 or set(self._history[0] + self._history[1])
                == [(i, j) for i in range(GomokuState.board_size)
                    for j in range(GomokuState.board_size)])

    def execute_action(self, action: GomokuAction) -> "GomokuState":
        """ Execute the action and return the new state
        :param action: The (player, action) tuple
        :return: The new state
        """
        if (action.position in self._history[0]
                or action.position in self._history[1]):
            raise ValueError("The position has already been taken")
        new_state = self.__copy__()
        new_state._player = 1 - self._player
        new_state._history[action.player].append(action.position)
        return new_state

    def go(self, position: tuple) -> None:
        self.execute_action(GomokuAction(self._player, position))

    # TODO
    def visualize(self):
        pass
