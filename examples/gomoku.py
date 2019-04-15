from copy import deepcopy
from state import AbstractState, AbstractAction
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure


class GomokuAction(AbstractAction):
    def __init__(self, player: int, position: (int, int)):
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
        return ("Action (player {0} takes position {1})"
                .format('WHITE' if self.player else 'BLACK', self.position))


class GomokuState(AbstractState):
    board_size = 9
    winning_length = 5

    @staticmethod
    def is_in_board(position: (int, int)) -> bool:
        """ Determine if a position tuple is a valid position on the Gomoku
            board
        """
        return (0 <= position[0] < GomokuState.board_size and
                0 <= position[1] < GomokuState.board_size)

    def __init__(self, reward_player: int = 0,
                 use_default_heuristics: bool = False):
        """ Create a GomokuState object
            A GomokuState object is uniquely identified by the black and white
            pieces
        :param reward_player: The player based on whom reward is calculated
        :param use_default_heuristics: Whether the default heuristics is used to
            generate possible actions. The rule is, if the board has no
            pieces, then place it at the center; if the board is not empty, then
            place a piece only at locations where at least one of its neighbors
            is occupied
        """
        if reward_player != 0 and reward_player != 1:
            raise ValueError("The player index must be 0 or 1")
        self._history = [[], []]
        self._player = 0  # the player to place the first piece is black
        self._reward_player = reward_player
        self._use_heuristics = use_default_heuristics

    def __eq__(self, other: "GomokuState") -> bool:
        """ Determine two GomokuState objects are identical
            Two GomokuState objects are identical as long as the black pieces
            and white pieces are the same, regardless of the order by which they
            are placed
        """
        return (self.__class__ == other.__class__ and
                set(self._history[0]) == set(other._history[0]) and
                set(self._history[1]) == set(other._history[1]))

    def __hash__(self) -> tuple:
        return tuple(self._history[0] + self._history[1])

    def __copy__(self) -> "GomokuState":
        new_state = GomokuState(reward_player=self._reward_player,
                                use_default_heuristics=self._use_heuristics)
        new_state._player = self._player
        new_state._history = deepcopy(self._history)
        return new_state

    def __str__(self) -> str:
        black = set(self._history[0])
        white = set(self._history[1])
        s = ""
        for i in range(GomokuState.board_size):
            for j in range(GomokuState.board_size):
                position = (i, j)
                if position in black:
                    s += 'B'
                elif position in white:
                    s += 'W'
                else:
                    s += '#'
            s += '\n'
        return s

    @property
    def heuristics(self) -> bool:
        return self._use_heuristics

    @heuristics.setter
    def heuristics(self, flag: bool = False) -> None:
        self._use_heuristics = flag

    @property
    def player(self) -> int:
        return self._player

    @property
    def possible_actions(self) -> list:
        """ Get all possible positions to place the next piece
            When heuristics is not used, return all Action's for unoccupied
            positions; when heuristics is turned on, return a position if and
            only if it is unoccupied and at least one of its neighboring
            position is occupied; when heuristics is enabled but no piece has
            been placed, the only action allowed is to place the piece at the
            center
        :return: A list of actions
        """
        occupied = set(self._history[0] + self._history[1])
        unoccupied = set((i, j) for i in range(GomokuState.board_size)
                         for j in range(GomokuState.board_size)
                         if (i, j) not in occupied)
        if self._use_heuristics:
            if occupied:
                occupied_expanded = set((i + m, j + n) for i, j in occupied
                                        for m in [-1, 0, 1] for n in [-1, 0, 1])
                return [GomokuAction(self._player, position) for
                        position in occupied_expanded.intersection(unoccupied)]
            else:
                return [GomokuAction(self._player,
                                     (GomokuState.board_size // 2,
                                      GomokuState.board_size // 2))]
        else:
            return [GomokuAction(self._player, position)
                    for position in unoccupied]

    @property
    def side(self) -> int:
        return self._reward_player

    @side.setter
    def side(self, reward_player: int = 0) -> None:
        if reward_player != 0 and reward_player != 1:
            raise ValueError("The side (based on which reward is calculated) "
                             "can only be 0 for black or 1 for white")
        self._reward_player = reward_player

    @staticmethod
    def _max_line_seg_len(pts: set, direction: (int, int), start: (int, int))\
            -> int:
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
        while GomokuState.is_in_board(pt):
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
            If all possible positions are occupied, the game terminates
        """
        return self.reward != 0 or len(self.possible_actions) == 0

    def execute_action(self, action: GomokuAction) -> "GomokuState":
        """ Execute the action in a copy of the current state and return the new
            state
        :param action: Action(player, position)
        :return: The new state
        """
        if (action.position in self._history[0]
                or action.position in self._history[1]):
            raise ValueError("The position has already been taken")
        new_state = self.__copy__()
        new_state._history[action.player].append(action.position)
        new_state._player = 1 - self._player
        return new_state

    def go(self, position: (int, int)) -> "GomokuState":
        """ Place the piece at the given position in the current state
        :param position: The position to place the piece
        """
        if not GomokuState.is_in_board(position):
            raise ValueError("The position is out of board")
        if position in self._history[0] or position in self._history[1]:
            raise ValueError("The position has already been taken")
        self._history[self.player].append(position)
        self._player = 1 - self._player
        return self

    def visualize(self, size: (int, int) = (15, 15), grid_width: float = 1.5,
                  buffer_size: float = 0.5, piece_radius: float = 0.35,
                  tick_font_size: int = 16, title_font_size: int = 24,
                  count_font_size: int = 30) -> Figure:
        """ Visualize the state of the Gomoku game
        :param size: The size of the figure
        :param grid_width: The width of the board grid
        :param buffer_size: The size beyond the board to display
        :param piece_radius: The size of a piece
        :param tick_font_size: The font size for x- and y-ticks
        :param title_font_size: The font size of the title
        :param count_font_size: The font size for the order of pieces
        :return: The figure
        """
        if not (size[0] > 0 and size[1] > 0):
            raise ValueError("The figure size must be a tuple of two positive"
                             "numbers")

        # Initialization
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.axis('equal')
        ax.set_xlim(-buffer_size, GomokuState.board_size - 1 + buffer_size)
        ax.set_ylim(-buffer_size, GomokuState.board_size - 1 + buffer_size)
        plt.xticks(range(GomokuState.board_size),
                   range(1, GomokuState.board_size + 1))
        plt.yticks(range(GomokuState.board_size),
                   map(chr, range(65, 65 + GomokuState.board_size)))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font_size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font_size)
        plt.xlabel('')
        plt.ylabel('')

        # Grid
        plt.hlines(y=range(GomokuState.board_size), xmin=0,
                   xmax=GomokuState.board_size - 1, color='k',
                   linewidth=grid_width, zorder=0)
        plt.vlines(x=range(GomokuState.board_size), ymin=0,
                   ymax=GomokuState.board_size - 1, color='k',
                   linewidth=grid_width, zorder=0)

        # State
        title = "Gomoku\n"
        if self.is_terminal:
            if self.black_reward == 1:
                title += "Black Has Won"
            elif self.black_reward == -1:
                title += "White Has Won"
            else:
                title += "Tie"
        else:
            title += "Game in Progress"
        plt.title(title, fontsize=title_font_size)
        count = 1
        player = 0
        history = deepcopy(self._history)
        colors = ['black', 'white']
        while history[0] or history[1]:
            if history[player]:
                i, j = history[player].pop(0)
                ax.add_patch(Circle(xy=(i, j), radius=piece_radius,
                             edgecolor='k', facecolor=colors[player],
                                    linewidth=grid_width * 2, fill=True,
                                    zorder=1))
                plt.text(i, j, str(count),
                         fontsize=count_font_size, ha='center', va='center',
                         color=colors[1 - player], zorder=2)
            count += 1
            player = 1 - player

        plt.show()
        return fig
