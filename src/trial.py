from gomoku import *
from mcts import *

state = GomokuState()
state.go((GomokuState.board_size // 2, GomokuState.board_size // 2))
state.go((GomokuState.board_size // 2 + 1, GomokuState.board_size // 2))
state.go((GomokuState.board_size // 2, GomokuState.board_size // 2 + 1))
state.go((GomokuState.board_size // 2 + 1, GomokuState.board_size // 2 + 1))
mcts = MonteCarloSearchTree(samples=200)
action = mcts.search_for_action(state)
print(action)
