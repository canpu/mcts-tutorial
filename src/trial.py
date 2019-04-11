from mcts import *
from gomoku import *

S1 = GomokuState()
S2 = GomokuState().go((1, 3))
A = Node(S1)
B = Node(S2)
B.parent = A
print(B.depth)
