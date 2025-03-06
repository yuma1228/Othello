import numpy as np

from SizedOthello import SizedOthello

State = list[list[list[int]]]
SIZE=4


def decode_state(state:State)->SizedOthello:
    othello=SizedOthello(SIZE)
    color=state[2][0][0]
    othello.color=color
    othello.board=[[0]*SIZE for _ in range(SIZE)]
    for x in range(othello.size):
            for y in range(othello.size):
                if state[0][y][x]==1:
                    othello.board[y][x]=1
                elif state[1][y][x]==1:
                    othello.board[y][x]=2
    return othello



