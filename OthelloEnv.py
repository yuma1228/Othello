import torch
from SizedOthello import SizedOthello

SIZE = 6


def encode_state(othello: SizedOthello, color: int) -> torch.Tensor:
    """盤面を3ch tensor に変換 (自分の石, 相手の石, 手番)"""
    state = torch.zeros(3, SIZE, SIZE, dtype=torch.float32)
    opponent = 3 - color
    for y in range(SIZE):
        for x in range(SIZE):
            if othello.board[y][x] == color:
                state[0][y][x] = 1.0
            elif othello.board[y][x] == opponent:
                state[1][y][x] = 1.0
    if color == 1:
        state[2] = 1.0
    return state


def get_legal_moves_mask(othello: SizedOthello, color: int) -> torch.Tensor:
    """合法手のマスク (36,)"""
    mask = torch.zeros(SIZE * SIZE, dtype=torch.float32)
    for x, y in othello.possible_puts(color):
        mask[y * SIZE + x] = 1.0
    return mask


def action_to_xy(action: int) -> tuple[int, int]:
    y = action // SIZE
    x = action % SIZE
    return x, y


def xy_to_action(x: int, y: int) -> int:
    return y * SIZE + x
