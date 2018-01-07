import numpy as np


class Gomoku:
    def __init__(self, board_size: int, win_length: int=5):
        self.board_size = board_size
        self.win_length = win_length
        self.board = self.make_new_board()

    def make_new_board(self):
        return np.zeros((self.board_size, self.board_size), dtype=np.float32)