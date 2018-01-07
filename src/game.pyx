import copy
import warnings
from typing import List, Tuple
import numpy as np
import src.exceptions as exceptions
from src.player import Player
from src.constant import Color, Outcome, Transition
from src.utils import place_move, board_to_state
from src.exceptions import Occupied
import src.config as config
cimport numpy as np
cimport cython

cdef class Gomoku:
    cdef int board_size, win_length
    cdef list moves
    cdef dict players
    cdef current_color, board
    def __init__(self, int board_size, int win_length):
        self.board_size    = board_size
        self.win_length    = win_length
        self.current_color = None
        self.board         = None
        self.moves         = None
        self.players       = None

    def __getstate__(self):
        return {
            'board_size': self.board_size,
            'win_length': self.win_length,
        }
    
    def __setstate__(self, state):
        self.board_size = state['board_size']
        self.win_length = state['win_length']
        self.board = self.current_color = self.moves = self.players = None

    cdef play_one_move(self, last_move):
        move = self.current_player.get_action(last_move)
        place_move(self.board, move, self.current_color)
        self.moves.append(move)

    cpdef start_self_play(self, player, float temperature=0.1):
        cdef int n
        data = {
            'states': [],
            'probs': [],
        }
        current_color = Color.Black
        player.set_self_play(True)
        player.reset()
        self.reset()
        outcome = Outcome.Nothing
        while outcome == outcome.Nothing:
            data['states'].append(board_to_state(self.board))
            move, probs = player.get_action(None, return_probs=True, temperature=temperature)
            data['probs'].append(np.array(probs))
            place_move(self.board, move, current_color)
            outcome = self.check_outcome_fast(self.board, move, current_color)
            current_color = current_color.get_opposite()

        if outcome == outcome.Tie:
            z = [0.5] * len(data['states'])
        elif outcome == outcome.Win:
            z = tuple(reversed([1-2*(i % 2) for i in range(len(data['states']))]))
        elif outcome == outcome.Loss:
            z = tuple(reversed([2*(i % 2)-1 for i in range(len(data['states']))]))
        data['z'] = z
        return [Transition(state, prob, z) for state, prob, z in zip(data["states"], data["probs"], data["z"])]

    cpdef start_game(self, players: List[Player]):
        self.players = {
            Color.Black: players[0],
            Color.White: players[1],
        }
        self.reset()
        while 1:
            last_move = self.moves[-1] if self.moves else None
            self.play_one_move(last_move)
            outcome = self.check_outcome_fast(self.board, self.moves[-1], self.current_color)
            if outcome == Outcome.Win:
                return self.current_color
            elif outcome == Outcome.Loss:
                return self.current_color.get_opposite()
            elif outcome == Outcome.Tie:
                return None
            self.current_color = self.current_color.get_opposite()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef check_outcome_fast(self, np.ndarray[np.int8_t, ndim=2] board, move, current_color):
        cdef int n, t, x0, y0, dx, dy, x, y
        cdef int color_value = current_color.value
        if (board == 0).sum() == 0:
            return Outcome.Tie
        y0, x0 = move
        for dy, dx in [(1, 0), (0, 1), (1, -1), (1, 1)]:
            n = 1
            t = 1
            x, y = t * dx + x0, t * dy + y0
            while max(x, y) < config.BOARD_SIZE and min(x, y) >= 0 and board[y, x] == color_value:
                t += 1
                n += 1
                x, y = t * dx + x0, t * dy + y0
            t = -1
            x, y = t * dx + x0, t * dy + y0
            while max(x, y) < config.BOARD_SIZE and min(x, y) >= 0 and board[y, x] == color_value:
                t -= 1
                n += 1
                x, y = t * dx + x0, t * dy + y0
            if n == config.WIN_LENGTH:
                return Outcome.Win
            elif n > config.WIN_LENGTH:
                return Outcome.Loss
        return Outcome.Nothing

    def save(self, filename):
        # TODO: save game
        raise NotImplementedError

    @property
    def current_player(self) -> Player:
        return self.players[self.current_color]

    def get_board(self) -> np.ndarray:
        return self.board.copy()

    def get_moves(self) -> List[Tuple[int]]:
        return copy.copy(self.moves)

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.moves = []
        self.current_color = Color.Black
