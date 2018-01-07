import numpy as np
from src.constant import Color
import src.exceptions as exceptions
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int8_t, ndim=2] place_move(np.ndarray[np.int8_t, ndim=2] board, move, color):
    check_validation(board, move)
    board[move] = color.value
    return board


@cython.boundscheck(False)
@cython.wraparound(False)
def check_validation(np.ndarray[np.int8_t, ndim=2] board, move: tuple):
    cdef int board_size = board.shape[0], x, y
    y, x = move
    if not 0 <= x < board_size:
        raise exceptions.OutOfRange("X coordinate `{}` out of range [0, {})".format(x, board_size))
    if not 0 <= y < board_size:
        raise exceptions.OutOfRange("Y coordinate `{}` out of range [0, {})".format(y, board_size))
    color = Color.from_int(int(board[y, x]))
    if color:
        raise exceptions.Occupied("Coordinate ({x}, {y}) is already occupied with {color}\n{board}".format(
            x=x, y=y, color=color.name, board=board
        ))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list available_moves(np.ndarray[np.int8_t, ndim=2] board):
    cdef np.ndarray[np.int64_t, ndim=1] row, col
    row, col = np.where(board == 0)
    return list(zip(row, col))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=3] board_to_state(np.ndarray[np.int8_t, ndim=2] board):
    # cdef np.ndarray[bint, ndim=2] b, w, c
    b = board > 0
    w = board < 0
    c = (np.sum(board > 0) - np.sum(board < 0)) * np.ones_like(board)
    state = np.stack([b, w, c], 0)
    return np.asarray(state, dtype=np.float32)
