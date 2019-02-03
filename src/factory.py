import src.config as config
from src.net import ResidualAlpha
from src.game import Gomoku
import src.config as config

def create_gomoku():
    return Gomoku(config.BOARD_SIZE, config.WIN_LENGTH)


def create_network():
    network = ResidualAlpha(config.NUM_RESIDUALS, config.BOARD_SIZE)
    return network
