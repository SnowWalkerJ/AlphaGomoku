from enum import Enum, IntEnum
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'probs', 'z'))


class Color(IntEnum):
    Black = 1
    White = -1

    def get_opposite(self):
        if self.value == 1:
            return Color.White
        else:
            return Color.Black

    @staticmethod
    def from_int(value):
        if value == 1:
            return Color.Black
        elif value == -1:
            return Color.White
        else:
            return None


class Outcome(Enum):
    Win = 0
    Loss = 1
    Tie = 2
    Nothing = 3
