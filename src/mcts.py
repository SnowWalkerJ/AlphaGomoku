import random

import numpy as np
from src.utils import place_move, available_moves, check_validation
from src.constant import Color, Outcome
import src.config as config


class Node:
    def __init__(self, parent, prior_prob: float):
        self._parent = parent
        self._prior_prob = prior_prob
        self._q_numerator = 0
        self._visit_count = 0
        self._children = {}

    def select(self, c: float):
        return max(self._children.items(), key=lambda item: item[1].get_value(c) + random.random() * 0.001)

    def expand(self, priors):
        for action, prior_prob in priors.items():
            self._children[action] = Node(self, prior_prob)

    def update(self, leaf_value):
        self._q_numerator += leaf_value
        self._visit_count += 1

    def recursive_update(self, leaf_value):
        if not self.is_root():
            self._parent.recursive_update(-leaf_value)
        self.update(leaf_value)

    @property
    def Q(self):
        return self._q_numerator / max(1, self._visit_count)

    def get_value(self, c_puct):
        U = self._prior_prob / (1 + self._visit_count) * self._parent.visit_count ** 0.5
        return self.Q + c_puct * U

    def is_leaf(self) -> bool:
        return not self._children

    def is_root(self) -> bool:
        return self._parent is None

    def as_root(self):
        self._parent = None
        return self

    @property
    def visit_count(self):
        return self._visit_count

    @property
    def children(self):
        return self._children


class MCTS:
    def __init__(self, policy_value_fn, c_puct: float):
        self._root = Node(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct

    def _playout_once(self, game):
        board = game.get_board()
        color = Color.Black if board.sum() == 0 else Color.White
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            place_move(board, action, color)
            color = color.get_opposite()
        try:
            outcome = game.check_outcome_fast(board, action, color.get_opposite())
        except NameError:
            outcome = Outcome.Tie if (board == 0).sum() == 0 else Outcome.Nothing

        if outcome == Outcome.Nothing:
            prior_probs, leaf_value = self._policy_value_fn(board)
            leaf_value = - leaf_value
            assert len(prior_probs) > 0
            node.expand(prior_probs)
        elif outcome == Outcome.Tie:
            leaf_value = 0.0
        elif outcome == Outcome.Win:
            leaf_value = 1.0
        elif outcome == Outcome.Loss:
            leaf_value = -1.0
        else:
            raise TypeError("Unexpected outcome value `{}`".format(outcome))
        node.recursive_update(leaf_value)

    def get_action_probs(self, game, num_playouts: int, temperature: float) -> dict:
        for _ in range(num_playouts):
            self._playout_once(game)
        assert not self._root.is_leaf()
        action_visits = [(action, node.visit_count) for action, node in self._root.children.items()]
        action, visit_count = zip(*action_visits)
        scaled_visit_count = np.array(visit_count) ** (1 / temperature)
        probs = scaled_visit_count / scaled_visit_count.sum()
        return dict(zip(action, probs))

    def update_with_move(self, move):
        if move in self._root.children:
            self._root = self._root.children[move].as_root()
        else:
            self._root = Node(None, 1.0)
