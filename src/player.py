from functools import partial
from abc import ABC, abstractmethod
import random

import numpy as np
import torch as th
import torch.nn as nn
from src.constant import Color
from src.mcts import MCTS
from src.utils import board_to_state, available_moves, check_validation
import src.config as config


class Player(ABC):
    def __init__(self, game):
        self._game = game

    @abstractmethod
    def get_action(self, last_move, return_probs: bool=False, temperature: int=1):
        pass


class AlphaPlayer(Player):
    def __init__(self, game, network: nn.Module, c_puct: float=5):
        self._network = network
        network.eval()
        self._c_puct = c_puct
        self._mcts = MCTS(partial(self.policy_value_fn, to_dict=True), c_puct)
        self._self_play = False
        super(AlphaPlayer, self).__init__(game)

    def get_action(self, last_move, return_probs=False, temperature=0.1):
        if last_move:
            self._mcts.update_with_move(last_move)
        action_probs = self._mcts.get_action_probs(self._game, num_playouts=config.NUM_PLAYOUTS, temperature=temperature)
        actions, probs = zip(*action_probs.items())
        if self._self_play:
            i = np.random.choice(np.arange(len(actions)), p=np.asarray(probs)*0.75+np.random.dirichlet(0.3*np.ones(len(actions)))*0.25)
        else:
            i = np.random.choice(np.arange(len(actions)), p=np.asarray(probs))
        action = actions[i]
        self._mcts.update_with_move(action)
        if return_probs:
            full_probs = np.zeros(config.BOARD_SIZE*config.BOARD_SIZE)
            rows, cols = zip(*actions)
            full_probs[np.array(rows) * config.BOARD_SIZE + np.array(cols)] = np.array(probs)
            return action, full_probs.ravel()
        else:
            return action

    def set_self_play(self, value: bool):
        self._self_play = value

    def policy_value_fn(self, board, to_dict=False):
        x = board_to_state(board)
        x = th.tensor(x).float().to(self._network.device).unsqueeze(0)
        with th.no_grad():
            prior_probs, value = self._network(x)
        if to_dict:
            prior_probs = prior_probs.cpu().view(*board.shape).numpy()
            moves = available_moves(board)
            rows, cols = zip(*moves)
            prior_probs = dict(zip(moves, prior_probs[np.array(rows), np.array(cols)]))
            value = value[0, 0].item()
        return prior_probs, value

    def reset(self):
        self._mcts = MCTS(partial(self.policy_value_fn, to_dict=True), self._c_puct)


class BasicPlayer(Player):
    def __init__(self, game, num_playouts: int, c_puct: float=5):
        self.num_playouts = num_playouts
        self._mcts = MCTS(partial(self.policy_value_fn, to_dict=True), c_puct)
        self._self_play = False
        self._c_puct = c_puct
        super(BasicPlayer, self).__init__(game)

    def get_action(self, last_move, return_probs=False, temperature=0.1):
        if last_move:
            self._mcts.update_with_move(last_move)
        action_probs = self._mcts.get_action_probs(self._game, num_playouts=self.num_playouts, temperature=temperature)
        actions, probs = zip(*action_probs.items())
        i = np.random.choice(np.arange(len(actions)), p=np.asarray(probs))
        print(actions, probs, i)
        action = actions[i]
        self._mcts.update_with_move(action)
        if return_probs:
            full_probs = np.zeros(config.BOARD_SIZE*config.BOARD_SIZE)
            rows, cols = zip(*actions)
            full_probs[np.array(rows) * config.BOARD_SIZE + np.array(cols)] = np.array(probs)
            return action, full_probs.ravel()
        else:
            return action

    def policy_value_fn(self, board, to_dict=False):
        value = 0
        moves = available_moves(board)
        p = 1 / len(moves)
        prior_probs = {move: p for move in moves}
        return prior_probs, value

    def reset(self):
        self._mcts = MCTS(partial(self.policy_value_fn, to_dict=True), self._c_puct)


class HumanPlayer(Player):
    def __init__(self, game, get_action):
        super().__init__(game)
        self.__get_action = get_action

    def get_action(self, last_move, return_probs=False, temperature=0.1):
        return self.__get_action(last_move, return_probs, temperature)
