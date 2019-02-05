from itertools import chain
import sys

import pygame
import torch as th
import fire

import src.config as config
from src.player import AlphaPlayer, BasicPlayer, HumanPlayer
from src.factory import create_gomoku, create_network
from src.torch_utils import device


pygame.init()
HEIGHT, WIDTH = 600, 600


class Client:
    def __init__(self, game):
        self.game = game
        self.board = None
        self.screen = None
        self.ready = False
        self.next_move = None

    def draw_board(self, board):
        self.screen.fill((200, 220, 220))
        for i in range(1, self.board_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * WIDTH // self.board_size), (HEIGHT, i * WIDTH // self.board_size))
            pygame.draw.line(self.screen, (0, 0, 0), (i * HEIGHT // self.board_size, 0), (i * HEIGHT // self.board_size, WIDTH))
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != 0:
                    color = (0, 0, 0) if board[i, j] == 1 else (255, 255, 255)
                    pygame.draw.ellipse(self.screen, color, pygame.Rect(
                        (j - 0.5) * WIDTH // self.board_size,
                        (i - 0.5) * HEIGHT // self.board_size,
                        WIDTH // self.board_size,
                        HEIGHT // self.board_size,
                    ))
        pygame.display.flip()

    def is_ready(self):
        return self.ready

    def run(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.board_size = config.BOARD_SIZE
        self.draw_board(self.get_board())
        # while 1:
        #     self.handle_events()
        #     self.draw_board(self.game.get_board())

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.click(event)

    def get_action(self, last_move, return_probs=False, temperature=0.1):
        self.ready = True
        self.next_move = None
        self.board = None
        while self.is_ready():
            self.handle_events()
            self.draw_board(self.get_board())
        assert self.next_move is not None
        return self.next_move

    def click(self, event):
        y, x = event.pos
        print(event.pos)
        x = int(x * self.board_size / WIDTH + 0.5)
        y = int(y * self.board_size / HEIGHT + 0.5)
        self.next_move = (x, y)
        self.ready = False
        event.pos = None

    def get_board(self):
        if self.board is None:
            self.board = self.game.get_board()
        return self.board


def main(ai, save_file=None, num_playouts=None):
    game = create_gomoku()
    client = Client(game)

    if ai == "alpha":
        network = create_network().to(device)
        network.load_state_dict(th.load(save_file))
        opponent = AlphaPlayer(game, network)
    elif ai == "basic":
        opponent = BasicPlayer(game, int(num_playouts), 0.05)
    else:
        raise ValueError(f"Player [{ai}] not supported yet")
    player = HumanPlayer(game, client.get_action)
    client.screen = pygame.display.set_mode((WIDTH, HEIGHT))
    client.board_size = config.BOARD_SIZE
    result = game.start_game([player, opponent])


if __name__ == "__main__":
    fire.Fire(main)
