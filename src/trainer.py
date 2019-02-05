import queue
import random
import uuid
from collections import deque
from time import sleep

import numpy as np
import torch as th
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorboardX import SummaryWriter

import src.config as config
from src.torch_utils import device
from src.constant import Color
from src.factory import create_gomoku, create_network
from src.player import AlphaPlayer, BasicPlayer


class SelfPlayer(mp.Process):
    def __init__(self, game, player, network, target_network, queue):
        self.game = game
        self.target_network = target_network
        self.player = player
        self.queue = queue
        self.network = network
        super(SelfPlayer, self).__init__()

    def sync_network(self):
        self.network.load_state_dict(self.target_network.state_dict())

    def run(self):
        n = 0
        while 1:
            self.sync_network()
            data = self.game.start_self_play(self.player, temperature=0.99 ** (n/50))
            self.queue.put(data)
            n += 1


class Evaluator(mp.Process):
    def __init__(self, game, network, target_network, trigger, queue, num_rounds: int):
        super(Evaluator, self).__init__()
        self.network = network
        self.queue = queue
        self.num_rounds = num_rounds
        self.trigger = trigger
        self.target_network = target_network
        self.game = game
        self.highest_score = 0
        self.level = 0
        self.win_count_series = deque(maxlen=50)

    def sync_network(self):
        """
        copy the parameters from the target network
        """
        self.network.load_state_dict(self.target_network.state_dict())

    def evaluate(self, player_first: bool):
        """
        Play a game with the basic player (pure MCTS) and returns the average
        winning rate (1 for a win and 0.5 for a tie)
        """
        num_playouts = 3000
        player = AlphaPlayer(self.game, self.network, c_puct=5)
        coach = BasicPlayer(self.game, num_playouts)
        players = [player, coach] if player_first else [coach, player]
        color = Color.Black if player_first else Color.White
        outcome = self.game.start_game(players)
        if outcome is None:
            return 0.5
        elif outcome == color:
            return 1
        else:
            return 0

    def run(self):
        i = 0
        while 1:
            self.trigger.wait()
            self.sync_network()
            score = self.evaluate(i % 2)
            self.win_count_series.append(score)
            average_score = np.mean(self.win_count_series)
            i += 1
            if i > 20 and i % 10 == 0:
                th.save(self.network.state_dict(), "models/model.ckpt.%03d" % (i // 10))
                self.queue.put((i // 10, average_score))


class Trainer:
    def __init__(self, num_processes: int):
        self.writer = SummaryWriter("tensorboard/{}".format(uuid.uuid4()))
        self.queue = mp.Queue()
        self.evaluation_queue = mp.Queue()
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.trigger = mp.Event()
        self.network = create_network().to(th.device("cuda: 0"))
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
        self.scheduler = LambdaLR(self.optimizer, self.scheduler_fn)
        self.evaluator = Evaluator(create_gomoku(), create_network().to(th.device("cuda: 0")), self.network, self.trigger, self.evaluation_queue, num_rounds=config.NUM_ROUNDS)
        self.selfplayers = []
        self.rounds_selfplay = 0
        self.loss = None
        self.steps = 0
        for i in range(num_processes):
            game = create_gomoku()
            network = create_network().to(th.device("cuda: 1"))
            player = AlphaPlayer(game, network)
            selfplayer = SelfPlayer(game, player, network, self.network, self.queue)
            self.selfplayers.append(selfplayer)

    @staticmethod
    def scheduler_fn(step):
        epoch = step * 1e-5
        if epoch < 4:
            return 1
        elif epoch < 6:
            return 0.1
        else:
            return 0.01

    def start(self):
        """
        1. start all the self-players and the estimator
        2. collect self-play data until the buffer is full
        3. try to update the buffer with new data; periodically trigger estimator; 
        and train the network
        """
        for selfplayer in self.selfplayers:
            selfplayer.start()
        self.evaluator.start()

        print("Collecting self play data")
        bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()])
        bar.start(max_value=config.BUFFER_SIZE)
        while len(self.buffer) < config.BUFFER_SIZE:
            self.buffer.extend(self.queue.get())
            bar.update(len(self.buffer))
        bar.finish()
        print("Buffer filled.")

        self.trigger.set()

        while 1:
            self.get_data()
            self.train_once(batch_size=config.BATCH_SIZE)
            sleep(0.1) # Lower training frequency to allow the self-players have more resource
            try:
                step, score = self.evaluation_queue.get_nowait()
                self.writer.add_scalar("WinRate", score, global_step=step)
                self.writer.add_scalar("Loss", self.loss, global_step=step)
                self.writer.add_scalar("Train Steps", self.steps, global_step=step)
                self.steps = 0
            except queue.Empty:
                pass
        
    def get_data(self):
        try:
            while 1:
                data = self.queue.get_nowait()
                self.rounds_selfplay += 1
                self.buffer.extend(data)
        except queue.Empty:
            pass

    def train_once(self, batch_size: int, l2_c: float=1e-4):
        device = th.device("cuda: 0")
    
        batch_data = random.sample(self.buffer, batch_size)
        states, probs, z = zip(*batch_data)
        states, probs = self.transform(states, probs)
        v_state = th.tensor(np.stack(states, 0)).to(device)

        probs_, z_ = self.network(v_state)
        loss_entropy = -(th.tensor(probs).to(device).float() * probs_.log()).sum(1).mean()
        loss_value = (th.tensor(z).to(device).float() - z_).pow(2).mean()
        loss_l2 = sum(p.pow(2).sum() for p in self.network.parameters())
        loss = loss_entropy + loss_value + l2_c * loss_l2
        self.loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.scheduler.step()
        self.optimizer.step()
        self.steps += 1

    @staticmethod
    def transform(states, probs):
        """
        Randomly flips and rotates the board, to force the invariance of rotation and flipping.
        """
        data = []
        for state, prob in zip(states, probs):
            transform_flip = random.choice((True, False))
            transform_rotate = random.choice((0, 1, 2, 3))
            if transform_flip or transform_rotate:
                prob = prob.reshape(*state.shape[1:])
                if transform_flip:
                    prob = prob[:, ::-1]
                    state = state[:, :, ::-1]
                if transform_rotate:
                    prob = np.rot90(prob, transform_rotate)
                    state = np.rot90(state, transform_rotate, axes=(1, 2))
                prob = prob.ravel()
            data.append((state, prob))
        states, probs = zip(*data)
        return states, probs
