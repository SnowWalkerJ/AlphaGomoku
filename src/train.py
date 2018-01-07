from src.trainer import Trainer
import src.config as config
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    trainer = Trainer(config.NUM_SELFPLAYERS)
    trainer.start()