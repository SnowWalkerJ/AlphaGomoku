import torch as th


if th.cuda.is_available():
    device = th.device("cuda: 0")
else:
    device = th.device("cpu")
