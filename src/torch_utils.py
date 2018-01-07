import torch


USE_CUDA = torch.cuda.is_available()


class Variable(torch.autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(self, data, *args, **kwargs)
