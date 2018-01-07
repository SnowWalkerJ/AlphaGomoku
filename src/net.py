import torch
import torch.nn as nn
import torch.nn.functional as F
import src.config as config
 

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=config.RESIDUAL_KERNELS, out_channels=config.RESIDUAL_KERNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=config.RESIDUAL_KERNELS),
            nn.ReLU(),
            nn.Conv2d(in_channels=config.RESIDUAL_KERNELS, out_channels=config.RESIDUAL_KERNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=config.RESIDUAL_KERNELS),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        y = self.relu(x + res)
        return y


class PolicyHead(nn.Module):
    def __init__(self, board_size):
        super(PolicyHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=config.RESIDUAL_KERNELS, out_channels=2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x


class ValueHead(nn.Module):
    def __init__(self, board_size):
        super(ValueHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=config.RESIDUAL_KERNELS, out_channels=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(board_size*board_size, config.RESIDUAL_KERNELS),
            nn.ReLU(),
            nn.Linear(config.RESIDUAL_KERNELS, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


class ResidualAlpha(nn.Module):
    def __init__(self, num_residuals, board_size):
        super(ResidualAlpha, self).__init__()
        conv_bottom = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config.RESIDUAL_KERNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=config.RESIDUAL_KERNELS),
            nn.ReLU()
        )
        residual_blocks = [ResidualBlock() for _ in range(num_residuals)]
        self.residual_blocks = nn.Sequential(conv_bottom, *residual_blocks)
        self.policy_head = PolicyHead(board_size)
        self.value_head  = ValueHead(board_size)

    def forward(self, x):
        x = self.residual_blocks(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return policy, value
