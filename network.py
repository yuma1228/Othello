import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_size=6, num_res_blocks=5, num_filters=64, input_channels=3):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.initial_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(self, state):
        """単一盤面の推論 (MCTS用)"""
        self.eval()
        with torch.no_grad():
            x = state.unsqueeze(0)
            policy_logits, value = self.forward(x)
            policy = F.softmax(policy_logits, dim=1).squeeze(0)
            return policy, value.item()
