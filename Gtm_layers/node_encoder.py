import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
 
    def __init__(self,hidden_channels, out_channels,in_channels=4, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
