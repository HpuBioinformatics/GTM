import torch
import torch.nn as nn

class NodeGatedFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Linear(2 * in_dim, in_dim)
    def forward(self, h1, h2):
        concat = torch.cat([h1, h2], dim=1)
        gate = torch.sigmoid(self.gate(concat))
        return gate * h1 + (1 - gate) * h2

class EdgeGatedFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Linear(2 * in_dim, in_dim)
    def forward(self, e1, e2):
        concat = torch.cat([e1, e2], dim=1)
        gate = torch.sigmoid(self.gate(concat))
        return gate * e1 + (1 - gate) * e2