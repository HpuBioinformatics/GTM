from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
from Gtm_layers.Transformer import GraphTransformerLayer
from Gtm_layers.Mamba import GraphMambaLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

import Gtm_layers as layers


class SymGatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, normalization, dropout=None):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.SymGatedGCN(hidden_features, hidden_features, normalization, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for conv in self.convs:
            h, e = conv(graph, h, e)
        return h, e


class GatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, normalization, dropout=None):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GatedGCN(hidden_features, hidden_features, normalization, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for conv in self.convs:
            h, e = conv(graph, h, e)
        return h, e


class GCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            GraphConv(hidden_features, hidden_features, weight=True, bias=True)
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs) - 1):
            h = F.gelu(self.convs[i](graph, h))
        h = self.convs[-1](graph, h)
        return h, e


class GAT_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout=0.0, num_heads=3):
        super().__init__()
        dropout = 0.0 if dropout is None else dropout
        self.num_heads = num_heads

        self.convs = nn.ModuleList([
            GATConv(
                hidden_features,
                hidden_features,
                num_heads=self.num_heads,
                feat_drop=dropout,
                attn_drop=dropout
            )
            for _ in range(num_layers)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(self.num_heads * hidden_features, hidden_features)
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs) - 1):
            heads = self.convs[i](graph, h)  # [N, H, D]
            h = heads.flatten(1)             # [N, H*D]
            h = self.linears[i](h)
            h = F.gelu(h)

        heads = self.convs[-1](graph, h)
        h = heads.flatten(1)
        h = self.linears[-1](h)
        return h, e


class SAGE_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout=0.0):
        super().__init__()
        dropout = 0.0 if dropout is None else dropout
        self.convs = nn.ModuleList([
            SAGEConv(hidden_features, hidden_features, "mean", feat_drop=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs) - 1):
            h = F.gelu(self.convs[i](graph, h))
        h = self.convs[-1](graph, h)
        return h, e


class GT_processor(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_features,
        num_heads=8,
        dropout=0.2,
        layer_norm=True,
        batch_norm=False,
        residual=True,
        use_bias=False,
        edge_bias_weight=0.1,
    ):
        super().__init__()

        assert hidden_features % num_heads == 0, (
            f"hidden_features ({hidden_features}) must be divisible by num_heads ({num_heads})"
        )

        self.convs = nn.ModuleList([
            GraphTransformerLayer(
                in_dim=hidden_features,
                out_dim=hidden_features,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                residual=residual,
                use_bias=use_bias,
                edge_bias_weight=edge_bias_weight,
            )
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for conv in self.convs:
            h, e = conv(graph, h, e)
        return h, e


class GM_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout=0.2, batch_norm=True):
        super().__init__()
        self.convs = nn.ModuleList([
            GraphMambaLayer(
                in_dim=hidden_features,
                out_dim=hidden_features,
                dropout=dropout,
                batch_norm=batch_norm,
            )
            for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for conv in self.convs:
            h, e = conv(graph, h, e)
        return h, e
