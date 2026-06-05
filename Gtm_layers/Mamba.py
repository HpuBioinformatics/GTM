import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, dim, expand=2, dt_rank="auto", bias=False):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.hidden_dim = dim * expand
        self.dt_rank = dt_rank if dt_rank != "auto" else max(dim // 16, 1)

        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
            bias=bias,
        )

        self.x_proj = nn.Linear(
            self.hidden_dim,
            self.dt_rank + 2 * self.hidden_dim,
            bias=bias,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.hidden_dim, bias=bias)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        self.act = nn.SiLU()

    def forward(self, x):
        squeeze_back = False
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1, L, D]
            squeeze_back = True

        # [B, L, D] -> [B, L, hidden_dim]
        x = self.in_proj(x)

        # depthwise conv over sequence dimension
        x = x.transpose(1, 2)          # [B, hidden_dim, L]
        x = self.conv1d(x)
        x = x.transpose(1, 2)          # [B, L, hidden_dim]
        x = self.act(x)

        # simplified selective update
        x_proj = self.x_proj(x)
        dt, A, B_proj = torch.split(
            x_proj,
            [self.dt_rank, self.hidden_dim, self.hidden_dim],
            dim=-1,
        )

        dt = self.dt_proj(dt)
        dt = torch.sigmoid(dt) * 0.1

        A = torch.sigmoid(A)
        B_proj = self.act(B_proj)

        x = dt * (A * x + B_proj)

        x = self.out_proj(x)

        if squeeze_back:
            x = x.squeeze(0)

        return x


class GraphMambaLayer(nn.Module):
  
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout=0.2,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        expand=2,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.node_mamba = MambaBlock(dim=out_dim, expand=expand)
        self.edge_mamba = MambaBlock(dim=out_dim, expand=expand)

        self.proj_h = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.proj_e = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        # input projection
        h = self.proj_h(h)
        e = self.proj_e(e)

        h_in1 = h
        e_in1 = e

        # graph message aggregation for node branch
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["e"] = e

            g.update_all(
                fn.u_add_e("h", "e", "m"),
                fn.sum("m", "h_neigh"),
            )

            h_neigh = g.ndata["h_neigh"]
            h = h + h_neigh

        # mamba blocks
        h = self.node_mamba(h)
        e = self.edge_mamba(e)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            h = h_in1 + h
            e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h
        e_in2 = e

        h = self.FFN_h_layer1(h)
        h = F.gelu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        e = self.FFN_e_layer1(e)
        e = F.gelu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e
