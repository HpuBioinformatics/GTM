import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def src_dot_dst(src_field, dst_field, out_field):
  
    def func(edges):
        return {out_field: edges.src[src_field] * edges.dst[dst_field]}
    return func


def scaling(field, scale_constant):
    """
    Scale attention logits before reduction.
    """
    def func(edges):
        return {field: edges.data[field] / scale_constant}
    return func


def imp_exp_attn(implicit_attn, explicit_edge, edge_bias_weight=0.1):
    """
    Add explicit edge bias to implicit attention logits.

    implicit_attn: [E, H, D]
    explicit_edge: [E, H, 1] or [E, H, D]
    """
    def func(edges):
        return {
            implicit_attn: edges.data[implicit_attn] + edge_bias_weight * edges.data[explicit_edge]
        }
    return func


def save_edge_output(edge_feat, out_name="e_out"):
    """
    Save edge features for downstream FFN_e.
    Usually this should be the pre-softmax edge-aware score, not exp(score).
    """
    def func(edges):
        return {out_name: edges.data[edge_feat]}
    return func


def exp_reduce_lastdim(field):
    """
    Reduce the last dimension to obtain one scalar attention logit per head,
    then exponentiate for softmax normalization.

    Input:
        edges.data[field]: [E, H, D] or [E, H, 1]
    Output:
        field: [E, H, 1]
    """
    def func(edges):
        logits = edges.data[field].sum(-1, keepdim=True).clamp(-5, 5)
        return {field: torch.exp(logits)}
    return func


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head graph attention with explicit edge bias.

    h input shape: [N, in_dim]
    e input shape: [E, in_dim]

    Internally:
        Q_h, K_h, V_h: [N, H, D]
        proj_e:        [E, H, 1]
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias=False, edge_bias_weight=0.1):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_bias_weight = edge_bias_weight

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        # More stable than projecting edge features to [H, D] and summing later.
        # One scalar bias per head is usually enough.
        self.proj_e = nn.Linear(in_dim, num_heads, bias=use_bias)

    def propagate_attention(self, g):
        # 1) Implicit node-node compatibility: [E, H, D]
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

        # 2) Scale
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # 3) Add explicit edge bias
        g.apply_edges(imp_exp_attn('score', 'proj_e', self.edge_bias_weight))

        # 4) Save pre-softmax edge-aware score for edge branch
        g.apply_edges(save_edge_output('score', out_name='e_out'))

        # 5) Convert to positive attention weights
        g.apply_edges(exp_reduce_lastdim('score'))  # [E, H, 1]

        # 6) Aggregate to destination nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'm'), fn.sum('m', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'm_score'), fn.sum('m_score', 'z'))

    def forward(self, g, h, e):
        with g.local_scope():
            Q_h = self.Q(h)
            K_h = self.K(h)
            V_h = self.V(h)
            proj_e = self.proj_e(e)

            g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
            g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

            # [E, H] -> [E, H, 1]
            g.edata['proj_e'] = proj_e.view(-1, self.num_heads, 1)

            self.propagate_attention(g)

            # [N, H, D] / [N, H, 1] -> [N, H, D]
            h_out = g.ndata['wV'] / (g.ndata['z'] + 1e-6)

            # Edge output keeps pre-softmax edge-aware score: [E, H, D]
            e_out = g.edata['e_out']

            return h_out, e_out


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.2,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
        edge_bias_weight=0.1,
    ):
        super().__init__()

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=use_bias,
            edge_bias_weight=edge_bias_weight,
        )

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for node features
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for edge features
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        h_in1 = h
        e_in1 = e

        # Multi-head attention
        h_attn_out, e_attn_out = self.attention(g, h, e)

        # [N, H, D] -> [N, out_dim]
        h = h_attn_out.view(-1, self.out_channels)
        # [E, H, D] -> [E, out_dim]
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

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

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.gelu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"heads={self.num_heads}, "
            f"residual={self.residual})"
        )
