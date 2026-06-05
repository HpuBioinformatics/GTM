import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import Gtm_layers as layers

from Gtm_layers.processor import GT_processor
from Gtm_layers.processor import GM_processor
from Gtm_layers.fusion import NodeGatedFusion
from Gtm_layers.fusion import EdgeGatedFusion


class SymGatedGCNModel(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_layers,
        hidden_edge_scores,
        normalization,
        dropout=None,
    ):
        super().__init__()
        self.linear1_node = nn.Linear(node_features, hidden_ne_features, bias=True)
        self.linear2_node = nn.Linear(hidden_ne_features, hidden_features, bias=True)
        self.linear1_edge = nn.Linear(edge_features, hidden_ne_features, bias=True)
        self.linear2_edge = nn.Linear(hidden_ne_features, hidden_features, bias=True)

        self.gnn = layers.SymGatedGCN_processor(
            num_layers, hidden_features, normalization, dropout=dropout
        )
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)
        self.act = nn.GELU()

    def forward(self, graph, x, e, pe=None):
        x = self.linear2_node(self.act(self.linear1_node(x)))
        e = self.linear2_edge(self.act(self.linear1_edge(e)))
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class GatedGCNModel(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_layers,
        hidden_edge_scores,
        normalization,
        dropout=None,
        directed=True,
    ):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(
            hidden_ne_features, hidden_features, node_features
        )
        self.edge_encoder = layers.EdgeEncoder(
            hidden_ne_features, hidden_features, edge_features
        )
        self.gnn = layers.GatedGCN_processor(
            num_layers, hidden_features, normalization, dropout=dropout
        )
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe=None):
        x_gcn, e_gcn = self.get_features(graph, x, e)
        scores = self.predictor(graph, x_gcn, e_gcn)
        return scores

    def get_features(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)

        if self.directed:
            x, e = self.gnn(graph, x, e)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            e = torch.cat((e, e), dim=0)
            x, e = self.gnn(g, x, e)
            e = e[:graph.num_edges()]

        return x, e


class GCNModel(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_layers,
        hidden_edge_scores,
        normalization,
        dropout=None,
        directed=True,
    ):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(
            hidden_ne_features, hidden_features, node_features
        )
        self.edge_encoder = layers.EdgeEncoder(
            hidden_ne_features, hidden_features, edge_features
        )
        self.gnn = layers.GCN_processor(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe=None):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class GATModel(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_layers,
        hidden_edge_scores,
        normalization,
        dropout=None,
        directed=True,
    ):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(
            hidden_ne_features, hidden_features, node_features
        )
        self.edge_encoder = layers.EdgeEncoder(
            hidden_ne_features, hidden_features, edge_features
        )
        self.gnn = layers.GAT_processor(
            num_layers, hidden_features, dropout=dropout, num_heads=3
        )
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe=None):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class SAGEModel(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_layers,
        hidden_edge_scores,
        normalization,
        dropout=None,
        directed=True,
    ):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(
            hidden_ne_features, hidden_features, node_features
        )
        self.edge_encoder = layers.EdgeEncoder(
            hidden_ne_features, hidden_features, edge_features
        )
        self.gnn = layers.SAGE_processor(num_layers, hidden_features, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe=None):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class MyTransformerModelNet(nn.Module):
    def __init__(self, hidden_features, num_layers, nb_pos_enc):
        super().__init__()
        self.linear_pe = nn.Linear(nb_pos_enc, hidden_features)
        self.linear_fuse = nn.Linear(2 * hidden_features, hidden_features)
        self.fuse_norm = nn.LayerNorm(hidden_features)

        self.graph_transformer = GT_processor(
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_heads=8,
            dropout=0.2,
            layer_norm=True,
            batch_norm=False,
            residual=True,
            use_bias=False,
            edge_bias_weight=0.1,
        )
        self.act = nn.GELU()

    def get_features(self, graph, x, e, pe):
        pe_proj = self.linear_pe(pe)
        x = torch.cat([x, pe_proj], dim=-1)
        x = self.act(self.linear_fuse(x))
        x = self.fuse_norm(x)
        x, e = self.graph_transformer(graph, x, e)
        return x, e


class MyMambaModelNet(nn.Module):
    def __init__(self, hidden_features, num_layers, nb_pos_enc):
        super().__init__()
        self.linear_pe = nn.Linear(nb_pos_enc, hidden_features)
        self.linear_fuse = nn.Linear(2 * hidden_features, hidden_features)
        self.fuse_norm = nn.LayerNorm(hidden_features)

        self.graph_mamba = GM_processor(
            num_layers=num_layers,
            hidden_features=hidden_features,
            dropout=0.2,
            batch_norm=True,
        )
        self.act = nn.GELU()

    def get_features(self, graph, x, e, pe):
        pe_proj = self.linear_pe(pe)
        x = torch.cat([x, pe_proj], dim=-1)
        x = self.act(self.linear_fuse(x))
        x = self.fuse_norm(x)
        x, e = self.graph_mamba(graph, x, e)
        return x, e


class FusionModelNet(nn.Module):
    """
    Full dual-branch fusion:
    Transformer + Mamba + gated fusion
    """
    def __init__(
        self,
        hidden_features,
        hidden_edge_features,
        num_transformer_layers,
        num_mamba_layers,
        hidden_edge_scores,
        nb_pos_enc,
    ):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_edge_features),
            nn.GELU(),
            nn.Linear(hidden_edge_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )

        self.transformer_model = MyTransformerModelNet(
            hidden_features=hidden_features,
            num_layers=num_transformer_layers,
            nb_pos_enc=nb_pos_enc,
        )
        self.mamba_model = MyMambaModelNet(
            hidden_features=hidden_features,
            num_layers=num_mamba_layers,
            nb_pos_enc=nb_pos_enc,
        )

        self.node_fusion = NodeGatedFusion(hidden_features)
        self.edge_fusion = EdgeGatedFusion(hidden_features)
        self.node_norm = nn.LayerNorm(hidden_features)
        self.edge_norm = nn.LayerNorm(hidden_features)

        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        e = self.edge_encoder(e)

        x_t, e_t = self.transformer_model.get_features(graph, x, e, pe)
        x_m, e_m = self.mamba_model.get_features(graph, x, e, pe)

        x_fused = self.node_norm(self.node_fusion(x_t, x_m) + x)
        e_fused = self.edge_norm(self.edge_fusion(e_t, e_m) + e)

        scores = self.predictor(graph, x_fused, e_fused)
        return scores


class TransformerOnlyFusionNet(nn.Module):
    """
    Single branch:
    Transformer only
    """
    def __init__(
        self,
        hidden_features,
        hidden_edge_features,
        num_transformer_layers,
        hidden_edge_scores,
        nb_pos_enc,
    ):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_edge_features),
            nn.GELU(),
            nn.Linear(hidden_edge_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )

        self.transformer_model = MyTransformerModelNet(
            hidden_features=hidden_features,
            num_layers=num_transformer_layers,
            nb_pos_enc=nb_pos_enc,
        )

        self.node_norm = nn.LayerNorm(hidden_features)
        self.edge_norm = nn.LayerNorm(hidden_features)

        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        e = self.edge_encoder(e)

        x_t, e_t = self.transformer_model.get_features(graph, x, e, pe)

        x_out = self.node_norm(x_t + x)
        e_out = self.edge_norm(e_t + e)

        scores = self.predictor(graph, x_out, e_out)
        return scores


class MambaOnlyFusionNet(nn.Module):
    """
    Single branch:
    Mamba only
    """
    def __init__(
        self,
        hidden_features,
        hidden_edge_features,
        num_mamba_layers,
        hidden_edge_scores,
        nb_pos_enc,
    ):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_edge_features),
            nn.GELU(),
            nn.Linear(hidden_edge_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )

        self.mamba_model = MyMambaModelNet(
            hidden_features=hidden_features,
            num_layers=num_mamba_layers,
            nb_pos_enc=nb_pos_enc,
        )

        self.node_norm = nn.LayerNorm(hidden_features)
        self.edge_norm = nn.LayerNorm(hidden_features)

        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        e = self.edge_encoder(e)

        x_m, e_m = self.mamba_model.get_features(graph, x, e, pe)

        x_out = self.node_norm(x_m + x)
        e_out = self.edge_norm(e_m + e)

        scores = self.predictor(graph, x_out, e_out)
        return scores


class DualBranchModel(nn.Module):
    """
    Ablation #2:
    Transformer + Mamba only
    No GatedGCN branch
    """
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_edge_scores,
        hidden_edge_features,
        num_transformer_layers,
        num_mamba_layers,
        nb_pos_enc,
        dropout=None,
    ):
        super().__init__()

        self.raw_node_proj = nn.Linear(node_features, hidden_features)
        self.raw_edge_proj = nn.Linear(edge_features, hidden_features)

        self.fusion_model = FusionModelNet(
            hidden_features=hidden_features,
            hidden_edge_features=hidden_edge_features,
            num_transformer_layers=num_transformer_layers,
            num_mamba_layers=num_mamba_layers,
            hidden_edge_scores=hidden_edge_scores,
            nb_pos_enc=nb_pos_enc,
        )

    def forward(self, graph, x, e, pe=None):
        if pe is None:
            raise ValueError("DualBranchModel requires positional encoding `pe`.")

        x = F.gelu(self.raw_node_proj(x))
        e = F.gelu(self.raw_edge_proj(e))
        scores = self.fusion_model(graph, x, e, pe)
        return scores


class GatedGCN_TransformerOnlyModel(nn.Module):
    """
    Ablation #4:
    GatedGCN + Transformer
    (Full - Mamba)
    """
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_gcn_layers,
        num_transformer_layers,
        hidden_edge_scores,
        normalization,
        hidden_edge_features,
        nb_pos_enc,
        dropout=None,
        directed=True,
    ):
        super().__init__()

        self.gated_gcn = GatedGCNModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_layers=num_gcn_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            dropout=dropout,
            directed=directed,
        )

        self.transformer_only_model = TransformerOnlyFusionNet(
            hidden_features=hidden_features,
            hidden_edge_features=hidden_edge_features,
            num_transformer_layers=num_transformer_layers,
            hidden_edge_scores=hidden_edge_scores,
            nb_pos_enc=nb_pos_enc,
        )

        self.raw_node_proj = nn.Linear(node_features, hidden_features)
        self.raw_edge_proj = nn.Linear(edge_features, hidden_features)

        self.node_proj = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
        )

    def forward(self, graph, x, e, pe=None):
        if pe is None:
            raise ValueError("GatedGCN_TransformerOnlyModel requires positional encoding `pe`.")

        x_raw = F.gelu(self.raw_node_proj(x))
        e_raw = F.gelu(self.raw_edge_proj(e))

        x_gcn, e_gcn = self.gated_gcn.get_features(graph, x, e)

        x_input = torch.cat([x_raw, x_gcn], dim=-1)
        e_input = torch.cat([e_raw, e_gcn], dim=-1)

        x_input = self.node_proj(x_input)
        e_input = self.edge_proj(e_input)

        scores = self.transformer_only_model(graph, x_input, e_input, pe)
        return scores


class GatedGCN_MambaOnlyModel(nn.Module):
    """
    Ablation #3:
    GatedGCN + Mamba
    (Full - Transformer)
    """
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_gcn_layers,
        num_mamba_layers,
        hidden_edge_scores,
        normalization,
        hidden_edge_features,
        nb_pos_enc,
        dropout=None,
        directed=True,
    ):
        super().__init__()

        self.gated_gcn = GatedGCNModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_layers=num_gcn_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            dropout=dropout,
            directed=directed,
        )

        self.mamba_only_model = MambaOnlyFusionNet(
            hidden_features=hidden_features,
            hidden_edge_features=hidden_edge_features,
            num_mamba_layers=num_mamba_layers,
            hidden_edge_scores=hidden_edge_scores,
            nb_pos_enc=nb_pos_enc,
        )

        self.raw_node_proj = nn.Linear(node_features, hidden_features)
        self.raw_edge_proj = nn.Linear(edge_features, hidden_features)

        self.node_proj = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
        )

    def forward(self, graph, x, e, pe=None):
        if pe is None:
            raise ValueError("GatedGCN_MambaOnlyModel requires positional encoding `pe`.")

        x_raw = F.gelu(self.raw_node_proj(x))
        e_raw = F.gelu(self.raw_edge_proj(e))

        x_gcn, e_gcn = self.gated_gcn.get_features(graph, x, e)

        x_input = torch.cat([x_raw, x_gcn], dim=-1)
        e_input = torch.cat([e_raw, e_gcn], dim=-1)

        x_input = self.node_proj(x_input)
        e_input = self.edge_proj(e_input)

        scores = self.mamba_only_model(graph, x_input, e_input, pe)
        return scores


class GatedGCN_FusionModel(nn.Module):
    """
    Full model:
    GatedGCN + Transformer + Mamba
    """
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        hidden_ne_features,
        num_gcn_layers,
        num_transformer_layers,
        num_mamba_layers,
        hidden_edge_scores,
        normalization,
        hidden_edge_features,
        batch_norm,
        nb_pos_enc,
        dropout=None,
        directed=True,
    ):
        super().__init__()

        self.gated_gcn = GatedGCNModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_layers=num_gcn_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            dropout=dropout,
            directed=directed,
        )

        self.fusion_model = FusionModelNet(
            hidden_features=hidden_features,
            hidden_edge_features=hidden_edge_features,
            num_transformer_layers=num_transformer_layers,
            num_mamba_layers=num_mamba_layers,
            hidden_edge_scores=hidden_edge_scores,
            nb_pos_enc=nb_pos_enc,
        )

        self.raw_node_proj = nn.Linear(node_features, hidden_features)
        self.raw_edge_proj = nn.Linear(edge_features, hidden_features)

        self.node_proj = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout if dropout is not None else 0.0),
        )

    def forward(self, graph, x, e, pe=None):
        if pe is None:
            raise ValueError("GatedGCN_FusionModel requires positional encoding `pe`.")

        x_raw = F.gelu(self.raw_node_proj(x))
        e_raw = F.gelu(self.raw_edge_proj(e))

        x_gcn, e_gcn = self.gated_gcn.get_features(graph, x, e)

        x_input = torch.cat([x_raw, x_gcn], dim=-1)
        e_input = torch.cat([e_raw, e_gcn], dim=-1)

        x_input = self.node_proj(x_input)
        e_input = self.edge_proj(e_input)

        scores = self.fusion_model(graph, x_input, e_input, pe)
        return scores
