import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import random
from datetime import datetime

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from graph_dataset import AssemblyGraphDataset
from configs.hyperparameters import get_hyperparameters
from configs.config import get_config
import Gtm_models as models
import Gtm_utils.utils as utils  # type: ignore
import Gtm_utils.metrics as metrics  # type: ignore


val_metis_cache = {}


# =========================
# Losses
# =========================
def sample_for_balanced_training(logits, labels, pos_keep_ratio=5.0):

    logits = logits.view(-1)
    labels = labels.float().view(-1)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(-1)
    neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(-1)

    num_pos = pos_indices.numel()
    num_neg = neg_indices.numel()

    if num_pos == 0 or num_neg == 0:
        return logits, labels

    num_pos_keep = min(int(num_neg * pos_keep_ratio), num_pos)
    perm = torch.randperm(num_pos, device=logits.device)
    pos_indices_selected = pos_indices[perm[:num_pos_keep]]

    keep_indices = torch.cat([neg_indices, pos_indices_selected], dim=0)

    return logits[keep_indices], labels[keep_indices]


def path_consistency_loss(graph, probs, labels=None):
   
    probs = probs.view(-1)

    lg = dgl.line_graph(graph, backtracking=False)
    lg_src, lg_dst = lg.edges(order="eid")

    if lg_src.numel() == 0:
        return probs.new_tensor(0.0)

    if labels is not None:
        labels = labels.view(-1)
        mask = (labels[lg_src] > 0.5) & (labels[lg_dst] > 0.5)
        if mask.sum() == 0:
            return probs.new_tensor(0.0)
        lg_src = lg_src[mask]
        lg_dst = lg_dst[mask]

    return torch.abs(probs[lg_src] - probs[lg_dst]).mean()


def confidence_loss(probs):
    probs = probs.view(-1)
    return torch.mean(probs * (1.0 - probs))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = (1 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_pos=3.0, gamma_neg=1.0, clip=0.0, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)

        xs_pos = probs
        xs_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_pos > 0:
            loss_pos = loss_pos * torch.pow(1.0 - xs_pos, self.gamma_pos)

        if self.gamma_neg > 0:
            loss_neg = loss_neg * torch.pow(1.0 - xs_neg, self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def symmetry_loss(org_scores, rev_scores, labels, edge_criterion, alpha=1.0):
    labels = labels.float()
    bce_org = edge_criterion(org_scores, labels)
    bce_rev = edge_criterion(rev_scores, labels)
    sym = torch.mean(torch.abs(torch.sigmoid(org_scores) - torch.sigmoid(rev_scores)))
    return bce_org + bce_rev + alpha * sym


def build_training_classification_loss(
    logits,
    labels,
    train_cls_criterion=None,
    use_asl=False,
    use_focal_loss=False,
    pos_keep_ratio=5.0,
    neg_aux_weight=1.0,
):
 
    logits = logits.view(-1)
    labels = labels.float().view(-1)

    sampled_logits, sampled_labels = sample_for_balanced_training(
        logits, labels, pos_keep_ratio=pos_keep_ratio
    )

    if use_asl or use_focal_loss:
        cls_loss = train_cls_criterion(sampled_logits, sampled_labels)
    else:
        cls_loss = F.binary_cross_entropy_with_logits(sampled_logits, sampled_labels)

    neg_aux_loss = logits.new_tensor(0.0)
    if neg_aux_weight > 0:
        neg_mask = (labels == 0)
        if neg_mask.sum() > 0:
            neg_logits = logits[neg_mask]
            neg_labels = labels[neg_mask]
            neg_aux_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
            cls_loss = cls_loss + neg_aux_weight * neg_aux_loss

    return cls_loss, neg_aux_loss



class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


def positional_encoding(num_nodes, d_model, device):
    if d_model % 2 != 0:
        raise ValueError("The embedding dimension (d_model) must be even.")

    pe = torch.zeros(num_nodes, d_model, device=device)
    position = torch.arange(0, num_nodes, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def compute_fp_fn_rates(TP, TN, FP, FN):
    fp_rate = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    fn_rate = FN / (FN + TP) if (FN + TP) != 0 else 0.0
    return fp_rate, fn_rate


def compute_metrics(logits, labels, loss, threshold=0.5):
    TP, TN, FP, FN = metrics.calculate_tfpn(logits, labels, threshold=threshold)
    acc, precision, recall, f1 = metrics.calculate_metrics(TP, TN, FP, FN)
    acc_inv, precision_inv, recall_inv, f1_inv = metrics.calculate_metrics_inverse(TP, TN, FP, FN)
    fp_rate, fn_rate = compute_fp_fn_rates(TP, TN, FP, FN)

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        pos_rate = preds.mean().item()

    return {
        "loss": loss,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc_inv": acc_inv,
        "precision_inv": precision_inv,
        "recall_inv": recall_inv,
        "f1_inv": f1_inv,
        "pos_rate": pos_rate,
    }


def average_epoch_metrics(metrics_dict):
    return {key: np.mean(values) for key, values in metrics_dict.items()}


def save_checkpoint(epoch, model, optimizer, loss_train, loss_valid, ckpt_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "loss_train": loss_train,
        "loss_valid": loss_valid,
    }
    torch.save(checkpoint, ckpt_path)


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


def get_or_build_val_parts(g_cpu, idx, num_clusters, extra_hops):
    global val_metis_cache
    key = (int(idx), int(g_cpu.num_nodes()), int(g_cpu.num_edges()), int(num_clusters), int(extra_hops))

    if key not in val_metis_cache:
        g_cpu_long = g_cpu.long()
        print(f"Metis Partitioning for Validation Graph {idx}...")
        d = dgl.metis_partition(g_cpu_long, num_clusters, extra_cached_hops=extra_hops)
        val_metis_cache[key] = list(d.values())

    return val_metis_cache[key]


def mask_graph_strandwise(g, fraction):
    g_cpu = g.to("cpu")

    keep_node_idx_half = torch.rand(g_cpu.num_nodes() // 2) < fraction
    keep_node_idx = torch.empty(
        keep_node_idx_half.size(0) * 2,
        dtype=keep_node_idx_half.dtype,
    )
    keep_node_idx[0::2] = keep_node_idx_half
    keep_node_idx[1::2] = keep_node_idx_half

    sub_g = dgl.node_subgraph(g_cpu, keep_node_idx, store_ids=True)

    print(f"Masking fraction: {fraction}")
    print(f"Original graph: N={g_cpu.num_nodes()}, E={g_cpu.num_edges()}")
    print(f"Subsampled graph: N={sub_g.num_nodes()}, E={sub_g.num_edges()}")

    return sub_g


# =========================
# Feature / batch helpers
# =========================
def get_full_ne_features(g):
    x = g.ndata["x"]
    e = g.edata["e"]
    return x, e


def get_partition_ne_features(sub_g, g):
    sub_g_ids = sub_g.ndata["_ID"].to(g.device)
    x = g.ndata["x"][sub_g_ids]

    e_ids = sub_g.edata["_ID"].to(g.device)
    e = g.edata["e"][e_ids]
    return x, e


def get_full_batch(g, model, device):
    x, e = get_full_ne_features(g)
    x = x.to(device)
    e = e.to(device)
    pe = g.ndata["pe"].to(device)
    logits = model(g, x, e, pe).squeeze(-1)
    labels = g.edata["y"].to(device).float()
    return logits, labels


def get_partition_batch(sub_g, g, model, device):
    sub_g = sub_g.to(device)
    x, e = get_partition_ne_features(sub_g, g)
    x = x.to(device)
    e = e.to(device)

    sub_node_ids = sub_g.ndata["_ID"].to(g.device)
    pe = g.ndata["pe"][sub_node_ids].to(device)

    logits = model(sub_g, x, e, pe).squeeze(-1)

    edge_ids = sub_g.edata["_ID"].to(g.edata["y"].device)
    labels = g.edata["y"][edge_ids].to(device).float()
    return sub_g, logits, labels


def get_bce_loss_full(g, model, edge_criterion, device):
    logits, labels = get_full_batch(g, model, device)
    loss = edge_criterion(logits, labels)
    return loss, logits, labels


def get_bce_loss_partition(sub_g, g, model, edge_criterion, device):
    sub_g, logits, labels = get_partition_batch(sub_g, g, model, device)
    loss = edge_criterion(logits, labels)
    return loss, logits, labels


def get_symmetry_loss_full(g, model, edge_criterion, alpha, device):
    logits_org, labels = get_full_batch(g, model, device)

    g_rev = dgl.reverse(g, copy_ndata=True, copy_edata=True)
    x_rev, e_rev = get_full_ne_features(g_rev)
    x_rev = x_rev.to(device)
    e_rev = e_rev.to(device)
    pe_rev = g_rev.ndata["pe"].to(device)
    logits_rev = model(g_rev, x_rev, e_rev, pe_rev).squeeze(-1)

    loss = symmetry_loss(logits_org, logits_rev, labels, edge_criterion, alpha=alpha)
    return loss, logits_org, labels


def get_symmetry_loss_partition(sub_g, g, model, edge_criterion, alpha, device):
    sub_g, logits_org, labels = get_partition_batch(sub_g, g, model, device)

    sub_g_rev = dgl.reverse(sub_g, copy_ndata=True, copy_edata=True)
    x_rev, e_rev = get_partition_ne_features(sub_g_rev, g)
    x_rev = x_rev.to(device)
    e_rev = e_rev.to(device)

    sub_node_ids_rev = sub_g_rev.ndata["_ID"].to(g.device)
    pe_rev = g.ndata["pe"][sub_node_ids_rev].to(device)
    logits_rev = model(sub_g_rev, x_rev, e_rev, pe_rev).squeeze(-1)

    loss = symmetry_loss(logits_org, logits_rev, labels, edge_criterion, alpha=alpha)
    return loss, logits_org, labels


# =========================
# Train
# =========================
def train(train_path, valid_path, out, assembler, overfit=False, dropout=None, seed=None, resume=False, gpu=None, exp_name=None):

    train_losses = []
    valid_losses = []
    train_f1_scores = []
    valid_f1_scores = []

    early_stopping = EarlyStopping(patience=5, delta=0.0)
    hyperparameters = get_hyperparameters(exp_name)

    if gpu is not None:
        if torch.cuda.is_available() and gpu < torch.cuda.device_count():
            device = f"cuda:{gpu}"
            print(f"Successfully using GPU {gpu} ({torch.cuda.get_device_name(gpu)})")
        else:
            print(f"GPU {gpu} is not available. Using CPU instead.")
            device = "cpu"
    else:
        device = "cpu"

    if seed is None:
        seed = hyperparameters["seed"]

    num_epochs = hyperparameters["num_epochs"]
    num_gnn_layers = hyperparameters["num_gnn_layers"]
    hidden_features = hyperparameters["dim_latent"]
    patience = hyperparameters["patience"]
    lr = hyperparameters["lr"]

    normalization = hyperparameters["normalization"]
    hidden_ne_features = hyperparameters["hidden_ne_features"]
    hidden_edge_scores = hyperparameters["hidden_edge_scores"]
    decay = hyperparameters["decay"]
    wandb_mode = hyperparameters["wandb_mode"]
    wandb_project = hyperparameters["wandb_project"]
    num_nodes_per_cluster = hyperparameters["num_nodes_per_cluster"]
    k_extra_hops = hyperparameters["k_extra_hops"]
    masking = hyperparameters["masking"]
    mask_frac_low = hyperparameters["mask_frac_low"]
    mask_frac_high = hyperparameters["mask_frac_high"]
    masking_valid = hyperparameters.get("masking_valid", False)

    use_symmetry_loss = hyperparameters["use_symmetry_loss"]
    alpha = hyperparameters["alpha"]

    pred_threshold = float(hyperparameters.get("pred_threshold", 0.70))
    use_dynamic_threshold = bool(hyperparameters.get("use_dynamic_threshold", False))

    use_focal_loss = bool(hyperparameters.get("use_focal_loss", False))
    focal_gamma = float(hyperparameters.get("focal_gamma", 2.0))

    use_asl = bool(hyperparameters.get("use_asl", True))
    asl_gamma_pos = float(hyperparameters.get("asl_gamma_pos", 3.0))
    asl_gamma_neg = float(hyperparameters.get("asl_gamma_neg", 1.0))
    asl_clip = float(hyperparameters.get("asl_clip", 0.0))

    lambda_path = float(hyperparameters.get("lambda_path", 0.05))
    lambda_conf = float(hyperparameters.get("lambda_conf", 0.02))
    conf_warmup_epochs = int(hyperparameters.get("conf_warmup_epochs", 8))

    pos_keep_ratio = float(hyperparameters.get("train_pos_keep_ratio", 5.0))
    neg_aux_weight = float(hyperparameters.get("neg_aux_weight", 1.0))

    num_transformer_layers = int(hyperparameters.get("num_transformer_layers", 4))
    num_mamba_layers = int(hyperparameters.get("num_mamba_layers", 2))

    utils.set_seed(seed)

    if not overfit:
        ds_train = AssemblyGraphDataset(train_path, assembler=assembler)
        ds_valid = AssemblyGraphDataset(valid_path, assembler=assembler)
    else:
        ds_train = ds_valid = AssemblyGraphDataset(train_path, assembler=assembler)

    config = get_config()
    checkpoints_path = os.path.abspath(config["checkpoints_path"])
    models_path = os.path.abspath(config["models_path"])

    node_features = 4
    edge_features = 4

    print("Calculating dataset statistics...")
    total_pos = 0
    total_neg = 0
    for _, g in ds_train:
        y_rounded = torch.round(g.edata["y"])
        total_pos += (y_rounded == 1).sum().item()
        total_neg += (y_rounded == 0).sum().item()

    pos_ratio = total_pos / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.0
    print(f"[INFO] total_pos = {total_pos}, total_neg = {total_neg}, pos_ratio = {pos_ratio:.6f}")

    time_start = datetime.now()
    timestamp = time_start.strftime("%Y-%b-%d-%H-%M-%S")

    if out is None:
        out = timestamp

    assert train_path is not None, "train_path not specified!"
    assert valid_path is not None, "valid_path not specified!"

    pe_dim = hyperparameters["nb_pos_enc"]

    for i in range(len(ds_train.graph_list)):
        idx, g = ds_train.graph_list[i]
        g = g.to(device)
        g.ndata["pe"] = positional_encoding(g.num_nodes(), pe_dim, device)
        ds_train.graph_list[i] = (idx, g)

    for i in range(len(ds_valid.graph_list)):
        idx, g = ds_valid.graph_list[i]
        g = g.to(device)
        g.ndata["pe"] = positional_encoding(g.num_nodes(), pe_dim, device)
        ds_valid.graph_list[i] = (idx, g)

    if dropout is None:
        dropout = hyperparameters["dropout"]

    edge_criterion = lambda logits, labels: F.binary_cross_entropy_with_logits(
        logits, labels.float()
    )

    if use_asl:
        train_cls_criterion = AsymmetricLoss(
            gamma_pos=asl_gamma_pos,
            gamma_neg=asl_gamma_neg,
            clip=asl_clip,
            reduction="mean",
        )
    elif use_focal_loss:
        train_cls_criterion = FocalLoss(gamma=focal_gamma)
    else:
        train_cls_criterion = None

    model_type = hyperparameters["model_type"]
    if model_type == "gatedgcn":
        model = models.GatedGCNModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_layers=num_gnn_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            dropout=dropout,
            directed=True,
        )

    elif model_type == "dualbranch":
        model = models.DualBranchModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_edge_scores=hidden_edge_scores,
            hidden_edge_features=hyperparameters["hidden_edge_features"],
            num_transformer_layers=num_transformer_layers,
            num_mamba_layers=num_mamba_layers,
            nb_pos_enc=hyperparameters["nb_pos_enc"],
            dropout=dropout,
        )

    elif model_type == "wo_transformer":
        model = models.GatedGCN_MambaOnlyModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_gcn_layers=num_gnn_layers,
            num_mamba_layers=num_mamba_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            hidden_edge_features=hyperparameters["hidden_edge_features"],
            nb_pos_enc=hyperparameters["nb_pos_enc"],
            dropout=dropout,
            directed=True,
        )

    elif model_type == "wo_mamba":
        model = models.GatedGCN_TransformerOnlyModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_gcn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            hidden_edge_features=hyperparameters["hidden_edge_features"],
            nb_pos_enc=hyperparameters["nb_pos_enc"],
            dropout=dropout,
            directed=True,
        )

    elif model_type == "full":
        model = models.GatedGCN_FusionModel(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            hidden_ne_features=hidden_ne_features,
            num_gcn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            num_mamba_layers=num_mamba_layers,
            hidden_edge_scores=hidden_edge_scores,
            normalization=normalization,
            hidden_edge_features=hyperparameters["hidden_edge_features"],
            batch_norm=(hyperparameters["normalization"] == "batch"),
            nb_pos_enc=hyperparameters["nb_pos_enc"],
            dropout=dropout,
            directed=True,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    if model_type in ["dualbranch", "wo_mamba", "full"] and hidden_features % 8 != 0:
        raise ValueError(
            f"hidden_features={hidden_features} must be divisible by the number of attention heads (num_heads=8)"
        )

    if not os.path.exists(models_path):
        os.makedirs(models_path)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    out = out + f"_seed{seed}"
    model_path = os.path.join(models_path, f"model_{out}.pt")
    ckpt_path = os.path.join(checkpoints_path, f"ckpt_{out}.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=decay, patience=patience, verbose=True)
    max_norm = 0.5

    start_epoch = 0
    loss_per_epoch_train, loss_per_epoch_valid = [], []

   
    best_valid_f1_inv = -1.0
    best_valid_fp_rate = float("inf")

    print("----- TRAIN CONFIGURAION SUMMARY -----")
    print(f"Using device: {device}")
    print(f"Using seed: {seed}")
    print(f"Experiment name: {exp_name}")
    print(f"Model type: {model_type}")
    print(f"Transformer layers: {num_transformer_layers}")
    print(f"Mamba layers: {num_mamba_layers}")
    print(f"Model path: {model_path}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Number of network parameters: {view_model_param(model)}")
    print(f"Normalization type: {normalization}")
    print("--------------------------------------\n")

    if resume:
        checkpoint = torch.load(ckpt_path)
        print("Loading checkpoint from:", ckpt_path, sep="\t")
        model_path = os.path.join(models_path, f"model_{out}_resumed-{num_epochs}.pt")
        ckpt_path = os.path.join(checkpoints_path, f"ckpt_{out}_resumed-{num_epochs}.pt")
        print("Saving resumed model to:", model_path, sep="\t")
        print("Saving new checkpoint to:", ckpt_path, sep="\t")

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch: {start_epoch}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

        loss_per_epoch_train.append(checkpoint["loss_train"])
        loss_per_epoch_valid.append(checkpoint["loss_valid"])

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f"Loading data done. Elapsed time: {elapsed}")

    try:
        with wandb.init(project=wandb_project, config=hyperparameters, mode=wandb_mode, name=out):
            for epoch in range(start_epoch, num_epochs):
                print("\n===> TRAINING")
                epoch_metrics_list_train = []
                random.shuffle(ds_train.graph_list)
                model.train()

                for idx, g in ds_train:
                    print(f"\n(TRAIN: Epoch = {epoch:3}) NEW GRAPH: index = {idx}")

                    if masking:
                        fraction = random.randint(mask_frac_low, mask_frac_high) / 100.0
                        g = mask_graph_strandwise(g, fraction)

                    num_clusters = g.num_nodes() // num_nodes_per_cluster + 1

                    if num_nodes_per_cluster >= g.num_nodes():
                        print("\nUse METIS: False")
                        print("Use full graph")
                        g = g.to(device)

                        logits, labels = get_full_batch(g, model, device)

                        if use_symmetry_loss:
                            loss, _, labels = get_symmetry_loss_full(g, model, edge_criterion, alpha, device)
                        else:
                            cls_loss, neg_aux_loss = build_training_classification_loss(
                                logits=logits,
                                labels=labels,
                                train_cls_criterion=train_cls_criterion,
                                use_asl=use_asl,
                                use_focal_loss=use_focal_loss,
                                pos_keep_ratio=pos_keep_ratio,
                                neg_aux_weight=neg_aux_weight,
                            )

                            probs = torch.sigmoid(logits)
                            p_loss = path_consistency_loss(g, probs, labels)
                            c_loss = confidence_loss(probs)

                            loss = cls_loss + lambda_path * p_loss
                            if epoch >= conf_warmup_epochs:
                                loss = loss + lambda_conf * c_loss

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                        optimizer.step()

                        epoch_metrics_list_train.append(
                            compute_metrics(logits, labels, loss.item(), threshold=pred_threshold)
                        )

                    else:
                        print("\nUse METIS: True")
                        print("Number of clusters:", num_clusters)

                        g_cpu = g.to("cpu")
                        d = dgl.metis_partition(g_cpu.long(), num_clusters, extra_cached_hops=k_extra_hops)
                        sub_gs = list(d.values())
                        sub_gs = [sg.to(device) for sg in sub_gs]
                        random.shuffle(sub_gs)

                        for sub_g in sub_gs:
                            sub_g, logits, labels = get_partition_batch(sub_g, g, model, device)

                            if use_symmetry_loss:
                                loss, _, labels = get_symmetry_loss_partition(sub_g, g, model, edge_criterion, alpha, device)
                            else:
                                cls_loss, neg_aux_loss = build_training_classification_loss(
                                    logits=logits,
                                    labels=labels,
                                    train_cls_criterion=train_cls_criterion,
                                    use_asl=use_asl,
                                    use_focal_loss=use_focal_loss,
                                    pos_keep_ratio=pos_keep_ratio,
                                    neg_aux_weight=neg_aux_weight,
                                )

                                probs = torch.sigmoid(logits)
                                p_loss = path_consistency_loss(sub_g, probs, labels)
                                c_loss = confidence_loss(probs)

                                loss = cls_loss + lambda_path * p_loss
                                if epoch >= conf_warmup_epochs:
                                    loss = loss + lambda_conf * c_loss

                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                            optimizer.step()

                            epoch_metrics_list_train.append(
                                compute_metrics(logits, labels, loss.item(), threshold=pred_threshold)
                            )

                aggregated_metrics = {
                    key: [m[key] for m in epoch_metrics_list_train]
                    for key in epoch_metrics_list_train[0]
                }
                epoch_mean_metrics_train = average_epoch_metrics(aggregated_metrics)

                torch.cuda.empty_cache()

                train_loss_epoch = epoch_mean_metrics_train["loss"]
                train_fp_rate_epoch = epoch_mean_metrics_train["fp_rate"]
                train_fn_rate_epoch = epoch_mean_metrics_train["fn_rate"]
                train_f1_epoch = epoch_mean_metrics_train["f1"]
                train_f1_inv_epoch = epoch_mean_metrics_train["f1_inv"]
                train_pos_rate_epoch = epoch_mean_metrics_train["pos_rate"]

                loss_per_epoch_train.append(train_loss_epoch)
                train_losses.append(train_loss_epoch)
                train_f1_scores.append(train_f1_epoch)

                lr_value = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch:3}] Train loss = {train_loss_epoch:.4f} | pos_rate = {train_pos_rate_epoch:.6f}")

                if overfit:
                    if len(loss_per_epoch_valid) == 1 or loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        torch.save(model.state_dict(), model_path)
                        print(
                            f"\nEpoch {epoch:3}: Model saved (overfitting)! -> Train Loss = {train_loss_epoch:.6f}"
                            f"\nTrain  F1 = {train_f1_epoch:.4f}\tTrain inv-F1 = {train_f1_inv_epoch:.4f}"
                            f"\nTrain FPR = {train_fp_rate_epoch:.4f}\tTrain FNR = {train_fn_rate_epoch:.4f}\n"
                        )

                    save_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        loss_train=min(loss_per_epoch_train),
                        loss_valid=0.0,
                        ckpt_path=ckpt_path,
                    )

                    log_data = {f"train/{k}": v for k, v in epoch_mean_metrics_train.items()}
                    log_data["lr_value"] = lr_value
                    log_data["pred_threshold"] = pred_threshold
                    log_data["use_focal_loss"] = use_focal_loss
                    log_data["focal_gamma"] = focal_gamma
                    log_data["use_asl"] = use_asl
                    log_data["asl_gamma_pos"] = asl_gamma_pos
                    log_data["asl_gamma_neg"] = asl_gamma_neg
                    log_data["asl_clip"] = asl_clip
                    log_data["train_pos_keep_ratio"] = pos_keep_ratio
                    log_data["neg_aux_weight"] = neg_aux_weight
                    wandb.log(log_data)
                    continue

                # =========================
                # Validation
                # =========================
                with torch.no_grad():
                    print("\n===> VALIDATION")
                    epoch_metrics_list_valid = []
                    all_val_logits, all_val_labels = [], []
                    model.eval()

                    for idx, g in ds_valid:
                        print(f"\n(VALID Epoch = {epoch:3}) NEW GRAPH: index = {idx}")

                        if masking_valid:
                            fraction = random.randint(mask_frac_low, mask_frac_high) / 100.0
                            g = mask_graph_strandwise(g, fraction)

                        num_clusters = g.num_nodes() // num_nodes_per_cluster + 1

                        if num_nodes_per_cluster >= g.num_nodes():
                            print("\nUse METIS: False")
                            print("Use full graph")
                            g = g.to(device)

                            if use_symmetry_loss:
                                loss, logits, labels = get_symmetry_loss_full(g, model, edge_criterion, alpha, device)
                            else:
                                loss, logits, labels = get_bce_loss_full(g, model, edge_criterion, device)

                            all_val_logits.append(logits.detach().float().view(-1).cpu())
                            all_val_labels.append(labels.detach().float().view(-1).cpu())

                            epoch_metrics_list_valid.append(
                                compute_metrics(logits, labels, loss.item(), threshold=pred_threshold)
                            )

                        else:
                            print("\nUse METIS: True")
                            print("Num clusters:", num_clusters)

                            g_cpu = g.to("cpu")
                            sub_gs = get_or_build_val_parts(g_cpu, idx, num_clusters, k_extra_hops)
                            sub_gs = [sg.to(device) for sg in sub_gs]

                            for sub_g in sub_gs:
                                if use_symmetry_loss:
                                    loss, logits, labels = get_symmetry_loss_partition(sub_g, g, model, edge_criterion, alpha, device)
                                else:
                                    loss, logits, labels = get_bce_loss_partition(sub_g, g, model, edge_criterion, device)

                                all_val_logits.append(logits.detach().float().view(-1).cpu())
                                all_val_labels.append(labels.detach().float().view(-1).cpu())

                                epoch_metrics_list_valid.append(
                                    compute_metrics(logits, labels, loss.item(), threshold=pred_threshold)
                                )

                    aggregated_metrics = {
                        key: [m[key] for m in epoch_metrics_list_valid]
                        for key in epoch_metrics_list_valid[0]
                    }
                    epoch_mean_metrics_valid = average_epoch_metrics(aggregated_metrics)

                    valid_loss_epoch = epoch_mean_metrics_valid["loss"]
                    valid_fp_rate_epoch = epoch_mean_metrics_valid["fp_rate"]
                    valid_fn_rate_epoch = epoch_mean_metrics_valid["fn_rate"]
                    valid_f1_epoch = epoch_mean_metrics_valid["f1"]
                    valid_f1_inv_epoch = epoch_mean_metrics_valid["f1_inv"]
                    valid_pos_rate_epoch = epoch_mean_metrics_valid["pos_rate"]

                    loss_per_epoch_valid.append(valid_loss_epoch)
                    valid_losses.append(valid_loss_epoch)
                    valid_f1_scores.append(valid_f1_epoch)

                    print(
                        f"[Epoch {epoch:3}] Valid loss = {valid_loss_epoch:.4f} | "
                        f"pos_rate = {valid_pos_rate_epoch:.6f} | "
                        f"f1_inv = {valid_f1_inv_epoch:.4f} | "
                        f"fp_rate = {valid_fp_rate_epoch:.4f}"
                    )

                    if use_dynamic_threshold and len(all_val_logits) > 0:
                        probs = torch.sigmoid(torch.cat(all_val_logits))
                        ytrue = torch.cat(all_val_labels)

                        cand = torch.linspace(0.50, 0.95, steps=10)
                        best_t, best_score = pred_threshold, -1.0

                        for t in cand:
                            pred = (probs >= t).float()
                            TP = (pred * ytrue).sum()
                            FP = (pred * (1 - ytrue)).sum()
                            TN = ((1 - pred) * (1 - ytrue)).sum()
                            FN = ((1 - pred) * ytrue).sum()

                            _, _, _, f1_inv_tmp = metrics.calculate_metrics_inverse(
                                TP.item(), TN.item(), FP.item(), FN.item()
                            )

                            score = f1_inv_tmp
                            if score > best_score:
                                best_score = score
                                best_t = float(t.item())

                        print(f"[Epoch {epoch:3d}] threshold_scan best_t={best_t:.2f} (inv-F1={best_score:.4f})")
                        pred_threshold = best_t
                        try:
                            wandb.log({
                                "valid/threshold_best_t": best_t,
                                "valid/threshold_best_inv_f1": best_score,
                            })
                        except Exception:
                            pass

                    scheduler.step(valid_loss_epoch)

                    early_stopping(valid_loss_epoch)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break

                    should_save = False
                    if valid_f1_inv_epoch > best_valid_f1_inv:
                        should_save = True
                    elif abs(valid_f1_inv_epoch - best_valid_f1_inv) < 1e-6 and valid_fp_rate_epoch < best_valid_fp_rate:
                        should_save = True

                    if should_save:
                        best_valid_f1_inv = valid_f1_inv_epoch
                        best_valid_fp_rate = valid_fp_rate_epoch
                        torch.save(model.state_dict(), model_path)
                        print(
                            f"\nEpoch {epoch:3}: Model saved! -> Val Loss = {valid_loss_epoch:.6f}"
                            f"\nVal F1 = {valid_f1_epoch:.4f}\tVal inv-F1 = {valid_f1_inv_epoch:.4f}"
                            f"\nVal FPR = {valid_fp_rate_epoch:.4f}\tVal FNR = {valid_fn_rate_epoch:.4f}\n"
                        )

                    save_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        loss_train=min(loss_per_epoch_train),
                        loss_valid=min(loss_per_epoch_valid),
                        ckpt_path=ckpt_path,
                    )

                try:
                    log_data = {f"train/{k}": v for k, v in epoch_mean_metrics_train.items()}
                    log_data.update({f"valid/{k}": v for k, v in epoch_mean_metrics_valid.items()})
                    log_data["lr_value"] = lr_value
                    log_data["pred_threshold"] = pred_threshold
                    log_data["use_focal_loss"] = use_focal_loss
                    log_data["focal_gamma"] = focal_gamma
                    log_data["use_asl"] = use_asl
                    log_data["asl_gamma_pos"] = asl_gamma_pos
                    log_data["asl_gamma_neg"] = asl_gamma_neg
                    log_data["asl_clip"] = asl_clip
                    log_data["lambda_path"] = lambda_path
                    log_data["lambda_conf"] = lambda_conf
                    log_data["train_pos_keep_ratio"] = pos_keep_ratio
                    log_data["neg_aux_weight"] = neg_aux_weight
                    log_data["num_transformer_layers"] = num_transformer_layers
                    log_data["num_mamba_layers"] = num_mamba_layers
                    wandb.log(log_data)

                except Exception as e:
                    print("WandB exception occurred!")
                    print(e)

    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        print("Keyboard Interrupt...")
        print("Exiting...")

    finally:
        torch.cuda.empty_cache()

    print("\n=== Training and Validation Metrics ===")
    print("Epoch | Train Loss | Valid Loss | Train F1 | Valid F1")
    for epoch in range(min(len(train_losses), len(valid_losses))):
        print(
            f"{epoch+1:5} | "
            f"{train_losses[epoch]:.4f} | "
            f"{valid_losses[epoch]:.4f} | "
            f"{train_f1_scores[epoch]:.4f} | "
            f"{valid_f1_scores[epoch]:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Path to the dataset")
    parser.add_argument("--valid", type=str, help="Path to the dataset")
    parser.add_argument("--asm", type=str, help="Assembler used")
    parser.add_argument("--name", type=str, default=None, help="Name for the model")
    parser.add_argument("--overfit", action="store_true", help="Overfit on the training data")
    parser.add_argument("--resume", action="store_true", help="Resume in case training failed")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate for the model")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for ablation/hyperparameter study")
    parser.add_argument("--gpu", type=int, default=None, help="Index of a GPU to train on (unspecified = cpu)")
    args = parser.parse_args()

    train(
        train_path=args.train,
        valid_path=args.valid,
        assembler=args.asm,
        out=args.name,
        overfit=args.overfit,
        dropout=args.dropout,
        seed=args.seed,
        resume=args.resume,
        gpu=args.gpu,
        exp_name=args.exp_name,
    )
