import os

import torch
import numpy as np
import dgl

from Bio import Seq, SeqIO
from scipy import sparse as sp

from configs.hyperparameters import get_hyperparameters


def extract_hifiasm_contigs(path, idx):
    gfa_path = os.path.join(path, f'{idx}_asm.bp.p_ctg.gfa')
    asm_path = os.path.join(path, f'{idx}_assembly.fasta')

    contigs = []

    with open(gfa_path) as f:
        n = 0
        for line in f.readlines():
            line = line.strip()

            if not line:
                continue

            if line[0] != 'S':
                continue

            seq = Seq.Seq(line.split()[2])
            ctg = SeqIO.SeqRecord(
                seq,
                description=f'contig_{n}',
                id=f'contig_{n}'
            )
            contigs.append(ctg)
            n += 1

        SeqIO.write(contigs, asm_path, 'fasta')


def _zscore(x):
   
    x = x.float()

    if x.numel() <= 1:
        return torch.zeros_like(x)

    mean = x.mean()
    std = x.std(unbiased=False)

    if std < 1e-8:
        return torch.zeros_like(x)

    return (x - mean) / (std + 1e-8)


def preprocess_graph(g):
 
    g = g.int()


    required_node_fields = [
        'read_length',
        'gc_content',
    ]

    required_edge_fields = [
        'overlap_similarity',
        'overlap_length',
        'edit_distance',
        'prefix_length_ratio',
    ]

    for key in required_node_fields:
        if key not in g.ndata:
            raise KeyError(
                f"Missing node field g.ndata['{key}']. "
                f"Available node keys: {list(g.ndata.keys())}"
            )

    for key in required_edge_fields:
        if key not in g.edata:
            raise KeyError(
                f"Missing edge field g.edata['{key}']. "
                f"Available edge keys: {list(g.edata.keys())}. "
                f"请确认生成图时使用了 get_similarities=True。"
            )

    # ----------------------
    # Node features: 4 dims
    # ----------------------
    read_length = g.ndata['read_length'].float()
    gc_content = g.ndata['gc_content'].float()

    in_deg_raw = g.in_degrees().float()
    out_deg_raw = g.out_degrees().float()

    read_length = _zscore(read_length)
    in_deg = _zscore(in_deg_raw)
    out_deg = _zscore(out_deg_raw)

 
    g.ndata['in_deg'] = in_deg_raw
    g.ndata['out_deg'] = out_deg_raw

    g.ndata['x'] = torch.stack(
        [
            read_length,
            gc_content,
            in_deg,
            out_deg,
        ],
        dim=1
    ) 

    # ----------------------
    # Edge features: 4 dims
    # ----------------------
    ol_sim = g.edata['overlap_similarity'].float()
    ol_len = g.edata['overlap_length'].float()
    edit_dist = g.edata['edit_distance'].float()
    prefix_ratio = g.edata['prefix_length_ratio'].float()

    ol_len = _zscore(ol_len)
    edit_dist = _zscore(edit_dist)
    prefix_ratio = _zscore(prefix_ratio)

    g.edata['e'] = torch.stack(
        [
            ol_sim,
            ol_len,
            edit_dist,
            prefix_ratio,
        ],
        dim=1
    )  # [num_edges, 4]

    return g


def add_positional_encoding(g):


    g.ndata['in_deg'] = g.in_degrees().float()
    g.ndata['out_deg'] = g.out_degrees().float()

    pe_dim = get_hyperparameters()['nb_pos_enc']
    pe_type = get_hyperparameters()['type_pos_enc']

    if pe_dim == 0:
        return g

    if pe_type == 'RW':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")

        Dinv = sp.diags(
            dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0,
            dtype=float
        )

        RW = A @ Dinv
        M = RW

        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M

        for _ in range(pe_dim - 1):
            M_power = M_power @ M
            PE.append(torch.from_numpy(M_power.diagonal()).float())

        PE = torch.stack(PE, dim=-1)
        g.ndata['pe'] = PE

    if pe_type == 'PR':
        # k-step PageRank features
        A = g.adjacency_matrix(scipy_fmt="csr")

        D = A.sum(axis=1)  # out degree
        Dinv = 1.0 / (D + 1e-9)
        Dinv[D < 1e-9] = 0

        Dinv = sp.diags(np.squeeze(np.asarray(Dinv)), dtype=float)
        P = (Dinv @ A).T

        n = A.shape[0]
        One = np.ones([n])
        x = One / n

        PE = []
        alpha = 0.95

        for _ in range(pe_dim):
            x = alpha * P.dot(x) + (1.0 - alpha) / n * One
            PE.append(torch.from_numpy(x).float())

        PE = torch.stack(PE, dim=-1)
        g.ndata['pe'] = PE

    return g
