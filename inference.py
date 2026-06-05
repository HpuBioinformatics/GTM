import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import math
from datetime import datetime
import gc

import torch
import dgl
import numpy as np

from graph_dataset import AssemblyGraphDataset
from configs.hyperparameters import get_hyperparameters
import Gtm_models as models
import Gtm_utils.evaluate as evaluate
import Gtm_utils.utils as utils



DEBUG = False
RANDOM = False
p_threshold = 0.06
early_stopping = False


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


def positional_encoding_from_node_ids(node_ids, d_model, device):
  
    if d_model % 2 != 0:
        raise ValueError("The embedding dimension (d_model) must be even.")

    node_ids = node_ids.to(device=device, dtype=torch.float32).unsqueeze(1)

    pe = torch.zeros(node_ids.shape[0], d_model, device=device)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(node_ids * div_term)
    pe[:, 1::2] = torch.cos(node_ids * div_term)
    return pe


def is_cuda_oom(err):
   
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True

    msg = str(err).lower()
    return (
        "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or ("out of memory" in msg and "cuda" in msg)
    )


def clear_cuda_cache(device):
   
    gc.collect()

    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def edge_logits_to_log_probs(edge_predictions, pred_threshold):
  
    edge_predictions = edge_predictions.reshape(-1)

    probs = torch.sigmoid(edge_predictions)

    probs = torch.where(
        probs >= pred_threshold,
        probs,
        torch.full_like(probs, 1e-9)
    )

    probs = probs.clamp(min=1e-9, max=1 - 1e-9)

    return torch.log(probs)


def infer_logprobs_full_graph(model, g_cpu, device, pe_dim, pred_threshold):
 
    g_dev = g_cpu.to(device)

    x = g_dev.ndata["x"]
    e = g_dev.edata["e"]

    pe = positional_encoding(g_dev.num_nodes(), pe_dim, device)

    edge_predictions = model(g_dev, x, e, pe).squeeze()
    log_probs = edge_logits_to_log_probs(edge_predictions, pred_threshold)

    log_probs = log_probs.detach().cpu()

    del g_dev, x, e, pe, edge_predictions
    clear_cuda_cache(device)

    return log_probs


def expand_nodes_by_hops(g, seed_nodes, num_hops):
   
    seed_nodes = seed_nodes.to(device=torch.device("cpu"), dtype=g.idtype).unique()

    if seed_nodes.numel() == 0:
        return seed_nodes

    node_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    node_mask[seed_nodes.long()] = True

    frontier = seed_nodes

    for _ in range(num_hops):
        if frontier.numel() == 0:
            break

        in_src, in_dst = g.in_edges(frontier)
        out_src, out_dst = g.out_edges(frontier)

        new_nodes = torch.cat([in_src, in_dst, out_src, out_dst]).to(dtype=g.idtype).unique()
        new_nodes = new_nodes[~node_mask[new_nodes.long()]]

        if new_nodes.numel() == 0:
            break

        node_mask[new_nodes.long()] = True
        frontier = new_nodes

    return torch.nonzero(node_mask, as_tuple=False).squeeze(1).to(dtype=g.idtype)


def infer_logprobs_metis_once(
    model,
    g_cpu,
    device,
    pe_dim,
    pred_threshold,
    num_parts,
    halo_hops,
    balance_edges=True,
):
   
    g_cpu = g_cpu.to("cpu")

    if g_cpu.num_edges() == 0:
        return torch.empty(0, dtype=torch.float32)

    num_parts = max(1, min(int(num_parts), g_cpu.num_nodes()))
    halo_hops = max(1, int(halo_hops))

    print(
        f"[WARN] Falling back to METIS subgraph inference: "
        f"num_parts={num_parts}, halo_hops={halo_hops}, balance_edges={balance_edges}"
    )

    
    if g_cpu.idtype != torch.int64:
        try:
            g_for_metis = g_cpu.long()
        except Exception as exc:
            raise RuntimeError(
                "DGL METIS partition usually requires int64 graph idtype. "
                "Please rebuild/load the graph with int64 IDs, or make sure DGLGraph.long() is available."
            ) from exc
    else:
        g_for_metis = g_cpu

    node_part = dgl.metis_partition_assignment(
        g_for_metis,
        num_parts,
        balance_edges=balance_edges,
    ).cpu().long()

    log_probs_all = torch.full(
        (g_cpu.num_edges(),),
        math.log(1e-9),
        dtype=torch.float32,
    )

    assigned = torch.zeros(g_cpu.num_edges(), dtype=torch.bool)

    for part_id in range(num_parts):
        core_nodes = torch.nonzero(node_part == part_id, as_tuple=False).squeeze(1)

        if core_nodes.numel() == 0:
            continue

        core_nodes = core_nodes.to(dtype=g_cpu.idtype)

        
        sub_nodes = expand_nodes_by_hops(g_cpu, core_nodes, halo_hops)

        sub_g_cpu = dgl.node_subgraph(
            g_cpu,
            sub_nodes,
            relabel_nodes=True,
            store_ids=True,
        )

        if sub_g_cpu.num_edges() == 0:
            continue

        raw_nids = sub_g_cpu.ndata[dgl.NID].cpu().long()
        raw_eids = sub_g_cpu.edata[dgl.EID].cpu().long()

        src_local, _ = sub_g_cpu.edges()
        src_raw = raw_nids[src_local.cpu().long()]

      
        write_mask = node_part[src_raw] == part_id

        if write_mask.sum().item() == 0:
            continue

        sub_g = sub_g_cpu.to(device)

        x_sub = sub_g.ndata["x"]
        e_sub = sub_g.edata["e"]

        pe_sub = positional_encoding_from_node_ids(raw_nids, pe_dim, device)

        edge_predictions = model(sub_g, x_sub, e_sub, pe_sub).squeeze()
        log_probs_sub = edge_logits_to_log_probs(edge_predictions, pred_threshold)

        write_eids = raw_eids[write_mask]
        log_probs_all[write_eids] = log_probs_sub.detach().cpu()[write_mask]
        assigned[write_eids] = True

        print(
            f"[METIS] part {part_id + 1}/{num_parts}: "
            f"sub_nodes={sub_g_cpu.num_nodes()}, "
            f"sub_edges={sub_g_cpu.num_edges()}, "
            f"write_edges={int(write_mask.sum().item())}"
        )

        del sub_g, x_sub, e_sub, pe_sub, edge_predictions, log_probs_sub
        clear_cuda_cache(device)

    if not assigned.all().item():
        num_missing = int((~assigned).sum().item())
        print(
            f"[WARN] {num_missing} edges were not assigned by METIS fallback. "
            f"They keep log(1e-9)."
        )

    return log_probs_all


def infer_logprobs_metis_oom_safe(
    model,
    g_cpu,
    device,
    pe_dim,
    pred_threshold,
    num_parts,
    max_parts,
    halo_hops,
    balance_edges=True,
):
   
    cur_parts = max(1, int(num_parts))
    max_parts = max(cur_parts, int(max_parts))

    while True:
        try:
            return infer_logprobs_metis_once(
                model=model,
                g_cpu=g_cpu,
                device=device,
                pe_dim=pe_dim,
                pred_threshold=pred_threshold,
                num_parts=cur_parts,
                halo_hops=halo_hops,
                balance_edges=balance_edges,
            )

        except RuntimeError as err:
            if not is_cuda_oom(err):
                raise

            clear_cuda_cache(device)

            if cur_parts >= max_parts:
                print(
                    f"[ERROR] METIS fallback still OOM at num_parts={cur_parts}. "
                    f"Try increasing metis_max_parts or reducing metis_halo_hops."
                )
                raise

            next_parts = min(cur_parts * 2, max_parts)
            print(
                f"[WARN] METIS subgraph inference OOM at num_parts={cur_parts}. "
                f"Retrying with num_parts={next_parts}..."
            )
            cur_parts = next_parts


def infer_logprobs_with_oom_fallback(
    model,
    g_cpu,
    device,
    pe_dim,
    pred_threshold,
    metis_num_parts=8,
    metis_max_parts=64,
    metis_halo_hops=1,
    metis_balance_edges=True,
):
   
    try:
        print("Computing the scores with the full graph model...\n")

        return infer_logprobs_full_graph(
            model=model,
            g_cpu=g_cpu,
            device=device,
            pe_dim=pe_dim,
            pred_threshold=pred_threshold,
        )

    except RuntimeError as err:
        if not is_cuda_oom(err):
            raise

        print("[WARN] CUDA OOM during full-graph inference. Switching to METIS fallback.")
        clear_cuda_cache(device)

        return infer_logprobs_metis_oom_safe(
            model=model,
            g_cpu=g_cpu,
            device=device,
            pe_dim=pe_dim,
            pred_threshold=pred_threshold,
            num_parts=metis_num_parts,
            max_parts=metis_max_parts,
            halo_hops=metis_halo_hops,
            balance_edges=metis_balance_edges,
        )


def get_contig_length(walk, graph):
    if len(walk) == 0:
        return 0.0
    if len(walk) == 1:
        return float(graph.ndata["read_length"][walk[0]].item())

    id_dtype = graph.idtype
    dev = graph.device

    idx_src = torch.tensor(walk[:-1], dtype=id_dtype, device=dev)
    idx_dst = torch.tensor(walk[1:], dtype=id_dtype, device=dev)

    eids = graph.edge_ids(idx_src, idx_dst)
    prefix = graph.edata["prefix_length"][eids]

    total_length = float(prefix.sum().item())
    total_length += float(graph.ndata["read_length"][walk[-1]].item())
    return total_length


def get_subgraph(g, visited, device):
    """Remove the visited nodes from the graph."""
    if len(visited) == 0:
        sub_g = g
        sub_g.ndata["idx_nodes"] = torch.arange(
            sub_g.num_nodes(), dtype=torch.int32, device=device
        )
        map_subg_to_g = torch.arange(
            sub_g.num_nodes(), dtype=torch.int32, device=device
        )
        return sub_g, map_subg_to_g

    remove_node_idx = torch.tensor(list(visited), dtype=torch.int32)
    list_node_idx = torch.arange(g.num_nodes(), dtype=torch.int32)

    keep_node_mask = torch.ones(g.num_nodes(), dtype=torch.bool)
    keep_node_mask[remove_node_idx.long()] = False

    keep_node_idx = list_node_idx[keep_node_mask].to(device=device, dtype=torch.int32)

    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    sub_g.ndata["idx_nodes"] = torch.arange(
        sub_g.num_nodes(), dtype=torch.int32, device=device
    )
    map_subg_to_g = sub_g.ndata[dgl.NID].to(torch.int32)
    return sub_g, map_subg_to_g


def sample_edges(prob_edges, nb_paths):
    """Sample edges with categorical sampling."""
    if prob_edges.shape[0] > 2**24:
        prob_edges = prob_edges[:2**24]

    if prob_edges.numel() == 0:
        return torch.empty(0, dtype=torch.int32)

    if RANDOM:
        return torch.randint(0, prob_edges.shape[0], (nb_paths,), dtype=torch.int32)

    prob_edges = prob_edges.masked_fill(prob_edges < 1e-9, 1e-9)
    prob_sum = prob_edges.sum()

    if prob_sum.item() <= 0:
        prob_edges = torch.ones_like(prob_edges) / prob_edges.numel()
    else:
        prob_edges = prob_edges / prob_sum

    dist = torch.distributions.Categorical(prob_edges)
    idx_edges = dist.sample((nb_paths,)).to(torch.int32)
    return idx_edges


def greedy_forwards(start, logProbs, neighbors, predecessors, edges, visited_old):
    current = start
    walk = []
    visited = set()
    sumLogProb = 0.0

    while True:
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)

        neighs_current = neighbors[current]
        if len(neighs_current) == 0:
            break

        if len(neighs_current) == 1:
            neighbor = neighs_current[0]
            if neighbor in visited_old or neighbor in visited:
                break
            else:
                sumLogProb += float(logProbs[edges[current, neighbor]].item())
                current = neighbor
                continue

        masked_neighbors = [n for n in neighs_current if not (n in visited_old or n in visited)]
        if len(masked_neighbors) == 0:
            break

        neighbor_edges = [edges[current, n] for n in masked_neighbors]
        neighbor_p = logProbs[neighbor_edges]

        if early_stopping and (neighbor_p < math.log(p_threshold)).all().item():
            return walk, visited, sumLogProb

        if RANDOM:
            index = torch.randint(0, neighbor_p.shape[0], (1,)).item()
            logProb = float(neighbor_p[index].item())
        else:
            logProb, index = torch.topk(neighbor_p, k=1, dim=0)
            index = index.item()
            logProb = float(logProb.item())

        sumLogProb += logProb
        current = masked_neighbors[index]

    return walk, visited, sumLogProb


def greedy_backwards_rc(start, logProbs, predecessors, neighbors, edges, visited_old):
    current = start ^ 1
    walk = []
    visited = set()
    sumLogProb = 0.0

    while True:
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)

        neighs_current = neighbors[current]
        if len(neighs_current) == 0:
            break

        if len(neighs_current) == 1:
            neighbor = neighs_current[0]
            if neighbor in visited_old or neighbor in visited:
                break
            else:
                sumLogProb += float(logProbs[edges[current, neighbor]].item())
                current = neighbor
                continue

        masked_neighbors = [n for n in neighs_current if not (n in visited_old or n in visited)]
        if len(masked_neighbors) == 0:
            break

        neighbor_edges = [edges[current, n] for n in masked_neighbors]
        neighbor_p = logProbs[neighbor_edges]

        if early_stopping and (neighbor_p < math.log(p_threshold)).all().item():
            walk = list(reversed([w ^ 1 for w in walk]))
            return walk, visited, sumLogProb

        if RANDOM:
            index = torch.randint(0, neighbor_p.shape[0], (1,)).item()
            logProb = float(neighbor_p[index].item())
        else:
            logProb, index = torch.topk(neighbor_p, k=1, dim=0)
            index = index.item()
            logProb = float(logProb.item())

        sumLogProb += logProb
        current = masked_neighbors[index]

    walk = list(reversed([w ^ 1 for w in walk]))
    return walk, visited, sumLogProb


def run_greedy_both_ways(src, dst, logProbs, succs, preds, edges, visited):
    tmp_visited = visited | {src, src ^ 1, dst, dst ^ 1}
    walk_f, visited_f, sumLogProb_f = greedy_forwards(
        dst, logProbs, succs, preds, edges, tmp_visited
    )
    walk_b, visited_b, sumLogProb_b = greedy_backwards_rc(
        src, logProbs, preds, succs, edges, tmp_visited | visited_f
    )
    return walk_f, walk_b, visited_f, visited_b, sumLogProb_f, sumLogProb_b


def get_contigs_greedy(
    g,
    succs,
    preds,
    edges,
    len_threshold,
    nb_paths=50,
    use_labels=False,
    checkpoint_dir=None,
    load_checkpoint=False,
):
    g = g.to("cpu")
    all_contigs = []
    all_walks_len = []
    all_contigs_len = []
    visited = set()
    idx_contig = -1

    if use_labels:
        print("Decoding with labels...")
        probs = g.edata["y"].float().clamp(min=1e-9, max=1 - 1e-9)
        g.edata["score"] = torch.log(probs)

    logProbs = g.edata["score"].to("cpu")

    print("Starting to decode with greedy...")
    print(f"num_candidates: {nb_paths}\n")

    ckpt_file = os.path.join(checkpoint_dir, "checkpoint.pkl")
    if load_checkpoint and os.path.isfile(ckpt_file):
        print(f"Loading checkpoint from: {checkpoint_dir}\n")
        with open(ckpt_file, "rb") as f:
            checkpoint = pickle.load(f)
        all_contigs = checkpoint["walks"]
        visited = checkpoint["visited"]
        idx_contig = len(all_contigs) - 1
        all_walks_len = checkpoint["all_walks_len"]
        all_contigs_len = checkpoint["all_contigs_len"]

    while True:
        idx_contig += 1
        time_start_sample_edges = datetime.now()

        sub_g, map_subg_to_g = get_subgraph(g, visited, "cpu")
        if sub_g.num_edges() == 0:
            print("No edges left in the subgraph. Stopping...")
            break

        if use_labels:
            prob_edges = sub_g.edata["y"]
        else:
            prob_edges = torch.exp(sub_g.edata["score"]).squeeze()

        idx_edges = sample_edges(prob_edges, nb_paths)
        if idx_edges.numel() == 0:
            print("No sampled edges. Stopping...")
            break

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_sample_edges)
        print(f"Elapsed time (sample edges): {elapsed}")

        all_walks = []
        all_visited_iter = []
        all_contig_lens = []
        all_sumLogProbs = []
        all_meanLogProbs = []
        all_meanLogProbs_scaled = []

        print(
            f"\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, "
            f"nb_remaining_nodes: {g.num_nodes() - len(visited)}, nb_original_nodes: {g.num_nodes()}"
        )

        time_start_get_candidates = datetime.now()

        src_nodes_sub, dst_nodes_sub = sub_g.edges()

        for indx, idx in enumerate(idx_edges):
            idx = idx.item()
            src_init_edges = map_subg_to_g[src_nodes_sub[idx]].item()
            dst_init_edges = map_subg_to_g[dst_nodes_sub[idx]].item()

            walk_f, walk_b, visited_f, visited_b, sumLogProb_f, sumLogProb_b = run_greedy_both_ways(
                src_init_edges,
                dst_init_edges,
                logProbs,
                succs,
                preds,
                edges,
                visited,
            )

            walk_it = walk_b + walk_f
            visited_iter = visited_f | visited_b
            sumLogProb_it = float(sumLogProb_f + sumLogProb_b)
            len_walk_it = len(walk_it)
            len_contig_it = get_contig_length(walk_it, g)

            if src_init_edges == dst_init_edges:
                len_walk_it = 1

            if len_walk_it > 2:
                meanLogProb_it = sumLogProb_it / (len_walk_it - 2)
                meanLogProb_scaled_it = meanLogProb_it / math.sqrt(max(len_contig_it, 1.0))
            elif len_walk_it == 2:
                meanLogProb_it = 0.0
                meanLogProb_scaled_it = 0.0
            else:
                len_contig_it = 0.0
                sumLogProb_it = 0.0
                meanLogProb_it = 0.0
                meanLogProb_scaled_it = 0.0
                print("SELF-LOOP!")

            print(
                f"{indx:<3}: src={src_init_edges:<8} dst={dst_init_edges:<8} "
                f"len_walk={len_walk_it:<8} len_contig={len_contig_it:<12.1f} "
                f"sumLogProb={sumLogProb_it:<12.3f} meanLogProb={meanLogProb_it:<12.4f} "
                f"meanLogProb_scaled={meanLogProb_scaled_it:<12.4f}"
            )

            all_walks.append(walk_it)
            all_visited_iter.append(visited_iter)
            all_contig_lens.append(len_contig_it)
            all_sumLogProbs.append(sumLogProb_it)
            all_meanLogProbs.append(meanLogProb_it)
            all_meanLogProbs_scaled.append(meanLogProb_scaled_it)

        if len(all_contig_lens) == 0:
            print("No candidate walks generated. Stopping...")
            break

        best = max(all_contig_lens)
        idxx = all_contig_lens.index(best)

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_candidates)
        print(f"Elapsed time (get_candidates): {elapsed}")

        best_walk = all_walks[idxx]
        best_visited = all_visited_iter[idxx]

        time_start_get_visited = datetime.now()
        trans = set()
        for ss, dd in zip(best_walk[:-1], best_walk[1:]):
            t1 = set(succs[ss]) & set(preds[dd])
            t2 = {t ^ 1 for t in t1}
            trans = trans | t1 | t2
        best_visited = best_visited | trans

        best_contig_len = all_contig_lens[idxx]
        best_sumLogProb = all_sumLogProbs[idxx]
        best_meanLogProb = all_meanLogProbs[idxx]
        best_meanLogProb_scaled = all_meanLogProbs_scaled[idxx]

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_visited)
        print(f"Elapsed time (get visited): {elapsed}")

        print(
            f"\nChosen walk with index: {idxx}\n"
            f"len_walk={len(best_walk):<8} len_contig={best_contig_len:<12.1f} "
            f"sumLogProb={best_sumLogProb:<12.3f} meanLogProb={best_meanLogProb:<12.4f} "
            f"meanLogProb_scaled={best_meanLogProb_scaled:<12.4f}\n"
        )

        if best_contig_len < len_threshold:
            print(
                f"Best contig length {best_contig_len:.1f} < len_threshold {len_threshold}, stopping..."
            )
            break

        all_contigs.append(best_walk)
        visited |= best_visited
        all_walks_len.append(len(best_walk))
        all_contigs_len.append(best_contig_len)

        print(f"All walks len: {all_walks_len}")
        print(f"All contigs len: {all_contigs_len}\n")

        if len(all_contigs) % 10 == 0 and checkpoint_dir is not None:
            checkpoint = {
                "walks": all_contigs,
                "visited": visited,
                "all_walks_len": all_walks_len,
                "all_contigs_len": all_contigs_len,
            }
            if not DEBUG:
                try:
                    tmp_path = os.path.join(checkpoint_dir, "checkpoint_tmp.pkl")
                    final_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
                    with open(tmp_path, "wb") as f:
                        pickle.dump(checkpoint, f)
                    os.replace(tmp_path, final_path)
                except OSError:
                    print(f"Checkpoint was not saved. Last available checkpoint: {final_path}")
                    raise

    return all_contigs


def inference(data_path, model_path, assembler, savedir, device="cpu", dropout=None, exp_name=None):
    """Using a pretrained model, get walks and contigs on new data."""
    hyperparameters = get_hyperparameters(exp_name)

    seed = hyperparameters["seed"]
    num_gnn_layers = hyperparameters["num_gnn_layers"]
    hidden_features = hyperparameters["dim_latent"]
    normalization = hyperparameters["normalization"]
    hidden_ne_features = hyperparameters["hidden_ne_features"]
    hidden_edge_scores = hyperparameters["hidden_edge_scores"]

    strategy = hyperparameters["strategy"]
    nb_paths = hyperparameters["num_decoding_paths"]
    len_threshold = hyperparameters["len_threshold"]
    use_labels = hyperparameters["decode_with_labels"]
    load_checkpoint = hyperparameters["load_checkpoint"]

    pred_threshold = hyperparameters.get("pred_threshold", 0.5)

    num_transformer_layers = int(hyperparameters.get("num_transformer_layers", 4))
    num_mamba_layers = int(hyperparameters.get("num_mamba_layers", 2))

  
    metis_num_parts = int(hyperparameters.get("metis_num_parts", 8))
    metis_max_parts = int(hyperparameters.get("metis_max_parts", 64))
    metis_balance_edges = bool(hyperparameters.get("metis_balance_edges", True))

    metis_halo_hops = int(
        hyperparameters.get("metis_halo_hops", max(1, num_gnn_layers))
    )

    node_features = 4
    edge_features = 4

    if dropout is None:
        dropout = hyperparameters["dropout"]

    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    print(f"[INFO] Inference using device: {device}")

    utils.set_seed(seed)
    time_start = datetime.now()

    model_type = hyperparameters.get("model_type", "gatedgcn")

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

    print(f"[INFO] Using exp_name = {exp_name}")
    print(f"[INFO] Using model_type = {model_type}")
    print(f"[INFO] Transformer layers = {num_transformer_layers}")
    print(f"[INFO] Mamba layers = {num_mamba_layers}")
    print(f"[INFO] METIS num_parts = {metis_num_parts}")
    print(f"[INFO] METIS max_parts = {metis_max_parts}")
    print(f"[INFO] METIS halo_hops = {metis_halo_hops}")
    print(f"[INFO] METIS balance_edges = {metis_balance_edges}")
    print(f"[INFO] Loading model parameters from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ds = AssemblyGraphDataset(data_path, assembler)

    inference_dir = os.path.join(savedir, "decode")
    os.makedirs(inference_dir, exist_ok=True)

    checkpoint_root = os.path.join(savedir, "checkpoint")
    os.makedirs(checkpoint_root, exist_ok=True)

    assembly_dir = os.path.join(savedir, "assembly")
    os.makedirs(assembly_dir, exist_ok=True)

    walks_per_graph = []
    contigs_per_graph = []

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f"\nelapsed time (loading network and data): {elapsed}\n")

    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    threshold_tag = str(pred_threshold).replace(".", "p")

    for idx, g in ds:
        print(f"==== Processing graph {idx} ====")

        with torch.no_grad():
            time_start_get_scores = datetime.now()

     
            g = g.to("cpu")

            pe_dim = hyperparameters["nb_pos_enc"]

            if use_labels:
                print("Decoding with labels...")
                g.edata["score"] = (
                    g.edata["y"]
                    .float()
                    .clamp(min=1e-9, max=1 - 1e-9)
                    .log()
                    .cpu()
                )

            else:
                print("Decoding with model scores...")

                predicts_path = os.path.join(
                    inference_dir,
                    f"{idx}_{model_tag}_thr{threshold_tag}_logprobs.pt"
                )

                if os.path.isfile(predicts_path):
                    print(f"Loading cached log-probs from:\n{predicts_path}\n")
                    g.edata["score"] = torch.load(predicts_path, map_location="cpu")

                elif RANDOM:
                    probs = torch.ones_like(g.edata["prefix_length"], dtype=torch.float32)
                    probs = probs.clamp(min=1e-9, max=1 - 1e-9)
                    g.edata["score"] = torch.log(probs).cpu()

                else:
                    g.edata["score"] = infer_logprobs_with_oom_fallback(
                        model=model,
                        g_cpu=g,
                        device=device,
                        pe_dim=pe_dim,
                        pred_threshold=pred_threshold,
                        metis_num_parts=metis_num_parts,
                        metis_max_parts=metis_max_parts,
                        metis_halo_hops=metis_halo_hops,
                        metis_balance_edges=metis_balance_edges,
                    )

                    torch.save(g.edata["score"].cpu(), predicts_path)

            elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_scores)
            print(f"elapsed time (get_scores): {elapsed}")

        print("Loading successors...")
        with open(f"{data_path}/{assembler}/info/{idx}_succ.pkl", "rb") as f_succs:
            succs = pickle.load(f_succs)

        print("Loading predecessors...")
        with open(f"{data_path}/{assembler}/info/{idx}_pred.pkl", "rb") as f_preds:
            preds = pickle.load(f_preds)

        print("Loading edges...")
        with open(f"{data_path}/{assembler}/info/{idx}_edges.pkl", "rb") as f_edges:
            edges = pickle.load(f_edges)

        print("Done loading the auxiliary graph data!")

        time_start_get_walks = datetime.now()

        g.edata["prefix_length"] = g.edata["prefix_length"].masked_fill(
            g.edata["prefix_length"] < 0, 0
        )

        graph_checkpoint_dir = os.path.join(checkpoint_root, str(idx))
        os.makedirs(graph_checkpoint_dir, exist_ok=True)

        if strategy == "greedy":
            walks = get_contigs_greedy(
                g,
                succs,
                preds,
                edges,
                len_threshold,
                nb_paths,
                use_labels,
                graph_checkpoint_dir,
                load_checkpoint,
            )
        else:
            raise ValueError("Invalid decoding strategy")

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_walks)
        print(f"elapsed time (get_walks): {elapsed}")

        inference_path = os.path.join(inference_dir, f"{idx}_walks.pkl")
        with open(inference_path, "wb") as f:
            pickle.dump(walks, f)

        print("Loading reads...")
        with open(f"{data_path}/{assembler}/info/{idx}_reads.pkl", "rb") as f_reads:
            reads = pickle.load(f_reads)
        print("Done!")

        time_start_get_contigs = datetime.now()
        contigs = evaluate.walk_to_sequence(walks, g, reads, edges)
        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_contigs)
        print(f"elapsed time (get_contigs): {elapsed}")

        evaluate.save_assembly(contigs, assembly_dir, idx)
        walks_per_graph.append(walks)
        contigs_per_graph.append(contigs)

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f"elapsed time (total): {elapsed}")

    if DEBUG:
        exit(0)

    print(f"Found contigs for {data_path}!")
    print(f"Model used: {model_path}")
    print(f"Assembly saved in: {savedir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the dataset")
    parser.add_argument("--asm", type=str, help="Assembler used")
    parser.add_argument("--out", type=str, help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Path to the model")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for ablation/hyperparameter study")
    args = parser.parse_args()

    data = args.data
    asm = args.asm
    out = args.out
    model = args.model
    device = args.device

    if not model:
        model = "weights/weights.pt"

    inference(
        data_path=data,
        assembler=asm,
        model_path=model,
        savedir=out,
        device=device,
        exp_name=args.exp_name,
    )
