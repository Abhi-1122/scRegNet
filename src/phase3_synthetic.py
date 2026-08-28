"""
Phase 3 synthetic sanity check (DIFFERENTIAL_GRN_PLAN.md Sec 5.2, using the plan's own
documented fallback since GRouNdGAN itself is out of scope for this pass -- Sec 7).

A parametric negative-binomial simulator seeded from hESC/TFs+500's REAL reference network
edge structure and REAL per-gene mean/dispersion. Constructs one KNOWN-ground-truth rewired
variant (moderate severity, ~15%) and runs the full Sec 3 labeling pipeline + a freshly
trained `full` scDiffTransNet_GCN model against it, reporting:
  (a) the trained model's AUROC/AUPRC against the TRUE (by-construction) 4-way labels
  (b) the Sec 3 silver-labeling procedure's own accuracy against those same true labels
      (the calibration check the plan calls for).

Scope: one severity (moderate, ~15%), one cell type (hESC), per work.md's scoping entry.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

from src.models import scDiffTransNet_GCN
from src.utils import load_data, set_seed
from src.differential_labels import (
    compute_differential_labels, TRAINABLE_CLASSES, CLASS_TO_IDX, stratified_split,
)


def fit_nb_params(counts: np.ndarray):
    """Method-of-moments NB fit per gene. counts: (n_genes, n_cells) raw (non-negative)."""
    mean = counts.mean(axis=1)
    var = counts.var(axis=1)
    mean = np.clip(mean, 1e-3, None)
    r = np.where(var > mean, mean ** 2 / np.clip(var - mean, 1e-6, None), 1e6)
    r = np.clip(r, 1e-2, 1e6)
    return mean, r


def simulate_condition(mu, r, edge_matrix, gene_names, tf_idx_set, n_cells, beta, rng):
    """
    mu, r: (n_genes,) NB mean/dispersion per gene (shared across conditions -- realism from
        real data; only the co-expression STRUCTURE differs between conditions via
        edge_matrix).
    edge_matrix: (n_genes, n_genes) binary, edge_matrix[i, j] = 1 if TF i -> target j in
        THIS condition's network.
    tf_idx_set: set of gene-row indices that are TFs (only these get a "driver" role).
    beta: regulatory-strength constant controlling how strongly a target's mean shifts with
        its regulators' expression.
    Returns raw counts (n_genes, n_cells).
    """
    n_genes = len(mu)

    def nb_sample(mu_arr, r_arr, size):
        mu_arr = np.clip(mu_arr, 1e-3, None)
        p = r_arr / (r_arr + mu_arr)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return rng.negative_binomial(r_arr, p, size=size)

    # Step 1: draw all genes' baseline counts independently first (gives every gene,
    # including TFs, its own realistic marginal distribution).
    counts = nb_sample(np.tile(mu[:, None], (1, n_cells)), np.tile(r[:, None], (1, n_cells)),
                        size=(n_genes, n_cells))

    # Step 2: standardize TF rows to get a per-cell "regulatory activity" signal z.
    z = np.zeros((n_genes, n_cells))
    tf_rows = np.array(sorted(tf_idx_set))
    tf_counts = counts[tf_rows]
    tf_mean = tf_counts.mean(axis=1, keepdims=True)
    tf_std = tf_counts.std(axis=1, keepdims=True)
    tf_std = np.where(tf_std == 0, 1.0, tf_std)
    z[tf_rows] = (tf_counts - tf_mean) / tf_std

    # Step 3: for every gene with >=1 regulator in edge_matrix, redraw its counts with a
    # per-cell mean shifted by the mean z of its regulators (vectorized).
    reg_count = edge_matrix.sum(axis=0)  # (n_genes,) number of regulators per target
    has_reg = reg_count > 0
    mean_reg_z = np.zeros((n_genes, n_cells))
    if has_reg.any():
        summed = edge_matrix.T @ z  # (n_genes, n_cells): sum of regulator z per gene
        mean_reg_z[has_reg] = summed[has_reg] / reg_count[has_reg, None]
        mean_reg_z = np.clip(mean_reg_z, -3, 3)
        mu_shifted = mu[:, None] * np.exp(beta * mean_reg_z)
        r_full = np.tile(r[:, None], (1, n_cells))
        regulated_counts = nb_sample(mu_shifted, r_full, size=(n_genes, n_cells))
        counts[has_reg] = regulated_counts[has_reg]

    return counts


def build_rewired_network(net_df, rng, severity=0.15, tf_idx_set=None, gene_names=None,
                           n_regular_tfs=30, targets_per_regular=10, hub_out_degree=200):
    """
    Builds a SIMPLIFIED single-regulator-per-target synthetic network -- seeded from the
    real reference network only via real TF/gene identities (source TFs are drawn from
    the real TF.csv list, targets from the real BL--ExpressionData.csv gene panel), NOT
    reusing the real network's multi-regulator topology directly.

    This is a deliberate, documented simplification (see work.md): the real hESC
    reference network has mean in-degree ~5.6 (most targets have several independent
    regulators), which dilutes any single TF-target pair's *marginal* correlation well
    below the |r|>=0.3 detection floor at n~92-188 cells (verified empirically while
    building this script -- real-network-topology simulation gave near-zero separation
    between true "conserved" edges and true "neither" random pairs, and a real reference
    network dominated by one 79%-of-genes hub TF (TFAP2A) also makes "random" pairs
    accidentally share a common regulator, so they are not a clean negative control
    either). Restricting the injected structure to one regulator per target sidesteps
    both problems and gives a labeling+model sanity check that is actually informative,
    at the documented cost of not replicating the real network's combinatorial-regulation
    topology.

    TF1 (hub, `hub_out_degree` private targets) loses `severity` fraction of its
    out-edges; TF2 (a `n_regular_tfs`-sized regular TF) gains that many new out-edges to
    fresh, previously-unassigned targets. All other regular TFs' edges are conserved
    unchanged. Returns (g_a_edges, g_b_edges, removed_edges, added_edges, tf1, tf2,
    involved_genes) where involved_genes is every gene used as a regulator or target
    anywhere in G_A/G_B (needed to sample a clean, unconfounded "neither" set).
    """
    tf_list = [gene_names[i] for i in sorted(tf_idx_set)]
    rng.shuffle(tf_list)
    tf_set_names = set(tf_list)
    non_tf_genes = [g for g in gene_names if g not in tf_set_names]
    rng.shuffle(non_tf_genes)

    tf1, tf2 = tf_list[0], tf_list[1]
    regular_tfs = tf_list[2:2 + n_regular_tfs]

    assigned_targets = set()
    g_a_edges = set()

    pool = [g for g in non_tf_genes if g not in assigned_targets]
    tf1_targets = pool[:hub_out_degree]
    assigned_targets.update(tf1_targets)
    for t in tf1_targets:
        g_a_edges.add((tf1, t))

    for tf in regular_tfs:
        pool = [g for g in non_tf_genes if g not in assigned_targets]
        targets = pool[:targets_per_regular]
        assigned_targets.update(targets)
        for t in targets:
            g_a_edges.add((tf, t))

    tf1_edges = [e for e in g_a_edges if e[0] == tf1]
    n_remove = max(1, int(round(len(tf1_edges) * severity)))
    removed_idx = rng.choice(len(tf1_edges), size=n_remove, replace=False)
    removed_edges = set(tf1_edges[i] for i in removed_idx)

    pool = [g for g in non_tf_genes if g not in assigned_targets]
    n_add = n_remove
    tf2_new_targets = pool[:n_add]
    assigned_targets.update(tf2_new_targets)
    added_edges = set((tf2, t) for t in tf2_new_targets)

    g_b_edges = (g_a_edges - removed_edges) | added_edges
    involved_genes = set(tf_list[:2 + n_regular_tfs]) | assigned_targets
    return g_a_edges, g_b_edges, removed_edges, added_edges, tf1, tf2, involved_genes


def run(args):
    rng = np.random.RandomState(args.random_seed)
    path = os.path.join(args.data_folder, f'{args.cell_type}/TFs+{args.num_TF}/')
    data_input = pd.read_csv(os.path.join(path, 'BL--ExpressionData.csv'), index_col=0)
    net_df = pd.read_csv(os.path.join(path, 'BL--network.csv'))
    tf_df = pd.read_csv(os.path.join(path, 'TF.csv'), index_col=0)

    gene_names = data_input.index.to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    tf_idx_set = set(tf_df['index'].values.astype(int))

    g_a_edges, g_b_edges, removed_edges, added_edges, tf1, tf2, involved_genes = build_rewired_network(
        net_df, rng, severity=args.severity, tf_idx_set=tf_idx_set, gene_names=gene_names,
        n_regular_tfs=args.n_regular_tfs, targets_per_regular=args.targets_per_regular,
        hub_out_degree=args.hub_out_degree)
    print(f"TF1 (loses {args.severity:.0%} out-edges) = {tf1} ({len(removed_edges)} removed)")
    print(f"TF2 (gains {args.severity:.0%}-of-TF1 new out-edges) = {tf2} ({len(added_edges)} added)")

    def edges_to_matrix(edge_set):
        m = np.zeros((n_genes, n_genes), dtype=np.float32)
        for (s, t) in edge_set:
            if s in gene_to_idx and t in gene_to_idx:
                m[gene_to_idx[s], gene_to_idx[t]] = 1.0
        return m

    M_A = edges_to_matrix(g_a_edges)
    M_B = edges_to_matrix(g_b_edges)

    # Real per-gene NB params from the real hESC expression data (all cells pooled).
    # BL--ExpressionData.csv is BEELINE-preprocessed log2(count+1) data (verified directly:
    # values are continuous floats in [0, ~19], ~28% exact zeros), not raw counts -- so we
    # invert the log2(x+1) transform to recover count-like values before an NB fit, rather
    # than fitting NB moments directly to log-scale data (which collapses almost all
    # per-gene variance and yields a near-Poisson/degenerate dispersion fit that cannot
    # produce realistic scRNA-seq overdispersion).
    log_vals = data_input.values.astype(np.float64)
    raw = np.clip(np.round(np.power(2.0, log_vals) - 1.0), 0, None)
    mu, r = fit_nb_params(raw)

    n_a, n_b = args.n_cells_a, args.n_cells_b
    counts_a = simulate_condition(mu, r, M_A, gene_names, tf_idx_set, n_a, args.beta, rng)
    counts_b = simulate_condition(mu, r, M_B, gene_names, tf_idx_set, n_b, args.beta, rng)

    data_a = pd.DataFrame(counts_a, index=gene_names, columns=[f"A_{i}" for i in range(n_a)])
    data_b = pd.DataFrame(counts_b, index=gene_names, columns=[f"B_{i}" for i in range(n_b)])

    # Candidate edges + TRUE 4-way labels (known by construction).
    conserved_edges = g_a_edges & g_b_edges
    n_neither = args.n_neither
    # Clean, unconfounded "neither" pairs: BOTH genes must be outside `involved_genes`
    # (never used as a regulator or target anywhere in G_A/G_B) so they carry no injected
    # correlation signal by construction -- the negative control the real-network topology
    # could not provide (see build_rewired_network's docstring).
    uninvolved_tfs = [gene_names[i] for i in sorted(tf_idx_set) if gene_names[i] not in involved_genes]
    uninvolved_genes = [g for g in gene_names if g not in involved_genes]
    neither_candidates = []
    seen = set()
    attempts = 0
    while len(neither_candidates) < n_neither and attempts < n_neither * 50 and uninvolved_tfs and uninvolved_genes:
        attempts += 1
        s = uninvolved_tfs[rng.randint(len(uninvolved_tfs))]
        t = uninvolved_genes[rng.randint(len(uninvolved_genes))]
        if s == t or (s, t) in seen:
            continue
        seen.add((s, t))
        neither_candidates.append((s, t))

    rows = []
    for (s, t) in conserved_edges:
        rows.append((s, t, 'conserved'))
    for (s, t) in removed_edges:
        rows.append((s, t, 'A-specific'))
    for (s, t) in added_edges:
        rows.append((s, t, 'B-specific'))
    for (s, t) in neither_candidates:
        rows.append((s, t, 'neither'))
    true_df = pd.DataFrame(rows, columns=['TF', 'Target', 'true_label'])
    true_df['TF_idx'] = true_df['TF'].map(gene_to_idx)
    true_df['Target_idx'] = true_df['Target'].map(gene_to_idx)
    candidate_edges = true_df[['TF_idx', 'Target_idx']].values.astype(np.int64)

    print(f"Synthetic candidate space: {len(true_df)} edges "
          f"(conserved={len(conserved_edges)}, A-specific={len(removed_edges)}, "
          f"B-specific={len(added_edges)}, neither={len(neither_candidates)})")

    # ---- (b) Silver-labeling calibration check: run the REAL Sec 3 pipeline on the
    # synthetic data and compare its output to the TRUE labels above. ----
    silver_df = compute_differential_labels(
        data_a, data_b, candidate_edges,
        corr_method=args.corr_method, corr_min_r=args.corr_min_r, fdr_q=args.fdr_q,
    )
    merged = true_df.merge(silver_df[['TF_idx', 'Target_idx', 'label']], on=['TF_idx', 'Target_idx'])
    silver_acc_all = accuracy_score(merged['true_label'], merged['label'])
    # Accuracy restricted to the trainable label space only (silver's ambiguous/inconsistent
    # calls have no analogue in the true labels, so those rows can never "match" -- report
    # both the strict all-rows accuracy and the trainable-subset accuracy).
    trainable_rows = merged[merged['label'].isin(TRAINABLE_CLASSES)]
    silver_acc_trainable = accuracy_score(trainable_rows['true_label'], trainable_rows['label']) if len(trainable_rows) else None
    silver_cm = confusion_matrix(merged['true_label'], merged['label'],
                                  labels=TRAINABLE_CLASSES + ['ambiguous', 'inconsistent'])

    # ---- (a) Train the `full` model on this synthetic pair, evaluate against TRUE labels ----
    y_true_all = true_df['true_label'].map(CLASS_TO_IDX).values.astype(np.int64)
    edges_train, y_train, edges_test, y_test = stratified_split(
        candidate_edges, y_true_all, args.diff_train_frac, args.random_seed)

    device = torch.device(f"cuda:{args.single_gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # scFM embeddings: reuse the REAL Geneformer embeddings for this gene panel (condition-
    # independent gene-identity embeddings -- exactly as intended, synthetic expression does
    # not change gene identity).
    scFM_embs = os.path.join(args.scFM_folder, "Geneformer")
    embs = pd.read_csv(os.path.join(scFM_embs, f'{args.cell_type}_{args.num_TF}_gene_embeddings.csv'))
    final_df = pd.read_csv(os.path.join(scFM_embs, f'{args.cell_type}_{args.num_TF}.csv'))
    zero_vec = np.zeros((len(embs.columns) - 1))
    gene_embeddings = []
    for _, row in final_df.iterrows():
        eid = row['ensembl_id']
        if eid in embs['Unnamed: 0'].values:
            gene_embeddings.append(embs[embs['Unnamed: 0'] == eid].values[0][1:].astype(np.float32))
        else:
            gene_embeddings.append(zero_vec)
    feat1 = torch.from_numpy(np.array(gene_embeddings)).float().to(device)

    x_a = torch.from_numpy(load_data(data_a).exp_data()).to(device)
    x_b = torch.from_numpy(load_data(data_b).exp_data()).to(device)

    # Shared adjacency: built from the UNION of G_A/G_B edges (the "static" structural
    # graph both conditions are encoded over), symmetrized exactly like Adj_Generate.
    union_edges = g_a_edges | g_b_edges
    adj_dense = np.zeros((n_genes, n_genes), dtype=np.float32)
    for (s, t) in union_edges:
        i, j = gene_to_idx[s], gene_to_idx[t]
        adj_dense[i, j] = 1.0
        adj_dense[j, i] = 1.0
    adj_coo = torch.from_numpy(adj_dense).to_sparse().to(device)

    # Auxiliary static task: use the union edges as positives + an equal-ish random-negative
    # sample, single shared label set (static structure isn't what's under test here).
    pos = np.array([[gene_to_idx[s], gene_to_idx[t], 1] for (s, t) in union_edges])
    n_neg = min(len(pos) * 3, n_genes * n_genes // 4)
    neg_s = rng.randint(0, n_genes, size=n_neg)
    neg_t = rng.randint(0, n_genes, size=n_neg)
    neg = np.stack([neg_s, neg_t, np.zeros(n_neg, dtype=int)], axis=1)
    static_all = np.concatenate([pos, neg], axis=0).astype(np.int64)
    rng.shuffle(static_all)
    n_static_train = int(len(static_all) * 0.8)
    static_train = torch.from_numpy(static_all[:n_static_train]).to(device)
    static_test = torch.from_numpy(static_all[n_static_train:]).to(device)

    class Args:
        pass
    margs = Args()
    for k, v in vars(args).items():
        setattr(margs, k, v)
    margs.gnn_hidden_dims = [227, 29, 34]
    margs.mlp_hidden_dims = [198]
    margs.gnn_num_layers = 3
    margs.mlp_num_layers = 1
    margs.dropout = 0.17477888900593389
    margs.gnn_dim_hidden = 256
    margs.type = 'MLP'
    margs.flag = True
    margs.diff_ablation = 'full'
    margs.use_cross_attention = False
    margs.diff_head_hidden = 128

    model = scDiffTransNet_GCN(input_dim_A=n_a, input_dim_B=n_b, args=margs, gene_dim=feat1.shape[1], device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0014411386699467767, weight_decay=1.0449337997892844e-05)

    edges_train_t = torch.from_numpy(edges_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    edges_test_t = torch.from_numpy(edges_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    class_counts_train = np.bincount(y_train, minlength=4).astype(np.float64)
    class_counts_safe = np.where(class_counts_train == 0, 1, class_counts_train)
    inv_freq = class_counts_safe.sum() / (4 * class_counts_safe)
    ce_weight = torch.tensor(inv_freq, dtype=torch.float32, device=device)

    def two_hot(y_int, n):
        out = np.zeros((n, 2), dtype=np.float32)
        out[np.arange(n), y_int] = 1.0
        return out

    static_train_y = torch.from_numpy(two_hot(static_train[:, -1].cpu().numpy(), len(static_train))).to(device)

    epochs = args.epochs
    best_f1, best_metrics, patience = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        static_pred_A, static_pred_B, diff_logits, r_A, r_B = model.forward_pair(
            x_a, x_b, adj_coo, feat1, static_train, edges_train_t)
        pred_A = torch.softmax(static_pred_A, dim=1)
        pred_B = torch.softmax(static_pred_B, dim=1)
        loss_static = F.binary_cross_entropy(pred_A, static_train_y) + F.binary_cross_entropy(pred_B, static_train_y)
        loss_ce = F.cross_entropy(diff_logits, y_train_t, weight=ce_weight)

        r_A_n = F.normalize(r_A, dim=1, eps=1e-8)
        r_B_n = F.normalize(r_B, dim=1, eps=1e-8)
        conserved_idx = torch.nonzero(y_train_t == CLASS_TO_IDX['conserved'], as_tuple=True)[0]
        specific_mask = (y_train_t == CLASS_TO_IDX['A-specific']) | (y_train_t == CLASS_TO_IDX['B-specific'])
        loss_infonce = torch.tensor(0.0, device=device)
        if conserved_idx.numel() >= 1:
            sim_all = torch.matmul(r_A_n[conserved_idx], r_B_n.T) / args.contrastive_temperature
            loss_infonce = F.cross_entropy(sim_all, conserved_idx)
        loss_margin = torch.tensor(0.0, device=device)
        if specific_mask.sum() >= 1:
            sim_s = (r_A_n[specific_mask] * r_B_n[specific_mask]).sum(dim=1)
            loss_margin = torch.clamp(sim_s - args.contrastive_margin, min=0).mean()

        loss = loss_static + loss_ce + 0.5 * (loss_infonce + loss_margin)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                _, _, diff_logits_test, _, _ = model.forward_pair(x_a, x_b, adj_coo, feat1, static_test, edges_test_t)
                probs = torch.softmax(diff_logits_test, dim=1).cpu().numpy()
                pred = probs.argmax(axis=1)
                y_np = y_test_t.cpu().numpy()
                from sklearn.metrics import f1_score
                f1s = []
                for c in range(4):
                    if (y_np == c).sum() > 0:
                        f1s.append(f1_score((y_np == c).astype(int), (pred == c).astype(int), zero_division=0))
                macro_f1 = float(np.mean(f1s)) if f1s else 0.0
                if macro_f1 >= best_f1:
                    best_f1 = macro_f1
                    per_class_auroc, per_class_auprc = {}, {}
                    for c in range(4):
                        name = TRAINABLE_CLASSES[c]
                        if (y_np == c).sum() > 0 and (y_np != c).sum() > 0:
                            y_bin = (y_np == c).astype(int)
                            per_class_auroc[name] = float(roc_auc_score(y_bin, probs[:, c]))
                            per_class_auprc[name] = float(average_precision_score(y_bin, probs[:, c]))
                        else:
                            per_class_auroc[name] = None
                            per_class_auprc[name] = None
                    overall_auroc_ovr = float(np.mean([v for v in per_class_auroc.values() if v is not None])) if any(v is not None for v in per_class_auroc.values()) else None
                    overall_auprc_ovr = float(np.mean([v for v in per_class_auprc.values() if v is not None])) if any(v is not None for v in per_class_auprc.values()) else None
                    best_metrics = {
                        'macro_f1': macro_f1,
                        'per_class_auroc': per_class_auroc,
                        'per_class_auprc': per_class_auprc,
                        'macro_auroc_ovr': overall_auroc_ovr,
                        'macro_auprc_ovr': overall_auprc_ovr,
                        'epoch': epoch + 1,
                        'confusion_matrix': confusion_matrix(y_np, pred, labels=list(range(4))).tolist(),
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= 10:
                        break

    result = {
        'severity': args.severity,
        'tf1_removed_from': tf1, 'tf2_added_to': tf2,
        'n_removed_edges': len(removed_edges), 'n_added_edges': len(added_edges),
        'n_conserved_edges': len(conserved_edges), 'n_neither_edges': len(neither_candidates),
        'silver_label_accuracy_vs_true_ALL_ROWS': float(silver_acc_all),
        'silver_label_accuracy_vs_true_TRAINABLE_SUBSET': float(silver_acc_trainable) if silver_acc_trainable is not None else None,
        'silver_confusion_matrix_labels_rows_true_cols_silver': TRAINABLE_CLASSES + ['ambiguous', 'inconsistent'],
        'silver_confusion_matrix': silver_cm.tolist(),
        'model_metrics_vs_true_labels': best_metrics,
        'diff_train_class_counts': {TRAINABLE_CLASSES[c]: int((y_train == c).sum()) for c in range(4)},
        'diff_test_class_counts': {TRAINABLE_CLASSES[c]: int((y_test == c).sum()) for c in range(4)},
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", type=str, default="./data")
    ap.add_argument("--scFM_folder", type=str, default="./scFM")
    ap.add_argument("--cell_type", type=str, default="hESC")
    ap.add_argument("--num_TF", type=str, default="500")
    ap.add_argument("--severity", type=float, default=0.15)
    ap.add_argument("--n_cells_a", type=int, default=92)
    ap.add_argument("--n_cells_b", type=int, default=188)
    ap.add_argument("--n_neither", type=int, default=120)
    ap.add_argument("--beta", type=float, default=4.0, help="Regulatory-strength constant")
    ap.add_argument("--n_regular_tfs", type=int, default=15)
    ap.add_argument("--targets_per_regular", type=int, default=6)
    ap.add_argument("--hub_out_degree", type=int, default=250)
    ap.add_argument("--corr_method", type=str, default="spearman")
    ap.add_argument("--corr_min_r", type=float, default=0.3)
    ap.add_argument("--fdr_q", type=float, default=0.05)
    ap.add_argument("--diff_train_frac", type=float, default=0.7)
    ap.add_argument("--contrastive_temperature", type=float, default=0.1)
    ap.add_argument("--contrastive_margin", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--single_gpu", type=int, default=0)
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.random_seed)
    result = run(args)
    print(json.dumps(result, indent=2, default=str))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    main()
