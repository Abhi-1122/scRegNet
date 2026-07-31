"""One-off script: build PANDA's three inputs (expression, motif/regulatory
prior, TF-TF PPI) for hESC/TFs+500 and run netZooPy's Panda, persisting the
refined TF x gene regulatory-score matrix for use by Experiments B and C.

Also builds a leakage-safe variant for training: PANDA's motif prior IS the
set of train-positive edges (weight=1.0), so a train-positive edge's own
PANDA score is almost a direct echo of "this was given weight 1" (verified:
train-positive scores average ~10 vs ~0 for everything else, while test
scores show no such separation since test edges were never in any prior).
Using that score as a training feature is target leakage. We fix it with
K-fold out-of-fold scoring: for each fold, rerun PANDA with that fold's
positives excluded from the prior, and use those held-out scores for the
train-positive edges in that fold. Test-edge scoring is untouched (already
leakage-free, since test edges are never part of any prior).
"""
import os
import numpy as np
import pandas as pd
from netZooPy.panda import Panda

DATA_DIR = "./data/hESC/TFs+500"
OUT_DIR = os.path.join(DATA_DIR, "PANDA")
K_FOLDS = 5
SEED = 42


def build_expression_df():
    return pd.read_csv(os.path.join(DATA_DIR, "BL--ExpressionData.csv"), index_col=0)


def build_motif_df(train_pos_df, idx_to_gene):
    motif_df = pd.DataFrame({
        "source": train_pos_df["TF"].map(idx_to_gene),
        "target": train_pos_df["Target"].map(idx_to_gene),
        "weight": 1.0,
    })
    assert motif_df["source"].notna().all() and motif_df["target"].notna().all()
    return motif_df


def build_ppi_df():
    tf_df = pd.read_csv(os.path.join(DATA_DIR, "TF.csv"), index_col=0)
    tf_set = set(tf_df["TF"])
    edges = pd.read_csv(os.path.join(DATA_DIR, "STRING", "hESC_TFs500_STRING_PPI.edges.csv"))
    tf_edges = edges[edges["Gene1"].isin(tf_set) & edges["Gene2"].isin(tf_set)]
    ppi_df = pd.DataFrame({0: tf_edges["Gene1"].values, 1: tf_edges["Gene2"].values, 2: 1.0})
    return ppi_df, tf_df


def run_panda_raw(motif_df, expression_df, ppi_df, tf_df, target_df, idx_to_gene, num_genes):
    """Runs PANDA and returns the raw (unscaled), re-indexed (num_tfs, num_genes) matrix
    plus the TF global-index array giving each row's position in the 910-gene space."""
    panda_obj = Panda(
        expression_df, motif_df, ppi_df,
        computing="cpu", precision="double",
        save_memory=True, remove_missing=False, modeProcess="union",
    )
    W = panda_obj.panda_network  # DataFrame: index=TFs, columns=genes (alphabetically sorted)
    all_genes_by_idx = [idx_to_gene[i] for i in range(num_genes)]
    tf_global_idx = tf_df["index"].values.astype(np.int64)
    tf_symbols_ordered = [idx_to_gene[i] for i in tf_global_idx]
    W_reindexed = W.reindex(index=tf_symbols_ordered, columns=all_genes_by_idx, fill_value=0.0)
    return W_reindexed.values.astype(np.float32), tf_global_idx


def scatter_full(raw_values, tf_global_idx, num_genes, scale):
    """Scale (not center, so zero-padding stays 0) and scatter (num_tfs, num_genes)
    rows into a full (num_genes, num_genes) matrix indexed by global gene-index."""
    full = np.zeros((num_genes, num_genes), dtype=np.float32)
    full[tf_global_idx, :] = raw_values / (scale + 1e-8)
    return full


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    expression_df = build_expression_df()
    ppi_df, tf_df = build_ppi_df()
    target_df = pd.read_csv(os.path.join(DATA_DIR, "Target.csv"), index_col=0)
    idx_to_gene = dict(zip(target_df["index"], target_df["Gene"]))
    num_genes = len(target_df)

    train = pd.read_csv(os.path.join(DATA_DIR, "Train_set.csv"), index_col=0)
    train_pos = train[train["Label"] == 1]
    print(f"expression: {expression_df.shape} (genes x cells)")
    print(f"motif prior: {len(train_pos)} positive TF-target edges")
    print(f"TF-TF PPI: {len(ppi_df)} edges out of {len(tf_df)} TFs")

    # --- Full-prior run: used for test-edge scoring (Experiment C) and the per-gene
    # profile feature (Experiment B). Both are leakage-safe as-is: test edges are never
    # in the prior, and Experiment B's per-gene feature dilutes any single edge's
    # membership across 410 dimensions rather than exposing it as one raw scalar. ---
    motif_df_full = build_motif_df(train_pos, idx_to_gene)
    raw_full, tf_global_idx = run_panda_raw(motif_df_full, expression_df, ppi_df, tf_df, target_df, idx_to_gene, num_genes)
    scale = raw_full.std()
    print(f"raw PANDA values (full prior): mean={raw_full.mean():.4f} std={scale:.4f} max={raw_full.max():.4f}")
    panda_full = scatter_full(raw_full, tf_global_idx, num_genes, scale)
    np.save(os.path.join(OUT_DIR, "panda_full_matrix.npy"), panda_full)
    print(f"Saved {OUT_DIR}/panda_full_matrix.npy (full-prior; use for test-edge scoring "
          f"and Experiment B's per-gene profile)")

    # --- K-fold out-of-fold run: used for train-edge scoring in Experiment C only. For
    # each of the ~3030 train-positive edges, score it using a prior that EXCLUDES it
    # (and its fold-mates), so its own score can't be a direct echo of its own label. ---
    rng = np.random.RandomState(SEED)
    pos_positions = train_pos.index.to_numpy()
    shuffled = rng.permutation(pos_positions)
    folds = np.array_split(shuffled, K_FOLDS)

    panda_full_train = panda_full.copy()  # negatives (never in any prior) stay as-is
    for k, held_out in enumerate(folds):
        remaining = train_pos.loc[~train_pos.index.isin(held_out)]
        motif_df_fold = build_motif_df(remaining, idx_to_gene)
        raw_fold, tf_idx_fold = run_panda_raw(motif_df_fold, expression_df, ppi_df, tf_df, target_df, idx_to_gene, num_genes)
        full_fold = scatter_full(raw_fold, tf_idx_fold, num_genes, scale)  # reuse the SAME scale as the full run

        held_out_rows = train_pos.loc[held_out]
        panda_full_train[held_out_rows["TF"].values, held_out_rows["Target"].values] = \
            full_fold[held_out_rows["TF"].values, held_out_rows["Target"].values]
        print(f"fold {k+1}/{K_FOLDS}: {len(held_out)} held-out positive edges re-scored out-of-fold")

    np.save(os.path.join(OUT_DIR, "panda_full_matrix_train_oof.npy"), panda_full_train)
    print(f"Saved {OUT_DIR}/panda_full_matrix_train_oof.npy (leakage-safe; use for "
          f"train-edge scoring in Experiment C)")

    # sanity: confirm the leakage is actually fixed
    train_scores_leaky = panda_full[train_pos["TF"].values, train_pos["Target"].values]
    train_scores_safe = panda_full_train[train_pos["TF"].values, train_pos["Target"].values]
    print(f"train-positive scores BEFORE fix: mean={train_scores_leaky.mean():.4f} std={train_scores_leaky.std():.4f}")
    print(f"train-positive scores AFTER  fix: mean={train_scores_safe.mean():.4f} std={train_scores_safe.std():.4f}")


if __name__ == "__main__":
    main()
