"""
DiffCorr-style Fisher z differential-correlation labeling pipeline.
DIFFERENTIAL_GRN_PLAN.md Sec 3 -- implemented exactly per the decision table in Sec 3.3.

Candidate edge space (Sec 3.1): TF->target pairs that are positive edges in the cell type's
BL--network.csv (== rows of Label.csv, which is the same edge list with Gene1/Gene2 already
mapped to the row-index of BL--ExpressionData.csv -- verified directly, Label.csv row i has
the same (TF, Target) gene pair as BL--network.csv row i, just as integer indices).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests

try:
    from utils import load_data
except ImportError:
    from src.utils import load_data


def _zscored_ranks(expr: np.ndarray) -> np.ndarray:
    """
    expr: (num_genes, num_cells) raw expression. Returns per-gene z-scored ranks
    (genes x cells), i.e. rank-transform each row then standardize it. Spearman
    correlation between two genes == Pearson correlation of these z-scored ranks
    (dot product / n), which lets us vectorize all candidate-edge correlations at once
    instead of calling scipy.stats.spearmanr per edge.
    """
    ranks = np.apply_along_axis(rankdata, 1, expr)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # guard constant rows (shouldn't happen post-rank, but safe)
    return (ranks - mean) / std


def _pearson_zscored(expr: np.ndarray) -> np.ndarray:
    mean = expr.mean(axis=1, keepdims=True)
    std = expr.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (expr - mean) / std


def compute_condition_corr(data_input_cond: pd.DataFrame, i_idx: np.ndarray, j_idx: np.ndarray,
                            corr_method: str = "spearman") -> np.ndarray:
    """
    data_input_cond: genes x cells_cond DataFrame restricted to one condition's cells (raw
    counts/values, same gene row order as the parent BL--ExpressionData.csv).
    i_idx, j_idx: integer gene-row indices (TF, Target) for each candidate edge.
    Returns r (num_edges,) -- Spearman (default) or Pearson correlation per edge, computed
    over this condition's cells only.

    Per plan Sec 3.2 step 1, this runs on the z-scored expression matrix produced by the
    existing `load_data.data_normalize` pipeline. Note: per-gene StandardScaler z-scoring is
    a monotonic-increasing transform, so it cannot change Spearman rank correlation -- we
    still route through load_data(...).exp_data() for pipeline consistency with the rest of
    the codebase, then apply the corr_method-specific transform below.
    """
    normalized = load_data(data_input_cond).exp_data()  # (num_genes, num_cells_cond), float32
    if corr_method == "spearman":
        z = _zscored_ranks(normalized)
    elif corr_method == "pearson":
        z = _pearson_zscored(normalized)
    else:
        raise ValueError(f"Unknown corr_method '{corr_method}'")
    n = normalized.shape[1]
    r = np.sum(z[i_idx] * z[j_idx], axis=1) / n
    r = np.clip(r, -0.999999, 0.999999)  # guard exact +/-1 before arctanh
    return r


LABEL_CONSERVED = "conserved"
LABEL_A_SPECIFIC = "A-specific"
LABEL_B_SPECIFIC = "B-specific"
LABEL_NEITHER = "neither"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_INCONSISTENT = "inconsistent"

# 4-way trainable classes, fixed integer encoding used throughout the differential model.
TRAINABLE_CLASSES = [LABEL_CONSERVED, LABEL_A_SPECIFIC, LABEL_B_SPECIFIC, LABEL_NEITHER]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAINABLE_CLASSES)}


def _decide_label(sig_a: bool, sig_b: bool, sig_diff: bool) -> str:
    if sig_a and sig_b and not sig_diff:
        return LABEL_CONSERVED
    if sig_a and (not sig_b) and sig_diff:
        return LABEL_A_SPECIFIC
    if (not sig_a) and sig_b and sig_diff:
        return LABEL_B_SPECIFIC
    if (not sig_a) and (not sig_b):
        return LABEL_NEITHER
    if sig_a and sig_b and sig_diff:
        return LABEL_AMBIGUOUS
    # remaining cases: (sig_a, not sig_b, not sig_diff) or (not sig_a, sig_b, not sig_diff)
    return LABEL_INCONSISTENT


def compute_differential_labels(data_a: pd.DataFrame, data_b: pd.DataFrame,
                                 candidate_edges: np.ndarray,
                                 corr_method: str = "spearman",
                                 corr_min_r: float = 0.3,
                                 fdr_q: float = 0.05) -> pd.DataFrame:
    """
    data_a, data_b: genes x cells_cond DataFrames already restricted to condition A / B's
        cells (e.g. via differential_data.load_condition_subsets), same gene row order as
        the parent BL--ExpressionData.csv.
    candidate_edges: (num_edges, 2) int array of (TF_row_idx, Target_row_idx) into that
        shared gene row order -- i.e. Label.csv's (TF, Target) columns.
    Returns a DataFrame with columns:
        TF, Target, TF_idx, Target_idx, r_A, r_B, p_A, p_B, p_diff,
        BH_p_A, BH_p_B, BH_p_diff, sig_A, sig_B, sig_diff, label
    per the exact decision table in plan Sec 3.3.
    """
    genes = data_a.index.to_numpy()
    i_idx = candidate_edges[:, 0].astype(int)
    j_idx = candidate_edges[:, 1].astype(int)
    n_a = data_a.shape[1]
    n_b = data_b.shape[1]

    r_a = compute_condition_corr(data_a, i_idx, j_idx, corr_method)
    r_b = compute_condition_corr(data_b, i_idx, j_idx, corr_method)

    z_a = np.arctanh(r_a)
    z_b = np.arctanh(r_b)

    # within-condition significance: z_X * sqrt(n_X - 3) ~ N(0,1) under H0: r_X = 0
    stat_a = z_a * np.sqrt(max(n_a - 3, 1))
    stat_b = z_b * np.sqrt(max(n_b - 3, 1))
    p_a = 2 * (1 - norm.cdf(np.abs(stat_a)))
    p_b = 2 * (1 - norm.cdf(np.abs(stat_b)))

    # differential (two-sample Fisher z) test
    se_diff = np.sqrt(1.0 / max(n_a - 3, 1) + 1.0 / max(n_b - 3, 1))
    z_diff = (z_a - z_b) / se_diff
    p_diff = 2 * (1 - norm.cdf(np.abs(z_diff)))

    # BH-FDR, three SEPARATE corrections (plan Sec 3.2 step 5)
    _, bh_p_a, _, _ = multipletests(p_a, alpha=fdr_q, method="fdr_bh")
    _, bh_p_b, _, _ = multipletests(p_b, alpha=fdr_q, method="fdr_bh")
    _, bh_p_diff, _, _ = multipletests(p_diff, alpha=fdr_q, method="fdr_bh")

    sig_a = (bh_p_a < fdr_q) & (np.abs(r_a) >= corr_min_r)
    sig_b = (bh_p_b < fdr_q) & (np.abs(r_b) >= corr_min_r)
    sig_diff = bh_p_diff < fdr_q

    labels = [
        _decide_label(bool(sa), bool(sb), bool(sd))
        for sa, sb, sd in zip(sig_a, sig_b, sig_diff)
    ]

    out = pd.DataFrame({
        "TF": genes[i_idx],
        "Target": genes[j_idx],
        "TF_idx": i_idx,
        "Target_idx": j_idx,
        "r_A": r_a,
        "r_B": r_b,
        "p_A": p_a,
        "p_B": p_b,
        "p_diff": p_diff,
        "BH_p_A": bh_p_a,
        "BH_p_B": bh_p_b,
        "BH_p_diff": bh_p_diff,
        "sig_A": sig_a,
        "sig_B": sig_b,
        "sig_diff": sig_diff,
        "label": labels,
    })
    return out


def get_candidate_edges(data_folder, cell_type, num_tf):
    """Load candidate edges (TF_idx, Target_idx) from Label.csv, matching BL--network.csv's
    row order (verified: Label.csv is BL--network.csv with Gene1/Gene2 mapped to
    BL--ExpressionData.csv row indices, same row order)."""
    path = os.path.join(data_folder, f"{cell_type}/TFs+{num_tf}/")
    label_file = os.path.join(path, "Label.csv")
    label_df = pd.read_csv(label_file, index_col=0)
    return label_df[["TF", "Target"]].values.astype(np.int64)


def get_cache_path(data_folder, cell_type, num_tf, condition_a, condition_b):
    d = os.path.join(data_folder, f"{cell_type}/TFs+{num_tf}/differential")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{condition_a}_vs_{condition_b}_labels.csv")


def load_or_compute_labels(data_folder, cell_type, num_tf, condition_a, condition_b,
                            corr_method="spearman", corr_min_r=0.3, fdr_q=0.05,
                            force_recompute=False) -> pd.DataFrame:
    try:
        from differential_data import load_condition_subsets
    except ImportError:
        from src.differential_data import load_condition_subsets

    cache_path = get_cache_path(data_folder, cell_type, num_tf, condition_a, condition_b)
    if os.path.exists(cache_path) and not force_recompute:
        return pd.read_csv(cache_path)

    exp_file = os.path.join(data_folder, f"{cell_type}/TFs+{num_tf}/BL--ExpressionData.csv")
    data_input = pd.read_csv(exp_file, index_col=0)
    cols_a, cols_b = load_condition_subsets(data_input, cell_type, condition_a, condition_b)
    candidate_edges = get_candidate_edges(data_folder, cell_type, num_tf)

    result = compute_differential_labels(
        cols_a, cols_b, candidate_edges,
        corr_method=corr_method, corr_min_r=corr_min_r, fdr_q=fdr_q,
    )
    result.to_csv(cache_path, index=False)
    return result


def class_counts(labels_df: pd.DataFrame) -> dict:
    return labels_df["label"].value_counts().to_dict()


def stratified_split(edges: np.ndarray, y: np.ndarray, train_frac: float, seed: int):
    """
    Deterministic stratified train/test split (plan Sec 5.1: "70/30, stratified by class").
    Guards tiny classes: a class with a single example goes entirely to train (cannot be
    split); a class with >=2 examples always keeps >=1 example in test. Deterministic given
    (edges, y, seed) so every script (DiffTrainer, posthoc_baseline, naive_baseline) that
    calls this with the same seed on the same cached labels gets an IDENTICAL split --
    required for a fair same-test-set comparison across configs.
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        if len(idx) == 1:
            train_idx.extend(idx)
        else:
            n_train = int(round(len(idx) * train_frac))
            n_train = min(max(n_train, 1), len(idx) - 1)
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx)) if test_idx else np.array([], dtype=int)
    return edges[train_idx], y[train_idx], edges[test_idx], y[test_idx]


if __name__ == "__main__":
    # Phase 0 dry run: labeling-yield check across all 9 condition pairs, CPU-only, no
    # model training (DIFFERENTIAL_GRN_PLAN.md Sec 7 "Phase 0").
    import argparse
    try:
        from differential_data import all_pairs
    except ImportError:
        from src.differential_data import all_pairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", type=str, default="./data")
    ap.add_argument("--num_tf", type=str, default="500")
    ap.add_argument("--corr_min_r", type=float, default=0.3)
    ap.add_argument("--fdr_q", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for cell_type, cond_a, cond_b, priority, rationale in all_pairs():
        print(f"=== {cell_type}: {cond_a} vs {cond_b} ({priority}) -- {rationale}")
        df = load_or_compute_labels(
            args.data_folder, cell_type, args.num_tf, cond_a, cond_b,
            corr_min_r=args.corr_min_r, fdr_q=args.fdr_q,
        )
        counts = class_counts(df)
        total = len(df)
        row = {"cell_type": cell_type, "pair": f"{cond_a}_vs_{cond_b}", "priority": priority, "n_candidate_edges": total}
        for c in TRAINABLE_CLASSES + [LABEL_AMBIGUOUS, LABEL_INCONSISTENT]:
            row[c] = counts.get(c, 0)
        rows.append(row)
        print(row)

    summary = pd.DataFrame(rows)
    print("\n=== Phase 0 summary (all 9 pairs) ===")
    print(summary.to_string(index=False))
    out_path = os.path.join(args.data_folder, "phase0_yield_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
