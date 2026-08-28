"""
naive_expression_baseline (DIFFERENTIAL_GRN_PLAN.md Sec 5.1(b)): logistic regression on
(r_A, r_B, TF mean-expression log2 fold-change, target mean-expression log2 fold-change)
only -- no GNN, no scFM, no learned embeddings. Operationalizes the question "does a
GRN-level differential claim beat gene-level differential statistics alone."

Standalone script, no GPU needed, run once per condition pair (not per ablation).
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

from src.differential_data import load_condition_subsets
from src.differential_labels import load_or_compute_labels, TRAINABLE_CLASSES, CLASS_TO_IDX


def build_features(args):
    path = os.path.join(args.data_folder, f'{args.cell_type}/TFs+{args.num_TF}/')
    data_input = pd.read_csv(os.path.join(path, 'BL--ExpressionData.csv'), index_col=0)
    data_a, data_b = load_condition_subsets(data_input, args.cell_type, args.condition_a, args.condition_b)

    labels_df = load_or_compute_labels(
        args.data_folder, args.cell_type, args.num_TF, args.condition_a, args.condition_b,
        corr_method=args.corr_method, corr_min_r=args.corr_min_r, fdr_q=args.fdr_q,
    )
    trainable = labels_df[labels_df['label'].isin(TRAINABLE_CLASSES)].reset_index(drop=True)

    mean_a = data_a.mean(axis=1)
    mean_b = data_b.mean(axis=1)
    eps = 1e-6
    log2fc = np.log2((mean_b + eps) / (mean_a + eps))

    tf_fc = trainable['TF'].map(log2fc).values
    tgt_fc = trainable['Target'].map(log2fc).values

    X = np.stack([trainable['r_A'].values, trainable['r_B'].values, tf_fc, tgt_fc], axis=1).astype(np.float64)
    y = trainable['label'].map(CLASS_TO_IDX).values.astype(np.int64)
    edges = trainable[['TF_idx', 'Target_idx']].values.astype(np.int64)
    return X, y, edges


def run(args):
    X, y, edges = build_features(args)

    # Same deterministic stratified-split LOGIC as differential_labels.stratified_split
    # (same seed, same per-class shuffle order since y's per-class row order matches the
    # edges/labels_df ordering used everywhere else) -- reimplemented inline here only
    # because we need X's rows split in lockstep with y/edges, which stratified_split's
    # (edges, y) signature doesn't return.
    rng = np.random.RandomState(args.random_seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        if len(idx) == 1:
            train_idx.extend(idx)
        else:
            n_train = int(round(len(idx) * args.diff_train_frac))
            n_train = min(max(n_train, 1), len(idx) - 1)
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx)) if test_idx else np.array([], dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    present_classes = np.unique(y_train)
    if len(present_classes) < 2:
        # Degenerate: only one class in training data -- logistic regression can't fit.
        y_pred = np.full(len(y_test), present_classes[0] if len(present_classes) else 0)
        probs = None
    else:
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        probs_present = clf.predict_proba(X_test)
        probs = np.zeros((len(y_test), 4))
        for k, c in enumerate(clf.classes_):
            probs[:, c] = probs_present[:, k]

    floor = args.diff_class_count_floor
    class_counts = np.bincount(y_test, minlength=4)
    per_class = {}
    f1s, included = [], []
    for c in range(4):
        name = TRAINABLE_CLASSES[c]
        n = int(class_counts[c])
        entry = {'n_test': n}
        if n >= floor:
            y_bin = (y_test == c).astype(int)
            entry['f1'] = float(f1_score(y_bin, (y_pred == c).astype(int), zero_division=0))
            if probs is not None:
                try:
                    entry['auroc'] = float(roc_auc_score(y_bin, probs[:, c]))
                    entry['auprc'] = float(average_precision_score(y_bin, probs[:, c]))
                except Exception:
                    entry['auroc'], entry['auprc'] = None, None
            f1s.append(entry['f1'])
            included.append(name)
        else:
            entry['f1'] = None
        per_class[name] = entry
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    cm = confusion_matrix(y_test, y_pred, labels=list(range(4))).tolist()

    return {
        'macro_f1': macro_f1,
        'per_class': per_class,
        'macro_f1_classes_included': included,
        'confusion_matrix': cm,
        'confusion_matrix_labels': TRAINABLE_CLASSES,
        'diff_train_class_counts': {TRAINABLE_CLASSES[c]: int((y_train == c).sum()) for c in range(4)},
        'diff_test_class_counts': {TRAINABLE_CLASSES[c]: int((y_test == c).sum()) for c in range(4)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", type=str, default="./data")
    ap.add_argument("--cell_type", type=str, required=True)
    ap.add_argument("--num_TF", type=str, default="500")
    ap.add_argument("--condition_a", type=str, required=True)
    ap.add_argument("--condition_b", type=str, required=True)
    ap.add_argument("--corr_method", type=str, default="spearman")
    ap.add_argument("--corr_min_r", type=float, default=0.3)
    ap.add_argument("--fdr_q", type=float, default=0.05)
    ap.add_argument("--diff_train_frac", type=float, default=0.7)
    ap.add_argument("--diff_class_count_floor", type=int, default=20)
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    metrics = run(args)
    print(json.dumps(metrics, indent=2, default=str))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    return metrics


if __name__ == "__main__":
    main()
