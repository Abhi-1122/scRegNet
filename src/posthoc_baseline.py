"""
posthoc_baseline -- adaptation of RegGAIN's differential-testing procedure to a per-edge
question (DIFFERENTIAL_GRN_PLAN.md Sec 5.1(a)).

Trains two INDEPENDENT static scTransNet_GCN models (today's unmodified architecture, one
per condition, each seeing only that condition's expression columns), scores every
candidate edge under each model, then runs a Wilcoxon rank-sum test per TF (grouping that
TF's out-edge scores under condition A vs condition B, exactly RegGAIN's per-TF procedure)
and collapses the result into the same 4-way label space used elsewhere: significant +
higher-under-A -> A-specific; significant + higher-under-B -> B-specific; not significant
-> conserved. This baseline has NO distinct "neither" call (RegGAIN's procedure never
attempted to test "is this edge active at all" separately from "did it change") -- true
`neither`-labeled test edges are therefore always missed by construction, which is reported
plainly as a structural limitation of the baseline, not smoothed over.
"""

import os
import copy
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import ranksums
from sklearn.metrics import f1_score, confusion_matrix
import warnings

from src.models import scTransNet_GCN
from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, set_seed, set_logging
from src.args import parse_args
from src.differential_data import load_condition_subsets
from src.differential_labels import (
    load_or_compute_labels, get_candidate_edges, TRAINABLE_CLASSES, CLASS_TO_IDX, stratified_split,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


def _get_geneformer_embeddings(args, data_input):
    scFM_embs = os.path.join(args.scFM_folder, "Geneformer")
    embs = pd.read_csv(os.path.join(scFM_embs, f'{args.cell_type}_{args.num_TF}_gene_embeddings.csv'))
    final_df = pd.read_csv(os.path.join(scFM_embs, f'{args.cell_type}_{args.num_TF}.csv'))
    x = np.zeros((len(embs.columns) - 1))
    gene_embeddings = []
    for _, row in final_df.iterrows():
        ensembl_id = row['ensembl_id']
        if ensembl_id in embs['Unnamed: 0'].values:
            gene_emb = embs[embs['Unnamed: 0'] == ensembl_id].values[0][1:]
            gene_embeddings.append(np.array(gene_emb, dtype=np.float32))
        else:
            gene_embeddings.append(x)
    return torch.from_numpy(np.array(gene_embeddings)).float()


def _train_static(x, adj, feat1, train_load, test_data_t, args, device, tag):
    model = scTransNet_GCN(input_dim=x.shape[1], args=args, gene_dim=feat1.shape[1], device=device).to(device)
    optimizer = getattr(optim, args.optimizer_name)(model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)

    max_auc, patience, best_state, best_aupr = 0.0, 0, None, 0.0
    for epoch in tqdm(range(args.gnn_epochs), desc=f"posthoc[{tag}]"):
        model.train()
        for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            train_y = train_y.to(device) if args.flag else train_y.to(device).view(-1, 1)
            pred = model(x, adj, train_x, feat1)
            pred = torch.softmax(pred, dim=1) if args.flag else torch.sigmoid(pred)
            loss = F.binary_cross_entropy(pred, train_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.gnn_eval_interval == 0:
            model.eval()
            with torch.no_grad():
                score = model(x, adj, test_data_t, feat1)
                score = torch.softmax(score, dim=1) if args.flag else torch.sigmoid(score)
                auc, aupr, _ = Evaluation(y_pred=score, y_true=test_data_t[:, -1], flag=args.flag)
            if auc > max_auc:
                max_auc, best_aupr, patience = auc, aupr, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= args.diff_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max_auc, best_aupr


def run_posthoc_baseline(args):
    device = torch.device(f"cuda:{args.single_gpu}") if torch.cuda.is_available() else torch.device("cpu")
    path = os.path.join(args.data_folder, f'{args.cell_type}/TFs+{args.num_TF}/')
    data_input = pd.read_csv(os.path.join(path, 'BL--ExpressionData.csv'), index_col=0)
    train_data = pd.read_csv(os.path.join(path, 'Train_set.csv'), index_col=0).values
    test_data = pd.read_csv(os.path.join(path, 'Test_set.csv'), index_col=0).values
    tf = pd.read_csv(os.path.join(path, 'TF.csv'), index_col=0)['index'].values.astype(np.int64)
    tf_t = torch.from_numpy(tf).to(device)
    gene_num = data_input.shape[0]

    feat1 = _get_geneformer_embeddings(args, data_input).to(device)

    cols_a, cols_b = load_condition_subsets(data_input, args.cell_type, args.condition_a, args.condition_b)
    x_a = torch.from_numpy(load_data(cols_a).exp_data()).to(device)
    x_b = torch.from_numpy(load_data(cols_b).exp_data()).to(device)

    train_load = scRNADataset(train_data, gene_num, flag=args.flag)
    adj = adj2saprse_tensor(train_load.Adj_Generate(tf_t, loop=args.loop)).to(device)
    test_data_t = torch.from_numpy(test_data).to(device)

    model_a, auc_a, aupr_a = _train_static(x_a, adj, feat1, train_load, test_data_t, args, device, f"{args.condition_a}")
    model_b, auc_b, aupr_b = _train_static(x_b, adj, feat1, train_load, test_data_t, args, device, f"{args.condition_b}")
    logger.critical(f"posthoc static models: {args.condition_a} AUROC={auc_a:.4f} AUPRC={aupr_a:.4f} | "
                     f"{args.condition_b} AUROC={auc_b:.4f} AUPRC={aupr_b:.4f}")

    candidate_edges = get_candidate_edges(args.data_folder, args.cell_type, args.num_TF)
    i_idx = torch.from_numpy(candidate_edges[:, 0]).to(device)
    j_idx = torch.from_numpy(candidate_edges[:, 1]).to(device)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        _ = model_a(x_a, adj, test_data_t, feat1)  # populate full-gene tf/target embeddings
        tf_embed_a, target_embed_a = model_a.get_embedding()
        scores_a = model_a.decode(tf_embed_a[i_idx], target_embed_a[j_idx])
        scores_a = torch.softmax(scores_a, dim=1)[:, -1] if args.flag else scores_a.view(-1)
        scores_a = scores_a.cpu().numpy()

        _ = model_b(x_b, adj, test_data_t, feat1)
        tf_embed_b, target_embed_b = model_b.get_embedding()
        scores_b = model_b.decode(tf_embed_b[i_idx], target_embed_b[j_idx])
        scores_b = torch.softmax(scores_b, dim=1)[:, -1] if args.flag else scores_b.view(-1)
        scores_b = scores_b.cpu().numpy()

    # Per-TF Wilcoxon rank-sum test (RegGAIN's own procedure), collapsed to a per-edge call.
    tf_ids = candidate_edges[:, 0]
    df = pd.DataFrame({'TF_idx': tf_ids, 'Target_idx': candidate_edges[:, 1], 'score_A': scores_a, 'score_B': scores_b})
    posthoc_label = np.empty(len(df), dtype=object)
    tf_pvalues = {}
    for tf_idx, group in df.groupby('TF_idx'):
        sa, sb = group['score_A'].values, group['score_B'].values
        if len(sa) >= 2:
            try:
                stat, p = ranksums(sa, sb)
            except Exception:
                p = 1.0
        else:
            p = 1.0  # cannot test a single out-edge TF; default to "not significant" -> conserved
        tf_pvalues[int(tf_idx)] = float(p)
        if p < 0.05:
            label = "A-specific" if sa.mean() > sb.mean() else "B-specific"
        else:
            label = "conserved"
        posthoc_label[group.index.values] = label
    df['posthoc_label'] = posthoc_label

    # Evaluate against the SAME silver-label diff-test split the `full`/naive configs use.
    labels_df = load_or_compute_labels(
        args.data_folder, args.cell_type, args.num_TF, args.condition_a, args.condition_b,
        corr_method=args.corr_method, corr_min_r=args.corr_min_r, fdr_q=args.fdr_q,
    )
    trainable = labels_df[labels_df['label'].isin(TRAINABLE_CLASSES)].reset_index(drop=True)
    edges = trainable[['TF_idx', 'Target_idx']].values.astype(np.int64)
    y = trainable['label'].map(CLASS_TO_IDX).values.astype(np.int64)
    _, _, edges_test, y_test = stratified_split(edges, y, args.diff_train_frac, args.random_seed)

    df_idx = df.set_index(['TF_idx', 'Target_idx'])
    pred_labels = []
    for tf_idx, tgt_idx in edges_test:
        pred_labels.append(df_idx.loc[(tf_idx, tgt_idx), 'posthoc_label'])
    y_pred = np.array([CLASS_TO_IDX[l] for l in pred_labels])

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
            f1s.append(entry['f1'])
            included.append(name)
        else:
            entry['f1'] = None
        per_class[name] = entry
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    cm = confusion_matrix(y_test, y_pred, labels=list(range(4))).tolist()

    return {
        'static_auc_A': float(auc_a), 'static_aupr_A': float(aupr_a),
        'static_auc_B': float(auc_b), 'static_aupr_B': float(aupr_b),
        'macro_f1': macro_f1,
        'per_class': per_class,
        'per_class_macro_f1_classes_included': included,
        'confusion_matrix': cm,
        'confusion_matrix_labels': TRAINABLE_CLASSES,
        'n_TFs_tested': len(tf_pvalues),
        'diff_test_class_counts': {TRAINABLE_CLASSES[c]: int((y_test == c).sum()) for c in range(4)},
    }


def main():
    set_logging()
    args = parse_args()
    set_seed(random_seed=args.random_seed)
    logger.critical(f"posthoc_baseline: {args.cell_type} {args.condition_a} vs {args.condition_b}")
    metrics = run_posthoc_baseline(args)
    logger.info(json.dumps(metrics, indent=2, default=str))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    return metrics


if __name__ == "__main__":
    metrics = main()
    print(json.dumps(metrics, indent=2, default=str))
