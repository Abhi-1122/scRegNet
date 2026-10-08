"""Item 1: score test edges with each information source on its own, no model."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import load, ppi_tf_adj, metrics, D

tr, te, genes, expr, tfs = load()
C = np.corrcoef(expr)
np.fill_diagonal(C, 0)
P = ppi_tf_adj(genes, tfs)
panda = np.load(D + 'PANDA/panda_full_matrix.npy')  # full prior; test edges never in prior -> leak-free
n = len(genes)

# train-positive adjacency (TF x gene), the only label info any signal may use
Wtr = np.zeros((n, n), dtype=np.float32)
pos = tr[tr.Label == 1]
Wtr[pos.TF.values, pos.Target.values] = 1


def signals(df, loo):
    t, g, y = df.TF.values, df.Target.values, df.Label.values
    self_pos = (y == 1).astype(float) if loo else 0  # leave-one-out: drop the row's own label
    W = Wtr
    tf_npos = W.sum(1)[t] - self_pos
    tf_nrows = tr.groupby('TF').size().reindex(range(n), fill_value=0).values[t] - (1 if loo else 0)
    s = {
        'TF positive rate in train (TF identity only)': tf_npos / np.maximum(tf_nrows, 1),
        'target train degree (target identity only)': W.sum(0)[g] - self_pos,
        'raw corr(TF, target)': C[t, g],
        '|corr(TF, target)|': np.abs(C[t, g]),
    }
    # guilt by association = PANDA's "availability" in one step: mean corr of target with TF's train targets
    ga = (W[t] * C[g]).sum(1) - self_pos * C[g, g]  # C[g,g]=0 anyway
    s['guilt-by-assoc: mean corr(target, TF\'s other targets)'] = ga / np.maximum(tf_npos, 1)
    # PANDA's "responsibility" in one step: how many of the TF's PPI partners target this gene
    s['PPI partners that target it'] = (P[t] * W[:, g].T).sum(1)
    s['PANDA score'] = panda[t, g]
    return s, y, t


s_tr, y_tr, t_tr = signals(tr, loo=True)
s_te, y_te, t_te = signals(te, loo=False)

rows = []
for k in s_te:
    m = metrics(y_te, s_te[k], t_te)
    rows.append((k, m))

# degree-only model: logistic regression on TF rate + target degree (LOO features on train)
X = lambda s: np.column_stack([s['TF positive rate in train (TF identity only)'],
                               np.log1p(s['target train degree (target identity only)'])])
lr = LogisticRegression(max_iter=1000).fit(X(s_tr), y_tr)
rows.append(('DEGREE-ONLY model (TF rate + target degree, LR)', metrics(y_te, lr.decision_function(X(s_te)), t_te)))

X3 = lambda s: np.column_stack([X(s), s['PANDA score']])
# train rows: PANDA OOF matrix (train positives scored with themselves excluded from the prior)
s_tr['PANDA score'] = np.load(D + 'PANDA/panda_full_matrix_train_oof.npy')[tr.TF.values, tr.Target.values]
lr3 = LogisticRegression(max_iter=1000).fit(X3(s_tr), y_tr)
rows.append(('DEGREE + PANDA (3-feature LR, no deep model)', metrics(y_te, lr3.decision_function(X3(s_te)), t_te)))

import os
models = {'no_grn': 'Geneformer only (no GNN)', 'no_scfm': 'GNN only (no Geneformer)',
          'baseline_rerun': 'FULL scRegNet (GNN + Geneformer)', 'panda_scfm_only': 'Geneformer + PANDA (Exp D)',
          'panda_only': 'GNN + PANDA profile, no Geneformer (Exp B)', 'panda_edge': 'FULL + PANDA scalar (Exp C)'}
for k, name in models.items():
    f = f'analysis/out/scores_tf_500_hESC_{"ablation_" if k.startswith("no_") else ""}{k}.npz'
    if os.path.exists(f):
        z = np.load(f)
        rows.append(('MODEL: ' + name, metrics(y_te, z['test_logits'][:, 1] - z['test_logits'][:, 0], t_te)))

print(f"{'signal':58s} AUROC  AUPRC  withinTF_AUROC")
for k, m in rows:
    print(f"{k:58s} {m['AUROC']:.4f} {m['AUPRC']:.4f} {m['withinTF_AUROC']:.4f}  (TFs={m['n_TF']})")
print('test positive rate:', y_te.mean().round(4))
