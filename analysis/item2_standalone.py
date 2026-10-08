"""Item 2, no-training part: score each PANDA variant alone and inside a degree LR."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, 'analysis')
from common import load, metrics, D

tr, te, genes, expr, tfs = load()
assert np.array_equal(np.load(D + 'PANDA_variants/full/panda_full_matrix_train_oof.npy'),
                      np.load(D + 'PANDA/panda_full_matrix_train_oof.npy')), 'OOF mismatch'
print('OOF matrix of driver == original build_panda.py OOF matrix: exact match')

n = len(genes)
W = np.zeros((n, n)); p = tr[tr.Label == 1]; W[p.TF.values, p.Target.values] = 1
rows_per_tf = tr.groupby('TF').size().reindex(range(n), fill_value=0).values


def degree_feats(df, loo):
    t, g, y = df.TF.values, df.Target.values, df.Label.values
    self_pos = (y == 1) * 1.0 if loo else 0
    return np.column_stack([(W.sum(1)[t] - self_pos) / np.maximum(rows_per_tf[t] - loo, 1),
                            np.log1p(W.sum(0)[g] - self_pos)])


Xtr, Xte = degree_feats(tr, 1), degree_feats(te, 0)
ytr, yte, tte = tr.Label.values, te.Label.values, te.TF.values
print(f"{'variant':12s} {'alone: AUROC AUPRC withinTF':28s} {'degree+variant LR: AUROC AUPRC withinTF'}")
for v in ['full', 'no_coexpr', 'no_ppi', 'prior_only', 'raw_corr']:
    test_m = np.load(D + f'PANDA_variants/{v}/panda_full_matrix.npy')
    oof_m = np.load(D + f'PANDA_variants/{v}/panda_full_matrix_train_oof.npy')
    s_te = test_m[te.TF.values, te.Target.values]
    s_tr = oof_m[tr.TF.values, tr.Target.values]
    a = metrics(yte, s_te, tte)
    lr = LogisticRegression(max_iter=1000).fit(np.column_stack([Xtr, s_tr]), ytr)
    b = metrics(yte, lr.decision_function(np.column_stack([Xte, s_te])), tte)
    print(f"{v:12s} {a['AUROC']:.4f} {a['AUPRC']:.4f} {a['withinTF_AUROC']:.4f}        "
          f"{b['AUROC']:.4f} {b['AUPRC']:.4f} {b['withinTF_AUROC']:.4f}")
