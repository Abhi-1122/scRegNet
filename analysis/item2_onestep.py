"""Item 2 check: is it PANDA's normalization+Tanimoto comparison, or the iteration, that makes
co-expression useful?  Compare one-step terms (raw vs PANDA-normalized) with the converged score."""
import sys
import numpy as np
from netZooPy.panda import calculations as calc
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, 'analysis')
import item2_panda_variants as V  # reuses its fixed TF/gene universe and matrices
from common import metrics, D

tr, te, genes, n = V.tr, V.te, V.genes, V.n
t, g, y = te.TF.values, te.Target.values, te.Label.values


def onestep(pos):
    motif = np.zeros((len(V.tf_univ), n))
    motif[[V.t2r[genes[x]] for x in pos.TF], pos.Target.values] = 1.0
    M, C, P = (calc.normalize_network(x) for x in (motif, V.corr, V.ppi))
    out = {}
    for name, mat in [('availability (co-expr), PANDA-normalized, 1 step', calc.t_function(M, C)),
                      ('responsibility (PPI), PANDA-normalized, 1 step', calc.t_function(P, M))]:
        full = np.zeros((n, n)); full[V.tf_row] = mat; out[name] = full
    return out


tp = tr[tr.Label == 1]
test_terms = onestep(tp)
oof = {k: np.zeros((n, n)) for k in test_terms}  # out-of-fold for train rows, as in build_panda
for k in oof: oof[k][:] = test_terms[k]
rng = np.random.RandomState(42)
for held in np.array_split(rng.permutation(tp.index.to_numpy()), 5):
    f = onestep(tp.loc[~tp.index.isin(held)]); h = tp.loc[held]
    for k in oof: oof[k][h.TF.values, h.Target.values] = f[k][h.TF.values, h.Target.values]

W = np.zeros((n, n)); W[tp.TF.values, tp.Target.values] = 1
rows = tr.groupby('TF').size().reindex(range(n), fill_value=0).values
def feats(df, loo):
    tt, gg, yy = df.TF.values, df.Target.values, df.Label.values
    sp = (yy == 1) * 1.0 if loo else 0
    return np.column_stack([(W.sum(1)[tt] - sp) / np.maximum(rows[tt] - loo, 1), np.log1p(W.sum(0)[gg] - sp)])
Xtr, Xte = feats(tr, 1), feats(te, 0)
print(f"{'term':52s} alone: AUROC withinTF | +degree LR: AUROC AUPRC withinTF")
for k in test_terms:
    s_te, s_tr = test_terms[k][t, g], oof[k][tr.TF.values, tr.Target.values]
    a = metrics(y, s_te, t)
    b = metrics(y, LogisticRegression(max_iter=1000).fit(np.column_stack([Xtr, s_tr]), tr.Label)
                .decision_function(np.column_stack([Xte, s_te])), t)
    print(f"{k:52s} {a['AUROC']:.4f} {a['withinTF_AUROC']:.4f}      | {b['AUROC']:.4f} {b['AUPRC']:.4f} {b['withinTF_AUROC']:.4f}")
