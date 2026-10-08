"""Item 2 checks: (a) is the one-step PPI term just target popularity?  (b) paired bootstrap
CIs on the degree+variant LR differences, resampling test TFs (the unit that actually varies)."""
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis')
from common import load, ppi_tf_adj, D

tr, te, genes, expr, tfs = load()
n = len(genes)
W = np.zeros((n, n)); p = tr[tr.Label == 1]; W[p.TF.values, p.Target.values] = 1
P = ppi_tf_adj(genes, tfs)
t, g, y = te.TF.values, te.Target.values, te.Label.values

# (a) within each TF: Spearman between "partners that target it" and target train degree
ppi_term = (P[t] * W[:, g].T).sum(1)
deg = W.sum(0)[g]
rs = [spearmanr(ppi_term[t == k], deg[t == k])[0] for k in np.unique(t) if ppi_term[t == k].std() > 0]
print(f'(a) within-TF Spearman(PPI-partner term, target degree): median {np.median(rs):.3f} '
      f'over {len(rs)} TFs with PPI partners; {sum(1 for k in np.unique(t) if P[k].sum() == 0)} TFs have no TF partner in STRING')
v = {}
for name in ['full', 'no_coexpr', 'no_ppi', 'prior_only']:
    m = np.load(D + f'PANDA_variants/{name}/panda_full_matrix.npy')[t, g]
    v[name] = np.median([spearmanr(m[t == k], deg[t == k])[0] for k in np.unique(t)])
print('    within-TF Spearman(PANDA variant score, target degree):', {k: round(x, 3) for k, x in v.items()})

# (b) LR scores, then bootstrap over TFs
rows = tr.groupby('TF').size().reindex(range(n), fill_value=0).values


def feats(df, loo):
    tt, gg, yy = df.TF.values, df.Target.values, df.Label.values
    sp = (yy == 1) * 1.0 if loo else 0
    return np.column_stack([(W.sum(1)[tt] - sp) / np.maximum(rows[tt] - loo, 1), np.log1p(W.sum(0)[gg] - sp)])


Xtr, Xte = feats(tr, 1), feats(te, 0)
S = {'degree': LogisticRegression(max_iter=1000).fit(Xtr, tr.Label).decision_function(Xte)}
for name in ['full', 'no_coexpr', 'no_ppi', 'prior_only', 'raw_corr']:
    a = np.load(D + f'PANDA_variants/{name}/panda_full_matrix_train_oof.npy')[tr.TF.values, tr.Target.values]
    b = np.load(D + f'PANDA_variants/{name}/panda_full_matrix.npy')[t, g]
    S[name] = LogisticRegression(max_iter=1000).fit(np.column_stack([Xtr, a]), tr.Label).decision_function(np.column_stack([Xte, b]))

tf_ids = [k for k in np.unique(t) if 0 < y[t == k].sum() < (t == k).sum()]
per = {name: np.array([roc_auc_score(y[t == k], s[t == k]) for k in tf_ids]) for name, s in S.items()}
rng = np.random.RandomState(0)
B = rng.randint(0, len(tf_ids), (5000, len(tf_ids)))
print('\n(b) within-TF AUROC differences, 95% CI from 5000 bootstrap resamples of the 33 test TFs')
for a, b in [('full', 'degree'), ('no_coexpr', 'degree'), ('no_ppi', 'degree'), ('prior_only', 'degree'),
             ('raw_corr', 'degree'), ('full', 'no_coexpr'), ('full', 'no_ppi')]:
    d = per[a] - per[b]
    boot = d[B].mean(1)
    print(f'  {a:10s} - {b:10s}: {d.mean():+.4f}  [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]'
          f'  TFs improved: {(d > 0).sum()}/{len(d)}')
