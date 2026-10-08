"""Item 3: which test edges does PANDA (Exp C vs baseline) move, stratified by graph properties.
Per-edge measure: within-TF percentile of the edge's score (0 = bottom of that TF's list, 1 = top).
For a true target, moving up is good; for a non-target, moving down is good."""
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis')
from common import load, ppi_tf_adj, D

tr, te, genes, expr, tfs = load()
n = len(genes); t, g, y = te.TF.values, te.Target.values, te.Label.values
z = lambda r: np.load(f'analysis/out/scores_tf_500_hESC_{r}.npz')
sb = z('baseline_rerun')['test_logits']; sb = sb[:, 1] - sb[:, 0]
sc = z('panda_edge')['test_logits']; sc = sc[:, 1] - sc[:, 0]


def pct(x):
    r = np.empty(len(x))
    for k in np.unique(t):
        m = t == k; r[m] = (np.argsort(np.argsort(x[m])) + 0.5) / m.sum()
    return r


pb, pc = pct(sb), pct(sc)
gain = np.where(y == 1, pc - pb, pb - pc)  # >0 means the edge moved the right way

p = tr[tr.Label == 1]
A = np.zeros((n, n)); A[p.TF.values, p.Target.values] = A[p.Target.values, p.TF.values] = 1  # GCN graph
dist = shortest_path(csr_matrix(A), unweighted=True, indices=np.unique(t))
dmap = dict(zip(np.unique(t), dist))
d = np.array([dmap[a][b] for a, b in zip(t, g)])
tdeg = A.sum(0)[g]
Wtr = np.zeros((n, n)); Wtr[p.TF.values, p.Target.values] = 1
P = ppi_tf_adj(genes, tfs)
partners_target = (P[t] * Wtr[:, g].T).sum(1)  # how many of the TF's STRING partners target this gene
tf_deg = Wtr.sum(1)[t]
C = np.nan_to_num(np.corrcoef(expr)); np.fill_diagonal(C, 0)
# does the gene co-express with the TF's train targets more than with its non-targets? (Item 2's winner)
coexp_contrast = np.array([C[b, Wtr[a] == 1].mean() - C[b, Wtr[a] == 0].mean() for a, b in zip(t, g)])

df = pd.DataFrame(dict(y=y, gain=gain, dist=d, tdeg=tdeg, ppi=partners_target, tfdeg=tf_deg,
                       coexp=coexp_contrast, tf=t))


def report(col, bins, labels):
    df['bin'] = pd.cut(df[col], bins=bins, labels=labels, right=True)
    print(f'\n-- by {col} --')
    print(f"{'stratum':22s} {'n_pos':>5s} {'n_neg':>5s}  {'pos moved up':>12s} {'neg moved down':>14s}  {'withinTF AUROC base -> ExpC':>28s}")
    for lab in labels:
        s = df[df.bin == lab]
        if len(s) == 0: continue
        sp, sn = s[s.y == 1], s[s.y == 0]
        # AUROC inside the stratum, within TF, then averaged over TFs that have both classes
        aucs = []
        for k in s.tf.unique():
            m = (df.bin == lab).values & (t == k)
            if 0 < y[m].sum() < m.sum():
                aucs.append((roc_auc_score(y[m], pb[m]), roc_auc_score(y[m], pc[m])))
        a = np.mean(aucs, 0) if aucs else (np.nan, np.nan)
        print(f'{str(lab):22s} {len(sp):5d} {len(sn):5d}  {sp.gain.mean():+12.4f} {sn.gain.mean():+14.4f}  '
              f'{a[0]:.4f} -> {a[1]:.4f} ({len(aucs)} TFs)')


print('mean percentile change, all positives: %+.4f, all negatives: %+.4f' %
      (df[df.y == 1].gain.mean(), df[df.y == 0].gain.mean()))
print('graph distance TF->target in the GCN training graph:', pd.Series(d).value_counts().sort_index().to_dict())
report('dist', [0, 1, 2, 3, np.inf], ['1 hop', '2 hops', '3 hops', '>3 / unreachable'])
report('tdeg', [-1, 0, 2, 5, 100], ['target deg 0', 'target deg 1-2', 'target deg 3-5', 'target deg 6+'])
report('tfdeg', [-1, 20, 100, 1000], ['TF with <=20 targets', 'TF with 21-100', 'TF with >100'])
report('ppi', [-1, 0, 1, 100], ['0 partners target it', '1 partner', '2+ partners'])
report('coexp', [-1, -0.02, 0.02, 1], ['co-exp contrast < -0.02', '-0.02..0.02', '> +0.02'])

print('\n-- per TF: within-TF AUROC baseline -> Exp C (sorted by gain) --')
rows = []
for k in np.unique(t):
    m = t == k
    if 0 < y[m].sum() < m.sum():
        rows.append((genes[k], int(Wtr[k].sum()), int(P[k].sum()), roc_auc_score(y[m], pb[m]), roc_auc_score(y[m], pc[m])))
r = pd.DataFrame(rows, columns=['TF', 'train_targets', 'STRING_TF_partners', 'base', 'expC'])
r['delta'] = r.expC - r.base
print(r.sort_values('delta', ascending=False).round(4).to_string(index=False))
print('Spearman(delta, train_targets) = %.3f; Spearman(delta, STRING partners) = %.3f' %
      (r[['delta', 'train_targets']].corr('spearman').iloc[0, 1], r[['delta', 'STRING_TF_partners']].corr('spearman').iloc[0, 1]))
