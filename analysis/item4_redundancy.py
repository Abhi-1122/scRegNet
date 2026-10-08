"""Item 4: how much of PANDA's signal does the baseline already have, and what kind of signal is new?"""
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis')
from common import load, metrics, D

tr, te, genes, expr, tfs = load()
t, g, y = te.TF.values, te.Target.values, te.Label.values
z = lambda r: np.load(f'analysis/out/scores_tf_500_hESC_{r}.npz')
base, expc = z('baseline_rerun'), z('panda_edge')
s_base = base['test_logits'][:, 1] - base['test_logits'][:, 0]
s_c = expc['test_logits'][:, 1] - expc['test_logits'][:, 0]
panda = np.load(D + 'PANDA/panda_full_matrix.npy')[t, g]

# (a) Exp C's score = node terms + w * PANDA. How much of its within-TF variation is the PANDA term?
Wc = expc['W']; w_panda = Wc[1, -1] - Wc[0, -1]
p_term = w_panda * panda
share = [np.var(p_term[t == k]) / np.var(s_c[t == k]) for k in np.unique(t)]
print(f'(a) Exp C decoder weight on PANDA (logit units per PANDA std): {w_panda:.3f}')
print(f'    PANDA term / total within-TF score variance: median {np.median(share):.3f} '
      f'(IQR {np.percentile(share, 25):.3f}-{np.percentile(share, 75):.3f})')

# (b) does PANDA still predict the label once the baseline score is known? (within-TF ranks)
def within_rank(x):
    r = np.empty_like(x, dtype=float)
    for k in np.unique(t):
        m = t == k; r[m] = (np.argsort(np.argsort(x[m])) + 0.5) / m.sum()
    return r
rb, rp, ry = within_rank(s_base), within_rank(panda), y.astype(float)
res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
print(f'(b) within-TF rank corr: label~PANDA {np.corrcoef(ry, rp)[0,1]:.3f}, label~baseline {np.corrcoef(ry, rb)[0,1]:.3f}, '
      f'PANDA~baseline {np.corrcoef(rp, rb)[0,1]:.3f}; partial corr(label, PANDA | baseline) = '
      f'{np.corrcoef(res(ry, rb), res(rp, rb))[0,1]:.3f}')

# (c) post-hoc stacking: baseline score + PANDA in an LR, cross-validated over TFs (grouped 5-fold)
tfu = np.unique(t); rng = np.random.RandomState(0); fold = dict(zip(tfu, rng.permutation(len(tfu)) % 5))
f = np.array([fold[k] for k in t]); stack = np.zeros(len(t))
X = np.column_stack([s_base, panda])
for k in range(5):
    stack[f == k] = LogisticRegression(max_iter=1000).fit(X[f != k], y[f != k]).decision_function(X[f == k])
print('(c) metrics (AUROC, AUPRC, within-TF AUROC):')
for name, s in [('baseline', s_base), ('baseline + PANDA, post-hoc stacked LR', stack), ('Exp C (PANDA as input feature)', s_c)]:
    m = metrics(y, s, t); print(f'    {name:40s} {m["AUROC"]:.4f} {m["AUPRC"]:.4f} {m["withinTF_AUROC"]:.4f}')

# (d) ceiling for ANY model that ranks genes the same way for every TF:
#     use the TEST labels themselves to build the best single gene ordering (cheating, upper bound only)
gene_rate = np.bincount(g, weights=y, minlength=len(genes)) / np.maximum(np.bincount(g, minlength=len(genes)), 1)
m = metrics(y, gene_rate[g] + 1e-6 * np.random.RandomState(1).rand(len(g)), t)
print(f'(d) oracle TF-agnostic gene ordering (built from test labels): within-TF AUROC ceiling = {m["withinTF_AUROC"]:.4f}')

# (e) PANDA itself: how much of its within-TF variation is a TF-agnostic gene effect?
Pm = np.load(D + 'PANDA/panda_full_matrix.npy')[np.unique(t)]
Pc = Pm - Pm.mean(1, keepdims=True)
gene_effect = Pc.mean(0, keepdims=True)
print(f'(e) share of PANDA within-TF variance that is one TF-agnostic gene effect: '
      f'{1 - ((Pc - gene_effect) ** 2).sum() / (Pc ** 2).sum():.3f} (the rest is TF-specific)')
