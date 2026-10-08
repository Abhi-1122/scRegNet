"""Item 2 diagnosis: stack the BASELINE model's score with each PANDA variant (LR, grouped 5-fold CV
over TFs). Compares 'what each variant adds on top of what scRegNet already knows' without any
neural retraining, to separate information content from how well training exploits it."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, 'analysis')
from common import load, metrics, D
tr, te, genes, expr, tfs = load()
t, g, y = te.TF.values, te.Target.values, te.Label.values
b = np.load('analysis/out/scores_tf_500_hESC_baseline_rerun.npz')['test_logits']; b = b[:, 1] - b[:, 0]
tfu = np.unique(t); fold = dict(zip(tfu, np.random.RandomState(0).permutation(len(tfu)) % 5)); f = np.array([fold[k] for k in t])
print(f"{'baseline score +':20s} AUROC  AUPRC  within-TF")
for v in ['none', 'full', 'no_coexpr', 'no_ppi', 'prior_only', 'raw_corr']:
    X = b[:, None] if v == 'none' else np.column_stack([b, np.load(D + f'PANDA_variants/{v}/panda_full_matrix.npy')[t, g]])
    s = np.zeros(len(t))
    for k in range(5):
        s[f == k] = LogisticRegression(max_iter=1000).fit(X[f != k], y[f != k]).decision_function(X[f == k])
    m = metrics(y, s, t); print(f"{v:20s} {m['AUROC']:.4f} {m['AUPRC']:.4f} {m['withinTF_AUROC']:.4f}")
