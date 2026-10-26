"""Held-out genes, hESC: late fusion (z-averaged logits, matched fold and seed) vs the same-cost seed average.
Run: python3.12 analysis/w6_heldout_fusion.py"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
O = 'analysis/out/pair/'
S = [42, 43, 44, 45, 46]
z = lambda v: (v - v.mean()) / v.std()
ld = lambda p, f, s: np.load(f'{O}{p[:2]}_ho_gene{f}_{p[3:]}_s{s}.npz')


def pertf(members):
    acc = {}
    for f in range(5):
        for i, s in enumerate(S):
            ds = [ld(p, f, S[(i + k) % 5]) for p, k in members]
            tf, y = ds[0]['tf'], ds[0]['y']
            sc = sum(z(d['logit']) for d in ds)
            for t in np.unique(tf):
                if 1 <= y[tf == t].sum() < (tf == t).sum():
                    acc.setdefault(t, []).append(roc_auc_score(y[tf == t], sc[tf == t]))
    return {t: np.mean(v) for t, v in acc.items()}


rows = {'hero': [('w5:hero', 0)], 'hero x2 seeds': [('w5:hero', 0), ('w5:hero', 1)],
        'Geneformer hero x2 seeds': [('w6:herogf', 0), ('w6:herogf', 1)],
        'hero + np had': [('w5:hero', 0), ('w5:nphad', 0)],
        'Geneformer hero + np had (GF)': [('w6:herogf', 0), ('w6:nphadgf', 0)],
        'hero x2 + Geneformer hero x2 [4 models]': [('w5:hero', 0), ('w5:hero', 1), ('w6:herogf', 0), ('w6:herogf', 1)],
        'hero + GF hero + np had + np had GF [4 models]': [('w5:hero', 0), ('w6:herogf', 0), ('w5:nphad', 0), ('w6:nphadgf', 0)]}
R = {k: pertf(v) for k, v in rows.items()}
print('HELD-OUT GENES, hESC: per-TF within-TF AUROC averaged over 5 folds x 5 seeds')
for k, v in R.items():
    ref = 'hero x2 seeds' if '[4' not in k else 'hero x2 + Geneformer hero x2 [4 models]'
    d = np.array([v[t] - R[ref][t] for t in v])
    print(f"{k:48s} within-TF {np.mean(list(v.values())):.4f}   vs {ref}: {d.mean():+.4f} ({(d > 0).sum()}/{len(d)}, p={wilcoxon(d).pvalue if d.any() else 1:.2g})")
