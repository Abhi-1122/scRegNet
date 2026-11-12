"""Idea 0 (13 Nov): is the signal 'TF activity read from known targets' rather than the TF's own mRNA?
Scores every test pair (t, g) by a correlation across cells, no training:
  tf_mrna   corr(x_t, x_g)                                   the TF's own expression
  centroid  corr(mean of x over t's training targets, x_g)   activity as the known-target average
  tfa       corr(A_t, x_g), A = least-squares TF activity from X ~ P A (P = training prior, genes x TFs)
plus |corr| versions (repressed targets anti-correlate). Random split, test rows, prior = all training positives.
Run: python3.12 analysis/w13_activity.py"""
import sys
from types import SimpleNamespace
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, '.')
from src.pair_gnn import prepare, within_tf
from sklearn.metrics import roc_auc_score

def corr_rows(A, B):  # row-wise Pearson of A (k x c) with every row of B (n x c) -> k x n
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-8)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-8)
    return A @ B.T / A.shape[1]

def per_tf(y, s, tf):
    return {t: roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if 0 < y[tf == t].sum() < (tf == t).sum()}

L = ['Idea 0: correlation across cells between the candidate gene and (a) the TF mRNA, (b) the mean of the TF\'s training targets, (c) least-squares TF activity. Test rows, random split, within-TF AUROC; no training.']
for cell in ['hESC', 'hHEP', 'mESC', 'mHSC-E']:
    a = SimpleNamespace(cell_type=cell, num_TF='500', data_folder='data', scfm_folder='scFM', protocol='random', fold=0, scfm=0, val_frac=0.0, split_seed=0)
    D = prepare(a, 'cpu')
    tr, te, X = D['tr'], D['te'], D['x']
    tfs = np.unique(tr.TF.values)
    pos = tr[tr.Label == 1]
    P = np.zeros((X.shape[0], len(tfs)), np.float32)
    ti = {t: i for i, t in enumerate(tfs)}
    P[pos.Target.values, [ti[t] for t in pos.TF.values]] = 1
    cen = (P.T @ X) / np.maximum(P.sum(0)[:, None], 1)          # TFs x cells, known-target average
    A = np.linalg.lstsq(P, X, rcond=None)[0]                     # TFs x cells, X ~ P A
    S = {'tf_mrna': corr_rows(X[tfs], X), 'centroid': corr_rows(cen, X), 'tfa': corr_rows(A, X)}
    deg = P.sum(1)
    y, tf, g = te.Label.values, te.TF.values, te.Target.values
    r = np.array([ti[t] for t in tf])
    sc = {k: v[r, g] for k, v in S.items()}
    sc.update({f'|{k}|': np.abs(v) for k, v in list(sc.items())})
    sc['popularity'] = deg[g] + 1e-3 * np.random.RandomState(0).rand(len(g))
    L.append(f'\n== {cell} ({len(tfs)} TFs, {len(te)} test pairs)')
    L.append(f"{'score':14s} {'AUROC':>7s} {'within-TF':>9s}")
    for k, s in sc.items():
        L.append(f'{k:14s} {roc_auc_score(y, s):7.4f} {within_tf(y, s, tf):9.4f}')
    pt = {k: per_tf(y, s, tf) for k, s in sc.items()}
    for b_, a_ in [('centroid', 'tf_mrna'), ('|centroid|', '|tf_mrna|'), ('tfa', 'centroid'), ('|tfa|', '|centroid|')]:
        ks = sorted(set(pt[a_]) & set(pt[b_])); d = np.array([pt[b_][k] - pt[a_][k] for k in ks])
        L.append(f'  {b_} - {a_}: {d.mean():+.4f}, TFs better {int((d > 0).sum())}/{len(d)}, Wilcoxon p = {wilcoxon(d).pvalue:.2g}')
    # does activity add to popularity? rank-sum of the two, within TF
    from scipy.stats import rankdata
    for k in ['centroid', '|centroid|', 'tfa']:
        comb = np.zeros(len(y))
        for t in np.unique(tf):
            m = tf == t; comb[m] = rankdata(sc['popularity'][m]) + rankdata(sc[k][m])
        L.append(f'  popularity + {k} (within-TF rank sum): within-TF {within_tf(y, comb, tf):.4f}')
open('analysis/out/w13_activity.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
