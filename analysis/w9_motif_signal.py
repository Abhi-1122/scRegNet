"""Item 7 round 1: does a motif-only TF -> gene prior carry signal? CPU only, no GNN. hESC (TFs+500).
a) benchmark split (Train_set / Test_set): motif_max and motif_hits alone on test pairs of TFs that have a motif.
b) held-out TFs (the 5 folds of src.pair_gnn --protocol tf): popularity (log training in-degree of the gene), motif_max, motif_hits, rank averages and a
   logistic regression on [popularity, motif] fitted on the fold's training rows; paired per-TF Wilcoxon against popularity, pooled over folds.
Only TFs with a motif and genes with a promoter are scored; popularity is scored on the same rows. Run: python3.12 -P analysis/w9_motif_signal.py"""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '.')
from src.pair_gnn import heldout_split

P = 'data/hESC/TFs+500/'
tr0, te0 = pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)
L = pd.concat([tr0, te0], ignore_index=True)
n = 910
MM, MH = np.load(P + 'motif/motif_max.npy'), np.load(P + 'motif/motif_hits.npy')
out = []
pr = lambda s='': (print(s), out.append(s))


def tf_auc(y, s, t):  # per-TF AUROC for TFs with both classes
    return {int(u): roc_auc_score(y[t == u], s[t == u]) for u in np.unique(t) if 0 < y[t == u].sum() < (t == u).sum()}


def indeg(tr):
    A = np.zeros((n, n)); A[tr.TF[tr.Label == 1], tr.Target[tr.Label == 1]] = 1
    return A


# ---- a) benchmark split
ok = lambda d: np.isfinite(MM[d.TF.values, d.Target.values]) & np.isfinite(MH[d.TF.values, d.Target.values])
for nm, d in [('train', tr0), ('test', te0)]:
    pr(f'{nm} pairs: {len(d)}, scored (TF has a motif and gene has a promoter): {int(ok(d).sum())}, TFs scored: {d.TF[ok(d)].nunique()} of {d.TF.nunique()}')
A = indeg(tr0)
te = te0[ok(te0)]
t, g, y = te.TF.values, te.Target.values, te.Label.values
pr('\n== a) benchmark split, test pairs of covered TFs only (popularity from Train_set positives, same rows, for reference)')
pr(f"{'score':16s} {'AUROC':>7s} {'AUPRC':>7s} {'within-TF':>10s} (TFs {len(tf_auc(y, y * 0.0 + MM[t, g], t))}, positives {int(y.sum())}/{len(y)}, base AUPRC {y.mean():.3f})")
for k, s in [('motif_max', MM[t, g]), ('motif_hits', MH[t, g]), ('popularity', np.log1p(A.sum(0))[g])]:
    pr(f'{k:16s} {roc_auc_score(y, s):7.4f} {average_precision_score(y, s):7.4f} {np.mean(list(tf_auc(y, s, t).values())):10.4f}')

# ---- b) held-out TFs
r = lambda v: rankdata(v) / len(v)
NAMES = ['popularity', 'motif_max', 'motif_hits', 'rank avg pop + max', 'rank avg pop + hits', 'LR [pop, max]', 'LR [pop, hits]']
tfs = {k: {} for k in NAMES}                  # per-TF AUROC pooled over folds
fold_res = {k: [] for k in NAMES}             # (AUROC, within-TF) per fold
pop_all = []                                  # popularity on ALL rows of the fold, for comparison with w5_heldout_tf_transfer.txt
for fold in range(5):
    tr, tst = heldout_split(L, SimpleNamespace(protocol='tf', fold=fold), n)
    A = indeg(tr); deg = A.sum(0)
    t, g, y = tst.TF.values, tst.Target.values, tst.Label.values
    pop_all.append(np.mean(list(tf_auc(y, np.log1p(deg)[g], t).values())))
    m = ok(tst); t, g, y = t[m], g[m], y[m]
    k = ok(tr); ttr, gtr, ytr = tr.TF.values[k], tr.Target.values[k], tr.Label.values[k]
    # training rows: leave the row's own TF out of the gene's in-degree, as for a held-out TF
    ptr = np.log1p(deg[gtr] - A[ttr, gtr])
    pop = np.log1p(deg)[g]
    sc = {'popularity': pop, 'motif_max': MM[t, g], 'motif_hits': MH[t, g]}
    sc['rank avg pop + max'] = r(pop) + r(sc['motif_max'])
    sc['rank avg pop + hits'] = r(pop) + r(sc['motif_hits'])
    for nm, f_tr, f_te in [('LR [pop, max]', MM[ttr, gtr], MM[t, g]), ('LR [pop, hits]', np.log1p(MH[ttr, gtr]), np.log1p(MH[t, g]))]:
        Xtr = np.c_[ptr, f_tr]; ss = StandardScaler().fit(Xtr)
        sc[nm] = LogisticRegression(C=1.0, max_iter=1000).fit(ss.transform(Xtr), ytr).decision_function(ss.transform(np.c_[pop, f_te]))
    for nm, s in sc.items():
        d = tf_auc(y, s, t)
        tfs[nm].update(d)
        fold_res[nm].append((roc_auc_score(y, s), np.mean(list(d.values())) if d else np.nan))
    pr_n = len(tf_auc(y, pop, t))
    pr(f'fold {fold}: held-out TFs {tst.TF.nunique()}, with a motif and both classes {pr_n}, rows scored {len(y)}/{len(tst)}')
ks = sorted(set.intersection(*[set(v) for v in tfs.values()]))
pr(f'\n== b) held-out TFs, 5 folds; TFs with a motif only ({len(ks)} TFs pooled); popularity on ALL held-out rows (w5 reference, fold mean): {np.mean(pop_all):.4f}')
pr(f"{'score':22s} {'AUROC (fold mean ± std)':>26s} {'within-TF, pooled TFs':>22s} {'within-TF, fold mean ± std':>28s}   diff vs popularity (95% bootstrap CI), TFs better, Wilcoxon p")
B = np.array([tfs['popularity'][q] for q in ks])
res = {}
for nm in NAMES:
    a = np.array([x[0] for x in fold_res[nm]]); w = np.array([x[1] for x in fold_res[nm]])
    V = np.array([tfs[nm][q] for q in ks]); d = V - B
    line = f'{nm:22s} {a.mean():.4f} ± {a.std():.4f}'.ljust(49) + f'{V.mean():22.4f}' + f'{np.nanmean(w):15.4f} ± {np.nanstd(w):.4f}'
    if nm != 'popularity':
        boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        res[nm] = (V.mean(), p)
        line += f'   {d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  {int((d > 0).sum())}/{len(d)}  p={p:.3g}'
    pr(line)
best = max(res, key=lambda k: res[k][0])
pr(f'\nbest variant by pooled within-TF (picked on these test rows, so optimistic): {best}, {res[best][0]:.4f} vs popularity {B.mean():.4f}, Wilcoxon p={res[best][1]:.3g}')
G = pd.read_csv(P + 'Target.csv', index_col=0).Gene.tolist()
for nm in ('motif_max', 'motif_hits'):
    V = np.array([tfs[nm][q] for q in ks])
    pr(f'{nm} alone vs chance (per-TF AUROC - 0.5): mean {V.mean() - 0.5:+.4f}, TFs above 0.5: {int((V > 0.5).sum())}/{len(V)}, Wilcoxon p={wilcoxon(V - 0.5).pvalue:.3g}')
pr('per-TF AUROC, popularity vs motif_max vs motif_hits: ' + '; '.join(f'{G[ks[i]]}: {B[i]:.2f}/{tfs["motif_max"][ks[i]]:.2f}/{tfs["motif_hits"][ks[i]]:.2f}' for i in range(len(ks))))
open('analysis/out/w9_motif_r1.txt', 'w').write('\n'.join(out) + '\n')
