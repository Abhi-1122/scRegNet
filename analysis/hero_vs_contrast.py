"""Does the hero's within-TF ranking agree with the co-expression contrast / PANDA number, beyond gene popularity?
Per TF over its test candidates: Spearman(logit, number) and partial Spearman controlling for popularity
(log1p of # training positives of the target). Output: analysis/out/hero_vs_contrast.txt (also printed)."""
import glob
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

P, OUT = 'data/hESC/TFs+500/', 'analysis/out/pair/'
tr, te = pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)
S = {k: np.load(P + f'scalars/{k}_v0.0_s0_ev.npy') for k in ('contrast', 'panda')}
n = S['contrast'].shape[0]
pop = np.log1p(np.bincount(tr[tr.Label == 1].Target.values, minlength=n))
t, g = te.TF.values, te.Target.values
num = {k: M[t, g] for k, M in S.items()}
popv = pop[g]

def partial(a, b, c):  # Spearman(a, b | c): rank, regress c out of both, correlate residuals
    a, b, c = rankdata(a), rankdata(b), rankdata(c)
    res = lambda x: x - np.polyval(np.polyfit(c, x, 1), c) if c.std() > 0 else x - x.mean()
    ra, rb = res(a), res(b)
    return np.corrcoef(ra, rb)[0, 1] if ra.std() > 1e-12 and rb.std() > 1e-12 else np.nan

tfs = [u for u in np.unique(t) if (t == u).sum() >= 8 and 0 < te.Label.values[t == u].sum() < (t == u).sum()]
ms = lambda v: f'{np.mean(v):.3f} ± {np.std(v):.3f}'
lines = [f'hESC test; {len(tfs)} TFs with >=8 test rows and both classes; per-TF correlation over the TF\'s test candidates, averaged over TFs, mean ± std over 5 seeds',
         'popularity = log1p(# training positives of the target gene)']
# sanity: number alone
for k in num:
    w = [roc_auc_score(te.Label.values[t == u], num[k][t == u]) for u in tfs]
    lines.append(f'sanity: within-TF AUROC of the {k} number alone: {np.mean(w):.4f}')
for k in num:
    r = [spearmanr(num[k][t == u], popv[t == u])[0] for u in tfs]
    lines.append(f'context: within-TF Spearman({k}, popularity) = {np.nanmean(r):+.3f}')

def stats(z):
    assert np.array_equal(z['tf'], t) and np.array_equal(z['tgt'], g) and np.array_equal(z['y'], te.Label.values)
    o = {}
    for k in num:
        o[k + '_rho'] = [spearmanr(z['logit'][t == u], num[k][t == u])[0] for u in tfs]
        o[k + '_part'] = [partial(z['logit'][t == u], num[k][t == u], popv[t == u]) for u in tfs]
    o['pop_rho'] = [spearmanr(z['logit'][t == u], popv[t == u])[0] for u in tfs]
    return o

cfgs = ['d01_C_default', 'f04_C_tuned', 'd02_B_default', 'f07_C_gcn', 'f08_C_add', 'f01_A_tuned', 'f02_A_paper']
cols = ['contrast_rho', 'contrast_part', 'panda_rho', 'panda_part', 'pop_rho']
lines += ['', f"{'config':16s} seeds " + ''.join(f'{c:>18s}' for c in cols)]
tfmean = {}
for c in cfgs:
    zs = [np.load(f) for f in sorted(glob.glob(f'{OUT}{c}_s4?.npz'))]
    R = [stats(z) for z in zs]
    tfmean[c] = {k: np.nanmean([r[k] for r in R], 0) for k in cols}  # per-TF, seed-averaged
    lines.append(f'{c:16s} {len(zs):5d} ' + ''.join(f'{ms([np.nanmean(r[k]) for r in R]):>18s}' for k in cols))

def paired(a, b, key):
    A, B = tfmean[a][key], tfmean[b][key]
    ok = ~(np.isnan(A) | np.isnan(B)); d = (A - B)[ok]
    boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
    return (f'  {a} vs {b} [{key}]: diff {d.mean():+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  '
            f'TFs hero higher {int((d > 0).sum())}/{len(d)}  Wilcoxon p={wilcoxon(d).pvalue:.3g}')

lines.append('\npaired tests across TFs (per-TF values averaged over seeds):')
for b in ('f08_C_add', 'f02_A_paper'):
    for key in ('contrast_part', 'panda_part', 'contrast_rho'):
        lines.append(paired('d01_C_default', b, key))
txt = '\n'.join(lines)
open('analysis/out/hero_vs_contrast.txt', 'w').write(txt)
print(txt)
