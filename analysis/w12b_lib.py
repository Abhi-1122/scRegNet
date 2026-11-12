"""Shared helpers of the W12 NBFNet / SEAL reports: seed-filtered tables (seeds 42-44), paired per-TF tests, popularity and path-count diagnostics."""
import glob, json, re, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis')
from w3_report import OUT, ms, per_tf, paired, HDR

SEEDS = (42, 43, 44)
HERO = 'w3_hero'


def runs(c, seeds=SEEDS):
    return [json.load(open(f'{OUT}{c}_s{s}.json')) for s in seeds if glob.glob(f'{OUT}{c}_s{s}.json')]


def npzs(c, seeds=SEEDS):
    return [np.load(f'{OUT}{c}_s{s}.npz') for s in seeds if glob.glob(f'{OUT}{c}_s{s}.npz')]


def tf_mean(c, seeds=SEEDS):
    d = [per_tf(z['tf'], z['y'], z['logit']) for z in npzs(c, seeds)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))} if d else {}


def row(nm, c):
    r = runs(c)
    if not r: return f'{nm:46s} (no runs)'
    g = lambda m: ms([x['refit'][m] for x in r])
    return f"{nm:46s} {len(r)}  {g('auc'):>16s} {g('aupr'):>16s} {g('within'):>16s} {g('within8'):>16s} {ms([x['val']['auc'] for x in r]):>16s} {np.mean([x['minutes'] for x in r]):7.2f}"


def table(title, items, ref=HERO):
    L = [f'\n== {title}', HDR]
    L += [row(nm, c) for nm, c in items]
    B = tf_mean(ref)
    L.append(f'paired per-TF tests vs hero (per-TF test AUROC averaged over seeds {SEEDS}, Wilcoxon over TFs):')
    L += [paired(tf_mean(c), B, nm) if npzs(c) else f'  {nm}: no runs' for nm, c in items if c != ref]
    R = {r['args']['seed']: r['refit'] for r in runs(ref)}
    L.append('same-seed differences vs hero (mean over seeds, seeds better):')
    for nm, c in items:
        if c != ref and runs(c):
            d = {m: [r['refit'][m] - R[r['args']['seed']][m] for r in runs(c) if r['args']['seed'] in R] for m in ('auc', 'aupr')}
            L.append(f"  {nm:46s} AUROC {np.mean(d['auc']):+.4f} ({sum(x > 0 for x in d['auc'])}/{len(d['auc'])})  AUPRC {np.mean(d['aupr']):+.4f} ({sum(x > 0 for x in d['aupr'])}/{len(d['aupr'])})")
    return L


def train_graph(cell='hESC'):
    """directed training-positive matrix A[tf, gene] (all training rows, as in the refit) and gene in-degree"""
    tr = pd.read_csv(f'data/{cell}/TFs+500/Train_set.csv', index_col=0)
    n = pd.read_csv(f'data/{cell}/TFs+500/Target.csv').shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    p = tr[tr.Label == 1]
    A[p.TF.values, p.Target.values] = 1
    return A, A.sum(0)


def popularity(c, seeds=SEEDS):
    """mean over TFs and seeds of the within-TF Spearman correlation between the score and the gene's training in-degree"""
    _, deg = train_graph()
    out = []
    for z in npzs(c, seeds):
        v = [spearmanr(z['logit'][z['tf'] == t], deg[z['tgt'][z['tf'] == t]])[0] for t in np.unique(z['tf'])]
        out.append(np.nanmean(v))
    return float(np.mean(out)) if out else float('nan')


def subset_within(c, mask_fn, seeds=SEEDS):
    """within-TF AUROC (mean over seeds) restricted to test pairs where mask_fn(tf, tgt) is true"""
    res = []
    for z in npzs(c, seeds):
        m = mask_fn(z['tf'], z['tgt'])
        v = [roc_auc_score(z['y'][m & (z['tf'] == t)], z['logit'][m & (z['tf'] == t)]) for t in np.unique(z['tf'])
             if 0 < z['y'][m & (z['tf'] == t)].sum() < (m & (z['tf'] == t)).sum()]
        res.append(np.mean(v))
    return float(np.mean(res)) if res else float('nan')


def path_diag(items, ref=HERO):
    """within-TF AUROC on test pairs with / without 3-paths t -> g' <- t' -> g in the training graph, plus the path count as a score"""
    A, deg = train_graph()
    P3 = A @ A.T @ A
    P2 = A @ A  # t -> x -> g (x is itself a TF)
    L = [f'\n-- path-count diagnostic (test pairs split by training-graph 3-paths t->g\'<-t\'->g; popularity = gene in-degree)',
         f"{'model':46s} {'within all':>10s} {'3-paths=0':>10s} {'3-paths>0':>10s} {'2-paths>0':>10s} {'pop. corr':>10s}"]
    z0 = npzs(ref)[0]
    frac = lambda f: np.mean(f(z0['tf'], z0['tgt']))
    L.append(f"  share of test pairs: 3-paths>0 {frac(lambda t, g: P3[t, g] > 0):.2f}, 2-paths>0 {frac(lambda t, g: P2[t, g] > 0):.2f}")
    for nm, c in [('3-path count as score', None)] + list(items):
        if c is None:
            f = lambda m: np.mean([roc_auc_score(z0['y'][m & (z0['tf'] == t)], P3[t, z0['tgt']][m & (z0['tf'] == t)]) for t in np.unique(z0['tf']) if 0 < z0['y'][m & (z0['tf'] == t)].sum() < (m & (z0['tf'] == t)).sum()])
            allm = np.ones(len(z0['y']), bool)
            L.append(f"  {nm:44s} {f(allm):10.4f} {'':>10s} {'':>10s}")
            continue
        if not npzs(c): continue
        a = lambda fn: subset_within(c, fn)
        L.append(f"  {nm:44s} {a(lambda t, g: np.ones(len(t), bool)):10.4f} {a(lambda t, g: P3[t, g] == 0):10.4f} {a(lambda t, g: P3[t, g] > 0):10.4f} {a(lambda t, g: P2[t, g] > 0):10.4f} {popularity(c):10.3f}")
    return L


def ho_table(proto, pairs, hero='w5_ho_{p}{f}_hero'):
    """held-out gene / TF protocols, 5 folds x seeds 42, 43: [(label, run-prefix with {p}{f})] vs the hero; per-fold mean +- std over seeds, pooled paired per-(fold, TF) test"""
    sd = (42, 43)
    L = [f'\n== held-out {proto}s (5 folds x seeds {sd}); mean over folds of the seed-mean', f"{'model':40s} {'AUROC':>8s} {'AUPRC':>8s} {'within':>8s} {'within>=8':>9s}"]
    keyed = lambda tpl: {(f, k): v for f in range(5) for k, v in tf_mean(tpl.format(p=proto, f=f), sd).items()}
    cols = lambda tpl: [np.mean([[r['refit'][m] for r in runs(tpl.format(p=proto, f=f), sd)] for f in range(5)], 1).mean() if all(runs(tpl.format(p=proto, f=f), sd) for f in range(5)) else float('nan') for m in ('auc', 'aupr', 'within', 'within8')]
    H = keyed(hero)
    for nm, tpl in [('hero', hero)] + pairs:
        c = cols(tpl)
        L.append(f'{nm:40s} {c[0]:8.4f} {c[1]:8.4f} {c[2]:8.4f} {c[3]:9.4f}')
    for nm, tpl in pairs:
        K = keyed(tpl)
        if K: L.append(paired(K, H, nm + ' vs hero (per fold-TF)'))
    return L
