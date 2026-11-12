"""Shared helpers for the W12 reports: w3_report functions restricted to seeds 42-44 (the W12 seed set), plus a popularity-correlation helper."""
import glob, json, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis')
import w3_report as w
from scipy.stats import spearmanr

SEEDS = (42, 43, 44)
OUT = w.OUT


def runs(c, seeds=SEEDS):
    return [r for r in w.runs(c) if r['args']['seed'] in seeds]


def npzs(c, seeds=SEEDS):
    return [np.load(f'{OUT}{c}_s{s}.npz') for s in seeds if glob.glob(f'{OUT}{c}_s{s}.npz')]


def tf_mean(c, seeds=SEEDS):
    d = [w.per_tf(z['tf'], z['y'], z['logit']) for z in npzs(c, seeds)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))} if d else {}


def row(nm, c, seeds=SEEDS):
    r = runs(c, seeds)
    if not r: return f'{nm:46s} (no runs)'
    g = lambda m: w.ms([x['refit'][m] for x in r])
    return f"{nm:46s} {len(r)}  {g('auc'):>16s} {g('aupr'):>16s} {g('within'):>16s} {g('within8'):>16s} {w.ms([x['val']['auc'] for x in r]):>16s} {np.mean([x['minutes'] for x in r]):7.2f}"


def table(title, items, ref, ref_name='hero', seeds=SEEDS):
    L = [f'\n== {title}', w.HDR] + [row(nm, c, seeds) for nm, c in items]
    L.append(f'paired per-TF tests vs {ref_name} ({ref}), seeds {list(seeds)}:')
    B = tf_mean(ref, seeds)
    L += [w.paired(tf_mean(c, seeds), B, nm) if npzs(c, seeds) else f'  {nm}: no runs' for nm, c in items if c != ref]
    R = {r['args']['seed']: r['refit'] for r in runs(ref, seeds)}
    L.append(f'same-seed differences vs {ref_name} (mean over seeds, seeds better):')
    for nm, c in items:
        if c != ref and runs(c, seeds):
            d = {m: [r['refit'][m] - R[r['args']['seed']][m] for r in runs(c, seeds) if r['args']['seed'] in R] for m in ('auc', 'aupr', 'within')}
            L.append(f"  {nm:46s} AUROC {np.mean(d['auc']):+.4f} ({sum(x > 0 for x in d['auc'])}/{len(d['auc'])})  AUPRC {np.mean(d['aupr']):+.4f}  within-TF {np.mean(d['within']):+.4f} ({sum(x > 0 for x in d['within'])}/{len(d['within'])})")
    return L


def popularity(cell='hESC', num_TF='500'):
    """training in-degree of each gene (number of TFs with a positive training label), from Train_set.csv"""
    import pandas as pd
    tr = pd.read_csv(f'data/{cell}/TFs+{num_TF}/Train_set.csv', index_col=0)
    n = pd.read_csv(f'data/{cell}/TFs+{num_TF}/BL--ExpressionData.csv', index_col=0).shape[0]
    return np.bincount(tr.Target.values[tr.Label.values == 1], minlength=n)


def pop_corr(z, pop):
    """mean over TFs of Spearman(score over all genes, gene popularity) from an npz with grid/grid_tf"""
    return float(np.mean([spearmanr(g, pop).statistic for g in z['grid']]))


def pair_keys(cell='hESC', num_TF='500'):
    """per test pair: gene popularity (training in-degree), and max cosine of the gene's expression profile to the TF's known (training) targets"""
    import pandas as pd
    P = f'data/{cell}/TFs+{num_TF}/'
    tr, te = pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)
    x = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0).values.astype(np.float32)
    xn = x - x.mean(1, keepdims=True); xn /= np.linalg.norm(xn, axis=1, keepdims=True) + 1e-9
    pop = popularity(cell, num_TF)
    tg = {t: tr.Target.values[(tr.TF.values == t) & (tr.Label.values == 1)] for t in np.unique(te.TF.values)}
    mx = np.array([(xn[tg[t][tg[t] != g]] @ xn[g]).max() if (tg[t] != g).any() else np.nan for t, g in zip(te.TF.values, te.Target.values)])
    return dict(pop=pop[te.Target.values].astype(float), maxcos=mx, y=te.Label.values, tf=te.TF.values)


def strata_delta(A, B, key, edges, seeds=SEEDS):
    """pooled test-pair AUROC of run family A minus family B inside bins of `key` (bin edges `edges`), averaged over seeds; returns [(lo, hi, n, nPos, auA, auB)]"""
    za, zb = npzs(A, seeds), npzs(B, seeds)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi) & ~np.isnan(key); y = za[0]['y'][m]
        if y.sum() < 5 or y.sum() == len(y): out.append((lo, hi, int(m.sum()), int(y.sum()), np.nan, np.nan)); continue
        out.append((lo, hi, int(m.sum()), int(y.sum()), np.mean([roc_auc_score(y, z['logit'][m]) for z in za]), np.mean([roc_auc_score(y, z['logit'][m]) for z in zb])))
    return out


def strata_lines(A, B, key, name, nb=3, seeds=SEEDS):
    q = np.nanquantile(key, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9; q = np.unique(q)
    return [f'  {name} [{lo:.2f}, {hi:.2f}) n={n} pos={p}: AUROC {a:.4f} vs {b:.4f} (diff {a - b:+.4f})' for lo, hi, n, p, a, b in strata_delta(A, B, key, q, seeds)]


def ho_runs(prot, variant, ref=False, seeds=(42, 43)):
    """held-out runs (5 folds x seeds). variant = W12 variant name (w12_ho_<prot><f>_<variant>_s<seed>) or, with ref=True, the w5 hero ('hero')"""
    if ref and variant.startswith('td01'):  # earlier hero + tf_drop runs have no prefix
        return [(f, s, f'ho_{prot}{f}_{variant}_s{s}') for f in range(5) for s in seeds]
    pre = 'w5' if ref else 'w12'
    return [(f, s, f'{pre}_ho_{prot}{f}_{variant}_s{s}') for f in range(5) for s in seeds]


def ho_report(prot, variants, ref='hero', seeds=(42, 43)):
    """table over the 10 (fold, seed) runs + paired per-TF test (per-TF AUROC averaged over fold x seed) vs the w5 hero"""
    import os
    def load(variant, is_ref=False):
        js, zs = [], []
        for f, s, n in ho_runs(prot, variant, is_ref, seeds):
            if os.path.exists(f'{OUT}{n}.json') and os.path.exists(f'{OUT}{n}.npz'): js.append(json.load(open(f'{OUT}{n}.json'))); zs.append(np.load(f'{OUT}{n}.npz'))
        return js, zs
    def per_tf_mean(zs):
        d = {}
        for z in zs:
            for k, v in w.per_tf(z['tf'], z['y'], z['logit']).items(): d.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in d.items()}
    Rj, Rz = load(ref, True)
    L = [f'held-out {prot}s: 5 folds x seeds {list(seeds)}', f"{'model':40s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within>=8':>16s}"]
    for nm, v, is_ref in [(('hero (w5)' if ref == 'hero' else ref), ref, True)] + [(v, v, False) for v in variants]:
        js, _ = load(v, is_ref)
        if js: L.append(f"{nm:40s} {len(js)}  " + ' '.join(f"{w.ms([j['refit'][m] for j in js]):>16s}" for m in ('auc', 'aupr', 'within', 'within8')))
        else: L.append(f'{nm:40s} (no runs)')
    B = per_tf_mean(Rz)
    for v in variants:
        js, zs = load(v)
        if zs: L.append(w.paired(per_tf_mean(zs), B, v + f' (n={len(zs)})'))
    return L


def ho_per_tf(prot, variant, ref=False, seeds=(42, 43)):
    """per-TF test AUROC averaged over the available (fold, seed) held-out runs"""
    import os
    d = {}
    for f, s, n in ho_runs(prot, variant, ref, seeds):
        if os.path.exists(f'{OUT}{n}.npz'):
            z = np.load(f'{OUT}{n}.npz')
            for k, v in w.per_tf(z['tf'], z['y'], z['logit']).items(): d.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in d.items()}
