"""W14: does the hero read gene-level properties of the expression profile or the co-expression pattern across cells?
Variants of the node features (--feat): full (hero), cellperm, stats, rand. Seeds 42-44; held-out genes hESC 5 folds x seeds 42,43.
Baselines: hESC w3_hero, other cells w4_<cell>_hero, held-out genes w5_ho_gene<f>_hero. Run: python3.12 analysis/w14_report.py"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis'); sys.path.insert(0, '.')
import w12_common as C
import w3_report as w
from src.pair_gnn import feat_variant

CELLS = ['hESC', 'hHEP', 'mESC', 'mHSC-E']
NAMES = ['mean', 'std', 'CV', 'detection rate', 'mean of nonzero', 'skewness', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99']
HERO = {c: ('w3_hero' if c == 'hESC' else f'w4_{c}_hero') for c in CELLS}
ITEMS = lambda c: [('full (hero)', HERO[c]), ('cellperm', f'w14_{c}_cellperm'), ('stats', f'w14_{c}_stats'), ('rand', f'w14_{c}_rand')]
L = ['W14: node-feature variants of the SAGE hero (no Geneformer). Test_set, seeds 42-44, mean ± std; within-TF p = paired per-TF Wilcoxon (per-TF AUROC averaged over seeds).',
     'kept share = (variant - rand) / (full - rand).']
SUM = {}
for c in CELLS:
    its = ITEMS(c)
    L += [f'\n== {c}', f"{'features':14s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s}   vs full: within-TF diff (TFs better, p)    vs rand: diff (p)    kept share AUROC / AUPRC / within-TF"]
    T = {k: C.tf_mean(v) for k, v in its}
    R = {k: C.runs(v) for k, v in its}
    m = lambda k, key: np.mean([r['refit'][key] for r in R[k]]) if R[k] else np.nan
    def pc(A, B):
        ks = sorted(set(A) & set(B)); d = np.array([A[k] - B[k] for k in ks])
        return d.mean(), int((d > 0).sum()), len(d), (wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
    for k, v in its:
        if not R[k]: L.append(f'{k:14s} (no runs)'); continue
        g = lambda key: w.ms([r['refit'][key] for r in R[k]])
        s = f'{k:14s} {len(R[k])}  {g("auc"):>16s} {g("aupr"):>16s} {g("within"):>16s}'
        if k != 'full (hero)' and T[k]:
            d, nb, n, p = pc(T[k], T['full (hero)']); s += f'   {d:+.4f} ({nb}/{n}, p={p:.3g})'
            if k != 'rand':
                d2, _, _, p2 = pc(T[k], T['rand']); s += f'   {d2:+.4f} (p={p2:.3g})'
            else: s += ' ' * 20
            if k != 'rand':
                ks = [(m(k, q) - m('rand', q)) / (m('full (hero)', q) - m('rand', q)) for q in ('auc', 'aupr', 'within')]
                s += '   ' + ' / '.join(f'{x:5.2f}' for x in ks); SUM[(c, k)] = ks
        L.append(s)
    if T['rand'] and T['full (hero)']:
        d, nb, n, p = pc(T['full (hero)'], T['rand']); L.append(f'  full - rand within-TF: {d:+.4f} (TFs better {nb}/{n}, p={p:.3g})')

# 2. held-out genes (hESC)
L += ['\n== Held-out genes, hESC: 5 folds x seeds 42,43 (n = 10 runs per row); the test genes were never in the training graph']
def ho(prefix):
    js, zs = [], []
    for f in range(5):
        for s in (42, 43):
            n = f'{prefix.format(f=f)}_s{s}'
            if os.path.exists(f'{C.OUT}{n}.json') and os.path.exists(f'{C.OUT}{n}.npz'): js.append(json.load(open(f'{C.OUT}{n}.json'))); zs.append(np.load(f'{C.OUT}{n}.npz'))
    d = {}
    for z in zs:
        for k, v in w.per_tf(z['tf'], z['y'], z['logit']).items(): d.setdefault(k, []).append(v)
    return js, {k: float(np.mean(v)) for k, v in d.items()}
H = {'full (hero, w5)': ho('w5_ho_gene{f}_hero'), 'cellperm': ho('w14_ho_gene{f}_cellperm'), 'stats': ho('w14_ho_gene{f}_stats'), 'rand': ho('w14_ho_gene{f}_rand')}
L.append(f"{'features':16s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s}   vs full: within-TF diff (TFs better, p)")
for k, (js, t) in H.items():
    if not js: L.append(f'{k:16s} (no runs)'); continue
    s = f'{k:16s} {len(js)}  ' + ' '.join(f"{w.ms([j['refit'][q] for j in js]):>16s}" for q in ('auc', 'aupr', 'within'))
    if not k.startswith('full'):
        ks = sorted(set(t) & set(H['full (hero, w5)'][1])); d = np.array([t[q] - H['full (hero, w5)'][1][q] for q in ks])
        s += f'   {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)}, p={wilcoxon(d).pvalue:.3g})'
    L.append(s)
f_, r_ = (np.mean([j['refit']['within'] for j in H[k][0]]) for k in ('full (hero, w5)', 'rand'))
L.append('kept share of the within-TF gain over rand (variant - rand) / (full - rand): ' + ', '.join(f"{k} {(np.mean([j['refit']['within'] for j in H[k][0]]) - r_) / (f_ - r_):.2f}" for k in ('cellperm', 'stats')))
# popularity cannot help here: AUROC of the test pairs ranked by training in-degree is 0.5 by construction (new genes have no training edges)
L.append('(a new gene has training in-degree 0, so popularity gives 0.5 here; anything above 0.5 comes from the features and the structure of the known genes)')

# 3. single-statistic baselines
L += ['\n== Single raw statistic as a score: within-TF AUROC of ranking each TF\'s test genes by the statistic alone (signed; below 0.5 = the opposite direction ranks better), and Spearman of the statistic with gene training in-degree (popularity)']
SS = {}
for c in CELLS:
    P = f'data/{c}/TFs+500/'
    te = pd.read_csv(P + 'Test_set.csv', index_col=0)
    raw = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0).values.astype(float)
    S = feat_variant('stats', np.zeros_like(raw, dtype=np.float32), raw) # standardised columns: ranks are the same as raw
    pop = C.popularity(c)
    y, tf, g = te.Label.values, te.TF.values, te.Target.values
    SS[c] = {}
    for j, nm in enumerate(NAMES):
        sc = S[g, j]
        SS[c][nm] = (w_ := np.mean(list(w.per_tf(tf, y, sc).values())), spearmanr(S[:, j], pop).statistic)
    SS[c]['popularity'] = (np.mean(list(w.per_tf(tf, y, pop[g] + 1e-3 * np.random.RandomState(0).rand(len(g))).values())), 1.0)
L.append(f"{'statistic':16s} " + ' '.join(f'{c + " within":>13s} {"rho(pop)":>8s}' for c in CELLS))
for nm in NAMES + ['popularity']:
    L.append(f'{nm:16s} ' + ' '.join(f'{SS[c][nm][0]:13.4f} {SS[c][nm][1]:8.3f}' for c in CELLS))
best = {c: max(NAMES, key=lambda n: max(SS[c][n][0], 1 - SS[c][n][0])) for c in CELLS}
for c in CELLS:
    a = SS[c][best[c]][0]; L.append(f'  best single statistic {c}: {best[c]} (within-TF {a:.4f}{"" if a >= 0.5 else f", reversed {1 - a:.4f}"}); popularity {SS[c]["popularity"][0]:.4f}')

# 4. score / popularity Spearman
L += ['\n== Spearman of test scores with gene popularity (training in-degree), within TF (test genes of one TF), mean over TFs; mean over seeds']
L.append(f"{'features':14s} " + ' '.join(f'{c:>9s}' for c in CELLS))
def pcor(name, cell, pop, te):
    out = []
    for z in C.npzs(name):
        r = [spearmanr(z['logit'][z['tf'] == t], pop[z['tgt'][z['tf'] == t]]).statistic for t in np.unique(z['tf']) if (z['tf'] == t).sum() > 2]
        out.append(np.nanmean(r))
    return np.mean(out) if out else np.nan
pops = {c: C.popularity(c) for c in CELLS}
for k in ['full (hero)', 'cellperm', 'stats', 'rand']:
    L.append(f'{k:14s} ' + ' '.join(f'{pcor(dict(ITEMS(c))[k], c, pops[c], None):9.3f}' for c in CELLS))
L.append(f"{'popularity':14s} " + ' '.join(f'{1.0:9.3f}' for c in CELLS))

# verdict helper
L += ['\n== Kept share of the full model\'s within-TF gain over rand (rule: >= 0.80 = reads gene-level properties; near 0 = needs the co-expression pattern)']
for c in CELLS:
    L.append(f'  {c}: ' + ', '.join(f'{k} {SUM[(c, k)][2]:.2f}' for k in ('cellperm', 'stats') if (c, k) in SUM))
open('analysis/out/w14_feat.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
