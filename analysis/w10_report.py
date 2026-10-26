"""Item 8 round 2: hero trained on one cell state (columns of the expression matrix) vs matched controls; 5 seeds (42-46), epoch chosen on validation AUROC, refit.
early = 00h+12h+24h (260 cells), late = 260 of the 326 cells at 72h+96h (new draw per seed), rand = 260 of all 758 cells (new draw per seed), all = existing w3_hero (758 cells).
Run: python3.12 analysis/w10_report.py"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, 'analysis')
import w3_report as w

ITEMS = [('early (00h+12h+24h, 260 cells)', 'w10_early'), ('late (72h+96h, 260 of 326, per-seed draw)', 'w10_late'),
         ('rand (260 of 758, per-seed draw)', 'w10_rand'), ('all (758 cells, w3_hero)', 'w3_hero')]
L = ['Item 8 round 2: SAGE hero (no Geneformer), expression columns restricted to one cell state. Test_set, mean ± std over 5 seeds.', w.HDR] + [w.row(a, b) for a, b in ITEMS]
T = {c: w.tf_mean(c) for _, c in ITEMS}
L += ['\npaired per-TF tests (per-TF test AUROC averaged over seeds; difference, bootstrap CI, TFs better, Wilcoxon p):']
for a, b in [('early', 'rand'), ('late', 'rand'), ('early', 'late'), ('all', 'rand')]:
    f = lambda k: 'w3_hero' if k == 'all' else f'w10_{k}'
    L.append(w.paired(T[f(a)], T[f(b)], f'{a} - {b}'))
L.append('same-seed within-TF differences (seeds better of 5): ' + '; '.join(
    f"{a} - {b}: {np.mean([x['refit']['within'] - y['refit']['within'] for x, y in zip(w.runs(f'w10_{a}'), w.runs(f'w10_{b}'))]):+.4f} "
    f"({sum(x['refit']['within'] > y['refit']['within'] for x, y in zip(w.runs(f'w10_{a}'), w.runs(f'w10_{b}')))}/5)" for a, b in [('early', 'rand'), ('late', 'rand'), ('early', 'late')]))

# per-TF biology: does the early-minus-late AUROC follow the TF's early-minus-late expression?
P = 'data/hESC/TFs+500/'
x = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0)
genes = x.index.tolist()
pre = np.array(['_'.join(c.split('_')[:2]) for c in x.columns])
ec, lc = np.isin(pre, ['H9_00hb4s', 'H9_12h', 'H9_24h']), np.isin(pre, ['H9_72h', 'H9_96h'])
E, Lt = T['w10_early'], T['w10_late']
ks = sorted(set(E) & set(Lt))
dA = np.array([E[k] - Lt[k] for k in ks])
dE = np.array([x.values[k, ec].mean() - x.values[k, lc].mean() for k in ks])
rho = spearmanr(dE, dA)
L += [f'\nper-TF biology check ({len(ks)} TFs): AUROC(early model) - AUROC(late model), per-TF test AUROC averaged over seeds, against the TF\'s mean expression in early cells minus late cells (file values, 260 early / 326 late cells)',
      f'Spearman across TFs: rho = {rho.statistic:+.3f}, p = {rho.pvalue:.3g}; TFs expressed higher early (dE > 0): {int((dE > 0).sum())}, of which early model better {int(((dE > 0) & (dA > 0)).sum())}; '
      f'TFs expressed higher late: {int((dE < 0).sum())}, of which late model better {int(((dE < 0) & (dA < 0)).sum())}',
      f"{'TF':8s} {'AUROC early':>11s} {'AUROC late':>10s} {'early-late':>10s} {'expr early-late':>15s}"]
for g in ['POU5F1', 'NANOG', 'SOX2', 'SOX17', 'GATA4', 'GATA6', 'EOMES']:
    k = genes.index(g)
    if k in E and k in Lt:
        L.append(f'{g:8s} {E[k]:11.3f} {Lt[k]:10.3f} {E[k] - Lt[k]:+10.3f} {x.values[k, ec].mean() - x.values[k, lc].mean():+15.2f}')
open('analysis/out/w10_states_r2.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
