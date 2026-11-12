"""W12 idea 6 (hub control): screen table (seed 42), 3-seed confirmations, popularity correlation, per-TF gain vs hub share of the TF's targets,
held-out stages when present. Run: python3.12 analysis/w12_6_report.py"""
import glob, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
import w12_common as C
import w3_report as w
from scipy.stats import spearmanr

REF = 'w3_hero'
fams = sorted({re.sub(r'_s\d+\.json$', '', f.split('/')[-1]) for f in glob.glob(f'{C.OUT}w12_hub*_s*.json') if '_ho_' not in f})
nm = lambda c: c.replace('w12_', '')
L = ['W12 idea 6: hub control. hESC, random split, no Geneformer, hero settings, epoch chosen on validation AUROC, refit.']
L += C.table('Screen, seed 42 only (n=1 per row)', [('hero (w3_hero)', REF)] + [(nm(c), c) for c in fams], REF, seeds=(42,))
conf = [c for c in fams if len(C.runs(c)) >= 3]
L += C.table('Confirmation, seeds 42-44', [('hero (w3_hero)', REF)] + [(nm(c), c) for c in conf], REF)
pop = C.popularity()
L.append('\nScore/popularity: Spearman over all genes between the score and the gene\'s training in-degree, mean over TFs and seeds')
for c in [REF] + conf:
    z = C.npzs(c)
    if z: L.append(f'  {nm(c):34s} {np.mean([C.pop_corr(q, pop) for q in z]):+.3f}')
tr = pd.read_csv('data/hESC/TFs+500/Train_set.csv', index_col=0)
hubthr = np.quantile(pop[pop > 0], 0.75)
L.append(f'\nPer-TF gain vs hub share of the TF\'s training targets (hub = training in-degree >= {hubthr:.0f}, top quartile of genes with any edge):')
B = C.tf_mean(REF)
for c in conf:
    A = C.tf_mean(c); ks = sorted(set(A) & set(B))
    share = np.array([np.mean(pop[tr.Target.values[(tr.TF.values == k) & (tr.Label.values == 1)]] >= hubthr) if ((tr.TF.values == k) & (tr.Label.values == 1)).any() else np.nan for k in ks])
    gain = np.array([A[k] - B[k] for k in ks]); ok = ~np.isnan(share)
    r = spearmanr(share[ok], gain[ok]); L.append(f'  {nm(c):34s} Spearman(hub share, within-TF gain) = {r.statistic:+.3f} (p={r.pvalue:.2g}, {ok.sum()} TFs)')
L.append('\nGene in-degree (training positives) per cell: hubs are mild in hESC and heavy elsewhere')
for cell in ('hESC', 'hHEP', 'mESC', 'mHSC-E'):
    pp = C.popularity(cell); L.append(f'  {cell:7s} genes {len(pp)}, max in-degree {pp.max()}, genes with >10 TFs {(pp > 10).sum()}, edges {pp.sum()}, edges removed by cap 10: {int(np.maximum(pp - 10, 0).sum())}, cap 20: {int(np.maximum(pp - 20, 0).sum())}')
L.append('\n== Stage 3: other cell types (3 seeds; baseline = w4 hero runs of the cell, no Geneformer)')
for cell in ('hHEP', 'mESC', 'mHSC-E'):
    ref = f'w4_{cell}_hero'
    L += C.table(cell, [(f'hero ({cell})', ref)] + [(v, f'w12_{cell}_hub_{v}') for v in ('idf05', 'idf1', 'cap10', 'cap20')], ref)
    for v in ('idf05', 'idf1', 'cap10', 'cap20'):
        z, zr = C.npzs(f'w12_{cell}_hub_{v}'), C.npzs(ref)
        if z and zr: L.append(f'  score/popularity Spearman {v}: {np.mean([C.pop_corr(q, C.popularity(cell)) for q in z]):+.3f} vs hero {np.mean([C.pop_corr(q, C.popularity(cell)) for q in zr]):+.3f}')
for prot in ('gene', 'tf'):
    L += ['\n== Stage 2'] + C.ho_report(prot, ['hub_idf05'])
open('analysis/out/w12_6_hubs.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
