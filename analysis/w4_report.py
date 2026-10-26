"""Week 3 part 5: the other three cell types. Per cell: mean ± std over seeds, paired per-TF tests vs that cell's hero (w4_<cell>_hero),
the scRegNet published recipe (rule A = validation AUROC), and the stacker rows. Run: python3.12 analysis/w4_report.py"""
import json
import numpy as np
from scipy.stats import wilcoxon
import sys
sys.path.insert(0, 'analysis')
import w3_report as w

CELLS = ['hHEP', 'mESC', 'mHSC-E']
HESC = dict(hero='w3_hero', herogf='w3_herogf', pgU_k100='w3_pgU_k100', np_had='w3_np_had', np_tani='w3_np_full', A_paper='f02_A_paper')  # hESC rows from the earlier tables
name = lambda c, k: HESC[k] if c == 'hESC' else f'w4_{c}_{k}'
ROWS = [('hero', 'hero (GNN, GRN graph)'), ('herogf', 'hero + Geneformer'), ('pgU_k100', 'GRN + PANDA K=100 safe (union)'), ('np_had', 'neural PANDA RCP, had decoder'),
        ('np_tani', 'neural PANDA RCP, Tanimoto decoder'), ('A_paper', 'scRegNet published recipe (rule A)')]
L = ['Week 3 part 5: hHEP (human), mESC and mHSC-E (mouse). GPU, 5 seeds (42-46), epoch chosen on validation AUROC, refit on all training rows.',
     'Hero settings were set on hESC and not re-tuned here. STRING PPI built with make_ppi.py (STRING v12, score >= 700, functional), same as hESC.']
for c in CELLS:
    L += w.table(f'{c}', [(nm, name(c, k)) for k, nm in ROWS], name(c, 'hero'))
    S = json.load(open(f'analysis/out/w3_stack_{c}.json')); Z = np.load(f'analysis/out/w3_stack_{c}.npz')
    B = w.tf_mean(name(c, 'hero'))
    base = 'label_+_contrast_+_PANDA_score_(best_in_cs_baselines)_[GBM]'
    bt = w.per_tf(Z['tf'], Z['y'], Z[base])
    L += [f'\n{c}: stacker (single fit; test rows full PANDA, training rows own-fold PANDA)',
          f"{'stack':72s} {'AUROC':>7s} {'AUPRC':>7s} {'within':>7s} {'vs best cs_baselines (p)':>26s} {'vs hero (p)':>14s}"]
    for k, v in S.items():
        if k.startswith('single') or k.replace(' ', '_') not in Z.files: continue
        t = w.per_tf(Z['tf'], Z['y'], Z[k.replace(' ', '_')])
        pv = lambda X, Y: wilcoxon(d).pvalue if np.any(d := [X[q] - Y[q] for q in X if q in Y]) else 1.0
        L.append(f"{k:72s} {v['auc']:7.4f} {v['aupr']:7.4f} {v['within']:7.4f} {np.mean([t[q] - bt[q] for q in t]):+.4f} (p={pv(t, bt):.2g}) {np.mean([t[q] - B[q] for q in t if q in B]):+.4f} (p={pv(t, B):.2g})")
CELLS = ['hESC'] + CELLS
# cross-cell summary: within-TF difference vs hero (p) and AUROC / AUPRC differences
L.append('\n== Cross-cell summary: AUROC / AUPRC / within-TF (mean over seeds); within-TF difference to the same cell\'s hero in brackets (Wilcoxon p)')
L.append(f"{'model':40s} " + ' '.join(f'{c:>38s}' for c in CELLS))
for k, nm in ROWS + [('stack_all_LR', 'stacker: all features, LR'), ('stack_base_GBM', 'stacker: label + contrast + PANDA score, GBM')]:
    cells = []
    for c in CELLS:
        if k.startswith('stack'):
            S = json.load(open(f'analysis/out/w3_stack_{c}.json')); Z = np.load(f'analysis/out/w3_stack_{c}.npz')
            key = 'all [LR]' if k == 'stack_all_LR' else 'label + contrast + PANDA score (best in cs_baselines) [GBM]'
            v = S[key]; t = w.per_tf(Z['tf'], Z['y'], Z[key.replace(' ', '_')]); B = w.tf_mean(name(c, 'hero')); d = np.array([t[q] - B[q] for q in t if q in B])
            m = (v['auc'], v['aupr'], v['within'])
        else:
            r = w.runs(name(c, k))
            if not r: cells.append('(no runs)'); continue
            m = tuple(np.mean([x['refit'][q] for x in r]) for q in ('auc', 'aupr', 'within'))
            B, A = w.tf_mean(name(c, 'hero')), w.tf_mean(name(c, k)); d = np.array([A[q] - B[q] for q in A if q in B])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        cells.append(f'{m[0]:.4f}/{m[1]:.4f}/{m[2]:.4f} ({d.mean():+.3f}, p={p:.2g})' if k != 'hero' else f'{m[0]:.4f}/{m[1]:.4f}/{m[2]:.4f}')
    L.append(f'{nm:40s} ' + ' '.join(f'{x:>38s}' for x in cells))
open('analysis/out/w4_report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
