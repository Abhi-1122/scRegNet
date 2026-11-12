"""W11 task B1/B2 tables, appended to analysis/out/w11_edge_checks.txt after w11_edge_audit.py. Run: python3.12 analysis/w11_edge_audit.py && python3.12 analysis/w11_edge_report.py"""
import sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, 'analysis')
import w3_report as w

L = ['', 'B1 own-edge memorisation (hESC, hero config without Geneformer, 5 seeds 42-46, refit model).',
     'Scored after training: a random 10% of the training-positive undirected edges (n=NEDGES) + as many training negatives, fixed seed 123; graph = all training edges ("present") vs the same graph with those edges removed ("removed").']
rows = {}
for nm in ('w11_own_batch', 'w11_own_none'):
    R = w.runs(nm); rows[nm] = [r['own_edge_diag'] for r in R]
L[2] = L[2].replace('NEDGES', str(rows['w11_own_batch'][0]['n_edges']))
H = f"{'split':8s} {'n':>2s} {'pos logit gap (present-removed)':>32s} {'median gap':>11s} {'neg gap':>9s} {'pos logit present/removed':>27s} {'AUROC present':>16s} {'AUROC removed':>16s} {'test AUROC/AUPRC/within-TF':>40s}"
L.append(H)
for nm, lab in (('w11_own_batch', 'batch'), ('w11_own_none', 'none')):
    d = rows[nm]; R = w.runs(nm); g = lambda k: [x[k] for x in d]
    L.append(f"{lab:8s} {len(d):2d} {w.ms(g('gap_pos_mean')):>32s} {np.mean(g('gap_pos_median')):11.3f} {np.mean(g('gap_neg_mean')):9.3f} "
             f"{np.mean(g('pos_logit_present')):+13.2f}/{np.mean(g('pos_logit_removed')):+.2f} {w.ms(g('auc_present')):>16s} {w.ms(g('auc_removed')):>16s}   "
             f"{w.ms([r['refit']['auc'] for r in R])} / {w.ms([r['refit']['aupr'] for r in R])} / {w.ms([r['refit']['within'] for r in R])}")
L.append('per-seed gap (batch): ' + ' '.join(f'{x["gap_pos_mean"]:+.3f}' for x in rows['w11_own_batch']) + '   (none): ' + ' '.join(f'{x["gap_pos_mean"]:+.3f}' for x in rows['w11_own_none']))
L.append('negatives mean logit present/removed (batch, none): %.2f/%.2f, %.2f/%.2f' % tuple(np.mean([x[k] for x in rows[nm]]) for nm in rows for k in ('neg_logit_present', 'neg_logit_removed')))

L += ['', 'B2 fixed 70/30 split vs per-batch removal (hESC, hero config without Geneformer, 5 seeds, test set, epoch on validation AUROC, refit).',
      'Reference w3_hero = --split batch, scfm 0, same args (checked). w11_own_batch = identical args plus the diagnostic flag (reproducibility check).', w.HDR]
items = [('hero, split batch (w3_hero)', 'w3_hero'), ('hero + diag flag (w11_own_batch)', 'w11_own_batch'), ('split none (w11_own_none)', 'w11_own_none'), ('split fixed mp_frac 0.7 (w11_fixed)', 'w11_fixed')]
L += [w.row(n, c) for n, c in items]
L.append('best validation epoch per seed: ' + '; '.join(f"{c}: {[r['best_val_epoch'] for r in w.runs(c)]}" for _, c in items))
L.append('paired per-TF tests (per-TF AUROC averaged over seeds, Wilcoxon over TFs):')
for c in ('w11_fixed', 'w11_own_none', 'w11_own_batch'):
    L.append(w.paired(w.tf_mean(c), w.tf_mean('w3_hero'), f'{c} - w3_hero'))
L.append(w.paired(w.tf_mean('w11_fixed'), w.tf_mean('w11_own_none'), 'w11_fixed - w11_own_none'))
a0, a1 = w.runs('w3_hero')[0]['args'], w.runs('w11_own_batch')[0]['args']
L.append('args differ w3_hero vs w11_own_batch: ' + str({k: (a0.get(k), a1.get(k)) for k in set(a0) | set(a1) if a0.get(k) != a1.get(k) and k != 'name'}))
open('analysis/out/w11_edge_checks.txt', 'a').write('\n'.join(L) + '\n'); print('\n'.join(L))
