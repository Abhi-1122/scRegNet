"""Week 4 part A: GCN vs SAGE (+ Geneformer) on the four cell types, then the tuned winner. Per cell: mean ± std over 5 seeds, paired per-TF tests
(per-TF AUROC averaged over seeds, Wilcoxon over test TFs; helpers from w3_report.py). Run: python3.12 analysis/w8_report.py"""
import json, sys
import numpy as np
sys.path.insert(0, 'analysis')
import w3_report as w

CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']
nm = lambda c, k: {'hero': 'w3_hero', 'sage': 'w3_herogf', 'paper': 'f02_A_paper'}.get(k) if c == 'hESC' and k in ('hero', 'sage', 'paper') else \
    {'hero': f'w4_{c}_hero', 'sage': f'w4_{c}_herogf', 'paper': f'w4_{c}_A_paper'}.get(k, f'w8_{c}_{k}')
ROWS = [('sage', 'SAGE + Geneformer'), ('gcngf', 'GCN + Geneformer'), ('hero', 'hero (SAGE, no Geneformer)'), ('paper', 'scRegNet published recipe (rule A)')]
L = ['Week 4 part A. 5 seeds (42-46), epoch chosen on validation AUROC, refit on all training rows. mean ± std over seeds.']
best = {}
for c in CELLS:
    items = [(t, nm(c, k)) for k, t in ROWS]
    L += [f'\n== {c}: part 1', w.HDR] + [w.row(t, n) for t, n in items]
    m = {k: np.mean([x['refit']['within'] for x in w.runs(nm(c, k))]) for k in ('sage', 'gcngf')}
    v = {k: np.mean([x['val']['auc'] for x in w.runs(nm(c, k))]) for k in ('sage', 'gcngf')}
    b = max(m, key=m.get)
    best[c] = b
    L += ['paired per-TF tests:', w.paired(w.tf_mean(nm(c, 'gcngf')), w.tf_mean(nm(c, 'sage')), 'GCN+GF - SAGE+GF'),
          w.paired(w.tf_mean(nm(c, b)), w.tf_mean(nm(c, 'paper')), f'best ({b}) - published recipe'),
          w.paired(w.tf_mean(nm(c, b)), w.tf_mean(nm(c, 'hero')), f'best ({b}) - hero without Geneformer'),
          f"best by test within-TF: {b} ({m['sage']:.4f} SAGE vs {m['gcngf']:.4f} GCN); by validation AUROC: {max(v, key=v.get)} ({v['sage']:.4f} vs {v['gcngf']:.4f})"]
    if w.runs(nm(c, 'tuned')):
        W = nm(c, b)
        L += [f'\n== {c}: part 2 (tuned {b})', w.HDR] + [w.row(t, n) for t, n in [(f'untuned {b} + Geneformer', W), ('tuned', nm(c, 'tuned')), ('published recipe', nm(c, 'paper'))]]
        L += ['paired per-TF tests:', w.paired(w.tf_mean(nm(c, 'tuned')), w.tf_mean(W), 'tuned - untuned'), w.paired(w.tf_mean(nm(c, 'tuned')), w.tf_mean(nm(c, 'paper')), 'tuned - published recipe')]
open('analysis/out/w8_report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
json.dump(best, open('analysis/out/w8_best.json', 'w'))
