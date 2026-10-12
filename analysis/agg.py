"""Aggregate seeds: python3.12 analysis/agg.py PREFIX [PREFIX...] -> mean +/- std over seeds, per config.
Columns: val-selected TEST (the clean number), REFIT TEST (retrained on all training rows for the chosen
epoch; comparable to earlier full-data numbers), and the old test-selected number for reference."""
import glob, json, re, sys
import numpy as np
from collections import defaultdict
groups = defaultdict(list)
for pre in sys.argv[1:]:
    for f in sorted(glob.glob(f'analysis/out/pair/{pre}*.json')):
        r = json.load(open(f)); groups[re.sub(r'_s\d+$', '', r['name'])].append(r)
ms = lambda v: f'{np.mean(v):.4f}±{np.std(v):.4f}'
print(f"{'config':34s} n | val-sel TEST AUROC  AUPRC   withinTF       | REFIT TEST AUROC   AUPRC   withinTF       | epochs")
for k, rs in groups.items():
    g = lambda key, m: [r[key][m] for r in rs if key in r]
    ref = ' '.join(ms(g('refit', m)) for m in ('auc', 'aupr', 'within')) if all('refit' in r for r in rs) else '-'
    print(f"{k:34s} {len(rs)} | {ms(g('test','auc'))} {ms(g('test','aupr'))} {ms(g('test','within'))} | {ref} | {[r['best_val_epoch'] for r in rs]}")
