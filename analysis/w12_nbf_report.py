"""W12 idea 2 (NBFNet) report, hESC. Run: python3.12 analysis/w12_nbf_report.py   (writes analysis/out/w12_nbf_report.txt)
Seeds 42-44; hero = w3_hero; all choices made on validation (the sweep table's val columns)."""
import glob, json, sys
sys.path.insert(0, 'analysis')
from w12b_lib import *

L = ['W12 idea 2, NBFNet, hESC. Test numbers are refit on all training rows at the epoch chosen on validation AUROC; seeds 42-44.']
L.append('\n== Sweep (seed 42 only; choice by val AUROC). L = layers, tfi = TF-gene expression interaction in the final MLP')
for f in sorted(glob.glob('analysis/out/pair/w12_nbf_*_s42.json')):
    r = json.load(open(f)); v, t = r['val'], r['refit']
    L.append(f"  {r['name'][8:]:30s} ep {r['best_val_epoch']:3d} val AUROC {v['auc']:.4f} within {v['within']:.4f} | test AUROC {t['auc']:.4f} AUPRC {t['aupr']:.4f} within {t['within']:.4f}")
H = json.load(open('analysis/out/pair/w3_hero_s42.json'))
L.append(f"  {'hero (seed 42)':30s} ep {H['best_val_epoch']:3d} val AUROC {H['val']['auc']:.4f} within {H['val']['within']:.4f} | test AUROC {H['refit']['auc']:.4f} AUPRC {H['refit']['aupr']:.4f} within {H['refit']['within']:.4f}")
items = [('hero', HERO), ('NBFNet L6 pna + tfi (chosen)', 'w12_nbf_tfi_L6pna'), ('NBFNet pure paths (no features)', 'w12_nbf_paths_L6pna'), ('NBFNet no paths (0 layers, features only)', 'w12_nbf_nopath'), ('NBFNet L6 pna + tfi + gene degree (diagnostic)', 'w12_nbf_tfi_L6pna_deg'),
         ('NBFNet no paths, hidden 256 (diagnostic)', 'w12_nbf_nopath_h256'), ('NBFNet L3 sum hidden 128 + tfi (diagnostic)', 'w12_nbf_tfi_L3sum_h128')]
L += table('Stage 1, 3 seeds', items)
L += path_diag(items)
L += ho_table('tf', [('NBFNet L6 pna + tfi', 'w12_ho_{p}{f}_nbf_tfi_L6pna')])
L += ho_table('gene', [('NBFNet L6 pna + tfi', 'w12_ho_{p}{f}_nbf_tfi_L6pna')])
open('analysis/out/w12_nbf_report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
