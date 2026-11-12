"""W12 idea 3 (SEAL) report, hESC. Run: python3.12 analysis/w12_seal_report.py   (writes analysis/out/w12_seal_report.txt)
Seeds 42-44; hero = w3_hero; all choices made on validation (the sweep table's val columns)."""
import glob, json, sys
sys.path.insert(0, 'analysis')
from w12b_lib import *

L = ['W12 idea 3, SEAL, hESC. Test numbers are refit on all training rows at the epoch chosen on validation AUROC; seeds 42-44.']
L.append('\n== Sweep (seed 42 only; choice by val AUROC, ties by val within-TF)')
for f in sorted(glob.glob('analysis/out/pair/w12_seal_*_s42.json')):
    r = json.load(open(f)); v, t = r['val'], r['refit']
    L.append(f"  {r['name'][9:]:30s} ep {r['best_val_epoch']:3d} val AUROC {v['auc']:.4f} within {v['within']:.4f} | test AUROC {t['auc']:.4f} AUPRC {t['aupr']:.4f} within {t['within']:.4f}")
H = json.load(open('analysis/out/pair/w3_hero_s42.json'))
L.append(f"  {'hero (seed 42)':30s} ep {H['best_val_epoch']:3d} val AUROC {H['val']['auc']:.4f} within {H['val']['within']:.4f} | test AUROC {H['refit']['auc']:.4f} AUPRC {H['refit']['aupr']:.4f} within {H['refit']['within']:.4f}")
items = [('hero', HERO), ('SEAL 1-hop cap 30, tt pooling (chosen)', 'w12_seal_h1tt'), ('SEAL without DRNL labels', 'w12_seal_h1tt_nodrnl'),
         ('SEAL without expression (structure only)', 'w12_seal_h1tt_noexpr'), ('SEAL without degree features', 'w12_seal_h1tt_nodeg'),
         ('SEAL expression PCA 128 (diagnostic)', 'w12_seal_h1tt_pca128'), ('SEAL expression PCA 256 (diagnostic)', 'w12_seal_h1tt_pca256')]
L += table('Stage 1, 3 seeds', items)
L += path_diag(items)
L += ho_table('tf', [('SEAL 1-hop tt', 'w12_ho_{p}{f}_seal_h1tt')])
L += ho_table('gene', [('SEAL 1-hop tt', 'w12_ho_{p}{f}_seal_h1tt')])
open('analysis/out/w12_seal_report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
