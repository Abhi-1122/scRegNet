"""Sweep table for a W12 module: python3.12 analysis/w12_sweep_table.py nbf|seal  (seed 42 runs, validation fields first; selection is on val)"""
import glob, json, sys
for f in sorted(glob.glob(f'analysis/out/pair/w12_{sys.argv[1]}_*_s42.json')):
    r = json.load(open(f)); v, t = r['val'], r['refit']
    print(f"{r['name']:38s} ep {r['best_val_epoch']:3d} val auc {v['auc']:.4f} within {v['within']:.4f} | refit auc {t['auc']:.4f} aupr {t['aupr']:.4f} within {t['within']:.4f} | {r['minutes']:.1f} min")
