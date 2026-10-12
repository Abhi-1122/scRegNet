"""Print finished pair_gnn runs: python3.12 analysis/show.py PREFIX"""
import glob, json, sys
rows = []
for f in sorted(glob.glob(f'analysis/out/pair/{sys.argv[1]}*.json')):
    r = json.load(open(f)); rows.append(r)
print(f"{'run':46s} ep  valAUC valWithin | testAUC testAUPR testWithin | test-selected AUC/Within | min")
for r in rows:
    v, t, s = r['val'], r['test'], r['test_selected']
    print(f"{r['name']:46s} {r['best_val_epoch']:3d} {v['auc']:.4f} {v['within']:.4f}   | {t['auc']:.4f} {t['aupr']:.4f} {t['within']:.4f}   | {s['auc']:.4f} {s['within']:.4f} | {r['minutes']:.1f}")
