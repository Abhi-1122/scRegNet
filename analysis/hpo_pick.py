"""Pick each family's best trial by VALIDATION AUROC (tuning seed 0). Prints the winning flags.
Usage: python3.12 analysis/hpo_pick.py  -> writes analysis/out/hpo_best.json"""
import glob, json, re
best = {}
for f in glob.glob('analysis/out/pair/hpo_*.json'):
    r = json.load(open(f)); fam = re.match(r'hpo_(\w+?)_(t\d+|paper)$', r['name']).group(1)
    if fam not in best: best[fam] = []
    best[fam].append((r['val']['auc'], r['name'], r['args'], r['best_val_epoch']))
out = {}
for fam, rows in sorted(best.items()):
    rows.sort(reverse=True)
    v, nm, a, ep = rows[0]
    keep = {k: a[k] for k in ['optimizer', 'lr', 'wd', 'dropout', 'mlp_dim', 'batch_size', 'gnn_dims', 'eval_interval', 'epochs']}
    out[fam] = dict(trial=nm, val_auc=v, best_epoch=ep, n_trials=len(rows), flags=keep)
    print(f"{fam:16s} {len(rows):2d} trials  best {nm}  val AUROC {v:.4f}  epoch {ep}  {keep}")
    print(f"{'':16s} runner-up val AUROC {rows[1][0]:.4f}, median {sorted(r[0] for r in rows)[len(rows)//2]:.4f}")
json.dump(out, open('analysis/out/hpo_best.json', 'w'), indent=1)
