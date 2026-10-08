"""Item 2, Result C: Exp C retrained with each PANDA variant (3 seeds each). Dumps each run's scores
with analysis/model_scores.py if needed, then reports AUROC, AUPRC and within-TF AUROC."""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, 'analysis')
from common import load, metrics

tr, te, genes, expr, tfs = load()
log = open('analysis/out/item2_runs.log').read() if os.path.exists('analysis/out/item2_runs.log') else ''
finished = lambda r: any(l.startswith(r + '\t') for l in log.splitlines())
t, y = te.TF.values, te.Label.values
runs = {
    'baseline (no PANDA)': ['tf_500_hESC_baseline_rerun', 'tf_500_hESC_pandavar_none_seed43', 'tf_500_hESC_pandavar_none_seed44'],
    'full (Exp C)': ['tf_500_hESC_panda_edge', 'tf_500_hESC_panda_edge_seed43', 'tf_500_hESC_panda_edge_seed44'],
}
for v in ['no_coexpr', 'no_ppi', 'prior_only', 'raw_corr', 'full_symoof', 'no_ppi_symoof', 'no_coexpr_symoof']:
    runs[v] = [f'tf_500_hESC_pandavar_{v}_seed{s}' for s in (42, 43, 44)]

print(f"{'variant':20s} {'seeds':>5s}  {'AUROC':>15s} {'AUPRC':>15s} {'within-TF AUROC':>17s}")
for name, rs in runs.items():
    m = []
    for r in rs:
        f = f'analysis/out/scores_{r}.npz'
        if 'pandavar' in r and not finished(r): continue  # a checkpoint exists mid-training; wait for the run to end
        if not os.path.exists(f):
            if not os.path.exists(f'out/GCN/Geneformer/{r}/ckpt/args.json'): continue
            subprocess.run([sys.executable, '-m', 'analysis.model_scores', r], capture_output=True)
        z = np.load(f)['test_logits']
        x = metrics(y, z[:, 1] - z[:, 0], t); m.append((x['AUROC'], x['AUPRC'], x['withinTF_AUROC']))
    if not m: continue
    m = np.array(m); mu, sd = m.mean(0), m.std(0)
    print(f'{name:20s} {len(m):5d}  {mu[0]:.4f}±{sd[0]:.4f} {mu[1]:.4f}±{sd[1]:.4f} {mu[2]:.4f}±{sd[2]:.4f}'
          + ''.join(f'\n{"":27s}seed run {r}: {a:.4f} {b:.4f} {c:.4f}' for r, (a, b, c) in zip(rs, m)))
