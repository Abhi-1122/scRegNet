"""W16: validation-mode scores of the DC-SBM, i.e. the same model as the test-time one but fitted WITHOUT the validation rows (train minus validation slice),
so that a blend weight can be chosen on validation without leakage. Writes analysis/out/pair/<name>.npz/.json (grid over TFs x genes + the validation rows' scores).
  --kind std : learned DC-SBM (w12_stat model 2), k / l2 / steps = the validation-chosen values stored with the test-time fit (analysis/out/stat/<cell>_dcsbm_learned_s<seed>.json)
  --kind hog : fixed-cluster DC-SBM with Geneformer soft blocks on held-out genes (gene protocol, fold f), config = the one stored with analysis/out/stat/hESC_hog<f>_dcsbm_gf_soft_s<seed>.json
Run: python3.12 analysis/w16_dcval.py --name X --kind std --cell hESC --seed 42   (ada: analysis/ada.sh run JOBFILE P analysis.w16_dcval)"""
import argparse, json, os, sys, time
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import w12_stat as st

p = argparse.ArgumentParser()
p.add_argument('--name', required=True); p.add_argument('--kind', choices=['std', 'hog'], required=True)
p.add_argument('--cell', default='hESC'); p.add_argument('--fold', type=int, default=0); p.add_argument('--seed', type=int, default=42)
p.add_argument('--out', default='analysis/out/pair')
a = p.parse_args()
t0 = time.time()
if a.kind == 'std':
    meta = json.load(open(f'analysis/out/stat/{a.cell}_dcsbm_learned_s{a.seed}.json'))['cfg']
    D, val, full = st.load(NS(cell=a.cell, protocol='random', fold=0))
    g = st.fit_torch('sbm', val, dict(k=meta['k'], l2=meta['l2'], ra=None), a.seed, [meta['steps']])[meta['steps']]
else:
    meta = json.load(open(f'analysis/out/stat/{a.cell}_hog{a.fold}_dcsbm_gf_soft_s{a.seed}.json'))['cfg']
    D, val, full = st.load(NS(cell=a.cell, protocol='gene', fold=a.fold))
    F = st.gene_feats(D)
    R0 = st.posterior('gf', F['gf'], meta['cov'], meta['K'], a.seed)
    R = R0 if meta['soft'] else st.hard(R0)
    g = st.dcsbm_grid(val, R, dict(alpha=meta['alpha'], shrink=meta['shrink'], b=meta['b'], dc=meta['dc']))[0]
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(f'{a.out}/{a.name}.npz', grid=g.astype(np.float32), grid_tf=val.tfids, logit=val.score(g), tf=val.ev_tf, tgt=val.ev_g, y=val.ev_y)
m = val.metrics(g)
json.dump(dict(name=a.name, args=vars(a), cfg=meta, val=m, minutes=(time.time() - t0) / 60), open(f'{a.out}/{a.name}.json', 'w'), default=str)
print(f'{a.name}\tval AUROC {m["auc"]:.4f} withinTF {m["within"]:.4f}\t{(time.time() - t0) / 60:.1f} min')
