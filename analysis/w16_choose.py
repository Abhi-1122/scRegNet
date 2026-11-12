"""W16 step 1: choose each cell's GNN configuration among the 24 grid configurations (w11) on VALIDATION only (seeds 42-46 mean of the `val` field).
Run: python3.12 analysis/w16_choose.py -> analysis/out/w16_choose.txt. Test numbers are never read here."""
import glob, itertools, json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from src import pair_gnn as pg
CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']
sys.argv = ['x', '--name', 'x']
DEF = vars(pg.parse())
IGN = {'name', 'seed', 'out', 'data_folder', 'scfm_folder', 'dec_hidden', 'panda_k', 'panda_safe', 'own_edge_diag', 'mp_frac'}
def key(args):
    a = {**DEF, **args}
    ig = set(IGN) | (set() if a['encoder'] == 'npanda' else {'relations', 'coexp'}) | (set() if a['protocol'] != 'random' else {'fold'})
    return tuple(sorted((k, json.dumps(v)) for k, v in a.items() if k not in ig))
def cfg(cell, enc, dec, split, gf):
    return dict(cell_type=cell, encoder=enc, decoder=dec, split=split, scfm=gf, gnn_dims=[256, 256] if enc == 'npanda' else [256], optimizer='Adam', lr=0.0001, wd=0.001, eval_interval=1, refit=1, relations='RCP')
runs = defaultdict(dict)
for f in glob.glob('analysis/out/pair/*.json'):
    nm = f.split('/')[-1][:-5]
    if re.search(r'_s\d+$', nm):
        r = json.load(open(f))
        if 'refit' in r and os.path.exists(f[:-4] + 'npz'): runs[key(r['args'])][r['args']['seed']] = nm
if __name__ == '__main__':
    L = ['Validation-only choice of the GNN configuration per cell (mean over seeds 42-46 of the best-epoch validation slice metrics; no test numbers).']
    for c in CELLS:
        rows = []
        for e, d, s, g in itertools.product(['gcn', 'sage', 'npanda'], ['add', 'had'], ['none', 'batch'], [0, 1]):
            nm = runs.get(key(cfg(c, e, d, s, g)), {})
            js = [json.load(open(f'analysis/out/pair/{nm[x]}.json')) for x in range(42, 47) if x in nm]
            if len(js) == 5: rows.append((np.mean([j['val']['within'] for j in js]), np.mean([j['val']['auc'] for j in js]), e, d, s, g, nm[42][:-4]))
        rows.sort(reverse=True)
        L.append(f'\n== {c}: top 6 of {len(rows)} by validation within-TF   (name, val within-TF, val AUROC; AUROC rank in brackets)')
        ra = {r[6]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r[1]))}
        for r in rows[:6]: L.append(f'  {r[2]:7s} {r[3]:4s} {r[4]:6s} gf{r[5]}  val within {r[0]:.4f}  val AUROC {r[1]:.4f} [{ra[r[6]]}]   {r[6]}')
    open('analysis/out/w16_choose.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
