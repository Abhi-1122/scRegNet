"""W11 task C report: consolidated grid encoder {gcn, sage (1 layer), npanda (2 layers)} x decoder {add, had} x split {none, batch} x Geneformer {0, 1}, hero hyper-parameters,
4 cells x seeds 42-46, plus the scRegNet recipe. Master tables, main effects (paired per-TF), best config vs recipe. Run: python3.12 analysis/w11_grid_report.py"""
import glob, itertools, json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from src import pair_gnn as pg
import w3_report as w

CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']
RECIPE = {'hESC': 'f02_A_paper', 'hHEP': 'w4_hHEP_A_paper', 'mHSC-E': 'w4_mHSC-E_A_paper', 'mESC': 'w4_mESC_A_paper'}
ENC, DEC, SPL, GF = ['gcn', 'sage', 'npanda'], ['add', 'had'], ['none', 'batch'], [0, 1]
sys.argv = ['x', '--name', 'x']
DEF = vars(pg.parse())
IGN = {'name', 'seed', 'out', 'data_folder', 'scfm_folder', 'dec_hidden', 'panda_k', 'panda_safe', 'own_edge_diag', 'mp_frac'}


def key(args):  # same matching rule as w11_grid_inventory.py
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
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'


def load(names):
    """names: {seed: run name} -> (metrics list, per-TF AUROC averaged over seeds)"""
    R = [json.load(open(f'analysis/out/pair/{n}.json'))['refit'] for _, n in sorted(names.items())]
    T = []
    for _, n in sorted(names.items()):
        z = np.load(f'analysis/out/pair/{n}.npz'); T.append(w.per_tf(z['tf'], z['y'], z['logit']))
    ks = sorted(set.intersection(*map(set, T)))
    return R, {k: float(np.mean([t[k] for t in T])) for k in ks}


L = ['W11 grid. 5 seeds (42-46), epoch chosen on validation AUROC, refit on all training rows; mean ± sd over seeds. Encoders: gcn and sage at 1 layer (256), npanda at 2 layers (256 256, relations RCP).',
     'Decoder add = TF + gene terms, no interaction; had = add + Hadamard interaction. Split none / batch (own batch edges removed from the training graph). GF = Geneformer.',
     'Hero hyper-parameters: Adam lr 1e-4, wd 1e-3, dropout 0.175, batch 193, mlp 198. Recipe = scRegNet published recipe (gcn, add, none, GF, 3 layers 227/29/34, SGD).']
G = {}  # (cell, enc, dec, split, gf) -> (metrics, tfmeans) ; None if missing
bad = []
for c, e, d, s, g in itertools.product(CELLS, ENC, DEC, SPL, GF):
    nm = runs.get(key(cfg(c, e, d, s, g)), {})
    if len([x for x in nm if 42 <= x <= 46]) == 5:
        G[(c, e, d, s, g)] = load({x: nm[x] for x in range(42, 47)})
    else:
        G[(c, e, d, s, g)] = None; bad.append((c, e, d, s, g, len(nm)))
Rec = {c: load({sd: f'{RECIPE[c]}_s{sd}' for sd in range(42, 47)}) for c in CELLS}
if bad: L.append('\nMISSING configs (seeds present): ' + '; '.join(f'{b[0]} {b[1]} {b[2]} {b[3]} gf{b[4]} ({b[5]})' for b in bad))

for c in CELLS:
    L += [f'\n== {c}: master table', f"{'encoder':8s}{'dec':4s}{'split':6s}{'GF':3s}{'AUROC':>17s}{'AUPRC':>17s}{'within-TF':>17s}"]
    for e, d, s, g in itertools.product(ENC, DEC, SPL, GF):
        v = G[(c, e, d, s, g)]
        L.append(f'{e:8s}{d:4s}{s:6s}{g:<3d}' + (''.join(f"{ms([r[m] for r in v[0]]):>17s}" for m in ('auc', 'aupr', 'within')) if v else '  (missing)'))
    R = Rec[c][0]
    L.append(f'{"recipe":21s}' + ''.join(f"{ms([r[m] for r in R]):>17s}" for m in ('auc', 'aupr', 'within')) + f'   ({RECIPE[c]})')

# main effects
def contrasts(factor):
    if factor == 'encoder': return [('sage - gcn', 'gcn', 'sage'), ('npanda - gcn', 'gcn', 'npanda'), ('npanda - sage', 'sage', 'npanda')]
    return {'decoder': [('had - add', 'add', 'had')], 'split': [('batch - none', 'none', 'batch')], 'Geneformer': [('GF1 - GF0', 0, 1)]}[factor]
pos = {'encoder': 1, 'decoder': 2, 'split': 3, 'Geneformer': 4}
L += ['\n== Main effects (within-TF AUROC). Pairs: configs that differ only in the factor; per cell the pairs are pooled over the other factors.',
      'For each TF the per-TF AUROC difference (seed-averaged) is averaged over the pairs; mean diff = mean of that over TFs; Wilcoxon over TFs. "pairs +" = pairs whose mean per-TF difference is > 0.',
      f"{'cell':8s}{'factor':12s}{'contrast (B - A)':16s}{'pairs':>6s}{'mean diff':>11s}{'pairs +':>9s}{'TFs +':>8s}{'Wilcoxon p':>12s}"]
for f in pos:
    for lab, A, B in contrasts(f):
        for c in CELLS:
            ds, pw = [], []
            for e, d, s, g in itertools.product(ENC, DEC, SPL, GF):
                k = [e, d, s, g]
                if k[pos[f] - 1] != A: continue
                k2 = list(k); k2[pos[f] - 1] = B
                a, b = G[(c, *k)], G[(c, *k2)]
                if a is None or b is None: continue
                ks = sorted(set(a[1]) & set(b[1]))
                ds.append({t: b[1][t] - a[1][t] for t in ks}); pw.append(np.mean([b[1][t] - a[1][t] for t in ks]))
            if not ds: L.append(f'{c:8s}{f:12s}{lab:16s}     0'); continue
            tfs = sorted(set.intersection(*map(set, ds)))
            dm = np.array([np.mean([x[t] for x in ds]) for t in tfs])
            p = wilcoxon(dm).pvalue if np.any(dm != 0) else 1.0
            L.append(f'{c:8s}{f:12s}{lab:16s}{len(ds):6d}{dm.mean():+11.4f}{sum(x > 0 for x in pw):6d}/{len(pw):<2d}{int((dm > 0).sum()):5d}/{len(dm):<2d}{p:12.3g}')
        L.append('')

# best config
L.append('== Best config per cell (highest 5-seed mean within-TF among the 24 grid configs; chosen on the test metric, so optimistic) vs the recipe')
for c in CELLS:
    cand = {k: v for k, v in G.items() if k[0] == c and v}
    k = max(cand, key=lambda q: np.mean([r['within'] for r in cand[q][0]]))
    R = cand[k][0]
    L.append(f'{c}: {k[1]} {k[2]} {k[3]} gf{k[4]}   AUROC {ms([r["auc"] for r in R])}  AUPRC {ms([r["aupr"] for r in R])}  within-TF {ms([r["within"] for r in R])}')
    L.append(w.paired(cand[k][1], Rec[c][1], 'best config - recipe'))
    top = sorted(cand, key=lambda q: -np.mean([r['within'] for r in cand[q][0]]))[:3]
    L.append('   top 3 by within-TF: ' + '; '.join(f'{q[1]} {q[2]} {q[3]} gf{q[4]} {np.mean([r["within"] for r in cand[q][0]]):.4f}' for q in top))
open('analysis/out/w11_grid.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
