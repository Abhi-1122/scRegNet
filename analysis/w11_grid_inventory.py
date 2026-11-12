"""W11 task C: inventory of the consolidated grid, no runs. Grid = encoder {gcn, sage, npanda} x decoder {add (the harness default), had} x split {none, batch} x
Geneformer {0, 1}, gcn/sage at 1 GNN layer (256) and npanda at 2 layers (256 256), hero hyper-parameters, 4 cells x seeds 42-46. A run counts as existing when its json has all five seeds with args equal to the
grid config on every functionally relevant argument (older jsons are padded with the defaults of the flags they predate). Writes analysis/out/w11_grid_inventory.txt and
analysis/jobs/w11_grid.txt (missing runs, NOT submitted). Run: python3.12 analysis/w11_grid_inventory.py"""
import glob, itertools, json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
from src import pair_gnn as pg

CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']
SEEDS = range(42, 47)
HERO = '--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1'
sys.argv = ['x', '--name', 'x']
DEF = vars(pg.parse())
IGN = {'name', 'seed', 'out', 'data_folder', 'scfm_folder', 'dec_hidden', 'panda_k', 'panda_safe', 'own_edge_diag', 'mp_frac'}


def key(args, strict_np=False):
    a = {**DEF, **args}
    ig = set(IGN) | (set() if a['encoder'] == 'npanda' else {'relations', 'coexp'}) | (set() if a['split'] == 'fixed' else {'mp_frac'}) | (set() if a['protocol'] != 'random' else {'fold'})
    return tuple(sorted((k, json.dumps(v)) for k, v in a.items() if k not in ig))


def cfg(cell, enc, dec, split, gf, layers=None):
    layers = layers or (2 if enc == 'npanda' else 1)  # npanda is the 2-layer model that was reported
    return dict(cell_type=cell, encoder=enc, decoder=dec, split=split, scfm=gf, gnn_dims=[256] * layers, optimizer='Adam', lr=0.0001, wd=0.001, eval_interval=1, refit=1, relations='RCP')


runs = defaultdict(dict)   # key -> seed -> json name
mins = {}
for f in glob.glob('analysis/out/pair/*.json'):
    nm = f.split('/')[-1][:-5]
    if not re.search(r'_s\d+$', nm): continue
    r = json.load(open(f))
    if 'refit' not in r or not os.path.exists(f[:-4] + 'npz'): continue  # per-TF logits are needed for the paired tests; older g*/h* runs have none
    runs[key(r['args'])][r['args']['seed']] = nm
    mins[nm] = (r['minutes'], r['args'])

L = ['W11 task C: consolidated-grid inventory (no runs). Existing = all 5 seeds present with matching args on every functionally relevant flag.', '']
miss, exist = [], []
tab = {}
for cell, enc, dec, split, gf in itertools.product(CELLS, ['gcn', 'sage', 'npanda'], ['add', 'had'], ['none', 'batch'], [0, 1]):
    k = key(cfg(cell, enc, dec, split, gf))
    have = runs.get(k, {})
    full = all(s in have for s in SEEDS)
    tab[(cell, enc, dec, split, gf)] = (len(have), sorted(set(have.values()))[:1] and re.sub(r'_s\d+$', '', next(iter(have.values()))) if have else '')
    (exist if full else miss).append((cell, enc, dec, split, gf, have))

n_cfg = len(tab); n_full = len(exist); n_part = sum(1 for m in miss if m[5])
L.append(f'{n_cfg} configs (4 cells x 24), {n_full} complete at 5 seeds, {n_part} partly present, {n_cfg - n_full - n_part} absent. Missing runs: {sum(5 - len(m[5]) for m in miss)} of {5 * n_cfg}.')
L.append('Decoder column: add = TF + gene terms, no interaction (the harness default; there is no dot decoder), had = + Hadamard interaction. npanda = relations RCP, coexp full (flags of the w3/w4/w5 *np_had* runs), at 2 layers (as reported).')
L += ['', 'EXISTING (run-name prefix):']
for c, e, d, s, g, have in exist:
    L.append(f'  {c:7s} {e:6s} {d:3s} {s:5s} gf{g}  {re.sub(r"_s[0-9]+$", "", next(iter(have.values())))}')
L += ['', 'GRID (rows = encoder/decoder/split/GF, columns = cells; number = seeds already present, E = complete):']
L.append(f"{'encoder dec split gf':26s}" + ''.join(f'{c:>8s}' for c in CELLS))
for enc, dec, split, gf in itertools.product(['gcn', 'sage', 'npanda'], ['add', 'had'], ['none', 'batch'], [0, 1]):
    L.append(f'{enc + " " + dec + " " + split + " gf" + str(gf):26s}' + ''.join(f'{("E" if tab[(c, enc, dec, split, gf)][0] >= 5 else str(tab[(c, enc, dec, split, gf)][0])):>8s}' for c in CELLS))

# recipe rows
L += ['', 'scRegNet recipe rows (gcn, add, split none, Geneformer, 3 layers [227 29 34], SGD, lr 0.00144, 600 epochs; not part of the 1-layer grid):']
for c, n in zip(CELLS, ['f02_A_paper', 'w4_hHEP_A_paper', 'w4_mHSC-E_A_paper', 'w4_mESC_A_paper']):
    L.append(f'  {c:7s} {n}: {sum(1 for s in SEEDS if f"{n}_s{s}" in mins)} seeds')
# npanda at 2 layers (what exists)
two = [(c, d, s, g) for c, d, s, g in itertools.product(CELLS, ['add', 'had'], ['none', 'batch'], [0, 1]) if len(runs.get(key(cfg(c, 'npanda', d, s, g, 2)), {})) >= 5]
L.append('npanda at 2 layers [256 256] (what the w3/w4/w5 runs used) complete at 5 seeds: ' + (', '.join(f'{c} {d} {s} gf{g}' for c, d, s, g in two) or 'none'))

# runtime estimate: mean minutes per run of existing hero-type runs (Adam lr 1e-4, refit, random protocol, no extras; any depth/decoder/GF), pooled by
# (cell, family, split); runs were executed 4 at a time so the minutes already include sharing the machine. Family = npanda or plain GNN (gcn, sage).
pool = defaultdict(list)
for nm, (m, a) in mins.items():
    a = {**DEF, **a}
    if a['optimizer'] == 'Adam' and a['lr'] == 0.0001 and a['protocol'] == 'random' and a['mp_graph'] == 'grn' and a['scalar'] == 'none' and a['rank_w'] == 0 and a['aux_panda'] == 0 and a['refit']:
        pool[(a['cell_type'], 'np' if a['encoder'] == 'npanda' else 'gnn', a['split'])].append(m)
mean = lambda c, f, s: float(np.mean(pool[(c, f, s)])) if pool.get((c, f, s)) else None
ratio = {f: (mean('hESC', f, 'none') / mean('hESC', f, 'batch')) if mean('hESC', f, 'none') else mean('hESC', 'gnn', 'none') / mean('hESC', 'gnn', 'batch') for f in ('gnn', 'np')}
def m_of(c, e, s):
    f = 'np' if e == 'npanda' else 'gnn'
    return mean(c, f, s) or mean(c, f, 'batch') * ratio[f]
L += ['', 'Runtime: minutes per run = mean over existing hero-type runs of the same cell/family/split (family = npanda or gcn+sage; any depth, decoder, Geneformer). Runs ran 4 at a time, so the minutes include sharing the machine.',
      f"{'cell':8s}{'family':8s}{'split':7s}{'min/run':>9s}{'n runs':>8s}"]
for c, f, s in itertools.product(CELLS, ('gnn', 'np'), ('none', 'batch')):
    L.append(f"{c:8s}{f:8s}{s:7s}{m_of(c, 'npanda' if f == 'np' else 'gcn', s):9.2f}{len(pool.get((c, f, s), [])):8d}" + ('' if pool.get((c, f, s)) else f'   (no runs: batch value x none/batch ratio {ratio[f]:.2f} from hESC)'))
per = defaultdict(float); nrun = defaultdict(int)
for cell, enc, dec, split, gf, have in miss:
    per[cell] += (5 - len(have)) * m_of(cell, enc, split); nrun[cell] += 5 - len(have)
L += ['', 'Estimated cost of the missing runs (run-minutes = sum of per-run minutes; wall time at 4 slots = run-minutes / 4; treat as +/- a factor 2, epochs-to-convergence vary):']
for c in CELLS: L.append(f'  {c:7s} {nrun[c]:4d} runs {per[c]:8.1f} run-min  ~{per[c] / 4 / 60:5.2f} h wall')
L.append(f'  total   {sum(nrun.values()):4d} runs {sum(per.values()):8.1f} run-min  ~{sum(per.values()) / 4 / 60:5.2f} h wall')

# job file
J = []
for cell, enc, dec, split, gf, have in sorted(miss, key=lambda m: m[0] == 'mESC'):
    for s in SEEDS:
        if s in have: continue
        nm = f'w11_grid_{cell}_{enc}_{dec}_{split}_gf{gf}_s{s}'
        extra = ' --relations RCP' if enc == 'npanda' else ''
        J.append(f'{nm} {HERO} --gnn_dims {"256 256" if enc == "npanda" else "256"} --encoder {enc} --decoder {dec} --split {split} --scfm {gf} --cell_type {cell}{extra} --seed {s}')
open('analysis/jobs/w11_grid.txt', 'w').write('\n'.join(J) + '\n')
L.append(f'job file analysis/jobs/w11_grid.txt: {len(J)} lines (not submitted)')
open('analysis/out/w11_grid_inventory.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
