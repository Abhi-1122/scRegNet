"""W22 fair baselines for complete network inference: report from the saved score grids in analysis/out/xcell_base.

  python3.12 analysis/w22_report.py   -> analysis/out/w22_baselines.txt
Per-TF AUROC of seed runs is averaged over seeds 42-44; the scRegNet recipe is chosen on mean source-validation within-TF.
Wilcoxon: paired per TF, two-sided, no multiplicity correction. W20 / W21 rows are added when their runs exist.
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')
from src.xcell_gnn import CELLS, cell, evaluate, load_labels
from src.xcell_base import OUT
sys.path.insert(0, 'analysis')
import w20_xcell_report as w20  # read-only use of W20's run index, chooser and scorer

SS = [('hESC', 'hHEP'), ('hHEP', 'hESC'), ('mESC', 'mHSC-E'), ('mHSC-E', 'mESC')]
SETS = {f'ss_{s}-{t}': ([s], t) for s, t in SS} | {f'loo_{t}': ([c for c in CELLS if c != t], t) for t in CELLS}
UNSUP = ['panda', 'panda-noppi', 'genie3-et', 'genie3-et-hard', 'genie3-et-soft', 'genie3-gbm', 'genie3-gbm-hard', 'genie3-gbm-soft']
LABEL = {'motif': 'motif alone', 'panda-nz': 'PANDA netZooPy (W20 run)', 'panda': 'PANDA', 'panda-noppi': 'PANDA no PPI',
         'genie3-et': 'GENIE3 (ExtraTrees)', 'genie3-et-hard': 'GENIE3 + hard motif pruning', 'genie3-et-soft': 'GENIE3 + soft motif (rank avg)',
         'genie3-gbm': 'GRNBoost2-style (GBM)', 'genie3-gbm-hard': 'GBM + hard motif pruning', 'genie3-gbm-soft': 'GBM + soft motif (rank avg)',
         'scr-paper': 'scRegNet-motif, published SGD recipe', 'scr-tuned': 'scRegNet-motif, Adam-tuned recipe', 'scr-chosen': 'scRegNet-motif, recipe chosen on source val',
         'w20': 'W20 learned PANDA (full, chosen on source val)', 'w20mlp': 'W20 feature-only MLP (chosen on source val)'}


def ev_grid(c, grid_tf, grid):
    e = evaluate(c, grid_tf, grid); return e['per_tf'], e['auc'], e['aupr']


def score_files(names):
    """seed runs -> (per-TF AUROC averaged over runs, mean pooled AUROC, mean AUPRC, jsons)"""
    pt, au, ap, js = defaultdict(list), [], [], []
    for n in names:
        j = json.load(open(f'{OUT}/{n}.json')); g = np.load(f'{OUT}/{n}.npz')
        p, a, b = ev_grid(j['tgt'], g['grid_tf'], g['grid'])
        for t, v in p.items(): pt[t].append(v)
        au.append(a); ap.append(b); js.append(j)
    return {t: float(np.mean(v)) for t, v in pt.items()}, float(np.mean(au)), float(np.mean(ap)), js


def motif_scores(c):
    D = cell(c); Mx = D['Mx'][D['tf']]
    return ev_grid(c, D['tf'], np.where(np.isnan(Mx), np.nanmin(D['Mx']) - 1, Mx))


m_ = w20.m_
wx = w20.wx


def other_models():
    """W20 full model per setting (chosen as in its report); W21 if its run index exists. {(setting, key): (per_tf, auc, aupr)}"""
    out = {}
    try:
        R = w20.runs()
        for st in SETS:
            for pre, key in (('full', 'w20'), ('mlp', 'w20mlp')):
                cfg = w20.choose(R, st, pre); out[(st, key)] = w20.score_runs(R[(st, cfg)].values())
    except Exception as e:
        print('W20 rows skipped:', repr(e))
    # W21 (analysis/out/xcell2) is left out until its final report analysis/out/w21_2view.txt exists
    return out


def report():
    out = []
    P = lambda *x: out.append(' '.join(str(v) for v in x))
    sym = {c: cell(c)['genes'] for c in CELLS}
    lab_tfs = {c: np.unique(load_labels(c).TF.values) for c in CELLS}
    lab_sym = {c: set(sym[c][lab_tfs[c]]) for c in CELLS}
    U = {}  # (cell, key) -> (per_tf, auc, aupr)
    for c in CELLS:
        U[(c, 'motif')] = motif_scores(c)
        b = np.load(f'analysis/out/xcell/base_{c}.npz'); U[(c, 'panda-nz')] = ev_grid(c, b['grid_tf'], b['panda'])
        for k in UNSUP:
            if os.path.exists(f'{OUT}/w22_{k}_{c}.json'): U[(c, k)] = score_files([f'w22_{k}_{c}'])[:3]
    S, CH = {}, {}
    for st in SETS:
        val = {}
        for r in ('paper', 'tuned'):
            names = sorted(os.path.basename(f)[:-5] for f in glob.glob(f'{OUT}/w22_{st}__scr-{r}__s*.json'))
            if names:
                pt, au, ap, js = score_files(names); S[(st, f'scr-{r}')] = (pt, au, ap); val[r] = (np.mean([j['val_within'] for j in js]), [j['best_epoch'] for j in js], len(js))
        if len(val) == 2:
            CH[st] = max(val, key=lambda r: val[r][0]); S[(st, 'scr-chosen')] = S[(st, f'scr-{CH[st]}')]
        CH[st + ':val'] = val
    S.update(other_models())

    P('W22 fair baselines for complete network inference (src/xcell_base.py, analysis/w22_report.py)')
    P('within-TF AUROC = mean over target TFs with both classes of the per-TF AUROC on every labelled pair of the target cell.')
    P('Unsupervised baselines (PANDA, GENIE3/GBM) use only the target cell; scRegNet-motif is trained on source labels only.')
    P('p = paired per-TF Wilcoxon against motif alone (two-sided, no multiplicity correction).')
    P()
    P('== A. Unsupervised baselines on each cell: within-TF all TFs (p vs motif) | TFs with motif | TFs without motif ; pooled AUROC / AUPRC')
    for c in CELLS:
        D = cell(c); tfs = lab_tfs[c]; hm = [t for t in tfs if D['has_m'][t]]; nm = [t for t in tfs if not D['has_m'][t]]
        P(f'{c}: labelled TFs {len(tfs)} (motif {len(hm)}, no motif {len(nm)})')
        for k in ['motif', 'panda-nz'] + UNSUP:
            if (c, k) not in U: P(f'   {LABEL[k]:34s} (missing)'); continue
            pt, au, ap = U[(c, k)]
            p = '' if k == 'motif' else f'(p {wx(pt, U[(c, "motif")][0], sorted(set(pt) & set(U[(c, "motif")][0]))):.2g})'
            P(f'   {LABEL[k]:34s} {m_(pt, tfs):.4f} {p:10s} | {m_(pt, hm):.4f} | {m_(pt, nm):.4f} ; {au:.4f} / {ap:.4f}')
    P()
    P('== B. Recipe choice for scRegNet-motif (mean source-validation within-TF over seeds; best epochs per seed)')
    for st in SETS:
        for r, (v, ep, n) in CH[st + ':val'].items():
            P(f'{st:16s} {r:5s} val {v:.4f} epochs {ep} ({n} seeds){"  <- chosen" if CH.get(st) == r else ""}')
    P()
    P('== C. Transfer settings: within-TF AUROC on all labelled target TFs (p vs motif alone)')
    keys = ['scr-paper', 'scr-tuned', 'scr-chosen', 'w20', 'w20mlp']
    P(f'{"setting":16s} {"nTF":>3s} {"motif":>6s} {"PANDA":>6s} {"GENIE3":>6s} {"G3-hard":>7s} {"G3-soft":>7s}   ' + '   '.join(f'{k:>16s}' for k in keys))
    for st, (srcs, t) in SETS.items():
        tfs = sorted(lab_tfs[t]); mo = U[(t, 'motif')][0]
        row = f'{st:16s} {len(tfs):3d} {m_(mo, tfs):.4f} ' + ' '.join(f'{m_(U[(t, k)][0], tfs):.4f}'.rjust(7) if (t, k) in U else '      -' for k in ('panda', 'genie3-et', 'genie3-et-hard', 'genie3-et-soft'))
        for k in keys:
            row += '   ' + (f'{m_(S[(st, k)][0], tfs):.4f} (p {wx(S[(st, k)][0], mo, [x for x in tfs if x in S[(st, k)][0]]):.1e})'.rjust(16) if (st, k) in S else '-'.rjust(16))
        P(row)
    P()
    base = ['panda', 'genie3-et', 'genie3-et-soft', 'genie3-gbm-soft', 'scr-chosen']
    for wk in ('w20', 'w20mlp'):
      P(f'== C2. {LABEL[wk]} minus each baseline: within-TF difference (paired per-TF Wilcoxon p)')
      P(f'{"setting":16s} ' + ' '.join(f'{k:>20s}' for k in base))
      for st, (srcs, t) in SETS.items():
        if (st, wk) not in S: continue
        w = S[(st, wk)][0]; tfs = sorted(lab_tfs[t]); row = f'{st:16s} '
        for k in base:
            o = S[(st, k)][0] if (st, k) in S else U.get((t, k), (None,))[0]
            row += (f'{m_(w, tfs) - m_(o, tfs):+.4f} ({wx(w, o, [x for x in tfs if x in o and x in w]):.1e})'.rjust(20) if o else '-'.rjust(20)) + ' '
        P(row)
      P()
    P('== D. Transfer settings: pooled AUROC / AUPRC (seed means)')
    for st, (srcs, t) in SETS.items():
        P(f'{st:16s} ' + '  '.join(f'{k} {S[(st, k)][1]:.4f}/{S[(st, k)][2]:.4f}' for k in keys if (st, k) in S))
    P()
    P('== E. Supervised breakdown: within-TF on target TFs with a motif | without a motif | not labelled in any source cell (upper-cased symbols)')
    P('   values: motif alone / scRegNet paper / scRegNet tuned / W20 full / W20 MLP; p = chosen scRegNet vs motif')
    for st, (srcs, t) in SETS.items():
        D = cell(t); allt = sorted(lab_tfs[t]); src_sym = set().union(*[lab_sym[s] for s in srcs])
        subs = {'motif': [x for x in allt if D['has_m'][x]], 'nomotif': [x for x in allt if not D['has_m'][x]], 'notinsrc': [x for x in allt if sym[t][x] not in src_sym]}
        ks = [k for k in ('scr-paper', 'scr-tuned', 'w20', 'w20mlp') if (st, k) in S]
        cells_ = []
        for nm_, v in subs.items():
            vals = '/'.join(f'{m_(x, v):.3f}' for x in [U[(t, "motif")][0]] + [S[(st, k)][0] for k in ks])
            p = wx(S[(st, 'scr-chosen')][0], U[(t, 'motif')][0], v) if (st, 'scr-chosen') in S and len(v) > 1 else float('nan')
            cells_.append(f'{nm_} n={len(v)} {vals} p={p:.2g}')
        P(f'{st:16s} ' + '  |  '.join(cells_))
    P()
    P('== E2. Additive decoder check: within-TF AUROC of each scRegNet-motif run vs the same grid averaged over TF rows (one gene ranking for all TFs)')
    d = []
    for f in sorted(glob.glob(f'{OUT}/w22_*__scr-*.npz')):
        g = np.load(f); j = json.load(open(f[:-4] + '.json')); G = g['grid']
        d.append(abs(evaluate(j['tgt'], g['grid_tf'], G)['within'] - evaluate(j['tgt'], g['grid_tf'], np.broadcast_to(G.mean(0), G.shape))['within']))
    P(f'{len(d)} runs, max |difference| {max(d):.2e}: the score is u_TF term + u_gene term, so every TF gets the same gene ranking')
    P()
    P('== F. Leak audit')
    P('src/xcell_base.leak_audit() (run by --selfcheck at the start of every ada job) parses the module: load_labels() is called only in run_scr')
    P('(source cells: training loss and source validation), and no other code names a label file. Target labels are read only inside')
    P('src/xcell_gnn.evaluate(). PANDA and GENIE3 inputs, and scRegNet node features and graph, are built under xcell_gnn.no_labels(), which')
    P('raises on any read of Train_set.csv / Test_set.csv / Label.csv / BL--network.csv; run_scr asserts the target is not a source.')
    P()
    J = [json.load(open(f)) for f in glob.glob(f'{OUT}/w22_*.json') if 'smoke' not in f]
    gm = sum(j.get('minutes', 0) for j in J if j['name'].split('__')[-1].startswith('s4'))
    cm = sum(j.get('minutes', 0) for j in J if 'genie3' in j['name'] and j['name'].endswith(tuple(CELLS)) and '-hard' not in j['name'] and '-soft' not in j['name'])
    cm += sum(j.get('minutes', 0) for j in J if j['name'].startswith('w22_panda'))
    P(f'== Compute: {len(J)} saved grids; scRegNet runs {gm:.0f} GPU-process minutes (sum of per-run wall time, up to 4 sharing one GPU); '
      f'PANDA + GENIE3/GBM {cm:.0f} wall minutes (GENIE3/GBM on 12 CPU workers)')
    open('analysis/out/w22_baselines.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    report()
