"""W20 learned PANDA across cell types: job lists and report.

  python3.12 analysis/w20_xcell_report.py --jobs 1   -> analysis/jobs/w20_p1.txt (grids for selection)
  python3.12 analysis/w20_xcell_report.py --jobs 2   -> analysis/jobs/w20_p2.txt (ablations at the chosen config, CV seeds 43-44)
  python3.12 analysis/w20_xcell_report.py            -> analysis/out/w20_xcell.txt (from the saved score grids)
Every choice (config, epoch) is made on source-validation within-TF AUROC stored in the run json; the target is never used.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '.')
from src.xcell_gnn import CELLS, OUT, evaluate, cell, cv_folds, load_labels

SS = [('hESC', 'hHEP'), ('hHEP', 'hESC'), ('mESC', 'mHSC-E'), ('mHSC-E', 'mESC')]
SEEDS = [42, 43, 44]
LRS = ['1e-4', '3e-4', '1e-3']
MKS = [0, 50, 100, 200]
ABL = {'noP': '--rels M', 'noM': '--rels P', 'nomotifpf': '--motif_pf 0', 'plusC': '--rels MPC', 'pluscorr': '--corr_pf 1', 'rank': '--loss rank', 'noGF': '--gf 0'}


def settings():
    """setting id -> (source args, target, cv fold tfs or None)"""
    out = {f'ss_{s}-{t}': (f'--src {s} --tgt {t}', t) for s, t in SS}
    out.update({f'loo_{t}': (f"--src {' '.join(c for c in CELLS if c != t)} --tgt {t}", t) for t in CELLS})
    out.update({f'cv_{c}_f{k}': (f'--cv_cell {c} --fold {k}', c) for c in CELLS for k in range(5)})
    return out


def line(st, cfg, seed, extra):
    return f"--name w20_{st}__{cfg}__s{seed} {settings()[st][0]} --seed {seed} {extra}"


def runs():
    """{(setting, cfg): {seed: json}}"""
    R = defaultdict(dict)
    for f in glob.glob(f'{OUT}/w20_*.json'):
        j = json.load(open(f)); st, cfg, s = j['name'][4:].split('__')
        R[(st, cfg)][int(s[1:])] = j
    return R


def choose(R, st, prefix):
    """config with the best mean source-validation within-TF (over the seeds run for every candidate)"""
    cand = {cfg: v for (s, cfg), v in R.items() if s == st and cfg.startswith(prefix + '_')}
    common = set.intersection(*[set(v) for v in cand.values()])
    return max(cand, key=lambda c: np.mean([cand[c][s]['val_within'] for s in common]))


def jobs(phase):
    L = []
    if phase == 1:
        for st in settings():
            seeds = [42] if st.startswith('cv_') else SEEDS
            for s in seeds:
                L += [line(st, f'full_lr{lr}_mk{mk}', s, f'--lr {lr} --mk {mk}') for lr in LRS for mk in MKS]
                L += [line(st, f'mlp_lr{lr}', s, f'--model mlp --lr {lr}') for lr in LRS]
    else:
        R = runs()
        for st in settings():
            if st.startswith('cv_'):
                for pre in ('full', 'mlp'):
                    cfg = choose(R, st, pre); lr = cfg.split('_lr')[1].split('_')[0]
                    ex = f'--lr {lr}' + (f" --mk {cfg.split('_mk')[1]}" if pre == 'full' else ' --model mlp')
                    L += [line(st, cfg, s, ex) for s in (43, 44)]
            elif st.startswith('ss_'):
                cfg = choose(R, st, 'full'); lr, mk = cfg.split('_lr')[1].split('_mk')
                L += [line(st, f'abl-{k}_lr{lr}_mk{mk}', s, f'--lr {lr} --mk {mk} {v}') for k, v in ABL.items() for s in SEEDS]
    os.makedirs('analysis/jobs', exist_ok=True)
    open(f'analysis/jobs/w20_p{phase}.txt', 'w').write('\n'.join(L) + '\n')
    print(len(L), 'jobs -> analysis/jobs/w20_p%d.txt' % phase)


def gene_only(js):
    """diagnostic: each run's grid averaged over all TF rows (one gene ranking shared by every TF), scored like score_runs"""
    pt = defaultdict(list)
    for j in js:
        g = np.load(f"{OUT}/{j['name']}.npz"); gr = np.broadcast_to(g['grid'].mean(0), g['grid'].shape)
        for t, v in evaluate(j['tgt'], g['grid_tf'], gr, j['test_tfs'])['per_tf'].items(): pt[t].append(v)
    return {t: float(np.mean(v)) for t, v in pt.items()}


def score_runs(js):
    """seed runs -> per-TF AUROC averaged over seeds, mean pooled AUROC / AUPRC"""
    pt, au, ap = defaultdict(list), [], []
    for j in js:
        g = np.load(f"{OUT}/{j['name']}.npz")
        ev = evaluate(j['tgt'], g['grid_tf'], g['grid'], j['test_tfs'])
        for t, v in ev['per_tf'].items(): pt[t].append(v)
        au.append(ev['auc']); ap.append(ev['aupr'])
    return {t: float(np.mean(v)) for t, v in pt.items()}, float(np.mean(au)), float(np.mean(ap))


def base_scores(c):
    b = np.load(f'{OUT}/base_{c}.npz')
    out = {}
    for k in ('motif', 'corr', 'panda'):
        ev = evaluate(c, b['grid_tf'], b[k]); out[k] = (ev['per_tf'], ev['auc'], ev['aupr'])
    return out


def wx(a, b, tfs):
    d = np.array([a[t] - b[t] for t in tfs])
    if not np.any(d): return 1.0
    return float(wilcoxon(d).pvalue)


def m_(pt, tfs):
    tfs = [t for t in tfs if t in pt]
    return float(np.mean([pt[t] for t in tfs])) if tfs else float('nan')


def cv_popularity(c):
    """in-cell reference, not label-free: per test fold, score gene g by the fraction of the other folds' labelled TFs that target it"""
    L = load_labels(c); pt = {}
    for f in cv_folds(c):
        tr = L[~L.TF.isin(f)]; te = L[L.TF.isin(f)]
        pop = tr.groupby('Target').Label.mean().reindex(np.arange(cell(c)['n'])).fillna(0).values
        for t, g in te.groupby('TF'):
            if 0 < g.Label.sum() < len(g): pt[int(t)] = float(roc_auc_score(g.Label.values, pop[g.Target.values]))
    return pt


def report():
    R = runs(); S = settings(); out = []
    P = lambda *x: out.append(' '.join(str(v) for v in x))
    sym = {c: cell(c)['genes'] for c in CELLS}
    lab_tfs = {c: np.unique(load_labels(c).TF.values) for c in CELLS}
    lab_sym = {c: set(sym[c][lab_tfs[c]]) for c in CELLS}
    B = {c: base_scores(c) for c in CELLS}
    srcs = lambda st: S[st][0].split('--tgt')[0].replace('--src', '').split()
    P('W20 learned PANDA: complete network inference across cell types (src/xcell_gnn.py, analysis/w20_xcell_report.py)')
    P('within-TF AUROC = mean over target TFs with both classes of the AUROC of that TF\'s labelled genes; model per-TF AUROC averaged over seeds 42-44')
    P('configs and epochs chosen on source-validation within-TF (20% of source TFs held out); nothing chosen on the target. Wilcoxon: paired per-TF, two-sided, no multiplicity correction.')
    P()
    P('== Baselines on each cell, all labelled pairs (within all TFs | TFs with motif | TFs without motif ; pooled AUROC / AUPRC)')
    for c in CELLS:
        D = cell(c); hm = [t for t in lab_tfs[c] if D['has_m'][t]]; nm = [t for t in lab_tfs[c] if not D['has_m'][t]]
        b = np.load(f'{OUT}/base_{c}.npz'); inp = dict(zip(b['grid_tf'], b['in_panda']))
        P(f'{c}: labelled TFs {len(lab_tfs[c])} (motif {len(hm)}, none {len(nm)}; in PANDA network {sum(inp[t] for t in lab_tfs[c])})')
        for k, (pt, au, ap) in B[c].items():
            P(f'   {k:6s} {m_(pt, lab_tfs[c]):.4f} | {m_(pt, hm):.4f} | {m_(pt, nm):.4f} ; {au:.4f} / {ap:.4f}')
    P()
    T = {}  # (setting, model) -> (per_tf, auc, aupr, cfg)
    for st in S:
        if st.startswith('cv_'): continue
        for pre in ('full', 'mlp'):
            cfg = choose(R, st, pre); T[(st, pre)] = (*score_runs(R[(st, cfg)].values()), cfg)
        for k in ABL:
            js = [v for (s, cfg), v in R.items() if s == st and cfg.startswith(f'abl-{k}_')]
            if js: T[(st, k)] = (*score_runs(js[0].values()), [cfg for (s, cfg) in R if s == st and cfg.startswith(f'abl-{k}_')][0])
    P('== Chosen configs (source-validation within-TF, mean over seeds; best epochs per seed)')
    for st in [s for s in S if not s.startswith('cv_')]:
        for pre in ('full', 'mlp'):
            cfg = T[(st, pre)][3]; js = R[(st, cfg)]
            P(f'{st:16s} {pre:4s} {cfg:22s} val {np.mean([j["val_within"] for j in js.values()]):.4f} epochs {[js[s]["best_epoch"] for s in sorted(js)]}')
    P()
    P('== Main table: within-TF AUROC over all labelled target TFs; p = Wilcoxon full vs motif / PANDA / MLP')
    P(f'{"setting":16s} {"nTF":>3s} {"motif":>6s} {"corr":>6s} {"PANDA":>6s} {"MLP":>6s} {"full":>6s}   p(motif)  p(PANDA)  p(MLP)')
    for st in [s for s in S if not s.startswith('cv_')]:
        t = S[st][1]; tfs = sorted(T[(st, 'full')][0]); b = B[t]
        fm, ml = T[(st, 'full')][0], T[(st, 'mlp')][0]
        P(f'{st:16s} {len(tfs):3d} {m_(b["motif"][0], tfs):.4f} {m_(b["corr"][0], tfs):.4f} {m_(b["panda"][0], tfs):.4f} {m_(ml, tfs):.4f} {m_(fm, tfs):.4f}   '
          f'{wx(fm, b["motif"][0], tfs):.2g}  {wx(fm, b["panda"][0], tfs):.2g}  {wx(fm, ml, tfs):.2g}')
    P()
    P('== Diagnostic: the same full / MLP runs with each run\'s scores averaged over all TF rows (a gene-only ranking, no TF-specific part); within-TF')
    for st in [s for s in S if not s.startswith('cv_')]:
        fm = T[(st, 'full')]; ml = T[(st, 'mlp')]; tfs = sorted(fm[0])
        gf_, gm_ = gene_only(R[(st, fm[3])].values()), gene_only(R[(st, ml[3])].values())
        P(f'{st:16s} full {m_(fm[0], tfs):.4f} gene-only {m_(gf_, tfs):.4f} (p {wx(fm[0], gf_, tfs):.2g}) | MLP {m_(ml[0], tfs):.4f} gene-only {m_(gm_, tfs):.4f} (p {wx(ml[0], gm_, tfs):.2g})')
    P()
    P('== Pooled AUROC / AUPRC on all labelled target pairs')
    for st in [s for s in S if not s.startswith('cv_')]:
        t = S[st][1]; b = B[t]
        P(f'{st:16s} motif {b["motif"][1]:.4f}/{b["motif"][2]:.4f}  corr {b["corr"][1]:.4f}/{b["corr"][2]:.4f}  PANDA {b["panda"][1]:.4f}/{b["panda"][2]:.4f}  '
          f'MLP {T[(st, "mlp")][1]:.4f}/{T[(st, "mlp")][2]:.4f}  full {T[(st, "full")][1]:.4f}/{T[(st, "full")][2]:.4f}')
    P()
    P('== Breakdown: within-TF on target TFs with a motif | without a motif | not labelled in any source cell (motif / PANDA / MLP / full; p full vs MLP; pm = p full vs motif)')
    for st in [s for s in S if not s.startswith('cv_')]:
        t = S[st][1]; D = cell(t); b = B[t]; fm, ml = T[(st, 'full')][0], T[(st, 'mlp')][0]
        allt = sorted(fm); src_sym = set().union(*[lab_sym[s] for s in srcs(st)])
        subs = {'motif': [x for x in allt if D['has_m'][x]], 'nomotif': [x for x in allt if not D['has_m'][x]], 'notinsrc': [x for x in allt if sym[t][x] not in src_sym]}
        P(f'{st:16s} ' + '  |  '.join(f'{k} n={len(v)} {m_(b["motif"][0], v):.3f}/{m_(b["panda"][0], v):.3f}/{m_(ml, v):.3f}/{m_(fm, v):.3f} p={wx(fm, ml, v) if len(v) > 1 else float("nan"):.2g} pm={wx(fm, b["motif"][0], v) if len(v) > 1 else float("nan"):.2g}' for k, v in subs.items()))
    P()
    P('== Upper bound: within-cell 5-fold TF-disjoint CV (same model, config chosen per fold on seed-42 source-validation, seeds 42-44) vs transfer')
    P('CV popularity = fraction of the training-fold TFs that target the gene (uses in-cell labels; shows how much of the bound a per-gene ranking alone gives)')
    P(f'{"cell":7s} {"motif":>6s} {"PANDA":>6s} {"CV pop":>7s} {"CV MLP":>7s} {"CV full":>7s} {"same-sp":>7s} {"LOO":>6s}  p(CV full vs CV MLP)  p(CV full vs same-species full)')
    UB = {}
    for c in CELLS:
        pts = {}
        for pre in ('full', 'mlp'):
            pt = {}
            for k in range(5):
                cfg = choose(R, f'cv_{c}_f{k}', pre); pt.update(score_runs(R[(f'cv_{c}_f{k}', cfg)].values())[0])
            pts[pre] = pt
        UB[c] = pts
        ss = [s for s in S if s.startswith('ss_') and s.endswith('-' + c)][0]
        tfs = sorted(pts['full']); b = B[c]
        pop = cv_popularity(c)
        P(f'{c:7s} {m_(b["motif"][0], tfs):.4f} {m_(b["panda"][0], tfs):.4f} {m_(pop, tfs):.4f}  {m_(pts["mlp"], tfs):.4f}  {m_(pts["full"], tfs):.4f}  {m_(T[(ss, "full")][0], tfs):.4f}  {m_(T[("loo_" + c, "full")][0], tfs):.4f}  '
          f'{wx(pts["full"], pts["mlp"], tfs):.2g}  {wx(pts["full"], T[(ss, "full")][0], tfs):.2g}')
    P()
    P('== Ablations of the full model, same-species settings (within-TF all target TFs; delta vs full; p Wilcoxon vs full)')
    P(f'{"setting":16s} {"full":>6s} ' + ' '.join(f'{k:>20s}' for k in ABL))
    for st in [s for s in S if s.startswith('ss_')]:
        fm = T[(st, 'full')][0]; tfs = sorted(fm)
        cells_ = []
        for k in ABL:
            if (st, k) not in T: cells_.append(f'{"-":>20s}'); continue
            a = T[(st, k)][0]; cells_.append(f'{m_(a, tfs):.4f} {m_(a, tfs) - m_(fm, tfs):+.4f} ({wx(a, fm, tfs):.2g})'.rjust(20))
        P(f'{st:16s} {m_(fm, tfs):.4f} ' + ' '.join(cells_))
    P()
    P('== Leak audit')
    P('build_cell() runs under no_labels(): pd.read_csv / np.load / open raise on Train_set.csv, Test_set.csv, Label.csv, BL--network.csv; --selfcheck asserts the guard fires and every cell builds.')
    P('load_labels() is the only label reader; run() calls it for source cells (training loss; CV: test-fold TFs removed, asserted) and evaluate() for the target; transfer runs assert target not in sources.')
    P('Shared labelled TFs (upper-cased symbols) and shared genes between source(s) and target; result with shared labelled TFs removed is the "notinsrc" column above:')
    for st in [s for s in S if not s.startswith('cv_')]:
        t = S[st][1]; src_sym = set().union(*[lab_sym[s] for s in srcs(st)]); src_g = set().union(*[set(sym[s]) for s in srcs(st)])
        sh = sorted(lab_sym[t] & src_sym)
        P(f'{st:16s} shared labelled TFs {len(sh)}: {",".join(sh)}; shared genes {len(set(sym[t]) & src_g)} of {len(sym[t])}')
    P()
    mins = sum(j['minutes'] for v in R.values() for j in v.values())
    P(f'== Compute: {sum(len(v) for v in R.values())} runs, {mins:.0f} GPU-process minutes in total (sum of per-run wall time; 4 processes in parallel)')
    open('analysis/out/w20_xcell.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--jobs', type=int, default=0)
    a = p.parse_args()
    if a.jobs: jobs(a.jobs); sys.exit()
    report()
