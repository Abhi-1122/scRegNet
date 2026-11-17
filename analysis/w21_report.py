"""W21 two-view model across cell types: job lists and report.

  python3.12 analysis/w21_report.py --jobs 1   -> analysis/jobs/w21_p1.txt (grids of MLP, view A, view B incl. the co-expression graph; seeds 42-44)
  python3.12 analysis/w21_report.py --jobs 2   -> analysis/jobs/w21_p2.txt (fusion grids lr x mk x co-expression graph; seeds 42-44)
  python3.12 analysis/w21_report.py --jobs 3   -> analysis/jobs/w21_p3.txt (loss / regularisation ablations of the best fusion)
  python3.12 analysis/w21_report.py            -> analysis/out/w21_2view.txt (from the saved score grids in analysis/out/xcell2)
Every choice (config, epoch, best fusion) is made on source-validation within-TF AUROC stored in the run json; never on the target.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from src.xcell_gnn import CELLS, evaluate, cell, load_labels
from src.xcell_2view import OUT

SS = [('hESC', 'hHEP'), ('hHEP', 'hESC'), ('mESC', 'mHSC-E'), ('mHSC-E', 'mESC')]
SEEDS = [42, 43, 44]
LRS = ['1e-4', '3e-4', '1e-3']
MKS = [0, 50]
CX = [(k, s) for k in (5, 10, 20) for s in 'pa']  # co-expression mutual kNN k, p = positive r only, a = |r|
FUS = ['concat', 'gate', 'early']
ARMS = ['mlp', 'A', 'B'] + FUS
ABL = {'bce': '--loss bce', 'vd0': '--vdrop 0', 'c0.1': '--cons 0.1', 'c0.5': '--cons 0.5', 'c0.1t': '--cons 0.1 --cons_tgt 1', 'c0.5t': '--cons 0.5 --cons_tgt 1'}


def settings():
    out = {f'ss_{s}-{t}': (f'--src {s} --tgt {t}', t, [s]) for s, t in SS}
    out.update({f'loo_{t}': (f"--src {' '.join(c for c in CELLS if c != t)} --tgt {t}", t, [c for c in CELLS if c != t]) for t in CELLS})
    return out


def grid(arm):
    """arm -> {cfg name: args}"""
    G = {}
    for lr in LRS:
        if arm == 'mlp': G[f'mlp_lr{lr}'] = f'--model mlp --lr {lr}'
        if arm == 'A': G.update({f'A_lr{lr}_mk{mk}': f'--model A --lr {lr} --mk {mk}' for mk in MKS})
        if arm == 'B': G.update({f'B_lr{lr}_ck{k}{s}': f'--model B --lr {lr} --ck {k} --cabs {int(s == "a")}' for k, s in CX})
        if arm in FUS:
            G.update({f'{arm}_lr{lr}_mk{mk}_ck{k}{s}': f'--model {arm} --lr {lr} --mk {mk} --ck {k} --cabs {int(s == "a")}' for mk in MKS for k, s in CX})
    return G


def line(st, cfg, seed, extra):
    return f'--name w21_{st}__{cfg}__s{seed} {settings()[st][0]} --seed {seed} {extra}'


def runs(prefix='w21_'):
    R = defaultdict(dict)
    for f in glob.glob(f'{OUT}/{prefix}*.json'):
        j = json.load(open(f))
        if '__' not in j['name']: continue
        st, cfg, s = j['name'][4:].split('__')
        R[(st, cfg)][int(s[1:])] = j
    return R


def choose(R, st, arm):
    cand = {cfg: v for (s, cfg), v in R.items() if s == st and cfg.startswith(arm + '_')}
    common = set.intersection(*[set(v) for v in cand.values()])
    return max(cand, key=lambda c: np.mean([cand[c][s]['val_within'] for s in common]))


def val_mean(js):
    return float(np.mean([j['val_within'] for j in js.values()]))


def best_fusion(R):
    """the two-view fusion (concat or gate) with the higher source-validation within-TF, mean over the 8 settings"""
    v = {f: np.mean([val_mean(R[(st, choose(R, st, f))]) for st in settings()]) for f in ('concat', 'gate')}
    return max(v, key=v.get), v


def jobs(phase):
    L = []
    if phase in (1, 2):
        for st in settings():
            for s in SEEDS:
                for arm in (['mlp', 'A', 'B'] if phase == 1 else FUS):
                    L += [line(st, cfg, s, ex) for cfg, ex in grid(arm).items()]
    else:
        R = runs(); bf, _ = best_fusion(R)
        for st in settings():
            cfg = choose(R, st, bf); ex = grid(bf)[cfg]
            L += [line(st, f'abl-{k}_{cfg}', s, f'{ex} {v}') for k, v in ABL.items() for s in SEEDS]
    open(f'analysis/jobs/w21_p{phase}.txt', 'w').write('\n'.join(L) + '\n')
    print(len(L), f'jobs -> analysis/jobs/w21_p{phase}.txt')


def score_runs(js, out=OUT, tf_avg=False):
    """seed runs -> per-TF AUROC averaged over seeds, mean pooled AUROC / AUPRC over seeds;
    tf_avg: every TF gets the TF-averaged gene ranking (a pure gene prior made from the same grid)"""
    pt, au, ap = defaultdict(list), [], []
    for j in js:
        g = np.load(f"{out}/{j['name']}.npz"); G = g['grid']
        if tf_avg: G = np.broadcast_to(G.mean(0), G.shape)
        ev = evaluate(j['tgt'], g['grid_tf'], G)
        for t, v in ev['per_tf'].items(): pt[t].append(v)
        au.append(ev['auc']); ap.append(ev['aupr'])
    return {t: float(np.mean(v)) for t, v in pt.items()}, float(np.mean(au)), float(np.mean(ap))


def motif_scores(c):
    """motif_max alone (no motif -> below every scored pair), as the W20 baseline"""
    D = cell(c); Mx = D['Mx'][D['tf']]
    ev = evaluate(c, D['tf'], np.where(np.isnan(Mx), np.nanmin(D['Mx']) - 1, Mx))
    return ev['per_tf'], ev['auc'], ev['aupr']


def wx(a, b, tfs):
    d = np.array([a[t] - b[t] for t in tfs])
    return 1.0 if not np.any(d) else float(wilcoxon(d).pvalue)


def m_(pt, tfs):
    tfs = [t for t in tfs if t in pt]
    return float(np.mean([pt[t] for t in tfs])) if tfs else float('nan')


def w20_full():
    """W20's full model per setting, chosen exactly as its own report does (None if its runs are not there)"""
    try:
        import w20_xcell_report as w
        from src.xcell_gnn import OUT as O20
        R = w.runs(); out = {}
        for st in settings():
            cfg = w.choose(R, st, 'full'); js = R[(st, cfg)]
            out[st] = (*score_runs(js.values(), O20), cfg, val_mean(js))
        return out
    except Exception as e:
        print('W20 not available:', e); return None


def report():
    R = runs(); S = settings(); out = []
    P = lambda *x: out.append(' '.join(str(v) for v in x))
    sym = {c: cell(c)['genes'] for c in CELLS}
    lab_sym = {c: set(sym[c][np.unique(load_labels(c).TF.values)]) for c in CELLS}
    MO = {c: motif_scores(c) for c in CELLS}
    bf, bfv = best_fusion(R)
    T = {}  # (setting, model) -> (per_tf, auc, aupr, cfg, val)
    for st in S:
        for arm in ARMS:
            cfg = choose(R, st, arm); T[(st, arm)] = (*score_runs(R[(st, cfg)].values()), cfg, val_mean(R[(st, cfg)]))
        for k in ABL:
            c2 = f'abl-{k}_{T[(st, bf)][3]}'
            if (st, c2) in R: T[(st, k)] = (*score_runs(R[(st, c2)].values()), c2, val_mean(R[(st, c2)]))
    W20 = w20_full()
    P('W21 two-view model: motif + PPI graph (view A) and co-expression graph (view B), complete network inference across cell types')
    P('src/xcell_2view.py, analysis/w21_report.py. Same data, features, source-validation split and evaluation as W20 (src/xcell_gnn.py).')
    P('within-TF AUROC = mean over target TFs with both classes of that TF\'s AUROC over all its labelled pairs (Train_set + Test_set); per-TF AUROC averaged over seeds 42-44.')
    P('Every config, epoch and the best fusion chosen on source-validation within-TF (20% of source TFs held out); nothing chosen on the target.')
    P('Wilcoxon: paired per-TF, two-sided, no multiplicity correction. Loss TF-balanced BCE, view dropout 0.2 unless stated.')
    P()
    P(f'== Best fusion on source validation (mean over the 8 settings of the chosen config): ' + ', '.join(f'{k} {v:.4f}' for k, v in bfv.items()) + f' -> {bf}')
    P()
    P('== Chosen configs (source-validation within-TF, mean over seeds; best epoch per seed)')
    for st in S:
        for arm in ARMS:
            cfg = T[(st, arm)][3]; js = R[(st, cfg)]
            P(f'{st:16s} {arm:6s} {cfg:28s} val {T[(st, arm)][4]:.4f} epochs {[js[s]["best_epoch"] for s in sorted(js)]}')
    P()
    P('== Main table: within-TF AUROC over all labelled target TFs' + ('; W20 = W20 full model (plain BCE, view A graph only)' if W20 else ''))
    P(f'{"setting":16s} {"nTF":>3s} {"motif":>6s} {"MLP":>6s} {"A":>6s} {"B":>6s} {"concat":>6s} {"gate":>6s} {"early":>6s}' + (f' {"W20":>6s}' if W20 else ''))
    mean_rows = defaultdict(list)
    for st in S:
        t = S[st][1]; tfs = sorted(T[(st, 'A')][0])
        row = [m_(MO[t][0], tfs)] + [m_(T[(st, a)][0], tfs) for a in ARMS] + ([m_(W20[st][0], tfs)] if W20 else [])
        for i, v in enumerate(row): mean_rows[i].append(v)
        P(f'{st:16s} {len(tfs):3d} ' + ' '.join(f'{v:.4f}' for v in row))
    P(f'{"mean of 8":16s} {"":3s} ' + ' '.join(f'{np.mean(v):.4f}' for v in mean_rows.values()))
    P()
    P('== p-values (paired per-TF Wilcoxon) of each fusion against motif alone / view A only / MLP / view B only; sign = direction of the fusion minus the other')
    for st in S:
        t = S[st][1]; tfs = sorted(T[(st, 'A')][0]); cells_ = []
        for f in FUS:
            fm = T[(st, f)][0]
            cells_.append(f'{f} ' + '/'.join(f'{"+" if m_(fm, tfs) >= m_(o, tfs) else "-"}{wx(fm, o, tfs):.2g}' for o in (MO[t][0], T[(st, 'A')][0], T[(st, 'mlp')][0], T[(st, 'B')][0])))
        P(f'{st:16s} ' + '   '.join(cells_))
    P()
    P('== Pooled AUROC / AUPRC on all labelled target pairs')
    for st in S:
        t = S[st][1]
        P(f'{st:16s} motif {MO[t][1]:.4f}/{MO[t][2]:.4f}  ' + '  '.join(f'{a} {T[(st, a)][1]:.4f}/{T[(st, a)][2]:.4f}' for a in ARMS)
          + (f'  W20 {W20[st][1]:.4f}/{W20[st][2]:.4f}' if W20 else ''))
    P()
    P(f'== Breakdown: within-TF on target TFs with a motif | without a motif | not labelled in any source cell (motif / MLP / A / {bf}; p {bf} vs A)')
    for st in S:
        t = S[st][1]; D = cell(t); fm, A_ = T[(st, bf)][0], T[(st, 'A')][0]
        allt = sorted(A_); src_sym = set().union(*[lab_sym[s] for s in S[st][2]])
        subs = {'motif': [x for x in allt if D['has_m'][x]], 'nomotif': [x for x in allt if not D['has_m'][x]], 'notinsrc': [x for x in allt if sym[t][x] not in src_sym]}
        P(f'{st:16s} ' + '  |  '.join(f'{k} n={len(v)} {m_(MO[t][0], v):.3f}/{m_(T[(st, "mlp")][0], v):.3f}/{m_(A_, v):.3f}/{m_(fm, v):.3f} p={wx(fm, A_, v) if len(v) > 1 else float("nan"):.2g}' for k, v in subs.items()))
    P()
    P('== Source-validation vs target within-TF (val / target / gap = val - target), chosen config per arm')
    P(f'{"setting":16s} ' + ' '.join(f'{a:>22s}' for a in ARMS))
    gaps = defaultdict(list)
    for st in S:
        tfs = sorted(T[(st, 'A')][0]); cells_ = []
        for a in ARMS:
            v, tg = T[(st, a)][4], m_(T[(st, a)][0], tfs); gaps[a].append((v, tg)); cells_.append(f'{v:.3f}/{tg:.3f}/{v - tg:+.3f}'.rjust(22))
        P(f'{st:16s} ' + ' '.join(cells_))
    P(f'{"mean of 8":16s} ' + ' '.join(f'{np.mean([g[0] for g in gaps[a]]):.3f}/{np.mean([g[1] for g in gaps[a]]):.3f}/{np.mean([g[0] - g[1] for g in gaps[a]]):+.3f}'.rjust(22) for a in ARMS))
    if W20:
        P(f'{"W20 full":16s} ' + ' '.join(f'{st}: {W20[st][4]:.3f}/{m_(W20[st][0], sorted(T[(st, "A")][0])):.3f}' for st in S))
    P()
    P(f'== Ablations of the best fusion ({bf}) at its chosen config: within-TF all target TFs; delta vs {bf}; (p Wilcoxon vs {bf}); [source-val]')
    P('bce = plain BCE instead of TF-balanced; vd0 = no view dropout; cL = InfoNCE view consistency weight L on source graphs; cLt = also on the target graph (no target labels)')
    P(f'{"setting":16s} {bf:>14s} ' + ' '.join(f'{k:>28s}' for k in ABL))
    am = defaultdict(list)
    for st in S:
        fm = T[(st, bf)][0]; tfs = sorted(fm); cells_ = []
        am['base'].append(m_(fm, tfs))
        for k in ABL:
            if (st, k) not in T: cells_.append(f'{"-":>28s}'); continue
            a = T[(st, k)][0]; am[k].append(m_(a, tfs) - m_(fm, tfs))
            cells_.append(f'{m_(a, tfs):.4f} {m_(a, tfs) - m_(fm, tfs):+.4f} ({wx(a, fm, tfs):.2g}) [{T[(st, k)][4]:.3f}]'.rjust(28))
        P(f'{st:16s} {m_(fm, tfs):.4f} [{T[(st, bf)][4]:.3f}] ' + ' '.join(cells_))
    P(f'{"mean delta":16s} {np.mean(am["base"]):.4f}         ' + ' '.join(f'{np.mean(am[k]):+.4f}'.rjust(28) if am[k] else f'{"-":>28s}' for k in ABL))
    P()
    P('== Gene-prior check: within-TF of each chosen run vs the same grid averaged over TFs (one gene ranking for every TF); seed means')
    for st in S:
        tfs = sorted(T[(st, 'A')][0])
        P(f'{st:16s} ' + '  '.join(f'{a} {m_(T[(st, a)][0], tfs):.3f}/{m_(score_runs(R[(st, T[(st, a)][3])].values(), tf_avg=True)[0], tfs):.3f}' for a in ('mlp', 'A', 'B', 'concat', 'gate')))
    P()
    P('== Gate (gate fusion, chosen config): mean per-gene weight on view A over target genes, seed mean (1 = A only, 0 = B only)')
    P('  '.join(f'{st} {np.mean([np.load(f"{OUT}/{j["name"]}.npz")["gate"].mean() for j in R[(st, T[(st, "gate")][3])].values()]):.3f}' for st in S))
    P()
    P('== Leak audit')
    P('Source labels are loaded first (load_labels, training loss + source validation only). All graphs, features, training (incl. the target consistency term) and the target score grid run inside')
    P('xcell_gnn.no_labels(), which raises on any read of Train_set.csv, Test_set.csv, Label.csv, BL--network.csv; evaluate() then reads the target labels. Transfer runs assert target not in sources.')
    P('src/xcell_2view.py --selfcheck asserts: the guard fires; every cell\'s motif, PPI and co-expression graphs (all k, both signs) and features build under the guard; mutual kNN is symmetric with')
    P('at most k neighbours; a perfect grid scores within-TF and pooled AUROC 1; TF-balanced weights sum to 1 per TF; a 2-epoch source+target consistency run completes under the guard.')
    if W20:
        P()
        P('== W20 full model (chosen by its own report logic from analysis/out/xcell): ' + '  '.join(f'{st} {W20[st][3]}' for st in S))
        P('Differences in settings: W20 trains with plain BCE, motif graph mk in {0,50,100,200}, no co-expression, no view dropout; W21 view A uses TF-balanced BCE and mk in {0,50}.')
    P()
    mins = sum(j['minutes'] for v in R.values() for j in v.values())
    P(f'== Compute: {sum(len(v) for v in R.values())} runs, {mins:.0f} GPU-process minutes (sum of per-run wall time; 6 processes on one ada 2080 Ti)')
    open('analysis/out/w21_2view.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--jobs', type=int, default=0)
    a = p.parse_args()
    if a.jobs: jobs(a.jobs); sys.exit()
    report()
