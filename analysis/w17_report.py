"""W17 report: held-out TFs (popularity blend, Part 1 + Part 2) and held-out genes (DC-SBM Geneformer-soft blend) on all 4 cells, pipeline vs scRegNet design (Adam-tuned) vs hero.
Every choice (blend weight) is made on the protocol's validation split (3 held-out validation TFs / 10% held-out validation genes of the training rows), per fold from that fold's runs.
Run: python3.12 analysis/w17_report.py -> analysis/out/w17_heldout.txt"""
import json, os, sys
from types import SimpleNamespace as NS
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); sys.path.insert(0, '.')
import w3_report as w
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from src.pair_gnn import within_tf, within_tf8, prepare, heldout_split

P = w.OUT; ST = 'analysis/out/stat/'
CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']; SEEDS = (42, 43); WS = (0.0, 0.25, 0.5, 0.75, 1.0)
L = []
out = lambda s='': (L.append(s), print(s, flush=True))


def tfrank(s, tf):
    r = np.empty(len(s))
    for t in np.unique(tf):
        m = tf == t; r[m] = rankdata(s[m]) / m.sum()
    return r


def blend(g, d, tf, wt):
    """(within-TF rank blend, global rank blend): first for within-TF metrics, second for pooled AUROC/AUPRC"""
    if wt == 0: return g, g
    return (1 - wt) * tfrank(g, tf) + wt * tfrank(d, tf), (1 - wt) * rankdata(g) + wt * rankdata(d)


def met(y, tf, g, d, wt):
    a, b = blend(g, d, tf, wt)
    return dict(auc=roc_auc_score(y, b), aupr=average_precision_score(y, b), within=within_tf(y, a, tf), within8=within_tf8(y, a, tf)), a


def ptf(tf, y, s, minpos=1):
    return {int(t): roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if minpos <= y[tf == t].sum() < (tf == t).sum()}


def fmt(nm, M, tfm=True):
    return f'{nm:50s}' + ''.join(f"{w.ms([m[k] for m in M]):>17s}" for k in ('auc', 'aupr', 'within', 'within8'))


def pdiff(A, B):
    ks = sorted(set(A) & set(B)); d = np.array([A[k] - B[k] for k in ks])
    return float(d.mean()), (wilcoxon(d).pvalue if np.any(d != 0) else 1.0), int((d > 0).sum()), len(d)


def cmp_line(lab, A, B):
    m, p, k, n = pdiff(A, B); return f'  {lab:60s} {m:+.4f}  better {k}/{n}  p={p:.3g}'


def nm_run(cell, prot, f, arm, s, pipe_gene_hesc='dwd0'):
    """run name of an arm; hESC reuses existing runs"""
    if cell == 'hESC':
        return {'tuned': f'ho_{prot}{f}_f01_A_tuned_s{s}', 'hero': f'w5_ho_{prot}{f}_hero_s{s}',
                'pipe': f'w16_ho_gene{f}_dwd0_s{s}' if prot == 'gene' else f'w17_hESC_ho_tf{f}_pipe_s{s}'}[arm]
    return f'w17_{cell}_ho_{prot}{f}_{arm}_s{s}'


def load_runs(cell, prot, arm):
    R = {}
    for f in range(5):
        for s in SEEDS:
            n = nm_run(cell, prot, f, arm, s)
            if not os.path.exists(f'{P}{n}.npz'): return None
            R[(f, s)] = np.load(f'{P}{n}.npz')
    return R


_PD = {}
def pop_data(cell, f):
    """gene in-degree among TRAINING positives. test: all training rows of the TF protocol fold; validation: training rows minus the 3 validation TFs (their edges excluded)"""
    if (cell, f) in _PD: return _PD[(cell, f)]
    path = f'data/{cell}/TFs+500/'
    Lb = pd.concat([pd.read_csv(path + 'Train_set.csv', index_col=0), pd.read_csv(path + 'Test_set.csv', index_col=0)], ignore_index=True)
    n = len(pd.read_csv(path + 'Target.csv', index_col=0))
    tr_, te_ = heldout_split(Lb, NS(protocol='tf', fold=f), n)
    deg_te = np.bincount(tr_[tr_.Label == 1].Target.values, minlength=n).astype(float)
    D = prepare(NS(cell_type=cell, num_TF='500', scfm_folder='scFM', data_folder='data', protocol='tf', fold=f, split_seed=0, emb_feat='none', feat='full', scfm=0, val_frac=0.1), 'cpu')
    tr, tri, vai = D['tr'], D['tri'], D['vai']
    deg_va = np.bincount(tr.iloc[tri][tr.iloc[tri].Label == 1].Target.values, minlength=n).astype(float)
    v = tr.iloc[vai]
    assert set(v.TF) .isdisjoint(set(tr.iloc[tri].TF))
    _PD[(cell, f)] = (np.log1p(deg_te), np.log1p(deg_va), v.TF.values, v.Target.values, v.Label.values)
    return _PD[(cell, f)]


def gnn_val_tf(cell, f, s, vtf, vtgt):
    z = np.load(f'{P}{nm_run(cell, "tf", f, "pipe", s)}_sel.npz'); ix = {t: i for i, t in enumerate(z['tfu'])}
    return z['grid'][[ix[t] for t in vtf], vtgt]


def tf_block(cell):
    out(f'\n==================== {cell} held-out TFs (5 folds x seeds 42, 43): pipeline = chosen GNN + gene-popularity rank blend; w in {WS} chosen on the 3 validation TFs of each fold')
    R = {a: load_runs(cell, 'tf', a) for a in ('pipe', 'tuned', 'hero')}
    if R['pipe'] is None: out('pipeline runs missing'); return None
    # validation scores
    VS = {}  # (f, wt) -> mean over seeds of validation within-TF of the blend
    for f in range(5):
        pte, pva, vtf, vtgt, vy = pop_data(cell, f)
        for wt in WS:
            VS[(f, wt)] = np.mean([met(vy, vtf, gnn_val_tf(cell, f, s, vtf, vtgt), pva[vtgt], wt)[0]['within'] for s in SEEDS])
        # sanity: grid reproduces the recorded validation within-TF
        chk = np.mean([met(vy, vtf, gnn_val_tf(cell, f, s, vtf, vtgt), pva[vtgt], 0)[0]['within'] for s in SEEDS])
        rec = np.mean([json.load(open(f'{P}{nm_run(cell, "tf", f, "pipe", s)}.json'))['val']['within'] for s in SEEDS])
        assert abs(chk - rec) < 2e-3, (cell, f, chk, rec)
    wf = {f: max(WS, key=lambda k: VS[(f, k)]) for f in range(5)}
    wpool = max(WS, key=lambda k: np.mean([VS[(f, k)] for f in range(5)]))
    out('validation within-TF of the blend (mean over the 2 seeds), per fold:')
    out(f"{'fold':6s}" + ''.join(f'{"w=" + str(k):>10s}' for k in WS) + '   chosen w')
    for f in range(5): out(f'{f:<6d}' + ''.join(f'{VS[(f, k)]:10.4f}' for k in WS) + f'   {wf[f]}')
    out(f'{"mean":6s}' + ''.join(f'{np.mean([VS[(f, k)] for f in range(5)]):10.4f}' for k in WS) + f'   pooled choice {wpool}')

    def ev(wsel):
        M, Z = [], {}
        for f in range(5):
            pte, *_ = pop_data(cell, f); wt = wsel(f)
            for s in SEEDS:
                z = R['pipe'][(f, s)]; m, a = met(z['y'], z['tf'], z['logit'], pte[z['tgt']], wt); M.append(m)
                for k, v in ((1, ptf(z['tf'], z['y'], a, 1)), (8, ptf(z['tf'], z['y'], a, 8))): Z.setdefault(k, {}).setdefault(None, []).append(v)
        return M, {k: avg(v[None]) for k, v in Z.items()}

    def avg(dl): return {t: float(np.mean([d[t] for d in dl if t in d])) for t in set().union(*dl)}

    def ev_other(RR):
        M, Z = [], {1: [], 8: []}
        for (f, s), z in RR.items():
            M.append(dict(auc=roc_auc_score(z['y'], z['logit']), aupr=average_precision_score(z['y'], z['logit']), within=within_tf(z['y'], z['logit'], z['tf']), within8=within_tf8(z['y'], z['logit'], z['tf'])))
            for k in Z: Z[k].append(ptf(z['tf'], z['y'], z['logit'], k))
        return M, {k: avg(v) for k, v in Z.items()}

    def ev_pop():
        M, Z = [], {1: [], 8: []}
        for f in range(5):
            pte, *_ = pop_data(cell, f)
            for s in SEEDS:
                z = R['pipe'][(f, s)]; g = pte[z['tgt']]
                M.append(dict(auc=roc_auc_score(z['y'], g), aupr=average_precision_score(z['y'], g), within=within_tf(z['y'], g, z['tf']), within8=within_tf8(z['y'], g, z['tf'])))
                for k in Z: Z[k].append(ptf(z['tf'], z['y'], g, k))
        return M, {k: avg(v) for k, v in Z.items()}

    arms = {f'PIPELINE (GNN + popularity, w chosen per fold on validation)': ev(lambda f: wf[f]),
            f'  w chosen on validation pooled over folds (w={wpool})': ev(lambda f: wpool),
            'GNN alone (w=0)': ev(lambda f: 0.0), 'popularity alone (w=1)': ev_pop(),
            'sensitivity: fixed w=0.5 (NOT validation-chosen)': ev(lambda f: 0.5)}
    for a, nm in (('tuned', 'scRegNet design, Adam-tuned'), ('hero', 'hero (SAGE, no Geneformer)')):
        if R[a] is not None: arms[nm] = ev_other(R[a])
    out('\n' + f"{'model':50s}{'AUROC':>17s}{'AUPRC':>17s}{'within-TF':>17s}{'within>=8':>17s}")
    for nm, (M, _) in arms.items(): out(fmt(nm, M))
    names = list(arms); pipe = arms[names[0]]
    out('chosen w per fold: ' + ', '.join(f'fold {f}: {wf[f]}' for f in range(5)))
    for nm in names[1:]:
        for k, lab in ((1, 'all TFs'), (8, 'TFs with >= 8 test positives')):
            out(cmp_line(f'pipeline - {nm.strip()} [{lab}]', pipe[1][k], arms[nm][1][k]))
    for k, lab in ((1, 'all TFs'), (8, 'TFs with >= 8 test positives')):
        out(cmp_line(f'popularity alone - scRegNet design, Adam-tuned [{lab}]', arms['popularity alone (w=1)'][1][k], arms['scRegNet design, Adam-tuned'][1][k]))
    return dict(arms={nm: np.mean([m['within'] for m in v[0]]) for nm, v in arms.items()}, arms8={nm: np.mean([m['within8'] for m in v[0]]) for nm, v in arms.items()},
                full={nm: {k: np.mean([m[k] for m in v[0]]) for k in ('auc', 'aupr', 'within', 'within8')} for nm, v in arms.items()},
                names=names, tests={nm: {k: pdiff(pipe[1][k], arms[nm][1][k]) for k in (1, 8)} for nm in names[1:]}, wf=wf, wpool=wpool)


def gene_block(cell):
    out(f'\n==================== {cell} held-out genes (5 folds x seeds 42, 43): pipeline = chosen GNN + DC-SBM (Geneformer soft blocks) rank blend; w in (0, .25, .5, .75) chosen on the validation genes, pooled over the 10 runs')
    R = {a: load_runs(cell, 'gene', a) for a in ('pipe', 'tuned', 'hero')}
    if R['pipe'] is None: out('pipeline runs missing'); return None
    dct = lambda f, s: np.load(f'{ST}{cell}_hog{f}_dcsbm_gf_soft_s{s}.npz')
    dcv = lambda f, s: np.load(f'{P}{"w16_dcv_hog" + str(f) if cell == "hESC" else "w17_dcv_" + cell + "_hog" + str(f)}_s{s}.npz')
    D = {k: dct(*k) for k in R['pipe']}
    for k in D: assert (D[k]['tf'] == R['pipe'][k]['tf']).all() and (D[k]['tgt'] == R['pipe'][k]['tgt']).all() and (D[k]['y'] == R['pipe'][k]['y']).all()
    ws = (0.0, 0.25, 0.5, 0.75); sc = {k: [] for k in ws}; chk = []
    for f in range(5):
        for s in SEEDS:
            dv = dcv(f, s); z = np.load(f'{P}{nm_run(cell, "gene", f, "pipe", s)}_sel.npz'); ix = {t: i for i, t in enumerate(z['tfu'])}
            g = z['grid'][[ix[t] for t in dv['tf']], dv['tgt']]
            for wt in ws: sc[wt].append(met(dv['y'], dv['tf'], g, dv['logit'], wt)[0]['within'])
            chk.append(within_tf(dv['y'], dv['logit'], dv['tf']))
    wsel = max(ws, key=lambda k: np.mean(sc[k]))
    out('validation within-TF of the blend, mean over 10 runs: ' + '  '.join(f'w={k}: {np.mean(v):.4f}' for k, v in sc.items()) + f'   -> w = {wsel}   (DC-SBM alone {np.mean(chk):.4f})')

    def ev(wt, dc_only=False):
        M, Z = [], {1: [], 8: []}
        for (f, s), z in R['pipe'].items():
            g = D[(f, s)]['logit'] if dc_only else z['logit']; m, a = met(z['y'], z['tf'], g, D[(f, s)]['logit'], 0.0 if dc_only else wt); M.append(m)
            for k in Z: Z[k].append(ptf(z['tf'], z['y'], a, k))
        return M, {k: {t: float(np.mean([d[t] for d in v if t in d])) for t in set().union(*v)} for k, v in Z.items()}

    def ev_other(RR):
        M, Z = [], {1: [], 8: []}
        for z in RR.values():
            M.append(dict(auc=roc_auc_score(z['y'], z['logit']), aupr=average_precision_score(z['y'], z['logit']), within=within_tf(z['y'], z['logit'], z['tf']), within8=within_tf8(z['y'], z['logit'], z['tf'])))
            for k in Z: Z[k].append(ptf(z['tf'], z['y'], z['logit'], k))
        return M, {k: {t: float(np.mean([d[t] for d in v if t in d])) for t in set().union(*v)} for k, v in Z.items()}

    arms = {f'PIPELINE (GNN + DC-SBM, w={wsel})': ev(wsel), 'GNN alone': ev(0.0), 'DC-SBM Geneformer soft alone': ev(0.0, True)}
    for a, nm in (('tuned', 'scRegNet design, Adam-tuned'), ('hero', 'hero (SAGE, no Geneformer)')):
        if R[a] is not None: arms[nm] = ev_other(R[a])
    out('\n' + f"{'model':50s}{'AUROC':>17s}{'AUPRC':>17s}{'within-TF':>17s}{'within>=8':>17s}")
    for nm, (M, _) in arms.items(): out(fmt(nm, M))
    names = list(arms); pipe = arms[names[0]]
    for nm in names[1:]: out(cmp_line(f'pipeline - {nm}', pipe[1][1], arms[nm][1][1]))
    return dict(arms={nm: np.mean([m['within'] for m in v[0]]) for nm, v in arms.items()},
                full={nm: {k: np.mean([m[k] for m in v[0]]) for k in ('auc', 'aupr', 'within', 'within8')} for nm, v in arms.items()},
                names=names, tests={nm: {1: pdiff(pipe[1][1], arms[nm][1][1])} for nm in names[1:]}, w=wsel)


if __name__ == '__main__':
    cells = sys.argv[1:] or CELLS
    RES = {('tf', c): tf_block(c) for c in cells}
    RES.update({('gene', c): gene_block(c) for c in cells})

    # ---------------------------------------------------------------- consolidated tables
    f4 = lambda d: f"{d['auc']:.4f} / {d['aupr']:.4f} / {d['within']:.4f}" + (f" / {d['within8']:.4f}" if 'within8' in d else '')
    out('\n==================== PART 2 TABLE: AUROC / AUPRC / within-TF / within>=8 (mean over 5 folds x seeds 42, 43)')
    for prot in ('gene', 'tf'):
        out(f'\n-- held-out {prot}s')
        for c in cells:
            r = RES.get((prot, c))
            if not r: continue
            for nm in r['names']:
                out(f'{c:7s}{nm.strip():62s}{f4(r["full"][nm])}')
            out('')
    out('\n==================== SUMMARY: pipeline vs the scRegNet baseline on every cell and protocol (within-TF AUROC; paired per-TF Wilcoxon)')
    out('standard split (W16, seeds 42-44, baseline = published scRegNet recipe, parsed from analysis/out/w16_pipeline.txt) and held-out (seeds 42, 43 x 5 folds, baseline = scRegNet design Adam-tuned)')
    import re
    txt = open('analysis/out/w16_pipeline.txt').read().split('==================== VERDICT')[1].split('-- standard split, seed ensemble')[0]
    std = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))) for m in re.finditer(r'  (hESC|hHEP|mHSC-E|mESC)\s+([\d.]+) / ([\d.]+) / [\d.]+\s+vs recipe ([+-][\d.]+) \(p=([\d.e+-]+)\)', txt)}
    def verdict(d, p): return 'beats' if d > 0 and p < 0.05 else ('loses' if d < 0 and p < 0.05 else 'ties')
    out(f"{'cell':8s}{'protocol':14s}{'pipeline':>10s}{'baseline':>10s}{'diff':>9s}{'p':>9s}  verdict   |  vs hero: diff (p)")
    for c in cells:
        if c in std:
            a, b, d, p = std[c]; out(f"{c:8s}{'standard':14s}{a:10.4f}{b:10.4f}{d:+9.4f}{p:9.3g}  {verdict(d, p)}")
        for prot in ('gene', 'tf'):
            r = RES.get((prot, c))
            if not r: continue
            k = r['names'][0]; tn = next(n for n in r['names'] if n.startswith('scRegNet design')); hn = next(n for n in r['names'] if n.startswith('hero'))
            d, p, kk, nn = r['tests'][tn][1]; dh, ph, _, _ = r['tests'][hn][1]
            out(f"{c:8s}{'held-out ' + prot + 's':14s}{r['arms'][k]:10.4f}{r['arms'][tn]:10.4f}{d:+9.4f}{p:9.3g}  {verdict(d, p)}   |  {dh:+.4f} ({ph:.2g}) {verdict(dh, ph)}")
    open('analysis/out/w17_heldout.txt', 'w').write('\n'.join(L) + '\n')
