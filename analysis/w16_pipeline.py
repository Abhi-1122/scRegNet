"""W16 curated pipeline: per cell, the GNN configuration chosen on validation (w16_choose.py), decoder weight decay 0 or not, the uniform second TF summary (hHEP, mHSC-E only),
a within-TF rank blend with the DC-SBM (weight from {0, .25, .5, .75}), and seed averaging; every choice on validation only. Compared with the scRegNet published recipe
and with the chosen configuration alone. Run: python3.12 analysis/w16_pipeline.py -> analysis/out/w16_pipeline.txt"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, 'analysis'); sys.path.insert(0, '.')
import w3_report as w
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from src.pair_gnn import within_tf, within_tf8

P = w.OUT; ST = 'analysis/out/stat/'
CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']; SEEDS = (42, 43, 44); WS = (0.0, 0.25, 0.5, 0.75); QS_CELLS = ('hHEP', 'mHSC-E')
RECIPE = {'hESC': 'f02_A_paper', 'hHEP': 'w4_hHEP_A_paper', 'mHSC-E': 'w4_mHSC-E_A_paper', 'mESC': 'w4_mESC_A_paper'}
TESTBEST = {'hESC': 'w5_nphad_gf', 'hHEP': 'w11_grid_hHEP_npanda_had_batch_gf1', 'mHSC-E': 'w8_mHSC-E_gcngf', 'mESC': 'w11_grid_mESC_sage_had_none_gf1'}
CFGTXT = {'hESC': 'GCN 1 layer, Hadamard, batch split, Geneformer', 'hHEP': 'GCN 1 layer, Hadamard, batch split, Geneformer',
          'mHSC-E': 'neural PANDA 2 layers, Hadamard, batch split, Geneformer', 'mESC': 'neural PANDA 2 layers, Hadamard, no split, Geneformer'}
J = lambda n: json.load(open(f'{P}{n}.json'))
ex = lambda n: os.path.exists(f'{P}{n}.npz') and os.path.exists(f'{P}{n}.json')
L = []
out = lambda s='': (L.append(s), print(s))
ms = w.ms


def tfrank(s, tf):
    r = np.empty(len(s))
    for t in np.unique(tf):
        m = tf == t; r[m] = rankdata(s[m]) / m.sum()
    return r


def blend(g, d, tf, wt):
    """(within-TF rank blend, global rank blend): the first drives within-TF metrics, the second pooled AUROC/AUPRC (a within-TF rank would drop the TF base rate)"""
    if wt == 0: return g, g
    return (1 - wt) * tfrank(g, tf) + wt * tfrank(d, tf), (1 - wt) * rankdata(g) + wt * rankdata(d)


def met(y, tf, g, d, wt):
    a, b = blend(g, d, tf, wt)
    return dict(auc=roc_auc_score(y, b), aupr=average_precision_score(y, b), within=within_tf(y, a, tf), within8=within_tf8(y, a, tf)), w.per_tf(tf, y, a)


def unit_eval(units, tf, y, wt, ens):
    """units: list of (gnn logit, dcsbm logit) per seed. ens: average the logits first (one score), else score every seed separately.
    returns list of metric dicts and per-TF AUROC averaged over the units"""
    if ens: units = [(np.mean([u[0] for u in units], 0), np.mean([u[1] for u in units], 0))]
    R = [met(y, tf, g, d, wt) for g, d in units]
    pt = {k: float(np.mean([r[1][k] for r in R])) for k in set.intersection(*[set(r[1]) for r in R])}
    return [r[0] for r in R], pt


def fmt(nm, M):
    return f'{nm:46s}' + ''.join(f"{ms([m[k] for m in M]) if len(M) > 1 else f'{M[0][k]:.4f}':>17s}" for k in ('auc', 'aupr', 'within', 'within8'))


# ---------------------------------------------------------------- choices (validation only)
vmean = lambda c, v, k='within': np.mean([J(f'w16_{c}_{v}_s{s}')['val'][k] for s in SEEDS])
have = lambda c, v: all(ex(f'w16_{c}_{v}_s{s}') for s in SEEDS)
CH, W = {}, {}
out('W16 curated pipeline. Standard split, seeds 42-44; every choice below uses validation numbers only (best-epoch validation slice of the training rows; test is read only for the tables).')
out('\n== Choices on validation (mean over seeds 42-44; criterion = validation within-TF AUROC, validation AUROC shown for reference)')
for c in CELLS:
    if not have(c, 'base') or not have(c, 'dwd0'): out(f'{c}: arms missing'); continue
    wd = max(['base', 'dwd0'], key=lambda v: vmean(c, v))
    var = wd
    txt = f"{c:7s}wd: base val within {vmean(c, 'base'):.4f} (AUROC {vmean(c, 'base', 'auc'):.4f})  decoder wd 0 {vmean(c, 'dwd0'):.4f} ({vmean(c, 'dwd0', 'auc'):.4f}) -> {'decoder wd 0' if wd == 'dwd0' else 'wd 1e-3'}"
    if c in QS_CELLS and have(c, 'qs') and have(c, 'qsdwd0'):
        q = 'qsdwd0' if wd == 'dwd0' else 'qs'
        use = vmean(c, q) > vmean(c, wd)
        txt += f"; second summary: without {vmean(c, wd):.4f} with {vmean(c, q):.4f} (AUROC {vmean(c, wd, 'auc'):.4f} / {vmean(c, q, 'auc'):.4f}) -> {'use' if use else 'skip'}"
        var = q if use else wd
    CH[c] = dict(wd=wd, var=var, qs=var.startswith('qs'))
    out(txt)


# ---------------------------------------------------------------- loaders
def gnn_test(c, v, s):
    z = np.load(f'{P}w16_{c}_{v}_s{s}.npz'); return z['logit'], z['tf'], z['y'], z['tgt']


def dc_test(c, s): return np.load(f'{ST}{c}_dcsbm_learned_s{s}.npz')


def dc_val(c, s): return np.load(f'{P}w16_dcv_std_{c}_s{s}.npz')


def gnn_val(c, v, s, tf, tgt):
    z = np.load(f'{P}w16_{c}_{v}_s{s}_sel.npz'); ix = {t: i for i, t in enumerate(z['tfu'])}
    return z['grid'][[ix[t] for t in tf], tgt]


def test_units(c, v):
    U = []
    for s in SEEDS:
        g, tf, y, tgt = gnn_test(c, v, s); d = dc_test(c, s)
        assert (d['tf'] == tf).all() and (d['tgt'] == tgt).all() and (d['y'] == y).all()
        U.append((g, d['logit']))
    return U, tf, y


def val_units(c, v):
    U = []
    for s in SEEDS:
        d = dc_val(c, s); U.append((gnn_val(c, v, s, d['tf'], d['tgt']), d['logit']))
    return U, d['tf'], d['y']


out('\n== Blend weight w (DC-SBM share of the within-TF rank average) chosen on validation; the DC-SBM validation scores come from a fit without the validation rows')
for c in CH:
    if not all(os.path.exists(f'{P}w16_dcv_std_{c}_s{s}.npz') and os.path.exists(f'{P}w16_{c}_{CH[c]["var"]}_s{s}_sel.npz') for s in SEEDS): out(f'{c}: validation scores missing'); continue
    U, tf, y = val_units(c, CH[c]['var'])
    # sanity: the saved best-epoch grid reproduces the run's recorded validation within-TF
    chk = np.mean([within_tf(y, g, tf) for g, _ in U]); rec = vmean(c, CH[c]['var'])
    sc = {wt: np.mean([m['within'] for m in unit_eval(U, tf, y, wt, False)[0]]) for wt in WS}
    dcw = np.mean([within_tf(y, d, tf) for _, d in U])
    W[c] = max(WS, key=lambda k: sc[k])
    out(f"{c:7s}" + '  '.join(f'w={k}: {v:.4f}' for k, v in sc.items()) + f'   -> w = {W[c]}   (DC-SBM alone {dcw:.4f}; grid check {chk:.4f} vs recorded {rec:.4f})')

# ---------------------------------------------------------------- results per cell
RES = {}
from scipy.stats import wilcoxon
def pdiff(A, B):
    ks = sorted(set(A) & set(B)); d = np.array([A[k] - B[k] for k in ks])
    return float(d.mean()), (wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
VER = {}
RES_g = lambda arms: np.mean([m['within'] for m in arms[('gnn', 'seed')][0]])
RES_d = lambda arms: np.mean([m['within'] for m in arms[('dc', 'seed')][0]])
for c in CH:
    if c not in W: continue
    out(f'\n==================== {c}: {CFGTXT[c]}; decoder wd {"0" if CH[c]["wd"] == "dwd0" else "1e-3"}; second summary {"yes" if CH[c]["qs"] else "no"}; blend w = {W[c]}')
    base_v = CH[c]['wd'] if not CH[c]['qs'] else None
    Uf, tf, y = test_units(c, CH[c]['var'])
    Ub, _, _ = test_units(c, 'base')
    rec = [np.load(f'{P}{RECIPE[c]}_s{s}.npz') for s in SEEDS]
    assert all((r['tf'] == tf).all() and (r['y'] == y).all() for r in rec)
    Ur = [(r['logit'], r['logit']) for r in rec]
    ref = None
    if all(ex(f'{TESTBEST[c]}_s{s}') for s in SEEDS):
        zr = [np.load(f'{P}{TESTBEST[c]}_s{s}.npz') for s in SEEDS]; ref = [(z['logit'], z['logit']) for z in zr]
    arms = {}
    for ens in (False, True):
        k = 'ens' if ens else 'seed'
        arms[('recipe', k)] = unit_eval(Ur, tf, y, 0, ens)
        arms[('single', k)] = unit_eval(Ub, tf, y, 0, ens)           # chosen configuration, wd 1e-3, no extras, no blend
        arms[('gnn', k)] = unit_eval(Uf, tf, y, 0, ens)              # chosen GNN variant (wd / second summary choices), no blend
        arms[('pipe', k)] = unit_eval(Uf, tf, y, W[c], ens)          # full pipeline
        if ref: arms[('testbest', k)] = unit_eval(ref, tf, y, 0, ens)
        arms[('dc', k)] = unit_eval([(u[1], u[1]) for u in Uf], tf, y, 0, ens)
        oth = {'base': 'dwd0', 'dwd0': 'base', 'qs': 'qsdwd0', 'qsdwd0': 'qs'}[CH[c]['var']]
        Uo, _, _ = test_units(c, oth); arms[('otherwd', k)] = unit_eval(Uo, tf, y, W[c], ens)
        if c in QS_CELLS and have(c, 'qs'):
            wd_ = CH[c]['wd']; q_ = 'qs' if wd_ == 'base' else 'qsdwd0'
            Uq, _, _ = test_units(c, q_); Un, _, _ = test_units(c, wd_)
            arms[('with_qs', k)] = unit_eval(Uq, tf, y, W[c], ens); arms[('no_qs', k)] = unit_eval(Un, tf, y, W[c], ens)
    out(f"{'model':46s}{'AUROC':>17s}{'AUPRC':>17s}{'within-TF':>17s}{'within>=8':>17s}")
    names = [('recipe', 'scRegNet published recipe'), ('single', 'best single model (chosen config alone)'), ('gnn', 'chosen GNN variant, no blend'), ('dc', 'DC-SBM alone'), ('pipe', 'PIPELINE'), ('testbest', 'ref: test-picked grid best (optimistic)')]
    for k, lab in (('seed', 'per-seed mean'), ('ens', 'seed ensemble')):
        out(f'-- {lab} (seeds 42-44)')
        for key, nm in names:
            if (key, k) in arms: out(fmt(nm, arms[(key, k)][0]))
        for key, nm in (('recipe', 'recipe'), ('single', 'best single model')):
            out(w.paired(arms[('pipe', k)][1], arms[(key, k)][1], f'pipeline - {nm} ({lab})'))
        if ('testbest', k) in arms: out(w.paired(arms[('pipe', k)][1], arms[('testbest', k)][1], f'pipeline - test-picked grid best ({lab})'))
        out(w.paired(arms[('single', k)][1], arms[('recipe', k)][1], f'best single model - recipe ({lab})'))
    out('-- marginal effects in the pipeline context (within-TF; paired per-TF Wilcoxon)')
    for k, lab in (('seed', 'per-seed mean'), ('ens', 'seed ensemble')):
        out(w.paired(arms[('pipe', k)][1], arms[('gnn', k)][1], f'DC-SBM blend (w={W[c]} vs 0) [{lab}]'))
        out(w.paired(arms[('pipe', k)][1], arms[('otherwd', k)][1], f'chosen decoder wd vs the other value [{lab}]'))
        if ('with_qs', k) in arms: out(w.paired(arms[('with_qs', k)][1], arms[('no_qs', k)][1], f'second summary (with vs without) [{lab}]'))
    out(w.paired(arms[('pipe', 'ens')][1], arms[('pipe', 'seed')][1], 'seed ensemble vs per-seed mean (pipeline)'))
    out(w.paired(arms[('recipe', 'ens')][1], arms[('recipe', 'seed')][1], 'seed ensemble vs per-seed mean (recipe)'))
    out('   same-seed within-TF gaps pipeline - recipe: ' + ', '.join(f'{a["within"] - b["within"]:+.4f}' for a, b in zip(arms[('pipe', 'seed')][0], arms[('recipe', 'seed')][0])))
    from scipy.stats import spearmanr
    sp = np.mean([np.mean([spearmanr(g[tf == t], d[tf == t]).statistic for t in np.unique(tf)]) for g, d in Uf])
    out(f'   mean within-TF Spearman between the chosen GNN scores and the DC-SBM scores (test rows): {sp:.3f}; within-TF AUROC GNN {RES_g(arms):.4f}, DC-SBM {RES_d(arms):.4f}')
    RES[c] = {k: np.mean([m['within'] for m in v[0]]) for k, v in arms.items()}
    for k in ('seed', 'ens'):
        VER[(c, k)] = dict(pipe=RES[c][('pipe', k)], rec=RES[c][('recipe', k)], single=RES[c][('single', k)], vs_rec=pdiff(arms[('pipe', k)][1], arms[('recipe', k)][1]), vs_single=pdiff(arms[('pipe', k)][1], arms[('single', k)][1]))

# ---------------------------------------------------------------- hESC held-out genes and TFs
def ho_per_tf(zs):
    d = {}
    for tf, y, s in zs:
        for t, v in w.per_tf(tf, y, s).items(): d.setdefault(t, []).append(v)
    return {t: float(np.mean(v)) for t, v in d.items()}


def ho_load(prot, pat, seeds=(42, 43)):
    """pat(f, s) -> run name; returns {(f, s): npz} (None if any missing)"""
    R = {}
    for f in range(5):
        for s in seeds:
            n = pat(f, s)
            if not os.path.exists(f'{P}{n}.npz'): return None
            R[(f, s)] = np.load(f'{P}{n}.npz')
    return R


def ho_eval(R, wt=0.0, D=None, ens=False, seeds=(42, 43)):
    """R: {(f, s): npz of GNN}. D: {(f, s): DC-SBM test npz}. returns (metric dicts per unit, per-TF mean)"""
    M, Z = [], []
    for f in range(5):
        us = [(R[(f, s)]['logit'], D[(f, s)]['logit'] if D else R[(f, s)]['logit']) for s in seeds]
        z = R[(f, seeds[0])]; tf, y = z['tf'], z['y']
        if ens: us = [(np.mean([u[0] for u in us], 0), np.mean([u[1] for u in us], 0))]
        for g, d in us:
            m, pt = met(y, tf, g, d, wt); M.append(m); a_, _ = blend(g, d, tf, wt); Z.append((tf, y, a_))
    return M, ho_per_tf(Z)


def ho_block(prot):
    out(f'\n==================== hESC held-out {prot}s (5 folds x seeds 42, 43; GCN 1 layer, Hadamard, batch split, Geneformer)')
    wdc = CH.get('hESC', {}).get('wd')
    if wdc is None: out('hESC choice missing'); return
    mine = lambda f, s: f'w16_ho_{prot}{f}_{wdc}_s{s}'
    oth = lambda f, s: f'w16_ho_{prot}{f}_{"dwd0" if wdc == "base" else "base"}_s{s}'
    R, Ro = ho_load(prot, mine), ho_load(prot, oth)
    if R is None: out('GNN runs missing'); return
    base = {'recipe': ho_load(prot, lambda f, s: f'ho_{prot}{f}_f01_A_tuned_s{s}'), 'hero': ho_load(prot, lambda f, s: f'w5_ho_{prot}{f}_hero_s{s}')}
    arms = {}
    D, wsel = None, 0.0
    if prot == 'gene':
        D = {(f, s): np.load(f'{ST}hESC_hog{f}_dcsbm_gf_soft_s{s}.npz') for f in range(5) for s in (42, 43)}
        for k in R: assert (D[k]['tf'] == R[k]['tf']).all() and (D[k]['tgt'] == R[k]['tgt']).all() and (D[k]['y'] == R[k]['y']).all()
        ok = all(os.path.exists(f'{P}w16_dcv_hog{f}_s{s}.npz') and os.path.exists(f'{P}w16_ho_gene{f}_{wdc}_s{s}_sel.npz') for f in range(5) for s in (42, 43))
        if ok:
            sc = {wt: [] for wt in WS}
            for f in range(5):
                for s in (42, 43):
                    dv = np.load(f'{P}w16_dcv_hog{f}_s{s}.npz'); zs = np.load(f'{P}w16_ho_gene{f}_{wdc}_s{s}_sel.npz'); ix = {t: i for i, t in enumerate(zs['tfu'])}
                    g = zs['grid'][[ix[t] for t in dv['tf']], dv['tgt']]
                    for wt in WS: sc[wt].append(met(dv['y'], dv['tf'], g, dv['logit'], wt)[0]['within'])
            wsel = max(WS, key=lambda k: np.mean(sc[k]))
            out('validation (held-out validation genes), within-TF of the blend, mean over 10 runs: ' + '  '.join(f'w={k}: {np.mean(v):.4f}' for k, v in sc.items()) + f'   -> w = {wsel}')
        else: out('validation scores missing for the blend')
    arms['PIPELINE (GNN' + (f' + DC-SBM w={wsel}' if D else ' alone') + ')'] = ho_eval(R, wsel, D)
    if D: arms['GNN alone (chosen decoder wd)'] = ho_eval(R); arms['DC-SBM Geneformer soft alone'] = ho_eval({k: dict(logit=D[k]['logit'], tf=D[k]['tf'], y=D[k]['y']) for k in D})
    arms['GNN, other decoder wd'] = ho_eval(Ro)
    for k, v in base.items():
        if v is not None: arms['scRegNet design, Adam-tuned (recipe baseline)' if k == 'recipe' else 'hero (SAGE, no Geneformer)'] = ho_eval(v)
    out(f"{'model':46s}{'AUROC':>17s}{'AUPRC':>17s}{'within-TF':>17s}{'within>=8':>17s}")
    for nm, (M, pt) in arms.items(): out(fmt(nm, M))
    pipe = next(iter(arms.values()))[1]
    for nm, (M, pt) in list(arms.items())[1:]: out(w.paired(pipe, pt, f'pipeline - {nm}'))
    if prot == 'gene':
        out('seed ensemble over the 2 seeds per fold:')
        for nm, rr, dd, ww in (('pipeline', R, D, wsel), ('recipe', base['recipe'], None, 0.0)):
            if rr is not None: out(fmt(nm + ' (ensemble)', ho_eval(rr, ww, dd, True)[0]))
        if base['recipe'] is not None: out(w.paired(ho_eval(R, wsel, D, True)[1], ho_eval(base['recipe'], 0.0, None, True)[1], 'pipeline ens - recipe ens'))
    RES['ho_' + prot] = {k: np.mean([m['within'] for m in v[0]]) for k, v in arms.items()}
    names_ = list(arms)
    VER[('ho_' + prot, 'seed')] = dict(pipe=RES['ho_' + prot][names_[0]], names=names_, stats={nm: pdiff(pipe, arms[nm][1]) for nm in names_[1:]}, means=RES['ho_' + prot])


ho_block('gene'); ho_block('tf')

# ---------------------------------------------------------------- sensitivity: the grid's test-picked configurations instead of the validation-chosen ones (same pipeline otherwise: wd 1e-3, no second summary, w from validation)
ALT = {'hESC': ('npbase', 'neural PANDA 2 layers'), 'hHEP': ('npbase', 'neural PANDA 2 layers'), 'mHSC-E': ('gcnbase', 'GCN 1 layer'), 'mESC': ('sagebase', 'SAGE 1 layer')}
out('\n==================== Sensitivity: the grid configuration picked on TEST in section 24 (hESC/hHEP neural PANDA, mHSC-E GCN, mESC SAGE no split) instead of the validation choice; decoder wd 1e-3, no second summary, w chosen on validation')
for c, (v, txt) in ALT.items():
    if not all(os.path.exists(f'{P}w16_{c}_{v}_s{s}_sel.npz') and os.path.exists(f'{P}w16_{c}_{v}_s{s}.npz') for s in SEEDS): out(f'{c}: arms missing'); continue
    Uv, tfv, yv = val_units(c, v)
    sc = {wt: np.mean([m['within'] for m in unit_eval(Uv, tfv, yv, wt, False)[0]]) for wt in WS}; wt = max(WS, key=lambda k: sc[k])
    Ut, tf, y = test_units(c, v); rec = [np.load(f'{P}{RECIPE[c]}_s{s}.npz') for s in SEEDS]; Ur = [(r['logit'], r['logit']) for r in rec]
    out(f'{c:7s}{txt}; validation within-TF of the blend ' + '  '.join(f'w={k}: {x:.4f}' for k, x in sc.items()) + f' -> w = {wt}; validation within-TF of this config alone {sc[0.0]:.4f} (the validation-chosen config: {vmean(c, CH[c]["var"]):.4f})')
    for ens, lab in ((False, 'per-seed mean'), (True, 'seed ensemble')):
        a0, a1, ar = unit_eval(Ut, tf, y, 0, ens), unit_eval(Ut, tf, y, wt, ens), unit_eval(Ur, tf, y, 0, ens)
        pm = unit_eval(test_units(c, CH[c]['var'])[0], tf, y, W[c], ens)
        out(f'   [{lab}] config alone {np.mean([m["within"] for m in a0[0]]):.4f}, + blend {np.mean([m["within"] for m in a1[0]]):.4f}, recipe {np.mean([m["within"] for m in ar[0]]):.4f}, main pipeline {np.mean([m["within"] for m in pm[0]]):.4f}')
        out(w.paired(a1[1], ar[1], f'   sensitivity pipeline - recipe [{lab}]'))
        out(w.paired(a1[1], a0[1], f'   sensitivity pipeline - its config alone [{lab}]'))
        out(w.paired(a1[1], pm[1], f'   sensitivity pipeline - main pipeline [{lab}]'))
        VER[('sens_' + c, 'ens' if ens else 'seed')] = dict(pipe=np.mean([m['within'] for m in a1[0]]), single=np.mean([m['within'] for m in a0[0]]), rec=np.mean([m['within'] for m in ar[0]]), vs_rec=pdiff(a1[1], ar[1]))

# ---------------------------------------------------------------- verdict
out('\n==================== VERDICT (rule: PASS if the pipeline beats the recipe on within-TF in all 4 cells and on hESC held-out genes and TFs, and is not below the best single model anywhere)')
fails = []
for k, lab in (('seed', 'per-seed mean'), ('ens', 'seed ensemble')):
    out(f'-- standard split, {lab}: within-TF pipeline / recipe / best single model; pipeline - recipe (p); pipeline - best single (p)')
    for c in CELLS:
        v = VER.get((c, k))
        if not v: continue
        out(f"  {c:7s} {v['pipe']:.4f} / {v['rec']:.4f} / {v['single']:.4f}   vs recipe {v['vs_rec'][0]:+.4f} (p={v['vs_rec'][1]:.3g})   vs best single {v['vs_single'][0]:+.4f} (p={v['vs_single'][1]:.3g})")
        if v['vs_rec'][0] <= 0: fails.append(f'{c} below the recipe ({lab}) by {v["vs_rec"][0]:+.4f}')
        if v['vs_single'][0] < 0: fails.append(f'{c} below the best single model ({lab}) by {v["vs_single"][0]:+.4f} (p={v["vs_single"][1]:.2g})')
for prot in ('gene', 'tf'):
    v = VER.get(('ho_' + prot, 'seed'))
    if not v: continue
    out(f"-- hESC held-out {prot}s, within-TF: pipeline {v['pipe']:.4f}; " + '; '.join(f"{nm} {v['means'][nm]:.4f} (diff {st[0]:+.4f}, p={st[1]:.3g})" for nm, st in v['stats'].items() if 'recipe' in nm or 'hero' in nm or 'GNN alone' in nm))
    for nm, st in v['stats'].items():
        if ('recipe' in nm or 'hero' in nm) and st[0] <= 0: fails.append(f'held-out {prot}s below {nm.split(" (")[0]} by {st[0]:+.4f} (p={st[1]:.2g})')
out('VERDICT: ' + ('PASS' if not fails else 'FAIL; ' + '; '.join(fails)))
open('analysis/out/w16_pipeline.txt', 'w').write('\n'.join(L) + '\n')
