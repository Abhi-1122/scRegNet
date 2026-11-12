"""W12 statistical-model report: regenerates analysis/out/w12_stat.txt from the saved scores in analysis/out/stat/. Run: python3.12 analysis/w12_stat_report.py"""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis')
import w3_report as w
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

ST, PAIR = 'analysis/out/stat/', 'analysis/out/pair/'
SEEDS = (42, 43, 44)
CELLS = ['hESC', 'hHEP', 'mESC', 'mHSC-E']
HERO = {'hESC': 'w3_hero', 'hHEP': 'm_hHEP_e_sage_d256', 'mESC': 'w4_mESC_hero', 'mHSC-E': 'w4_mHSC-E_hero'}  # SAGE, had, batch split, no Geneformer (w11_grid_<cell>_sage_had_batch_gf0 equals these runs; the grid files were never produced)
BESTG = {'hESC': 'w8_hESC_tuned', 'hHEP': 'w8_hHEP_gcngf', 'mESC': 'w8_mESC_tuned', 'mHSC-E': 'w8_mHSC-E_gcngf'}  # best GNN per cell (PROGRESS section 21)
FIX = ['dcsbm_gf_soft', 'dcsbm_gf_hard', 'dcsbm_ex_soft', 'dcsbm_ex_hard']
LEARN = ['dcsbm_learned', 'hpf', 'lsm']
ALL = ['pop_indeg', 'pop'] + FIX + [f + s for f in FIX for s in ('_omegaonly', '_thetaonly')] + LEARN
CAND = FIX + LEARN  # candidates for "best statistical model" (selected on validation only)
L = []
P = lambda s='': L.append(s)


def jl(tag, m, seeds=SEEDS):
    return [json.load(open(f'{ST}{tag}_{m}_s{s}.json')) for s in seeds if glob.glob(f'{ST}{tag}_{m}_s{s}.json')]


def zl(tag, m, seeds=SEEDS):
    return [np.load(f'{ST}{tag}_{m}_s{s}.npz') for s in seeds if glob.glob(f'{ST}{tag}_{m}_s{s}.npz')]


def tfm(tag, m, seeds=SEEDS, key=None):
    d = [w.per_tf(z['tf'], z['y'], z['logit']) for z in zl(tag, m, seeds)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))} if d else {}


def hero_tf(c, seeds=SEEDS):
    d = [w.per_tf(z['tf'], z['y'], z['logit']) for z in (np.load(f'{PAIR}{c}_s{s}.npz') for s in seeds)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))}


def hero_row(c, seeds=SEEDS):
    r = [json.load(open(f'{PAIR}{c}_s{s}.json'))['refit'] for s in seeds]
    return r


def msd(v): return f'{np.mean(v):.4f} ± {np.std(v):.4f}'


def val_of(tag, m):
    j = jl(tag, m)
    return j[0].get('val', float('nan')) if j else float('nan')


def best_cand(tag, cands=CAND):
    c = [m for m in cands if jl(tag, m)]
    return max(c, key=lambda m: val_of(tag, m))


def stat_row(tag, m, nm=None):
    j = jl(tag, m)
    if not j: return f'{nm or m:34s} (none)'
    g = lambda k: msd([x['test'][k] for x in j])
    cfg = j[0]['cfg']
    return f"{nm or m:34s} {val_of(tag, m):7.4f} {g('auc'):>16s} {g('aupr'):>16s} {g('within'):>16s} {g('within8'):>16s}  {cfg if isinstance(cfg, str) else {k: v for k, v in cfg.items()}}"


HDR = f"{'model':34s} {'val wTF':>7s} {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within>=8':>16s}  chosen config (on validation)"


def pair_line(A, B, label):
    return w.paired(A, B, label)


def popgrid(cell):
    tr = pd.read_csv(f'data/{cell}/TFs+500/Train_set.csv', index_col=0)
    n = len(pd.read_csv(f'data/{cell}/TFs+500/Target.csv', index_col=0))
    return tr, n, np.bincount(tr.Target.values[tr.Label.values == 1], minlength=n)


def q_profile(z, cfg):
    """lift of every TF in every cluster: omega_tk / mean over TFs of omega_t'k (1 = the TF treats the cluster like the average TF)"""
    om, R, th = z['omega'], z['R'].astype(float), z['theta']
    raw = om * ((R.T @ th) / R.sum(0).clip(1e-9))[None] if cfg['dc'] else om
    return raw / raw.mean(0, keepdims=True)


def entropy_stats(tag, m):
    """normalised entropy of p_tk = q_tk / sum_k q_tk (1 = flat, no cluster preference) and the mass-weighted sd of log q (strength of preference)"""
    out = []
    for z, j in zip(zl(tag, m), jl(tag, m)):
        q = q_profile(z, j['cfg']).clip(1e-9); R = z['R'].astype(float); K = q.shape[1]
        p = q / q.sum(1, keepdims=True)
        H = -(p * np.log(p)).sum(1) / np.log(K)
        mass = R.sum(0) / R.sum()
        mu = (np.log(q) * mass).sum(1, keepdims=True)
        sd = np.sqrt((((np.log(q) - mu) ** 2) * mass).sum(1))
        out.append((H, sd))
    return np.mean([o[0] for o in out], 0), np.mean([o[1] for o in out], 0)


def main():
    P('W12 statistical network models (DC-SBM, Poisson factorisation, latent space) vs gene popularity and the GNN. CPU, 3 seeds (42-44) for the random split; held-out genes 5 folds x seeds 42-43.')
    P('Every knob (GMM covariance type, K, smoothing strengths, DC normalisation, rank / dimension, L2, number of steps) was chosen on the validation slice by mean-over-seeds within-TF AUROC; the model was refit on all training rows and the test rows scored once.')
    P('Test metrics are mean ± std over seeds. "val wTF" is the validation within-TF AUROC of the chosen config (mean over seeds).')
    best = {}
    for cell in CELLS:
        if not jl(cell, 'pop'): continue
        P(f'\n\n######## {cell}')
        P(HDR)
        hr = hero_row(HERO[cell]); bg = hero_row(BESTG[cell])
        P('-- reference GNNs (same test rows, same seeds)')
        for nm, r, c in (('hero SAGE ' + HERO[cell], hr, ''), ('best GNN ' + BESTG[cell], bg, '')):
            P(f"{nm[:34]:34s} {'':7s} " + ' '.join(f"{msd([x[k] for x in r]):>16s}" for k in ('auc', 'aupr', 'within', 'within8')))
        P('-- controls')
        for m in ('pop_indeg', 'pop'): P(stat_row(cell, m))
        P('-- DC-SBM, fixed GMM clusters (K chosen on validation; rows ending _omegaonly / _thetaonly drop one factor of the selected model)')
        for m in FIX:
            P(stat_row(cell, m))
            P(stat_row(cell, m + '_omegaonly', '    omega only (no theta)')); P(stat_row(cell, m + '_thetaonly', '    theta only (no program)'))
        P('-- DC-SBM per K (config chosen on validation within each K)')
        for m in FIX:
            P('  ' + m + ': ' + '   '.join(f"K={K}: val {val_of(cell, f'{m}_K{K}'):.4f} test within {np.mean([x['test']['within'] for x in jl(cell, f'{m}_K{K}')]):.4f}" for K in (5, 10, 20, 40)))
        P('-- learned block / factorisation / latent space')
        for m in LEARN: P(stat_row(cell, m))
        P(stat_row(cell, 'maineff', 'maineff (control: K=1, a_t + b_g only)'))
        b = best_cand(cell); best[cell] = b
        fixb = best_cand(cell, FIX)
        P(f'\nbest statistical model by validation: {b}   (best fixed-cluster DC-SBM: {fixb})')

        P('\nPaired per-TF tests (per-TF test AUROC averaged over seeds 42-44, difference, TFs better, Wilcoxon p):')
        B = tfm(cell, b); pop = tfm(cell, 'pop'); pi = tfm(cell, 'pop_indeg'); H = hero_tf(HERO[cell]); G = hero_tf(BESTG[cell])
        P(pair_line(B, pop, f'{b} - popularity (smoothed)')); P(pair_line(B, pi, f'{b} - popularity (in-degree)'))
        P(pair_line(B, H, f'{b} - hero')); P(pair_line(B, G, f'{b} - best GNN ({BESTG[cell]})'))
        P(pair_line(pop, H, 'popularity - hero')); P(pair_line(H, pop, 'hero - popularity'))
        P(pair_line(G, pop, 'best GNN - popularity'))
        P('program term: selected model vs its own theta-only version, and vs popularity, for every fixed-cluster family:')
        for m in FIX:
            P(pair_line(tfm(cell, m), tfm(cell, m + '_thetaonly'), f'{m} - theta-only')); P(pair_line(tfm(cell, m), pop, f'{m} - popularity'))
        for m in LEARN: P(pair_line(tfm(cell, m), pop, f'{m} - popularity'))
        P('program term of the learned models: model minus the Bernoulli main-effects control (a_t + b_g, same l2 / steps selection):')
        P(pair_line(tfm(cell, 'maineff'), pop, 'maineff - popularity'))
        for m in LEARN: P(pair_line(tfm(cell, m), tfm(cell, 'maineff'), f'{m} - maineff'))

    # ---------------------------------------------------------------- where the program term helps
    P('\n\n######## Where does the program term add over popularity? (per-TF gain of the selected fixed-cluster DC-SBM over smoothed popularity)')
    P('entropy = normalised entropy of the TF cluster-preference profile (1 = flat); sd = mass-weighted sd of log lift (omega_tk / mean over TFs of omega_t.k), averaged over seeds.')
    rows = {}
    for cell in CELLS:
        if not jl(cell, 'pop'): continue
        fm = best_cand(cell, FIX)
        tr, n, pop_deg = popgrid(cell)
        A, Pp = tfm(cell, fm), tfm(cell, 'pop')
        H, sd = entropy_stats(cell, fm)
        ids = zl(cell, fm)[0]['grid_tf']
        npos = tr[tr.Label == 1].groupby('TF').size().reindex(ids).fillna(0).values
        gain = np.array([A[t] - Pp[t] if t in A and t in Pp else np.nan for t in ids])
        ok = ~np.isnan(gain)
        r1, p1 = spearmanr(gain[ok], npos[ok]); r2, p2 = spearmanr(gain[ok], H[ok]); r3, p3 = spearmanr(gain[ok], sd[ok])
        rows[cell] = dict(fm=fm, gain=np.nanmean(gain), H=H.mean(), sd=sd.mean(), npos=npos, r=(r1, p1, r2, p2, r3, p3))
        P(f'{cell}: model {fm}: mean per-TF gain {np.nanmean(gain):+.4f} over {ok.sum()} TFs; Spearman(gain, #training targets) {r1:+.2f} (p={p1:.2g}); (gain, entropy) {r2:+.2f} (p={p2:.2g}); (gain, sd of log preference) {r3:+.2f} (p={p3:.2g})')
        lo, hi = npos <= np.median(npos), npos > np.median(npos)
        P(f'    gain for TFs with <= median ({np.median(npos):.0f}) training targets: {np.nanmean(gain[lo & ok]):+.4f}; with more: {np.nanmean(gain[hi & ok]):+.4f}')
        order = np.argsort(-np.nan_to_num(gain, nan=-9))
        P('    top gains: ' + ', '.join(f'TF{ids[i]} ({npos[i]:.0f} targets, {gain[i]:+.3f})' for i in order[:4]) + '; worst: ' + ', '.join(f'TF{ids[i]} ({npos[i]:.0f}, {gain[i]:+.3f})' for i in order[::-1][:3]))

    P('\nSame for the learned-block DC-SBM (dcsbm_learned) and the latent space model (lsm) vs smoothed popularity:')
    for cell in rows:
        tr, n, _ = popgrid(cell)
        for m in ('dcsbm_learned', 'lsm'):
            A, Pp = tfm(cell, m), tfm(cell, 'pop')
            ks = [t for t in A if t in Pp]
            npos = np.array([((tr.TF == t) & (tr.Label == 1)).sum() for t in ks]); g = np.array([A[t] - Pp[t] for t in ks])
            r, pv = spearmanr(g, npos)
            P(f'{cell} {m}: mean gain {g.mean():+.4f}; Spearman(gain, #training targets) {r:+.2f} (p={pv:.2g}); gain for TFs with <= median ({np.median(npos):.0f}) targets {g[npos <= np.median(npos)].mean():+.4f}, with more {g[npos > np.median(npos)].mean():+.4f}')

    P('\n######## Why hHEP is popularity-dominated: the four cells side by side')
    P(f"{'cell':8s} {'pop wTF':>8s} {'indeg wTF':>9s} {'hero wTF':>8s} {'fixedDC':>9s} {'stat-pop':>9s} {'omega-only':>10s} {'entropy':>8s} {'sd logq':>8s} {'#targets/TF':>11s} {'gini(indeg)':>11s} {'TF-TF cosine':>12s} {'pop share':>9s} {'learned-pop':>11s}")
    for cell in CELLS:
        if cell not in rows: continue
        fm = rows[cell]['fm']; tr, n, deg = popgrid(cell)
        g = np.sort(deg.astype(float)); gini = (2 * np.arange(1, n + 1) - n - 1) @ g / (n * g.sum())
        Y = np.zeros((len(np.unique(tr.TF)), n)); tfs = np.unique(tr.TF)
        for t, gg in zip(tr.TF[tr.Label == 1], tr.Target[tr.Label == 1]): Y[np.searchsorted(tfs, t), gg] = 1
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9); C = Yn @ Yn.T; cos = (C.sum() - np.trace(C[:, :])) / (len(C) * (len(C) - 1))
        # share of a TF's targets that are in the top-10% most popular genes
        top = deg >= np.sort(deg)[::-1][int(0.1 * n)]
        share = np.mean([Y[i][top].sum() / max(Y[i].sum(), 1) for i in range(len(Y)) if Y[i].sum() > 0])
        jf = lambda m: np.mean([x['test']['within'] for x in jl(cell, m)])
        P(f"{cell:8s} {jf('pop'):8.4f} {jf('pop_indeg'):9.4f} {np.mean([x['within'] for x in hero_row(HERO[cell])]):8.4f} {jf(fm):9.4f} {rows[cell]['gain']:+9.4f} {jf(fm + '_omegaonly'):10.4f} {rows[cell]['H']:8.3f} {rows[cell]['sd']:8.3f} {Y.sum(1).mean():11.1f} {gini:11.3f} {cos:12.3f} {share:9.3f} {jf('dcsbm_learned') - jf('pop'):+11.4f}")
    P('(pop wTF = smoothed popularity; omega-only = program term alone; entropy and sd are for the selected fixed-cluster family of that cell; TF-TF cosine = mean pairwise cosine of the TFs\' training target vectors;')
    P(' pop share = mean over TFs of the fraction of training targets that lie in the 10% most popular genes.)')
    P('\nSame comparison at fixed K=10, Geneformer clusters, soft assignment (config chosen on validation per cell):')
    for cell in CELLS:
        if not jl(cell, 'dcsbm_gf_soft_K10'): continue
        H, sd = entropy_stats(cell, 'dcsbm_gf_soft_K10')
        pp = tfm(cell, 'pop'); aa = tfm(cell, 'dcsbm_gf_soft_K10')
        P(f"  {cell:8s} entropy {H.mean():.3f}  sd log-preference {sd.mean():.3f}  gain over popularity {np.mean([aa[t] - pp[t] for t in aa if t in pp]):+.4f}  test within {np.mean([x['test']['within'] for x in jl(cell, 'dcsbm_gf_soft_K10')]):.4f}")

    # ---------------------------------------------------------------- key TF clusters, hESC
    for fm_ in ([rows['hESC']['fm'], 'dcsbm_gf_soft_K10'] if 'hESC' in rows else []):
        P(f'\n\n######## hESC: clusters preferred by key TFs ({fm_}, seed 42; preference = TF-normalised lift = (omega_tk / mean over TFs of omega_t.k) divided by the cluster-mass-weighted mean lift, so 1 = its own average; hard cluster = argmax posterior; members = the TF training targets in the cluster, most popular first)')
        fm = fm_; z = zl('hESC', fm)[0]; j = jl('hESC', fm)[0]
        q = q_profile(z, j['cfg']); R = z['R'].astype(float); q = q / (q * (R.sum(0) / R.sum())).sum(1, keepdims=True); hardc = R.argmax(1); K = q.shape[1]; ids = list(z['grid_tf'])
        genes = pd.read_csv('data/hESC/TFs+500/Target.csv', index_col=0).set_index('index')['Gene']
        tfn = pd.read_csv('data/hESC/TFs+500/TF.csv', index_col=0).set_index('TF')['index']
        tr, n, deg = popgrid('hESC')
        P(f'config {j["cfg"]}; cluster sizes: {np.bincount(hardc, minlength=K).tolist()}')
        for name in ('POU5F1', 'NANOG', 'SOX2', 'SOX17', 'GATA6', 'EOMES'):
            if name not in tfn.index: P(f'  {name}: not a TF in this data set'); continue
            t = int(tfn[name]); i = ids.index(t) if t in ids else None
            if i is None: P(f'  {name}: not in the TF set'); continue
            tp = set(tr.Target[(tr.TF == t) & (tr.Label == 1)])
            sizes = np.bincount(hardc, minlength=K)
            okc = [k for k in np.argsort(-q[i]) if sizes[k] >= 8][:3]
            P(f'  {name} ({len(tp)} training targets): ' + ' | '.join(
                f'cluster {k} (size {sizes[k]}, pref {q[i, k]:.2f}x): ' + ','.join(str(genes[g]) for g in [g for g in np.argsort(-deg) if hardc[g] == k and g in tp][:6]) + f' [{sum(1 for g in tp if hardc[g] == k)}/{len(tp)} targets]' for k in okc))
            P(f'      lowest preference: ' + ', '.join(f'cluster {k} ({q[i, k]:.2f}x)' for k in [k for k in np.argsort(q[i]) if sizes[k] >= 8][:2]))

    # ---------------------------------------------------------------- correlation with popularity
    P('\n\n######## Score correlation with gene popularity (per-TF Spearman over the test rows between score and the gene training in-degree, mean over TFs and seeds)')
    P('and mean per-TF Spearman between the model score and the hero logit on the test rows (seed-mean)')
    for cell in CELLS:
        if not jl(cell, 'pop'): continue
        tr, n, deg = popgrid(cell)
        hz = [np.load(f'{PAIR}{HERO[cell]}_s{s}.npz') for s in SEEDS]; bz = [np.load(f'{PAIR}{BESTG[cell]}_s{s}.npz') for s in SEEDS]
        sp = lambda zs: np.mean([np.nanmean([spearmanr(z['logit'][z['tf'] == t], deg[z['tgt'][z['tf'] == t]]).statistic for t in np.unique(z['tf']) if (z['tf'] == t).sum() > 5]) for z in zs])
        P(f'{cell}: hero {sp(hz):.3f}   best GNN {sp(bz):.3f}')
        for m in ['pop'] + FIX + [f + '_omegaonly' for f in FIX] + LEARN:
            zs = zl(cell, m)
            if not zs: continue
            # model-vs-hero: per TF Spearman over test rows (seed s with seed s)
            cs = []
            for z, h in zip(zs, hz):
                o = {t: spearmanr(z['logit'][z['tf'] == t], h['logit'][h['tf'] == t]).statistic for t in np.unique(z['tf']) if (z['tf'] == t).sum() > 5}
                cs.append(np.nanmean(list(o.values())))
            P(f'   {m:30s} pop-corr {sp(zs):+.3f}   corr with hero logit {np.mean(cs):+.3f}')

    # ---------------------------------------------------------------- held-out genes
    P('\n\n######## Held-out genes (hESC, protocol gene, folds 0-4, seeds 42-43). Statistical models see only gene features of the unseen genes; no popularity is available for them (theta = cluster mean).')
    P('Held-out TFs were skipped: an unseen TF has no omega row (and no latent position).')
    hm = ['dcsbm_gf_soft', 'dcsbm_gf_hard', 'dcsbm_gf_soft_thetaonly', 'dcsbm_gf_soft_omegaonly', 'lsm']
    folds = [f for f in range(5) if jl(f'hESC_hog{f}', 'lsm', (42, 43))]
    if folds:
        S2 = (42, 43)
        P(f"{'model':34s} {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within>=8':>16s}   (mean ± std over fold x seed; {len(folds)} folds)")
        hero = [json.load(open(f'{PAIR}w5_ho_gene{f}_hero_s{s}.json'))['refit'] for f in folds for s in S2]
        P(f"{'hero (w5_ho_gene*_hero)':34s} " + ' '.join(f"{msd([r[k] for r in hero]):>16s}" for k in ('auc', 'aupr', 'within', 'within8')))
        for m in hm:
            r = [x['test'] for f in folds for x in jl(f'hESC_hog{f}', m, S2)]
            P(f'{m:34s} ' + ' '.join(f"{msd([x[k] for x in r]):>16s}" for k in ('auc', 'aupr', 'within', 'within8')))
        P('per fold within-TF (statistical model mean over seeds | hero): ' + ' ; '.join(
            f"f{f}: gf_soft {np.mean([x['test']['within'] for x in jl(f'hESC_hog{f}', 'dcsbm_gf_soft', S2)]):.3f} lsm {np.mean([x['test']['within'] for x in jl(f'hESC_hog{f}', 'lsm', S2)]):.3f} | {np.mean([json.load(open(f'{PAIR}w5_ho_gene{f}_hero_s{s}.json'))['refit']['within'] for s in S2]):.3f}" for f in folds))
        for f in folds:
            j = jl(f'hESC_hog{f}', 'dcsbm_gf_soft', S2)[0]; l = jl(f'hESC_hog{f}', 'lsm', S2)[0]
            P(f'  fold {f} chosen: dcsbm_gf_soft {j["cfg"]} (val {j["val"]:.3f}); lsm {l["cfg"]} (val {l["val"]:.3f})')
        # paired per (fold, TF)
        def pooled(getter):
            d = {}
            for f in folds:
                for t, v in getter(f).items(): d[(f, t)] = v
            return d
        Hh = pooled(lambda f: hero_tf_f(f, S2))
        for m in hm[:2] + ['lsm']:
            M = pooled(lambda f: tfm(f'hESC_hog{f}', m, S2))
            P(pair_line(M, Hh, f'{m} - hero (per fold x TF)'))
        T_ = pooled(lambda f: tfm(f'hESC_hog{f}', 'dcsbm_gf_soft_thetaonly', S2)); M = pooled(lambda f: tfm(f'hESC_hog{f}', 'dcsbm_gf_soft', S2))
        P(pair_line(M, T_, 'dcsbm_gf_soft - its theta-only (program term, unseen genes)'))
    else:
        P('(held-out runs not finished)')
    open('analysis/out/w12_stat.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def hero_tf_f(f, seeds):
    d = [w.per_tf(z['tf'], z['y'], z['logit']) for z in (np.load(f'{PAIR}w5_ho_gene{f}_hero_s{s}.npz') for s in seeds)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))}


if __name__ == '__main__':
    main()
