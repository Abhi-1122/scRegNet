"""Item 8 round 1: does per-time-point co-expression match the hESC ChIP labels better than pooled co-expression? CPU only, no training, no label is used in any score.
Score of a (TF, gene) pair = |Spearman| between their expression over a set of cells: each of the 6 time points, all cells pooled, all cells after centring each gene within its time point.
Rows: Test_set, and all labelled rows (Train u Test). Within-TF AUROC is the headline. Time-point choice for the tests is made on the Train_set rows.
Run: python3.12 -P analysis/w10_states_r1.py"""
import numpy as np, pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, rankdata, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

P = 'data/hESC/TFs+500/'
x = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0)
genes = pd.read_csv(P + 'Target.csv', index_col=0).Gene.tolist()
assert (x.index == genes).all()
E = x.values
tp = np.array(['_'.join(c.split('_')[:2]) for c in x.columns])
TPS = ['H9_00hb4s', 'H9_12h', 'H9_24h', 'H9_36h', 'H9_72h', 'H9_96h']
LAB = {t: t.replace('H9_', '').replace('b4s', '') for t in TPS}  # 00h 12h 24h 36h 72h 96h
tr0, te0 = pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)
al0 = pd.concat([tr0, te0], ignore_index=True)
ROWS = {'Test_set': te0, 'all labelled': al0}
tfs = np.unique(al0.TF.values)
out = []
pr = lambda s='': (print(s, flush=True), out.append(s))


def abs_spearman(cols, centred=False):  # (910 x 910) |Spearman| for the labelled TFs' rows (other rows NaN)
    M = E[:, cols]
    if centred:  # remove each gene's mean within its time point, then rank over all cells
        M = M.copy()
        for t in TPS: M[:, tp[cols] == t] -= M[:, tp[cols] == t].mean(1, keepdims=True)
    R = np.apply_along_axis(rankdata, 1, M)
    R = R - R.mean(1, keepdims=True)
    sd = np.sqrt((R ** 2).sum(1)); sd[sd == 0] = np.inf  # constant genes get correlation 0
    Z = R / sd[:, None]
    S = np.full((len(genes), len(genes)), np.nan)
    S[tfs] = np.abs(Z[tfs] @ Z.T)
    return S


def tf_auc(y, s, t):
    return {int(u): roc_auc_score(y[t == u], s[t == u]) for u in np.unique(t) if 0 < y[t == u].sum() < (t == u).sum()}


def ev(S, d):
    t, g, y = d.TF.values, d.Target.values, d.Label.values
    s = S[t, g]
    a = tf_auc(y, s, t)
    return dict(within=np.mean(list(a.values())), auc=roc_auc_score(y, s), aupr=average_precision_score(y, s), tf=a)


def cmp(V, B):  # paired per-TF Wilcoxon: V - B
    ks = sorted(set(V) & set(B)); d = np.array([V[q] - B[q] for q in ks])
    boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return f'{d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  TFs improved {int((d > 0).sum())}/{len(d)}  p={p:.3g}'


allc = np.arange(E.shape[1])
SC = {LAB[t]: abs_spearman(np.where(tp == t)[0]) for t in TPS}
SC['pooled'] = abs_spearman(allc)
SC['pooled, centred within time point'] = abs_spearman(allc, centred=True)
RES = {rn: {k: ev(S, d) for k, S in SC.items()} for rn, d in ROWS.items()}
n_cells = {LAB[t]: int((tp == t).sum()) for t in TPS}

pr('Item 8 round 1: |Spearman| co-expression of each (TF, gene) pair, per time point / pooled / pooled after centring each gene within its time point.')
pr(f'Cells per time point: {n_cells}; labelled TFs: {len(tfs)}; Test_set rows {len(te0)}, all labelled rows {len(al0)}. No label enters any score.')
pr('\n== 0. Main table: within-TF AUROC (headline), AUROC, AUPRC; positive rate Test_set %.3f, all labelled %.3f' % (te0.Label.mean(), al0.Label.mean()))
pr(f"{'score':36s} {'cells':>5s} | {'Test_set: within-TF':>20s} {'AUROC':>7s} {'AUPRC':>7s} | {'all labelled: within-TF':>24s} {'AUROC':>7s} {'AUPRC':>7s}")
for k in SC:
    a, b = RES['Test_set'][k], RES['all labelled'][k]
    pr(f"{k:36s} {n_cells.get(k, len(allc)):5d} | {a['within']:20.4f} {a['auc']:7.4f} {a['aupr']:7.4f} | {b['within']:24.4f} {b['auc']:7.4f} {b['aupr']:7.4f}")

# the time point used in the tests: chosen on the Train_set rows (the scores never saw labels, but this keeps the Test_set comparison clean)
trs = {k: ev(S, tr0)['within'] for k, S in SC.items() if k in LAB.values()}
best = max(trs, key=trs.get)
pr(f'\nbest time point on the Train_set rows (within-TF): ' + ', '.join(f'{k} {v:.4f}' for k, v in trs.items()) + f'  ->  {best}')

# ---- 1. cell-count control
pr('\n== 1. Cell-count control: each time point subsampled to 66 cells (the size of 24h), 20 repeats, mean ± sd (24h has exactly 66 cells, so it does not vary)')
REP = 20
sub = {rn: {LAB[t]: [] for t in TPS} for rn in ROWS}
for rep in range(REP):
    rs = np.random.RandomState(rep)
    for t in TPS:
        idx = np.where(tp == t)[0]
        cols = idx if len(idx) == 66 else np.sort(rs.choice(idx, 66, replace=False))
        S = abs_spearman(cols)
        for rn, d in ROWS.items(): sub[rn][LAB[t]].append(ev(S, d))
pr(f"{'time point':10s} {'cells':>5s} | {'Test_set within-TF (full data)':>32s} {'66-cell mean ± sd':>20s} {'AUROC':>16s} {'AUPRC':>16s} | {'all labelled within-TF (full)':>30s} {'66-cell mean ± sd':>20s} {'AUROC':>16s} {'AUPRC':>16s}")
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'
for t in TPS:
    k = LAB[t]; line = f'{k:10s} {n_cells[k]:5d} |'
    for rn in ROWS:
        r = sub[rn][k]
        line += f" {RES[rn][k]['within']:32.4f} {ms([z['within'] for z in r]):>20s} {ms([z['auc'] for z in r]):>16s} {ms([z['aupr'] for z in r]):>16s} |"
    pr(line)
for rn in ROWS:
    W = np.array([[z['within'] for z in sub[rn][LAB[t]]] for t in TPS])  # (6, REP)
    full = np.array([RES[rn][LAB[t]]['within'] for t in TPS])
    rk = np.array([rankdata(-W[:, r]) for r in range(REP)])  # rank 1 = best, per repeat
    tau = kendalltau(full, W.mean(1)).statistic
    pr(f"{rn}: mean rank of the 6 time points over {REP} repeats (1 = best) " + ', '.join(f'{LAB[t]} {rk[:, i].mean():.2f}' for i, t in enumerate(TPS))
       + f"; share of repeats in which each is the best: " + ', '.join(f'{LAB[t]} {np.mean(rk[:, i] == 1):.2f}' for i, t in enumerate(TPS))
       + f"; Kendall tau, full-data vs subsampled-mean ordering: {tau:+.2f}")
    e, l = W[[0, 1]].mean(0), W[[4, 5]].mean(0)
    pr(f"   early (00h,12h) minus late (72h,96h) within-TF at 66 cells: {np.mean(e - l):+.4f} ± {np.std(e - l):.4f}; early better in {int((e > l).sum())}/{REP} repeats. Full-data ordering: " + ' > '.join(LAB[TPS[i]] for i in np.argsort(-full)))

# ---- 2. paired tests
pr('\n== 2. Paired per-TF tests on within-TF AUROC (difference [bootstrap 95% CI], TFs improved, Wilcoxon p)')
for rn in ROWS:
    R = RES[rn]
    pr(f'-- rows: {rn}' + (f' (best time point {best} chosen on Train_set)' if True else ''))
    avg = lambda ks: {q: np.mean([R[k]['tf'][q] for k in ks]) for q in R[ks[0]]['tf']}
    pr(f"  best ({best}) vs pooled                   {cmp(R[best]['tf'], R['pooled']['tf'])}")
    pr(f"  early (00h,12h) vs late (72h,96h)          {cmp(avg(['00h', '12h']), avg(['72h', '96h']))}")
    pr(f"  centred pooled vs pooled                   {cmp(R['pooled, centred within time point']['tf'], R['pooled']['tf'])}")
    pr(f"  best ({best}) vs the mean of the other 5 time points {cmp(R[best]['tf'], avg([LAB[t] for t in TPS if LAB[t] != best]))}")
    pr(f"  centred pooled vs best ({best})            {cmp(R['pooled, centred within time point']['tf'], R[best]['tf'])}")
    pr(f"  early (00h,12h) vs pooled                  {cmp(avg(['00h', '12h']), R['pooled']['tf'])}")

# ---- 3. per-TF biology
pr('\n== 3. Per-TF check (all labelled rows). Expression = mean of the log-expression values in the file; rho = Spearman across the 6 time points between the TF\'s mean expression and its AUROC')
R = RES['all labelled']
names = [LAB[t] for t in TPS]
rho, rows3 = {}, []
pr(f"{'TF':9s} {'best tp':>7s} {'peak expr tp':>12s} {'rho':>6s} | AUROC at 00h/12h/24h/36h/72h/96h{'':6s} | mean expression at 00h/12h/24h/36h/72h/96h")
for q in tfs:
    au = np.array([R[k]['tf'].get(int(q), np.nan) for k in names])
    ex = np.array([E[q, tp == t].mean() for t in TPS])
    if np.isnan(au).any(): continue
    rho[q] = spearmanr(ex, au).statistic
    rows3.append((q, int(np.argmax(au)), int(np.argmax(ex)), au, ex))
    pr(f"{genes[q]:9s} {names[np.argmax(au)]:>7s} {names[np.argmax(ex)]:>12s} {rho[q]:+6.2f} | " + ' '.join(f'{v:.3f}' for v in au) + ' | ' + ' '.join(f'{v:5.2f}' for v in ex))
rv = np.array([rho[r[0]] for r in rows3])
pr(f'\nsummary over {len(rows3)} TFs: mean rho {rv.mean():+.3f}, median {np.median(rv):+.3f}, rho > 0 for {int((rv > 0).sum())}/{len(rv)} TFs, Wilcoxon p (rho vs 0) = {wilcoxon(rv).pvalue:.3g}')
bi, pi = np.array([r[1] for r in rows3]), np.array([r[2] for r in rows3])
pr(f'best-AUROC time point equals the peak-expression time point for {int((bi == pi).sum())}/{len(bi)} TFs (chance about {len(bi) / 6:.1f}); Spearman across TFs between the two time-point indices: {spearmanr(bi, pi).statistic:+.3f} (p={spearmanr(bi, pi).pvalue:.3g})')
at_peak = {genes[r[0]]: r[3][r[2]] - np.delete(r[3], r[2]).mean() for r in rows3}
d = np.array(list(at_peak.values()))
pr(f'AUROC at the TF\'s peak-expression time point minus the mean AUROC at its other 5: {d.mean():+.4f}, higher for {int((d > 0).sum())}/{len(d)} TFs, Wilcoxon p={wilcoxon(d).pvalue:.3g}')
# also: is the correlation simply "more expressed = better estimated"? compare with the TF's expression minus its mean over time points, and the gene side is ignored here
EARLY, LATE = ['POU5F1', 'NANOG', 'SOX2'], ['SOX17', 'GATA4', 'GATA6', 'EOMES']
bt = {genes[r[0]]: (names[r[1]], names[r[2]]) for r in rows3}
pr('expected-direction TFs (best AUROC time point / peak expression time point): early group ' + ', '.join(f'{q} {bt[q][0]}/{bt[q][1]}' for q in EARLY if q in bt)
   + '; later group ' + ', '.join(f'{q} {bt[q][0]}/{bt[q][1]}' for q in LATE if q in bt))
if all(q in bt for q in EARLY + LATE):
    ie = [names.index(bt[q][0]) for q in EARLY]; il = [names.index(bt[q][0]) for q in LATE]
    pr(f'   mean best-time-point index (0=00h ... 5=96h): early group {np.mean(ie):.2f}, later group {np.mean(il):.2f} (Mann-Whitney p={mannwhitneyu(ie, il).pvalue:.3g}, 3 vs 4 TFs, so low power)')

# ---- 4. fusion with popularity
pr('\n== 4. Fusion with popularity: rank-avg(popularity, score); popularity = log1p Train_set in-degree of the gene')
A = np.zeros((len(genes),) * 2); A[tr0.TF[tr0.Label == 1], tr0.Target[tr0.Label == 1]] = 1
deg = A.sum(0)
rk = lambda v: rankdata(v) / len(v)
for rn, d in ROWS.items():
    t, g, y = d.TF.values, d.Target.values, d.Label.values
    pop = np.log1p(deg[g] - (A[t, g] if rn == 'all labelled' else 0))  # all labelled: leave the row's own label out of the in-degree
    sc = {'popularity alone': pop}
    for k in (best, 'pooled', 'pooled, centred within time point'):
        sc[f'rank avg pop + {k}'] = rk(pop) + rk(SC[k][t, g])
    sc[f'{best} alone'] = SC[best][t, g]; sc['pooled alone'] = SC['pooled'][t, g]
    base = tf_auc(y, sc[f'rank avg pop + pooled'], t)
    pr(f'-- rows: {rn}' + ('' if rn == 'Test_set' else ' (popularity with the row\'s own label removed)'))
    for k, s in sc.items():
        a = tf_auc(y, s, t)
        pr(f'  {k:42s} within-TF {np.mean(list(a.values())):.4f}  AUROC {roc_auc_score(y, s):.4f}  AUPRC {average_precision_score(y, s):.4f}'
           + (f'   vs rank avg pop + pooled: {cmp(a, base)}' if k.startswith('rank avg pop + ') and k != 'rank avg pop + pooled' else ''))
    pa = tf_auc(y, pop, t)
    pr(f"  rank avg pop + {best} vs popularity alone: {cmp(tf_auc(y, sc[f'rank avg pop + {best}'], t), pa)}")
    pr(f"  rank avg pop + pooled vs popularity alone: {cmp(base, pa)}")
open('analysis/out/w10_states_r1.txt', 'w').write('\n'.join(out) + '\n')
