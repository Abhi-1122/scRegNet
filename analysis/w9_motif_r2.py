"""Item 7 round 2: four diagnostics on the motif-only prior (hESC, TFs+500). Analysis only, no model training. Re-uses the round-1 caches
(data/external/promoters_hg38/sym_*, tr_*, seq_*, JASPAR file) and the round-1 folds. Run: python3.12 -P analysis/w9_motif_r2.py
A better motif score: per motif, -log10 p of the best hit (DP null) or the hit count at p < 1e-4, then the MAX over the TF's motifs; windows -500/+100, -1000/+200, -2000/+500.
B confound: CpG o/e and GC of the -1000/+200 promoter; motif AUROC after residualising on [popularity, CpG o/e, GC].
C motif-similarity transfer for held-out TFs.   D motif added to the trained model's test logits (benchmark split).
No ChIP-derived data is used. The -2000/+500 sequences are fetched once (cached as seqw_*.json)."""
import glob, json, os, re, sys, time
from types import SimpleNamespace
import numpy as np, pandas as pd, requests
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '.')
from src.pair_gnn import heldout_split

P, CACHE = 'data/hESC/TFs+500/', 'data/external/promoters_hg38/'
UP, DOWN, PV, BATCH = 2000, 500, 1e-4, 50
WIN = {'-500/+100': (500, 100), '-1000/+200': (1000, 200), '-2000/+500': (2000, 500)}
genes = pd.read_csv(P + 'Target.csv', index_col=0).Gene.tolist()
n, T = len(genes), UP + DOWN + 1
out = []
pr = lambda s='': (print(s, flush=True), out.append(s))

# ---------------------------------------------------------------- sequences (-2000/+500, gene orientation, TSS at index 2000)
sess = requests.Session()
sess.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'scRegNet-motif-prior (research; garimella.abhishek@research.iiit.ac.in)'})


def post(path, body, fn):
    if os.path.exists(CACHE + fn): return json.load(open(CACHE + fn))
    for k in range(12):
        time.sleep(0.5)
        try: r = sess.post('https://rest.ensembl.org' + path, data=json.dumps(body), timeout=300)
        except requests.exceptions.RequestException:
            time.sleep(10 * (k + 1)); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get('Retry-After', 10 * (k + 1)))); continue
        r.raise_for_status()
        json.dump(r.json(), open(CACHE + fn, 'w'))
        return r.json()
    raise RuntimeError(path)


canon, trs = {}, {}
for f in sorted(glob.glob(CACHE + 'sym_*.json')):
    for s, g in json.load(open(f)).items():
        if g and g.get('canonical_transcript'): canon[s.upper()] = g['canonical_transcript'].split('.')[0]
for f in sorted(glob.glob(CACHE + 'tr_*.json')): trs.update(json.load(open(f)))
reg = {}
for s, i in canon.items():
    t = trs.get(i)
    if t and t['seq_region_name'] in [str(c) for c in range(1, 23)] + ['X', 'Y']:
        a, b, st = t['start'], t['end'], t['strand']
        lo, hi = (a - UP, a + DOWN) if st == 1 else (b - DOWN, b + UP)
        reg[s] = (f'{t["seq_region_name"]}:{max(lo, 1)}..{hi}:{st}', max(lo, 1) - lo if st == 1 else 0)  # left padding when clipped at the chromosome start
syms = sorted(reg)
X = np.full((n, T), 4, dtype=np.int8)  # 0-3 = ACGT, 4 = N / padding
has = np.zeros(n, bool)
row = {g.upper(): i for i, g in enumerate(genes)}
for k in range(0, len(syms), BATCH):
    chunk = syms[k:k + BATCH]
    res = post('/sequence/region/human', {'regions': [reg[s][0] for s in chunk]}, f'seqw_{k:04d}.json')
    assert len(res) == len(chunk)
    for s, r in zip(chunk, res):
        v = np.frombuffer(r['seq'].upper().encode(), dtype=np.uint8)
        c = np.full(len(v), 4, np.int8)
        for j, ch in enumerate(b'ACGT'): c[v == ch] = j
        pad = reg[s][1]
        c = c[:T - pad]
        X[row[s], pad:pad + len(c)] = c
        has[row[s]] = True
# check: the -1000/+200 slice is the sequence used in round 1
bad = 0
for f in sorted(glob.glob(CACHE + 'seq_*.json')):
    k = int(re.findall(r'(\d+)', os.path.basename(f))[0])
    for s, r in zip(syms[k:k + BATCH], json.load(open(f))):
        v = np.frombuffer(r['seq'].upper().encode(), dtype=np.uint8)
        c = np.full(len(v), 4, np.int8)
        for j, ch in enumerate(b'ACGT'): c[v == ch] = j
        bad += int((X[row[s], UP - 1000:UP - 1000 + len(c)] != c).any()) if reg[s][1] == 0 else 0
pr(f'sequences: {int(has.sum())}/{n} genes; windows of the -2000/+500 fetch that differ from the round-1 -1000/+200 sequence: {bad}')
bgc = np.bincount(X[:, UP - 1000:UP + 201][X[:, UP - 1000:UP + 201] < 4].ravel(), minlength=4) / (X[:, UP - 1000:UP + 201] < 4).sum()
bg = np.array([(bgc[0] + bgc[3]) / 2, (bgc[1] + bgc[2]) / 2, (bgc[1] + bgc[2]) / 2, (bgc[0] + bgc[3]) / 2])  # same background as round 1

# ---------------------------------------------------------------- motif scores, 2 score types x 3 windows
KINDS = ['logp', 'hitsmax']
MP = P + 'motif/'
files = {(k, w): MP + f'r2_{k}_{w.replace("/", "_")}.npy' for k in KINDS for w in WIN}
if not all(os.path.exists(f) for f in files.values()):
    mot, cur = [], None
    for line in open('data/external/jaspar2024/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt'):
        if line.startswith('>'):
            cur = [line.split('\t')[1].strip(), []]; mot.append(cur)
        elif line.strip():
            cur[1].append([float(x) for x in re.findall(r'[\d.]+', line.split('[')[1])])
    tf = pd.read_csv(P + 'TF.csv', index_col=0)
    idx = {s.upper(): int(i) for s, i in zip(tf.TF, tf['index'])}
    by_tf = {}
    for name, cnt in mot:
        for part in re.sub(r'\(var\.\d\)', '', name).upper().split('::'):
            if part in idx: by_tf.setdefault(part, []).append(np.array(cnt).T)

    def pwm(c):  # log2-odds in 0.01-bit integers, so window sums sit exactly on the DP grid
        p = (c + 0.8 * bg) / (c.sum(1, keepdims=True) + 0.8)
        return np.round(np.log2(p / bg) * 100).astype(np.int32)

    def null(w):  # background distribution of the window score: sorted integer scores and P(score >= s)
        d = {0: 1.0}
        for col in w:
            nd = {}
            for a, pa in d.items():
                for b in range(4): nd[a + col[b]] = nd.get(a + col[b], 0) + pa * bg[b]
            d = nd
        keys = np.array(sorted(d)); pk = np.array([d[k] for k in keys])
        return keys, np.cumsum(pk[::-1])[::-1]

    M = {kw: np.full((n, n), np.nan, np.float32) for kw in files}
    for t, ms in by_tf.items():
        best = {kw: np.full(n, -np.inf) for kw in files}
        for c in ms:
            w = pwm(c); L = len(w)
            keys, tail = null(w)
            thr = keys[np.argmax(tail <= PV)] if (tail <= PV).any() else keys[-1] + 1
            strands = []
            for m in (w, w[::-1, ::-1]):
                m5 = np.vstack([m.T, np.full((1, L), -100000, np.int32)]).T
                strands.append(sum(m5[j][X[:, j:T - L + 1 + j]] for j in range(L)))
            for wn, (u, d_) in WIN.items():
                a, b = UP - u, UP + d_
                subs = [s[:, a:b - L + 2] for s in strands]
                smax = np.maximum(subs[0].max(1), subs[1].max(1))
                pv = tail[np.minimum(np.searchsorted(keys, smax, 'left'), len(keys) - 1)]
                lp = np.where(smax > -5000, -np.log10(pv), np.nan)
                hits = (subs[0] >= thr).sum(1) + (subs[1] >= thr).sum(1)
                best[('logp', wn)] = np.fmax(best[('logp', wn)], lp)
                best[('hitsmax', wn)] = np.maximum(best[('hitsmax', wn)], hits)
        for kw in files:
            v = best[kw].astype(np.float32); v[~has] = np.nan; v[~np.isfinite(v)] = np.nan
            M[kw][idx[t]] = v
    for kw, f in files.items(): np.save(f, M[kw])
M = {kw: np.load(f) for kw, f in files.items()}
VAR = [f'{k} {w}' for k in KINDS for w in WIN]
MV = {f'{k} {w}': M[(k, w)] for k in KINDS for w in WIN}

# ---------------------------------------------------------------- shared helpers (round-1 conventions)
tr0, te0 = pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)
L_all = pd.concat([tr0, te0], ignore_index=True)


def tf_auc(y, s, t):
    return {int(u): roc_auc_score(y[t == u], s[t == u]) for u in np.unique(t) if 0 < y[t == u].sum() < (t == u).sum()}


def indeg(tr):
    A = np.zeros((n, n)); A[tr.TF[tr.Label == 1], tr.Target[tr.Label == 1]] = 1
    return A


def cmp(V, B, nm='', chance=False):  # V, B: {tf: auc}; paired Wilcoxon over TFs; text 'diff [CI] improved p'
    ks = sorted(set(V) & set(B)); d = np.array([V[q] - B[q] for q in ks]) if not chance else np.array([V[q] - 0.5 for q in ks])
    boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return f'{d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] {int((d > 0).sum())}/{len(d)} p={p:.3g}'


mean = lambda d, ks=None: np.mean([d[q] for q in (ks or d)])


def lr_fit(ttr, gtr, ytr, A, M_, deg, kind):
    f = (lambda v: np.log1p(v)) if kind == 'hitsmax' else (lambda v: v)
    Xtr = np.c_[np.log1p(deg[gtr] - A[ttr, gtr]), f(M_[ttr, gtr])]  # training rows: own TF left out of the in-degree
    ss = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(ss.transform(Xtr), ytr)
    return lambda t, g: m.decision_function(ss.transform(np.c_[np.log1p(deg[g]), f(M_[t, g])]))


ok = lambda d, M_: np.isfinite(M_[d.TF.values, d.Target.values])
# per variant: popularity / motif / LR per-TF AUROCs on (a) the benchmark split and (b) the 5 held-out-TF folds
BEN, HO, TRAIN_SEL = {}, {}, {}
A0 = indeg(tr0); deg0 = A0.sum(0)
folds = []
for fold in range(5):
    tr, tst = heldout_split(L_all, SimpleNamespace(protocol='tf', fold=fold), n)
    A = indeg(tr); folds.append((tr, tst, A, A.sum(0)))
for v in VAR:
    M_, kind = MV[v], v.split()[0]
    k = ok(tr0, M_); trm = tr0[k]
    TRAIN_SEL[v] = tf_auc(trm.Label.values, M_[trm.TF.values, trm.Target.values], trm.TF.values)
    m = ok(te0, M_); te = te0[m]; t, g, y = te.TF.values, te.Target.values, te.Label.values
    f = lr_fit(trm.TF.values, trm.Target.values, trm.Label.values, A0, M_, deg0, kind)
    BEN[v] = dict(pop=tf_auc(y, np.log1p(deg0)[g], t), motif=tf_auc(y, M_[t, g], t), lr=tf_auc(y, f(t, g), t))
    H = dict(pop={}, motif={}, lr={})
    for tr, tst, A, deg in folds:
        m = ok(tst, M_); te = tst[m]; t, g, y = te.TF.values, te.Target.values, te.Label.values
        k = ok(tr, M_); ttr, gtr, ytr = tr.TF.values[k], tr.Target.values[k], tr.Label.values[k]
        f = lr_fit(ttr, gtr, ytr, A, M_, deg, kind)
        H['pop'].update(tf_auc(y, np.log1p(deg)[g], t)); H['motif'].update(tf_auc(y, M_[t, g], t)); H['lr'].update(tf_auc(y, f(t, g), t))
    HO[v] = H
best = max(VAR, key=lambda v: mean(TRAIN_SEL[v]))
pr(f'\n== A. motif score variants (6 = 2 score types x 3 windows); chosen on the benchmark-split TRAIN rows: within-TF AUROC of the motif alone on Train_set')
pr(f"{'variant':24s} {'train within-TF':>15s} | benchmark test, within-TF: {'pop':>6s} {'motif':>6s} {'LR':>6s}  LR vs pop | held-out TFs, within-TF: {'pop':>6s} {'motif':>6s} {'LR':>6s}  motif vs pop | LR vs pop")
for v in VAR:
    b, h = BEN[v], HO[v]
    pr(f"{v:24s} {mean(TRAIN_SEL[v]):15.4f} | {'':27s}{mean(b['pop']):6.4f} {mean(b['motif']):6.4f} {mean(b['lr']):6.4f}  {cmp(b['lr'], b['pop'])} | {'':25s}{mean(h['pop']):6.4f} {mean(h['motif']):6.4f} {mean(h['lr']):6.4f}  {cmp(h['motif'], h['pop'])} | {cmp(h['lr'], h['pop'])}"
       + ('   <- chosen' if v == best else ''))
pr(f'(benchmark test: {len(BEN[best]["pop"])} covered TFs with both classes; held-out: {len(HO[best]["pop"])} TFs pooled over 5 folds. Round 1 (raw max, -1000/+200, held-out): motif 0.5865, LR 0.7202 vs pop 0.7074.)')
pr(f'chosen variant: {best}')
Mb, kind = MV[best], best.split()[0]

# ---------------------------------------------------------------- B. confound check (-1000/+200 promoter)
S = X[:, UP - 1000:UP + 201]
cnt = lambda c: (S == c).sum(1)
Lv = (S < 4).sum(1).astype(float)
CG = ((S[:, :-1] == 1) & (S[:, 1:] == 2)).sum(1)
with np.errstate(all='ignore'):
    cpg = np.where(has, CG * Lv / (cnt(1) * cnt(2)), np.nan)
    gc = np.where(has, (cnt(1) + cnt(2)) / Lv, np.nan)
pr('\n== B. confound check: is the motif signal more than popularity, CpG o/e or GC? (chosen variant; -1000/+200 composition)')


def residual_auc(M_, tr_info, rows, cols):  # per TF: AUROC of the residual of motif ~ [1, cols] fitted on all genes with a motif score
    out_ = {}
    t, g, y = rows
    for u in np.unique(t):
        mu = M_[u]; fin = np.isfinite(mu) & np.all([np.isfinite(c) for c in cols], 0)
        Z = np.c_[np.ones(fin.sum()), *[c[fin] for c in cols]]
        beta = np.linalg.lstsq(Z, mu[fin], rcond=None)[0]
        res = np.full(n, np.nan); res[fin] = mu[fin] - Z @ beta
        m = t == u
        if 0 < y[m].sum() < m.sum() and np.isfinite(res[g[m]]).all(): out_[int(u)] = roc_auc_score(y[m], res[g[m]])
    return out_


def conf(rows, pop_vec):
    t, g, y = rows
    R = dict(pop=tf_auc(y, pop_vec[g], t), motif=tf_auc(y, Mb[t, g], t), cpg=tf_auc(y, cpg[g], t), gc=tf_auc(y, gc[g], t))
    R['resid [pop]'] = residual_auc(Mb, None, rows, [pop_vec])
    R['resid [CpG o/e, GC]'] = residual_auc(Mb, None, rows, [cpg, gc])
    R['resid [pop, CpG o/e, GC]'] = residual_auc(Mb, None, rows, [pop_vec, cpg, gc])
    return R


m = ok(te0, Mb) & np.isfinite(cpg[te0.Target.values]); te = te0[m]
BR = conf((te.TF.values, te.Target.values, te.Label.values), np.log1p(deg0))
HR = {}
for tr, tst, A, deg in folds:
    m = ok(tst, Mb) & np.isfinite(cpg[tst.Target.values]); te = tst[m]
    for k_, d_ in conf((te.TF.values, te.Target.values, te.Label.values), np.log1p(deg)).items(): HR.setdefault(k_, {}).update(d_)
for nm, R in [('benchmark test (covered TFs)', BR), ('held-out TFs (5 folds pooled)', HR)]:
    pr(f'{nm}: {len(R["motif"])} TFs; within-TF AUROC; vs chance 0.5 / vs the motif alone (paired Wilcoxon, TFs above / improved)')
    for k_ in R:
        pr(f'  {k_:26s} {mean(R[k_]):.4f}   vs 0.5: {cmp(R[k_], R["motif"], chance=True)}' + ('' if k_ in ('motif',) else f"   vs motif: {cmp(R[k_], R['motif'])}"))
sp = lambda v: np.nanmedian([spearmanr(Mb[t][np.isfinite(Mb[t])], v[np.isfinite(Mb[t])]).statistic for t in np.where(np.isfinite(Mb).any(1))[0]])
pr(f'median over covered TFs of the Spearman correlation between the motif score and: popularity (benchmark in-degree) {sp(np.log1p(deg0)):+.3f}, CpG o/e {sp(cpg):+.3f}, GC {sp(gc):+.3f}')

# ---------------------------------------------------------------- C. motif-similarity transfer for held-out TFs
xexp = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0).values
Cexp = np.nan_to_num(np.corrcoef(xexp))
r = lambda v: rankdata(v) / len(v)
cov = np.where(np.isfinite(Mb).any(1))[0]
gok = np.isfinite(Mb[cov]).all(0)
Cm = np.corrcoef(Mb[np.ix_(cov, np.where(gok)[0])])  # label-free similarity of motif-score profiles across the promoter genes
covpos = {int(t): i for i, t in enumerate(cov)}
pr(f'\n== C. motif-similarity transfer (similarity = Pearson correlation of {kind} {best.split()[1]} profiles across {int(gok.sum())} genes; {len(cov)} TFs with a motif)')
NAMES = ['popularity', 'co-expression transfer (w5)', 'motif transfer k=3', 'motif transfer k=5', 'motif transfer k=all positive',
         'rank avg pop + co-expr transfer', 'rank avg pop + motif k=3', 'rank avg pop + motif k=5', 'rank avg pop + motif k=all']
ALL, COV, FB = {k: {} for k in NAMES}, {k: {} for k in NAMES}, []
fold_all = {k: [] for k in NAMES}
for fold, (tr, tst, A, deg) in enumerate(folds):
    t, g, y = tst.TF.values, tst.Target.values, tst.Label.values
    tfs = np.unique(tr.TF); ut = np.unique(t)
    pop = np.log1p(deg)[g]
    W = np.clip(Cexp[np.ix_(ut, tfs)], 0, None)
    co = ((W @ A[tfs]) / (W.sum(1, keepdims=True) + 1e-9))[np.searchsorted(ut, t), g]
    trc = [int(s) for s in tfs if int(s) in covpos]
    sc = {'popularity': pop, 'co-expression transfer (w5)': co}
    for kname, kk in [('k=3', 3), ('k=5', 5), ('k=all positive', 99999)]:
        Tm = np.zeros(len(t)); fb = 0
        for u in ut:
            m = t == u
            sim = Cm[covpos[int(u)], [covpos[s] for s in trc]] if int(u) in covpos else np.array([])
            pos = np.where(sim > 0)[0]
            if len(pos) == 0:
                Tm[m] = pop[m]; fb += 1; continue
            top = pos[np.argsort(-sim[pos])[:kk]]
            Tm[m] = (sim[top][:, None] * A[[trc[i] for i in top]][:, g[m]]).sum(0)
        sc['motif transfer ' + kname] = Tm
        if kname == 'k=3': FB.append(fb)
    sc['rank avg pop + co-expr transfer'] = r(pop) + r(co)
    for kn in ('k=3', 'k=5', 'k=all'):
        sc['rank avg pop + motif ' + kn] = r(pop) + r(sc['motif transfer ' + ('k=all positive' if kn == 'k=all' else kn)])
    for nm, s in sc.items():
        d = tf_auc(y, s, t)
        ALL[nm].update(d); COV[nm].update({q: a for q, a in d.items() if q in covpos})
        fold_all[nm].append((roc_auc_score(y, s), np.mean(list(d.values()))))
pr(f'held-out TFs that fell back to popularity (uncovered, or no positively similar training TF with a motif; k=3 run), per fold: {FB} of {[len(np.unique(f[1].TF)) for f in folds]}')
for ttl, D, extra in [('all held-out rows, all TFs (uncovered TFs score by popularity); AUROC and within-TF are fold mean ± std as in w5_heldout_tf_transfer.txt', ALL, True),
                      ('TFs with a motif only, pooled over folds', COV, False)]:
    pr(f'-- {ttl}')
    for nm in NAMES:
        a = np.array(fold_all[nm])
        pr(f'  {nm:34s} ' + (f'AUROC {a[:, 0].mean():.4f} ± {a[:, 0].std():.4f}  within-TF {a[:, 1].mean():.4f} ± {a[:, 1].std():.4f}  ' if extra else f'within-TF {mean(D[nm]):.4f}  ')
           + (f'vs pop {cmp(D[nm], D["popularity"])}' if nm != 'popularity' else f'({len(D[nm])} TFs)'))

# ---------------------------------------------------------------- D. does the motif add to the trained model (benchmark split)?
pr('\n== D. motif added to the trained model on the benchmark split: z(model) + w * z(motif), test rows of covered TFs with a finite motif score, within-TF, 5 seeds')
zs = lambda v: (v - v.mean()) / v.std()
for fam, pat in [('SAGE + Geneformer (w3_herogf)', 'w3_herogf_s{}'), ('tuned SAGE + Geneformer (w8_hESC_tuned)', 'w8_hESC_tuned_s{}')]:
    Z = [np.load(f'analysis/out/pair/{pat.format(s)}.npz') for s in range(42, 47)]
    assert all((z['tf'] == te0.TF.values).all() and (z['tgt'] == te0.Target.values).all() for z in Z)
    m = ok(te0, Mb).values if hasattr(ok(te0, Mb), 'values') else ok(te0, Mb)
    t, g, y = te0.TF.values[m], te0.Target.values[m], te0.Label.values[m]
    mo = Mb[t, g]
    mot_pooled = zs(mo)
    tfz = np.zeros_like(mo)  # per-TF z of the motif score (second variant)
    for u in np.unique(t): tfz[t == u] = zs(mo[t == u]) if mo[t == u].std() > 0 else 0
    pr(f'-- {fam}: {len(y)} rows, {len(np.unique(t))} TFs ({len(tf_auc(y, y * 0. + mo, t))} with both classes)')
    base = [tf_auc(y, z['logit'][m], t) for z in Z]
    B = {q: np.mean([b[q] for b in base]) for q in base[0]}
    wi = [np.mean(list(b.values())) for b in base]
    pr(f'  {"model alone":38s} within-TF {np.mean(wi):.4f} ± {np.std(wi):.4f} over seeds (per-TF AUROC averaged over seeds: {mean(B):.4f})')
    for nm, mz, ws in [('pooled z(motif), w=0.25 (headline, fixed in advance)', mot_pooled, [0.25]),
                       ('pooled z(motif), grid chosen on test, optimistic', mot_pooled, [0.1, 0.5, 1.0]),
                       ('per-TF z(motif), grid chosen on test, optimistic', tfz, [0.1, 0.25, 0.5, 1.0])]:
        for w in ws:
            sd = [tf_auc(y, zs(z['logit'][m]) + w * mz, t) for z in Z]
            Vv = {q: np.mean([b[q] for b in sd]) for q in sd[0]}
            si = [np.mean(list(b.values())) for b in sd]
            pr(f'  {nm:52s} w={w:<4} within-TF {np.mean(si):.4f} ± {np.std(si):.4f}  seeds better {sum(a > b for a, b in zip(si, wi))}/5   vs model alone {cmp(Vv, B)}')
open('analysis/out/w9_motif_r2.txt', 'w').write('\n'.join(out) + '\n')
