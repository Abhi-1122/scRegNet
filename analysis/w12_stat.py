"""W12 statistical network models for GRN link prediction (CPU only). Run: python3.12 analysis/w12_stat.py --cell hESC [--protocol gene --fold f]
Every (TF, gene) pair is scored by a model fitted on the TRAINING labelled pairs only (unlabelled pairs are missing, not zero).
Knobs are chosen on the validation slice (same slice as the GNN runs: val_frac 0.1, split_seed 0), mean over seeds of the within-TF AUROC;
the chosen config is refit on all training rows and the test rows are scored once. Seeds change GMM / gradient initialisation.
Saved to analysis/out/stat/: <tag>_<model>_s<seed>.npz (+ _meta.json). The tables are made by analysis/w12_stat_report.py."""
import argparse, itertools, json, os, sys, time, warnings
from types import SimpleNamespace as NS
import numpy as np
import torch
sys.path.insert(0, '.')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from src.pair_gnn import prepare, metrics, within_tf

warnings.filterwarnings('ignore')
torch.set_num_threads(2)
OUT = 'analysis/out/stat/'
SEEDS = (42, 43, 44)
KS = (5, 10, 20, 40)


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------- data
class Split:
    """train matrices (T x n, values and observed mask) + the rows to score"""
    def __init__(s, tr_rows, ev_rows, tfids, n):
        pos = {t: i for i, t in enumerate(tfids)}
        s.T, s.n, s.tfids = len(tfids), n, tfids
        ti = np.array([pos[t] for t in tr_rows.TF.values])
        s.Y = np.zeros((s.T, n), np.float32); s.M = np.zeros((s.T, n), np.float32)
        s.Y[ti, tr_rows.Target.values] = tr_rows.Label.values; s.M[ti, tr_rows.Target.values] = 1
        s.ev_ti = np.array([pos[t] for t in ev_rows.TF.values]); s.ev_g = ev_rows.Target.values
        s.ev_y = ev_rows.Label.values; s.ev_tf = ev_rows.TF.values
        s.P, s.L = s.Y.sum(0), s.M.sum(0)  # gene in-degree and number of labelled pairs, in the training rows
        s.m = s.Y.sum() / s.M.sum()
        s.rt = (s.Y.sum(1) + 5 * s.m) / (s.M.sum(1) + 5)  # smoothed positive rate of every TF

    def score(s, grid):
        return grid[s.ev_ti, s.ev_g]

    def within(s, grid):
        return within_tf(s.ev_y, s.score(grid), s.ev_ti)

    def metrics(s, grid):
        return metrics(s.ev_y, s.score(grid), s.ev_tf)


def load(a):
    """val-mode split (train = train rows minus the validation slice, eval = validation) and full-mode split (train = all, eval = test)"""
    base = dict(cell_type=a.cell, num_TF='500', scfm_folder='scFM', data_folder='data', protocol=a.protocol, fold=a.fold, split_seed=0, emb_feat='none', feat='full')
    D = prepare(NS(**base, scfm=1, val_frac=0.1), 'cpu')
    D2 = prepare(NS(**base, scfm=0, val_frac=0.0), 'cpu')
    tr, te, n = D['tr'], D['te'], D['n']
    tfids = np.unique(tr.TF.values)
    assert set(te.TF.values) <= set(tfids)
    val = Split(tr.iloc[D['tri']], tr.iloc[D['vai']], tfids, n)
    full = Split(D2['tr'], te, tfids, n)
    return D, val, full


# ---------------------------------------------------------------- features / clusters
def gene_feats(D):
    sc = lambda z: z / np.sqrt(z.var(0).sum() / z.shape[1])  # mean per-dimension variance 1
    ex = sc(PCA(40, random_state=0).fit_transform(D['x']))
    gf = sc(PCA(30, random_state=0).fit_transform(D['f']))
    return dict(gf=gf, ex=ex)


_R = {}
def posterior(feat, F, cov, K, seed):
    key = (feat, cov, K, seed, len(F))
    if key not in _R:
        g = GaussianMixture(K, covariance_type=cov, reg_covar=0.05 if cov == 'full' else 1e-3, n_init=1, max_iter=100, random_state=seed)
        _R[key] = g.fit(F).predict_proba(F).astype(np.float64)
    return _R[key]


# ---------------------------------------------------------------- model 1: DC-SBM with fixed gene clusters
def theta(S, R, alpha, shrink):
    """smoothed positive rate of each gene among its labelled training pairs; prior mean = global rate or the soft-cluster rate"""
    if shrink == 'cluster':
        rk = (R.T @ S.P + S.m) / (R.T @ S.L + 1)  # cluster rate with a weak pull to the global rate
        prior = R @ rk
    else:
        prior = S.m
    return (S.P + alpha * prior) / (S.L + alpha)


def dcsbm_parts(S, R, b):
    num, den = S.Y @ R, S.M @ R
    return (num + b * S.rt[:, None]) / (den + b)  # omega (T x K): smoothed rate of TF t among its labelled pairs in cluster k


def dcsbm_grid(S, R, cfg, part='full', parts=None):
    th = theta(S, R, cfg['alpha'], cfg['shrink'])
    om = dcsbm_parts(S, R, cfg['b']) if parts is None else parts
    if cfg['dc']:
        om = om / ((R.T @ th) / R.sum(0).clip(1e-9))[None, :]
    prog = np.log((om @ R.T).clip(1e-9))
    if part == 'theta': return np.log(th)[None] + np.log(S.rt)[:, None], om, th  # TF rate added so pooled AUROC is comparable; within-TF unchanged
    if part == 'omega': return prog, om, th
    return np.log(th)[None] + prog, om, th


def hard(R):
    H = np.zeros_like(R); H[np.arange(len(R)), R.argmax(1)] = 1
    return H


def search_dcsbm(val, F, feats, ks, covs, log_):
    """val within-TF for every config; V[(feat, soft, cov, K, alpha, shrink, b, dc)] -> list over seeds"""
    V = {}
    for feat, cov, K in itertools.product(feats, covs, ks):
        for sd in SEEDS:
            R0 = posterior(feat, F[feat], cov, K, sd)
            for soft in (1, 0):
                R = R0 if soft else hard(R0)
                for b in (1, 3, 10, 30):
                    om0 = dcsbm_parts(val, R, b)
                    for alpha, shrink, dc in itertools.product((1, 3, 10), ('global', 'cluster'), (0, 1)):
                        cfg = dict(alpha=alpha, shrink=shrink, b=b, dc=dc)
                        V.setdefault((feat, soft, cov, K, alpha, shrink, b, dc), []).append(val.within(dcsbm_grid(val, R, cfg, parts=om0)[0]))
        log_(f'  dcsbm val search {feat} {cov} K={K} done')
    return V


# ---------------------------------------------------------------- models 2-4 (torch, full batch Adam on the observed training entries)
def fit_torch(kind, S, hp, seed, ckpt, feats=None):
    """returns {step: T x n score grid}. kind: sbm | hpf | lsm"""
    torch.manual_seed(seed)
    T, n, K = S.T, S.n, hp['k']
    Y, M = torch.tensor(S.Y), torch.tensor(S.M)
    No = M.sum()
    l2 = hp['l2']
    a = torch.zeros(T, 1); b = torch.zeros(1, n)
    a.data = torch.logit(torch.tensor(S.rt, dtype=torch.float32))[:, None].clone() if kind != 'hpf' else torch.log(torch.tensor(S.rt, dtype=torch.float32))[:, None].clone()
    tfids = torch.tensor(S.tfids)
    if kind == 'sbm':
        U = torch.randn(n, K) * 0.5; W = torch.randn(T, K) * 0.1; ps = [a, b, U, W]
    elif kind == 'hpf':
        Th = torch.randn(T, K) * 0.1 - 2.0; Ph = torch.randn(n, K) * 0.1 - 2.0; ps = [a, b, Th, Ph]
    else:
        Z = torch.randn(n, K) * 0.5; ps = [a, b, Z]
    for p in ps: p.requires_grad_(True)
    opt = torch.optim.Adam(ps, lr=hp.get('lr', 0.03))

    def logit():
        if kind == 'sbm':
            return a + b + torch.log((torch.exp(W) @ torch.softmax(U, 1).T).clamp_min(1e-9))
        if kind == 'hpf':
            th, ph = torch.nn.functional.softplus(Th), torch.nn.functional.softplus(Ph)
            return a + b + torch.log1p(th @ ph.T)  # log mu, mu = exp(a_t + b_g) (1 + theta_t . phi_g)
        d = Z[tfids][:, None, :] - Z[None, :, :]
        return a + b - torch.sqrt((d * d).sum(-1) + 1e-8)

    out = {}
    for step in range(1, max(ckpt) + 1):
        opt.zero_grad()
        s = logit()
        if kind == 'hpf':
            nll = (M * (torch.exp(s) - Y * s)).sum()
            pen = l2 * ((b ** 2).sum() + (torch.nn.functional.softplus(Th) ** 2).sum() + (torch.nn.functional.softplus(Ph) ** 2).sum())
        else:
            nll = (M * torch.nn.functional.softplus(-(2 * Y - 1) * s)).sum()  # Bernoulli NLL
            if kind == 'sbm': pen = l2 * ((b ** 2).sum() + (W ** 2).sum()) + 0.1 * (U ** 2).sum()
            else: pen = l2 * ((b ** 2).sum() + (Z ** 2).sum())
        ((nll + pen) / No).backward()
        opt.step()
        if step in ckpt:
            with torch.no_grad():
                g = logit().numpy().copy()
            if kind == 'lsm' and feats is not None and (S.L == 0).any():
                g = lsm_unseen(S, a.detach().numpy(), b.detach().numpy()[0], Z.detach().numpy(), feats, hp['ra'], tfids.numpy())
            out[step] = g
    return out


def lsm_unseen(S, a, b, Z, F, ra, tfids):
    """genes without labelled training pairs: z_g and b_g predicted from the gene features by ridge fitted on the seen genes"""
    seen = S.L > 0
    X = F - F[seen].mean(0)
    tgt = np.concatenate([Z, b[:, None]], 1)
    pred = Ridge(alpha=ra).fit(X[seen], tgt[seen]).predict(X)
    Z2, b2 = Z.copy(), b.copy()
    Z2[~seen] = pred[~seen, :-1]; b2[~seen] = pred[~seen, -1]
    d = Z2[tfids][:, None, :] - Z2[None, :, :]
    return a + b2[None] - np.sqrt((d * d).sum(-1) + 1e-8)


def search_torch(kind, val, ks, l2s, steps, feats=None, ras=(None,)):
    V = {}  # (k, l2, ra, step) -> list over seeds
    ckpt = list(steps)
    for k, l2, ra in itertools.product(ks, l2s, ras):
        hp = dict(k=k, l2=l2, ra=ra)
        for sd in SEEDS:
            for st, g in fit_torch(kind, val, hp, sd, ckpt, feats).items():
                V.setdefault((k, l2, ra, st), []).append(val.within(g))
    return V


# ---------------------------------------------------------------- driver
def best(V, filt=lambda k: True):
    ks = [k for k in V if filt(k)]
    k = max(ks, key=lambda q: np.mean(V[q]))
    return k, float(np.mean(V[k]))


def save(tag, name, sd, full, grid, meta, extra=None):
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f'{OUT}{tag}_{name}_s{sd}.npz', grid=grid.astype(np.float32), grid_tf=full.tfids, logit=full.score(grid), tf=full.ev_tf, tgt=full.ev_g, y=full.ev_y, **(extra or {}))
    m = full.metrics(grid)
    meta = {**meta, 'test': m, 'seed': sd}
    json.dump(meta, open(f'{OUT}{tag}_{name}_s{sd}.json', 'w'), default=str)
    return m


def run(a):
    global SEEDS
    if a.protocol == 'gene': SEEDS = (42, 43)  # held-out protocols use 2 seeds
    tag = a.cell if a.protocol == 'random' else f'{a.cell}_hog{a.fold}'
    t0 = time.time()
    D, val, full = load(a)
    F = gene_feats(D)
    gene = a.protocol == 'gene'
    log(f'{tag}: n={val.n} T={val.T} train labelled pairs val-mode {int(val.M.sum())} full {int(full.M.sum())} eval rows {len(val.ev_y)}/{len(full.ev_y)}')
    summary = {}
    row = lambda nm, ms, v: log(f'  {tag:12s} {nm:28s} val within {v:.4f}  test within {np.mean([m["within"] for m in ms]):.4f}  auc {np.mean([m["auc"] for m in ms]):.4f}  aupr {np.mean([m["aupr"] for m in ms]):.4f}')

    only = set(a.only.split(',')) if a.only else None
    if not gene and not only:
        # popularity controls
        g_in = np.broadcast_to(full.P[None], (full.T, full.n))
        ms = [save(tag, 'pop_indeg', sd, full, g_in, dict(cfg='training in-degree')) for sd in SEEDS]
        vm = val.within(np.broadcast_to(val.P[None], (val.T, val.n)))
        row('pop_indeg', ms, vm)
        Vp = {al: val.within(np.broadcast_to(np.log(theta(val, None, al, 'global'))[None], (val.T, val.n))) for al in (0.3, 1, 3, 10, 30)}
        al = max(Vp, key=Vp.get)
        ms = [save(tag, 'pop', sd, full, np.log(theta(full, None, al, 'global'))[None] + np.log(full.rt)[:, None], dict(cfg=dict(alpha=al), val=Vp[al])) for sd in SEEDS]
        row('pop', ms, Vp[al])

    # model 1: DC-SBM with fixed GMM clusters
    feats = ('gf',) if gene else ('gf', 'ex')
    V1 = search_dcsbm(val, F, feats, KS, ('diag', 'full'), log) if not only else {}
    for feat in (feats if not only else ()):
        for soft in (1, 0):
            fam = f'dcsbm_{feat}_{"soft" if soft else "hard"}'
            fil = lambda k, feat=feat, soft=soft: k[0] == feat and k[1] == soft
            perK = {K: best(V1, lambda k, K=K: fil(k) and k[3] == K) for K in KS}
            kb = max(perK, key=lambda K: perK[K][1])
            for K in KS:
                (_, _, cov, _, alpha, shrink, b, dc), v = perK[K]
                cfg = dict(alpha=alpha, shrink=shrink, b=b, dc=dc)
                ms, ms_o, ms_t = [], [], []
                for sd in SEEDS:
                    R0 = posterior(feat, F[feat], cov, K, sd)
                    R = R0 if soft else hard(R0)
                    meta = dict(cfg=dict(**cfg, cov=cov, K=K, feat=feat, soft=soft), val=v, val_seeds=V1[perK[K][0]])
                    g, om, th = dcsbm_grid(full, R, cfg)
                    ms.append(save(tag, f'{fam}_K{K}', sd, full, g, meta, dict(omega=om, R=R.astype(np.float32), theta=th, P=full.P, L=full.L)))
                    if K == kb:
                        save(tag, fam, sd, full, g, meta, dict(omega=om, R=R.astype(np.float32), theta=th, P=full.P, L=full.L))
                        ms_o.append(save(tag, fam + '_omegaonly', sd, full, dcsbm_grid(full, R, cfg, 'omega')[0], meta))
                        ms_t.append(save(tag, fam + '_thetaonly', sd, full, dcsbm_grid(full, R, cfg, 'theta')[0], meta))
                row(f'{fam} K={K}' + (' *' if K == kb else ''), ms, v)
                if K == kb:
                    row(fam + ' omega-only', ms_o, v); row(fam + ' theta-only', ms_t, v)

    # model 2: DC-SBM with learned (soft) gene blocks
    steps = tuple(range(50, 651, 50))
    Fg = F['gf']
    models = [('dcsbm_learned', 'sbm', (5, 10, 20), (1, 3, 10, 30, 100)), ('lsm', 'lsm', (2, 4, 8), (1, 3, 10, 30, 100))]
    if not gene: models.insert(1, ('hpf', 'hpf', (4, 8, 16), (0.03, 0.1, 0.3, 1)))
    if gene and a.skip_lsm: models = []
    if only: models = [('maineff', 'sbm', (1,), (1, 3, 10, 30, 100))] if 'maineff' in only else []  # K=1: no program term, logit = a_t + b_g (+ TF constant)
    for name, kind, ks, l2s in models:
        ras = (10, 100, 1000) if (gene and kind == 'lsm') else (None,)
        if gene and kind == 'sbm': continue  # no way to place an unseen gene in a learned block
        V = search_torch(kind, val, ks, l2s, steps if kind != 'hpf' else tuple(range(50, 1001, 50)), Fg if (gene and kind == 'lsm') else None, ras)
        (k, l2, ra, st), v = best(V)
        log(f'  {tag} {name}: val best k={k} l2={l2} ra={ra} steps={st} val within {v:.4f}')
        ms = []
        for sd in SEEDS:
            g = fit_torch(kind, full, dict(k=k, l2=l2, ra=ra), sd, [st], Fg if (gene and kind == 'lsm') else None)[st]
            ms.append(save(tag, name, sd, full, g, dict(cfg=dict(k=k, l2=l2, ra=ra, steps=st), val=v, val_seeds=V[(k, l2, ra, st)])))
        row(name, ms, v)
    log(f'{tag} done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='hESC')
    p.add_argument('--protocol', default='random', choices=['random', 'gene'])
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--skip_lsm', type=int, default=0, help='W17: gene protocol, stop after the fixed-cluster DC-SBM (skip the learned latent-space model)')
    p.add_argument('--only', default='', help='comma list; maineff = Bernoulli main-effects control (K=1 block model)')
    run(p.parse_args())
