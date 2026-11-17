"""W22 fair baselines for complete network inference (label-free inputs; ChIP labels only as source loss and in evaluate()).

1. PANDA (Glass et al. 2013), own numpy port of netZooPy's CPU loop: prior W0 = motif_hits > 0 (TF x gene),
   P0 = STRING TF-TF edges + unit diagonal (or identity: --noppi), C0 = Pearson co-expression of the cell; netZooPy's
   normalize_network, Tanimoto updates, alpha 0.1, stop at hamming < 0.001. Rows = every TF in TF.csv.
2. GENIE3 / GRNBoost2-style regression (SCENIC step 1) with scikit-learn: per target gene, tree ensemble on the
   expression of the TF.csv TFs (target excluded); scores = importances. et: ExtraTrees 500 trees, sqrt features,
   unit-variance output, unnormalised importances (GENIE3); gbm: GradientBoosting lr 0.01, <= 5000 trees,
   max_features 0.1, subsample 0.9, arboreto's OOB early stop (window 25), importance x n_trees (GRNBoost2).
   Step 2 motif pruning: hard (motif_hits > 0 first, importance rank inside each group) and soft (within-TF rank
   average of importance and motif_max). All settings fixed in advance.
3. scRegNet on the motif graph: published GCN 227/29/34, Geneformer joined after the GCN, Linear 198 + LeakyReLU,
   additive decoder; graph = motif_hits > 0 made undirected; GCN input = W20's 12 expression statistics; trained on all
   labelled pairs of the source cell(s), 20% of source TFs held out for validation (W20's split), epoch chosen on
   source-validation within-TF AUROC. --recipe paper (SGD) | tuned (Adam).

  python3.12 -m src.xcell_base --panda hESC [--noppi]
  python3.12 -m src.xcell_base --genie3 hESC --method et|gbm [--n_jobs 12]
  python3.12 -m src.xcell_base --src hESC --tgt hHEP --recipe paper --seed 42 --name X
  python3.12 -m src.xcell_base --jobs analysis/jobs/w22_x.txt --shard 0 --nshard 4
  python3.12 -m src.xcell_base --panda_check hESC     (own PANDA vs W20's saved netZooPy run, same TF set)
  python3.12 -m src.xcell_base --selfcheck
"""
import argparse, ast, copy, inspect, json, math, os, shlex, subprocess, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from src.xcell_gnn import CELLS, cell, evaluate, load_labels, motif_edges, no_labels, path, per_tf_auc
from src.pair_gnn import sym_edges

OUT = 'analysis/out/xcell_base'
RECIPES = {  # analysis/jobs/w4_paper.txt (A_paper) and analysis/jobs/finals.txt (A_tuned)
    'paper': dict(optimizer='SGD', lr=0.0014411386699467767, wd=1.0449337997892844e-05, dropout=0.17477888900593389, batch=193, eval_interval=5, epochs=600),
    'tuned': dict(optimizer='Adam', lr=4.29028e-05, wd=0.002, dropout=0.481, batch=512, eval_interval=1, epochs=300),
}


def save(name, grid, grid_tf, tgt, **meta):
    """W20 format: {name}.npz with grid (TF x gene) and grid_tf; json with the target metrics"""
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f'{OUT}/{name}.npz', grid=grid.astype(np.float32), grid_tf=grid_tf)
    ev = evaluate(tgt, grid_tf, grid)
    res = dict(name=name, tgt=tgt, target=dict(auc=ev['auc'], aupr=ev['aupr'], within=ev['within']), **meta)
    json.dump(res, open(f'{OUT}/{name}.json', 'w'), indent=1)
    print(f"{name}\ttarget AUROC {ev['auc']:.4f} AUPRC {ev['aupr']:.4f} within {ev['within']:.4f}\t{meta.get('minutes', 0):.1f} min", flush=True)


# ---------------------------------------------------------------- 1. PANDA

def normalize_network(x):
    """netZooPy calculations.normalize_network"""
    with np.errstate(all='ignore'):
        nc = (x - x.mean(0)) / x.std(0)
        nr = nc.T if x.shape[0] == x.shape[1] and np.allclose(x, x.T) else (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        nt = (x - x.mean()) / x.std(ddof=1)
    out = (nc + nr) / math.sqrt(2)
    a, b = np.isnan(nc), np.isnan(nr)
    out[a] = (nr[a] + nt[a]) / math.sqrt(2)
    out[b] = (nc[b] + nt[b]) / math.sqrt(2)
    out[a & b] = 2 * nt[a & b] / math.sqrt(2)
    return out


def tani(x, y=None):
    """continuous Tanimoto similarity (netZooPy t_function)"""
    if y is None:
        a = x @ x.T; s = (x * x).sum(1)
        return a / np.sqrt(s + s[:, None] - np.abs(a))
    a = x @ y
    return a / np.sqrt((y * y).sum(0) + (x * x).sum(1)[:, None] - np.abs(a))


def diag_update(m, num, alpha, step):
    np.fill_diagonal(m, np.nan)
    np.fill_diagonal(m, np.nanstd(m, 0) * num * math.exp(2 * alpha * step))
    return m


def panda(W0, P0, C0, alpha=0.1, thr=1e-3, max_steps=2000):
    """W0 (TF x gene) prior, P0 (TF x TF), C0 (gene x gene), all unnormalised; returns the PANDA network and the step count"""
    W, P, C = normalize_network(W0.astype(np.float64)), normalize_network(P0.astype(np.float64)), normalize_network(C0.astype(np.float64))
    step, ham = 0, 1.0
    while ham > thr:
        R = 0.5 * (tani(P, W) + tani(W, C))
        ham = np.abs(W - R).mean()
        W = (1 - alpha) * W + alpha * R
        P = (1 - alpha) * P + alpha * diag_update(tani(W), len(P), alpha, step)
        C = (1 - alpha) * C + alpha * diag_update(tani(W.T), len(C), alpha, step)
        step += 1
        assert np.isfinite(ham) and step < max_steps, f'PANDA did not converge (step {step}, hamming {ham})'
    return W, step


def panda_inputs(D, tfs, noppi):
    """label-free: binary motif prior, STRING TF-TF + diagonal (identity with noppi), Pearson of the raw expression (netZooPy NaN handling)"""
    with no_labels():
        W0 = (np.nan_to_num(D['Mh'][tfs]) > 0).astype(np.float64)
        P0 = np.eye(len(tfs))
        if not noppi:
            pos = {int(t): i for i, t in enumerate(tfs)}
            for u, v in D['P'].T:
                if u in pos and v in pos: P0[pos[u], pos[v]] = P0[pos[v], pos[u]] = 1.0
        C0 = np.corrcoef(D['expr'].values.astype(np.float64))
        if np.isnan(C0).any(): np.fill_diagonal(C0, 1); C0 = np.nan_to_num(C0)
    return W0, P0, C0


def run_panda(c, noppi):
    t0 = time.time(); D = cell(c)
    W, steps = panda(*panda_inputs(D, D['tf'], noppi))
    save(f"w22_panda{'-noppi' if noppi else ''}_{c}", W, D['tf'], c, steps=steps, minutes=(time.time() - t0) / 60)


def panda_check(c):
    """own PANDA vs W20's netZooPy run (analysis/out/xcell/base_<c>.npz) on the TFs netZooPy kept (motif or PPI TFs)"""
    D = cell(c); b = np.load(f'analysis/out/xcell/base_{c}.npz')
    assert (b['grid_tf'] == D['tf']).all()
    keep = b['in_panda']; tfs = D['tf'][keep]
    t0 = time.time(); W, steps = panda(*panda_inputs(D, tfs, False))
    ref = b['panda'][keep]
    r = np.corrcoef(W.ravel(), ref.ravel())[0, 1]
    print(f'{c}: {len(tfs)} TFs, {steps} steps, {time.time() - t0:.1f} s; Pearson r vs netZooPy {r:.6f}; max |diff| {np.abs(W - ref).max():.2e}')
    Wall, _ = panda(*panda_inputs(D, D['tf'], False))
    for nm, g in (('netZooPy (W20)', b['panda']), ('own, all TF rows', Wall)):
        ev = evaluate(c, D['tf'], g); print(f'   {nm:18s} AUROC {ev["auc"]:.4f} AUPRC {ev["aupr"]:.4f} within {ev["within"]:.4f}')


# ---------------------------------------------------------------- 2. GENIE3 / GRNBoost2 + motif pruning

def _fit_one(j, Xtf, tfs, y, method):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    inp = tfs != j  # a TF never regresses on itself
    if method == 'et':
        sd = y.std()
        m = ExtraTreesRegressor(n_estimators=500, max_features='sqrt', random_state=0, n_jobs=1).fit(Xtf[:, inp], y / sd if sd > 0 else y)
        imp = np.mean([e.tree_.compute_feature_importances(normalize=False) for e in m.estimators_], 0)
    else:
        def stop(i, est, _):  # arboreto EarlyStopMonitor(window_length=25) on the OOB improvement
            return i >= 24 and np.mean(est.oob_improvement_[max(0, i - 24):i]) < 0
        m = GradientBoostingRegressor(learning_rate=0.01, n_estimators=5000, max_features=0.1, subsample=0.9, random_state=0)
        m.fit(Xtf[:, inp], y, monitor=stop)
        imp = m.feature_importances_ * m.n_estimators_
    out = np.zeros(len(tfs)); out[inp] = imp
    return out


def importances(D, method, n_jobs):
    from joblib import Parallel, delayed
    with no_labels():
        E = D['expr'].values.astype(np.float64)  # genes x cells, as given (log-normalised); GENIE3 rescales each output
        Xtf = E[D['tf']].T
        cols = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(j, Xtf, D['tf'], E[j], method) for j in range(D['n']))
    return np.stack(cols, 1)  # TF x gene


def within_rank(S):
    """per TF row: average rank scaled to [0, 1)"""
    return np.stack([(rankdata(r) - 1) / len(r) for r in S])


def prune(D, imp):
    """hard: motif-supported pairs (motif_hits > 0) above all others, importance rank inside each group;
    soft: mean of the within-TF ranks of importance and motif_max (no motif = lowest)"""
    Mh = np.nan_to_num(D['Mh'][D['tf']]); Mx = D['Mx'][D['tf']]
    ri = within_rank(imp)
    hard = (Mh > 0) + ri
    soft = 0.5 * (ri + within_rank(np.where(np.isnan(Mx), -np.inf, Mx)))
    return hard, soft


def run_genie3(c, method, n_jobs):
    t0 = time.time(); D = cell(c)
    imp = importances(D, method, n_jobs)
    imp[np.arange(len(D['tf'])), D['tf']] = 0.0
    m = (time.time() - t0) / 60
    hard, soft = prune(D, imp)
    for suf, g in (('', imp), ('-hard', hard), ('-soft', soft)):
        save(f'w22_genie3-{method}{suf}_{c}', g, D['tf'], c, minutes=m)


# ---------------------------------------------------------------- 3. scRegNet on the motif graph

def scr_model(d_in, d_gf, dropout):
    from torch_geometric.nn import GCNConv

    class ScrMotif(nn.Module):
        """src/pair_gnn.py --encoder gcn --decoder add --scfm 1 --gnn_dims 227 29 34 --mlp_dim 198 (scTransNet_GCN)"""
        def __init__(s):
            super().__init__()
            dims = [d_in, 227, 29, 34]
            s.convs = nn.ModuleList(GCNConv(i, o) for i, o in zip(dims[:-1], dims[1:]))
            s.mlp, s.out = nn.Linear(34 + d_gf, 198), nn.Linear(2 * 198, 1)

        def nodes(s, x, gf, A):
            """A = dense GCN-normalised adjacency D^-1/2 (A + I) D^-1/2 (adj()): each layer is GCNConv's own math, A @ lin(h) + bias,
            as one matmul instead of PyG's per-step normalisation and scatter (same result, checked in --selfcheck)"""
            h = x
            for i, c in enumerate(s.convs):
                h = A @ c.lin(h) + c.bias
                if i < len(s.convs) - 1: h = F.dropout(F.relu(h), dropout, s.training)
            return F.leaky_relu(s.mlp(torch.cat([gf, h], 1)))

        def score(s, u, pr):
            return s.out(torch.cat([u[pr[:, 0]], u[pr[:, 1]]], 1)).squeeze(1)
    return ScrMotif()


def scr_inputs(D, dev):
    """label-free node features and graph of one cell: 12 expression statistics (GCN input), Geneformer (joined after the GCN),
    motif graph motif_hits > 0 made undirected (self pairs removed)"""
    with no_labels():
        T = lambda v, dt: torch.as_tensor(v, device=dev, dtype=dt)
        return T(D['X'][:, :12], torch.float32), T(D['X'][:, 12:908], torch.float32), adj(T(sym_edges(motif_edges(D, 0), D['n']), torch.long), D['n'])


def adj(ei, n):
    """dense D^-1/2 (A + I) D^-1/2 exactly as GCNConv normalises (gcn_norm, self loops weight 1)"""
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    e, w = gcn_norm(ei, None, n, add_self_loops=True)
    return torch.zeros(n, n, device=ei.device).index_put_((e[1], e[0]), w, accumulate=True)


def run_scr(a, dev):
    t0 = time.time(); r = {**RECIPES[a.recipe], **({'epochs': a.max_epochs} if a.max_epochs else {})}
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    assert a.tgt not in a.src, 'transfer: the target must not be a source'
    S = []
    for c in a.src:
        D = cell(c); L = load_labels(c)  # source labels: training loss and source validation only
        tfs = np.unique(L.TF.values)
        val = np.random.RandomState(a.seed).choice(tfs, max(1, int(round(0.2 * len(tfs)))), replace=False)  # = src/xcell_gnn.run
        vm = L.TF.isin(val).values
        T = lambda v, dt=torch.long: torch.as_tensor(v, device=dev, dtype=dt)
        X, GF, G = scr_inputs(D, dev)
        S.append(dict(c=c, X=X, GF=GF, G=G, Ptr=T(L[['TF', 'Target']].values[~vm]), Ytr=T(L.Label.values[~vm], torch.float32),
                      Pva=T(L[['TF', 'Target']].values[vm]), yva=L.Label.values[vm], tva=L.TF.values[vm]))
    model = scr_model(12, 896, r['dropout']).to(dev)
    opt = getattr(torch.optim, r['optimizer'])(model.parameters(), lr=r['lr'], weight_decay=r['wd'])

    def val_score():
        model.eval(); pt, ys, ss = {}, [], []
        with torch.no_grad():
            for s in S:
                sv = model.score(model.nodes(s['X'], s['GF'], s['G']), s['Pva']).cpu().numpy()
                pt.update({f"{s['c']}:{t}": v for t, v in per_tf_auc(s['yva'], sv, s['tva']).items()})
        return float(np.mean(list(pt.values())))

    hist, best, bad, state, best_ep = [], -1, 0, None, 0
    for ep in range(1, r['epochs'] + 1):
        model.train()
        bl = [(i, perm[j:j + r['batch']]) for i, s in enumerate(S) for perm in [torch.randperm(len(s['Ptr']), device=dev)] for j in range(0, len(perm), r['batch'])]
        for k in np.random.permutation(len(bl)):
            i, idx = bl[k]; s = S[i]
            loss = F.binary_cross_entropy_with_logits(model.score(model.nodes(s['X'], s['GF'], s['G']), s['Ptr'][idx]), s['Ytr'][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % r['eval_interval']: continue
        v = val_score(); hist.append(v)
        if v > best:
            best, bad, state, best_ep = v, 0, copy.deepcopy(model.state_dict()), ep
        else:
            bad += 1
            if bad >= 10: break
        print(f'ep {ep} val {v:.4f}', flush=True)
    model.load_state_dict(state); model.eval()
    Dt = cell(a.tgt)
    with torch.no_grad():
        X, GF, G = scr_inputs(Dt, dev)
        u = model.nodes(X, GF, G)
        pr = torch.as_tensor(np.array([(t, g) for t in Dt['tf'] for g in range(Dt['n'])]), device=dev)
        grid = torch.cat([model.score(u, pr[i:i + 65536]) for i in range(0, len(pr), 65536)]).cpu().numpy().reshape(len(Dt['tf']), Dt['n'])
    save(a.name, grid, Dt['tf'], a.tgt, args={k: v for k, v in vars(a).items() if k not in ('jobs', 'shard', 'nshard')}, srcs=a.src,
         best_epoch=best_ep, val_within=best, hist=hist, minutes=(time.time() - t0) / 60)


# ---------------------------------------------------------------- checks and driver

def leak_audit():
    """source-level check of this module: load_labels() is called only by run_scr (source cells), and no other code names a
    label file (the target's labels are read only inside the imported evaluate())"""
    tree = ast.parse(inspect.getsource(sys.modules[__name__]))
    readers, named = set(), set()
    for f in tree.body:
        if isinstance(f, ast.FunctionDef) and f.name in ('leak_audit', 'selfcheck'): continue
        for n in ast.walk(f):
            if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'load_labels': readers.add(f.name)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and any(k in n.value for k in ('Train_set', 'Test_set', 'Label.csv', 'BL--network')):
                named.add(getattr(f, 'name', '<module>'))
    assert readers == {'run_scr'} and not named, (readers, named)
    return readers


def selfcheck():
    print('label readers in src/xcell_base.py:', leak_audit())
    with no_labels():
        try:
            pd.read_csv(path('hHEP') + 'Train_set.csv'); raise SystemExit('guard failed')
        except AssertionError:
            pass
    # PANDA port: a converged network is a fixed point of the update; normalisation matches scipy's zscore rule
    rs = np.random.RandomState(0)
    W0 = (rs.rand(6, 20) > 0.7).astype(float); P0 = np.eye(6); P0[0, 1] = P0[1, 0] = 1; C0 = np.corrcoef(rs.randn(20, 50))
    W, st = panda(W0, P0, C0)
    assert W.shape == (6, 20) and np.isfinite(W).all() and st > 1
    from scipy.stats import zscore
    x = rs.rand(5, 7); nn_ = normalize_network(x)
    assert np.allclose(nn_, (zscore(x, axis=0) + zscore(x, axis=1)) / math.sqrt(2))
    # pruning orders: hard puts every motif pair first, soft without a motif = importance rank
    D = cell('mHSC-E'); imp = rs.rand(len(D['tf']), D['n'])
    hard, soft = prune(D, imp)
    Mh = np.nan_to_num(D['Mh'][D['tf']]) > 0
    i = np.where(Mh.any(1) & ~Mh.all(1))[0][0]
    assert hard[i][Mh[i]].min() > hard[i][~Mh[i]].max()
    j = np.where(~D['has_m'][D['tf']])[0][0]
    assert (np.argsort(soft[j]) == np.argsort(imp[j])).all()
    for c in CELLS:
        D = cell(c)
        panda_inputs(D, D['tf'], False); X, GF, A = scr_inputs(D, torch.device('cpu'))  # raise on any label read
        assert X.shape == (D['n'], 12) and GF.shape == (D['n'], 896) and torch.allclose(A, A.T)
        ei = sym_edges(motif_edges(D, 0), D['n'])
        ev = evaluate(c, D['tf'], np.where(np.isnan(D['Mx'][D['tf']]), np.nanmin(D['Mx']) - 1, D['Mx'][D['tf']]))
        print(c, 'ok; motif graph undirected edges', ei.shape[1] // 2, '; motif_max within-TF', round(ev['within'], 4))
    m = scr_model(12, 896, 0.0).eval(); ei = torch.as_tensor(ei)  # dense propagation = PyG GCNConv on the edge list (last cell)
    with torch.no_grad():
        h = X
        for i, c in enumerate(m.convs):
            h = c(h, ei)
            if i < 2: h = F.relu(h)
        ref = F.leaky_relu(m.mlp(torch.cat([GF, h], 1)))
        assert torch.allclose(m.nodes(X, GF, A), ref, atol=1e-4), 'dense GCN differs from GCNConv'
    print('selfcheck passed')


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name'); p.add_argument('--src', nargs='+', default=[]); p.add_argument('--tgt')
    p.add_argument('--recipe', default='paper', choices=list(RECIPES)); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_epochs', type=int, default=0, help='>0: cap the recipe\'s epochs (smoke tests only)')
    p.add_argument('--panda'); p.add_argument('--noppi', action='store_true'); p.add_argument('--panda_check')
    p.add_argument('--genie3'); p.add_argument('--method', default='et', choices=['et', 'gbm']); p.add_argument('--n_jobs', type=int, default=8)
    p.add_argument('--jobs'); p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--job_timeout', type=int, default=5400)
    p.add_argument('--selfcheck', action='store_true')
    return p.parse_args(argv)


def job_name(b):
    if b.name: return b.name
    if b.panda: return f"w22_panda{'-noppi' if b.noppi else ''}_{b.panda}"
    return f'w22_genie3-{b.method}-soft_{b.genie3}'  # written last by run_genie3


def main():
    a = parse()
    if a.selfcheck: return selfcheck()
    if a.panda_check: return panda_check(a.panda_check)
    if a.panda: return run_panda(a.panda, a.noppi)
    if a.genie3: return run_genie3(a.genie3, a.method, a.n_jobs)
    if a.jobs:
        lines = [l for l in open(a.jobs).read().splitlines() if l.strip() and not l.startswith('#')][a.shard::a.nshard]
        for l in lines:
            b = parse(shlex.split(l))
            nm = job_name(b)
            if os.path.exists(f'{OUT}/{nm}.json'): continue
            os.makedirs(OUT, exist_ok=True)
            try:  # dynamic queue: parallel workers on the same file each claim a line once (ada_w22.sh clears stale claims)
                os.close(os.open(f'{OUT}/{nm}.claim', os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            try:
                with open(f'{OUT}/{nm}.out', 'w') as log:  # per-run log; the last line (the result) goes to the job log
                    subprocess.run([sys.executable, '-m', 'src.xcell_base'] + shlex.split(l), timeout=a.job_timeout, stdout=log, stderr=subprocess.STDOUT)
                print(open(f'{OUT}/{nm}.out').read().strip().splitlines()[-1:], flush=True)
            except subprocess.TimeoutExpired:
                print(f'TIMEOUT {nm} after {a.job_timeout} s', flush=True)
        return
    torch.set_num_threads(2)
    run_scr(a, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


if __name__ == '__main__':
    main()
