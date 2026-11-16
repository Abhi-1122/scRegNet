"""W20 "learned PANDA": complete network inference, trained on source cell type(s), applied unchanged to a target.

Inputs per cell are label-free only: node features = 12 per-gene expression statistics (pair_gnn --feat stats),
Geneformer vector (896), is-TF and has-motif flags, each standardised within the dataset; relations
Mf (TF -> motif target), Mr (target -> TF), P (STRING among TFs), optional C (co-expression kNN);
pair features = motif_max, log1p motif_hits (z-scored over the dataset), missing flag, optional TF-gene Pearson.
ChIP labels (Train_set + Test_set) are read only by load_labels(): as the training loss of source cells and inside
evaluate() for the target. build_cell() runs under no_labels(), which fails on any read of a label file.

  python3.12 -m src.xcell_gnn --src hESC --tgt hHEP --name X [--model full|mlp] [--lr ..] [--mk 0|50|100|200] ...
  python3.12 -m src.xcell_gnn --cv_cell hESC --fold 0 --name X ...     (within-cell 5-fold TF-disjoint CV)
  python3.12 -m src.xcell_gnn --jobs analysis/jobs/w20_x.txt --shard 0 --nshard 4   (job lines = the args above)
  python3.12 -m src.xcell_gnn --baselines hHEP     (motif_max, |Pearson|, netZooPy PANDA with the motif prior)
  python3.12 -m src.xcell_gnn --selfcheck
"""
import argparse, builtins, contextlib, copy, glob, io, json, os, shlex, subprocess, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from src.pair_gnn import NPanda, coexp_graph, ppi_graph, feat_variant

CELLS = ['hESC', 'hHEP', 'mESC', 'mHSC-E']
LABEL_FILES = {'Train_set.csv', 'Test_set.csv', 'Label.csv', 'BL--network.csv'}
OUT = 'analysis/out/xcell'
path = lambda c: f'data/{c}/TFs+500/'


@contextlib.contextmanager
def no_labels():
    """any read of a ChIP label file (pd.read_csv, np.load, open) raises inside this block"""
    def guard(f):
        def g(p, *a, **k):
            if isinstance(p, (str, os.PathLike)) and os.path.basename(str(p)) in LABEL_FILES:
                raise AssertionError(f'label file read during label-free construction: {p}')
            return f(p, *a, **k)
        return g
    saved = pd.read_csv, np.load, builtins.open
    pd.read_csv, np.load, builtins.open = guard(saved[0]), guard(saved[1]), guard(saved[2])
    try:
        yield
    finally:
        pd.read_csv, np.load, builtins.open = saved


def load_labels(c):
    """the only reader of ChIP labels: all labelled pairs (Train_set + Test_set)"""
    p = path(c)
    return pd.concat([pd.read_csv(p + 'Train_set.csv', index_col=0), pd.read_csv(p + 'Test_set.csv', index_col=0)], ignore_index=True)


def z(v):
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)


def build_cell(c):
    with no_labels():
        p = path(c)
        tg = pd.read_csv(p + 'Target.csv', index_col=0)
        assert (tg['index'].values == np.arange(len(tg))).all()
        genes = tg.Gene.str.upper().values; n = len(genes)
        ex = pd.read_csv(p + 'BL--ExpressionData.csv', index_col=0)
        assert (ex.index.str.upper().values == genes).all()
        raw = ex.values.astype(np.float64)
        x = StandardScaler().fit_transform(raw.T).T.astype(np.float32)
        stats = feat_variant('stats', x, raw)
        e = pd.read_csv(f'scFM/Geneformer/{c}_500_gene_embeddings.csv').drop_duplicates('Unnamed: 0').set_index('Unnamed: 0')
        fd = pd.read_csv(f'scFM/Geneformer/{c}_500.csv')
        assert (fd.gene.str.upper().values == genes).all()
        gf = e.reindex(fd.ensembl_id.values).fillna(0.0).values.astype(np.float32)  # same as Trainer._get_embeddings (zeros if absent)
        gf = StandardScaler().fit_transform(gf).astype(np.float32)
        tf = pd.read_csv(p + 'TF.csv', index_col=0)['index'].values
        Mx, Mh = np.load(p + 'motif/motif_max.npy'), np.load(p + 'motif/motif_hits.npy')
        has_m = ~np.isnan(Mx).all(1)
        is_tf = np.isin(np.arange(n), tf)
        X = np.hstack([stats, gf, is_tf[:, None], has_m[:, None]]).astype(np.float32)
        P = ppi_graph(p, n); P = P[:, np.isin(P[0], tf) & np.isin(P[1], tf)]
        C = coexp_graph(x, False)
        R = np.nan_to_num(np.corrcoef(x)).astype(np.float32)
        miss = np.isnan(Mx)
        PF = np.stack([np.nan_to_num(z(Mx)), np.nan_to_num(z(np.log1p(Mh))), miss.astype(np.float32), z(R)], -1).astype(np.float32)
    return dict(cell=c, n=n, genes=genes, tf=tf, X=X, P=P, C=C, R=R, Mx=Mx, Mh=Mh, has_m=has_m, PF=PF, expr=ex)


def motif_edges(D, k):
    """TF -> gene motif edges: motif_hits > 0 (k=0) or the top-k genes per TF by motif_max; self pairs removed"""
    Mx = D['Mx']
    if k == 0:
        t, g = np.where(np.nan_to_num(D['Mh']) > 0)
    else:
        rows = np.where(D['has_m'])[0]
        S = np.where(np.isnan(Mx[rows]), -np.inf, Mx[rows]); S[np.arange(len(rows)), rows] = -np.inf
        top = np.argsort(-S, 1)[:, :k]
        t, g = np.repeat(rows, k), top.ravel()
        ok = np.isfinite(S[np.repeat(np.arange(len(rows)), k), g]); t, g = t[ok], g[ok]
    ok = t != g
    return np.stack([t[ok], g[ok]])


def graph(D, a, dev):
    T = lambda v: torch.as_tensor(v, device=dev, dtype=torch.long)
    G = {}
    if 'M' in a.rels:
        m = motif_edges(D, a.mk); G['Mf'] = T(m); G['Mr'] = T(m[::-1].copy())
    if 'P' in a.rels: G['P'] = T(D['P'])
    if 'C' in a.rels: G['C'] = T(D['C'])
    return G


def mean_agg(X, G):
    """per relation: mean of the source features over each node's incoming edges (0 without edges), as SAGEConv's mean aggregation"""
    out = {}
    for r, ei in G.items():
        src, dst = ei
        deg = torch.zeros(len(X), device=X.device).index_add_(0, dst, torch.ones(len(dst), device=X.device)).clamp(min=1)
        out[r] = torch.zeros_like(X).index_add_(0, dst, X[src]) / deg[:, None]
    return out


def pf_cols(a):
    return ([0, 1, 2] if a.motif_pf else []) + ([3] if a.corr_pf else [])


class Net(nn.Module):
    """2 NPanda layers (self term + one mean-SAGE per relation; rels=[] gives the feature-only MLP), Hadamard + pair features -> MLP"""
    def __init__(s, d_in, rels, n_pf, hid, drop):
        super().__init__()
        s.drop = drop
        s.l1, s.l2 = NPanda(d_in, hid, rels), NPanda(hid, hid, rels)
        s.dec = nn.Sequential(nn.Linear(hid + n_pf, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 1))

    def nodes(s, X, G, AX):
        """AX = mean_agg(X, G): layer 1's neighbour means of the fixed input, cached (same as s.l1(X, G), checked in --selfcheck)"""
        h1 = s.l1.self_(X) + sum(c.lin_l(AX[r]) for r, c in s.l1.rel.items())
        return s.l2(F.dropout(F.relu(h1), s.drop, s.training), G)

    def score(s, h, pr, PF):
        t, g = pr[:, 0], pr[:, 1]
        return s.dec(torch.cat([h[t] * h[g], PF[t, g]], 1)).squeeze(1)


def per_tf_auc(y, s, tf):
    return {int(t): float(roc_auc_score(y[tf == t], s[tf == t])) for t in np.unique(tf) if 0 < y[tf == t].sum() < (tf == t).sum()}


def evaluate(cell, grid_tf, grid, tfs=None):
    """the only place a target cell's labels are read: score a (TF x gene) grid on every labelled pair (or the TFs in tfs)"""
    L = load_labels(cell)
    if tfs is not None: L = L[L.TF.isin(tfs)]
    row = {int(t): i for i, t in enumerate(grid_tf)}
    s = grid[L.TF.map(row).values, L.Target.values]; y = L.Label.values
    pt = per_tf_auc(y, s, L.TF.values)
    return dict(auc=float(roc_auc_score(y, s)), aupr=float(average_precision_score(y, s)), within=float(np.mean(list(pt.values()))), per_tf=pt)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name')
    p.add_argument('--src', nargs='+', default=[])
    p.add_argument('--tgt')
    p.add_argument('--cv_cell'); p.add_argument('--fold', type=int, default=0)
    p.add_argument('--model', default='full', choices=['full', 'mlp'])
    p.add_argument('--rels', default='MP', help='letters of M (Mf+Mr), P, C; ignored for --model mlp')
    p.add_argument('--mk', type=int, default=0, help='motif graph: 0 = motif_hits > 0, k = top-k genes per TF by motif_max')
    p.add_argument('--motif_pf', type=int, default=1)
    p.add_argument('--corr_pf', type=int, default=0)
    p.add_argument('--gf', type=int, default=1, help='0: drop the Geneformer columns of the node features (ablation)')
    p.add_argument('--loss', default='bce', choices=['bce', 'rank'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.175)
    p.add_argument('--hid', type=int, default=256)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jobs'); p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--job_timeout', type=int, default=1200, help='--jobs: each job runs in its own process, killed after this many seconds')
    p.add_argument('--baselines'); p.add_argument('--selfcheck', action='store_true')
    return p.parse_args(argv)


_CACHE = {}


def cell(c):
    if c not in _CACHE: _CACHE[c] = build_cell(c)
    return _CACHE[c]


def cv_folds(c):
    """5 TF-disjoint folds of the cell's labelled TFs (same seed convention as pair_gnn.heldout_split)"""
    return np.array_split(np.random.RandomState(7).permutation(np.unique(load_labels(c).TF.values)), 5)


def run(a, dev):
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.cv_cell:
        srcs, tgt = [a.cv_cell], a.cv_cell
        test_tfs = cv_folds(a.cv_cell)[a.fold]
    else:
        srcs, tgt, test_tfs = a.src, a.tgt, None
        assert tgt not in srcs, 'transfer: the target must not be a source'
    rels = [] if a.model == 'mlp' else [r for l in a.rels for r in ({'M': ['Mf', 'Mr']}.get(l, [l]))]
    cols = pf_cols(a)
    fcol = np.r_[0:12, (12 if a.gf else 908):910]  # 12 stats [+ 896 Geneformer] + 2 flags
    S = []
    for c in srcs:
        D = cell(c); L = load_labels(c)  # source labels: training loss only
        if test_tfs is not None:
            L = L[~L.TF.isin(test_tfs)]
        tfs = np.unique(L.TF.values)
        val = np.random.RandomState(a.seed).choice(tfs, max(1, int(round(0.2 * len(tfs)))), replace=False)
        vm = L.TF.isin(val).values
        T = lambda v, dt=torch.long: torch.as_tensor(v, device=dev, dtype=dt)
        S.append(dict(c=c, X=T(D['X'][:, fcol], torch.float32), G=graph(D, argparse.Namespace(**{**vars(a), 'rels': a.rels if rels else ''}), dev),
                      PF=T(D['PF'][..., cols], torch.float32), Ptr=T(L[['TF', 'Target']].values[~vm]), Ytr=T(L.Label.values[~vm], torch.float32),
                      Pva=T(L[['TF', 'Target']].values[vm]), yva=L.Label.values[vm], tva=L.TF.values[vm]))
    for s in S: s['AX'] = mean_agg(s['X'], s['G'])
    if test_tfs is not None:
        assert not set(np.concatenate([np.unique(s['Ptr'][:, 0].cpu().numpy()) for s in S])) & set(test_tfs)
    model = Net(S[0]['X'].shape[1], rels, len(cols), a.hid, a.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def val_score():
        model.eval(); pt = {}
        with torch.no_grad():
            for s in S:
                sv = model.score(model.nodes(s['X'], s['G'], s['AX']), s['Pva'], s['PF']).cpu().numpy()
                pt.update({f"{s['c']}:{t}": v for t, v in per_tf_auc(s['yva'], sv, s['tva']).items()})
        return float(np.mean(list(pt.values())))

    hist, best, bad, state = [], -1, 0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        bl = [(i, perm[j:j + a.batch]) for i, s in enumerate(S) for perm in [torch.randperm(len(s['Ptr']), device=dev)] for j in range(0, len(perm), a.batch)]
        for k in np.random.permutation(len(bl)):
            i, idx = bl[k]; s = S[i]
            pr, y = s['Ptr'][idx], s['Ytr'][idx]
            out = model.score(model.nodes(s['X'], s['G'], s['AX']), pr, s['PF'])
            if a.loss == 'bce':
                loss = F.binary_cross_entropy_with_logits(out, y)
            else:  # within-TF pairwise: every (positive, negative) pair of the same TF in the batch
                m = (pr[:, 0, None] == pr[None, :, 0]) & (y[:, None] == 1) & (y[None, :] == 0)
                if not m.any(): continue
                loss = F.softplus(out[None, :] - out[:, None])[m].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = val_score(); hist.append(v)
        if v > best:
            best, bad, state, best_ep = v, 0, copy.deepcopy(model.state_dict()), ep
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(state); model.eval()
    Dt = cell(tgt)
    with torch.no_grad():
        Xt = torch.as_tensor(Dt['X'][:, fcol], device=dev); Gt = graph(Dt, argparse.Namespace(**{**vars(a), 'rels': a.rels if rels else ''}), dev)
        PFt = torch.as_tensor(Dt['PF'][..., cols], device=dev)
        pr = torch.as_tensor(np.array([(t, g) for t in Dt['tf'] for g in range(Dt['n'])]), device=dev)
        h = model.nodes(Xt, Gt, mean_agg(Xt, Gt))
        grid = torch.cat([model.score(h, pr[i:i + 65536], PFt) for i in range(0, len(pr), 65536)]).cpu().numpy().reshape(len(Dt['tf']), Dt['n'])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f'{OUT}/{a.name}.npz', grid=grid, grid_tf=Dt['tf'])
    ev = evaluate(tgt, Dt['tf'], grid, test_tfs)
    res = dict(name=a.name, args={k: v for k, v in vars(a).items() if k not in ('jobs', 'shard', 'nshard')}, srcs=srcs, tgt=tgt,
               test_tfs=None if test_tfs is None else [int(t) for t in test_tfs], best_epoch=best_ep, val_within=best, hist=hist,
               target=dict(auc=ev['auc'], aupr=ev['aupr'], within=ev['within']), minutes=(time.time() - t0) / 60)
    json.dump(res, open(f'{OUT}/{a.name}.json', 'w'), indent=1)
    print(f"{a.name}\tep {best_ep}\tval {best:.4f}\ttarget AUROC {ev['auc']:.4f} AUPRC {ev['aupr']:.4f} within {ev['within']:.4f}\t{res['minutes']:.1f} min", flush=True)


def baselines(c):
    """motif_max, |Pearson|, and netZooPy PANDA (prior = binary motif_hits > 0, PPI = STRING TF-TF, this cell's expression)"""
    D = cell(c)
    with no_labels():
        from netZooPy.panda import Panda
        g = D['genes']; sym = D['expr'].index.values  # original-case symbols, as in the data files
        Mh = np.nan_to_num(D['Mh'])
        t, q = np.where(Mh > 0)
        motif_df = pd.DataFrame({'source': sym[t], 'target': sym[q], 'weight': 1.0})
        P = D['P']; P = P[:, P[0] < P[1]]
        ppi_df = pd.DataFrame({0: sym[P[0]], 1: sym[P[1]], 2: 1.0})
        with contextlib.redirect_stdout(io.StringIO()):
            W = Panda(D['expr'], motif_df, ppi_df, computing='cpu', precision='double', save_memory=True, remove_missing=False, modeProcess='union').panda_network
        in_panda = np.isin(sym[D['tf']], W.index.values)
        Wp = W.reindex(index=sym[D['tf']], columns=sym, fill_value=0.0).values.astype(np.float32)
    Mx = D['Mx'][D['tf']]
    motif = np.where(np.isnan(Mx), np.nanmin(D['Mx']) - 1, Mx).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f'{OUT}/base_{c}.npz', grid_tf=D['tf'], motif=motif, corr=np.abs(D['R'][D['tf']]), panda=Wp, in_panda=in_panda)
    for k in ('motif', 'corr', 'panda'):
        ev = evaluate(c, D['tf'], {'motif': motif, 'corr': np.abs(D['R'][D['tf']]), 'panda': Wp}[k])
        print(f'{c}\t{k}\tAUROC {ev["auc"]:.4f} AUPRC {ev["aupr"]:.4f} within {ev["within"]:.4f}', flush=True)


def selfcheck(dev):
    with no_labels():
        try:
            pd.read_csv(path('hHEP') + 'Train_set.csv'); raise SystemExit('guard failed')
        except AssertionError:
            pass
    for c in CELLS:
        D = build_cell(c)  # raises if any label file is read
        assert D['X'].shape[1] == 12 + 896 + 2, D['X'].shape
        assert np.isfinite(D['X']).all() and np.isfinite(D['PF']).all()
        L = load_labels(c)
        grid = np.zeros((len(D['tf']), D['n']), np.float32)
        row = {t: i for i, t in enumerate(D['tf'])}
        grid[L.TF.map(row).values, L.Target.values] = L.Label.values
        ev = evaluate(c, D['tf'], grid)
        assert abs(ev['within'] - 1) < 1e-12 and abs(ev['auc'] - 1) < 1e-12, ev
        print(c, 'ok', D['X'].shape, 'P edges', D['P'].shape[1], 'Mf edges (hits>0)', motif_edges(D, 0).shape[1])
    a = parse(['--src', 'hHEP', '--tgt', 'hESC'])
    G = graph(D, a, torch.device('cpu')); Xt = torch.as_tensor(D['X'])
    net = Net(Xt.shape[1], ['Mf', 'Mr', 'P'], 3, 256, 0.175).eval()
    with torch.no_grad():
        ref = net.l2(F.relu(net.l1(Xt, G)), G)
        assert torch.allclose(net.nodes(Xt, G, mean_agg(Xt, G)), ref, atol=1e-4), 'cached aggregation differs from NPanda'
    print('selfcheck passed')


def main():
    a = parse()
    torch.set_num_threads(2)  # 4 parallel jobs on 16 cores; two phase-1 shards hung spinning with the default 10 threads each
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if a.selfcheck: return selfcheck(dev)
    if a.baselines: return baselines(a.baselines)
    if a.jobs:
        lines = [l for l in open(a.jobs).read().splitlines() if l.strip()][a.shard::a.nshard]
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        for l in lines:
            b = parse(shlex.split(l))
            if os.path.exists(f'{OUT}/{b.name}.json'): continue
            try:
                subprocess.run([sys.executable, '-m', 'src.xcell_gnn'] + shlex.split(l), env=env, timeout=a.job_timeout)
            except subprocess.TimeoutExpired:
                print(f'TIMEOUT {b.name} after {a.job_timeout} s', flush=True)
        return
    run(a, dev)


if __name__ == '__main__':
    main()
