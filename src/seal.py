"""SEAL-style subgraph classifier for TF -> target scoring (Zhang & Chen, NeurIPS 2018), same harness as src.pair_gnn.

Each pair (t, g) is classified from the --hops enclosing subgraph of the training graph (neighbours capped at --cap per node)
with the undirected edge t-g REMOVED for every pair (train, val and test alike), so the pair's own edge is never visible.
Node input: DRNL one-hot (double-radius node labelling), log in/out degree (own edge subtracted for t and g),
PCA(--pca) of the standardised expression. 3 x (SAGE|GCN), then pooling (mm = mean+max, tt = mm + the two target nodes, sort = SortPooling), MLP.
Subgraphs are extracted once per fit (the graph does not change across batches). Usage: python3.12 -m src.seal --name X --hops 1
"""
import argparse, json, multiprocessing as mp, os, sys, time
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import SortAggregation
from sklearn.decomposition import PCA
from src.pair_gnn import prepare, metrics

G = {}  # extraction state (shared with forked workers)
MAXZ = 30


def drnl(sub, k):
    """double-radius node labels of a subgraph whose nodes 0, 1 are the pair; distance to one end is computed with the other end removed"""
    z = np.zeros(k, dtype=np.int64)
    d = []
    for a, b in ((0, 1), (1, 0)):
        keep = np.array([i for i in range(k) if i != b])
        dd = shortest_path(sub[keep][:, keep], directed=False, unweighted=True, indices=int(np.where(keep == a)[0][0]))
        full = np.full(k, np.inf); full[keep] = dd; d.append(full)
    dx, dy = d
    ok = np.isfinite(dx) & np.isfinite(dy)
    s = np.where(ok, dx + dy, 0).astype(np.int64)
    zz = 1 + np.minimum(np.where(ok, dx, 0), np.where(ok, dy, 0)).astype(np.int64) + (s // 2) * (s // 2 + s % 2 - 1)
    z[ok] = np.minimum(zz[ok], MAXZ)
    z[0] = z[1] = 1
    return z


def extract(pair):
    t, g = int(pair[0]), int(pair[1])
    A, NB, hops, cap = G['A'], G['NB'], G['hops'], G['cap']
    nodes, seen, frontier = [t, g], {t, g}, [t, g]
    for h in range(hops):
        new = []
        for v in frontier:
            ex = {t: g, g: t}.get(v, -1) if h == 0 else -1  # own edge never used to grow the subgraph
            nb = NB[v]; nb = nb[nb != ex][:cap]
            for u in nb:
                if u not in seen: seen.add(u); new.append(u)
        nodes += new; frontier = new
    k = len(nodes)
    sub = A[nodes][:, nodes].tolil()
    sub[0, 1] = 0; sub[1, 0] = 0
    sub = sub.tocsr(); sub.eliminate_zeros()
    r, c = sub.nonzero()
    nodes = np.array(nodes)
    df = np.stack([np.log1p(G['out'][nodes]), np.log1p(G['inn'][nodes])], 1).astype(np.float32)
    e = G['own'].get(min(t, g) * G['n'] + max(t, g))  # (tf, target) of the removed edge, if the pair's edge is in the graph
    if e is not None:
        df[0, 0] = np.log1p(G['out'][t] - (e[0] == t)); df[0, 1] = np.log1p(G['inn'][t] - (e[1] == t))
        df[1, 0] = np.log1p(G['out'][g] - (e[0] == g)); df[1, 1] = np.log1p(G['inn'][g] - (e[1] == g))
    return nodes, drnl(sub, k) if G['drnl'] else np.where(np.arange(k) < 2, 1, 2), df, np.stack([r, c]).astype(np.int32)


def build_store(P, dev, workers):
    with mp.get_context('fork').Pool(workers) as pool:
        out = pool.map(extract, P, chunksize=64)
    nn_ = np.array([len(o[0]) for o in out]); ne = np.array([o[3].shape[1] for o in out])
    S = dict(gnode=np.concatenate([o[0] for o in out]), z=np.concatenate([o[1] for o in out]), df=np.concatenate([o[2] for o in out]), ei=np.concatenate([o[3] for o in out], 1),
             nptr=np.r_[0, np.cumsum(nn_)], eptr=np.r_[0, np.cumsum(ne)])
    return {k: torch.as_tensor(v.astype(np.int64) if v.dtype in (np.int32,) else v, device=dev) for k, v in S.items()} | dict(mean_nodes=float(nn_.mean()), mean_edges=float(ne.mean()))


def batch(S, ids):
    nl = S['nptr'][ids + 1] - S['nptr'][ids]; no = torch.cumsum(nl, 0) - nl
    nidx = torch.repeat_interleave(S['nptr'][ids] - no, nl) + torch.arange(int(nl.sum()), device=ids.device)
    el = S['eptr'][ids + 1] - S['eptr'][ids]; eo = torch.cumsum(el, 0) - el
    eidx = torch.repeat_interleave(S['eptr'][ids] - eo, el) + torch.arange(int(el.sum()), device=ids.device)
    ei = S['ei'][:, eidx] + torch.repeat_interleave(no, el)
    bvec = torch.repeat_interleave(torch.arange(len(ids), device=ids.device), nl)
    return S['gnode'][nidx], S['z'][nidx], S['df'][nidx], ei, bvec, no


class SEAL(nn.Module):
    def __init__(s, a, Xp):
        super().__init__()
        s.a, s.Xp, d = a, Xp, a.hidden
        s.zemb = nn.Embedding(MAXZ + 3, 32)
        din = 32 + (2 if a.deg else 0) + (Xp.shape[1] if a.expr else 0)
        s.inp = nn.Linear(din, d)
        mk = (lambda: SAGEConv(d, d)) if a.conv == 'sage' else (lambda: GCNConv(d, d))
        s.convs = nn.ModuleList([mk() for _ in range(a.layers)])
        s.sort = SortAggregation(a.k) if a.pool == 'sort' else None
        dp = {'mm': 2 * d, 'tt': 5 * d, 'sort': a.k * d}[a.pool]
        s.head = nn.Sequential(nn.Linear(dp, d), nn.ReLU(), nn.Dropout(a.dropout), nn.Linear(d, 1))

    def forward(s, S, ids):
        gn, z, df, ei, bvec, no = batch(S, ids)
        parts = [s.zemb(z)] + ([df] if s.a.deg else []) + ([s.Xp[gn]] if s.a.expr else [])
        h = F.relu(s.inp(torch.cat(parts, 1)))
        for c in s.convs:
            h = F.dropout(F.relu(c(h, ei)), s.a.dropout, s.training)
        B = len(ids)
        if s.a.pool == 'sort': return s.head(s.sort(h, bvec, dim_size=B)).squeeze(1)
        p = [global_mean_pool(h, bvec, B), global_max_pool(h, bvec, B)]
        if s.a.pool == 'tt': p += [h[no], h[no + 1], h[no] * h[no + 1]]
        return s.head(torch.cat(p, 1)).squeeze(1)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cell_type', default='hESC')
    p.add_argument('--num_TF', default='500')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hops', type=int, default=1)
    p.add_argument('--cap', type=int, default=30, help='max neighbours kept per node (fixed random subset)')
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--conv', default='sage', choices=['sage', 'gcn'])
    p.add_argument('--pool', default='tt', choices=['mm', 'tt', 'sort'])
    p.add_argument('--k', type=int, default=20, help='SortPooling size')
    p.add_argument('--drnl', type=int, default=1, help='0: only a target-node marker instead of DRNL labels')
    p.add_argument('--expr', type=int, default=1, help='0: no expression features (structure only)')
    p.add_argument('--deg', type=int, default=1, help='0: no degree features')
    p.add_argument('--pca', type=int, default=32)
    p.add_argument('--neg_frac', type=float, default=1.0, help='fraction of training negatives kept per epoch-fixed subsample')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--optimizer', default='Adam')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--eval_interval', type=int, default=1)
    p.add_argument('--select_metric', default='auc', choices=['auc', 'within'])
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--split_seed', type=int, default=0)
    p.add_argument('--refit', type=int, default=1)
    p.add_argument('--protocol', default='random', choices=['random', 'gene', 'tf'])
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--scfm', type=int, default=0)
    p.add_argument('--data_folder', default='data')
    p.add_argument('--scfm_folder', default='scFM')
    p.add_argument('--out', default='analysis/out/pair')
    return p.parse_args()


def setup_graph(a, D):
    n = D['n']; U0, U1 = D['U']; O = D['orient']
    A = sp.csr_matrix((np.ones(2 * len(U0), dtype=np.float32), (np.r_[U0, U1], np.r_[U1, U0])), shape=(n, n))
    rs = np.random.RandomState(0)
    NB = [rs.permutation(A.indices[A.indptr[v]:A.indptr[v + 1]]) for v in range(n)]
    G.update(A=A, NB=NB, hops=a.hops, cap=a.cap, n=n, drnl=a.drnl, out=np.bincount(O[:, 0], minlength=n), inn=np.bincount(O[:, 1], minlength=n),
             own={int(min(t, g) * n + max(t, g)): (int(t), int(g)) for t, g in O})


def fit(a, D, dev, epochs, select):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr, te, n, tri, vai = D['tr'], D['te'], D['n'], D['tri'], D['vai']
    setup_graph(a, D)
    Xp = torch.as_tensor(PCA(a.pca, random_state=0).fit_transform(D['x']).astype(np.float32), device=dev)
    Xp = Xp / Xp.std()
    pr = lambda df, rows: df[['TF', 'Target']].values[rows]
    t0 = time.time()
    S_tr = build_store(pr(tr, tri), dev, a.workers)
    S_va = build_store(pr(tr, vai), dev, a.workers) if len(vai) else None
    S_te = build_store(te[['TF', 'Target']].values, dev, a.workers)
    print(f'extracted in {time.time() - t0:.0f}s, mean nodes {S_tr["mean_nodes"]:.1f} edges {S_tr["mean_edges"]:.1f}', flush=True)
    Y_tr = torch.as_tensor(tr.Label.values[tri], dtype=torch.float32, device=dev)
    y_va = tr.Label.values[vai]; tf_va = tr.TF.values[vai]; y_te = te.Label.values; tf_te = te.TF.values
    keep = np.arange(len(tri))
    if a.neg_frac < 1:  # fixed subsample of training negatives
        neg = np.where(tr.Label.values[tri] == 0)[0]
        keep = np.sort(np.r_[np.where(tr.Label.values[tri] == 1)[0], np.random.RandomState(a.seed).choice(neg, int(a.neg_frac * len(neg)), replace=False)])
    keep = torch.as_tensor(keep, device=dev)
    model = SEAL(a, Xp).to(dev)
    opt = getattr(torch.optim, a.optimizer)(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def score(S):
        model.eval(); N = len(S['nptr']) - 1; o = []
        with torch.no_grad():
            for i in range(0, N, 1024): o.append(model(S, torch.arange(i, min(i + 1024, N), device=dev)))
        return torch.cat(o).cpu().numpy()

    def evaluate(with_val=True):
        st = score(S_te)
        return (metrics(y_va, score(S_va), tf_va) if with_val else None), metrics(y_te, st, tf_te), st

    hist, best, bad = [], None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = keep[torch.randperm(len(keep), device=dev)]
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size]
            loss = F.binary_cross_entropy_with_logits(model(S_tr, idx), Y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if select and ep % a.eval_interval == 0:
            v, t, _ = evaluate()
            hist.append(dict(epoch=ep, val=v, test=t))
            if best is None or v[a.select_metric] > best['val'][a.select_metric]:
                best, bad = hist[-1], 0
            else:
                bad += 1
                if bad >= a.patience: break
    if select: return hist, best
    _, t, st = evaluate(with_val=False)
    return t, (st, None, None)


def main():
    a = parse()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    D = prepare(a, dev)
    hist, best = fit(a, D, dev, a.epochs, select=True)
    os.makedirs(a.out, exist_ok=True)
    tsel = max(hist, key=lambda h: h['test']['auc'])
    res = dict(name=a.name, args=vars(a), best_val_epoch=best['epoch'], val=best['val'], test=best['test'], test_selected=tsel['test'], history=hist)
    if a.refit:
        a2 = SimpleNamespace(**{**vars(a), 'val_frac': 0.0})
        D2 = prepare(a2, dev)
        res['refit'], (st, _, _) = fit(a2, D2, dev, best['epoch'], select=False)
        np.savez_compressed(f'{a.out}/{a.name}.npz', logit=st, tf=D2['te'].TF.values, tgt=D2['te'].Target.values, y=D2['te'].Label.values)
    res['minutes'] = (time.time() - t0) / 60
    json.dump(res, open(f'{a.out}/{a.name}.json', 'w'), indent=1)
    t = res.get('refit', best['test'])
    print(f"{a.name}\tepoch {best['epoch']}\tval AUROC {best['val']['auc']:.4f}\t{'REFIT ' if a.refit else ''}TEST AUROC {t['auc']:.4f} AUPRC {t['aupr']:.4f} withinTF {t['within']:.4f}\t{res['minutes']:.1f} min")


if __name__ == '__main__':
    main()
