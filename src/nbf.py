"""NBFNet-style path-based link predictor for TF -> target scoring (Zhu et al., NeurIPS 2021), same harness as src.pair_gnn.

For every source TF t: boundary h0_v = q if v == t else 0, then L layers of
  h_v <- LN(h_v + W [h_v, AGG_{(u,r)->v}( h_u * w_r ) + boundary])        (DistMult message, sum / mean / pna aggregation)
over the training TF->target graph (relations: 0 TF->target, 1 target->TF, optional 2 co-expression, 3 STRING).
Score(t, g) = MLP([h_t^1(g) ... h_t^L(g), x_g projection (, x_t * x_g projection)]).
--split batch: the batch's own positive edges (both directions) leave the graph, as in pair_gnn.
Usage: python3.12 -m src.nbf --name X --layers 4 --aggr pna
"""
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from src.pair_gnn import prepare, metrics, coexp_graph, ppi_graph


class Layer(nn.Module):
    def __init__(s, d, R, aggr, feat):
        super().__init__()
        s.aggr, s.feat = aggr, feat
        s.rel = nn.Parameter(1 + 0.3 * torch.randn(R, d))
        k = 12 if aggr == 'pna' else 1
        s.lin = nn.Linear(d * (k + 1) + (d if feat == 'layer' else 0), d)
        s.norm = nn.LayerNorm(d)

    def forward(s, h, bound, xp, src, dst, rel, n):
        T = h.shape[0]
        msg = torch.cat([h[:, src] * s.rel[rel], bound], 1)  # boundary condition enters as one extra message per node
        idx = torch.cat([dst, torch.arange(n, device=h.device)])
        sc = lambda r, m=msg: scatter(m, idx, dim=1, dim_size=n, reduce=r)
        if s.aggr in ('sum', 'mean'):
            agg = sc(s.aggr)
        else:
            mean = sc('mean'); std = (sc('mean', msg * msg) - mean * mean).clamp_min(0).add(1e-6).sqrt()
            a = torch.cat([mean, sc('max'), sc('min'), std], -1)
            deg = torch.bincount(idx, minlength=n).float()
            ld = torch.log(deg + 1); amp = (ld / ld.mean())[None, :, None]; att = 1 / amp.clamp_min(1e-2)
            agg = torch.cat([a, a * amp, a * att], -1)
        z = torch.cat([h, agg] + ([xp.expand(T, -1, -1)] if s.feat == 'layer' else []), -1)
        return h + F.relu(s.norm(s.lin(z)))


class NBF(nn.Module):
    def __init__(s, dx, a, R):
        super().__init__()
        s.a, d = a, a.hidden
        s.proj = nn.Linear(dx, d) if a.feat != 'none' or a.tf_inter else None
        s.q = nn.Parameter(0.1 * torch.randn(d))
        s.qx = nn.Linear(d, d, bias=False) if a.tfq == 'feat' else None
        s.layers = nn.ModuleList([Layer(d, R, a.aggr, a.feat) for _ in range(a.layers)])
        din = d * a.layers + (d if a.feat != 'none' else 0) + (d if a.tf_inter else 0) + (2 if a.deg else 0)
        s.mlp = nn.Sequential(nn.Linear(din, d), nn.ReLU(), nn.Dropout(a.dropout), nn.Linear(d, 1))

    def forward(s, X, tfs, ei, rel, n, pairs_ti, pairs_g):
        """tfs: (T,) source TFs; pairs_ti: index into tfs of each scored pair; pairs_g: target gene of each pair"""
        T, d = len(tfs), s.a.hidden
        xp = F.dropout(s.proj(X), s.a.dropout, s.training)[None] if s.proj is not None else None  # (1,n,d)
        q = s.q.expand(T, d)
        if s.qx is not None: q = q + s.qx(xp[0][tfs])
        bound = torch.zeros(T, n, d, device=X.device)
        bound[torch.arange(T, device=X.device), tfs] = q
        h, outs = bound, []
        xl = xp if s.a.feat == 'layer' else None
        for L in s.layers:
            h = L(h, bound, xl, ei[0], ei[1], rel, n)
            outs.append(h[pairs_ti, pairs_g])
        parts = outs + ([xp[0][pairs_g]] if s.a.feat != 'none' else []) + ([xp[0][tfs[pairs_ti]] * xp[0][pairs_g]] if s.a.tf_inter else [])
        if s.a.deg:  # log in-degree (regulators) and out-degree (targets) of the scored gene in the CURRENT message-passing graph (own edge already removed in training)
            dst = ei[1]
            parts.append(torch.log1p(torch.stack([torch.bincount(dst[rel == 0], minlength=n), torch.bincount(dst[rel == 1], minlength=n)], 1).float())[pairs_g])
        return s.mlp(torch.cat(parts, 1)).squeeze(1)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cell_type', default='hESC')
    p.add_argument('--num_TF', default='500')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', default='batch', choices=['none', 'batch'])
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--aggr', default='pna', choices=['sum', 'mean', 'pna'])
    p.add_argument('--feat', default='final', choices=['none', 'final', 'layer'], help='gene expression projection: none (pure paths) | into the final MLP | into every layer and the final MLP')
    p.add_argument('--tfq', default='shared', choices=['shared', 'feat'], help='boundary query: one learned vector | + projection of the TF expression')
    p.add_argument('--tf_inter', type=int, default=0, help='1: final MLP also sees proj(x_t) * proj(x_g)')
    p.add_argument('--deg', type=int, default=0, help='1: final MLP also sees log in/out degree of the gene in the current graph (diagnostic)')
    p.add_argument('--rels', default='R', help='R TF->target + target->TF, C co-expression kNN, P STRING')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--optimizer', default='Adam')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=193)
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


def fit(a, D, dev, epochs, select):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr, te, n, tri, vai = D['tr'], D['te'], D['n'], D['tri'], D['vai']
    T = lambda v, dt=None: torch.as_tensor(v, device=dev, dtype=dt)
    X = T(D['x'])
    OT, OG = T(D['orient'][:, 0], torch.long), T(D['orient'][:, 1], torch.long)
    uid = T(D['uid'], torch.long); m = len(OT)
    stat = [(e, r) for e, r in ((coexp_graph(D['x'], False) if 'C' in a.rels else None, 2), (ppi_graph(D['path'], n) if 'P' in a.rels else None, 3)) if e is not None]
    S_src = torch.cat([T(e[0], torch.long) for e, _ in stat]) if stat else torch.zeros(0, dtype=torch.long, device=dev)
    S_dst = torch.cat([T(e[1], torch.long) for e, _ in stat]) if stat else torch.zeros(0, dtype=torch.long, device=dev)
    S_rel = torch.cat([torch.full((e.shape[1],), r, dtype=torch.long, device=dev) for e, r in stat]) if stat else torch.zeros(0, dtype=torch.long, device=dev)
    R = 2 + len(stat)

    def graph(drop=None):
        keep = torch.ones(m, dtype=torch.bool, device=dev)
        if drop is not None and len(drop): keep[drop] = False
        s, d = torch.cat([OT[keep], OG[keep], S_src]), torch.cat([OG[keep], OT[keep], S_dst])
        rel = torch.cat([torch.zeros(int(keep.sum()), dtype=torch.long, device=dev), torch.ones(int(keep.sum()), dtype=torch.long, device=dev), S_rel])
        return torch.stack([s, d]), rel

    P_tr = T(tr[['TF', 'Target']].values[tri], torch.long); Y_tr = T(tr.Label.values[tri], torch.float32)
    uid_tr = uid[T(tri, torch.long)]
    P_va = T(tr[['TF', 'Target']].values[vai], torch.long); y_va = tr.Label.values[vai]; tf_va = tr.TF.values[vai]
    P_te = T(te[['TF', 'Target']].values, torch.long); y_te = te.Label.values; tf_te = te.TF.values
    model = NBF(X.shape[1], a, R).to(dev)
    opt = getattr(torch.optim, a.optimizer)(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def run(p, ei, rel):
        tfs, inv = torch.unique(p[:, 0], return_inverse=True)
        return model(X, tfs, ei, rel, n, inv, p[:, 1])

    def score(P):
        out = []
        for i in range(0, len(P), 4096):
            out.append(run(P[i:i + 4096], *graph()))
        return torch.cat(out).cpu().numpy()

    def evaluate(with_val=True):
        model.eval()
        with torch.no_grad():
            st = score(P_te); sv = score(P_va) if with_val else None
        return (metrics(y_va, sv, tf_va) if with_val else None), metrics(y_te, st, tf_te), st

    hist, best, bad = [], None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(P_tr), device=dev)
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size]
            p, y = P_tr[idx], Y_tr[idx]
            drop = None
            if a.split == 'batch':
                dd = uid_tr[idx]; drop = dd[dd >= 0]
            loss = F.binary_cross_entropy_with_logits(run(p, *graph(drop)), y)
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
    with torch.no_grad():
        model.eval()
        tfu = np.unique(np.concatenate([tr.TF.values, te.TF.values]))
        pairs = T(np.array([(u, g) for u in tfu for g in range(n)]), torch.long)
        grid = score(pairs).reshape(len(tfu), n)
    return t, (st, tfu, grid)


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
        res['refit'], (st, tfu, grid) = fit(a2, D2, dev, best['epoch'], select=False)
        np.savez_compressed(f'{a.out}/{a.name}.npz', logit=st, tf=D2['te'].TF.values, tgt=D2['te'].Target.values, y=D2['te'].Label.values, grid_tf=tfu, grid=grid)
    res['minutes'] = (time.time() - t0) / 60
    json.dump(res, open(f'{a.out}/{a.name}.json', 'w'), indent=1)
    t = res.get('refit', best['test'])
    print(f"{a.name}\tepoch {best['epoch']}\tval AUROC {best['val']['auc']:.4f}\t{'REFIT ' if a.refit else ''}TEST AUROC {t['auc']:.4f} AUPRC {t['aupr']:.4f} withinTF {t['within']:.4f}\t{res['minutes']:.1f} min")


if __name__ == '__main__':
    main()
