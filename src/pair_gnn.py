"""Pair-aware GRN link predictor for the in-depth analysis (one file: data, model, training).

GCN over the training-positive graph, node features = standardized expression (optionally fused
with Geneformer), then a pluggable pair decoder:
  add : linear on [u_tf, u_gene]                 (what scRegNet does: TF term + gene term)
  had : linear on [u_tf, u_gene, u_tf * u_gene]  (biased matrix factorization: adds interaction)
  mlp : hidden layer on [u_tf, u_gene]           (concat + MLP)
  tani, tani_pure : 6-head Tanimoto similarity of u_tf and u_gene (tani keeps [u_tf, u_gene] too)
--split batch removes each batch's own positive edges from the message-passing graph, so a pair is
never scored through its own edge. Epochs are chosen on a held-out validation slice of the
training rows, never on the test set. Usage: python3.12 -m src.pair_gnn --name X --decoder had ...
"""
import argparse, glob, json, os, sys, time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class DirConv(nn.Module):
    """Directed message passing: TF->target and target->TF edges get separate weights."""
    def __init__(s, i, o):
        super().__init__()
        s.f, s.b = SAGEConv(i, o), SAGEConv(i, o)

    def forward(s, h, ei):
        return 0.5 * (s.f(h, ei[0]) + s.b(h, ei[1]))


class NPanda(nn.Module):
    """Neural PANDA layer: h' = lin_self(h) + sum over relations of SAGE_r(h, E_r); R = known targets, C = co-expression, P = PPI."""
    def __init__(s, i, o, rels):
        super().__init__()
        s.self_ = nn.Linear(i, o)
        s.rel = nn.ModuleDict({r: SAGEConv(i, o, root_weight=False) for r in rels})

    def forward(s, h, G):
        return s.self_(h) + sum(c(h, G[r]) for r, c in s.rel.items())


def make_conv(enc, i, o, rels='RCP'):
    if enc == 'gcn': return GCNConv(i, o)
    if enc == 'sage': return SAGEConv(i, o)
    if enc == 'gat': return GATConv(i, o, heads=4, concat=False)
    if enc == 'gin': return GINConv(nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o)), aggr='mean')  # sum aggregation overflows on hub genes
    if enc == 'dir': return DirConv(i, o)
    if enc == 'npanda': return NPanda(i, o, rels)
    raise ValueError(enc)


def sym_edges(ei, n):
    """undirected edge list (2, E) -> both directions, duplicates removed"""
    key = np.unique(np.minimum(ei[0], ei[1]) * n + np.maximum(ei[0], ei[1]))
    p, q = key // n, key % n
    return np.stack([np.concatenate([p, q]), np.concatenate([q, p])])


def panda_graph(M, tfs, k, n):
    """top-k genes per TF by PANDA score (self excluded), undirected"""
    M = M[tfs].copy(); M[np.arange(len(tfs)), tfs] = -np.inf
    top = np.argsort(-M, 1)[:, :k]
    return sym_edges(np.stack([np.repeat(tfs, k), top.ravel()]), n)


def coexp_graph(x, meta):
    """label-free gene-gene kNN (k=10, positive Pearson); meta: on 75 KMeans metacells of the first 30 PCs of the cells"""
    if meta:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        lab = KMeans(75, n_init=10, random_state=0).fit_predict(PCA(30, random_state=0).fit_transform(x.T))
        x = np.stack([x[:, lab == c].mean(1) for c in range(75)], 1)
    C = np.nan_to_num(np.corrcoef(x)); np.fill_diagonal(C, -1)
    nb = np.argsort(-C, 1)[:, :10]
    ei = np.stack([np.repeat(np.arange(len(C)), 10), nb.ravel()])
    return sym_edges(ei[:, C[ei[0], ei[1]] > 0], len(C))


def ppi_graph(path, n):
    """label-free STRING edges among the genes"""
    g = pd.read_csv(path + 'Target.csv', index_col=0).set_index('Gene')['index']
    e = pd.read_csv(glob.glob(path + 'STRING/*.symmetric_edges.csv')[0])
    e = e[e.Gene1.isin(g.index) & e.Gene2.isin(g.index) & (e.Gene1 != e.Gene2)]
    return sym_edges(np.stack([e.Gene1.map(g).values, e.Gene2.map(g).values]), n)


def tanimoto(a, b, H=6):
    a, b = a.reshape(len(a), H, -1), b.reshape(len(b), H, -1)
    ab = (a * b).sum(-1)
    return ab / ((a * a).sum(-1) + (b * b).sum(-1) - ab + 1e-6)


def within_tf(y, s, tf):
    out = [roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if 0 < y[tf == t].sum() < (tf == t).sum()]
    return float(np.mean(out))


def within_tf8(y, s, tf):
    """within-TF AUROC over TFs with at least 8 positives (less noisy for held-out-TF tests)"""
    out = [roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if y[tf == t].sum() >= 8 and y[tf == t].sum() < (tf == t).sum()]
    return float(np.mean(out)) if out else float('nan')


def metrics(y, s, tf):
    return dict(auc=float(roc_auc_score(y, s)), aupr=float(average_precision_score(y, s)), within=within_tf(y, s, tf), within8=within_tf8(y, s, tf))


def heldout_split(L, a, n):
    if a.protocol == 'gene':
        held = np.random.RandomState(1000 + a.fold).choice(n, size=int(0.2 * n), replace=False)
        m = L.Target.isin(held).values
    else:
        folds = np.array_split(np.random.RandomState(7).permutation(np.unique(L.TF.values)), 5)
        m = L.TF.isin(folds[a.fold]).values
    return L[~m].reset_index(drop=True), L[m].reset_index(drop=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cell_type', default='hESC')
    p.add_argument('--num_TF', default='500')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--decoder', default='add', choices=['add', 'had', 'mlp', 'tani', 'tani_pure'])
    p.add_argument('--split', default='none', choices=['none', 'batch'])
    p.add_argument('--scfm', type=int, default=1, help='0 off, 1 Geneformer, 2 Geneformer vectors shuffled across genes (control)')
    p.add_argument('--scalar', default='none', help='none | panda | panda_old | contrast | contrast_raw')
    p.add_argument('--encoder', default='gcn', choices=['gcn', 'sage', 'gat', 'gin', 'dir', 'npanda'])
    p.add_argument('--tf_drop', type=float, default=0.0, help='per batch, hide ALL training edges of this fraction of TFs (teaches the model to cope with TFs that have no known targets)')
    p.add_argument('--mp_graph', default='grn', choices=['grn', 'panda', 'union'], help='message-passing graph: training-positive GRN (default) | PANDA top-K | both')
    p.add_argument('--panda_k', type=int, default=100, help='top-K genes per TF in the PANDA graph')
    p.add_argument('--panda_safe', type=int, default=1, help='1: training batches use the fold-k PANDA graph (prior without the batch rows); 0: full PANDA graph (leaks the label)')
    p.add_argument('--relations', default='RCP', help='npanda encoder relations: R known targets, C co-expression kNN, P STRING PPI')
    p.add_argument('--coexp', default='full', choices=['full', 'meta'], help='npanda C graph on cells or on metacells')
    p.add_argument('--scalar_lr', type=float, default=1.0, help='!=1: scalars enter as a separate linear term trained at lr * scalar_lr, no weight decay (else concatenated into the decoder)')
    p.add_argument('--rank_w', type=float, default=0.0, help='weight of a within-TF BPR loss: each batch positive vs 4 training negatives of the same TF')
    p.add_argument('--rank_neg', default='rand', choices=['rand', 'pop', 'model'], help='same-TF training negative for --rank_w: random (x4) | most popular gene of 16 random | highest-scoring of 16 random')
    p.add_argument('--rank_loss', default='bpr', choices=['bpr', 'bce'], help='bpr: softplus(s_neg - s_pos); bce: extra BCE with label 0 on the chosen negatives')
    p.add_argument('--aux_panda', type=float, default=0.0, help='weight of an auxiliary MSE head that predicts the out-of-fold PANDA score on random (TF, gene) pairs')
    p.add_argument('--roles', default='shared', choices=['shared', 'sep'])
    p.add_argument('--gnn_dims', type=int, nargs='+', default=[227, 29, 34])
    p.add_argument('--mlp_dim', type=int, default=198)
    p.add_argument('--dec_hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.17477888900593389)
    p.add_argument('--optimizer', default='SGD')
    p.add_argument('--lr', type=float, default=0.0014411386699467767)
    p.add_argument('--wd', type=float, default=1.0449337997892844e-05)
    p.add_argument('--batch_size', type=int, default=193)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--eval_interval', type=int, default=5)
    p.add_argument('--select_metric', default='auc', choices=['auc', 'within'], help='validation metric that picks the epoch')
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--split_seed', type=int, default=0)
    p.add_argument('--graph_w', default='none', help='none | panda | contrast: out-of-fold score of each training edge as GCN edge weight (gcn encoder only)')
    p.add_argument('--refit', type=int, default=0, help='after choosing the epoch on validation, retrain on ALL training rows for that many epochs and score test once')
    p.add_argument('--protocol', default='random', choices=['random', 'gene', 'tf'], help='random = benchmark split; gene = hold out 20%% of target genes; tf = hold out ~7 of 34 TFs (5 folds)')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--data_folder', default='data')
    p.add_argument('--scfm_folder', default='scFM')
    p.add_argument('--out', default='analysis/out/pair')
    return p.parse_args()


def prepare(a, dev):
    path = f'{a.data_folder}/{a.cell_type}/TFs+{a.num_TF}/'
    tr = pd.read_csv(path + 'Train_set.csv', index_col=0)
    te = pd.read_csv(path + 'Test_set.csv', index_col=0)
    expr_df = pd.read_csv(path + 'BL--ExpressionData.csv', index_col=0)
    x = StandardScaler().fit_transform(expr_df.values.T).T.astype(np.float32)  # same as src.utils.load_data
    n = x.shape[0]
    if a.protocol != 'random':
        tr, te = heldout_split(pd.concat([tr, te], ignore_index=True), a, n)
    f = None
    if a.scfm:
        from src.train import Trainer
        ns = SimpleNamespace(llm_type='Geneformer', scFM_folder=a.scfm_folder, cell_type=a.cell_type, num_TF=a.num_TF)
        f = Trainer(ns)._get_embeddings(n, expr_df).numpy().astype(np.float32)
        if a.scfm == 2:  # control: same embedding vectors assigned to the wrong genes
            f = f[np.random.RandomState(0).permutation(n)]
    if a.val_frac > 0 and a.protocol != 'random':  # validation held out the same way as the test: unseen genes / unseen TFs
        rs = np.random.RandomState(a.split_seed)
        key, k = ('Target', max(1, int(a.val_frac * tr.Target.nunique()))) if a.protocol == 'gene' else ('TF', 3)
        vm = tr[key].isin(rs.choice(np.unique(tr[key].values), size=k, replace=False)).values
        tri, vai = np.where(~vm)[0], np.where(vm)[0]
    elif a.val_frac > 0:
        tri, vai = train_test_split(np.arange(len(tr)), test_size=a.val_frac, stratify=tr.Label.values, random_state=a.split_seed)
    else:
        tri, vai = np.arange(len(tr)), np.array([], dtype=int)
    tri, vai = np.sort(tri), np.sort(vai)
    P = tr[['TF', 'Target']].values
    pos_rows = tri[tr.Label.values[tri] == 1]
    pos_rows = pos_rows[P[pos_rows, 0] != P[pos_rows, 1]]
    lo, hi = P[pos_rows].min(1), P[pos_rows].max(1)
    uniq, first, inv = np.unique(lo * n + hi, return_index=True, return_inverse=True)
    orient = P[pos_rows][first]  # (TF, target) orientation of each undirected edge
    uid = -np.ones(len(tr), dtype=np.int64)
    uid[pos_rows] = inv
    return dict(tr=tr, te=te, x=x, f=f, n=n, tri=tri, vai=vai, U=(uniq // n, uniq % n), orient=orient, uid=uid, path=path)


class PairGNN(nn.Module):
    def __init__(s, d_in, d_sf, a, n_scalar):
        super().__init__()
        s.a = a
        s.convs = nn.ModuleList()
        prev = d_in
        for h in a.gnn_dims:
            s.convs.append(make_conv(a.encoder, prev, h, a.relations))
            prev = h
        fused = prev + (d_sf if a.scfm else 0)
        s.mlp = nn.Linear(fused, a.mlp_dim)
        s.mlp_t = nn.Linear(fused, a.mlp_dim) if a.roles == 'sep' else None
        D = a.mlp_dim
        s.sw = nn.Linear(n_scalar, 1, bias=False) if n_scalar and a.scalar_lr != 1 else None
        n_scalar = 0 if s.sw is not None else n_scalar
        if a.decoder == 'add':
            s.out = nn.Linear(2 * D + n_scalar, 1)
        elif a.decoder == 'had':
            s.out = nn.Linear(3 * D + n_scalar, 1)
        elif a.decoder == 'tani':
            s.out = nn.Linear(2 * D + 6 + n_scalar, 1)
        elif a.decoder == 'tani_pure':
            s.out = nn.Linear(6 + n_scalar, 1)
        else:
            s.out = nn.Sequential(nn.Linear(2 * D + n_scalar, a.dec_hidden), nn.ReLU(), nn.Linear(a.dec_hidden, 1))
        s.aux = nn.Linear(3 * D, 1) if a.aux_panda > 0 else None

    def nodes(s, x, f, ei, ew=None):
        h = x
        for i, c in enumerate(s.convs):
            h = c(h, ei, ew) if ew is not None else c(h, ei)
            if i < len(s.convs) - 1:
                h = F.dropout(F.relu(h), s.a.dropout, s.training)
        if s.a.scfm:
            h = torch.cat([f, h], 1)  # same order as scTransNet_GCN
        u = F.leaky_relu(s.mlp(h))
        return u, (F.leaky_relu(s.mlp_t(h)) if s.mlp_t is not None else u)

    def score(s, ut, ug, pairs, scalar=None):
        a, b = ut[pairs[:, 0]], ug[pairs[:, 1]]
        sx = None if scalar is None else (scalar[:, None] if scalar.dim() == 1 else scalar)
        parts = [a, b] + ([a * b] if s.a.decoder == 'had' else []) + ([sx] if sx is not None and s.sw is None else [])
        if s.a.decoder.startswith('tani'):
            parts = ([a, b] if s.a.decoder == 'tani' else []) + [tanimoto(a, b)] + parts[2:]
        out = s.out(torch.cat(parts, 1)).squeeze(1)
        return out + s.sw(sx).squeeze(1) if s.sw is not None else out


def fit(a, D, dev, epochs, select):
    """Train one model. select=True: evaluate every eval_interval epochs on the validation slice,
    early-stop, keep the best epoch. select=False: train exactly `epochs` epochs, score test once."""
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr, te, n, tri, vai = D['tr'], D['te'], D['n'], D['tri'], D['vai']
    T = lambda v, dt=None: torch.as_tensor(v, device=dev, dtype=dt)
    X = T(D['x']); Fe = T(D['f']) if D['f'] is not None else None
    U0, U1 = T(D['U'][0], torch.long), T(D['U'][1], torch.long)
    OT, OG = T(D['orient'][:, 0], torch.long), T(D['orient'][:, 1], torch.long)
    uid = T(D['uid'], torch.long)
    m = len(U0)

    S_ev = S_tr = None
    if a.scalar != 'none':
        import scalars
        kinds = a.scalar.split(',')  # several kinds -> (n, n, k)
        got = [scalars.get(SimpleNamespace(**{**vars(a), 'scalar': k}), D) for k in kinds]
        S_ev, S_tr = [T(got[0][j] if len(kinds) == 1 else np.stack([g[j] for g in got], -1), torch.float32) for j in (0, 1)]
    w_u = None
    if a.graph_w != 'none':
        import scalars
        _, Sg = scalars.get(SimpleNamespace(**{**vars(a), 'scalar': a.graph_w}), D)
        Sg = T(Sg, torch.float32)
        w_u = torch.zeros(m, device=dev)
        pr = T(tr[['TF', 'Target']].values[tri], torch.long)
        u_tri = uid[T(tri, torch.long)]
        ok = u_tri >= 0
        w_u[u_tri[ok]] = 0.5 + torch.sigmoid(Sg[pr[ok, 0], pr[ok, 1]])

    if a.rank_w > 0:  # training negatives grouped by TF, for same-TF negative sampling
        yt, tft = tr.Label.values[tri], tr.TF.values[tri]
        neg = np.where(yt == 0)[0]; neg = neg[np.argsort(tft[neg], kind='stable')]
        st_ = np.searchsorted(tft[neg], np.arange(n))
        NEG, NS, NC = T(neg, torch.long), T(st_, torch.long), T(np.searchsorted(tft[neg], np.arange(n), 'right') - st_, torch.long)
        pos_t = tr.Target.values[tri][yt == 1]
        DEG = T(np.bincount(pos_t, minlength=n), torch.float32)  # training in-degree of each gene (popularity)
    if a.aux_panda > 0:  # out-of-fold PANDA (training rows scored without their own fold's positives)
        import scalars
        M_aux = T(scalars.get(SimpleNamespace(**{**vars(a), 'scalar': 'panda'}), D)[1], torch.float32)
        aux_tf = T(np.unique(tr.TF.values), torch.long)

    PG = None
    if a.mp_graph != 'grn':
        assert a.encoder in ('gcn', 'sage', 'gat', 'gin') and a.protocol == 'random'
        import scalars
        a_ = SimpleNamespace(**{**vars(a), 'scalar': 'panda'})
        tfs = np.unique(np.concatenate([tr.TF.values, D['te'].TF.values]))
        PG = dict(ev=T(panda_graph(scalars.get(a_, D)[0], tfs, a.panda_k, n), torch.long),
                  fold=[T(panda_graph(M, tfs, a.panda_k, n), torch.long) for M in scalars.get_folds(a_, D)])
    G_C = T(coexp_graph(D['x'], a.coexp == 'meta'), torch.long) if a.encoder == 'npanda' and 'C' in a.relations else None
    G_P = T(ppi_graph(D['path'], n), torch.long) if a.encoder == 'npanda' and 'P' in a.relations else None

    def full_graph(drop=None, fold=None):
        keep = torch.ones(m, dtype=torch.bool, device=dev)
        if drop is not None and len(drop) > 0:
            keep[drop] = False
        p, q = U0[keep], U1[keep]
        ei = torch.stack([torch.cat([p, q]), torch.cat([q, p])])
        if a.encoder == 'dir':
            ei = (torch.stack([OT[keep], OG[keep]]), torch.stack([OG[keep], OT[keep]]))
        ew = torch.cat([w_u[keep], w_u[keep]]) if w_u is not None else None
        if PG is not None:
            pg = PG['ev'] if fold is None else PG['fold'][fold]
            return (pg if a.mp_graph == 'panda' else torch.cat([ei, pg], 1)), ew
        if a.encoder == 'npanda':
            return {r: g for r, g in zip('RCP', (ei, G_C, G_P)) if r in a.relations}, None
        return ei, ew

    P_tr = T(tr[['TF', 'Target']].values[tri], torch.long); Y_tr = T(tr.Label.values[tri], torch.float32)
    uid_tr = uid[T(tri, torch.long)]
    P_va = T(tr[['TF', 'Target']].values[vai], torch.long); y_va = tr.Label.values[vai]; tf_va = tr.TF.values[vai]
    P_te = T(te[['TF', 'Target']].values, torch.long); y_te = te.Label.values; tf_te = te.TF.values

    model = PairGNN(X.shape[1], Fe.shape[1] if Fe is not None else 0, a, 0 if S_tr is None else (1 if S_tr.dim() == 2 else S_tr.shape[2])).to(dev)
    groups = [{'params': [q for nm, q in model.named_parameters() if not nm.startswith('sw.')]}]
    if model.sw is not None:
        groups.append({'params': model.sw.parameters(), 'lr': a.lr * a.scalar_lr, 'weight_decay': 0.0})
    opt = getattr(torch.optim, a.optimizer)(groups, lr=a.lr, weight_decay=a.wd)
    sc = lambda S, P: None if S is None else S[P[:, 0], P[:, 1]]

    def evaluate(with_val=True):
        model.eval()
        with torch.no_grad():
            ei, ew = full_graph()
            ut, ug = model.nodes(X, Fe, ei, ew)
            st = model.score(ut, ug, P_te, sc(S_ev, P_te)).cpu().numpy()
            sv = model.score(ut, ug, P_va, sc(S_ev, P_va)).cpu().numpy() if with_val else None
        return (metrics(y_va, sv, tf_va) if with_val else None), metrics(y_te, st, tf_te), st

    safe = PG is not None and a.panda_safe  # fold-pure batches: batch rows come from one OOF fold, whose PANDA prior excludes them
    if safe:
        fold_rows = [T(h, torch.long) for h in np.array_split(np.random.RandomState(42).permutation(len(tri)), 5)]  # same folds as scalars.get

    hist, best, bad = [], None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(P_tr), device=dev)
        if safe:
            batches = [(k, r[torch.randperm(len(r), device=dev)][i:i + a.batch_size]) for k, r in enumerate(fold_rows) for i in range(0, len(r), a.batch_size)]
            batches = [batches[j] for j in np.random.permutation(len(batches))]
        else:
            batches = [(None, perm[i:i + a.batch_size]) for i in range(0, len(perm), a.batch_size)]
        for fk, idx in batches:
            p, y = P_tr[idx], Y_tr[idx]
            drop = None
            if a.split == 'batch':
                d = uid_tr[idx]; drop = d[d >= 0]
                if a.tf_drop > 0:
                    ex = torch.nonzero((torch.rand(n, device=dev) < a.tf_drop)[OT]).squeeze(1)
                    drop = torch.unique(torch.cat([drop, ex]))
            ei, ew = full_graph(drop, fk)
            ut, ug = model.nodes(X, Fe, ei, ew)
            loss = F.binary_cross_entropy_with_logits(model.score(ut, ug, p, sc(S_tr, p)), y)
            if a.rank_w > 0:
                pp = p[y == 1]; c = NC[pp[:, 0]]; pp, c = pp[c > 0], c[c > 0]
                if len(pp) and a.rank_neg == 'rand':
                    pp, c = pp.repeat(4, 1), c.repeat(4)
                    pn = P_tr[NEG[NS[pp[:, 0]] + (torch.rand(len(pp), device=dev) * c).long()]]
                elif len(pp):  # hard negative: pick 1 of 16 random same-TF negatives
                    pc = P_tr[NEG[NS[pp[:, 0], None] + (torch.rand(len(pp), 16, device=dev) * c[:, None]).long()]]
                    with torch.no_grad():
                        h = DEG[pc[..., 1]] if a.rank_neg == 'pop' else model.score(ut, ug, pc.reshape(-1, 2), sc(S_tr, pc.reshape(-1, 2))).reshape(len(pp), 16)
                    pn = pc[torch.arange(len(pp), device=dev), h.argmax(1)]
                if len(pp):
                    sn = model.score(ut, ug, pn, sc(S_tr, pn))
                    loss = loss + a.rank_w * (F.softplus(sn - model.score(ut, ug, pp, sc(S_tr, pp))).mean() if a.rank_loss == 'bpr' else F.binary_cross_entropy_with_logits(sn, torch.zeros_like(sn)))
            if a.aux_panda > 0:
                q = torch.stack([aux_tf[torch.randint(len(aux_tf), (512,), device=dev)], torch.randint(n, (512,), device=dev)], 1)
                ua, ub = ut[q[:, 0]], ug[q[:, 1]]
                loss = loss + a.aux_panda * F.mse_loss(model.aux(torch.cat([ua, ub, ua * ub], 1)).squeeze(1), M_aux[q[:, 0], q[:, 1]])
            opt.zero_grad(); loss.backward(); opt.step()
        if select and ep % a.eval_interval == 0:
            v, t, _ = evaluate()
            hist.append(dict(epoch=ep, val=v, test=t))
            if best is None or v[a.select_metric] > best['val'][a.select_metric]:
                best, bad = hist[-1], 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
    if select:
        return hist, best
    _, t, st = evaluate(with_val=False)
    with torch.no_grad():
        model.eval()
        ei, ew = full_graph()
        ut, ug = model.nodes(X, Fe, ei, ew)
        tfu = np.unique(np.concatenate([tr.TF.values, te.TF.values]))
        pairs = T(np.array([(u, g) for u in tfu for g in range(n)]), torch.long)
        grid = model.score(ut, ug, pairs, sc(S_ev, pairs)).cpu().numpy().reshape(len(tfu), n)
    return t, (st, tfu, grid)


def main():
    a = parse()
    sys.path.insert(0, 'analysis')
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    D = prepare(a, dev)
    hist, best = fit(a, D, dev, a.epochs, select=True)
    os.makedirs(a.out, exist_ok=True)
    tsel = max(hist, key=lambda h: h['test']['auc'])  # what the original code would report (selection on test)
    res = dict(name=a.name, args=vars(a), best_val_epoch=best['epoch'], val=best['val'], test=best['test'],
               test_selected=tsel['test'], history=hist)
    if a.refit:
        D2 = prepare(SimpleNamespace(**{**vars(a), 'val_frac': 0.0}), dev)
        a2 = SimpleNamespace(**{**vars(a), 'val_frac': 0.0})
        res['refit'], (st, tfu, grid) = fit(a2, D2, dev, best['epoch'], select=False)
        os.makedirs(a.out, exist_ok=True)
        np.savez_compressed(f'{a.out}/{a.name}.npz', logit=st, tf=D2['te'].TF.values, tgt=D2['te'].Target.values, y=D2['te'].Label.values, grid_tf=tfu, grid=grid)
    res['minutes'] = (time.time() - t0) / 60
    json.dump(res, open(f'{a.out}/{a.name}.json', 'w'), indent=1)
    t = res.get('refit', best['test'])
    print(f"{a.name}\tepoch {best['epoch']}\tval AUROC {best['val']['auc']:.4f}\t{'REFIT ' if a.refit else ''}TEST AUROC {t['auc']:.4f} AUPRC {t['aupr']:.4f} withinTF {t['within']:.4f}\t{res['minutes']:.1f} min")


if __name__ == '__main__':
    main()
