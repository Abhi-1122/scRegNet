"""Pair-aware GRN link predictor for the in-depth analysis (one file: data, model, training).

GCN over the training-positive graph, node features = standardized expression (optionally fused
with Geneformer), then a pluggable pair decoder:
  add : linear on [u_tf, u_gene]                 (what scRegNet does: TF term + gene term)
  had : linear on [u_tf, u_gene, u_tf * u_gene]  (biased matrix factorization: adds interaction)
  mlp : hidden layer on [u_tf, u_gene]           (concat + MLP)
  tani, tani_pure : 6-head Tanimoto similarity of u_tf and u_gene (tani keeps [u_tf, u_gene] too)
--split batch removes each batch's own positive edges from the message-passing graph, so a pair is
never scored through its own edge; --split fixed instead keeps a fixed --mp_frac of the edges as the training graph
(supervision = the rest + all negatives). Epochs are chosen on a held-out validation slice of the
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


class WSage(nn.Module):
    """SAGEConv with edge weights: lin_l(weighted mean of neighbours) + lin_r(x). Weights ew (E,) are used by --idf_beta."""
    def __init__(s, i, o):
        super().__init__()
        s.l, s.r = nn.Linear(i, o), nn.Linear(i, o, bias=False)

    def forward(s, h, ei, ew):
        src, dst = ei
        num = torch.zeros_like(h).index_add_(0, dst, ew[:, None] * h[src])
        den = torch.zeros(len(h), device=h.device).index_add_(0, dst, ew).clamp(min=1e-12)
        return s.l(num / den[:, None]) + s.r(h)


def make_pe(kind, ei, n, k, dev):
    """positional encoding (n, k) of an undirected graph given as an edge list (2, E) over n nodes.
    lap: k lowest non-trivial eigenvectors of the normalised Laplacian of the non-isolated nodes (x sqrt(n)); rw: diagonal of the first k powers of the random-walk matrix (RWSE).
    Isolated nodes get zeros."""
    A = np.zeros((n, n), np.float32); A[ei[0], ei[1]] = 1; A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    deg = A.sum(1); nz = deg > 0; pe = np.zeros((n, k), np.float32)
    if kind == 'lap':
        An = A[np.ix_(nz, nz)]; d = deg[nz]
        V = np.linalg.eigh(np.eye(len(d)) - An / np.sqrt(d[:, None] * d[None]))[1][:, 1:k + 1]
        pe[nz, :V.shape[1]] = V * np.sqrt(len(d))
    else:
        P = torch.as_tensor(A / np.maximum(deg, 1)[:, None], device=dev); Q = P.clone(); cols = []
        for _ in range(k):
            cols.append(torch.diagonal(Q)); Q = Q @ P
        pe = torch.stack(cols, 1).cpu().numpy(); pe = pe / (pe[nz].std(0) + 1e-6)
    return pe


def make_conv(enc, i, o, rels='RCP'):
    if enc == 'wsage': return WSage(i, o)
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
    p.add_argument('--feat', default='full', choices=['full', 'cellperm', 'stats', 'rand'], help='W14 diagnostic on node features: full = standardised expression profile | cellperm = each gene\'s values shuffled across cells | stats = 14 per-gene summary statistics of the raw expression | rand = fixed i.i.d. Gaussian of the same width')
    p.add_argument('--num_TF', default='500')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--decoder', default='add', choices=['add', 'had', 'mlp', 'tani', 'tani_pure', 'qatt', 'dot'], help='dot = plain dot product of the final node embeddings + one bias (LightGCN scorer; no MLP, no learned decoder weights) | qatt = had + query-aware TF summary: attention of the pair gene over the TF\'s known targets (W12 idea 1)')
    p.add_argument('--split', default='none', choices=['none', 'batch', 'fixed'], help="none | batch (each batch's own edges leave the graph) | fixed (a fixed --mp_frac of the undirected training edges is the training graph, supervision = the other edges + all negatives; validation/test graph = all training edges, as for batch)")
    p.add_argument('--mp_frac', type=float, default=0.7, help='--split fixed: fraction of training-positive edges in the training message-passing graph')
    p.add_argument('--own_edge_diag', type=int, default=0, help='1: after the refit, score a random 10%% of training-positive edges (+ as many training negatives) with and without those edges in the graph; stored in the json')
    p.add_argument('--scfm', type=int, default=1, help='0 off, 1 Geneformer, 2 Geneformer vectors shuffled across genes (control)')
    p.add_argument('--scalar', default='none', help='none | panda | panda_old | contrast | contrast_raw')
    p.add_argument('--encoder', default='gcn', choices=['gcn', 'sage', 'gat', 'gin', 'dir', 'npanda', 'lightgcn'])
    p.add_argument('--lgc_layers', type=int, default=2, help='lightgcn: propagation steps K (output = mean of E_0..E_K)')
    p.add_argument('--emb_dim', type=int, default=64, help='lightgcn: width of the free node embedding')
    p.add_argument('--emb_std', type=float, default=0.1, help='lightgcn: std of the normal init of the free embedding')
    p.add_argument('--emb_l2', type=float, default=0.0, help='lightgcn: L2 (Adam weight_decay) on the free embedding only; --wd then covers the rest')
    p.add_argument('--emb_feat', default='none', choices=['none', 'expr', 'gf', 'stats'], help='lightgcn: add a linear projection of node features to E_0 (gf needs --scfm 1; stats = the 14 --feat stats columns)')
    p.add_argument('--emb_free', type=int, default=1, help='lightgcn: 0 = no free embedding (features only; ablation)')
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
    p.add_argument('--qatt_heads', type=int, default=1)
    p.add_argument('--qatt_dim', type=int, default=64, help='total attention width (split over heads)')
    p.add_argument('--qatt_src', default='emb', choices=['emb', 'raw'], help='vectors that are compared: post-GNN embeddings or raw expression')
    p.add_argument('--qatt_topk', type=int, default=0, help='>0: attention only over the k best-matching known targets')
    p.add_argument('--qatt_uniform', type=int, default=0, help='control: uniform attention over the known targets (a plain mean, no query-awareness)')
    p.add_argument('--hub_cap', type=int, default=0, help='>0: each gene keeps at most K message-passing edges, redrawn every training batch (W12 idea 6)')
    p.add_argument('--hub_eval', default='cap', choices=['cap', 'all'], help='evaluation graph with --hub_cap: one fixed-seed cap draw, or all edges')
    p.add_argument('--idf_beta', type=float, default=0.0, help='>0 (sage): edge (t,g) weighted deg(g)^-beta in the neighbourhood mean, deg from the current graph')
    p.add_argument('--deg_feat', type=int, default=0, help='1: log(1+degree in the current message-passing graph) appended to node features')
    p.add_argument('--pe', default='none', choices=['none', 'lap', 'rw'], help='positional encoding of the genes (W12 idea 9)')
    p.add_argument('--pe_dim', type=int, default=16)
    p.add_argument('--pe_graph', default='coexp', choices=['grn', 'coexp', 'ppi'])
    p.add_argument('--pe_use', default='concat', choices=['concat', 'dist', 'prod'], help='concat to the GNN input | PEG-style: ||pe_t-pe_g|| / pe_t*pe_g into the decoder')
    p.add_argument('--pe_fold', type=int, default=1, help='grn PE: batches come from one of 5 folds and use the PE of the graph without that fold\'s edges (0: PE of the full training graph, own edge inside = leak)')
    p.add_argument('--pe_flip', type=int, default=1, help='random sign flip of lap eigenvectors per training batch')
    p.add_argument('--pe_rand', type=int, default=0, help='control: replace the PE by fixed random Gaussian vectors')
    p.add_argument('--fold_batches', type=int, default=0, help='1: fold-pure batches (as --pe_fold) even without a PE, as control')
    p.add_argument('--save_sel_grid', type=int, default=0, help='1: also save the best-validation-epoch model\'s TF x gene score grid, its gene embeddings and out-of-fold training scores (Correct&Smooth, W12 idea 4)')
    p.add_argument('--dec_wd', type=float, default=-1.0, help='W16: >=0 sets the weight decay of everything after the GNN encoder (node projection mlp/mlp_t, pair decoder, qatt parts, dot bias) to this value; --wd then covers the GNN convolutions only (and lightgcn proj). -1 = --wd for everything (default)')
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


def feat_variant(kind, x, raw):
    """W14: replace the node features (x: genes x cells, standardised per gene; raw: same shape, unstandardised). Seeds do not matter: every run sees the same matrix."""
    if kind == 'cellperm':
        return np.stack([x[i, np.random.RandomState(i).permutation(x.shape[1])] for i in range(x.shape[0])]).astype(np.float32)
    if kind == 'rand':
        return np.random.RandomState(0).randn(*x.shape).astype(np.float32)
    from scipy.stats import skew
    m, sd, nz = raw.mean(1), raw.std(1), (raw != 0).mean(1)
    with np.errstate(all='ignore'):
        mnz = np.where(nz > 0, raw.sum(1) / np.maximum(raw.shape[1] * nz, 1e-12), 0.0)
        S = np.column_stack([m, sd, np.where(m > 0, sd / np.where(m > 0, m, 1), 0.0), nz, mnz, skew(raw, axis=1)] + [np.percentile(raw, q, axis=1) for q in (10, 25, 50, 75, 90, 99)])
    S = np.nan_to_num(S)
    return StandardScaler().fit_transform(S).astype(np.float32)


def prepare(a, dev):
    path = f'{a.data_folder}/{a.cell_type}/TFs+{a.num_TF}/'
    tr = pd.read_csv(path + 'Train_set.csv', index_col=0)
    te = pd.read_csv(path + 'Test_set.csv', index_col=0)
    expr_df = pd.read_csv(path + 'BL--ExpressionData.csv', index_col=0)
    x = StandardScaler().fit_transform(expr_df.values.T).T.astype(np.float32)  # same as src.utils.load_data
    n = x.shape[0]
    xs = feat_variant('stats', x, expr_df.values.astype(np.float64)) if a.emb_feat == 'stats' else None
    if a.feat != 'full':
        x = feat_variant(a.feat, x, expr_df.values.astype(np.float64))
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
    return dict(tr=tr, te=te, x=x, xs=xs, f=f, n=n, tri=tri, vai=vai, U=(uniq // n, uniq % n), orient=orient, uid=uid, path=path)


class PairGNN(nn.Module):
    def __init__(s, d_in, d_sf, a, n_scalar, ctx=None):
        super().__init__()
        s.a, s.ctx = a, ctx  # ctx: per-call state set by fit (current PE, degree feature, TF->target mask)
        s.convs = nn.ModuleList()
        prev = d_in
        s.lgc = a.encoder == 'lightgcn'
        if s.lgc:
            s.emb = nn.Embedding(ctx.n, a.emb_dim); nn.init.normal_(s.emb.weight, std=a.emb_std)
            if not a.emb_free: s.emb.weight.requires_grad_(False); s.emb.weight.zero_()
            s.proj = nn.Linear(ctx.d_feat, a.emb_dim) if a.emb_feat != 'none' else None
            s.dot_b = nn.Parameter(torch.zeros(1))
            s.featmat = None  # set by fit
        for h in ([] if s.lgc else a.gnn_dims):
            s.convs.append(make_conv('wsage' if a.encoder == 'sage' and a.idf_beta > 0 else a.encoder, prev, h, a.relations))
            prev = h
        fused = a.emb_dim if s.lgc else prev + (d_sf if a.scfm else 0)
        s.mlp = nn.Linear(fused, a.mlp_dim)
        s.mlp_t = nn.Linear(fused, a.mlp_dim) if a.roles == 'sep' else None
        D = a.mlp_dim
        s.sw = nn.Linear(n_scalar, 1, bias=False) if n_scalar and a.scalar_lr != 1 else None
        n_scalar = 0 if s.sw is not None else n_scalar
        if a.pe != 'none' and a.pe_use != 'concat':
            n_scalar += 1 if a.pe_use == 'dist' else a.pe_dim
        if a.decoder == 'add':
            s.out = nn.Linear(2 * D + n_scalar, 1)
        elif a.decoder == 'dot':
            s.out = None
        elif a.decoder == 'had':
            s.out = nn.Linear(3 * D + n_scalar, 1)
        elif a.decoder == 'qatt':
            H, dk = a.qatt_heads, a.qatt_dim // a.qatt_heads
            ds = D if a.qatt_src == 'emb' else d_in
            s.wq, s.wk, s.wv = nn.Linear(ds, H * dk), nn.Linear(ds, H * dk), nn.Linear(ds, H * dk)
            s.cp = nn.Linear(H * dk, D, bias=False)
            s.out = nn.Linear(5 * D + 1 + n_scalar, 1)
        elif a.decoder == 'tani':
            s.out = nn.Linear(2 * D + 6 + n_scalar, 1)
        elif a.decoder == 'tani_pure':
            s.out = nn.Linear(6 + n_scalar, 1)
        else:
            s.out = nn.Sequential(nn.Linear(2 * D + n_scalar, a.dec_hidden), nn.ReLU(), nn.Linear(a.dec_hidden, 1))
        s.aux = nn.Linear(3 * D, 1) if a.aux_panda > 0 else None

    def lgc_nodes(s, ei):
        """LightGCN: E_0 = free embedding (+ linear projection of features); E_{k+1} = D^-1/2 A D^-1/2 E_k on the current graph; output = mean of E_0..E_K"""
        h = s.emb.weight + (s.proj(s.featmat) if s.proj is not None else 0)
        src, dst = ei
        deg = torch.zeros(len(h), device=h.device).index_add_(0, dst, torch.ones(len(dst), device=h.device))
        w = (deg[src] * deg[dst]).clamp(min=1).rsqrt()[:, None]
        out = h
        for _ in range(s.a.lgc_layers):
            h = torch.zeros_like(h).index_add_(0, dst, w * h[src]); out = out + h
        return out / (s.a.lgc_layers + 1)

    def nodes(s, x, f, ei, ew=None):
        if s.lgc:
            h = s.lgc_nodes(ei)
            if s.a.decoder == 'dot': return h, h
            u = F.leaky_relu(s.mlp(h)); return u, u
        ex = ([s.ctx.pe] if s.a.pe != 'none' and s.a.pe_use == 'concat' else []) + ([s.ctx.deg[:, None]] if s.a.deg_feat else [])
        h = torch.cat([x] + ex, 1) if ex else x
        s.raw = h
        for i, c in enumerate(s.convs):
            h = c(h, ei, ew) if ew is not None else c(h, ei)
            if i < len(s.convs) - 1:
                h = F.dropout(F.relu(h), s.a.dropout, s.training)
        if s.a.scfm:
            h = torch.cat([f, h], 1)  # same order as scTransNet_GCN
        u = F.leaky_relu(s.mlp(h))
        return u, (F.leaky_relu(s.mlp_t(h)) if s.mlp_t is not None else u)

    def qctx(s, ug, pairs, chunk=4096):
        """query-aware TF summary. For pair (t, g): softmax attention of gene g over t's known targets j (TF->target edges of the CURRENT
        message-passing graph, g itself removed); returns context (B, D), masked max cosine similarity (B,), attention entropy (B,), number of targets (B,)."""
        a = s.a; H = a.qatt_heads; dk = a.qatt_dim // H
        src = ug if a.qatt_src == 'emb' else s.raw
        Kx, Vx, un = s.wk(src).view(-1, H, dk), s.wv(src).view(-1, H, dk), F.normalize(ug, dim=1)
        out = []
        for i in range(0, len(pairs), chunk):
            t, g = pairs[i:i + chunk, 0], pairs[i:i + chunk, 1]
            mask = s.ctx.A[t].clone(); mask[torch.arange(len(g), device=g.device), g] = False
            sc = torch.einsum('bhd,nhd->bhn', s.wq(src[g]).view(-1, H, dk), Kx) / dk ** 0.5
            sc = sc.masked_fill(~mask[:, None, :], -1e9)
            if a.qatt_topk > 0:
                sc = sc.masked_fill(sc < sc.topk(min(a.qatt_topk, sc.shape[-1]), -1).values[..., -1:], -1e9)
            has = mask.any(1)
            al = torch.softmax(sc, -1) * has[:, None, None]
            if a.qatt_uniform: al = (mask / mask.sum(1, keepdim=True).clamp(min=1))[:, None, :].expand(-1, H, -1)
            c = s.cp(torch.einsum('bhn,nhd->bhd', al, Vx).reshape(len(g), H * dk))
            ms = (un[g] @ un.T).masked_fill(~mask, -2).max(1).values * has
            ent = -(al * torch.log(al + 1e-9)).sum(-1).mean(1)
            out.append((c, ms, ent, mask.sum(1)))
        return [torch.cat(z) for z in zip(*out)]

    def score(s, ut, ug, pairs, scalar=None):
        a, b = ut[pairs[:, 0]], ug[pairs[:, 1]]
        sx = None if scalar is None else (scalar[:, None] if scalar.dim() == 1 else scalar)
        if s.a.decoder == 'dot': return (a * b).sum(1) + s.dot_b
        parts = [a, b] + ([a * b] if s.a.decoder in ('had', 'qatt') else [])
        if s.a.decoder == 'qatt':
            c, ms, _, _ = s.qctx(ug, pairs)
            parts += [c, c * b, ms[:, None]]
        if s.a.pe != 'none' and s.a.pe_use != 'concat':
            pt, pg = s.ctx.pe[pairs[:, 0]], s.ctx.pe[pairs[:, 1]]
            parts.append((pt - pg).norm(dim=1, keepdim=True) if s.a.pe_use == 'dist' else pt * pg)
        parts += [sx] if sx is not None and s.sw is None else []
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

    ctx = SimpleNamespace(A=None, pe=None, deg=None)
    need_A = a.decoder == 'qatt'
    PE_ev, PE_fold, pe_fold = None, None, False
    if a.pe != 'none':
        k_ = a.pe_dim
        pe_fold = a.pe_graph == 'grn' and bool(a.pe_fold) and not a.pe_rand
        if a.pe_rand:
            PE_ev = torch.randn(n, k_, generator=torch.Generator().manual_seed(a.seed)).to(dev)
        else:
            if a.pe_graph == 'grn':
                pe_ei = np.stack([D['U'][0], D['U'][1]])
            else:
                pe_ei = coexp_graph(D['x'], False) if a.pe_graph == 'coexp' else ppi_graph(D['path'], n)
            PE_ev = T(make_pe(a.pe, pe_ei, n, k_, dev))
            if pe_fold:  # PE of the training graph without fold k's edges, for batches drawn from fold k (the pair's own edge is in no graph it sees)
                PE_fold = []
                for h in np.array_split(np.random.RandomState(42).permutation(len(tri)), 5):
                    u = np.unique(D['uid'][tri[h]]); u = u[u >= 0]
                    keep_ = np.ones(len(D['U'][0]), bool); keep_[u] = False
                    pf = T(make_pe(a.pe, np.stack([D['U'][0][keep_], D['U'][1][keep_]]), n, k_, dev))
                    if a.pe == 'lap': pf = pf * torch.where((pf * PE_ev).sum(0) < 0, -1.0, 1.0)  # align eigenvector signs with the full-graph PE
                    PE_fold.append(pf)
    ctx.pe = PE_ev

    def hub_cap(keep, gen):
        """each gene (OG) keeps at most a.hub_cap of its currently kept edges, chosen at random"""
        key = torch.where(keep, OG.double() * 2 + torch.rand(m, device=dev, generator=gen).double() * 0.999, torch.full((m,), 2.0 * n + 10, device=dev, dtype=torch.float64))
        order = torch.argsort(key)
        cnt = torch.bincount(OG[keep], minlength=n)
        rank = torch.arange(m, device=dev) - (torch.cumsum(cnt, 0) - cnt)[OG[order]]
        k2 = torch.zeros_like(keep); k2[order[keep[order] & (rank < a.hub_cap)]] = True
        return k2

    def full_graph(drop=None, fold=None, train=False):
        keep = torch.ones(m, dtype=torch.bool, device=dev)
        if drop is not None and len(drop) > 0:
            keep[drop] = False
        if a.hub_cap > 0 and (train or a.hub_eval == 'cap'):
            keep = hub_cap(keep, None if train else torch.Generator(device=dev).manual_seed(123))
        if need_A:
            ctx.A = torch.zeros(n, n, dtype=torch.bool, device=dev); ctx.A[OT[keep], OG[keep]] = True
        gdeg = torch.bincount(OG[keep], minlength=n).float()
        ctx.deg = torch.log1p(gdeg) / 3
        p, q = U0[keep], U1[keep]
        ei = torch.stack([torch.cat([p, q]), torch.cat([q, p])])
        if a.encoder == 'dir':
            ei = (torch.stack([OT[keep], OG[keep]]), torch.stack([OG[keep], OT[keep]]))
        ew = torch.cat([w_u[keep], w_u[keep]]) if w_u is not None else None
        if a.idf_beta > 0:
            w_e = gdeg[OG].clamp(min=1) ** -a.idf_beta
            ew = torch.cat([w_e[keep], w_e[keep]])
        if PG is not None:
            pg = PG['ev'] if fold is None else PG['fold'][fold]
            return (pg if a.mp_graph == 'panda' else torch.cat([ei, pg], 1)), ew
        if a.encoder == 'npanda':
            return {r: g for r, g in zip('RCP', (ei, G_C, G_P)) if r in a.relations}, None
        return ei, ew

    tri_sup = tri
    if a.split == 'fixed':  # fixed graph/supervision split by undirected edge (so an edge and its reverse row never straddle the split)
        mp_e = np.zeros(m, dtype=bool); mp_e[np.random.RandomState(a.seed).permutation(m)[:int(round(a.mp_frac * m))]] = True
        u_np = D['uid'][tri]
        tri_sup = tri[(u_np < 0) | ~mp_e[np.maximum(u_np, 0)]]  # negatives + positives not in the graph
        EI_FIX = full_graph(torch.as_tensor(np.where(~mp_e)[0], device=dev), train=True)
        FIX_STATE = (ctx.A, ctx.deg)
        D['fixed'] = (mp_e, tri_sup)  # exposed for the w11 audit
    P_tr = T(tr[['TF', 'Target']].values[tri_sup], torch.long); Y_tr = T(tr.Label.values[tri_sup], torch.float32)
    uid_tr = uid[T(tri_sup, torch.long)]
    P_va = T(tr[['TF', 'Target']].values[vai], torch.long); y_va = tr.Label.values[vai]; tf_va = tr.TF.values[vai]
    P_te = T(te[['TF', 'Target']].values, torch.long); y_te = te.Label.values; tf_te = te.TF.values

    ctx.n = n
    if a.encoder == 'lightgcn' and a.emb_feat != 'none':
        FM = {'expr': X, 'gf': Fe, 'stats': T(D['xs']) if D['xs'] is not None else None}[a.emb_feat]; ctx.d_feat = FM.shape[1]
    model = PairGNN(X.shape[1] + (a.pe_dim if a.pe != 'none' and a.pe_use == 'concat' else 0) + int(bool(a.deg_feat)), Fe.shape[1] if Fe is not None else 0, a, 0 if S_tr is None else (1 if S_tr.dim() == 2 else S_tr.shape[2]), ctx).to(dev)
    if a.encoder == 'lightgcn' and a.emb_feat != 'none': model.featmat = FM
    groups = [{'params': [q for nm, q in model.named_parameters() if not nm.startswith('sw.') and not (a.encoder == 'lightgcn' and nm == 'emb.weight')]}]
    if a.dec_wd >= 0:  # W16: decoder-side parameters get their own weight decay
        is_dec = lambda nm: not nm.startswith(('sw.', 'convs.', 'emb.', 'proj.'))
        groups = [{'params': [q for nm, q in model.named_parameters() if not is_dec(nm) and not nm.startswith('sw.') and nm != 'emb.weight']},
                  {'params': [q for nm, q in model.named_parameters() if is_dec(nm)], 'weight_decay': a.dec_wd}]
    if a.encoder == 'lightgcn': groups.append({'params': [model.emb.weight], 'weight_decay': a.emb_l2})
    if model.sw is not None:
        groups.append({'params': model.sw.parameters(), 'lr': a.lr * a.scalar_lr, 'weight_decay': 0.0})
    opt = getattr(torch.optim, a.optimizer)(groups, lr=a.lr, weight_decay=a.wd)
    sc = lambda S, P: None if S is None else S[P[:, 0], P[:, 1]]

    def evaluate(with_val=True):
        model.eval()
        with torch.no_grad():
            ctx.pe = PE_ev
            ei, ew = full_graph()
            ut, ug = model.nodes(X, Fe, ei, ew)
            st = model.score(ut, ug, P_te, sc(S_ev, P_te)).cpu().numpy()
            sv = model.score(ut, ug, P_va, sc(S_ev, P_va)).cpu().numpy() if with_val else None
        return (metrics(y_va, sv, tf_va) if with_val else None), metrics(y_te, st, tf_te), st

    tfu_all = np.unique(np.concatenate([tr.TF.values, te.TF.values]))

    def dump():
        """full TF x gene score grid, gene embeddings, and out-of-fold scores of all supervision rows (20 folds; a row is never scored through its own edge)"""
        model.eval()
        with torch.no_grad():
            ctx.pe = PE_ev
            ei, ew = full_graph()
            ut, ug = model.nodes(X, Fe, ei, ew)
            pairs = T(np.array([(u, g) for u in tfu_all for g in range(n)]), torch.long)
            grid = model.score(ut, ug, pairs, None).cpu().numpy().reshape(len(tfu_all), n)
            oof = np.zeros(len(tri_sup), np.float32)
            for h in np.array_split(np.random.RandomState(7).permutation(len(tri_sup)), 20):
                th = T(h, torch.long); d = uid_tr[th]
                ei, ew = full_graph(d[d >= 0])
                u2, g2 = model.nodes(X, Fe, ei, ew)
                oof[h] = model.score(u2, g2, P_tr[th], None).cpu().numpy()
        return dict(grid=grid, tfu=tfu_all, ug=ug.cpu().numpy(), oof=oof)

    safe = (PG is not None and a.panda_safe) or pe_fold or bool(a.fold_batches)  # fold-pure batches: batch rows come from one OOF fold, whose PANDA prior excludes them
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
            if a.split == 'fixed':
                (ei, ew), (ctx.A, ctx.deg) = EI_FIX, FIX_STATE
            else:
                ei, ew = full_graph(drop, fk, train=True)
            if PE_ev is not None:
                ctx.pe = PE_fold[fk] if PE_fold is not None else PE_ev
                if a.pe == 'lap' and a.pe_flip and a.pe_use == 'concat':
                    ctx.pe = ctx.pe * (torch.randint(0, 2, (a.pe_dim,), device=dev) * 2 - 1)
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
                if a.save_sel_grid: D['sel'] = dict(epoch=ep, **dump())
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
        tfu = tfu_all
        pairs = T(np.array([(u, g) for u in tfu for g in range(n)]), torch.long)
        grid = model.score(ut, ug, pairs, sc(S_ev, pairs)).cpu().numpy().reshape(len(tfu), n)
        if a.save_sel_grid: D['extra'] = dump()
        if a.decoder == 'qatt':  # attention diagnostics on the test pairs: entropy, number of known targets, masked max cosine similarity
            _, ms_, ent_, nb_ = model.qctx(ug, P_te)
            D['qdiag'] = dict(q_ent=ent_.cpu().numpy(), q_nb=nb_.cpu().numpy(), q_maxsim=ms_.cpu().numpy())
        if a.own_edge_diag:  # does the model read "my own edge is in the graph" as the label? (these edges are training examples, scored after training)
            rs = np.random.RandomState(123)
            sel = rs.choice(m, size=max(1, int(0.1 * m)), replace=False)
            neg_rows = tri[tr.Label.values[tri] == 0]
            Q = T(np.concatenate([D['orient'][sel], tr[['TF', 'Target']].values[rs.choice(neg_rows, size=len(sel), replace=False)]]), torch.long)
            yq = np.r_[np.ones(len(sel)), np.zeros(len(sel))]
            sq = {}
            for cond, drop in (('present', None), ('removed', T(sel, torch.long))):
                ei, ew = full_graph(drop)
                ut, ug = model.nodes(X, Fe, ei, ew)
                sq[cond] = model.score(ut, ug, Q, sc(S_ev, Q)).cpu().numpy()
            k = len(sel); gap = sq['present'] - sq['removed']
            D['diag'] = dict(n_edges=int(k), n_neg=int(k), gap_pos_mean=float(gap[:k].mean()), gap_pos_median=float(np.median(gap[:k])), gap_neg_mean=float(gap[k:].mean()),
                             pos_logit_present=float(sq['present'][:k].mean()), pos_logit_removed=float(sq['removed'][:k].mean()),
                             neg_logit_present=float(sq['present'][k:].mean()), neg_logit_removed=float(sq['removed'][k:].mean()),
                             auc_present=float(roc_auc_score(yq, sq['present'])), auc_removed=float(roc_auc_score(yq, sq['removed'])))
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
        if 'diag' in D2: res['own_edge_diag'] = D2['diag']
        os.makedirs(a.out, exist_ok=True)
        ex = {f'refit_{k}': v for k, v in D2.get('extra', {}).items() if k != 'grid'}
        np.savez_compressed(f'{a.out}/{a.name}.npz', logit=st, tf=D2['te'].TF.values, tgt=D2['te'].Target.values, y=D2['te'].Label.values, grid_tf=tfu, grid=grid, **D2.get('qdiag', {}), **ex)
        if 'sel' in D:
            np.savez_compressed(f'{a.out}/{a.name}_sel.npz', **D['sel'])
    res['minutes'] = (time.time() - t0) / 60
    json.dump(res, open(f'{a.out}/{a.name}.json', 'w'), indent=1)
    t = res.get('refit', best['test'])
    print(f"{a.name}\tepoch {best['epoch']}\tval AUROC {best['val']['auc']:.4f}\t{'REFIT ' if a.refit else ''}TEST AUROC {t['auc']:.4f} AUPRC {t['aupr']:.4f} withinTF {t['within']:.4f}\t{res['minutes']:.1f} min")


if __name__ == '__main__':
    main()
