"""W21 two-view model: complete network inference across cell types with a regulatory-prior view and a co-expression view.

View A: motif TF -> gene edges (Mf, reverse Mr) + STRING TF-TF edges (P) on all genes of the cell, 2-layer relational encoder.
View B: mutual kNN co-expression graph of the cell's own expression (k, positive-only or |r|; edge weight = Pearson r),
        2-layer weighted SAGE, own weights. Co-expression never scores a pair directly: it only shapes gene embeddings.
Fusion: concat (z = ReLU(W [zA; zB])), gate (z = g zA + (1-g) zB, g = sigmoid(w [zA; zB]) per gene), early (C as a
        4th relation inside encoder A). Single-view arms: A, B, mlp (self term only, no graph).
Decoder: MLP on [z_t * z_g ; motif_max, log1p motif_hits, missing flag] (W20 pair features 0-2).
Loss: TF-balanced BCE (every source TF weighs the same; positives and negatives of a TF weigh the same) or plain BCE;
      optional label-free InfoNCE between zA and zB of the same gene (--cons lambda), on the source graphs and, with
      --cons_tgt 1, also on the target graph. View dropout --vdrop: per batch, zero one view's embedding.
Protocol, data, features, validation split and evaluation are those of src/xcell_gnn.py (imported, not copied).
Leak audit: source labels are read first; graphs, features, training (incl. the target consistency term) and the
target score grid then run inside no_labels(), which raises on any read of a label file; the target's labels are
read only by evaluate() afterwards.

  python3.12 -m src.xcell_2view --src hESC --tgt hHEP --name X --model concat --mk 0 --ck 10 --cabs 0 --lr 3e-4
  python3.12 -m src.xcell_2view --jobs analysis/jobs/w21_p1.txt --shard 0 --nshard 16
  python3.12 -m src.xcell_2view --selfcheck
"""
import argparse, copy, json, os, shlex, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.xcell_gnn import CELLS, cell, evaluate, load_labels, motif_edges, no_labels, path, per_tf_auc

OUT = 'analysis/out/xcell2'
PF_COLS = [0, 1, 2]  # z(motif_max), z(log1p motif_hits), missing flag; never the Pearson column
RELS = {'mlp': [], 'A': ['Mf', 'Mr', 'P'], 'early': ['Mf', 'Mr', 'P', 'C'], 'concat': ['Mf', 'Mr', 'P'], 'gate': ['Mf', 'Mr', 'P']}
TWO = ('concat', 'gate')


def coexp_mknn(R, k, absolute):
    """mutual kNN on Pearson r (|r| if absolute, else r with r > 0 required); returns edges (2, E) and weights r"""
    n = len(R)
    S = np.abs(R) if absolute else R.copy()
    np.fill_diagonal(S, -np.inf)
    A = np.zeros((n, n), bool)
    A[np.repeat(np.arange(n), k), np.argsort(-S, 1)[:, :k].ravel()] = True
    A &= A.T
    if not absolute: A &= R > 0
    i, j = np.where(A)
    return np.stack([i, j]), R[i, j].astype(np.float32)


def val_tfs(L, seed):
    """the W20 source-validation split (src/xcell_gnn.run): 20% of the source's labelled TFs, seeded"""
    tfs = np.unique(L.TF.values)
    return np.random.RandomState(seed).choice(tfs, max(1, int(round(0.2 * len(tfs)))), replace=False)


def tf_balanced_w(tf, y):
    """per-pair weight: each TF sums to 1, split equally between its positives and negatives (one-class TF: uniform)"""
    w = np.zeros(len(y), np.float32)
    for t in np.unique(tf):
        m = tf == t; p = m & (y == 1); q = m & (y == 0)
        if p.any() and q.any(): w[p], w[q] = 0.5 / p.sum(), 0.5 / q.sum()
        else: w[m] = 1.0 / m.sum()
    return w


def inputs(c, a, dev):
    """label-free inputs of one cell on dev: node features, graphs {rel: (edges, weights)}, pair features"""
    D = cell(c)
    T = lambda v, dt=torch.long: torch.as_tensor(v, device=dev, dtype=dt)
    G = {}
    need = set(RELS.get(a.model, [])) | ({'C'} if a.model in TWO + ('B',) else set())
    if 'Mf' in need:
        m = motif_edges(D, a.mk); one = T(np.ones(m.shape[1]), torch.float32)
        G['Mf'], G['Mr'] = (T(m), one), (T(m[::-1].copy()), one)
    if 'P' in need: G['P'] = (T(D['P']), T(np.ones(D['P'].shape[1]), torch.float32))
    if 'C' in need:
        e, w = coexp_mknn(D['R'], a.ck, a.cabs); G['C'] = (T(e), T(w, torch.float32))
    X = T(D['X'], torch.float32)
    return dict(c=c, X=X, G=G, AX=agg(X, G), PF=T(D['PF'][..., PF_COLS], torch.float32))


def agg(X, G):
    """per relation: weighted mean of incoming neighbours, sum(w h) / sum(|w|) (signed r allowed); 0 without edges"""
    out = {}
    for r, (ei, w) in G.items():
        src, dst = ei
        num = torch.zeros_like(X).index_add_(0, dst, w[:, None] * X[src])
        den = torch.zeros(len(X), device=X.device).index_add_(0, dst, w.abs()).clamp(min=1e-12)
        out[r] = num / den[:, None]
    return out


class Enc(nn.Module):
    """2 layers of h' = W_self h + sum_r W_r agg_r(h); first-layer aggregation of the fixed input is cached (AX)"""
    def __init__(s, d, hid, rels, drop):
        super().__init__()
        s.drop, s.rels = drop, rels
        s.s1, s.s2 = nn.Linear(d, hid), nn.Linear(hid, hid)
        s.r1 = nn.ModuleDict({r: nn.Linear(d, hid, bias=False) for r in rels})
        s.r2 = nn.ModuleDict({r: nn.Linear(hid, hid, bias=False) for r in rels})

    def forward(s, X, G, AX):
        h = s.s1(X) + sum(s.r1[r](AX[r]) for r in s.rels)
        h = F.dropout(F.relu(h), s.drop, s.training)
        A2 = agg(h, {r: G[r] for r in s.rels})
        return s.s2(h) + sum(s.r2[r](A2[r]) for r in s.rels)


class TwoView(nn.Module):
    def __init__(s, d, a):
        super().__init__()
        s.kind, s.vdrop = a.model, a.vdrop
        s.encA = Enc(d, a.hid, RELS[a.model], a.dropout) if a.model != 'B' else None
        s.encB = Enc(d, a.hid, ['C'], a.dropout) if a.model in TWO + ('B',) else None
        if a.model == 'concat': s.fuse = nn.Linear(2 * a.hid, a.hid)
        if a.model == 'gate': s.gate = nn.Linear(2 * a.hid, 1)
        s.dec = nn.Sequential(nn.Linear(a.hid + len(PF_COLS), 128), nn.ReLU(), nn.Dropout(a.dropout), nn.Linear(128, 1))

    def views(s, I):
        return (s.encA(I['X'], I['G'], I['AX']) if s.encA else None), (s.encB(I['X'], I['G'], I['AX']) if s.encB else None)

    def nodes(s, zA, zB):
        if zB is None: return zA
        if zA is None: return zB
        if s.training and s.vdrop > 0 and np.random.rand() < s.vdrop:  # view dropout: one view zeroed for the whole batch
            if np.random.rand() < 0.5: zA = torch.zeros_like(zA)
            else: zB = torch.zeros_like(zB)
        cat = torch.cat([zA, zB], 1)
        if s.kind == 'concat': return F.relu(s.fuse(cat))
        g = torch.sigmoid(s.gate(cat))
        return g * zA + (1 - g) * zB

    def score(s, h, pr, PF):
        t, g = pr[:, 0], pr[:, 1]
        return s.dec(torch.cat([h[t] * h[g], PF[t, g]], 1)).squeeze(1)


def info_nce(zA, zB, tau):
    """label-free view consistency: zA_i should match zB_i against every other gene's zB (and symmetrically)"""
    a, b = F.normalize(zA, dim=1), F.normalize(zB, dim=1)
    lg = a @ b.T / tau; y = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y))


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--name')
    p.add_argument('--src', nargs='+', default=[])
    p.add_argument('--tgt')
    p.add_argument('--model', default='concat', choices=['concat', 'gate', 'early', 'A', 'B', 'mlp'])
    p.add_argument('--mk', type=int, default=0, help='motif graph: 0 = motif_hits > 0, k = top-k genes per TF by motif_max')
    p.add_argument('--ck', type=int, default=10, help='co-expression mutual kNN k')
    p.add_argument('--cabs', type=int, default=0, help='1: kNN on |r| (negative r edges kept, signed weights)')
    p.add_argument('--loss', default='bal', choices=['bal', 'bce'])
    p.add_argument('--vdrop', type=float, default=0.2)
    p.add_argument('--cons', type=float, default=0.0, help='weight of the InfoNCE view-consistency loss')
    p.add_argument('--cons_tgt', type=int, default=0, help='1: consistency loss also on the target graph (no labels)')
    p.add_argument('--tau', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.175)
    p.add_argument('--hid', type=int, default=256)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jobs'); p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--job_timeout', type=int, default=1800)
    p.add_argument('--selfcheck', action='store_true')
    return p.parse_args(argv)


def run(a, dev):
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    srcs, tgt = a.src, a.tgt
    assert tgt not in srcs, 'transfer: the target must not be a source'
    assert not (a.cons and a.model not in TWO), '--cons needs two views'
    Lab = {}
    for c in srcs:  # the only label reads before evaluate(): source cells, training loss and source validation
        L = load_labels(c); vm = L.TF.isin(val_tfs(L, a.seed)).values
        tr = L[~vm]
        Lab[c] = dict(Ptr=tr[['TF', 'Target']].values, Ytr=tr.Label.values,
                      Wtr=tf_balanced_w(tr.TF.values, tr.Label.values) if a.loss == 'bal' else np.ones(len(tr), np.float32),
                      Pva=L[['TF', 'Target']].values[vm], yva=L.Label.values[vm], tva=L.TF.values[vm])
    with no_labels():  # from here to the target grid any read of a label file raises
        T = lambda v, dt=torch.long: torch.as_tensor(v, device=dev, dtype=dt)
        S = []
        for c in srcs:
            I = inputs(c, a, dev); l = Lab[c]
            I.update(Ptr=T(l['Ptr']), Ytr=T(l['Ytr'], torch.float32), Wtr=T(l['Wtr'], torch.float32), Pva=T(l['Pva']), yva=l['yva'], tva=l['tva'])
            S.append(I)
        It = inputs(tgt, a, dev)  # target: label-free inputs only
        model = TwoView(S[0]['X'].shape[1], a).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)

        def val_score():
            model.eval(); pt = {}
            with torch.no_grad():
                for s in S:
                    sv = model.score(model.nodes(*model.views(s)), s['Pva'], s['PF']).cpu().numpy()
                    pt.update({f"{s['c']}:{t}": v for t, v in per_tf_auc(s['yva'], sv, s['tva']).items()})
            return float(np.mean(list(pt.values())))

        hist, best, bad, state, best_ep = [], -1, 0, None, 0
        for ep in range(1, a.epochs + 1):
            model.train()
            bl = [(i, perm[j:j + a.batch]) for i, s in enumerate(S) for perm in [torch.randperm(len(s['Ptr']), device=dev)] for j in range(0, len(perm), a.batch)]
            for k in np.random.permutation(len(bl)):
                i, idx = bl[k]; s = S[i]
                zA, zB = model.views(s)
                out = model.score(model.nodes(zA, zB), s['Ptr'][idx], s['PF'])
                w = s['Wtr'][idx]
                loss = (w * F.binary_cross_entropy_with_logits(out, s['Ytr'][idx], reduction='none')).sum() / w.sum()
                if a.cons:
                    loss = loss + a.cons * info_nce(zA, zB, a.tau)
                    if a.cons_tgt: loss = loss + a.cons * info_nce(*model.views(It), a.tau)
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
            zA, zB = model.views(It); h = model.nodes(zA, zB)
            gate = torch.sigmoid(model.gate(torch.cat([zA, zB], 1))).squeeze(1).cpu().numpy() if a.model == 'gate' else None
            pr = torch.as_tensor(np.array([(t, g) for t in Dt['tf'] for g in range(Dt['n'])]), device=dev)
            grid = torch.cat([model.score(h, pr[i:i + 65536], It['PF']) for i in range(0, len(pr), 65536)]).cpu().numpy().reshape(len(Dt['tf']), Dt['n'])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f'{OUT}/{a.name}.npz', grid=grid, grid_tf=Dt['tf'], **({} if gate is None else dict(gate=gate)))
    ev = evaluate(tgt, Dt['tf'], grid)  # the only read of the target's labels
    res = dict(name=a.name, args={k: v for k, v in vars(a).items() if k not in ('jobs', 'shard', 'nshard')}, srcs=srcs, tgt=tgt,
               best_epoch=best_ep, val_within=best, hist=hist, target=dict(auc=ev['auc'], aupr=ev['aupr'], within=ev['within']),
               minutes=(time.time() - t0) / 60)
    json.dump(res, open(f'{OUT}/{a.name}.json', 'w'), indent=1)
    print(f"{a.name}\tep {best_ep}\tval {best:.4f}\ttarget AUROC {ev['auc']:.4f} AUPRC {ev['aupr']:.4f} within {ev['within']:.4f}\t{res['minutes']:.1f} min", flush=True)


def selfcheck(dev):
    with no_labels():
        try:
            load_labels('hHEP'); raise SystemExit('guard failed')
        except AssertionError:
            pass
    a = parse(['--model', 'concat'])
    for c in CELLS:
        with no_labels():  # every graph and feature of every cell is built without label files
            for ck in (5, 10, 20):
                for cabs in (0, 1):
                    e, w = coexp_mknn(cell(c)['R'], ck, cabs)
                    assert (e[0] != e[1]).all() and np.isfinite(w).all() and (cabs or (w > 0).all())
                    M = set(map(tuple, e.T)); assert all((j, i) in M for i, j in M), 'mutual kNN not symmetric'
                    assert np.bincount(e[0], minlength=cell(c)['n']).max() <= ck
            for mk in (0, 50):
                I = inputs(c, argparse.Namespace(**{**vars(a), 'mk': mk}), dev)
                assert set(I['G']) == {'Mf', 'Mr', 'P', 'C'} and I['PF'].shape[-1] == 3 and torch.isfinite(I['X']).all()
        D = cell(c); L = load_labels(c)
        grid = np.zeros((len(D['tf']), D['n']), np.float32)
        grid[L.TF.map({t: i for i, t in enumerate(D['tf'])}).values, L.Target.values] = L.Label.values
        ev = evaluate(c, D['tf'], grid)
        assert abs(ev['within'] - 1) < 1e-12 and abs(ev['auc'] - 1) < 1e-12, ev
        w = tf_balanced_w(L.TF.values, L.Label.values)
        for t in np.unique(L.TF.values)[:5]:
            m = L.TF.values == t; assert abs(w[m].sum() - 1) < 1e-5
        print(c, 'ok: graphs and features label-free, perfect grid within-TF 1, TF weights sum to 1')
    # the target consistency term and the target grid run under no_labels(): one short source+target run must pass the guard
    os.makedirs(OUT, exist_ok=True)
    run(parse(['--src', 'mHSC-E', '--tgt', 'hHEP', '--name', 'w21_selfcheck', '--model', 'concat', '--cons', '0.1', '--cons_tgt', '1', '--epochs', '2']), dev)
    for f in ('json', 'npz'): os.remove(f'{OUT}/w21_selfcheck.{f}')
    print('selfcheck passed')


def main():
    a = parse()
    torch.set_num_threads(2)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if a.selfcheck: return selfcheck(dev)
    if a.jobs:
        lines = [l for l in open(a.jobs).read().splitlines() if l.strip() and not l.startswith('#')][a.shard::a.nshard]
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        for l in lines:
            b = parse(shlex.split(l))
            if os.path.exists(f'{OUT}/{b.name}.json'): continue
            try:
                subprocess.run([sys.executable, '-m', 'src.xcell_2view'] + shlex.split(l), env=env, timeout=a.job_timeout)
            except subprocess.TimeoutExpired:
                print(f'TIMEOUT {b.name} after {a.job_timeout} s', flush=True)
        return
    run(a, dev)


if __name__ == '__main__':
    main()
