"""W12 idea 4: Correct & Smooth on a graph of (TF, gene) pairs, post-hoc on a trained pair GNN.
Inputs per run <name> (trained with --save_sel_grid 1): <name>_sel.npz (best-validation-epoch model: score grid, gene embeddings, out-of-fold scores of
its training rows) and <name>.npz (refit model: same, trained on all training rows).
Pair graph: (t,g)~(t,g') for g' in the gene kNN of g (label-free co-expression, or cosine of the model's own gene embeddings), and optionally
(t,g)~(t',g) for the TFs t' with the largest Jaccard overlap of training targets.
Correct: E = (1-a1) R + a1 Op(E), R = y - sigmoid(out-of-fold score) on labelled training pairs; p1 = p + s E.
Smooth: Z = (1-a2) Z0 + a2 Op(Z), Z0 = p1 with the training labels clamped.
Hyperparameters are chosen per run on the validation slice (stage 1 model, training labels = the 90% training rows, validation rows scored),
criterion = validation AUROC (the metric that picks the epoch of the GNN itself); the alternative criterion mean(val AUROC, val within-TF) is also reported (validation within-TF is noisy: ~9 positives per TF, so the maximum over ~500 configurations is a winner's-curse pick); the same hyperparameters are then applied to the refit model with all
training labels and the test set is scored once. Out-of-fold scores mean no training pair is scored through its own edge.
Run: python3.12 analysis/w12_cs.py [run-family ...]   (default: w12_cs_hero w12_cs_herogf)   ->  analysis/out/w12_cs_<family>.json"""
import itertools, json, sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, 'analysis'); sys.path.insert(0, '.')
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
import w12_common as C
from src.pair_gnn import prepare, metrics, within_tf

KS = (5, 10, 20)
GRAPHS = ('coexp', 'emb')
EDGES = ('gene', 'tf', 'both')
A1S, SC, A2S = (0.0, 0.5, 0.8, 0.95), (0.5, 1.0, 2.0), (0.0, 0.5, 0.8, 0.95)
ITER = 30
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
T_ = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=DEV)


def knn_op(S, k):
    """symmetric kNN graph from a similarity matrix (diagonal excluded), D^-1/2 A D^-1/2"""
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, k, 1)[:, :k]
    A = np.zeros(S.shape, np.float32); A[np.repeat(np.arange(len(S)), k), nb.ravel()] = 1; A = np.maximum(A, A.T)
    d = np.maximum(A.sum(1), 1)
    return T_(A / np.sqrt(d[:, None] * d[None]))


def tf_op(Yp, k=5):
    """TF-TF operator: Jaccard overlap of positive training targets, top-k TFs per TF, symmetric-normalised"""
    B = (Yp > 0).astype(np.float32); inter = B @ B.T; J = inter / np.maximum(B.sum(1)[:, None] + B.sum(1)[None] - inter, 1)
    return knn_op(J + 1e-6 * np.random.RandomState(0).rand(*J.shape), min(k, len(J) - 1))


def propagate(Z0, wg, Sg, St, a, iters=ITER):
    Z = Z0
    for _ in range(iters):
        nz = (wg * (Z @ Sg) if wg > 0 else 0) + ((1 - wg) * (St @ Z) if wg < 1 else 0)
        Z = (1 - a) * Z0 + a * nz
    return Z


def run_cs(grid, lab, ops, hp):
    """grid: logits (T, n); lab: (t_idx, g_idx, y, oof_logit) of labelled training pairs; ops: (Sg, St); hp: (wg, a1, s, a2). Returns scores (T, n)."""
    wg, a1, s, a2 = hp
    Sg, St = ops
    ti, gi, y, oof = lab
    p = T_(expit(grid))
    p1 = p
    ti_, gi_ = torch.as_tensor(ti, device=DEV), torch.as_tensor(gi, device=DEV)
    if a1 > 0:
        R = torch.zeros_like(p); R[ti_, gi_] = T_(y - expit(oof))
        p1 = p + s * propagate(R, wg, Sg, St, a1)
    if a2 > 0:
        Z0 = p1.clamp(0, 1); Z0[ti_, gi_] = T_(y)
        return propagate(Z0, wg, Sg, St, a2).cpu().numpy()
    return p1.cpu().numpy()


def gene_ops(kind, k, x, ug):
    if kind == 'coexp':
        Cm = np.nan_to_num(np.corrcoef(x)); Cm[Cm < 0] = -1; return knn_op(Cm.astype(np.float32), k)
    u = ug / (np.linalg.norm(ug, axis=1, keepdims=True) + 1e-9); return knn_op(u @ u.T, k)


def stage(grid, tfu, ug, oof, tr, rows_lab, rows_eval, x):
    """returns a function hp -> scores on rows_eval, plus the labelled-row bundle"""
    ti_of = {t: i for i, t in enumerate(tfu)}
    ti = np.array([ti_of[t] for t in tr.TF.values[rows_lab]]); gi = tr.Target.values[rows_lab]; y = tr.Label.values[rows_lab].astype(np.float32)
    Yp = np.zeros(grid.shape, np.float32); Yp[ti[y == 1], gi[y == 1]] = 1
    St = tf_op(Yp)
    cache = {}

    def ops(kind, k):
        if (kind, k) not in cache: cache[(kind, k)] = gene_ops(kind, k, x, ug)
        return cache[(kind, k)], St
    return (ti, gi, y, oof), ops, ti_of


def grid_search(grid, tfu, ug, oof, tr, tri, vai, x):
    lab, ops, ti_of = stage(grid, tfu, ug, oof, tr, tri, vai, x)
    tv = np.array([ti_of[t] for t in tr.TF.values[vai]]); gv = tr.Target.values[vai]; yv = tr.Label.values[vai]; tfv = tr.TF.values[vai]
    res = []
    for kind, k, edge in itertools.product(GRAPHS, KS, EDGES):
        wg = {'gene': 1.0, 'tf': 0.0, 'both': 0.5}[edge]
        if edge == 'tf' and (kind, k) != (GRAPHS[0], KS[0]): continue  # the TF-only operator does not depend on the gene graph
        op = ops(kind, k)
        for a1, s, a2 in itertools.product(A1S, SC, A2S):
            if a1 == 0 and s != SC[0]: continue
            if a1 == 0 and a2 == 0: continue
            sv = run_cs(grid, lab, op, (wg, a1, s, a2))[tv, gv]
            res.append(dict(graph=kind, k=k, edge=edge, a1=a1, s=s, a2=a2, auc=float(roc_auc_score(yv, sv)), within=within_tf(yv, sv, tfv)))
    base = expit(grid)[tv, gv]
    b = dict(auc=float(roc_auc_score(yv, base)), within=within_tf(yv, base, tfv))
    for r in res: r['crit'] = r['auc']; r['crit_mix'] = 0.5 * (r['auc'] + r['within'])
    return b, sorted(res, key=lambda r: -r['crit'])


def final(hp, grid, tfu, ug, oof, tr, te, x):
    lab, ops, ti_of = stage(grid, tfu, ug, oof, tr, np.arange(len(tr)), None, x)
    wg = {'gene': 1.0, 'tf': 0.0, 'both': 0.5}[hp['edge']]
    Z = run_cs(grid, lab, ops(hp['graph'], hp['k']), (wg, hp['a1'], hp['s'], hp['a2']))
    return Z, (lab, ops, wg)


def evalz(Z, tfu, te):
    ti_of = {t: i for i, t in enumerate(tfu)}
    s = Z[[ti_of[t] for t in te.TF.values], te.Target.values]
    return s, metrics(te.Label.values, s, te.TF.values)


def one(name, pop):
    args = json.load(open(f'{C.OUT}{name}.json'))['args']
    a = SimpleNamespace(**{**args, 'scfm': 0})
    D = prepare(a, None); tr, te, tri, vai, x = D['tr'], D['te'], D['tri'], D['vai'], D['x']
    S, Rf = np.load(f'{C.OUT}{name}_sel.npz'), np.load(f'{C.OUT}{name}.npz')
    base_val, res = grid_search(S['grid'], S['tfu'], S['ug'], S['oof'], tr, tri, vai, x)
    best = res[0]
    mix = max(res, key=lambda r: r['crit_mix'])
    out = dict(name=name, seed=args['seed'], val_baseline=base_val, chosen=best, chosen_mix=mix, top5=res[:5], n_configs=len(res))
    tfu = Rf['grid_tf']; grid = Rf['grid']; ug = Rf['refit_ug']; oof = Rf['refit_oof']
    te_s0, m0 = evalz(expit(grid), tfu, te)
    Z, (lab, ops, wg) = final(best, grid, tfu, ug, oof, tr, te, x)
    s1, m1 = evalz(Z, tfu, te)
    out.update(test_raw=m0, test_cs=m1, pop_corr_raw=float(np.mean([C.spearmanr(g, pop).statistic for g in expit(grid)])),
               pop_corr_cs=float(np.mean([C.spearmanr(g, pop).statistic for g in Z])))
    # which part carries the gain: same hyperparameters with one part switched off (test scored once per variant, no re-selection)
    var = {'correct_only': (wg, best['a1'], best['s'], 0.0), 'smooth_only': (wg, 0.0, 0.5, best['a2']),
           'gene_edges_only': (1.0, best['a1'], best['s'], best['a2']), 'tf_edges_only': (0.0, best['a1'], best['s'], best['a2'])}
    op = ops(best['graph'], best['k'])
    Zm, _ = final(mix, grid, tfu, ug, oof, tr, te, x); out['test_cs_mix'] = evalz(Zm, tfu, te)[1]
    out['parts'] = {}
    for v, hp in var.items():
        if (hp[1] == 0 and hp[3] == 0): continue
        out['parts'][v] = evalz(run_cs(grid, lab, op, hp), tfu, te)[1]
    out['_s_cs'] = s1.tolist(); out['_s_raw'] = te_s0.tolist()
    return out


def main():
    fams = sys.argv[1:] or ['w12_cs_hero', 'w12_cs_herogf']
    pop = C.popularity()
    for fam in fams:
        outs = []
        for sd in C.SEEDS:
            nm = f'{fam}_s{sd}'
            try:
                outs.append(one(nm, pop))
            except FileNotFoundError as e:
                print('missing', nm, e)
        json.dump(outs, open(f'analysis/out/w12_cs_{fam.replace("w12_cs_", "")}.json', 'w'))
        for o in outs:
            print(o['name'], 'chosen', {k: o['chosen'][k] for k in ('graph', 'k', 'edge', 'a1', 's', 'a2')}, 'val AUROC base %.4f -> %.4f' % (o['val_baseline']['auc'], o['chosen']['crit']),
                  '| test within %.4f -> %.4f auc %.4f -> %.4f | within-mix pick: %s test within %.4f auc %.4f' % (o['test_raw']['within'], o['test_cs']['within'], o['test_raw']['auc'], o['test_cs']['auc'], {k: o['chosen_mix'][k] for k in ('graph', 'k', 'edge', 'a1', 's', 'a2')}, o['test_cs_mix']['within'], o['test_cs_mix']['auc']))


if __name__ == '__main__':
    main()
