"""Pair-level scalar features for src.pair_gnn, built so no row is scored with its own label.

kinds:
  contrast : z-scored co-expression of gene g with the TF's known targets minus with its non-targets
  panda    : PANDA score (prior = known targets, co-expression, TF-TF PPI)
Validation/test rows use a prior built from ALL graph positives (training rows minus the validation
slice). Training rows (positives AND negatives) are scored out-of-fold: a row in fold k is scored
with a prior that excludes the positives of fold k ("symmetric" out-of-fold).
Run once to fill the cache:  python3.12 analysis/scalars.py hESC 500
"""
import contextlib, glob, io, os, sys
import numpy as np
import pandas as pd
from netZooPy.panda import calculations as calc


def _fn_contrast(x, n):
    C = np.nan_to_num(np.corrcoef(x))
    np.fill_diagonal(C, 1)
    Cz = calc.normalize_network(C)
    np.fill_diagonal(Cz, 0)

    def fn(pos):
        W = np.zeros((n, n))
        W[pos.TF.values, pos.Target.values] = 1
        cnt = W.sum(1, keepdims=True)
        S = W @ Cz
        return S / np.maximum(cnt, 1) - (Cz.sum(0, keepdims=True) - S) / np.maximum(n - cnt - 1, 1)
    return fn


def _fn_panda(x, n, D, gp):
    path = D['path']
    genes = pd.read_csv(path + 'Target.csv', index_col=0).set_index('index')['Gene']
    idx = {g: i for i, g in genes.items()}
    tfs = pd.read_csv(path + 'TF.csv', index_col=0)
    tf_set = set(tfs['TF'])
    e = pd.read_csv(glob.glob(path + 'STRING/*.edges.csv')[0])
    e = e[e.Gene1.isin(tf_set) & e.Gene2.isin(tf_set)]
    tf_univ = sorted(set(e.Gene1) | set(e.Gene2) | set(genes[gp.TF.unique()]))  # fixed across all folds
    t2r = {t: r for r, t in enumerate(tf_univ)}
    tf_row = np.array([idx[t] for t in tf_univ])
    corr = np.nan_to_num(np.corrcoef(x))
    np.fill_diagonal(corr, 1)
    ppi = np.identity(len(tf_univ))
    a, b = e.Gene1.map(t2r).values, e.Gene2.map(t2r).values
    ppi[a, b] = ppi[b, a] = 1
    Cn, Pn = calc.normalize_network(corr), calc.normalize_network(ppi)

    def fn(pos):
        motif = np.zeros((len(tf_univ), n))
        motif[[t2r[genes[t]] for t in pos.TF], pos.Target.values] = 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            Wm = calc.compute_panda(Cn.copy(), Pn.copy(), calc.normalize_network(motif), alpha=0.1)
        full = np.zeros((n, n), dtype=np.float32)
        full[tf_row] = Wm
        return full
    fn.tf_index = tfs['index'].values
    return fn


def get(a, D):
    tr, tri, n = D['tr'], D['tri'], D['n']
    kind = a.scalar
    pr = getattr(a, 'protocol', 'random')
    cache = f"{D['path']}scalars/{kind}_v{a.val_frac}_s{a.split_seed}" + ('' if pr == 'random' else f"_{pr}{a.fold}")
    if os.path.exists(cache + '_ev.npy'):
        return np.load(cache + '_ev.npy'), np.load(cache + '_tr.npy')
    trn = tr.iloc[tri]
    gp = trn[trn.Label == 1]
    fn = _fn_contrast(D['x'], n) if kind == 'contrast' else _fn_panda(D['x'], n, D, gp)
    full = fn(gp)
    oof = full.copy()
    rng = np.random.RandomState(42)
    for held in np.array_split(rng.permutation(len(trn)), 5):
        rows = trn.iloc[held]
        f = fn(gp.drop(index=rows.index.intersection(gp.index)))
        oof[rows.TF.values, rows.Target.values] = f[rows.TF.values, rows.Target.values]
    scale_rows = getattr(fn, 'tf_index', np.unique(tr.TF.values))
    scale = full[scale_rows].std() + 1e-8
    full, oof = (full / scale).astype(np.float32), (oof / scale).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache + '_ev.npy', full)
    np.save(cache + '_tr.npy', oof)
    return full, oof


if __name__ == '__main__':
    sys.path.insert(0, '.')
    from types import SimpleNamespace
    from src.pair_gnn import prepare
    cell, ntf = sys.argv[1], sys.argv[2]
    kinds = sys.argv[3:] or ['contrast', 'panda']
    for k, vf in [(k, vf) for k in kinds for vf in (0.1, 0.0)]:
        a = SimpleNamespace(cell_type=cell, num_TF=ntf, data_folder='data', scfm=0, scfm_folder='scFM', val_frac=vf, split_seed=0, scalar=k)
        if k == 'panda' and not glob.glob(f'data/{cell}/TFs+{ntf}/STRING/*.edges.csv'):
            print(k, 'skipped: no STRING file for', cell); continue
        D = prepare(a, None)
        ev, tr_ = get(a, D)
        t, g, y = D['te'].TF.values, D['te'].Target.values, D['te'].Label.values
        from src.pair_gnn import within_tf
        from sklearn.metrics import roc_auc_score
        print(k, 'val_frac', vf, 'test AUROC of the scalar alone %.4f, within-TF %.4f' % (roc_auc_score(y, ev[t, g]), within_tf(y, ev[t, g], t)))
