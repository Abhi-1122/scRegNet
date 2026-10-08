"""Item 2: rebuild PANDA with inputs switched off, on one fixed TF/gene universe.

Runs netZooPy's own normalize + update loop (calc.normalize_network, calc.compute_panda)
directly, so every variant shares the full run's 910 genes and TF set. A missing input is
replaced by the identity matrix, which is what PANDA itself does when an input is absent.
Writes, per variant, the same two files src/build_panda.py writes:
  panda_full_matrix.npy            (full prior: used to score test edges)
  panda_full_matrix_train_oof.npy  (5-fold out-of-fold: used to score train edges, no leakage)
"""
import os, sys
import numpy as np
import pandas as pd
from netZooPy.panda import calculations as calc
sys.path.insert(0, 'analysis')
from common import load, D

tr, te, genes, expr, tfs = load()
n = len(genes)
idx = {g: i for i, g in genes.items()}
tf_set = set(tfs['TF'])
e = pd.read_csv(D + 'STRING/hESC_TFs500_STRING_PPI.edges.csv')
e = e[e.Gene1.isin(tf_set) & e.Gene2.isin(tf_set)]
train_pos = tr[tr.Label == 1]
# same TF universe as build_panda.py's union mode: TFs in the PPI file + TFs in the prior
tf_univ = sorted(set(e.Gene1) | set(e.Gene2) | set(genes[train_pos.TF.unique()]))
tf_row = np.array([idx[t] for t in tf_univ])  # global gene index of each PANDA row
t2r = {t: r for r, t in enumerate(tf_univ)}

corr = np.nan_to_num(np.corrcoef(expr))
np.fill_diagonal(corr, 1)
ppi = np.identity(len(tf_univ))
a, b = e.Gene1.map(t2r).values, e.Gene2.map(t2r).values
ppi[a, b] = ppi[b, a] = 1

VARIANTS = {  # name: (use co-expression, use PPI)
    'full': (True, True),
    'no_coexpr': (False, True),
    'no_ppi': (True, False),
    'prior_only': (False, False),
}


def run(pos, use_c, use_p):
    motif = np.zeros((len(tf_univ), n))
    motif[[t2r[genes[t]] for t in pos.TF], pos.Target.values] = 1.0
    C = corr if use_c else np.identity(n)
    P = ppi if use_p else np.identity(len(tf_univ))
    W = calc.compute_panda(calc.normalize_network(C), calc.normalize_network(P),
                           calc.normalize_network(motif), alpha=0.1)
    full = np.zeros((n, n), dtype=np.float32)
    full[tf_row] = W
    return full


def build(name, use_c, use_p):
    out = D + f'PANDA_variants/{name}/'
    os.makedirs(out, exist_ok=True)
    raw = run(train_pos, use_c, use_p)
    scale = raw[tfs["index"].values].std()  # same as build_panda.py: std over all 410 TF rows
    np.save(out + 'panda_full_matrix.npy', raw / scale)
    oof = raw / scale
    rng = np.random.RandomState(42)  # identical fold split to src/build_panda.py
    for held in np.array_split(rng.permutation(train_pos.index.to_numpy()), 5):
        f = run(train_pos.loc[~train_pos.index.isin(held)], use_c, use_p) / scale
        h = train_pos.loc[held]
        oof[h.TF.values, h.Target.values] = f[h.TF.values, h.Target.values]
    np.save(out + 'panda_full_matrix_train_oof.npy', oof)
    return raw / scale


if __name__ == '__main__':
    import contextlib, io
    for name in sys.argv[1:] or VARIANTS:
        with contextlib.redirect_stdout(io.StringIO()):  # compute_panda prints every step
            m = build(name, *VARIANTS[name])
        print(name, 'done')
        if name == 'full':  # validate the driver against the PANDA matrix the experiments used
            ref = np.load(D + 'PANDA/panda_full_matrix.npy')
            r = np.corrcoef(m[tf_row].ravel(), ref[tf_row].ravel())[0, 1]
            print(f'  driver vs original build_panda.py matrix: Pearson r = {r:.6f}, '
                  f'max abs diff = {np.abs(m - ref).max():.2e}')
    # control: raw TF-target correlation as the "PANDA" scalar (uses no labels, so no OOF needed)
    out = D + 'PANDA_variants/raw_corr/'
    os.makedirs(out, exist_ok=True)
    rc = np.zeros((n, n), dtype=np.float32)
    rc[tf_row] = corr[tf_row]
    rc /= rc[tfs["index"].values].std()
    np.save(out + 'panda_full_matrix.npy', rc)
    np.save(out + 'panda_full_matrix_train_oof.npy', rc)
    print('raw_corr done')
