"""Item 2 follow-up: build a SYMMETRIC out-of-fold matrix. build_panda.py re-scores only the train
POSITIVES out-of-fold and leaves train negatives on the full-prior matrix, so positives and negatives
are scored under different conditions. Here every train row (pos and neg) in fold k is scored by a
PANDA run whose prior excludes fold k's positives. Test edges keep the full-prior matrix as before."""
import contextlib, io, os, sys
import numpy as np
sys.path.insert(0, 'analysis')
import item2_panda_variants as V
from common import metrics, D

tr, te = V.tr, V.te
for name in sys.argv[1:] or ['full', 'no_ppi', 'no_coexpr']:
    use_c, use_p = V.VARIANTS[name]
    out = D + f'PANDA_variants/{name}_symoof/'
    os.makedirs(out, exist_ok=True)
    full = np.load(D + f'PANDA_variants/{name}/panda_full_matrix.npy')
    tf_scale = V.run(V.train_pos, use_c, use_p)[V.tfs['index'].values].std()  # same scale as the full matrix
    sym = full.copy()
    rng = np.random.RandomState(42)
    for held in np.array_split(rng.permutation(len(tr)), 5):
        rows = tr.iloc[held]
        prior = V.train_pos.loc[~V.train_pos.index.isin(rows.index)]
        with contextlib.redirect_stdout(io.StringIO()):
            f = V.run(prior, use_c, use_p) / tf_scale
        sym[rows.TF.values, rows.Target.values] = f[rows.TF.values, rows.Target.values]
    np.save(out + 'panda_full_matrix.npy', full)
    np.save(out + 'panda_full_matrix_train_oof.npy', sym)
    old = np.load(D + f'PANDA_variants/{name}/panda_full_matrix_train_oof.npy')
    trs = lambda M: metrics(tr.Label.values, M[tr.TF.values, tr.Target.values], tr.TF.values)['withinTF_AUROC']
    tes = metrics(te.Label.values, full[te.TF.values, te.Target.values], te.TF.values)['withinTF_AUROC']
    print(f'{name:10s} within-TF AUROC of the scalar on TRAIN rows: positives-only OOF {trs(old):.4f} -> '
          f'symmetric OOF {trs(sym):.4f}   (on TEST rows: {tes:.4f})')
