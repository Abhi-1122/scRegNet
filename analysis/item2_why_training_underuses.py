"""Item 2 diagnosis: why does retraining extract less from no_ppi/no_coexpr than post-hoc stacking does?
(1) learned decoder weight on the PANDA scalar vs. the weight post-hoc stacking wants;
(2) does the scalar behave the same on training edges (out-of-fold matrix) as on test edges (full prior)?"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'analysis')
from common import load, metrics, D
tr, te, genes, expr, tfs = load()
t, g, y = te.TF.values, te.Target.values, te.Label.values
b = np.load('analysis/out/scores_tf_500_hESC_baseline_rerun.npz')['test_logits']; b = b[:, 1] - b[:, 0]
runs = {'full': ['tf_500_hESC_panda_edge', 'tf_500_hESC_panda_edge_seed43', 'tf_500_hESC_panda_edge_seed44']}
for v in ['no_coexpr', 'no_ppi']: runs[v] = [f'tf_500_hESC_pandavar_{v}_seed{s}' for s in (42, 43, 44)]
print(f"{'variant':10s} {'learned weight (3 seeds)':>26s} {'stacking weight':>16s} | {'within-TF AUROC of the scalar: train(OOF) vs test(full)':>40s} | {'train-pos mean / test-pos mean (z)':>34s}")
for v, rs in runs.items():
    w = [(lambda W: W[1, -1] - W[0, -1])(np.load(f'analysis/out/scores_{r}.npz')['W']) for r in rs]
    full = np.load(D + f'PANDA_variants/{v}/panda_full_matrix.npy'); oof = np.load(D + f'PANDA_variants/{v}/panda_full_matrix_train_oof.npy')
    ste = full[t, g]
    lr = LogisticRegression(max_iter=1000).fit(np.column_stack([b, ste]), y)
    s_tr = oof[tr.TF.values, tr.Target.values]; ytr = tr.Label.values
    a_tr = metrics(ytr, s_tr, tr.TF.values)['withinTF_AUROC']; a_te = metrics(y, ste, t)['withinTF_AUROC']
    # standardized position of positives relative to negatives, within TF, train vs test
    def zpos(s, yy, tt):
        out = []
        for k in np.unique(tt):
            m = tt == k
            if 0 < yy[m].sum() < m.sum() and s[m].std() > 0:
                out.append((s[m][yy[m] == 1].mean() - s[m][yy[m] == 0].mean()) / s[m].std())
        return np.mean(out)
    print(f"{v:10s} {' '.join(f'{x:.3f}' for x in w):>26s} {lr.coef_[0][1]:16.3f} | {a_tr:19.4f} vs {a_te:.4f}{'':12s} | "
          f"{zpos(s_tr, ytr, tr.TF.values):15.3f} / {zpos(ste, y, t):.3f}")
