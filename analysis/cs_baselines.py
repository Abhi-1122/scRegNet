"""Link-prediction baselines from the literature review, CPU only, on the benchmark split.
Everything is computed from the TF x gene matrix of TRAINING positives. A pair's own edge never
enters its own score: test pairs are absent from the matrix, and for training rows (used only to fit
the stackers) scores come from 5-fold out-of-fold matrices that drop the fold's positives.
Run: python3.12 analysis/cs_baselines.py [cell_type]"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from src.pair_gnn import metrics

cell = sys.argv[1] if len(sys.argv) > 1 else 'hESC'
P = f'data/{cell}/TFs+500/'
tr = pd.read_csv(P + 'Train_set.csv', index_col=0)
te = pd.read_csv(P + 'Test_set.csv', index_col=0)
n = len(pd.read_csv(P + 'Target.csv', index_col=0))
tf_ids = np.unique(np.concatenate([tr.TF.values, te.TF.values]))
row = {t: i for i, t in enumerate(tf_ids)}
cos = lambda M: (M @ M.T) / (np.sqrt((M * M).sum(1, keepdims=True)) * np.sqrt((M * M).sum(1, keepdims=True)).T + 1e-9)


def feature_mats(A):
    """A: (n_tf, n) 0/1 training matrix. Returns dict of (n_tf, n) score matrices."""
    dt, dg = A.sum(1, keepdims=True), A.sum(0, keepdims=True)
    St = cos(A); np.fill_diagonal(St, 0)                       # TF-TF similarity from shared targets
    user_cf = (St @ A) / (St.sum(1, keepdims=True) + 1e-9)     # do TFs like me target g?
    Sg = cos(A.T); np.fill_diagonal(Sg, 0)                     # gene-gene similarity from shared regulators
    item_cf = (A @ Sg) / (dt + 1e-9)                           # does g look like my known targets?
    K3 = A @ A.T @ A                                           # number of 3-paths t - g' - t' - g
    np.fill_diagonal(K3[:, :0], 0) if False else None
    ra_t = (A / np.maximum(dg, 1)) @ A.T                       # TF-TF similarity weighting shared targets by 1/popularity
    np.fill_diagonal(ra_t, 0)
    ra = (ra_t @ A) / (ra_t.sum(1, keepdims=True) + 1e-9)
    return dict(pa=np.log1p(dt) + np.log1p(dg), user_cf=user_cf, item_cf=item_cf, k3=np.log1p(K3), ra3=ra,
                logdt=np.repeat(np.log1p(dt), n, 1), logdg=np.repeat(np.log1p(dg), len(A), 0))


def build(df, A_by_row, names):
    r = df.TF.map(row).values; g = df.Target.values
    return np.column_stack([A_by_row[k][r, g] for k in names])


A_full = np.zeros((len(tf_ids), n))
pos = tr[tr.Label == 1]
A_full[pos.TF.map(row).values, pos.Target.values] = 1
F_full = feature_mats(A_full)

# out-of-fold features for the training rows
rng = np.random.RandomState(42)
oof = {k: np.zeros(len(tr)) for k in F_full}
for held in np.array_split(rng.permutation(len(tr)), 5):
    rows = tr.iloc[held]
    A = A_full.copy()
    hp = rows[rows.Label == 1]
    A[hp.TF.map(row).values, hp.Target.values] = 0
    Fk = feature_mats(A)
    for k in F_full:
        oof[k][held] = Fk[k][rows.TF.map(row).values, rows.Target.values]

names = list(F_full)
y_te, tf_te = te.Label.values, te.TF.values
out = {}
print(f'{cell}: single heuristics on the test set (no learning)')
for k in ['pa', 'user_cf', 'item_cf', 'k3', 'ra3']:
    s = F_full[k][te.TF.map(row).values, te.Target.values]
    out[k] = metrics(y_te, s, tf_te)
    print(f"  {k:10s} AUROC {out[k]['auc']:.4f} AUPRC {out[k]['aupr']:.4f} within-TF {out[k]['within']:.4f}")

# optional PANDA / contrast scalars (hESC only; cached by analysis/scalars.py with val_frac 0)
extra = {}
for kind in ['contrast', 'panda']:
    f = f'{P}scalars/{kind}_v0.0_s0_'
    try:
        extra[kind] = (np.load(f + 'tr.npy'), np.load(f + 'ev.npy'))
    except FileNotFoundError:
        pass


def feats(sets):
    cols_tr = [oof[k][:, None] for k in sets if k in F_full]
    cols_te = [F_full[k][te.TF.map(row).values, te.Target.values][:, None] for k in sets if k in F_full]
    for k in sets:
        if k in extra:
            cols_tr.append(extra[k][0][tr.TF.values, tr.Target.values][:, None])
            cols_te.append(extra[k][1][te.TF.values, te.Target.values][:, None])
    return np.hstack(cols_tr), np.hstack(cols_te)


STACKS = {
    'popularity (logdt, logdg)': ['logdt', 'logdg'],
    '+ user-CF + item-CF': ['logdt', 'logdg', 'user_cf', 'item_cf'],
    '+ 3-paths + RA': ['logdt', 'logdg', 'user_cf', 'item_cf', 'k3', 'ra3'],
    '+ co-expression contrast': ['logdt', 'logdg', 'user_cf', 'item_cf', 'k3', 'ra3', 'contrast'],
    '+ contrast + PANDA': ['logdt', 'logdg', 'user_cf', 'item_cf', 'k3', 'ra3', 'contrast', 'panda'],
}
saved = {}
print(f'{cell}: stackers (trained on out-of-fold training features)')
for nm, sets in STACKS.items():
    if any(k in ('contrast', 'panda') and k not in extra for k in sets):
        continue
    Xtr, Xte = feats(sets)
    sc = StandardScaler().fit(Xtr)
    for mname, mdl in [('LR', LogisticRegression(max_iter=2000)), ('GBM', HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))]:
        mdl.fit(sc.transform(Xtr), tr.Label.values)
        s = mdl.decision_function(sc.transform(Xte)) if mname == 'LR' else mdl.predict_proba(sc.transform(Xte))[:, 1]
        m = metrics(y_te, s, tf_te); out[f'{nm} [{mname}]'] = m
        saved[f'{nm} [{mname}]'.replace(' ', '_')] = s
        print(f"  {nm:28s} {mname:3s} AUROC {m['auc']:.4f} AUPRC {m['aupr']:.4f} within-TF {m['within']:.4f}")
json.dump(out, open(f'analysis/out/cs_baselines_{cell}.json', 'w'), indent=1)
np.savez_compressed(f'analysis/out/cs_scores_{cell}.npz', tf=te.TF.values, tgt=te.Target.values, y=te.Label.values, **saved)
