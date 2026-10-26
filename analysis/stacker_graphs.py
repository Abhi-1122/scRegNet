"""Context-specific stacker: which GRAPH do the link-prediction features run on?
cs_baselines.py runs popularity / CF / 3-paths on the training ChIP-seq label graph only. Here the same features are also
computed on (a) the PANDA top-100 graph and (b) the label-free co-expression kNN graph (k=10), and stacked (LR, GBM).
Test rows use full-training PANDA; training rows use the PANDA of their own fold (that fold's positives are not in the prior),
same folds as cs_baselines. Run: python3.12 analysis/stacker_graphs.py   (hESC; imports cs_baselines, which re-runs it quietly)"""
import contextlib, io, json, sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
with contextlib.redirect_stdout(io.StringIO()):
    import cs_baselines as cb
from sklearn.metrics import roc_auc_score
from src.pair_gnn import coexp_graph, metrics, panda_graph, prepare
from types import SimpleNamespace
import scalars

tr, te, n, row, tf_ids = cb.tr, cb.te, cb.n, cb.row, cb.tf_ids
a = SimpleNamespace(cell_type=cb.cell, num_TF='500', data_folder='data', scfm=0, scfm_folder='scFM', val_frac=0.0, split_seed=0, scalar='panda', protocol='random', fold=0)
D = prepare(a, None)
Pfull, Pf = scalars.get(a, D)[0], scalars.get_folds(a, D)
folds = np.array_split(np.random.RandomState(42).permutation(len(tr)), 5)
rt, gt = tr.TF.map(row).values, tr.Target.values
re_, ge = te.TF.map(row).values, te.Target.values
K = 100


def pmat(M):  # binary PANDA top-K graph, (n_tf, n)
    E = panda_graph(M, tf_ids, K, n)
    A = np.zeros((len(tf_ids), n)); m = np.isin(E[0], tf_ids)
    A[[row[t] for t in E[0][m]], E[1][m]] = 1  # directed TF -> gene half only
    return A


# (a) PANDA-graph features: feature_mats on the PANDA graph; fold rows read their own fold's graph
Fp_full = cb.feature_mats(pmat(Pfull))
pg_tr = {k: np.zeros(len(tr)) for k in Fp_full}
for k, h in enumerate(folds):
    Fk = cb.feature_mats(pmat(Pf[k]))
    for f in Fk: pg_tr[f][h] = Fk[f][rt[h], gt[h]]
PG = ['pa', 'user_cf', 'item_cf', 'k3', 'ra3']
pg_tr = {f'pg_{f}': v for f, v in pg_tr.items() if f in PG}
pg_te = {f'pg_{f}': Fp_full[f][re_, ge] for f in PG}

# (b) co-expression-graph features (label-free except item-CF, which uses known targets out-of-fold)
Cn = np.zeros((n, n)); e = coexp_graph(D['x'], False); Cn[e[0], e[1]] = 1
corr = np.nan_to_num(np.corrcoef(D['x']))
deg = Cn.sum(1)
cn = Cn @ Cn
aa = (Cn / np.log(np.maximum(deg, 2))[None, :]) @ Cn
tfn = tf_ids  # global gene index of each label TF


def cx(r, g, A):  # r: row index into tf_ids; A: (n_tf, n) label matrix
    t = tfn[r]
    icf = (A @ Cn)[r, g] / (A.sum(1)[r] + 1e-9)
    return dict(cx_corr=corr[t, g], cx_cn=np.log1p(cn[t, g]), cx_aa=aa[t, g], cx_degg=np.log1p(deg[g]), cx_degt=np.log1p(deg[t]), cx_itemcf=icf)


cx_te = cx(re_, ge, cb.A_full)
cx_tr = {k: np.zeros(len(tr)) for k in cx_te}
for h in folds:
    A = cb.A_full.copy(); hp = tr.iloc[h]; hp = hp[hp.Label == 1]
    A[hp.TF.map(row).values, hp.Target.values] = 0
    c = cx(rt[h], gt[h], A)
    for k in c: cx_tr[k][h] = c[k]

LAB = ['logdt', 'logdg', 'user_cf', 'item_cf', 'k3', 'ra3']
RAW = ['panda']
PGF, CXF = list(pg_te), list(cx_te)
col = lambda src_tr, src_te, names: (np.column_stack([src_tr[k] for k in names]), np.column_stack([src_te[k] for k in names]))
tr_all = {**{k: cb.oof[k] for k in cb.oof}, **{'panda': cb.extra['panda'][0][tr.TF.values, tr.Target.values], 'contrast': cb.extra['contrast'][0][tr.TF.values, tr.Target.values]}, **pg_tr, **cx_tr}
te_all = {**{k: cb.F_full[k][re_, ge] for k in cb.F_full}, **{'panda': cb.extra['panda'][1][te.TF.values, te.Target.values], 'contrast': cb.extra['contrast'][1][te.TF.values, te.Target.values]}, **pg_te, **cx_te}
y_te, tf_te = te.Label.values, te.TF.values
out, saved = {}, {}
print('single features on the test set (raw AUROC, sign not flipped)')
for k in PGF + CXF + ['panda']:
    s = te_all[k]; out[f'single {k}'] = metrics(y_te, s, tf_te)
    print(f"  {k:12s} AUROC {out[f'single {k}']['auc']:.4f} AUPRC {out[f'single {k}']['aupr']:.4f} within-TF {out[f'single {k}']['within']:.4f}")
STK = {'label stack (3-paths + RA)': LAB, 'label + contrast + PANDA score (best in cs_baselines)': LAB + ['contrast', 'panda'],
       'context only (PANDA graph + PANDA score + co-expression graph)': PGF + RAW + CXF, 'PANDA graph only': PGF + RAW, 'co-expression graph only': CXF,
       'label + PANDA graph': LAB + PGF + RAW, 'label + co-expression graph': LAB + CXF, 'all': LAB + ['contrast'] + RAW + PGF + CXF}
from sklearn.ensemble import HistGradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
print('stacks (trained on out-of-fold training features)')
for nm, names in STK.items():
    Xtr, Xte = col(tr_all, te_all, names)
    sc = StandardScaler().fit(Xtr)
    for mn, mdl in [('LR', LR(max_iter=3000)), ('GBM', GB(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))]:
        mdl.fit(sc.transform(Xtr), tr.Label.values)
        s = mdl.decision_function(sc.transform(Xte)) if mn == 'LR' else mdl.predict_proba(sc.transform(Xte))[:, 1]
        out[f'{nm} [{mn}]'] = metrics(y_te, s, tf_te); saved[f'{nm} [{mn}]'.replace(' ', '_')] = s
        m = out[f'{nm} [{mn}]']
        print(f"  {nm:62s} {mn:3s} AUROC {m['auc']:.4f} AUPRC {m['aupr']:.4f} within-TF {m['within']:.4f}")
json.dump(out, open(f'analysis/out/w3_stack_{cb.cell}.json', 'w'), indent=1)
np.savez_compressed(f'analysis/out/w3_stack_{cb.cell}.npz', tf=te.TF.values, tgt=te.Target.values, y=y_te, **saved)
