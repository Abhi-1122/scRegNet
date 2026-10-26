"""Held-out TFs, CPU only: score a TF with no training targets by borrowing the targets of similar TFs
(PANDA's 'responsibility' step), similarity from co-expression or STRING PPI; compared with popularity.
Run: python3.12 analysis/w5_heldout_tf_transfer.py"""
import glob, sys
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, '.')
from src.pair_gnn import heldout_split, metrics
P = 'data/hESC/TFs+500/'
L = pd.concat([pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)], ignore_index=True)
x = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0).values
n = len(x)
C = np.nan_to_num(np.corrcoef(x))
g = pd.read_csv(P + 'Target.csv', index_col=0).set_index('Gene')['index']
e = pd.read_csv(glob.glob(P + 'STRING/*.symmetric_edges.csv')[0])
e = e[e.Gene1.isin(g.index) & e.Gene2.isin(g.index)]
PPI = np.zeros((n, n)); PPI[e.Gene1.map(g), e.Gene2.map(g)] = PPI[e.Gene2.map(g), e.Gene1.map(g)] = 1
r = lambda v: rankdata(v) / len(v)
res = {}
for fold in range(5):
    tr, te = heldout_split(L, SimpleNamespace(protocol='tf', fold=fold), n)
    pos = tr[tr.Label == 1]
    A = np.zeros((n, n)); A[pos.TF, pos.Target] = 1
    tfs = np.unique(tr.TF)
    t, gg, y = te.TF.values, te.Target.values, te.Label.values
    pop = np.log1p(A.sum(0))[gg]
    def transfer(S):
        W = np.clip(S[np.ix_(np.unique(t), tfs)], 0, None)  # similarity of each held-out TF to the training TFs
        Sc = (W @ A[tfs]) / (W.sum(1, keepdims=True) + 1e-9)
        return Sc[np.searchsorted(np.unique(t), t), gg]
    sc = {'popularity': pop, 'co-expression transfer': transfer(C), 'PPI transfer': transfer(PPI),
          'direct co-expression corr(t,g)': C[t, gg]}
    sc['popularity + co-expr transfer (rank avg)'] = r(pop) + r(sc['co-expression transfer'])
    sc['popularity + corr(t,g) (rank avg)'] = r(pop) + r(sc['direct co-expression corr(t,g)'])
    for k, v in sc.items():
        res.setdefault(k, []).append(metrics(y, v, t))
print('HELD-OUT TFs, hESC, 5 folds (same folds as src.pair_gnn --protocol tf); mean ± std over folds')
for k, v in res.items():
    f = lambda m: f"{np.mean([z[m] for z in v]):.4f} ± {np.std([z[m] for z in v]):.4f}"
    print(f"{k:44s} AUROC {f('auc')}  AUPRC {f('aupr')}  within-TF {f('within')}  within>=8 {f('within8')}")
