"""How much of the truth sits inside the PANDA top-K graph, and how much of that is the prior echoing itself.
For K in 50/100/200 (full-training PANDA, val_frac 0): recall of test positives, share of the graph's test pairs that are
positive, recall of training positives in the FULL graph (naive: contains the prior) vs in their OWN fold-k graph (safe).
Run: python3.12 analysis/w3_graph_recall.py [cell]"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from types import SimpleNamespace
from src.pair_gnn import prepare
import scalars

cell = sys.argv[1] if len(sys.argv) > 1 else 'hESC'
a = SimpleNamespace(cell_type=cell, num_TF='500', data_folder='data', scfm=0, scfm_folder='scFM', val_frac=0.0, split_seed=0, scalar='panda', protocol='random', fold=0)
D = prepare(a, None)
tr, te, n = D['tr'], D['te'], D['n']
full, M = scalars.get(a, D)[0], scalars.get_folds(a, D)
tfs = np.unique(np.concatenate([tr.TF.values, te.TF.values]))
folds = np.array_split(np.random.RandomState(42).permutation(len(tr)), 5)
fold_of = np.zeros(len(tr), int)
for k, h in enumerate(folds): fold_of[h] = k
L = [f'{cell}: PANDA top-K graph diagnostics (test base rate {te.Label.mean():.3f}, train base rate {tr.Label.mean():.3f})',
     f"{'K':>4s} {'edges':>6s} {'test pos recall':>16s} {'test pairs in graph':>20s} {'precision':>10s} {'train pos recall FULL':>22s} {'train pos recall OWN-FOLD':>26s}"]
for K in (50, 100, 200):
    def topk(M):  # directed (TF, gene) keys of the top-K genes per TF, self excluded
        M = M[tfs].copy(); M[np.arange(len(tfs)), tfs] = -np.inf
        return np.repeat(tfs, K) * n + np.argsort(-M, 1)[:, :K].ravel()
    ins = lambda E, t, g: np.isin(t * n + g, E)
    Ef = topk(full)
    t, g, y = te.TF.values, te.Target.values, te.Label.values
    h = ins(Ef, t, g)
    tp, tg, ty = tr.TF.values, tr.Target.values, tr.Label.values
    full_hit = ins(Ef, tp, tg)
    own = np.zeros(len(tr), bool)
    for k in range(5):
        r = fold_of == k
        own[r] = ins(topk(M[k]), tp[r], tg[r])
    L.append(f'{K:4d} {len(Ef):6d} {h[y == 1].mean():16.3f} {h.sum():20d} {y[h].mean():10.3f} {full_hit[ty == 1].mean():22.3f} {own[ty == 1].mean():26.3f}')
K = 100
L.append(f'training positives among the K=100 edges of the full graph: {np.isin(topk(full), tp[ty == 1] * n + tg[ty == 1]).sum()} of {K * len(tfs)}')
L.append('train-pos recall FULL minus OWN-FOLD = the share of training positives that sit in the graph only because they are in the prior (the leak).')
open(f'analysis/out/w3_graph_recall_{cell}.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
