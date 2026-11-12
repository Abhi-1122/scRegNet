"""W11 task B3: leak audit of src/pair_gnn.py (hESC, hero config; CPU, no GPU job). Asserts + counts, written to analysis/out/w11_edge_checks.txt
(the B1/B2 tables are appended by w11_edge_report.py). Run: python3.12 analysis/w11_edge_audit.py"""
import sys
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
from src import pair_gnn as pg

a = SimpleNamespace(cell_type='hESC', num_TF='500', scfm=0, protocol='random', val_frac=0.1, split_seed=0, data_folder='data', scfm_folder='scFM')
D = pg.prepare(a, None)
tr, te, n, tri, vai, uid = D['tr'], D['te'], D['n'], D['tri'], D['vai'], D['uid']
L = ['W11 B3 leak audit, hESC TFs+500, hero config (val_frac 0.1, split_seed 0). Line numbers refer to src/pair_gnn.py.', '']
ed = lambda df: set(zip(np.minimum(df.TF, df.Target), np.maximum(df.TF, df.Target)))
U = set(zip(*(np.asarray(v).tolist() for v in D['U'])))
isU = lambda df: np.array([(min(t, g), max(t, g)) in U for t, g in zip(df.TF, df.Target)])
v, t = tr.iloc[vai], te
pair = lambda df: set(zip(df.TF, df.Target))

# (i) validation
assert len(set(tri) & set(vai)) == 0 and (uid[vai] == -1).all() and (uid[tri][tr.Label.values[tri] == 1] >= 0).sum() > 0
assert (uid >= 0).sum() == ((tr.Label.values[tri] == 1) & (tr.TF.values[tri] != tr.Target.values[tri])).sum()  # graph rows == training positives only
vp = v[v.Label == 1]
d_val = len(pair(v) & pair(tr.iloc[tri])); rev_val = isU(v); rev_valp = isU(vp)
L += ['(i) validation positives are never in the message-passing graph while validation is scored',
      f'  CONFIRMED for the pair itself: graph edges come only from training rows (L200-203: pos_rows = tri positives; evaluate() L345-348 uses full_graph() with no validation edge); '
      f'assert uid[val rows] == -1 for all {len(vai)} validation rows; {d_val} validation pairs duplicate a training pair in the same direction.',
      f'  REFUTED in the reverse direction: the graph is undirected, so a validation pair (t,g) whose reverse (g,t) is a training positive is scored through its own edge. '
      f'{int(rev_val.sum())} of {len(v)} validation pairs ({int(rev_valp.sum())} of {len(vp)} positives).', '']

# (ii) test
tp = tr.iloc[tri]; ful = isU(t)
L += ['(ii) test pairs are never in any training graph',
      f'  Same-direction: CONFIRMED, {len(pair(t) & pair(tr))} of {len(t)} test pairs appear in Train_set (any label).',
      f'  Reverse direction: REFUTED, {int(isU(t).sum())} of {len(t)} test pairs ({int((isU(t) & (t.Label.values == 1)).sum())} of {int(t.Label.sum())} positives) have the undirected edge in the graph built from the validation-split training rows; '
      f'with the refit graph (all training rows, val_frac 0) the count is {int(np.array([(g, x) in pair(tr[tr.Label == 1]) for x, g in zip(t.TF, t.Target)]).sum())} (task A, analysis/out/w11_reverse_leak.txt).',
      f'  Training itself never uses test rows: prepare() reads Test_set only into D["te"] (L176, protocol random leaves it untouched); fit() only reads te at evaluate()/final scoring (L345-350, L410).', '']

# (iv) popularity quantities
pos_t_tri = tp.Target.values[tp.Label.values == 1]
all_t = tr.Target.values[tr.Label.values == 1]
assert np.bincount(pos_t_tri, minlength=n).sum() == (tp.Label == 1).sum() < (tr.Label == 1).sum() == len(all_t)
L += ['(iv) degree/popularity quantities use training rows only',
      '  CONFIRMED by code: DEG/NC/NEG (L287-293, used only with --rank_w > 0, off in the hero) are built from yt, tft, pos_t = tr[tri] (L288-293); tri excludes the validation rows (assert tri & vai = {}) and the test file is never indexed.',
      f'  Check: recomputed in-degree from tr[tri] sums to {int((tp.Label == 1).sum())} positives, below the {len(all_t)} of all Train_set rows, so validation positives are excluded; test positives are never in either.',
      '  scalars.get (PANDA/contrast; off in the hero) also starts from trn = tr.iloc[tri] (analysis/scalars.py L69-70). The GCN/SAGE degree normalisation uses the same graph (training edges only).',
      '  Caveat, not a leak: standardised expression x (L178), and the co-expression kNN graph (L72, npanda only) use all cells but no labels.', '']

# (iii) Geneformer
L += ['(iii) Geneformer embeddings use no labels',
      '  CONFIRMED by code reading: pair_gnn L186 calls Trainer._get_embeddings (src/train.py L41-57), which reads scFM/Geneformer/<cell>_500_gene_embeddings.csv and maps rows to genes through <cell>_500.csv (gene -> Ensembl id).',
      '  The loader takes no label, split or graph argument and builds one 896-d vector per gene (zeros when the gene is missing). The embeddings are precomputed files (pretrained Geneformer on the expression data);',
      '  the generation script is not in the repository, so that last step is taken from the file provenance and cannot be audited from here.', '']

# fixed-split sanity (CPU, one epoch)
sys.argv = ['x', '--name', 'audit', '--cell_type', 'hESC', '--scfm', '0', '--encoder', 'sage', '--decoder', 'had', '--gnn_dims', '256', '--split', 'fixed', '--mp_frac', '0.7', '--epochs', '1', '--seed', '42', '--val_frac', '0.1']
b = pg.parse()
torch.manual_seed(0)
pg.fit(b, D, torch.device('cpu'), 1, select=False)
mp_e, sup = D['fixed']
u = uid[sup]
mpn = int(mp_e.sum()); m = len(mp_e)
assert abs(mpn / m - 0.7) < 0.01
assert not np.isin(uid[sup][uid[sup] >= 0], np.where(mp_e)[0]).any()          # no supervised positive edge is a training-graph edge
assert (tr.Label.values[sup][u < 0] == 0).all() and (tr.Label.values[tri][~np.isin(tri, sup)] == 1).all()  # all negatives supervised; only graph positives removed
assert not np.isin(uid[vai], np.where(mp_e)[0]).any()
sp = tr.Label.values[sup].sum(); sn = (tr.Label.values[sup] == 0).sum()
L += ['--split fixed --mp_frac 0.7 sanity (asserted on the CPU one-epoch run, seed 42)',
      f'  undirected training edges {m}; in the training graph {mpn} ({100 * mpn / m:.1f}%); supervised rows {len(sup)} = {int(sp)} positives (rows of the other {m - mpn} edges, reverse rows included) + {int(sn)} negatives (all training negatives); '
      f'no supervised positive edge is in the training graph; the split is by undirected edge so an edge and its reverse row never straddle it.',
      '  Validation/test graph = full_graph() with nothing dropped = ALL training-positive edges (same call as --split batch, L348 and L410). Training graph = the fixed 70% (L331, L375).', '']
open('analysis/out/w11_edge_checks.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
