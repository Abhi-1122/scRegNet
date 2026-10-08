"""Item 5: test the biological assumptions behind PANDA's PPI term and behind using STRING as a
message-passing graph. This is a data analysis, so it uses all labelled positives (train+test).
(a) TF cooperativity: do STRING-connected TFs share more targets than expected from their sizes?
(b) Regulator homophily: do genes linked in a graph share regulators (the property a GNN needs)?"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, 'analysis')
from common import load, D

tr, te, genes, expr, tfs = load()
n = len(genes)
lab = pd.concat([tr, te])
R = np.zeros((n, n)); p = lab[lab.Label == 1]; R[p.TF.values, p.Target.values] = 1  # TF x gene, all positives
cand = np.zeros((n, n), bool); cand[lab.TF.values, lab.Target.values] = True        # pairs with a label at all
bench_tfs = np.unique(lab.TF.values)
idx = {g: i for i, g in genes.items()}

S = pd.read_csv(D + 'STRING/hESC_TFs500_STRING_PPI.symmetric_edges.csv')
S = S[S.Gene1.isin(idx) & S.Gene2.isin(idx)]
Sa = np.zeros((n, n)); Sa[S.Gene1.map(idx), S.Gene2.map(idx)] = S.score.values
Sa = np.maximum(Sa, Sa.T)

# (a) for each pair of benchmark TFs: target overlap vs. expectation, over genes labelled for both
rows = []
for i, a in enumerate(bench_tfs):
    for b in bench_tfs[i + 1:]:
        m = cand[a] & cand[b]
        na, nb, k, N = R[a, m].sum(), R[b, m].sum(), (R[a, m] * R[b, m]).sum(), m.sum()
        if na == 0 or nb == 0: continue
        exp = na * nb / N
        rows.append(dict(A=genes[a], B=genes[b], string=Sa[a, b], na=na, nb=nb, overlap=k,
                         expected=exp, log2_enrich=np.log2((k + 0.5) / (exp + 0.5))))
df = pd.DataFrame(rows)
con, unc = df[df.string > 0], df[df.string == 0]
print('(a) TF-TF target overlap, log2(observed / expected given set sizes); 0 = chance')
print(f'    STRING-connected TF pairs: n={len(con)}, median {con.log2_enrich.median():+.3f}, mean {con.log2_enrich.mean():+.3f}')
print(f'    unconnected TF pairs:      n={len(unc)}, median {unc.log2_enrich.median():+.3f}, mean {unc.log2_enrich.mean():+.3f}')
print(f'    Mann-Whitney (connected > unconnected): p = {mannwhitneyu(con.log2_enrich, unc.log2_enrich, alternative="greater").pvalue:.3g}')
print(f'    all pairs above chance (log2 > 0): {(df.log2_enrich > 0).mean():.2f} of TF pairs')
print(f'    Spearman(STRING score, enrichment) among connected pairs: {spearmanr(con.string, con.log2_enrich)[0]:+.3f}')
core = ['POU5F1', 'SOX2', 'NANOG']
print('    core pluripotency trio:')
print(df[df.A.isin(core) & df.B.isin(core)][['A', 'B', 'string', 'na', 'nb', 'overlap', 'expected', 'log2_enrich']]
      .round(3).to_string(index=False))
print('    top 8 most-enriched connected pairs:')
print(con.sort_values('log2_enrich', ascending=False).head(8)[['A', 'B', 'string', 'overlap', 'expected', 'log2_enrich']]
      .round(3).to_string(index=False))

# (b) regulator homophily: for gene pairs linked by a graph, how similar are their regulator sets?
reg = R[bench_tfs].T  # gene x benchmark-TF regulator profile
norm = np.linalg.norm(reg, axis=1)
has = norm > 0
cos = lambda u, v: (reg[u] * reg[v]).sum(1) / (norm[u] * norm[v])
C = np.nan_to_num(np.corrcoef(expr)); np.fill_diagonal(C, -2)
rng = np.random.RandomState(0)


def pairs_from(A):
    u, v = np.nonzero(np.triu(A, 1)); keep = has[u] & has[v]
    return u[keep], v[keep]


graphs = {}
graphs['STRING (score >= 0.7, used by PPI Methods 1/2)'] = pairs_from(Sa > 0)
k = 10
knn = np.zeros((n, n)); top = np.argsort(-C, 1)[:, :k]; knn[np.repeat(np.arange(n), k), top.ravel()] = 1
graphs[f'co-expression top-{k} neighbours (raw corr)'] = pairs_from(np.maximum(knn, knn.T))
Cz = (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True); np.fill_diagonal(Cz, -9)
knz = np.zeros((n, n)); topz = np.argsort(-(Cz + Cz.T), 1)[:, :k]; knz[np.repeat(np.arange(n), k), topz.ravel()] = 1
graphs[f'co-expression top-{k} neighbours (z-scored corr)'] = pairs_from(np.maximum(knz, knz.T))
u_all = rng.choice(np.where(has)[0], 20000); v_all = rng.choice(np.where(has)[0], 20000); ok = u_all != v_all
base = cos(u_all[ok], v_all[ok]).mean()
# degree-matched random pairs: shuffle within regulator-count bins, to separate "both popular" from "same regulators"
deg = reg.sum(1)
print(f'\n(b) regulator-profile cosine similarity between linked genes (random gene pairs: {base:.3f})')
for name, (u, v) in graphs.items():
    obs = cos(u, v).mean()
    bins = np.digitize(deg, [1, 2, 3, 5, 8, 13, 100])
    null = []
    for _ in range(20):
        u2 = np.array([rng.choice(np.where((bins == bins[x]) & has)[0]) for x in u])
        v2 = np.array([rng.choice(np.where((bins == bins[x]) & has)[0]) for x in v])
        m = u2 != v2; null.append(cos(u2[m], v2[m]).mean())
    print(f'    {name:52s} pairs={len(u):5d}  cosine {obs:.3f}  degree-matched null {np.mean(null):.3f}  '
          f'excess {obs - np.mean(null):+.3f}')
