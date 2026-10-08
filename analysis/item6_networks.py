"""Item 6: compare the networks each method predicts, structurally and biologically (GO BP 2023,
background = the 910-gene panel so the TF-heavy panel doesn't inflate 'transcription' terms)."""
import json, sys, warnings
import numpy as np
import pandas as pd
import gseapy
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, 'analysis')
from common import load, D
warnings.filterwarnings('ignore')

tr, te, genes, expr, tfs = load()
n = len(genes); t, g, y = te.TF.values, te.Target.values, te.Label.values
GO = json.load(open('analysis/data/GO_BP_2023.json'))
BG = list(genes.values)
z = lambda r: np.load(f'analysis/out/scores_tf_500_hESC_{r}.npz')

# ---- scores on the test candidate pairs ----
W = np.zeros((n, n)); p = tr[tr.Label == 1]; W[p.TF.values, p.Target.values] = 1
C = np.nan_to_num(np.corrcoef(expr)); np.fill_diagonal(C, 0)
Cz = (C - C.mean(1, keepdims=True)) / C.std(1, keepdims=True)
def coexp_contrast(a, b):  # Item 2's winner: z-scored corr with TF's train targets minus with its non-targets
    tg, nt = W[a] == 1, W[a] == 0
    nt[b] = False
    return Cz[b, tg].mean() - Cz[b, nt].mean() if tg.any() else 0.0
rows = tr.groupby('TF').size().reindex(range(n), fill_value=0).values
def feats(df, loo):
    tt, gg, yy = df.TF.values, df.Target.values, df.Label.values; sp = (yy == 1) * 1.0 if loo else 0
    return np.column_stack([(W.sum(1)[tt] - sp) / np.maximum(rows[tt] - loo, 1), np.log1p(W.sum(0)[gg] - sp)])
S = {
    'baseline scRegNet': (lambda x: x[:, 1] - x[:, 0])(z('baseline_rerun')['test_logits']),
    'Exp C (+PANDA)': (lambda x: x[:, 1] - x[:, 0])(z('panda_edge')['test_logits']),
    'PANDA alone': np.load(D + 'PANDA/panda_full_matrix.npy')[t, g],
    'co-exp contrast alone': np.array([coexp_contrast(a, b) for a, b in zip(t, g)]),
    'degree-only LR': LogisticRegression(max_iter=1000).fit(feats(tr, 1), tr.Label).decision_function(feats(te, 0)),
}
K = int(y.sum())
nets = {'TRUE test edges': set(np.flatnonzero(y))}
for k, s in S.items():
    nets[k] = set(np.argsort(-s)[:K])


def tf_specificity(edge_ids):  # mean Jaccard between the target sets of different TFs (lower = more TF-specific)
    sets = {}
    for e in edge_ids: sets.setdefault(t[e], set()).add(g[e])
    big = [v for v in sets.values() if len(v) >= 5]
    js = [len(a & b) / len(a | b) for i, a in enumerate(big) for b in big[i + 1:]]
    return np.mean(js) if js else np.nan, len(sets)


true_per_tf = pd.Series(t[y == 1]).value_counts()
print(f'=== each method\'s network = its top {K} test pairs (K = number of true test edges) ===')
print(f"{'network':22s} {'precision':>9s} {'#TFs':>5s} {'TF-target-set overlap':>22s} {'edges/TF vs truth (Spearman)':>29s}")
for k, e in nets.items():
    prec = len(e & nets['TRUE test edges']) / K
    spec, ntf = tf_specificity(e)
    per = pd.Series(t[list(e)]).value_counts().reindex(true_per_tf.index, fill_value=0)
    print(f'{k:22s} {prec:9.3f} {ntf:5d} {spec:22.3f} {spearmanr(per, true_per_tf)[0]:29.3f}')
names = list(nets)
print('\nJaccard overlap between networks:')
print(pd.DataFrame([[len(nets[a] & nets[b]) / len(nets[a] | nets[b]) for b in names] for a in names],
                   index=names, columns=[x[:10] for x in names]).round(2).to_string())


def go(genelist, label, top=6):
    r = gseapy.enrich(gene_list=list(genelist), gene_sets=GO, background=BG, outdir=None, cutoff=1.0, verbose=False).results
    r = r[r['Overlap'].str.split('/').str[0].astype(int) >= 3].sort_values('Adjusted P-value').head(top)
    print(f'\n--- GO: {label} ({len(genelist)} genes) ---')
    for _, x in r.iterrows():
        print(f"  {x['Adjusted P-value']:.2e}  {x['Overlap']:>7s}  {x['Term'][:80]}")


# 1) what the baseline thinks a "target" is: its single TF-agnostic gene ranking
allp, alll = z('baseline_rerun')['all_pairs'], z('baseline_rerun')['all_logits']
row = allp[:, 0] == allp[0, 0]
gene_score = pd.Series((alll[row, 1] - alll[row, 0]), index=genes[allp[row, 1]].values)
tf_names = set(tfs['TF'])
nontf = gene_score[~gene_score.index.isin(tf_names)]
go(nontf.sort_values(ascending=False).index[:100], 'baseline\'s 100 most "target-like" non-TF genes (same list for every TF)')
go(nontf.sort_values().index[:100], 'baseline\'s 100 least "target-like" non-TF genes')
pos_all = pd.concat([tr, te]); pos_all = pos_all[pos_all.Label == 1]
popular = pos_all.Target.value_counts()
go(genes[popular.index[:100]].values, 'the 100 genes bound by the most TFs in the ground truth')

# 2) EZH2: real targets vs. each model's top predictions (excluding EZH2's training targets)
ezh2 = genes[genes == 'EZH2'].index[0]
go(genes[pos_all[pos_all.TF == ezh2].Target].values, 'EZH2 true targets (train+test)')
for run, label in [('baseline_rerun', 'baseline'), ('panda_edge', 'Exp C')]:
    zz = z(run); m = zz['all_pairs'][:, 0] == ezh2
    sc = pd.Series(zz['all_logits'][m, 1] - zz['all_logits'][m, 0], index=zz['all_pairs'][m, 1])
    sc = sc.drop(np.flatnonzero(W[ezh2]), errors='ignore')
    top = sc.sort_values(ascending=False).index[:60]
    test_pos = set(te[(te.TF == ezh2) & (te.Label == 1)].Target)
    print(f'\n{label}: EZH2 top-60 new predictions contain {len(set(top) & test_pos)}/{len(test_pos)} held-out EZH2 targets')
    go(genes[top].values, f'{label}: EZH2 top-60 predicted new targets')
pm = np.load(D + 'PANDA/panda_full_matrix.npy')[ezh2]; pm[W[ezh2] == 1] = -np.inf
go(genes[np.argsort(-pm)[:60]].values, 'PANDA alone: EZH2 top-60 predicted new targets')

# 3) the edges Exp C adds to the network that the baseline did not have
added = nets['Exp C (+PANDA)'] - nets['baseline scRegNet']
removed = nets['baseline scRegNet'] - nets['Exp C (+PANDA)']
yy = set(np.flatnonzero(y))
print(f'\nExp C vs baseline network: {len(added)} edges added ({len(added & yy)} real), '
      f'{len(removed)} removed ({len(removed & yy)} real)')
print('  TFs of added edges:', pd.Series(genes[t[list(added)]].values).value_counts().head(8).to_dict())
