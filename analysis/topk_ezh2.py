"""(1) precision@k within each TF's test candidates (what a biologist looks at: the top of one TF's list)
(2) EZH2: do the model's top NEW predictions (genes that are not training targets) recover held-out EZH2
    targets, and do they form the developmental program that EZH2's real targets form (GO enrichment)?"""
import glob, json, re, warnings
import numpy as np, pandas as pd, gseapy
from scipy.stats import hypergeom
warnings.filterwarnings('ignore')
O = 'analysis/out/pair/'
P = 'data/hESC/TFs+500/'
genes = pd.read_csv(P + 'Target.csv', index_col=0).set_index('index')['Gene']
tr = pd.read_csv(P + 'Train_set.csv', index_col=0); te = pd.read_csv(P + 'Test_set.csv', index_col=0)
GO = json.load(open('analysis/data/GO_BP_2023.json')); BG = list(genes.values)
CFG = {'f01_A_tuned': 'scRegNet design, tuned', 'f04_C_tuned': 'GNN-only hero (SAGE)', 'f03_B_tuned': 'hero + Geneformer', 'f06_C_panda': 'hero + PANDA'}
Z = {c: [np.load(f) for f in sorted(glob.glob(f'{O}{c}_s*.npz'))] for c in CFG}
out = []

def p_at_k(z, k):
    res = []
    for t in np.unique(z['tf']):
        m = z['tf'] == t
        if z['y'][m].sum() >= k:
            top = np.argsort(-z['logit'][m])[:k]
            res.append(z['y'][m][top].mean())
    return np.mean(res), len(res)

out.append('precision@k inside each TF\'s test candidates (TFs with at least k test positives; base rate in brackets)')
for k in (10, 20, 50):
    base = np.mean([te.Label[te.TF == t].mean() for t in te.TF.unique() if te.Label[te.TF == t].sum() >= k])
    row = ' | '.join(f'{nm}: {np.mean([p_at_k(z, k)[0] for z in Z[c]]):.3f}' for c, nm in CFG.items() if Z[c])
    n = p_at_k(Z['f04_C_tuned'][0], k)[1] if Z['f04_C_tuned'] else 0
    out.append(f'  k={k:2d} ({n} TFs, base rate {base:.3f}): {row}')

ez = int(genes[genes == 'EZH2'].index[0])
train_t = set(tr[(tr.TF == ez) & (tr.Label == 1)].Target); test_t = set(te[(te.TF == ez) & (te.Label == 1)].Target)
cand = set(te[te.TF == ez].Target)                      # EZH2's labelled test candidates (real test targets + non-targets)
out.append(f'\nEZH2: {len(train_t)} training targets, {len(test_t)} held-out real targets among {len(cand)} test candidates')
out.append('top-30 of EZH2\'s TEST candidates: held-out targets recovered (chance level ~ %.1f)' % (30 * len(test_t) / len(cand)))
for c, nm in CFG.items():
    if not Z[c]: continue
    g = np.mean([z['grid'][list(z['grid_tf']).index(ez)] for z in Z[c]], 0)
    rank = [i for i in np.argsort(-g) if i in cand][:30]
    hit = sum(i in test_t for i in rank)
    top60 = [i for i in np.argsort(-g) if i not in train_t and i != ez][:60]
    r = gseapy.enrich(gene_list=list(genes[top60].values), gene_sets=GO, background=BG, outdir=None, cutoff=1.0, verbose=False).results
    r = r[r['Overlap'].str.split('/').str[0].astype(int) >= 3].sort_values('Adjusted P-value').head(3)
    terms = '; '.join(f"{x['Term'].split(' (')[0]} (p={x['Adjusted P-value']:.2g})" for _, x in r.iterrows())
    out.append(f'  {nm:26s} recovered {hit}/30 | GO of top-60 new predictions: {terms}')
txt = '\n'.join(out); open('analysis/out/topk_ezh2.txt', 'w').write(txt); print(txt)
