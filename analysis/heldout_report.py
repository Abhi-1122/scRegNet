"""Held-out-gene and held-out-TF results: mean over folds and seeds. Popularity cannot help these tests:
a held-out gene has no training edges, a held-out TF has none either."""
import glob, json, re
import numpy as np
O = 'analysis/out/pair/'
CFG = {'f01_A_tuned': 'scRegNet design, tuned', 'f03_B_tuned': 'hero + Geneformer', 'f04_C_tuned': 'GNN-only hero (SAGE)', 'f05_C_contrast': 'GNN-only hero + co-expression number'}
import sys
from types import SimpleNamespace
import pandas as pd
sys.path.insert(0, '.')
from src.pair_gnn import heldout_split, metrics
P = 'data/hESC/TFs+500/'
L = pd.concat([pd.read_csv(P + 'Train_set.csv', index_col=0), pd.read_csv(P + 'Test_set.csv', index_col=0)], ignore_index=True)
n = len(pd.read_csv(P + 'Target.csv', index_col=0))
def popularity_reference(pr):
    """what popularity alone gives: TF base rate (held-out genes) or gene degree (held-out TFs), from training rows only"""
    res = []
    for fold in range(5):
        tr_, te_ = heldout_split(L, SimpleNamespace(protocol=pr, fold=fold), n)
        posr = tr_[tr_.Label == 1]
        if pr == 'gene':
            rate = tr_.groupby('TF').Label.mean(); s = te_.TF.map(rate).fillna(0).values
        else:
            deg = posr.Target.value_counts(); s = np.log1p(te_.Target.map(deg).fillna(0).values)
        res.append(metrics(te_.Label.values, s, te_.TF.values))
    return {k: np.mean([r[k] for r in res]) for k in res[0]}, {k: np.std([r[k] for r in res]) for k in res[0]}
lines = []
for pr, title in [('gene', 'HELD-OUT GENES (20% of target genes removed from training and from the graph; 5 folds x 2 seeds)'),
                  ('tf', 'HELD-OUT TFs (about 7 of 34 TFs removed from training and from the graph; 5 folds x 2 seeds)')]:
    lines.append(title)
    lines.append(f"{'model':40s} runs {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within-TF(>=8 pos)':>20s}")
    for c, nm in CFG.items():
        rs = [json.load(open(f)) for f in glob.glob(f'{O}ho_{pr}[0-9]_{c}_s*.json')]
        if not rs: continue
        fold_means = {}
        for r in rs:
            fold_means.setdefault(r['args']['fold'], []).append(r['refit'])
        g = lambda m: np.array([np.mean([x[m] for x in v]) for v in fold_means.values()])
        f = lambda m: f'{g(m).mean():.4f} ± {g(m).std():.4f}'
        lines.append(f"{nm:40s} {len(rs):3d}  {f('auc'):>16s} {f('aupr'):>16s} {f('within'):>16s} {f('within8'):>20s}")
    m_, s_ = popularity_reference(pr)
    nm = 'popularity only: TF base rate' if pr == 'gene' else 'popularity only: gene degree'
    lines.append(f"{nm:40s}  -   {m_['auc']:.4f} ± {s_['auc']:.4f}  {m_['aupr']:.4f} ± {s_['aupr']:.4f}  {m_['within']:.4f} ± {s_['within']:.4f}  {m_['within8']:.4f} ± {s_['within8']:.4f}")
    lines.append('')
txt = '\n'.join(lines); open('analysis/out/heldout_report.txt', 'w').write(txt); print(txt)

# ---- paired comparison over the 34 held-out TFs (each TF is held out in exactly one fold) ----
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
def per_tf_heldout(cfg):
    acc = {}
    for f in sorted(glob.glob(f'{O}ho_tf[0-9]_{cfg}_s*.npz')):
        z = np.load(f)
        for t in np.unique(z['tf']):
            m = z['tf'] == t
            if z['y'][m].sum() >= 8 and z['y'][m].sum() < m.sum():
                acc.setdefault(int(t), []).append(roc_auc_score(z['y'][m], z['logit'][m]))
    return {t: float(np.mean(v)) for t, v in acc.items()}
pop = {}
for fold in range(5):
    tr_, te_ = heldout_split(L, SimpleNamespace(protocol='tf', fold=fold), n)
    deg = tr_[tr_.Label == 1].Target.value_counts()
    s = np.log1p(te_.Target.map(deg).fillna(0).values)
    for t in te_.TF.unique():
        m = (te_.TF == t).values
        if te_.Label.values[m].sum() >= 8 and te_.Label.values[m].sum() < m.sum():
            pop[int(t)] = roc_auc_score(te_.Label.values[m], s[m])
extra = ['\nPAIRED over held-out TFs with >= 8 test positives (per-TF AUROC, averaged over seeds):']
refs = {'popularity (gene degree)': pop}
for c, nm in CFG.items():
    refs[nm] = per_tf_heldout(c)
def cmp(a, b):
    ks = sorted(set(refs[a]) & set(refs[b]))
    d = np.array([refs[a][k] - refs[b][k] for k in ks])
    boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
    return f'  {a} vs {b}: n={len(d)} TFs, mean diff {d.mean():+.4f}, 95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], TFs better {int((d>0).sum())}/{len(d)}, Wilcoxon p={wilcoxon(d).pvalue:.3g}'
for a, b in [('GNN-only hero (SAGE)', 'scRegNet design, tuned'), ('GNN-only hero (SAGE)', 'popularity (gene degree)'),
             ('scRegNet design, tuned', 'popularity (gene degree)'), ('hero + Geneformer', 'popularity (gene degree)'),
             ('GNN-only hero + co-expression number', 'popularity (gene degree)')]:
    if a in refs and b in refs and refs[a] and refs[b]:
        extra.append(cmp(a, b))
txt2 = '\n'.join(extra); print(txt2)
open('analysis/out/heldout_report.txt', 'a').write(txt2)
