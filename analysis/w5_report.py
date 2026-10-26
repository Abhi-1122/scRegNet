"""Round-1 new ideas on hESC (GPU, refit, epoch by validation AUROC): within-TF ranking loss, PANDA teacher
head, neural PANDA (had) + Geneformer, and neural PANDA on held-out TFs / genes.
Run: python3.12 analysis/w5_report.py"""
import glob, json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

O = 'analysis/out/pair/'
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'


def per_tf(z):
    tf, y, s = z['tf'], z['y'], z['logit']
    return {int(t): roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if 1 <= y[tf == t].sum() < (tf == t).sum()}


def pooled(pat):
    """per-TF AUROC averaged over every run matching pat (seeds, and folds for held-out protocols)"""
    acc = {}
    for f in sorted(glob.glob(O + pat + '.npz')):
        for k, v in per_tf(np.load(f)).items():
            acc.setdefault(k, []).append(v)
    return {k: np.mean(v) for k, v in acc.items()}


def paired(A, B):
    d = np.array([A[k] - B[k] for k in A if k in B])
    return f'within-TF diff {d.mean():+.4f}  TFs better {(d > 0).sum()}/{len(d)}  p={wilcoxon(d).pvalue:.2g}'


def row(nm, pat):
    r = [json.load(open(f))['refit'] for f in sorted(glob.glob(O + pat + '.json'))]
    return f"{nm:34s} {len(r):2d}  {ms([x['auc'] for x in r]):>16s} {ms([x['aupr'] for x in r]):>16s} {ms([x['within'] for x in r]):>16s} {ms([x['within8'] for x in r]):>16s}"


H = f"{'model':34s} {'n':>2s}  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within>=8':>16s}"
print('== Benchmark split, 5 seeds; paired per-TF vs the GPU hero (w3_hero) or hero + Geneformer (w3_herogf)')
print(H)
R = [('hero', 'w3_hero_s*', None), ('hero + Geneformer', 'w3_herogf_s*', None)]
R += [(f'hero + rank loss w={w}', f'w5_rank{w}_s*', 'w3_hero_s*') for w in ('0.5', '1', '2')]
R += [(f'hero + PANDA teacher w={w}', f'w5_aux{w}_s*', 'w3_hero_s*') for w in ('0.1', '0.3', '1')]
R += [(f'hero + {k} number, lr x{L}', f'w5_sc_{k}_lr{L}_s*', 'w3_hero_s*') for k in ('panda', 'contrast', 'both') for L in ('1', '100', '1000') if (k, L) != ('contrast', '1') and (k, L) != ('both', '1')]
R += [('hero+GF + both numbers, lr x100', 'w5_sc_both_lr100_gf_s*', 'w3_herogf_s*')]
R += [(f'hero + hard neg {ng} {ls} w={w}', f'w6_hn_{ng}_{ls}_w{w}_s*', 'w3_hero_s*') for ng in ('pop', 'model') for ls in ('bpr', 'bce') for w in ('0.1', '0.3')]
R += [('hero + hard neg model bpr w=1', 'w6_hn_model_bpr_w1_s*', 'w3_hero_s*')]
R += [('np had (RCP)', 'w3_np_had_s*', 'w3_hero_s*'), ('np had + Geneformer', 'w5_nphad_gf_s*', 'w3_herogf_s*')]
for nm, pat, _ in R:
    print(row(nm, pat))
for nm, pat, ref in R:
    if ref:
        print(f'  {nm:32s} vs {ref[:-3]:10s} {paired(pooled(pat), pooled(ref))}')

for pr, title in [('tf', 'HELD-OUT TFs (5 folds x 2 seeds; per-TF AUROC pooled over folds)'), ('gene', 'HELD-OUT GENES (5 folds x 5 seeds)')]:
    print(f'\n== {title}')
    print(H)
    ms_ = [m for m in ('w5:hero', 'w5:nphad', 'w5:npRC', 'w6:npRP', 'w6:herogf', 'w6:nphadgf') if glob.glob(f"{O}{m[:2]}_ho_{pr}[0-9]_{m[3:]}_s*.json")]
    pt = lambda m: f"{m[:2]}_ho_{pr}[0-9]_{m[3:]}_s*"
    for m in ms_:
        print(row(m[3:], pt(m)))
    for m in ms_[1:]:
        ref = 'w6:herogf' if m == 'w6:nphadgf' else 'w5:hero'
        print(f"  {m[3:]} vs {ref[3:]}: {paired(pooled(pt(m)), pooled(pt(ref)))}")
