"""Aggregate the 5-seed final runs (analysis/jobs/finals.txt) and compare models pair-by-pair across TFs.
Output: analysis/out/final_report.txt (also printed)."""
import glob, json, re, sys
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

OUT = 'analysis/out/pair/'
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'


def runs(cfg):
    return [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}{cfg}_s*.json')) if re.search(rf'{cfg}_s\d+\.json$', f)]


def per_tf(z, min_pos=1):
    out = {}
    for t in np.unique(z['tf']):
        m = z['tf'] == t
        y = z['y'][m]
        if min_pos <= y.sum() < m.sum():
            out[int(t)] = roc_auc_score(y, z['logit'][m])
    return out


def per_tf_mean(cfg):
    zs = [np.load(f) for f in sorted(glob.glob(f'{OUT}{cfg}_s*.npz'))]
    d = [per_tf(z) for z in zs]
    ks = sorted(set.intersection(*[set(x) for x in d])) if d else []
    return {k: float(np.mean([x[k] for x in d])) for k in ks}


def paired(a, b, label):
    A, B = per_tf_mean(a), per_tf_mean(b)
    ks = sorted(set(A) & set(B))
    if not ks:
        return f'  {label}: no overlap'
    d = np.array([A[k] - B[k] for k in ks])
    rng = np.random.RandomState(0)
    boot = d[rng.randint(0, len(d), (5000, len(d)))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return (f'  {label:44s} within-TF diff {d.mean():+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  '
            f'TFs better {int((d > 0).sum())}/{len(d)}  Wilcoxon p={p:.3g}')


names = {
    'f01_A_tuned': 'scRegNet design, tuned (Adam)', 'f02_A_paper': 'scRegNet design, original SGD recipe',
    'f03_B_tuned': 'SAGE + interacting + split + Geneformer', 'f04_C_tuned': 'SAGE + interacting + split (GNN only)',
    'f05_C_contrast': '  + co-expression contrast number', 'f06_C_panda': '  + PANDA number',
    'f07_C_gcn': '  ablation: GCN encoder', 'f08_C_add': '  ablation: additive final layer',
    'f09_C_nosplit': '  ablation: no edge split', 'f10_C_mlp': '  ablation: hidden-layer final layer',
    'f11_B_panda': 'SAGE + interacting + split + Geneformer + PANDA',
    'd01_C_default': 'hero, hand-set default settings (GNN only)', 'd02_B_default': 'hero + Geneformer, hand-set default settings', 'td01_hero_tfdrop0.3': 'hero (default) + whole-TF edge dropout 0.3',
    'f12_A_shuffled': 'control: scRegNet design, Geneformer vectors shuffled across genes',
    'f13_B_shuffled': 'control: hero + Geneformer, vectors shuffled across genes',
}
lines = ['hESC, refit protocol, mean ± std over seeds (test set scored once)',
         f"{'model':52s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within-TF(>=8 pos)':>18s}"]
for k, nm in names.items():
    r = runs(k)
    if not r:
        continue
    g = lambda m: [x['refit'][m] for x in r]
    lines.append(f"{nm:52s} {len(r)}  {ms(g('auc')):>16s} {ms(g('aupr')):>16s} {ms(g('within')):>16s} {ms(g('within8')):>18s}")
lines += ['reference rows:',
          '  original scRegNet (epoch chosen on test; 1 seed)             0.8870           0.6245           0.7215',
          '  last week scRegNet + PANDA, symmetric fix (3 seeds)          0.8915 ± 0.0004 0.6337 ± 0.0009 0.7471 ± 0.0009']

# stacker & original references for the paired tests
S = np.load('analysis/out/cs_scores_hESC.npz')
stack = per_tf(dict(tf=S['tf'], y=S['y'], logit=S['+_contrast_+_PANDA_[LR]']))
orig = np.load('analysis/out/scores_tf_500_hESC_baseline_rerun.npz')
tp = orig['test_pairs']
orig_tf = per_tf(dict(tf=tp[:, 0], y=tp[:, 2], logit=orig['test_logits'][:, 1] - orig['test_logits'][:, 0]))
lines.append('stacker without any neural network (popularity + CF + 3-paths + co-expression contrast + PANDA, logistic regression):')
st = json.load(open('analysis/out/cs_baselines_hESC.json'))['+ contrast + PANDA [LR]']
lines.append(f"  AUROC {st['auc']:.4f}  AUPRC {st['aupr']:.4f}  within-TF {st['within']:.4f}")

lines.append('\npaired comparisons across the test TFs (per-TF AUROC averaged over seeds):')
for a, b, lab in [('f04_C_tuned', 'f01_A_tuned', 'GNN-only hero vs tuned scRegNet design'),
                  ('f04_C_tuned', 'f02_A_paper', 'GNN-only hero vs scRegNet original recipe'),
                  ('f03_B_tuned', 'f01_A_tuned', 'hero + Geneformer vs tuned scRegNet design'),
                  ('f04_C_tuned', 'f08_C_add', 'interacting vs additive final layer'),
                  ('f04_C_tuned', 'f09_C_nosplit', 'edge split vs none'),
                  ('f04_C_tuned', 'f07_C_gcn', 'SAGE vs GCN'),
                  ('f04_C_tuned', 'f10_C_mlp', 'multiplication vs hidden-layer final layer'),
                  ('f05_C_contrast', 'f04_C_tuned', 'adding co-expression contrast to hero'),
                  ('f06_C_panda', 'f04_C_tuned', 'adding PANDA to hero'),
                  ('f03_B_tuned', 'f04_C_tuned', 'Geneformer on top of hero'),
                  ('f11_B_panda', 'f03_B_tuned', 'adding PANDA to hero + Geneformer'),
                  ('d01_C_default', 'f02_A_paper', 'default hero vs published scRegNet recipe'),
                  ('d02_B_default', 'f02_A_paper', 'default hero + Geneformer vs published recipe'),
                  ('d02_B_default', 'd01_C_default', 'Geneformer on top of default hero'),
                  ('d01_C_default', 'f04_C_tuned', 'default hero vs validation-tuned hero')]:
    lines.append(paired(a, b, lab))
for a, ref, lab in [('f04_C_tuned', stack, 'GNN-only hero vs no-neural-network stacker'),
                    ('d01_C_default', stack, 'default hero vs no-neural-network stacker'),
                    ('d02_B_default', stack, 'default hero + Geneformer vs stacker'),
                    ('d01_C_default', orig_tf, 'default hero vs original scRegNet checkpoint'),
                    ('d02_B_default', orig_tf, 'default hero + Geneformer vs original checkpoint'),
                    ('f04_C_tuned', orig_tf, 'GNN-only hero vs original scRegNet checkpoint'),
                    ('f01_A_tuned', orig_tf, 'tuned scRegNet design vs original checkpoint')]:
    A = per_tf_mean(a)
    ks = sorted(set(A) & set(ref))
    d = np.array([A[k] - ref[k] for k in ks])
    boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
    lines.append(f'  {lab:44s} within-TF diff {d.mean():+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  '
                 f'TFs better {int((d > 0).sum())}/{len(d)}  Wilcoxon p={wilcoxon(d).pvalue:.3g}')
from sklearn.metrics import average_precision_score
def metr(y, s, t):
    w = [roc_auc_score(y[t == u], s[t == u]) for u in np.unique(t) if 0 < y[t == u].sum() < (t == u).sum()]
    return roc_auc_score(y, s), average_precision_score(y, s), float(np.mean(w))
zs = lambda v: (v - v.mean()) / v.std()
lines.append('\nensembles (no fitting; logits z-scored then averaged):')
for cfg in ('f04_C_tuned', 'f03_B_tuned', 'd01_C_default', 'd02_B_default'):
    Z = [np.load(f) for f in sorted(glob.glob(f'{OUT}{cfg}_s*.npz'))]
    if not Z: continue
    ens = np.mean([zs(z['logit']) for z in Z], 0)
    a, b, c = metr(Z[0]['y'], ens, Z[0]['tf'])
    lines.append(f'  {cfg} {len(Z)}-seed ensemble: AUROC {a:.4f} AUPRC {b:.4f} within-TF {c:.4f}')
    if cfg == 'f04_C_tuned' and np.array_equal(Z[0]['tgt'], S['tgt']) and np.array_equal(Z[0]['tf'], S['tf']):
        mix = (ens + zs(S['+_contrast_+_PANDA_[LR]'])) / 2
        a, b, c = metr(Z[0]['y'], mix, Z[0]['tf'])
        lines.append(f'  {cfg} ensemble averaged with the no-neural-network stacker: AUROC {a:.4f} AUPRC {b:.4f} within-TF {c:.4f}')
txt = '\n'.join(lines)
open('analysis/out/final_report.txt', 'w').write(txt)
print(txt)
