"""Which ingredient carries the late-fusion gain? Diagnostic only (weights scanned on the test set, so the
best-w numbers are optimistic; they say WHICH information complements the hero, not a final score).
score = z(hero) + w * z(ingredient), seed-matched, 5 seeds; within-TF vs the hero x2-seed blend (same cost).
Run: python3.12 analysis/w7_attribution.py"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

O = 'analysis/out/pair/'
S = [42, 43, 44, 45, 46]
W = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
z = lambda v: (v - v.mean()) / v.std()


def per_tf(tf, y, s):
    return {t: roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if 1 <= y[tf == t].sum() < (tf == t).sum()}


for cell in ['hESC', 'hHEP', 'mESC', 'mHSC-E']:
    R = 'w3_' if cell == 'hESC' else f'w4_{cell}_'
    stk = np.load(f'analysis/out/w3_stack_{cell}.npz')
    tf, y, tg = stk['tf'], stk['y'], stk['tgt']
    run = lambda c, s: np.load(f'{O}{R}{c}_s{s}.npz')['logit']
    sc = lambda k: np.load(f'data/{cell}/TFs+500/scalars/{k}_v0.0_s0_ev.npy')[tf, tg]
    ing = {'Geneformer hero': lambda s: run('herogf', s), 'neural PANDA': lambda s: run('np_had', s),
           'stacker (all)': lambda s: stk['all_[LR]'], 'stacker (label graph only)': lambda s: stk['label_stack_(3-paths_+_RA)_[LR]'],
           'PANDA score': lambda s: sc('panda'), 'contrast score': lambda s: sc('contrast')}
    mean_tf = lambda scores: {t: np.mean([p[t] for p in scores]) for t in scores[0]}
    ctrl = mean_tf([per_tf(tf, y, z(run('hero', s)) + z(run('hero', S[(i + 1) % 5]))) for i, s in enumerate(S)])
    print(f'\n{cell}: hero x2 seeds within-TF {np.mean(list(ctrl.values())):.4f}')
    for nm, f in ing.items():
        best = None
        for w in W:
            m = mean_tf([per_tf(tf, y, z(run('hero', s)) + w * z(f(s))) for s in S])
            v = np.mean(list(m.values()))
            if best is None or v > best[1]:
                best = (w, v, m)
        w, v, m = best
        d = np.array([m[t] - ctrl[t] for t in ctrl])
        print(f'  hero + w*{nm:28s} best w={w:<4} within-TF {v:.4f}  vs hero x2 {d.mean():+.4f} ({(d > 0).sum()}/{len(d)}, p={wilcoxon(d).pvalue:.2g})')

# Is the blend better than the best single model (hero + Geneformer), at the same cost?
print('\nSingle hero+Geneformer, its 4-seed average, and the 4-model blend (within-TF; paired vs Geneformer hero x4)')
for cell in ['hESC', 'hHEP', 'mESC', 'mHSC-E']:
    R = 'w3_' if cell == 'hESC' else f'w4_{cell}_'
    stk = np.load(f'analysis/out/w3_stack_{cell}.npz')
    tf, y = stk['tf'], stk['y']
    run = lambda c, s: np.load(f'{O}{R}{c}_s{s}.npz')['logit']
    mean_tf = lambda scores: {t: np.mean([p[t] for p in scores]) for t in scores[0]}
    single = mean_tf([per_tf(tf, y, run('herogf', s)) for s in S])
    gf4 = mean_tf([per_tf(tf, y, sum(z(run('herogf', S[(i + k) % 5])) for k in range(4))) for i in range(5)])
    blend = mean_tf([per_tf(tf, y, z(run('hero', s)) + z(run('herogf', s)) + z(run('np_had', s)) + z(stk['all_[LR]'])) for s in S])
    d = np.array([blend[t] - gf4[t] for t in gf4])
    print(f"  {cell:7s} GF hero {np.mean(list(single.values())):.4f}   GF hero x4 {np.mean(list(gf4.values())):.4f}   blend {np.mean(list(blend.values())):.4f}   blend - GF x4 {d.mean():+.4f} ({(d > 0).sum()}/{len(d)}, p={wilcoxon(d).pvalue:.2g})")
