"""Post-hoc ensembles on hESC: average of z-scored test logits, seed by seed (seed s of each member).
Fair-compute control: the hero ensembled with itself at another seed (s and s+1).
Run: python3.12 analysis/w5_ensembles.py [cell]"""
import sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, '.')
from src.pair_gnn import metrics
from sklearn.metrics import roc_auc_score
per_tf = lambda tf, y, s: {int(t): roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if 1 <= y[tf == t].sum() < (tf == t).sum()}

cell = sys.argv[1] if len(sys.argv) > 1 else 'hESC'
O = 'analysis/out/pair/'
S = [42, 43, 44, 45, 46]
z = lambda v: (v - v.mean()) / v.std()
stk = np.load(f'analysis/out/w3_stack_{cell}.npz')
R = 'w3_' if cell == 'hESC' else f'w4_{cell}_'
y, tf = stk['y'], stk['tf']


def logit(c, s):
    if c.startswith('stack:'):
        return stk[c[6:]]
    if c.startswith('scal:'):  # PANDA / contrast number alone (prior from all training positives; test pairs never in it)
        return np.load(f'data/{cell}/TFs+500/scalars/{c[5:]}_v0.0_s0_ev.npy')[tf, stk['tgt']]
    d = np.load(f'{O}{c}_s{s}.npz')
    assert (d['tf'] == tf).all() and (d['y'] == y).all()
    return d['logit']


def ens(members):
    """members: run prefixes or 'stack:<key>'; 'name@k' takes that member at seed offset k (self-ensemble controls)"""
    sp = [(c.split('@')[0], int(c.split('@')[1]) if '@' in c else 0) for c in members]
    return [sum(z(logit(c, S[(i + k) % 5])) for c, k in sp) for i in range(5)]


hero, gf, nph = R + 'hero', R + 'herogf', R + 'np_had'
B = ens([hero])
pb = [per_tf(tf, y, s) for s in B]
base = {k: np.mean([p[k] for p in pb]) for k in pb[0]}
p4 = [per_tf(tf, y, s) for s in ens([hero, hero + '@1', hero + '@2', hero + '@3'])]
base4 = {k: np.mean([p[k] for p in p4]) for k in p4[0]}  # same-compute control for 4-member ensembles
best_stack = 'stack:all_[LR]'
rows = {
    'hero (single model)': [hero],
    'hero x2 seeds  [compute control]': [hero, hero + '@1'],
    'hero x3 seeds  [compute control]': [hero, hero + '@1', hero + '@2'],
    'hero x4 seeds  [compute control]': [hero, hero + '@1', hero + '@2', hero + '@3'],
    'hero + Geneformer hero': [hero, gf],
    'hero + np had': [hero, nph],
    'hero + PANDA score': [hero, 'scal:panda'],
    'hero + contrast score': [hero, 'scal:contrast'],
    'hero + PANDA + contrast scores': [hero, 'scal:panda', 'scal:contrast'],
    'hero + stacker(all, LR)': [hero, best_stack],
    'hero + stacker(label stack, LR)': [hero, 'stack:label_stack_(3-paths_+_RA)_[LR]'],
    'hero + Geneformer hero + np had': [hero, gf, nph],
    'hero + Geneformer hero + stacker': [hero, gf, best_stack],
    'hero + Geneformer hero + np had + stacker': [hero, gf, nph, best_stack],
}
print(f'{cell}: ensembles of z-scored test logits, 5 seeds (member seeds matched); paired per-TF vs single hero')
print(f"{'ensemble':48s} {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s}   diff    TFs  p")
for nm, spec in rows.items():
    try:
        E = ens(spec)
    except FileNotFoundError as e:
        print(f'{nm:48s} missing {e.filename}'); continue
    M = [metrics(y, e, tf) for e in E]
    f = lambda k: f"{np.mean([m[k] for m in M]):.4f} ± {np.std([m[k] for m in M]):.4f}"
    pe = [per_tf(tf, y, e) for e in E]
    d = np.array([np.mean([p[k] for p in pe]) - base[k] for k in base])
    p = wilcoxon(d).pvalue if np.any(d) else 1.0
    d4 = np.array([np.mean([q[k] for q in pe]) - base4[k] for k in base4])
    p4_ = wilcoxon(d4).pvalue if np.any(d4) else 1.0
    print(f"{nm:48s} {f('auc'):>16s} {f('aupr'):>16s} {f('within'):>16s}  {d.mean():+.4f}  {(d > 0).sum():2d}/{len(d)}  {p:.2g}   vs hero x4: {d4.mean():+.4f} ({(d4 > 0).sum()}/{len(d4)}, p={p4_:.2g})")
