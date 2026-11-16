# Within-TF AUROC with gene popularity matched: only compare a positive and a negative
# of the same TF whose genes have the SAME training in-degree bin.
import numpy as np, pandas as pd, glob
from scipy.stats import rankdata, wilcoxon
def strat(tf, tgt, y, s, deg, nb=10):
    b = deg[tgt].astype(int); nb = b.max()+1
    out = {}
    for t in np.unique(tf):
        m = tf == t; conc = tot = 0.0
        for k in np.unique(b[m]):
            mm = m & (b == k); p = y[mm] == 1; n1, n0 = p.sum(), (~p).sum()
            if n1 == 0 or n0 == 0: continue
            r = rankdata(s[mm]); conc += r[p].sum() - n1*(n1+1)/2; tot += n1*n0
        if tot >= 30: out[t] = conc/tot
    return out
def plain(tf, tgt, y, s):
    out = {}
    for t in np.unique(tf):
        m = tf == t; p = y[m] == 1; n1, n0 = p.sum(), (~p).sum()
        if n1 < 3 or n0 < 3: continue
        r = rankdata(s[m]); out[t] = (r[p].sum() - n1*(n1+1)/2)/(n1*n0)
    return out
runs = {c: dict(full_expr=f'm_{c}_g08_had_nosf_batch_d256' if c!='hESC' else 'd01_C_default', random_feat=f'w14_{c}_rand', cell_shuffled=f'w14_{c}_cellperm',
                add_gf0=f'w11_grid_{c}_sage_add_batch_gf0', had_gf0=f'w11_grid_{c}_gcn_had_batch_gf0', add_gf1=f'w11_grid_{c}_gcn_add_batch_gf1', had_nosplit_gf0=f'w11_grid_{c}_sage_had_none_gf0', had_nosplit_gf1=f'w11_grid_{c}_sage_had_none_gf1') for c in ['hESC','hHEP','mESC','mHSC-E']}
for c, R in runs.items():
    tr = pd.read_csv(f'data/{c}/TFs+500/Train_set.csv')
    ng = pd.read_csv(f'data/{c}/TFs+500/Target.csv')['index'].max() + 1
    deg = np.bincount(tr.Target[tr.Label == 1], minlength=ng).astype(float)
    files = {k: sorted(glob.glob(f'analysis/out/pair/{v}_s4[234].npz')) for k, v in R.items()}
    files['dcsbm'] = sorted(glob.glob(f'analysis/out/stat/{c}_dcsbm_learned_s4[234].npz'))
    print(f'\n== {c}')
    for k, fs in files.items():
        if not fs: print(f'  {k:16s} no files'); continue
        P, S = [], []
        for f in fs:
            z = np.load(f); a = (z['tf'], z['tgt'], z['y'], z['logit'])
            P.append(plain(*a)); S.append(strat(*a, deg))
        p = pd.DataFrame(P).mean(); s = pd.DataFrame(S).mean().dropna()
        w = wilcoxon(s - 0.5).pvalue
        print(f'  {k:16s} seeds {len(fs)}  within-TF {p.mean():.3f}   popularity-matched {s.mean():.3f}  ({(s>0.5).sum()}/{len(s)} TFs > 0.5, p={w:.1e})')
    z = np.load(fs[0]); a = (z['tf'], z['tgt'], z['y'], deg[z['tgt']] + 1e-6*np.random.rand(len(z['y'])))
    print(f'  popularity          within-TF {np.mean(list(plain(*a).values())):.3f}   popularity-matched {np.mean(list(strat(*a, deg).values())):.3f}')
