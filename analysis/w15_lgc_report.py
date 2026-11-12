"""W15 part 2: LightGCN (free node embedding + parameter-free symmetric-normalised propagation, mean of layers) as a GRN link predictor.
Sweep on hESC at seed 42 (validation AUROC), final runs on four cells at seeds 42-44, held-out genes/TFs on hESC, explanation diagnostics. Run: python3.12 analysis/w15_lgc_report.py"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from src import pair_gnn as pg
import w3_report as w
import w12_common as C

SEEDS = (42, 43, 44)
CELLS = ['hESC', 'hHEP', 'mESC', 'mHSC-E']
OUT = 'analysis/out/pair/'
HERO = {c: ('w3_hero' if c == 'hESC' else f'w4_{c}_hero') for c in CELLS}
sys.argv = ['x', '--name', 'x']
DEF = vars(pg.parse())
IGN = {'name', 'seed', 'out', 'data_folder', 'scfm_folder', 'dec_hidden', 'panda_k', 'panda_safe', 'own_edge_diag', 'mp_frac'}


def key(args):
    a = {**DEF, **args}
    ig = set(IGN) | (set() if a['encoder'] == 'npanda' else {'relations', 'coexp'}) | (set() if a['protocol'] != 'random' else {'fold'})
    return tuple(sorted((k, json.dumps(v)) for k, v in a.items() if k not in ig))


BEST = {'hESC': ('npanda', [256, 256], 'batch'), 'hHEP': ('npanda', [256, 256], 'batch'), 'mHSC-E': ('gcn', [256], 'batch'), 'mESC': ('sage', [256], 'none')}


def best_grid_names(cell):
    e, d, s = BEST[cell]
    tgt = key(dict(cell_type=cell, encoder=e, decoder='had', split=s, scfm=1, gnn_dims=d, optimizer='Adam', lr=0.0001, wd=0.001, eval_interval=1, refit=1, relations='RCP'))
    found = {}
    for f in sorted(glob.glob(OUT + '*.json')):
        nm = f.split('/')[-1][:-5]
        if re.search(r'_s\d+$', nm) and not nm.startswith('w15'):
            r = json.load(open(f))
            if r['args']['seed'] in SEEDS and 'refit' in r and key(r['args']) == tgt: found.setdefault(r['args']['seed'], nm)
    return found


def per_tf_from(paths):
    d = [w.per_tf(*(lambda z: (z['tf'], z['y'], z['logit']))(np.load(p))) for p in paths if os.path.exists(p)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))} if d else {}


fam = lambda prefix: per_tf_from([f'{OUT}{prefix}_s{s}.npz' for s in SEEDS])
mets = lambda prefix: [json.load(open(f'{OUT}{prefix}_s{s}.json')) for s in SEEDS if os.path.exists(f'{OUT}{prefix}_s{s}.json')]


def pc(A, B):
    ks = sorted(set(A) & set(B)); d = np.array([A[k] - B[k] for k in ks])
    return d.mean(), int((d > 0).sum()), len(d), (wilcoxon(d).pvalue if np.any(d != 0) else 1.0)


L = []
# ---------------- 1. sweep
sw = []
for f in glob.glob(OUT + 'w15_sw_*_s42.json') + glob.glob(OUT + 'w15_swdot_*_s42.json'):
    j = json.load(open(f)); a = j['args']
    sw.append(dict(name=j['name'], val=j['val']['auc'], valw=j['val']['within'], K=a['lgc_layers'], d=a['emb_dim'], dec=a['decoder'], l2=a['emb_l2'], lr=a['lr'], feat=a['emb_feat'], std=a['emb_std'], pat=a['patience'], ep=j['best_val_epoch'], refit=j['refit']['auc']))
S = pd.DataFrame(sw).sort_values('val', ascending=False)
main = S[S.name.str.startswith('w15_sw_')]
L += ['W15 part 2: LightGCN on hESC (seed 42 sweep, chosen on validation AUROC; hero hyper-parameters otherwise: Adam, wd 1e-3 on the decoder, batch split, scfm 0).',
      f'\n== Sweep: {len(main)} runs (K x d x decoder x embedding L2 x lr x emb_feat), default init std 0.1, patience 10. Mean validation AUROC by level (all other factors pooled), then the top 10.']
for col in ('K', 'd', 'dec', 'l2', 'lr', 'feat'):
    L.append(f'  {col:5s} ' + '   '.join(f'{v}: {g.val.mean():.4f}' for v, g in main.groupby(col)))
L.append('  best validation AUROC per cell of the grid: ' + '; '.join(f'{dec}/{ft} {g.val.max():.4f}' for (dec, ft), g in main.groupby(['dec', 'feat'])))
L.append(f"{'run':52s} {'val AUROC':>9s} {'val within':>10s} epoch")
for _, r in S.head(10).iterrows(): L.append(f'{r["name"]:52s} {r.val:9.4f} {r.valw:10.4f} {r.ep}')
dx = S[S.name.str.startswith('w15_swdot_')]
L.append(f'\n== Extra dot-decoder sweep (init std 0.5 / 1.0, patience 30; the default init stalls the dot product): {len(dx)} runs, best validation AUROC {dx.val.max():.4f} ({dx.iloc[0]["name"]}); best had run {main[main.dec == "had"].val.max():.4f}')
bad = S[S.refit < S.val - 0.1]; L.append(f'  runs whose refit test AUROC fell more than 0.1 below the validation AUROC (refit instability): ' + (', '.join(f'{r["name"]} (val {r.val:.3f}, test {r.refit:.3f})' for _, r in bad.iterrows()) or 'none'))

# ---------------- 2. cells
VAR = [('pure', 'LightGCN (free embedding only)'), ('hyb', 'hybrid, + projected expression'), ('gf', 'hybrid, + projected Geneformer'), ('stats', 'hybrid, + projected 14 statistics'),
       ('featexpr', 'ablation: expression only (no free emb.)'), ('featgf', 'ablation: Geneformer only (no free emb.)')]
T = {}
for c in CELLS:
    items = [(lab, f'w15_lgc_{c}_{v}') for v, lab in VAR]
    if c == 'hESC': items.append(('LightGCN, dot decoder', 'w15_lgc_hESC_dot'))
    gn = best_grid_names(c)
    items += [('hero (SAGE, expression)', HERO[c]), ('SAGE random features (w14)', f'w14_{c}_rand')]
    L += [f'\n== {c}: seeds 42-44, test, mean ± sd', f"{'model':44s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'val AUROC':>16s}"]
    tf = {}
    for lab, p in items:
        m = mets(p); tf[lab] = fam(p)
        if not m: L.append(f'{lab:44s} (no runs)'); continue
        L.append(f'{lab:44s} {len(m)}  ' + ' '.join(f"{w.ms([x['refit'][q] for x in m]):>16s}" for q in ('auc', 'aupr', 'within')) + f" {w.ms([x['val']['auc'] for x in m]):>16s}")
    if len(gn) == 3:
        R = [json.load(open(f'{OUT}{n}.json')) for _, n in sorted(gn.items())]; lab = f'best grid GNN ({BEST[c][0]}, scfm 1)'
        L.append(f'{lab:44s} {len(R)}  ' + ' '.join(f"{w.ms([x['refit'][q] for x in R]):>16s}" for q in ('auc', 'aupr', 'within')) + f" {w.ms([x['val']['auc'] for x in R]):>16s}")
        tf[lab] = per_tf_from([f'{OUT}{n}.npz' for _, n in sorted(gn.items())])
    ds = [f'analysis/out/stat/{c}_dcsbm_learned_s{s}.npz' for s in SEEDS]
    tf['DC-SBM learned'] = per_tf_from(ds)
    wd = [np.mean(list(w.per_tf(z['tf'], z['y'], z['logit']).values())) for z in map(np.load, ds)]
    L.append(f"{'DC-SBM learned (section 31)':44s} 3  {'':>16s} {'':>16s} {w.ms(wd):>16s}")
    T[c] = tf
    L.append('paired per-TF within-TF tests (diff, TFs better, Wilcoxon p):')
    refs = [k for k in tf if k.startswith(('hero', 'SAGE random', 'best grid', 'DC-SBM'))]
    for lab in ('LightGCN (free embedding only)', 'hybrid, + projected expression', 'hybrid, + projected Geneformer', 'hybrid, + projected 14 statistics'):
        if not tf.get(lab): continue
        L.append(f'  {lab}:')
        for r in refs + ([l for l in ('LightGCN (free embedding only)',) if l != lab] if lab != 'LightGCN (free embedding only)' else []):
            if tf.get(r):
                d, nb, n, p = pc(tf[lab], tf[r]); L.append(f'      vs {r:36s} {d:+.4f} ({nb}/{n}, p={p:.3g})')

# ---------------- 3. held-out
def ho(fmt, prot):
    R, Tt = [], {}
    for f in range(5):
        for s in (42, 43):
            n = fmt.format(prot=prot, f=f, s=s)
            if os.path.exists(f'{OUT}{n}.json') and os.path.exists(f'{OUT}{n}.npz'):
                R.append(json.load(open(f'{OUT}{n}.json'))['refit']); z = np.load(f'{OUT}{n}.npz')
                for k, v in w.per_tf(z['tf'], z['y'], z['logit']).items(): Tt.setdefault(k, []).append(v)
    return R, {k: float(np.mean(v)) for k, v in Tt.items()}


for prot in ('gene', 'tf'):
    L += [f'\n== hESC held-out {prot}s: 5 folds x seeds 42,43 (n = 10)', f"{'model':40s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within>=8':>16s}   vs hero (w5): within-TF diff (TFs better, p)"]
    base = ho('w5_ho_{prot}{f}_hero_s{s}', prot)
    rows = [('hero (w5)', base)] + [(lab, ho('w15_ho_{prot}{f}_lgc' + v + '_s{s}', prot)) for v, lab in VAR[:4]]
    if prot == 'gene': rows.insert(1, ('SAGE random features (w14)', ho('w14_ho_gene{f}_rand_s{s}', prot)))
    for lab, (R, Tt) in rows:
        if not R: L.append(f'{lab:40s} (no runs)'); continue
        s = f'{lab:40s} {len(R)}  ' + ' '.join(f"{w.ms([r[q] for r in R]):>16s}" for q in ('auc', 'aupr', 'within', 'within8'))
        if lab != 'hero (w5)' and base[1]:
            d, nb, n, p = pc(Tt, base[1]); s += f'   {d:+.4f} ({nb}/{n}, p={p:.3g})'
        L.append(s)

# ---------------- 4. explanation (hESC)
c = 'hESC'; tf = T[c]
L += ['\n== Explanation (hESC)']
tr = pd.read_csv(f'data/{c}/TFs+500/Train_set.csv', index_col=0)
ntr = tr[tr.Label == 1].groupby('TF').size()
pop = C.popularity(c)
for lab, ref in (('LightGCN (free embedding only)', 'hero (SAGE, expression)'), ('LightGCN (free embedding only)', 'SAGE random features (w14)'), ('hybrid, + projected expression', 'hero (SAGE, expression)')):
    if tf.get(lab) and tf.get(ref):
        ks = sorted(set(tf[lab]) & set(tf[ref])); d = np.array([tf[lab][k] - tf[ref][k] for k in ks]); nt = np.array([ntr.get(k, 0) for k in ks])
        rho = spearmanr(d, nt)
        small = nt <= np.median(nt)
        L.append(f'  per-TF gain {lab} - {ref}: Spearman with number of training targets rho={rho.statistic:+.2f} (p={rho.pvalue:.2g}); TFs with <= median targets ({int(np.median(nt))}): {d[small].mean():+.4f}, with more: {d[~small].mean():+.4f}')
L.append('  Spearman of the test scores with gene popularity (training in-degree), within TF, mean over TFs and seeds:')
for lab, p in [(l, f'w15_lgc_{c}_{v}') for v, l in VAR[:6]] + [('hero', HERO[c]), ('SAGE random features', f'w14_{c}_rand')]:
    out = []
    for s in SEEDS:
        fn = f'{OUT}{p}_s{s}.npz'
        if os.path.exists(fn):
            z = np.load(fn); out.append(np.nanmean([spearmanr(z['logit'][z['tf'] == t], pop[z['tgt'][z['tf'] == t]]).statistic for t in np.unique(z['tf']) if (z['tf'] == t).sum() > 2]))
    L.append(f'      {lab:44s} {np.mean(out):.3f}' if out else f'      {lab:44s} (no runs)')
L += ['\n== Diagnostic (mESC, seed 42; pure LightGCN trails the hero there): wider embedding, one layer, no decoder weight decay. Not used for any choice.']
for f in sorted(glob.glob(OUT + 'w15_diag_mESC_*_s42.json')) + [OUT + 'w15_lgc_mESC_pure_s42.json']:
    j = json.load(open(f)); a = j['args']
    L.append(f"  {j['name']:40s} K={a['lgc_layers']} d={a['emb_dim']} wd={a['wd']}  val AUROC {j['val']['auc']:.4f}  test within-TF {j['refit']['within']:.4f}  AUROC {j['refit']['auc']:.4f}  epoch {j['best_val_epoch']}")
# own-edge check on the pure model is covered by the batch split (same removal as SAGE); no extra diagnostic needed
open('analysis/out/w15_lgc.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
