"""W15 part 3: does Geneformer help through biology or through identity? Best grid configuration per cell, seeds 42-44:
no Geneformer (--scfm 0) / real Geneformer (--scfm 1) / Geneformer vectors shuffled across genes (--scfm 2); plus hESC held-out genes (hero SAGE config).
none and real runs are the existing grid runs (found by matching all arguments, as in w11_grid_report.py). Run: python3.12 analysis/w15_gf_report.py"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
from src import pair_gnn as pg
import w3_report as w

SEEDS = (42, 43, 44)
CELLS = ['hESC', 'hHEP', 'mHSC-E', 'mESC']
sys.argv = ['x', '--name', 'x']
DEF = vars(pg.parse())
IGN = {'name', 'seed', 'out', 'data_folder', 'scfm_folder', 'dec_hidden', 'panda_k', 'panda_safe', 'own_edge_diag', 'mp_frac'}


def key(args):
    a = {**DEF, **args}
    ig = set(IGN) | (set() if a['encoder'] == 'npanda' else {'relations', 'coexp'}) | (set() if a['protocol'] != 'random' else {'fold'})
    return tuple(sorted((k, json.dumps(v)) for k, v in a.items() if k not in ig))


BEST = {'hESC': ('npanda', [256, 256], 'batch'), 'hHEP': ('npanda', [256, 256], 'batch'), 'mHSC-E': ('gcn', [256], 'batch'), 'mESC': ('sage', [256], 'none')}


def cfg(cell, gf):
    e, d, s = BEST[cell]
    return dict(cell_type=cell, encoder=e, decoder='had', split=s, scfm=gf, gnn_dims=d, optimizer='Adam', lr=0.0001, wd=0.001, eval_interval=1, refit=1, relations='RCP')


runs = defaultdict(dict)
for f in glob.glob('analysis/out/pair/*.json'):
    nm = f.split('/')[-1][:-5]
    if re.search(r'_s\d+$', nm) and 'refit' in (r := json.load(open(f))) and os.path.exists(f[:-4] + 'npz') and r['args']['seed'] in SEEDS and r['args']['protocol'] == 'random':
        runs[key(r['args'])].setdefault(r['args']['seed'], nm)  # first match wins (names sort grid/w3 before w15)


def load(names):
    R = [json.load(open(f'analysis/out/pair/{n}.json'))['refit'] for _, n in sorted(names.items())]
    T = [w.per_tf(*(lambda z: (z['tf'], z['y'], z['logit']))(np.load(f'analysis/out/pair/{n}.npz'))) for _, n in sorted(names.items())]
    ks = sorted(set.intersection(*map(set, T)))
    return R, {k: float(np.mean([t[k] for t in T])) for k in ks}


def pc(A, B):
    ks = sorted(set(A) & set(B)); d = np.array([A[k] - B[k] for k in ks])
    return d.mean(), int((d > 0).sum()), len(d), (wilcoxon(d).pvalue if np.any(d != 0) else 1.0)


def block(title, V, note=''):
    """V: {'none': (R, T), 'real': ..., 'shuf': ...}"""
    L = [f'\n== {title}{note}', f"{'Geneformer':22s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s}"]
    for k, lab in (('none', 'none (scfm 0)'), ('real', 'real (scfm 1)'), ('shuf', 'shuffled (scfm 2)')):
        if V.get(k) is None: L.append(f'{lab:22s} (missing)'); continue
        R = V[k][0]; L.append(f'{lab:22s} {len(R)}  ' + ' '.join(f"{w.ms([r[m] for r in R]):>16s}" for m in ('auc', 'aupr', 'within')))
    if all(V.get(k) for k in ('none', 'real', 'shuf')):
        mu = {k: {m: np.mean([r[m] for r in V[k][0]]) for m in ('auc', 'aupr', 'within')} for k in V}
        for a_, b_, lab in (('real', 'none', 'real - none'), ('shuf', 'none', 'shuffled - none'), ('shuf', 'real', 'shuffled - real')):
            d, nb, n, p = pc(V[a_][1], V[b_][1]); L.append(f'  within-TF {lab:16s} {d:+.4f} (TFs better {nb}/{n}, p={p:.3g})   AUROC {mu[a_]["auc"] - mu[b_]["auc"]:+.4f}  AUPRC {mu[a_]["aupr"] - mu[b_]["aupr"]:+.4f}')
        sh = {m: (mu['shuf'][m] - mu['none'][m]) / (mu['real'][m] - mu['none'][m]) if abs(mu['real'][m] - mu['none'][m]) > 1e-9 else float('nan') for m in mu['none']}
        L.append(f"  share of the real-Geneformer gain kept by shuffled vectors: AUROC {sh['auc']:.2f}  AUPRC {sh['aupr']:.2f}  within-TF {sh['within']:.2f}")
    return L


L = ['W15 part 3. Geneformer: none / real / shuffled across genes. Each cell\'s best grid configuration (section 24), hero hyper-parameters, seeds 42-44, epoch chosen on validation AUROC, refit on all training rows. Mean ± sd over seeds; p = paired per-TF Wilcoxon (per-TF AUROC averaged over seeds).']
for c in CELLS:
    V = {}
    for k, g in (('none', 0), ('real', 1), ('shuf', 2)):
        nm = runs.get(key(cfg(c, g)), {})
        V[k] = load({s: nm[s] for s in SEEDS}) if all(s in nm for s in SEEDS) else None
        if V[k] is None: print('missing', c, k, nm)
    e, d, s = BEST[c]
    L += block(f'{c}: {e} {d} split {s}, Hadamard', V)

# held-out genes
def ho(fmt):
    R, T = [], {}
    for f in range(5):
        for s in (42, 43):
            n = fmt.format(f=f, s=s)
            if os.path.exists(f'analysis/out/pair/{n}.json') and os.path.exists(f'analysis/out/pair/{n}.npz'):
                R.append(json.load(open(f'analysis/out/pair/{n}.json'))['refit']); z = np.load(f'analysis/out/pair/{n}.npz')
                for k, v in w.per_tf(z['tf'], z['y'], z['logit']).items(): T.setdefault(k, []).append(v)
    return (R, {k: float(np.mean(v)) for k, v in T.items()}) if R else None


V = {'none': ho('w5_ho_gene{f}_hero_s{s}'), 'real': ho('w6_ho_gene{f}_herogf_s{s}'), 'shuf': ho('w15_ho_gene{f}_herogf_shuf_s{s}')}
L += block('hESC held-out genes (hero SAGE 256, Hadamard, batch; 5 folds x seeds 42,43)', V, '; the test genes were never in the training graph')
L.append('(a new gene has no training edges, so the identity role of a vector cannot be learned for it; what remains is what the vector says about the gene)')
# earlier shuffled runs with other configurations, for context only
L.append('\nEarlier shuffled-Geneformer runs (other configurations, hESC, seeds 42-44, no matched real-Geneformer run of the same configuration):')
for nm in ('f12_A_shuffled', 'f13_B_shuffled'):
    r = [json.load(open(f'analysis/out/pair/{nm}_s{s}.json'))['refit'] for s in SEEDS]
    L.append(f"  {nm:16s} AUROC {w.ms([x['auc'] for x in r])}  AUPRC {w.ms([x['aupr'] for x in r])}  within-TF {w.ms([x['within'] for x in r])}")
open('analysis/out/w15_gf.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
