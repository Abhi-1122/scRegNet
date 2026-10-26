"""Week 3 report (hESC): mean ± std over seeds, paired per-TF tests against the GPU hero (per-TF AUROC averaged over seeds,
Wilcoxon over the test TFs; same test as tune2.py report). Run: python3.12 analysis/w3_report.py"""
import glob, json, re
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

OUT = 'analysis/out/pair/'
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'
runs = lambda c: [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}{c}_s*.json')) if re.search(rf'{c}_s\d+\.json$', f)]
npzs = lambda c: [np.load(f) for f in sorted(glob.glob(f'{OUT}{c}_s*.npz')) if re.search(rf'{c}_s\d+\.npz$', f)]


def per_tf(tf, y, s):
    return {int(t): roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf) if 1 <= y[tf == t].sum() < (tf == t).sum()}


def tf_mean(c):
    d = [per_tf(z['tf'], z['y'], z['logit']) for z in npzs(c)]
    return {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))} if d else {}


def paired(A, B, label):
    ks = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in ks])
    boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return f'  {label:46s} within-TF diff {d.mean():+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  TFs better {int((d > 0).sum())}/{len(d)}  p={p:.3g}'


HDR = f"{'model':46s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within>=8':>16s} {'val AUROC':>16s} {'min/run':>7s}"


def row(nm, c):
    r = runs(c)
    if not r: return f'{nm:46s} (no runs)'
    g = lambda m: ms([x['refit'][m] for x in r])
    return f"{nm:46s} {len(r)}  {g('auc'):>16s} {g('aupr'):>16s} {g('within'):>16s} {g('within8'):>16s} {ms([x['val']['auc'] for x in r]):>16s} {np.mean([x['minutes'] for x in r]):7.2f}"


def table(title, items, ref, ref_name='hero'):
    L = [f'\n== {title}', HDR]
    L += [row(nm, c) for nm, c in items]
    L.append(f'paired per-TF tests vs {ref_name} ({ref}):')
    B = tf_mean(ref)
    L += [paired(tf_mean(c), B, nm) if npzs(c) else f'  {nm}: no runs' for nm, c in items if c != ref]
    L.append(f'same-seed differences vs {ref_name} (mean over seeds, seeds better of 5):')
    R = {r['args']['seed']: r['refit'] for r in runs(ref)}
    for nm, c in items:
        if c != ref and runs(c):
            d = {m: [r['refit'][m] - R[r['args']['seed']][m] for r in runs(c) if r['args']['seed'] in R] for m in ('auc', 'aupr')}
            L.append(f"  {nm:46s} AUROC {np.mean(d['auc']):+.4f} ({sum(x > 0 for x in d['auc'])}/{len(d['auc'])})  AUPRC {np.mean(d['aupr']):+.4f} ({sum(x > 0 for x in d['aupr'])}/{len(d['aupr'])})")
    return L


def main():
    L = ['Week 3, hESC. All pair_gnn rows: GPU, 5 seeds (42-46), hero optimiser settings, epoch chosen on validation AUROC, refit on all training rows.']
    L += table('Reference', [('hero (GNN, GRN graph, no scFM)', 'w3_hero'), ('hero + Geneformer', 'w3_herogf')], 'w3_hero')
    L += table('Part 1: PANDA top-K as the message-passing graph (safe = fold-pure batches)',
               [('hero (GRN graph)', 'w3_hero'), ('PANDA graph K=50 safe', 'w3_pg_k50'), ('PANDA graph K=100 safe', 'w3_pg_k100'), ('PANDA graph K=200 safe', 'w3_pg_k200'),
                ('GRN + PANDA K=100 safe (union)', 'w3_pgU_k100'), ('PANDA graph K=100 NAIVE (leaks)', 'w3_pg_k100_naive'), ('GRN + PANDA K=100 NAIVE (union)', 'w3_pgU_k100_naive'),
                ('PANDA graph K=200 safe + Geneformer', 'w3_pg_k200_gf'), ('GRN + PANDA K=100 safe + Geneformer', 'w3_pgU_k100_gf')], 'w3_hero')
    L.append('\nleak measurement, NAIVE minus SAFE (same K, same seeds):')
    for a_, b_ in (('w3_pg_k100_naive', 'w3_pg_k100'), ('w3_pgU_k100_naive', 'w3_pgU_k100')):
        L.append(paired(tf_mean(a_), tf_mean(b_), a_ + ' - safe'))
        L.append('  epochs chosen: naive %s, safe %s' % ([r['best_val_epoch'] for r in runs(a_)], [r['best_val_epoch'] for r in runs(b_)]))
    L += table('Part 2: neural PANDA encoder (R known targets, C co-expression kNN, P STRING PPI; 2 layers, Tanimoto decoder unless stated)',
               [('hero (sage, had)', 'w3_hero'), ('hero with Tanimoto decoder', 'w3_hero_tani'), ('np full RCP', 'w3_np_full'), ('np without C (RP)', 'w3_np_noC'),
                ('np without P (RC)', 'w3_np_noP'), ('np without R (CP)', 'w3_np_noR'), ('np full, had decoder', 'w3_np_had'), ('np full, 1 layer', 'w3_np_1L'),
                ('np full, metacell C', 'w3_np_meta'), ('np full, tani_pure decoder', 'w3_np_pure'), ('np full + Geneformer', 'w3_np_gf')], 'w3_hero')
    L += table('Part 2b (added): where does the np had-decoder gain come from? (had decoder everywhere)',
               [('hero (1 layer sage)', 'w3_hero'), ('hero, 2 layers sage', 'w3_hero_2L'), ('np R only', 'w3_nph_R'), ('np RC', 'w3_nph_RC'), ('np RP', 'w3_nph_RP'),
                ('np CP', 'w3_nph_CP'), ('np RCP', 'w3_np_had')], 'w3_hero')
    L.append('\nnp full + Geneformer vs hero + Geneformer:')
    L.append(paired(tf_mean('w3_np_gf'), tf_mean('w3_herogf'), 'np_gf - herogf'))
    for c in ('w3_pg_k200_gf', 'w3_pgU_k100_gf'): L.append(paired(tf_mean(c), tf_mean('w3_herogf'), c + ' - herogf'))

    S = json.load(open('analysis/out/w3_stack_hESC.json')); Z = np.load('analysis/out/w3_stack_hESC.npz')
    B = tf_mean('w3_hero'); base = 'label_+_contrast_+_PANDA_score_(best_in_cs_baselines)_[GBM]'
    L += ['\n== Part 3: context-specific stacker (single run; test rows use full PANDA, training rows their own fold PANDA)',
          f"{'stack':72s} {'AUROC':>7s} {'AUPRC':>7s} {'within':>7s} {'vs best cs_baselines (p)':>26s} {'vs hero (p)':>14s}"]
    bt = per_tf(Z['tf'], Z['y'], Z[base])
    for k, v in S.items():
        if k.startswith('single') or k.replace(' ', '_') not in Z.files: continue
        t = per_tf(Z['tf'], Z['y'], Z[k.replace(' ', '_')])
        pv = lambda X, Y: wilcoxon(d).pvalue if np.any(d := [X[q] - Y[q] for q in X if q in Y]) else 1.0
        L.append(f"{k:72s} {v['auc']:7.4f} {v['aupr']:7.4f} {v['within']:7.4f} {np.mean([t[q] - bt[q] for q in t]):+.4f} (p={pv(t, bt):.2g}) {np.mean([t[q] - B[q] for q in t if q in B]):+.4f} (p={pv(t, B):.2g})")
    L.append('hero within-TF (mean over seeds, same TFs) for reference: %.4f' % np.mean(list(B.values())))
    L.append('\nsingle features (raw test AUROC / within-TF):')
    L += [f"  {k[7:]:12s} {v['auc']:.4f} / {v['within']:.4f}" for k, v in S.items() if k.startswith('single')]
    L.append('\n== Graph diagnostics and leak\n' + open('analysis/out/w3_graph_recall_hESC.txt').read())
    open('analysis/out/w3_report_hESC.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
