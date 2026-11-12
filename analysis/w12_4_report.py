"""W12 idea 4 (Correct & Smooth on the pair graph): reads analysis/out/w12_cs_<family>.json made by analysis/w12_cs.py (run that first).
Run: python3.12 analysis/w12_4_report.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, 'analysis')
import w12_common as C
import w3_report as w
from sklearn.metrics import roc_auc_score, average_precision_score

L = ['W12 idea 4: Correct & Smooth on a graph of (TF, gene) pairs. hESC, seeds 42-44. Hyperparameters chosen per seed on the validation slice (stage-1 model, training labels = 90% of training rows); applied to the refit model with all training labels; test scored once.']
for fam, ref, nm in (('hero', 'w3_hero', 'hero'), ('herogf', 'w3_herogf', 'hero + Geneformer')):
    f = f'analysis/out/w12_cs_{fam}.json'
    if not os.path.exists(f): L.append(f'\n{nm}: {f} missing'); continue
    O = json.load(open(f))
    if not O: continue
    z = [np.load(f'{C.OUT}{o["name"]}.npz') for o in O]
    ms = lambda v: w.ms(v)
    L += [f'\n== {nm} (family w12_cs_{fam}, {len(O)} seeds)',
          f'{"":34s} {"AUROC":>16s} {"AUPRC":>16s} {"within-TF":>16s}']
    for lab, key in (('refit model, raw (rerun of the hero)', 'test_raw'), ('+ C&S, chosen on val AUROC', 'test_cs'), ('+ C&S, chosen on mean(val AUROC, val within-TF)', 'test_cs_mix')):
        L.append(f'{lab:34s} ' + ' '.join(f'{ms([o[key][m] for o in O]):>16s}' for m in ('auc', 'aupr', 'within')))
    L.append(f'{"original run " + ref:34s} ' + ' '.join(f'{ms([r["refit"][m] for r in C.runs(ref)]):>16s}' for m in ('auc', 'aupr', 'within')))
    L.append('chosen hyperparameters (val AUROC base -> chosen): ' + '; '.join(f'{o["name"][-3:]}: {o["chosen"]["graph"]} k={o["chosen"]["k"]} {o["chosen"]["edge"]} a1={o["chosen"]["a1"]} s={o["chosen"]["s"]} a2={o["chosen"]["a2"]} ({o["val_baseline"]["auc"]:.4f} -> {o["chosen"]["auc"]:.4f})' for o in O))
    # paired per-TF tests: C&S scores vs the same model's raw scores, per-TF AUROC averaged over seeds
    pt = lambda key, which: [w.per_tf(zz['tf'], zz['y'], np.array(o[key])) for zz, o in zip(z, O)]
    avg = lambda d: {k: float(np.mean([x[k] for x in d])) for k in sorted(set.intersection(*map(set, d)))}
    cs, raw = avg(pt('_s_cs', 0)), avg(pt('_s_raw', 0))
    L.append(w.paired(cs, raw, 'C&S - same model without C&S'))
    L.append(w.paired(cs, C.tf_mean(ref), f'C&S - original {ref}'))
    L.append('where the effect comes from (test, same hyperparameters, one part off; mean over seeds; the full method gives within-TF %.4f, AUROC %.4f):' % (np.mean([o['test_cs']['within'] for o in O]), np.mean([o['test_cs']['auc'] for o in O])))
    for k in ('correct_only', 'smooth_only', 'gene_edges_only', 'tf_edges_only'):
        v = [o['parts'][k] for o in O if k in o['parts']]
        if v: L.append(f'  {k:18s} (n={len(v)}) within-TF {np.mean([x["within"] for x in v]):.4f}  AUROC {np.mean([x["auc"] for x in v]):.4f}')
    L.append(f'score/popularity Spearman over all genes (mean over TFs, seeds): raw {np.mean([o["pop_corr_raw"] for o in O]):+.3f}, C&S {np.mean([o["pop_corr_cs"] for o in O]):+.3f}')
open('analysis/out/w12_4_cs.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
