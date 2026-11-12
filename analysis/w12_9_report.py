"""W12 idea 9 (positional encodings): screen table (seed 42), 3-seed confirmations, leak check (fold-safe vs full-graph GRN PE),
random-PE ablation, and where the gain sits (strata by distance of the gene to the TF's known targets and by gene popularity).
Run: python3.12 analysis/w12_9_report.py"""
import glob, re, sys
import numpy as np
sys.path.insert(0, 'analysis')
import w12_common as C
import w3_report as w

REF = 'w3_hero'
fams = sorted({re.sub(r'_s\d+\.json$', '', f.split('/')[-1]) for f in glob.glob(f'{C.OUT}w12_pe*_s*.json')})
nm = lambda c: c.replace('w12_', '')
L = ['W12 idea 9: positional encodings. hESC, random split, no Geneformer, hero settings, epoch chosen on validation AUROC, refit.']
L += C.table('Screen, seed 42 only (n=1 per row; hero noise between identical GPU reruns is about 0.001 AUROC, 0.001 within-TF)', [('hero (w3_hero)', REF)] + [(nm(c), c) for c in fams], REF, seeds=(42,))
conf = [c for c in fams if len(C.runs(c)) >= 3]
L += C.table('Confirmation, seeds 42-44', [('hero (w3_hero)', REF)] + [(nm(c), c) for c in conf], REF)
L.append('\nLeak check, GRN positional encoding: per-fold graphs (fold-safe, own edge in no graph) vs full training graph (own edge inside), paired per-TF:')
for k in ('lap', 'rw'):
    a, b = f'w12_pe_{k}_grn_k16_leak', f'w12_pe_{k}_grn_k16_fold'
    if C.npzs(a) and C.npzs(b): L.append(w.paired(C.tf_mean(a), C.tf_mean(b), f'{k}: leak minus fold-safe'))
L.append('Fold-batch control (same batching as the fold-safe PE, no PE): ' + (w.paired(C.tf_mean('w12_pe_foldctl'), C.tf_mean(REF), 'foldctl - hero') if C.npzs('w12_pe_foldctl') else 'no runs'))
L.append('Random-vector control vs each confirmed PE: ' + ('' if C.npzs('w12_pe_rand_k16') else 'no runs'))
for c in conf:
    if c != 'w12_pe_rand_k16' and C.npzs('w12_pe_rand_k16') and 'rand' not in c: L.append(w.paired(C.tf_mean(c), C.tf_mean('w12_pe_rand_k16'), nm(c) + ' - random PE'))
K = C.pair_keys()
best = max([c for c in conf if 'rand' not in c and 'foldctl' not in c], key=lambda c: np.mean([r['val']['auc'] + r['val']['within'] for r in C.runs(c)]), default=None)
if best:
    L.append(f'\nWhere the gain sits (best confirmed family by mean validation AUROC + within-TF: {best}; pooled test-pair AUROC, family vs hero, mean over seeds):')
    L += C.strata_lines(best, REF, K['maxcos'], 'max cosine to the TF\'s known targets (low = far from any known target)')
    L += C.strata_lines(best, REF, K['pop'], 'gene popularity (training in-degree)')
    L.append(f'  score/popularity Spearman over all genes (mean over TFs): hero {np.mean([C.pop_corr(q, C.popularity()) for q in C.npzs(REF)]):+.3f}, {best} {np.mean([C.pop_corr(q, C.popularity()) for q in C.npzs(best)]):+.3f}')
for prot in ('gene', 'tf'):
    L += ['\n== Stage 2'] + C.ho_report(prot, ['pe_lap_ppi_k16'])
open('analysis/out/w12_9_pe.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
