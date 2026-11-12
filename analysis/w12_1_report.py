"""W12 idea 1 (query-aware TF summary, --decoder qatt): stage-1 table, paired per-TF tests vs hero (seeds 42-44 unless a family has fewer),
and explanation tests: per-TF gain vs multi-modality of the TF's training targets, attention entropy, non-parametric max-similarity score.
Run: python3.12 analysis/w12_1_report.py"""
import glob, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); sys.path.insert(0, '.')
import w12_common as C
import w3_report as w
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score

REF = 'w3_hero'
fams = lambda pat: sorted({re.sub(r'_s\d+\.json$', '', f.split('/')[-1]) for f in glob.glob(f'{C.OUT}{pat}_s*.json')})
S1 = fams('w12_qatt_*')
L = ['W12 idea 1: query-aware TF summary. hESC, random split, no Geneformer, epoch chosen on validation AUROC, refit. Seed-42 screen rows have n=1.']
L += C.table('Screen (seed 42 only; knob choice by val AUROC / within-TF in the val column)', [('hero (w3_hero)', REF)] + [(c.replace('w12_', ''), c) for c in S1 if len(C.runs(c)) == 1], REF, seeds=(42,))
conf = [c for c in S1 + ['w12_qattdrop', 'w12_qattuni'] if len(C.runs(c)) >= 3]
if conf:
    L += C.table('Confirmation (seeds 42-44)', [('hero', REF)] + [(c.replace('w12_', ''), c) for c in conf], REF)
    pr = lambda c: C.tf_mean(c)
    best = conf[0]
    # --- explanation tests on the first confirmed family (edit BEST to inspect another)
    BEST = best
    tr = pd.read_csv('data/hESC/TFs+500/Train_set.csv', index_col=0)
    x = pd.read_csv('data/hESC/TFs+500/BL--ExpressionData.csv', index_col=0).values.astype(np.float32)
    xn = (x - x.mean(1, keepdims=True)); xn /= np.linalg.norm(xn, axis=1, keepdims=True) + 1e-9
    A, B = C.tf_mean(BEST), C.tf_mean(REF)
    ks = sorted(set(A) & set(B)); gain = np.array([A[k] - B[k] for k in ks])
    L.append(f'\n== Explanation tests for {BEST}')
    props = {}
    for k in ks:
        tg = tr.Target.values[(tr.TF.values == k) & (tr.Label.values == 1)]
        if len(tg) < 3: props[k] = (len(tg), np.nan, np.nan); continue
        S = xn[tg] @ xn[tg].T; iu = np.triu_indices(len(tg), 1)
        nc = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, metric='cosine', linkage='average').fit(xn[tg]).n_clusters_
        props[k] = (len(tg), float(S[iu].mean()), nc)
    P = np.array([props[k] for k in ks], float)
    for j, nm in enumerate(['number of training targets', 'mean pairwise cosine of the targets (low = multi-modal)', 'number of clusters (average link, cosine distance < 1)']):
        ok = ~np.isnan(P[:, j]); r = spearmanr(P[ok, j], gain[ok]); L.append(f'  Spearman(per-TF within-TF gain vs hero, {nm}) = {r.statistic:+.3f}, p = {r.pvalue:.3g}, {ok.sum()} TFs')
    # attention entropy and max-similarity on test pairs
    z = C.npzs(BEST)
    if z and 'q_ent' in z[0].files:
        ent = np.concatenate([q['q_ent'] / np.log(np.maximum(q['q_nb'], 2)) for q in z]); y = np.concatenate([q['y'] for q in z])
        L.append(f'  normalised attention entropy (entropy / log #targets), test pairs: positives {ent[y == 1].mean():.3f}, negatives {ent[y == 0].mean():.3f}')
        ms = np.concatenate([q['q_maxsim'] for q in z]); L.append(f'  masked max cosine of the learnt embeddings, positives {ms[y == 1].mean():.3f}, negatives {ms[y == 0].mean():.3f}; AUROC of this single number {roc_auc_score(y, ms):.4f}')
    # non-parametric: max / top3-mean cosine of the expression profile of g to the TF's known targets (excluding g), and popularity alone
    te = pd.read_csv('data/hESC/TFs+500/Test_set.csv', index_col=0)
    pop = C.popularity()
    tgs = {t: tr.Target.values[(tr.TF.values == t) & (tr.Label.values == 1)] for t in np.unique(te.TF.values)}
    mx, t3 = np.zeros(len(te)), np.zeros(len(te))
    for i, (t, g) in enumerate(zip(te.TF.values, te.Target.values)):
        tg = tgs[t][tgs[t] != g]
        if len(tg): s = np.sort(xn[tg] @ xn[g])[::-1]; mx[i], t3[i] = s[0], s[:3].mean()
    y = te.Label.values; tf = te.TF.values
    from src.pair_gnn import within_tf
    L.append('  non-parametric single scores on the test set (no training; known targets = all training positives of the TF):')
    for nm, sc_ in [('max cosine to known targets', mx), ('mean of top-3 cosines', t3), ('gene popularity (training in-degree)', pop[te.Target.values].astype(float))]:
        L.append(f'    {nm:42s} AUROC {roc_auc_score(y, sc_):.4f}  within-TF {within_tf(y, sc_, tf):.4f}')
    L.append(f'    hero ({REF}) AUROC {np.mean([r["refit"]["auc"] for r in C.runs(REF)]):.4f}  within-TF {np.mean([r["refit"]["within"] for r in C.runs(REF)]):.4f}')
    pp = C.popularity()
    L.append(f'  score/popularity Spearman over all genes (mean over TFs): hero {np.mean([C.pop_corr(q, pp) for q in C.npzs(REF)]):+.3f}, {BEST} {np.mean([C.pop_corr(q, pp) for q in z]):+.3f}')
L.append('\n== Stage 3: other cell types (no Geneformer; baseline = w4 hero runs of the cell, seeds 42-44) and + Geneformer on hESC')
for cell in ('hHEP', 'mESC', 'mHSC-E'):
    ref = f'w4_{cell}_hero'
    L += C.table(f'{cell}', [(f'hero ({cell})', ref), ('qatt_h1_emb_k0', f'w12_{cell}_qatt'), ('qatt, uniform attention (control)', f'w12_{cell}_qattuni')], ref)
L += C.table('hESC + Geneformer', [('hero + Geneformer (w3_herogf)', 'w3_herogf'), ('qatt + Geneformer', 'w12_qattgf')], 'w3_herogf')
# held-out genes: where does the qatt gain sit? per-TF gain vs the TF's multi-modality
A, B = C.ho_per_tf('gene', 'qatt_h1_emb_k0'), C.ho_per_tf('gene', 'hero', ref=True)
ks = sorted(set(A) & set(B))
if ks:
    trn = pd.read_csv('data/hESC/TFs+500/Train_set.csv', index_col=0)
    xx = pd.read_csv('data/hESC/TFs+500/BL--ExpressionData.csv', index_col=0).values.astype(np.float32)
    xx = xx - xx.mean(1, keepdims=True); xx /= np.linalg.norm(xx, axis=1, keepdims=True) + 1e-9
    gain = np.array([A[k] - B[k] for k in ks]); nt, mc = [], []
    for k in ks:
        tg = trn.Target.values[(trn.TF.values == k) & (trn.Label.values == 1)]
        nt.append(len(tg)); mc.append(float((xx[tg] @ xx[tg].T)[np.triu_indices(len(tg), 1)].mean()) if len(tg) > 2 else np.nan)
    L.append('\nHeld-out genes, per-TF gain of qatt_h1_emb_k0 over the hero (per-TF AUROC averaged over 5 folds x 2 seeds):')
    for nm_, v in (('number of training targets of the TF', np.array(nt, float)), ('mean pairwise cosine of the TF\'s targets (low = multi-modal)', np.array(mc))):
        ok = ~np.isnan(v); r = spearmanr(v[ok], gain[ok]); L.append(f'  Spearman(gain, {nm_}) = {r.statistic:+.3f}, p = {r.pvalue:.3g}, {ok.sum()} TFs')
    lo = np.array(nt) < np.median(nt)
    L.append(f'  TFs with fewer than the median number of targets ({np.median(nt):.0f}): mean gain {gain[lo].mean():+.4f} (n={lo.sum()}); with more: {gain[~lo].mean():+.4f} (n={(~lo).sum()})')
for prot in ('gene', 'tf'):
    L += ['\n== Stage 2'] + C.ho_report(prot, ['qatt_h1_emb_k0', 'qattdrop', 'qattuni'])
    L += ['  same, against the matched control hero + tf_drop 0.3 (earlier td01 runs):'] + C.ho_report(prot, ['qattdrop'], ref='td01_hero_tfdrop0.3')[1:]
open('analysis/out/w12_1_query_tf.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
