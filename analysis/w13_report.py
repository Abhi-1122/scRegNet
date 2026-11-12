"""W13 leads: (A) second TF summary (--decoder qatt, uniform) and (B) learned DC-SBM score as a decoder scalar (--scalar dcsbm), both on each cell's best grid
configuration (section 24, Geneformer on). Seeds 42-44, paired per-TF Wilcoxon vs the matching baseline. Run: python3.12 analysis/w13_report.py -> analysis/out/w13_leads.txt"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); sys.path.insert(0, '.')
import w12_common as C
import w3_report as w
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from src.pair_gnn import within_tf

SEEDS = C.SEEDS
BASE = {'hESC': 'w5_nphad_gf', 'hHEP': 'w11_grid_hHEP_npanda_had_batch_gf1', 'mHSC-E': 'w8_mHSC-E_gcngf', 'mESC': 'w11_grid_mESC_sage_had_none_gf1'}
CFG = {'hESC': 'npanda 2 layers, batch split', 'hHEP': 'npanda 2 layers, batch split', 'mHSC-E': 'GCN 1 layer, batch split', 'mESC': 'SAGE 1 layer, no split'}
L = ['W13 leads. Best grid configuration of each cell (Hadamard decoder, Geneformer on, hero hyper-parameters), epoch chosen on validation AUROC, refit on all training rows; seeds 42-44;',
     'paired per-TF Wilcoxon (per-TF test AUROC averaged over the three seeds) against the matching baseline run of the grid. qs = second TF summary (qatt, uniform, 1 head, post-GNN embeddings);',
     'dc = learned DC-SBM score as a decoder input (--scalar dcsbm, out-of-fold for training rows); dclr10 = same with a separate linear term at 10x learning rate.']
chosen = {}
L.append('\n== Choice of the dcsbm variant per cell (validation AUROC at seed 42)')
for c in BASE:
    v = {k: json.load(open(f'{C.OUT}w13_{c}_{k}_s42.json'))['val']['auc'] for k in ('dc', 'dclr10') if os.path.exists(f'{C.OUT}w13_{c}_{k}_s42.json')}
    if v:
        chosen[c] = max(v, key=v.get); L.append(f'{c:8s} ' + '  '.join(f'{k} val AUROC {x:.4f}' for k, x in v.items()) + f'  -> {chosen[c]}')
fam = lambda c, k: f'w13_{c}_{k}'
ntr = lambda c, t: None
for c in BASE:
    items = [('baseline (' + CFG[c] + ')', BASE[c]), ('+ summary (qs)', fam(c, 'qs'))]
    if c in chosen:
        k = chosen[c]; o = 'dc' if k == 'dclr10' else 'dclr10'
        items += [(f'+ dcsbm scalar ({k})', fam(c, k)), (f'+ both (qs + {k})', fam(c, 'qs' + k))]
        items += [(f'  not chosen: dcsbm ({o})', fam(c, o))] if len(C.runs(fam(c, o))) >= 2 else []
    L += C.table(f'{c}', items, BASE[c], 'baseline')

# ---- rank-average diagnostic: 50/50 rank average of the baseline GNN scores and the DC-SBM test scores (analysis/out/stat/)
L.append('\n== Diagnostic (not a method): 50/50 rank average of baseline GNN and learned DC-SBM test scores (global ranks over the test pairs), seeds 42-44')
L.append(f"{'cell':8s}{'model':26s}{'AUROC':>16s}{'AUPRC':>16s}{'within-TF':>16s}")
stat_npz = lambda c, s: np.load(f'analysis/out/stat/{c}_dcsbm_learned_s{s}.npz')
for c in BASE:
    rows = {'GNN baseline': [], 'DC-SBM alone': [], 'rank average': []}; pt = {k: [] for k in rows}
    for s in SEEDS:
        g, d = np.load(f'{C.OUT}{BASE[c]}_s{s}.npz'), stat_npz(c, s)
        assert (g['tf'] == d['tf']).all() and (g['tgt'] == d['tgt']).all() and (g['y'] == d['y']).all()
        ra = (rankdata(g['logit']) + rankdata(d['logit'])) / 2
        for k, sc in (('GNN baseline', g['logit']), ('DC-SBM alone', d['logit']), ('rank average', ra)):
            rows[k].append((roc_auc_score(g['y'], sc), average_precision_score(g['y'], sc), within_tf(g['y'], sc, g['tf']))); pt[k].append(w.per_tf(g['tf'], g['y'], sc))
    for k, r in rows.items():
        r = np.array(r); L.append(f'{c:8s}{k:26s}' + ''.join(f'{w.ms(r[:, j]):>16s}' for j in range(3)))
    tm = lambda k: {t: float(np.mean([x[t] for x in pt[k]])) for t in pt[k][0]}
    L.append(w.paired(tm('rank average'), tm('GNN baseline'), f'{c}: rank average - GNN baseline'))
    L.append(w.paired(tm('rank average'), tm('DC-SBM alone'), f'{c}: rank average - DC-SBM alone'))
    L.append(w.paired(tm('DC-SBM alone'), tm('GNN baseline'), f'{c}: DC-SBM alone - GNN baseline'))
    L.append(f"   Spearman between the two models' test scores: {np.mean([spearmanr(np.load(f'{C.OUT}{BASE[c]}_s{s}.npz')['logit'], stat_npz(c, s)['logit']).statistic for s in SEEDS]):.3f}")

# ---- why: per-TF gain vs number of training targets and vs the DC-SBM's own per-TF advantage
L.append('\n== Where the gain sits (per-TF test AUROC, seed-averaged). n_t = number of training targets of the TF; adv = DC-SBM minus baseline per-TF AUROC')
for c in BASE:
    tr = pd.read_csv(f'data/{c}/TFs+500/Train_set.csv', index_col=0)
    nt = tr[tr.Label == 1].groupby('TF').size()
    B = C.tf_mean(BASE[c])
    dc = {t: float(np.mean([w.per_tf(stat_npz(c, s)['tf'], stat_npz(c, s)['y'], stat_npz(c, s)['logit'])[t] for s in SEEDS])) for t in B}
    for lab, f_ in [('qs', fam(c, 'qs'))] + ([(chosen[c], fam(c, chosen[c])), ('both', fam(c, 'qs' + chosen[c]))] if c in chosen else []):
        A = C.tf_mean(f_)
        ks = sorted(set(A) & set(B))
        if len(ks) < 5: continue
        gain = np.array([A[k] - B[k] for k in ks]); n_ = np.array([nt.get(k, 0) for k in ks], float); adv = np.array([dc[k] - B[k] for k in ks])
        r1, r2 = spearmanr(n_, gain), spearmanr(adv, gain); lo = n_ < np.median(n_)
        r3 = spearmanr([dc[k] for k in ks], gain)
        L.append(f'{c:8s}{lab:8s} gain vs n_t: Spearman {r1.statistic:+.3f} (p={r1.pvalue:.2g});  gain vs DC-SBM advantage: Spearman {r2.statistic:+.3f} (p={r2.pvalue:.2g}) [shares -baseline with gain; vs DC-SBM per-TF AUROC alone: {r3.statistic:+.3f} (p={r3.pvalue:.2g})];  '
                 f'mean gain TFs below / above median n_t ({np.median(n_):.0f}): {gain[lo].mean():+.4f} / {gain[~lo].mean():+.4f}')

# ---- diagnostic: which factor removes the section-25 gain on hHEP (Geneformer or the 2-layer neural PANDA encoder)?
L.append('\n== Diagnostic on hHEP: the summary (qs) on top of four baselines that differ in encoder and Geneformer (batch split, Hadamard; seeds 42-44)')
for nm, b, q in [('sage 1 layer, no Geneformer (section 25)', 'w4_hHEP_hero', 'w12_hHEP_qattuni'), ('sage 1 layer, Geneformer', 'w4_hHEP_herogf', 'w13_hHEP_sagegf1_qs'),
                 ('npanda 2 layers, no Geneformer', 'w4_hHEP_np_had', 'w13_hHEP_npgf0_qs'), ('npanda 2 layers, Geneformer (best config)', BASE['hHEP'], fam('hHEP', 'qs'))]:
    if len(C.runs(q)) < 3: L.append(f'  {nm}: runs missing'); continue
    L.append(f"  {nm:44s} baseline within-TF {np.mean([r['refit']['within'] for r in C.runs(b)]):.4f}, +qs {np.mean([r['refit']['within'] for r in C.runs(q)]):.4f}")
    L.append('  ' + w.paired(C.tf_mean(q), C.tf_mean(b), '   + qs - baseline'))

# ---- hHEP held-out genes, 5 folds x seeds 42,43
L.append('\n== hHEP held-out genes (5 folds x seeds 42, 43; best hHEP configuration with and without the summary)')
def ho(v):
    js, d = [], {}
    for f in range(5):
        for s in (42, 43):
            n = f'{C.OUT}w13_ho_gene{f}_hHEP_{v}_s{s}'
            if os.path.exists(n + '.json') and os.path.exists(n + '.npz'):
                js.append(json.load(open(n + '.json')))
                z = np.load(n + '.npz')
                for k, x in w.per_tf(z['tf'], z['y'], z['logit']).items(): d.setdefault(k, []).append(x)
    return js, {k: float(np.mean(x)) for k, x in d.items()}
(jb, tb), (jq, tq) = ho('base'), ho('qs')
L.append(f"{'model':16s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s}")
for nm, j in (('baseline', jb), ('+ summary', jq)):
    L.append(f'{nm:16s} {len(j)}  ' + ' '.join(f"{w.ms([x['refit'][m] for x in j]):>16s}" for m in ('auc', 'aupr', 'within')) if j else f'{nm}: no runs')
if jb and jq:
    L.append(w.paired(tq, tb, 'hHEP held-out genes: + summary - baseline'))
    tr = pd.read_csv('data/hHEP/TFs+500/Train_set.csv', index_col=0); nt = tr[tr.Label == 1].groupby('TF').size()
    ks = sorted(set(tq) & set(tb)); gain = np.array([tq[k] - tb[k] for k in ks]); n_ = np.array([nt.get(k, 0) for k in ks], float)
    r = spearmanr(n_, gain); lo = n_ < np.median(n_)
    L.append(f'  gain vs number of training targets: Spearman {r.statistic:+.3f} (p={r.pvalue:.2g}); below / above median ({np.median(n_):.0f}): {gain[lo].mean():+.4f} / {gain[~lo].mean():+.4f}')
open('analysis/out/w13_leads.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
