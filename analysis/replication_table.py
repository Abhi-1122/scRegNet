"""Markdown tables for the four cell types: reference rows + harness configs (refit, mean ± std over seeds)."""
import glob, json, re
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def within(y, s, t):
    return np.mean([roc_auc_score(y[t == u], s[t == u]) for u in np.unique(t) if 0 < y[t == u].sum() < (t == u).sum()])

ORIG = {'hESC': 'hESC_baseline_rerun', 'hHEP': 'hHEP_best', 'mHSC-E': 'mHSC-E_best', 'mESC': 'mESC_best'}
CFG = [('g01_add_sf_none', 'scRegNet design (add, no split, Geneformer, 3-layer GCN)'),
       ('g02_add_sf_batch', '  + edge split'),
       ('g08_had_nosf_batch_d256', 'GCN 1-layer, interacting, split, no Geneformer'),
       ('g09_mlp_nosf_batch_d256', 'GCN 1-layer, hidden-layer final, split, no Geneformer'),
       ('g11_had_sf_batch_d256', 'GCN 1-layer, interacting, split, Geneformer'),
       ('e_sage_d256', 'SAGE 1-layer, interacting, split, no Geneformer'),
       ('k04_sage_had_nosf_d256_contrast', '  + co-expression number')]
ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'
for cell in ['hESC', 'hHEP', 'mHSC-E', 'mESC']:
    print(f'\n**{cell}**\n\n| Model | Seeds | AUROC | AUPRC | Within-TF |\n|---|---|---|---|---|')
    z = np.load(f'analysis/out/scores_tf_500_{ORIG[cell]}.npz'); tp = z['test_pairs']; s = z['test_logits'][:, 1] - z['test_logits'][:, 0]
    print(f'| original scRegNet checkpoint (paper recipe, epoch chosen on test) | 1 | {roc_auc_score(tp[:,2], s):.4f} | {average_precision_score(tp[:,2], s):.4f} | {within(tp[:,2], s, tp[:,0]):.4f} |')
    d = json.load(open(f'analysis/out/cs_baselines_{cell}.json'))
    for k, nm in [('popularity (logdt, logdg) [LR]', 'popularity only (two counts, logistic regression)'), ('+ co-expression contrast [LR]', 'no-neural-network stacker (popularity + CF + 3-paths + co-expression number)')]:
        print(f"| {nm} | - | {d[k]['auc']:.4f} | {d[k]['aupr']:.4f} | {d[k]['within']:.4f} |")
    for c, nm in CFG:
        pre = c if cell == 'hESC' else f'm_{cell}_{c}'
        rs = [json.load(open(f)) for f in sorted(glob.glob(f'analysis/out/pair/{pre}_s*.json')) if re.search(rf'{re.escape(pre)}_s\d+\.json$', f)]
        if rs:
            g = lambda m: [r['refit'][m] for r in rs]
            print(f"| {nm} | {len(rs)} | {ms(g('auc'))} | {ms(g('aupr'))} | {ms(g('within'))} |")
