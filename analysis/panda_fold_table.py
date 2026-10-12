"""Markdown table: each model with and without the PANDA / co-expression number (or PANDA edge weights)."""
import glob, json, re
import numpy as np
def get(pre):
    rs = [json.load(open(f)) for f in sorted(glob.glob(f'analysis/out/pair/{pre}_s*.json')) if re.search(rf'{re.escape(pre)}_s\d+\.json$', f)]
    return None if not rs else {m: np.array([r['refit'][m] for r in rs]) for m in ('auc', 'aupr', 'within')} | {'n': len(rs)}
PAIRS = [  # (control, [(variant, label)])
    ('g08_had_nosf_batch_d256', 'GCN 1-layer, interacting, no Geneformer', [('h01_had_nosf_batch_d256_contrast', '+ co-expression number'), ('h02_had_nosf_batch_d256_panda', '+ PANDA number'), ('h03_had_nosf_batch_d256_gwpanda', 'PANDA as graph edge weights'), ('h04_had_nosf_batch_d256_gwpanda_panda', 'edge weights + PANDA number')]),
    ('g09_mlp_nosf_batch_d256', 'GCN 1-layer, hidden-layer final, no Geneformer', [('h05_mlp_nosf_batch_d256_contrast', '+ co-expression number'), ('h06_mlp_nosf_batch_d256_panda', '+ PANDA number')]),
    ('g11_had_sf_batch_d256', 'GCN 1-layer, interacting, Geneformer', [('h07_had_sf_batch_d256_contrast', '+ co-expression number'), ('h08_had_sf_batch_d256_panda', '+ PANDA number')]),
    ('g01_add_sf_none', "scRegNet's design", [('h09_add_sf_none_panda', '+ PANDA number'), ('h11_add_sf_none_contrast', '+ co-expression number')]),
    ('g02_add_sf_batch', "scRegNet's design + edge split", [('h10_add_sf_batch_panda', '+ PANDA number')]),
    ('g06_add_nosf_none_d256', 'GCN 1-layer, additive, no Geneformer', [('h12_add_nosf_none_d256_panda', '+ PANDA number')]),
    ('e_sage_d256', 'SAGE 1-layer, interacting, no Geneformer (default settings)', [('k04_sage_had_nosf_d256_contrast', '+ co-expression number')]),
    ('f04_C_tuned', 'SAGE 1-layer, interacting, no Geneformer (tuned settings)', [('f05_C_contrast', '+ co-expression number'), ('f06_C_panda', '+ PANDA number')]),
    ('f03_B_tuned', 'SAGE 1-layer, interacting, Geneformer (tuned settings)', [('f11_B_panda', '+ PANDA number')]),
]
print('| Model | Variant | Seeds | AUROC | AUPRC | Within-TF | Change in within-TF |\n|---|---|---|---|---|---|---|')
for c, nm, vs in PAIRS:
    C = get(c)
    if C is None: continue
    print(f"| {nm} | none (control) | {C['n']} | {C['auc'].mean():.4f} ± {C['auc'].std():.4f} | {C['aupr'].mean():.4f} ± {C['aupr'].std():.4f} | {C['within'].mean():.4f} ± {C['within'].std():.4f} | - |")
    for v, lab in vs:
        V = get(v)
        if V is None: print(f'| | {lab} | pending | | | | |'); continue
        print(f"| | {lab} | {V['n']} | {V['auc'].mean():.4f} ± {V['auc'].std():.4f} | {V['aupr'].mean():.4f} ± {V['aupr'].std():.4f} | {V['within'].mean():.4f} ± {V['within'].std():.4f} | {V['within'].mean()-C['within'].mean():+.4f} |")
