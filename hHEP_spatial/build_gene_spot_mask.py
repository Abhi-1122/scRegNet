"""
build_gene_spot_mask.py
───────────────────────
Run once from the hHEP_spatial/ directory:

    python build_gene_spot_mask.py

Reads spatial_expression.csv  (933 genes × 7982 spots, log-normalised)
and produces  gene_spot_mask.pt

gene_spot_mask is a Python list of length N_genes.
Each element is a 1-D LongTensor containing the indices of the spots that
express that gene (raw count > threshold after log-norm, default 0.0).

Saved object:
    {
        "gene_spot_mask" : list[LongTensor],   # length = num_genes
        "gene_order"     : list[str],          # gene names, same order
        "num_spots"      : int,
    }

The mask is used by AttentionPool in scRegNet-Spatial to pool spatially
informed spot embeddings into gene-level representations.
"""

import os
import numpy as np
import pandas as pd
import torch

# ── Config ─────────────────────────────────────────────────────────────────
EXPR_CSV   = "spatial_expression.csv"   # genes × spots (columns)
OUT_FILE   = "gene_spot_mask.pt"
THRESHOLD  = 0.0   # spots with log-norm expression > threshold are "expressing"
# ───────────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
expr_path  = os.path.join(script_dir, EXPR_CSV)
out_path   = os.path.join(script_dir, OUT_FILE)

print(f"Loading expression matrix from: {expr_path}")
expr = pd.read_csv(expr_path, index_col=0)   # rows=genes, cols=spots

num_genes, num_spots = expr.shape
print(f"  genes : {num_genes}")
print(f"  spots : {num_spots}")

expr_arr = expr.values.astype(np.float32)   # (num_genes, num_spots)

gene_spot_mask = []
nonzero_counts = []

for g_idx in range(num_genes):
    spot_indices = np.where(expr_arr[g_idx] > THRESHOLD)[0]
    gene_spot_mask.append(torch.tensor(spot_indices, dtype=torch.long))
    nonzero_counts.append(len(spot_indices))

nonzero_counts = np.array(nonzero_counts)
print(f"\nExpression sparsity stats (threshold={THRESHOLD}):")
print(f"  Mean spots per gene  : {nonzero_counts.mean():.1f}")
print(f"  Median spots per gene: {np.median(nonzero_counts):.1f}")
print(f"  Min / Max            : {nonzero_counts.min()} / {nonzero_counts.max()}")
print(f"  Genes with 0 spots   : {(nonzero_counts == 0).sum()}")

save_dict = {
    "gene_spot_mask": gene_spot_mask,
    "gene_order":     list(expr.index),
    "num_spots":      num_spots,
}
torch.save(save_dict, out_path)
print(f"\n✅  gene_spot_mask.pt saved to: {out_path}")
print(f"   List length (= num_genes): {len(gene_spot_mask)}")
