import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
import numpy as np

# Load
adata_L5  = sq.read.visium("L5/",  counts_file="filtered_feature_bc_matrix.h5")
adata_L18 = sq.read.visium("L18/", counts_file="filtered_feature_bc_matrix.h5")

# Fix warnings
adata_L5.var_names_make_unique()
adata_L18.var_names_make_unique()
adata_L5.obs_names  = [f"L5_{x}"  for x in adata_L5.obs_names]
adata_L18.obs_names = [f"L18_{x}" for x in adata_L18.obs_names]

adata_L5.obs["donor"]  = "L5"
adata_L18.obs["donor"] = "L18"

# Merge
adata = ad.concat([adata_L5, adata_L18], label="donor",
                  keys=["L5", "L18"], uns_merge="first")

# ── Step 1: Filter to only in-tissue spots
adata = adata[adata.obs["in_tissue"] == 1].copy()
print(f"In-tissue spots: {adata.n_obs}")   # expect ~6000-7000

# ── Step 2: Filter to your hHEP genes only
tf_df     = pd.read_csv("/home/abhishekg/Documents/sproj/paper_code/scRegNet/data/hHEP/TFs+500/TF.csv",     index_col=0)
target_df = pd.read_csv("/home/abhishekg/Documents/sproj/paper_code/scRegNet/data/hHEP/TFs+500/Target.csv", index_col=0)

your_genes = list((set(tf_df["TF"]) | set(target_df["Gene"])) & set(adata.var_names))
adata = adata[:, your_genes].copy()
print(f"Genes after filter: {adata.n_vars}")   # expect 933

# ── Step 3: Normalize
sc.pp.filter_spots = lambda a: a[a.obs["in_tissue"] == 1]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ── Step 4: Save spatial_expression.csv (genes × spots — same format as BL-ExpressionData)
import scipy.sparse as sp
X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
expr_df = pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names)
expr_df.to_csv("spatial_expression.csv")
print(f"spatial_expression.csv: {expr_df.shape}")   # (933 genes × ~6500 spots)

# ── Step 5: Save spatial_coords.csv
coords_df = pd.DataFrame(
    adata.obsm["spatial"],
    index=adata.obs_names,
    columns=["x", "y"]
)
coords_df.to_csv("spatial_coords.csv")
print(f"spatial_coords.csv: {coords_df.shape}")     # (~6500 spots × 2)

# ── Step 6: Rebuild TF/Target index for overlapping genes only
gene_list   = adata.var_names.tolist()
gene_to_idx = {g: i for i, g in enumerate(gene_list)}

tf_spatial = tf_df[tf_df["TF"].isin(gene_list)].copy()
tf_spatial["index"] = tf_spatial["TF"].map(gene_to_idx)
tf_spatial.to_csv("TF_spatial.csv")

target_spatial = target_df[target_df["Gene"].isin(gene_list)].copy()
target_spatial["index"] = target_spatial["Gene"].map(gene_to_idx)
target_spatial.to_csv("Target_spatial.csv")

print(f"\nTF_spatial.csv:     {len(tf_spatial)} TFs")
print(f"Target_spatial.csv: {len(target_spatial)} targets")
print("\n✅ All preprocessing files saved!")
print("Next: build spatial kNN graph")