"""
Step 1: Load Norman Perturb-seq data via GEARS and extract
(TF, gene, response_label) triples for pretraining.

Response labels:
  0 = no significant change
  1 = significantly upregulated
  2 = significantly downregulated
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats
from gears import PertData

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR       = str(PROJECT_ROOT / "data")
DATASET_DIR    = PROJECT_ROOT / "data" / "hESC" / "TFs+500"
TF_CSV         = DATASET_DIR / "TF.csv"
TARGET_CSV     = DATASET_DIR / "Target.csv"
OUT_DIR        = Path(__file__).resolve().parent / "perturb_triples"
FC_THRESHOLD   = 0.5                  # log2 fold-change cutoff
PVAL_CUTOFF    = 0.05                 # FDR-adjusted p-value cutoff
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Norman 2019 Perturb-seq ──────────────────────────────────────────
print("Loading Norman 2019 Perturb-seq via GEARS...")
pert_data = PertData(DATA_DIR)
pert_data.load(data_name="norman")
pert_data.prepare_split(split="simulation", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

adata = pert_data.adata           # AnnData object: cells × genes
print(f"AnnData shape: {adata.shape}")
print(f"Available perturbations: {len(adata.obs['condition'].unique())}")

# ── 2. Load your BEELINE TF/gene set ─────────────────────────────────────────
tf_df = pd.read_csv(TF_CSV)
target_df = pd.read_csv(TARGET_CSV)

if "TF" not in tf_df.columns:
    tf_df.columns = ["_idx", "TF", "index"]
if "Gene" not in target_df.columns:
    target_df.columns = ["_idx", "Gene", "index"]

beeline_tfs = set(tf_df["TF"].astype(str).unique())
beeline_genes = set(target_df["Gene"].astype(str).unique())
beeline_tfs_upper = {g.upper() for g in beeline_tfs}
tf_upper_to_name = {g.upper(): g for g in beeline_tfs}
print(f"BEELINE TFs: {len(beeline_tfs)}")
print(f"BEELINE target genes: {len(beeline_genes)}")

# Global overlap against GEARS perturbation catalog (matches pertub.py behavior)
norman_pert_names_upper = {str(g).upper() for g in pert_data.pert_names}
tf_overlap_with_norman = beeline_tfs_upper & norman_pert_names_upper
print(f"BEELINE TF overlap with Norman pert_names: {len(tf_overlap_with_norman)}")

# ── 3. Get control expression ─────────────────────────────────────────────────
ctrl_mask = adata.obs["condition"] == "ctrl"
X_ctrl = adata[ctrl_mask].X
if hasattr(X_ctrl, "toarray"):
    X_ctrl = X_ctrl.toarray()
ctrl_mean = X_ctrl.mean(axis=0)     # shape: (n_genes,)
ctrl_std  = X_ctrl.std(axis=0) + 1e-8
if "gene_name" in adata.var.columns:
    gene_symbols = adata.var["gene_name"].astype(str).tolist()
else:
    gene_symbols = list(map(str, adata.var_names))

# case-insensitive symbol lookup; keep first index when duplicates exist
gene2idx = {}
for idx, symbol in enumerate(gene_symbols):
    symbol_upper = symbol.upper()
    if symbol_upper not in gene2idx:
        gene2idx[symbol_upper] = idx

overlap_count = len({g.upper() for g in beeline_genes} & set(gene2idx.keys()))
print(f"BEELINE target genes overlapping Norman gene symbols: {overlap_count}")

# ── 4. Compute differential expression per TF perturbation ───────────────────
print("Computing differential expression for each TF perturbation...")

triples = []   # (tf_name, gene_name, label)

perturbations = adata.obs["condition"].unique()
def extract_single_gene_pert(condition: str):
    if condition == "ctrl":
        return None
    parts = condition.split("+")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2 and "ctrl" in parts:
        return parts[0] if parts[1] == "ctrl" else parts[1]
    return None


# Index all single-gene condition cells once for fast lookup
single_pert_cells = {}
for p in perturbations:
    gene = extract_single_gene_pert(str(p))
    if gene is not None:
        mask = adata.obs["condition"] == p
        X = adata[mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        single_pert_cells[gene.upper()] = X

tf_perturbations_single = []
tf_perturbations_double = []   # (condition, tf_name, reference_X)

for p in perturbations:
    p = str(p)
    parts = p.split("+")

    # Single perturbation
    if len(parts) == 2 and "ctrl" in parts:
        gene = extract_single_gene_pert(p)
        if gene is not None:
            tf_perturbations_single.append(p)

    # Double perturbation
    elif len(parts) == 2 and "ctrl" not in parts:
        g1, g2 = parts[0].upper(), parts[1].upper()
        if g1 in beeline_tfs_upper:
            ref_X = single_pert_cells.get(g2, X_ctrl)
            tf_perturbations_double.append((p, tf_upper_to_name.get(g1, g1), ref_X))
        if g2 in beeline_tfs_upper:
            ref_X = single_pert_cells.get(g1, X_ctrl)
            tf_perturbations_double.append((p, tf_upper_to_name.get(g2, g2), ref_X))

print(f"Single-gene perturbation conditions used for pretraining: {len(tf_perturbations_single)}")
print(f"Double TF perturbations with >=1 BEELINE TF: {len(tf_perturbations_double)}")


def run_de(X_pert, X_ref, tf_name, triples_out):
    if X_pert.shape[0] < 5 or X_ref.shape[0] < 5:
        return

    pert_mean = X_pert.mean(axis=0)
    ref_mean = X_ref.mean(axis=0)
    log2fc = np.log2((pert_mean + 1e-4) / (ref_mean + 1e-4))

    ttest_res = stats.ttest_ind(X_pert, X_ref, axis=0, equal_var=False, nan_policy="omit")
    pvals = np.asarray(ttest_res.pvalue)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)

    from statsmodels.stats.multitest import multipletests
    _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")

    for gene in beeline_genes:
        g_upper = gene.upper()
        if g_upper not in gene2idx:
            continue
        g_idx = gene2idx[g_upper]
        fc, pv = log2fc[g_idx], pvals_adj[g_idx]

        if pv < PVAL_CUTOFF and fc > FC_THRESHOLD:
            label = 1
        elif pv < PVAL_CUTOFF and fc < -FC_THRESHOLD:
            label = 2
        else:
            label = 0

        triples_out.append((tf_name, gene, label, float(fc), float(pv)))


# Process single perturbations (reference = ctrl)
for pert in tf_perturbations_single:
    raw_tf = extract_single_gene_pert(pert)
    if raw_tf is None:
        continue
    tf_name = tf_upper_to_name.get(raw_tf.upper(), raw_tf)
    mask = adata.obs["condition"] == pert
    X_pert = adata[mask].X
    if hasattr(X_pert, "toarray"):
        X_pert = X_pert.toarray()
    run_de(X_pert, X_ctrl, tf_name, triples)


# Process double perturbations (reference = other single KO or ctrl)
for pert, tf_name, X_ref in tf_perturbations_double:
    mask = adata.obs["condition"] == pert
    X_pert = adata[mask].X
    if hasattr(X_pert, "toarray"):
        X_pert = X_pert.toarray()
    run_de(X_pert, X_ref, tf_name, triples)

print(f"Total triples: {len(triples)}")

# ── 5. Save triples ───────────────────────────────────────────────────────────
df = pd.DataFrame(triples, columns=["TF", "Gene", "Label", "Log2FC", "PVal_adj"])
out_path = OUT_DIR / "perturb_triples.csv"
df.to_csv(out_path, index=False)

# Label distribution
print("\nLabel distribution:")
print(df["Label"].value_counts())
print(f"\nSaved to {out_path}")

# Also save gene names and TF names for encoder mapping
all_norman_single_tf_names = set()
for p in perturbations:
    gene = extract_single_gene_pert(str(p))
    if gene is not None:
        all_norman_single_tf_names.add(gene)

gene_list = sorted(
    all_norman_single_tf_names
    | set(df["TF"])
    | set(df["Gene"])
    | beeline_tfs
    | beeline_genes
)
with open(OUT_DIR / "gene_list.pkl", "wb") as f:
    pickle.dump(gene_list, f)
print(f"Gene vocab size: {len(gene_list)}")
