"""
Step 4: Visualise and compare baseline vs pretrained model.

Generates:
  - results/auroc_auprc_comparison.png   — bar chart per fold + mean
  - results/summary_table.csv            — clean summary
  - results/overlap_ablation.csv         — AUROC on overlap-TFs vs non-overlap-TFs
                                            (the cleanest ablation for the paper)
"""

import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, pickle, torch, inspect
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from encoder import FusedGeneEncoder

SCRIPT_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DATASET_DIR      = PROJECT_ROOT / "data" / "hESC" / "TFs+500"

RESULTS_DIR      = SCRIPT_DIR / "results"
LABEL_CSV        = DATASET_DIR / "Label.csv"
TRAIN_SET_CSV    = DATASET_DIR / "Train_set.csv"
TEST_SET_CSV     = DATASET_DIR / "Test_set.csv"
TARGET_CSV       = DATASET_DIR / "Target.csv"
TRIPLES_CSV      = SCRIPT_DIR / "perturb_triples" / "perturb_triples.csv"
GENE2IDX_PKL     = SCRIPT_DIR / "perturb_triples" / "gene_list.pkl"
FM_EMBEDDINGS    = DATASET_DIR / "fm_embeddings.pt"
PRIOR_EDGE_INDEX = SCRIPT_DIR / "BL-network_edge_index.pt"
PRETRAINED_CKPT  = SCRIPT_DIR / "checkpoints" / "pretrained_encoder.pt"
HIDDEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_torch_load(path, map_location):
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def load_beeline_edges() -> pd.DataFrame:
    label_df = pd.read_csv(LABEL_CSV)
    if {"TF", "Target", "Label"}.issubset(label_df.columns):
        return label_df[["TF", "Target", "Label"]].copy()

    train_df = pd.read_csv(TRAIN_SET_CSV)
    test_df = pd.read_csv(TEST_SET_CSV)
    idx_df = pd.concat([train_df, test_df], ignore_index=True)

    target_map = pd.read_csv(TARGET_CSV)
    idx_to_gene = dict(zip(target_map["index"], target_map["Gene"].astype(str)))
    idx_df["TF"] = idx_df["TF"].map(idx_to_gene)
    idx_df["Target"] = idx_df["Target"].map(idx_to_gene)
    idx_df = idx_df.dropna(subset=["TF", "Target", "Label"]).copy()
    idx_df["Label"] = idx_df["Label"].astype(int)
    return idx_df[["TF", "Target", "Label"]]

# ── 1. Load CV results ────────────────────────────────────────────────────────
base_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_cv_results.csv"))
pre_df  = pd.read_csv(os.path.join(RESULTS_DIR, "pretrained_cv_results.csv"))

summary = pd.DataFrame({
    "Model":       ["Baseline (random init)", "Pretrained (Norman Perturb-seq)"],
    "AUROC mean":  [base_df["AUROC"].mean(), pre_df["AUROC"].mean()],
    "AUROC std":   [base_df["AUROC"].std(),  pre_df["AUROC"].std()],
    "AUPRC mean":  [base_df["AUPRC"].mean(), pre_df["AUPRC"].mean()],
    "AUPRC std":   [base_df["AUPRC"].std(),  pre_df["AUPRC"].std()],
})
summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)
print(summary.to_string(index=False))

# ── 2. Bar chart ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
    folds = base_df["fold"].tolist()
    x = np.arange(len(folds))
    ax.bar(x - 0.2, base_df[metric], width=0.35, label="Baseline", color="#7a9cbf")
    ax.bar(x + 0.2, pre_df[metric],  width=0.35, label="Pretrained", color="#01696f")
    ax.axhline(base_df[metric].mean(), color="#7a9cbf", linestyle="--", alpha=0.6)
    ax.axhline(pre_df[metric].mean(),  color="#01696f", linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_title(metric); ax.legend(); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel(metric)

plt.suptitle("Baseline vs Perturbation-Pretrained scRegNet (5-fold CV)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "auroc_auprc_comparison.png"),
            dpi=150, bbox_inches="tight")
print("Saved bar chart.")

# ── 3. Overlap ablation ───────────────────────────────────────────────────────
# Key experiment for paper: do overlap TFs benefit MORE from pretraining?
label_df = load_beeline_edges()

perturb_df = pd.read_csv(TRIPLES_CSV)
perturbed_tfs = set(perturb_df["TF"].unique())

label_df["TF_in_norman"] = label_df["TF"].isin(perturbed_tfs)

with open(GENE2IDX_PKL, "rb") as f:
    gene_list = pickle.load(f)
gene2idx = {g: i for i, g in enumerate(gene_list)}
label_df = label_df[
    label_df["TF"].isin(gene2idx) & label_df["Target"].isin(gene2idx)
].reset_index(drop=True)

# Load FM matrix and edge index
fm_dict = safe_torch_load(FM_EMBEDDINGS, map_location="cpu")
if not isinstance(fm_dict, dict) or len(fm_dict) == 0:
    raise ValueError(f"Expected non-empty dict in {FM_EMBEDDINGS}, got {type(fm_dict)}")

sample_vec = next(iter(fm_dict.values()))
if not torch.is_tensor(sample_vec):
    sample_vec = torch.as_tensor(sample_vec)
sample_vec = sample_vec.reshape(-1)
FM_DIM = int(sample_vec.numel())
print(f"Detected FM embedding dimension: {FM_DIM}")

fm_matrix = torch.zeros(len(gene_list), FM_DIM)
for gene, idx in gene2idx.items():
    if gene in fm_dict:
        vec = fm_dict[gene]
        if not torch.is_tensor(vec):
            vec = torch.as_tensor(vec)
        vec = vec.reshape(-1)
        if vec.numel() == FM_DIM:
            fm_matrix[idx] = vec
fm_matrix = fm_matrix.to(DEVICE)
edge_index = safe_torch_load(PRIOR_EDGE_INDEX, map_location=DEVICE)

# Load pretrained encoder and a fresh baseline for this ablation
def get_predictions(use_pretrained):
    encoder = FusedGeneEncoder(fm_dim=FM_DIM, hidden_dim=HIDDEN).to(DEVICE)
    if use_pretrained:
        encoder.load_state_dict(safe_torch_load(PRETRAINED_CKPT, map_location=DEVICE))
    encoder.eval()
    with torch.no_grad():
        all_emb = encoder(fm_matrix, edge_index)

    head = nn.Sequential(
        nn.Linear(encoder.fused_dim * 2, encoder.fused_dim),
        nn.GELU(), nn.Linear(encoder.fused_dim, 1)
    ).to(DEVICE)

    scores = []
    for _, row in label_df.iterrows():
        tf_idx   = gene2idx[row["TF"]]
        gene_idx = gene2idx[row["Target"]]
        h = torch.cat([all_emb[tf_idx], all_emb[gene_idx]], dim=-1).unsqueeze(0)
        scores.append(torch.sigmoid(head(h)).item())
    return np.array(scores)

print("\nRunning ablation (may take a moment)...")
base_scores = get_predictions(use_pretrained=False)
pre_scores  = get_predictions(use_pretrained=True)

ablation_rows = []
for group_name, mask in [
    ("Overlap TFs (in Norman)",    label_df["TF_in_norman"].values),
    ("Non-overlap TFs (not Norman)", ~label_df["TF_in_norman"].values),
]:
    if mask.sum() < 10:
        continue
    y_true = label_df["Label"].values[mask]
    try:
        base_auroc = roc_auc_score(y_true, base_scores[mask])
        pre_auroc  = roc_auc_score(y_true, pre_scores[mask])
        base_auprc = average_precision_score(y_true, base_scores[mask])
        pre_auprc  = average_precision_score(y_true, pre_scores[mask])
    except Exception:
        continue
    ablation_rows.append({
        "Group": group_name,
        "n_edges": int(mask.sum()),
        "Baseline AUROC": round(base_auroc, 4),
        "Pretrained AUROC": round(pre_auroc, 4),
        "Delta AUROC": round(pre_auroc - base_auroc, 4),
        "Baseline AUPRC": round(base_auprc, 4),
        "Pretrained AUPRC": round(pre_auprc, 4),
        "Delta AUPRC": round(pre_auprc - base_auprc, 4),
    })

abl_df = pd.DataFrame(ablation_rows)
abl_df.to_csv(os.path.join(RESULTS_DIR, "overlap_ablation.csv"), index=False)
print("\nOverlap Ablation (key for paper):")
print(abl_df.to_string(index=False))
print(f"\nAll results in {RESULTS_DIR}/")
