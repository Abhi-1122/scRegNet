"""
Step 2: Pretrain the FusedGeneEncoder on Norman Perturb-seq triples.

Task: given (TF_embedding, Gene_embedding), predict whether the gene is
      0=no change / 1=upregulated / 2=downregulated when that TF is perturbed.

Loss: weighted cross-entropy (class 0 dominates, so we reweight).

After training, the encoder weights are saved to:
    ./checkpoints/pretrained_encoder.pt
"""

import os, pickle, numpy as np, pandas as pd, inspect
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from torch_geometric.data import Data

from encoder import FusedGeneEncoder   # from 02_encoder.py

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DATASET_DIR      = PROJECT_ROOT / "data" / "hESC" / "TFs+500"

TRIPLES_CSV      = SCRIPT_DIR / "perturb_triples" / "perturb_triples.csv"
GENE_LIST_PKL    = SCRIPT_DIR / "perturb_triples" / "gene_list.pkl"
FM_EMBEDDINGS    = DATASET_DIR / "fm_embeddings.pt"
PRIOR_EDGE_INDEX = SCRIPT_DIR / "BL-network_edge_index.pt"

CKPT_DIR  = SCRIPT_DIR / "checkpoints"
HIDDEN    = 256
EPOCHS    = 50
LR        = 1e-3
BATCH     = 512
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CKPT_DIR, exist_ok=True)


def safe_torch_load(path, map_location):
    """Load tensors with safer defaults across torch versions."""
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)

# ── 1. Load pretraining triples ───────────────────────────────────────────────
df = pd.read_csv(TRIPLES_CSV)
with open(GENE_LIST_PKL, "rb") as f:
    gene_list = pickle.load(f)
gene2idx = {g: i for i, g in enumerate(gene_list)}
print(f"Triples: {len(df)}, gene vocab: {len(gene_list)}")

required_cols = {"TF", "Gene", "Label"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in {TRIPLES_CSV}: {sorted(missing_cols)}")
if len(df) == 0:
    raise ValueError(
        "No perturbation triples found (0 rows). "
        "Run `python3 perturb_pretrain/01_prepare_perturb_data.py` to regenerate data, "
        "then rerun pretraining."
    )

# ── 2. Load FM embeddings ─────────────────────────────────────────────────────
print("Loading FM embeddings...")
fm_dict = safe_torch_load(FM_EMBEDDINGS, map_location="cpu")  # {gene: tensor}

if not isinstance(fm_dict, dict) or len(fm_dict) == 0:
    raise ValueError(f"Expected non-empty dict in {FM_EMBEDDINGS}, got {type(fm_dict)}")

sample_vec = next(iter(fm_dict.values()))
if not torch.is_tensor(sample_vec):
    sample_vec = torch.as_tensor(sample_vec)
if sample_vec.ndim != 1:
    sample_vec = sample_vec.reshape(-1)
FM_DIM = int(sample_vec.numel())
print(f"Detected FM embedding dimension: {FM_DIM}")

# Build embedding matrix for all genes in gene_list
# Genes missing from FM get zero embedding (rare — ESM-2 covers most human genes)
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

# ── 3. Load prior graph ───────────────────────────────────────────────────────
# If you haven't pre-built the edge_index, uncomment the block below:
#   import pandas as pd
#   bl = pd.read_csv("BL-network.csv")  # columns: TF, Target
#   src = [gene2idx[g] for g in bl["TF"]    if g in gene2idx]
#   dst = [gene2idx[g] for g in bl["Target"] if g in gene2idx]
#   edge_index = torch.tensor([src, dst], dtype=torch.long)
#   torch.save(edge_index, "./BL-network_edge_index.pt")

edge_index = safe_torch_load(PRIOR_EDGE_INDEX, map_location=DEVICE)

# ── 4. Dataset ────────────────────────────────────────────────────────────────
class PerturbDataset(Dataset):
    def __init__(self, df, gene2idx):
        self.tfs    = [gene2idx.get(g, 0) for g in df["TF"]]
        self.genes  = [gene2idx.get(g, 0) for g in df["Gene"]]
        self.labels = df["Label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tfs[i], self.genes[i], self.labels[i]

from sklearn.model_selection import train_test_split

n_samples = len(df)
if n_samples < 2:
    raise ValueError(f"Need at least 2 triples for train/val split, found {n_samples}.")

val_size = max(1, int(round(0.1 * n_samples)))
if val_size >= n_samples:
    val_size = 1

label_counts = df["Label"].value_counts()
can_stratify = (label_counts.min() >= 2) and (val_size >= label_counts.shape[0])

if can_stratify:
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["Label"], random_state=42
    )
else:
    print(
        "Warning: skipping stratified split due to small/imbalanced label counts; "
        "using random split instead."
    )
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)

train_ds = PerturbDataset(train_df, gene2idx)
val_ds   = PerturbDataset(val_df,   gene2idx)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

# ── 5. Model + pretraining head ───────────────────────────────────────────────
encoder = FusedGeneEncoder(fm_dim=FM_DIM, hidden_dim=HIDDEN).to(DEVICE)

# Pretraining head: (TF_fused || Gene_fused) → 3-class response
class PerturbHead(nn.Module):
    def __init__(self, fused_dim, n_classes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, n_classes)
        )
    def forward(self, h_tf, h_gene):
        return self.mlp(torch.cat([h_tf, h_gene], dim=-1))

head = PerturbHead(encoder.fused_dim).to(DEVICE)

# Class weights to handle imbalance (label 0 dominates)
counts = df["Label"].value_counts().reindex([0, 1, 2], fill_value=0).astype(float)
safe_counts = counts.clip(lower=1.0)
weights = 1.0 / safe_counts.values
weights = torch.tensor(weights / weights.sum(), dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(head.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── 6. Precompute all fused embeddings once per epoch ─────────────────────────
def get_all_embeddings():
    """Run encoder on ALL genes once; cache for fast batch lookup."""
    encoder.eval()
    with torch.no_grad():
        return encoder(fm_matrix, edge_index)  # (N_genes, fused_dim)

# ── 7. Training loop ──────────────────────────────────────────────────────────
best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    # Recompute embeddings each epoch (encoder weights change)
    all_emb = get_all_embeddings().detach()  # (N, fused_dim)

    encoder.train(); head.train()
    train_loss = 0.0
    for tf_idx, gene_idx, label in train_loader:
        tf_idx   = tf_idx.to(DEVICE)
        gene_idx = gene_idx.to(DEVICE)
        label    = label.to(DEVICE)

        h_tf   = all_emb[tf_idx]
        h_gene = all_emb[gene_idx]
        logits = head(h_tf, h_gene)
        loss   = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validation
    encoder.eval(); head.eval()
    all_emb = get_all_embeddings().detach()
    preds, truths = [], []
    with torch.no_grad():
        for tf_idx, gene_idx, label in val_loader:
            tf_idx   = tf_idx.to(DEVICE)
            gene_idx = gene_idx.to(DEVICE)
            logits   = head(all_emb[tf_idx], all_emb[gene_idx])
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(label.tolist())

    val_f1 = f1_score(truths, preds, average="macro", zero_division=0)
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val macro-F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, "pretrained_encoder.pt"))
        print(f"  → Saved best encoder (F1={best_val_f1:.4f})")

print(f"\nPretraining done. Best val macro-F1: {best_val_f1:.4f}")
print(f"Pretrained weights: {CKPT_DIR}/pretrained_encoder.pt")

# Full classification report on val
print("\nFinal validation report:")
print(classification_report(
    truths,
    preds,
    labels=[0, 1, 2],
    target_names=["No change", "Up", "Down"],
    zero_division=0,
))
