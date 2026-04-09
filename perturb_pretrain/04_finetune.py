"""
Step 3: Fine-tune the pretrained encoder on BEELINE GRN task.

Two models are trained for a fair ablation:
  (A) Baseline  — encoder initialized randomly    (standard scRegNet)
  (B) Pretrained — encoder loaded from pretraining (our new model)

Both use identical architecture and hyperparameters.
Results are saved to ./results/
"""

import os, torch, numpy as np, pandas as pd
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from encoder import FusedGeneEncoder   # from 02_encoder.py

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DATASET_DIR      = PROJECT_ROOT / "data" / "hESC" / "TFs+500"

LABEL_CSV        = DATASET_DIR / "Label.csv"
TRAIN_SET_CSV    = DATASET_DIR / "Train_set.csv"
TEST_SET_CSV     = DATASET_DIR / "Test_set.csv"
TARGET_CSV       = DATASET_DIR / "Target.csv"

FM_EMBEDDINGS    = DATASET_DIR / "fm_embeddings.pt"
PRIOR_EDGE_INDEX = SCRIPT_DIR / "BL-network_edge_index.pt"
GENE2IDX_PKL     = SCRIPT_DIR / "perturb_triples" / "gene_list.pkl"

PRETRAINED_CKPT  = SCRIPT_DIR / "checkpoints" / "pretrained_encoder.pt"
RESULTS_DIR      = SCRIPT_DIR / "results"

HIDDEN   = 256
EPOCHS   = 100
LR       = 5e-4
BATCH    = 256
N_FOLDS  = 5
FEW_SHOT_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 1.0]
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)


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

# ── 1. Load data ──────────────────────────────────────────────────────────────
import pickle
with open(GENE2IDX_PKL, "rb") as f:
    gene_list = pickle.load(f)
gene2idx = {g: i for i, g in enumerate(gene_list)}

# Extend vocab with any BEELINE genes missing from pretraining vocab
label_df = pd.read_csv(LABEL_CSV)
if "TF" not in label_df.columns:
    label_df.columns = ["TF", "Target", "Label"]

extra_genes = (set(label_df["TF"].astype(str)) | set(label_df["Target"].astype(str))) - set(gene_list)
if extra_genes:
    print(f"Adding {len(extra_genes)} BEELINE genes missing from pretraining vocab")
    gene_list = sorted(set(gene_list) | extra_genes)
    gene2idx = {g: i for i, g in enumerate(gene_list)}

df = load_beeline_edges()

# Filter to genes present in vocab
df = df[df["TF"].isin(gene2idx) & df["Target"].isin(gene2idx)].reset_index(drop=True)
print(f"Usable edges: {len(df)} | Positives: {df['Label'].sum()}")

# ── 2. FM embeddings ──────────────────────────────────────────────────────────
fm_dict = torch.load(FM_EMBEDDINGS, map_location="cpu")
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

edge_index = torch.load(PRIOR_EDGE_INDEX, map_location=DEVICE)

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class GRNDataset(Dataset):
    def __init__(self, df, gene2idx):
        self.tfs    = [gene2idx[g] for g in df["TF"]]
        self.genes  = [gene2idx[g] for g in df["Target"]]
        self.labels = df["Label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tfs[i], self.genes[i], self.labels[i]

# ── 4. GRN prediction head (identical for both models) ────────────────────────
class GRNHead(nn.Module):
    def __init__(self, fused_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, 1)
        )
    def forward(self, h_tf, h_gene):
        return self.mlp(torch.cat([h_tf, h_gene], dim=-1)).squeeze(-1)

# ── 5. Training function ──────────────────────────────────────────────────────
def train_model(train_ds, val_ds, pretrained: bool):
    encoder = FusedGeneEncoder(fm_dim=FM_DIM, hidden_dim=HIDDEN).to(DEVICE)
    head    = GRNHead(encoder.fused_dim).to(DEVICE)

    if pretrained:
        encoder.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=DEVICE))
        print("  → Loaded pretrained encoder weights")
    else:
        print("  → Random init (baseline)")

    pos_weight = torch.tensor(
        [(1 - df["Label"].mean()) / df["Label"].mean()], dtype=torch.float
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    best_auroc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # -- get all embeddings once per epoch --
        encoder.eval()
        with torch.no_grad():
            all_emb = encoder(fm_matrix, edge_index).detach()

        encoder.train(); head.train()
        for tf_idx, gene_idx, label in train_loader:
            tf_idx   = tf_idx.to(DEVICE)
            gene_idx = gene_idx.to(DEVICE)
            label    = label.float().to(DEVICE)
            h_tf   = all_emb[tf_idx]
            h_gene = all_emb[gene_idx]
            logits = head(h_tf, h_gene)
            loss   = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

        # -- validation --
        encoder.eval(); head.eval()
        with torch.no_grad():
            all_emb = encoder(fm_matrix, edge_index).detach()
        preds, truths = [], []
        with torch.no_grad():
            for tf_idx, gene_idx, label in val_loader:
                tf_idx   = tf_idx.to(DEVICE)
                gene_idx = gene_idx.to(DEVICE)
                logits   = head(all_emb[tf_idx], all_emb[gene_idx])
                preds.extend(torch.sigmoid(logits).cpu().tolist())
                truths.extend(label.tolist())

        try:
            auroc = roc_auc_score(truths, preds)
            auprc = average_precision_score(truths, preds)
        except Exception:
            auroc = auprc = 0.0

        scheduler.step(auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            best_auprc = auprc
            best_state = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "head":    {k: v.clone() for k, v in head.state_dict().items()},
            }

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    return best_auroc, best_auprc, best_state

# ── 6. Cross-validation loop ──────────────────────────────────────────────────
X = np.arange(len(df))
y = df["Label"].values

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = {"baseline": [], "pretrained": []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n=== Fold {fold}/{N_FOLDS} ===")
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]
    val_ds   = GRNDataset(val_df,   gene2idx)

    for model_name, use_pretrained in [("baseline", False), ("pretrained", True)]:
        print(f"\n  [{model_name}]")
        for fraction in FEW_SHOT_FRACTIONS:
            n_sub = max(1, int(round(len(train_df) * fraction)))
            train_df_sub = train_df.sample(n=n_sub, random_state=42)
            train_ds_sub = GRNDataset(train_df_sub, gene2idx)
            print(f"    fraction={fraction:.2f} | n_train={len(train_df_sub)}")
            auroc, auprc, _ = train_model(train_ds_sub, val_ds, pretrained=use_pretrained)
            results[model_name].append({
                "fold": fold,
                "fraction": fraction,
                "AUROC": auroc,
                "AUPRC": auprc,
            })
            print(f"    Best → AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

# ── 7. Save + report ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS (Few-shot + Full-data, 5-fold CV)")
print("="*60)
for model_name in ["baseline", "pretrained"]:
    r = pd.DataFrame(results[model_name])
    print(f"\n  {model_name.upper()}")
    for fraction in FEW_SHOT_FRACTIONS:
        rf = r[r["fraction"] == fraction]
        if len(rf) == 0:
            continue
        auroc_mean, auroc_std = rf["AUROC"].mean(), rf["AUROC"].std()
        auprc_mean, auprc_std = rf["AUPRC"].mean(), rf["AUPRC"].std()
        print(
            f"    frac={fraction:.2f} | "
            f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f} | "
            f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}"
        )

    r.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_fewshot_results.csv"), index=False)

    # Keep legacy outputs (used by 05_evaluate.py) for full-data setting only
    r_full = r[r["fraction"] == 1.0][["fold", "AUROC", "AUPRC"]].copy()
    r_full.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_cv_results.csv"), index=False)

print(f"\nResults saved to {RESULTS_DIR}/")
