import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv

from src.args import load_args
from src.utils import adj2saprse_tensor, load_data, scRNADataset


class TripletDataset(Dataset):
    def __init__(self, tf_idx, gene_idx, labels):
        self.tf_idx = tf_idx
        self.gene_idx = gene_idx
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.tf_idx[index],
            self.gene_idx[index],
            self.labels[index],
        )


class ScRegNetGCNPretrainer(nn.Module):
    def __init__(self, input_dim, gnn_hidden_dims, dropout=0.2, num_classes=3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        cur_dim = input_dim
        for hidden_dim in gnn_hidden_dims:
            self.convs.append(GCNConv(cur_dim, hidden_dim))
            cur_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(cur_dim * 2, cur_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cur_dim, num_classes),
        )

    def encode(self, x, adj):
        for index, conv in enumerate(self.convs):
            x = conv(x, adj)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, adj, tf_idx, gene_idx):
        emb = self.encode(x, adj)
        tf_emb = emb[tf_idx]
        gene_emb = emb[gene_idx]
        logits = self.head(torch.cat([tf_emb, gene_emb], dim=-1))
        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_args_json",
        type=str,
        default="./out/GCN/Geneformer/tf_500_hESC/best/ckpt/args.json",
    )
    parser.add_argument(
        "--triples_csv",
        type=str,
        default="./perturb_pretrain/perturb_triples/perturb_triples.csv",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./perturb_pretrain/checkpoints/pretrained_scregnet_gcn.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    return parser.parse_args()


def load_base_args(args_json_path):
    if not os.path.exists(args_json_path):
        raise FileNotFoundError(f"base args json not found: {args_json_path}")
    args_dir = str(Path(args_json_path).resolve().parent)
    base_args = load_args(args_dir)
    return base_args


def prepare_graph_and_features(base_args, device):
    data_dir = os.path.join(
        base_args.data_folder,
        f"{base_args.cell_type}/TFs+{base_args.num_TF}",
    )

    exp_file = os.path.join(data_dir, "BL--ExpressionData.csv")
    train_file = os.path.join(data_dir, "Train_set.csv")
    tf_file = os.path.join(data_dir, "TF.csv")

    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature2 = loader.exp_data()
    feature2 = torch.from_numpy(feature2).float().to(device)

    gene_num = feature2.shape[0]
    train_data = pd.read_csv(train_file, index_col=0).values

    tf_indices = pd.read_csv(tf_file, index_col=0)["index"].values.astype(np.int64)
    tf_tensor = torch.from_numpy(tf_indices).to(device)

    train_set = scRNADataset(train_data, gene_num, flag=bool(base_args.flag))
    adj = train_set.Adj_Generate(tf_tensor, loop=bool(base_args.loop))
    adj = adj2saprse_tensor(adj).to(device)

    gene_to_idx = {str(gene): idx for idx, gene in enumerate(data_input.index.astype(str))}
    return feature2, adj, gene_to_idx


def load_triples(triples_csv, gene_to_idx):
    if not os.path.exists(triples_csv):
        raise FileNotFoundError(f"triples csv not found: {triples_csv}")

    triples_df = pd.read_csv(triples_csv)
    required = {"TF", "Gene", "Label"}
    missing = required - set(triples_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {triples_csv}: {sorted(missing)}")

    triples_df = triples_df[triples_df["Label"].isin([0, 1, 2])].copy()
    triples_df["TF"] = triples_df["TF"].astype(str)
    triples_df["Gene"] = triples_df["Gene"].astype(str)

    triples_df["tf_idx"] = triples_df["TF"].map(gene_to_idx)
    triples_df["gene_idx"] = triples_df["Gene"].map(gene_to_idx)

    before = len(triples_df)
    triples_df = triples_df.dropna(subset=["tf_idx", "gene_idx"]).copy()
    triples_df["tf_idx"] = triples_df["tf_idx"].astype(np.int64)
    triples_df["gene_idx"] = triples_df["gene_idx"].astype(np.int64)

    kept = len(triples_df)
    if kept == 0:
        raise ValueError("No perturbation triples overlap with dataset genes.")

    print(f"Triples loaded: kept {kept}/{before} rows after gene-index overlap filtering")
    return triples_df


def main():
    cli_args = parse_cli_args()
    base_args = load_base_args(cli_args.base_args_json)

    set_seed(cli_args.seed)

    if torch.cuda.is_available():
        gpu_id = base_args.single_gpu if base_args.single_gpu < torch.cuda.device_count() else 0
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Architecture: gnn_hidden_dims={base_args.gnn_hidden_dims}, dropout={base_args.dropout}")

    feature2, adj, gene_to_idx = prepare_graph_and_features(base_args, device)
    triples_df = load_triples(cli_args.triples_csv, gene_to_idx)

    tf_idx = triples_df["tf_idx"].to_numpy()
    gene_idx = triples_df["gene_idx"].to_numpy()
    labels = triples_df["Label"].to_numpy(dtype=np.int64)

    n_samples = len(labels)
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for train/val split; found {n_samples}")

    val_size = max(1, int(round(cli_args.val_ratio * n_samples)))
    if val_size >= n_samples:
        val_size = 1

    label_counts = pd.Series(labels).value_counts()
    can_stratify = (label_counts.min() >= 2) and (val_size >= len(label_counts))

    split_kwargs = {"test_size": val_size, "random_state": cli_args.seed}
    if can_stratify:
        split_kwargs["stratify"] = labels

    tf_train, tf_val, gene_train, gene_val, y_train, y_val = train_test_split(
        tf_idx,
        gene_idx,
        labels,
        **split_kwargs,
    )

    train_ds = TripletDataset(
        torch.tensor(tf_train, dtype=torch.long),
        torch.tensor(gene_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TripletDataset(
        torch.tensor(tf_val, dtype=torch.long),
        torch.tensor(gene_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cli_args.batch_size,
        shuffle=True,
        num_workers=cli_args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ScRegNetGCNPretrainer(
        input_dim=feature2.shape[1],
        gnn_hidden_dims=base_args.gnn_hidden_dims,
        dropout=base_args.dropout,
        num_classes=3,
    ).to(device)

    counts = pd.Series(y_train).value_counts().reindex([0, 1, 2], fill_value=0).astype(float)
    safe_counts = counts.clip(lower=1.0)
    class_weights = (1.0 / safe_counts.values)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = cli_args.lr if cli_args.lr is not None else base_args.gnn_lr
    weight_decay = (
        cli_args.weight_decay if cli_args.weight_decay is not None else base_args.gnn_weight_decay
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli_args.epochs)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, cli_args.epochs + 1):
        model.train()
        running_loss = 0.0

        for tf_batch, gene_batch, y_batch in train_loader:
            tf_batch = tf_batch.to(device)
            gene_batch = gene_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(feature2, adj, tf_batch, gene_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for tf_batch, gene_batch, y_batch in val_loader:
                tf_batch = tf_batch.to(device)
                gene_batch = gene_batch.to(device)
                logits = model(feature2, adj, tf_batch, gene_batch)
                pred = torch.argmax(logits, dim=-1).cpu().numpy()
                preds.extend(pred.tolist())
                truths.extend(y_batch.numpy().tolist())

        val_f1 = f1_score(truths, preds, average="macro", zero_division=0)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:03d}/{cli_args.epochs} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "state_dict": model.state_dict(),
                "meta": {
                    "input_dim": int(feature2.shape[1]),
                    "gnn_hidden_dims": [int(v) for v in base_args.gnn_hidden_dims],
                    "dropout": float(base_args.dropout),
                    "best_val_macro_f1": float(best_f1),
                },
                "source_args_json": cli_args.base_args_json,
            }

    os.makedirs(os.path.dirname(cli_args.save_path), exist_ok=True)
    torch.save(best_state, cli_args.save_path)
    print(f"Saved pretrained scRegNet GCN checkpoint: {cli_args.save_path}")
    print(f"Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
