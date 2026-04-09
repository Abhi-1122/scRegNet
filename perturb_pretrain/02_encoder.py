"""
Shared encoder architecture used in both pretraining and fine-tuning.
Mirrors scRegNet's FM + GCN design but is standalone.

Architecture:
  GeneEncoder:  Linear projection of scFM embeddings → hidden_dim
  GCNEncoder:   2-layer GCN over prior TF-target graph → hidden_dim
  Fusion:       concat(gene_emb, gcn_emb) → fused_dim  (simple concat for now;
                swap in cross-attention if you want Idea #3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GeneEncoder(nn.Module):
    """Projects raw scFM embeddings (e.g. ESM-2, 1280-d) → hidden_dim."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GCNEncoder(nn.Module):
    """2-layer GCN over TF-target prior graph → hidden_dim per node."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim * 2)
        self.conv2 = GCNConv(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h


class FusedGeneEncoder(nn.Module):
    """
    Full shared encoder: scFM features + GCN graph features → fused per-gene embedding.

    Input:
        x_fm:        (N_genes, fm_dim)     — scFM embeddings for all genes
        edge_index:  (2, E)                — TF-target prior graph edges
    Output:
        (N_genes, hidden_dim * 2)          — fused gene embeddings
    """
    def __init__(self,
                 fm_dim: int    = 1280,
                 hidden_dim: int = 256,
                 dropout: float  = 0.2):
        super().__init__()
        self.gene_enc  = GeneEncoder(fm_dim,    hidden_dim, dropout)
        self.gcn_enc   = GCNEncoder(fm_dim,     hidden_dim, dropout)
        self.fused_dim = hidden_dim * 2

    def forward(self,
                x_fm: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h_fm  = self.gene_enc(x_fm)          # (N, hidden_dim)
        h_gcn = self.gcn_enc(x_fm, edge_index)  # (N, hidden_dim)
        return torch.cat([h_fm, h_gcn], dim=-1)  # (N, hidden_dim*2)


class CrossAttentionFusedEncoder(nn.Module):
    """
    Optional drop-in replacement for FusedGeneEncoder.
    Uses cross-attention instead of concat to fuse FM and GCN embeddings.
    (Idea #3 from the previous discussion)
    """
    def __init__(self,
                 fm_dim: int     = 1280,
                 hidden_dim: int  = 256,
                 n_heads: int     = 4,
                 dropout: float   = 0.2):
        super().__init__()
        self.gene_enc  = GeneEncoder(fm_dim, hidden_dim, dropout)
        self.gcn_enc   = GCNEncoder(fm_dim,  hidden_dim, dropout)
        self.fused_dim = hidden_dim

        # FM queries attend over GCN keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x_fm: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h_fm  = self.gene_enc(x_fm)              # (N, H)
        h_gcn = self.gcn_enc(x_fm, edge_index)   # (N, H)

        # Add batch dimension for MHA: (1, N, H)
        q = h_fm.unsqueeze(0)
        k = h_gcn.unsqueeze(0)
        attn_out, _ = self.cross_attn(q, k, k)  # (1, N, H)
        fused = self.norm(attn_out.squeeze(0) + h_fm)  # residual
        return fused  # (N, H)
