"""
spatial_gnn.py
──────────────
Phase 1 of scRegNet-Spatial.

Provides two modules:
  SpatialGNN    – 2-layer GraphSAGE on the Visium spot kNN graph.
                  Message-passes spatial context across neighbouring spots
                  so each spot embedding reflects its liver niche.

  AttentionPool – learnable weighted pooling of spot embeddings per gene.
                  Replaces the implicit mean pooling used in the original
                  scRegNet (raw expression matrix as GCN node features).

Typical data shapes for hHEP (933 genes, 7982 spots, k=6 neighbours):
  spot_feat    : (7982, 933)   – standardised gene expression per spot
  edge_index   : (2,  47892)  – kNN spatial graph in PyG format
  edge_weight  : (47892,)     – inverse-distance weights
  spot_emb     : (7982, D)    – after SpatialGNN
  gene_emb     : (933,  D)    – after AttentionPool
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# SpatialGNN
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGNN(nn.Module):
    """
    Two-layer GraphSAGE that propagates information across spatially adjacent
    spots on the Visium hexagonal grid.

    Args:
        in_channels  (int): Input feature dim per spot (= number of genes).
        hidden_channels (int): Hidden dim of first SAGE layer.
        out_channels (int): Output spot-embedding dim.
        dropout      (float): Dropout probability between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.out_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,           # (N_spots, in_channels)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor = None,  # (E,) – currently unused by SAGEConv
    ) -> torch.Tensor:             # (N_spots, out_channels)
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        return x  # (N_spots, out_channels)


# ─────────────────────────────────────────────────────────────────────────────
# AttentionPool
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Learnable attention pooling: for each gene, aggregate the embeddings of
    the spots that express it into a single gene-level embedding.

    The attention score for spot s given gene g is:
        score(s) = v^T · tanh(W · h_s)
    where h_s is the spatial embedding of spot s.

    Args:
        embed_dim (int): Dimensionality of spot (and gene) embeddings.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v = nn.Linear(embed_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(
        self,
        spot_emb: torch.Tensor,         # (N_spots, D)
        gene_spot_mask: list,           # list[N_genes] of LongTensor spot indices
    ) -> torch.Tensor:                  # (N_genes, D)
        """
        Vectorized attention pooling.

        1. Compute attention logits for EVERY spot in one pass (no Python loop).
        2. Build a dense (N_genes, N_spots) logit matrix, masked to -inf for
           non-expressing (spot, gene) pairs.
        3. Row-wise softmax → one matmul gives all gene embeddings at once.
        """
        N_spots, D = spot_emb.shape
        N_genes    = len(gene_spot_mask)
        device     = spot_emb.device

        # All-spot attention logits in a single forward pass
        logits_all = self.v(torch.tanh(self.W(spot_emb))).squeeze(-1)  # (N_spots,)

        # Build index tensors for non-empty genes
        nonempty = [(g, m) for g, m in enumerate(gene_spot_mask) if m.numel() > 0]

        if nonempty:
            gene_ids = torch.cat([
                torch.full((m.numel(),), g, dtype=torch.long, device=device)
                for g, m in nonempty
            ])                                                   # (total_expressing,)
            spot_ids = torch.cat([m for _, m in nonempty])      # (total_expressing,)

            # Dense score matrix: -inf so softmax ignores non-expressing spots
            score_mat = torch.full((N_genes, N_spots), float('-inf'), device=device)
            score_mat[gene_ids, spot_ids] = logits_all[spot_ids]

            # Genes with no expressing spots: row is all -inf → softmax gives NaN.
            # Use a zero row instead for those genes (no in-place op).
            has_expr = torch.tensor(
                [m.numel() > 0 for m in gene_spot_mask],
                dtype=torch.bool, device=device
            )                                                    # (N_genes,)
            raw_weights = torch.softmax(score_mat, dim=1)       # (N_genes, N_spots)
            # all-inf rows (no expressing spots) → NaN after softmax; replace with 0
            raw_weights = torch.nan_to_num(raw_weights, nan=0.0)
            weights = raw_weights * has_expr.unsqueeze(1)       # zero rows for empty genes
        else:
            weights = torch.zeros(N_genes, N_spots, device=device)

        gene_embs = weights @ spot_emb                          # (N_genes, D)
        return gene_embs
