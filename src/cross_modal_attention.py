import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignedCrossAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion for scRegNet.

    Design (VLM-inspired):
        1. Align GNN prior embeddings (128-dim) → scFM space (896-dim)
           via a learned projection. Never compress the richer modality down.
        2. scFM is Query  → scFM gets updated/grounded by graph structure
           GNN-aligned is Key/Value → graph topology is the information source
        3. Residual connection back onto scFM preserves pre-trained geometry.
           Output is a single 896-dim vector — attention IS the fusion, no concat needed.
        4. Optional final projection to a smaller output_dim for MLP head.

    Args:
        scfm_dim:   Dimension of scFM embeddings (896)
        gnn_dim:    Dimension of GNN prior embeddings (128)
        num_heads:  Number of attention heads (scfm_dim must be divisible)
        dropout:    Dropout rate
        output_dim: If set, project fused 896-dim output to this dim for the MLP.
                    If None, output stays at scfm_dim (896).
    """

    def __init__(
        self,
        scfm_dim: int,
        gnn_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = None,
    ):
        super().__init__()
        assert scfm_dim % num_heads == 0, (
            f"scfm_dim ({scfm_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.scfm_dim = scfm_dim
        self.num_heads = num_heads
        self.head_dim = scfm_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --- Step 1: Alignment --- #
        # Lift GNN (128) → scFM space (896). Never go the other direction.
        self.gnn_align = nn.Sequential(
            nn.Linear(gnn_dim, scfm_dim),
            nn.LayerNorm(scfm_dim),
            nn.GELU(),
        )

        # --- Step 2: Cross-Attention projections (all in scfm_dim space) --- #
        self.q_proj = nn.Linear(scfm_dim, scfm_dim)  # Q from scFM
        self.k_proj = nn.Linear(scfm_dim, scfm_dim)  # K from aligned GNN
        self.v_proj = nn.Linear(scfm_dim, scfm_dim)  # V from aligned GNN
        self.out_proj = nn.Linear(scfm_dim, scfm_dim)

        self.dropout = nn.Dropout(dropout)

        # Pre-norm on Q (scFM) before attention
        self.norm_q = nn.LayerNorm(scfm_dim)
        # Post-attention residual norm
        self.norm_out = nn.LayerNorm(scfm_dim)

        # FFN after attention (standard transformer block pattern)
        self.ffn = nn.Sequential(
            nn.Linear(scfm_dim, scfm_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scfm_dim * 2, scfm_dim),
        )
        self.norm_ffn = nn.LayerNorm(scfm_dim)

        # --- Step 4: Optional output projection --- #
        self._output_dim = output_dim if output_dim is not None else scfm_dim
        if output_dim is not None and output_dim != scfm_dim:
            self.final_proj = nn.Linear(scfm_dim, output_dim)
        else:
            self.final_proj = nn.Identity()

    def forward(self, scfm_embed: torch.Tensor, gnn_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scfm_embed: (N, scfm_dim=896)  — scFM gene embeddings
            gnn_embed:  (N, gnn_dim=128)   — GNN prior graph embeddings
        Returns:
            fused:      (N, output_dim)    — scFM grounded in graph topology
        """
        N = scfm_embed.size(0)

        # Step 1: Align GNN → scFM space
        gnn_aligned = self.gnn_align(gnn_embed)  # (N, 896)

        # Step 2: Multi-head cross-attention
        # Q: scFM (pre-normed), K/V: aligned GNN
        Q = self.q_proj(self.norm_q(scfm_embed))  # (N, 896)
        K = self.k_proj(gnn_aligned)               # (N, 896)
        V = self.v_proj(gnn_aligned)               # (N, 896)

        # Reshape for multi-head: (num_heads, N, head_dim)
        def split_heads(x):
            return x.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention: (num_heads, N, N)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (num_heads, N, head_dim) → (N, 896)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = attn_out.transpose(0, 1).contiguous().view(N, self.scfm_dim)
        attn_out = self.out_proj(attn_out)

        # Step 3: Residual onto scFM — graph info absorbed into transcript embedding
        x = self.norm_out(scfm_embed + self.dropout(attn_out))

        # FFN block (standard post-attention)
        x = self.norm_ffn(x + self.dropout(self.ffn(x)))

        # Step 4: Optional compression for MLP head
        return self.final_proj(x)  # (N, output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim