import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphFusion(nn.Module):
    """
    Generic fusion of two same-dim node embeddings (e.g. GRN-GNN and PPI-GNN
    embeddings), used to build embed_graph = fuse(embed_grn, embed_ppi) before
    it is passed downstream into the existing scFM-fusion + MLP pipeline.

    fusion_type:
        "concat"    - torch.cat, output dim = dim_a + dim_b
        "gate"      - learnable elementwise gate between a and b, output dim = dim
        "attention" - single/multi-head scaled dot-product cross-attention
                      (query=a, key/value=b) + residual + LayerNorm, output dim = dim
    """

    def __init__(self, dim_a: int, dim_b: int, fusion_type: str = "concat", num_heads: int = 1):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self._output_dim = dim_a + dim_b

        elif fusion_type == "gate":
            self._output_dim = dim_a
            if dim_b != dim_a:
                self.proj_b = nn.Linear(dim_b, dim_a)
            else:
                self.proj_b = nn.Identity()
            self.gate = nn.Linear(dim_a + dim_a, dim_a)

        elif fusion_type == "attention":
            assert dim_a % num_heads == 0, (
                f"dim_a ({dim_a}) must be divisible by num_heads ({num_heads})"
            )
            self._output_dim = dim_a
            self.num_heads = num_heads
            self.head_dim = dim_a // num_heads
            self.scale = self.head_dim ** -0.5

            if dim_b != dim_a:
                self.proj_b = nn.Linear(dim_b, dim_a)
            else:
                self.proj_b = nn.Identity()

            self.q_proj = nn.Linear(dim_a, dim_a)
            self.k_proj = nn.Linear(dim_a, dim_a)
            self.v_proj = nn.Linear(dim_a, dim_a)
            self.out_proj = nn.Linear(dim_a, dim_a)
            self.norm_out = nn.LayerNorm(dim_a)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, embed_a: torch.Tensor, embed_b: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat((embed_a, embed_b), dim=1)

        elif self.fusion_type == "gate":
            b = self.proj_b(embed_b)
            g = torch.sigmoid(self.gate(torch.cat((embed_a, b), dim=1)))
            return g * embed_a + (1 - g) * b

        elif self.fusion_type == "attention":
            N = embed_a.size(0)
            b = self.proj_b(embed_b)

            Q = self.q_proj(embed_a)
            K = self.k_proj(b)
            V = self.v_proj(b)

            def split_heads(x):
                return x.view(N, self.num_heads, self.head_dim).transpose(0, 1)

            Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

            attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)

            attn_out = torch.bmm(attn_weights, V)
            attn_out = attn_out.transpose(0, 1).contiguous().view(N, self._output_dim)
            attn_out = self.out_proj(attn_out)

            return self.norm_out(embed_a + attn_out)
