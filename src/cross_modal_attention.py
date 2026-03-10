import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing scFM embeddings and GNN embeddings.
    Allows bidirectional information flow between foundation model and graph features.
    """
    def __init__(self, scfm_dim, gnn_dim, hidden_dim, num_heads=4, dropout=0.1):
        """
        Args:
            scfm_dim: Dimension of scFM (foundation model) embeddings
            gnn_dim: Dimension of GNN embeddings
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossModalAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Project scFM embeddings to query space
        self.scfm_to_q = nn.Linear(scfm_dim, hidden_dim)
        self.scfm_to_k = nn.Linear(scfm_dim, hidden_dim)
        self.scfm_to_v = nn.Linear(scfm_dim, hidden_dim)
        
        # Project GNN embeddings to query space
        self.gnn_to_q = nn.Linear(gnn_dim, hidden_dim)
        self.gnn_to_k = nn.Linear(gnn_dim, hidden_dim)
        self.gnn_to_v = nn.Linear(gnn_dim, hidden_dim)
        
        # Output projections
        self.scfm_out = nn.Linear(hidden_dim, scfm_dim)
        self.gnn_out = nn.Linear(hidden_dim, gnn_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_scfm = nn.LayerNorm(scfm_dim)
        self.layer_norm_gnn = nn.LayerNorm(gnn_dim)
        
    def split_heads(self, x):
        """Split embedding into multiple heads"""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x):
        """Combine multiple heads back"""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
    
    def attention(self, query, key, value, mask=None):
        """
        Scaled dot-product attention
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, scfm_embed, gnn_embed):
        """
        Args:
            scfm_embed: (batch_size, scfm_dim) - Foundation model embeddings
            gnn_embed: (batch_size, gnn_dim) - GNN embeddings
            
        Returns:
            fused_scfm: Enhanced scFM embeddings with GNN information
            fused_gnn: Enhanced GNN embeddings with scFM information
        """
        # Add sequence dimension if needed (for attention mechanism)
        if len(scfm_embed.shape) == 2:
            scfm_embed = scfm_embed.unsqueeze(1)  # (batch, 1, scfm_dim)
            gnn_embed = gnn_embed.unsqueeze(1)    # (batch, 1, gnn_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # ===== Cross-Attention 1: scFM queries GNN =====
        # scFM embeddings ask: "What graph topology info is relevant to me?"
        scfm_q = self.split_heads(self.scfm_to_q(scfm_embed))  # Queries from scFM
        gnn_k = self.split_heads(self.gnn_to_k(gnn_embed))     # Keys from GNN
        gnn_v = self.split_heads(self.gnn_to_v(gnn_embed))     # Values from GNN
        
        scfm_attended, _ = self.attention(scfm_q, gnn_k, gnn_v)
        scfm_attended = self.combine_heads(scfm_attended)
        scfm_attended = self.scfm_out(scfm_attended)
        
        # Residual connection + Layer Norm
        scfm_enhanced = self.layer_norm_scfm(scfm_embed + self.dropout(scfm_attended))
        
        # ===== Cross-Attention 2: GNN queries scFM =====
        # GNN embeddings ask: "What foundation model info is relevant to me?"
        gnn_q = self.split_heads(self.gnn_to_q(gnn_embed))     # Queries from GNN
        scfm_k = self.split_heads(self.scfm_to_k(scfm_embed))  # Keys from scFM
        scfm_v = self.split_heads(self.scfm_to_v(scfm_embed))  # Values from scFM
        
        gnn_attended, _ = self.attention(gnn_q, scfm_k, scfm_v)
        gnn_attended = self.combine_heads(gnn_attended)
        gnn_attended = self.gnn_out(gnn_attended)
        
        # Residual connection + Layer Norm
        gnn_enhanced = self.layer_norm_gnn(gnn_embed + self.dropout(gnn_attended))
        
        # Remove sequence dimension if we added it
        if squeeze_output:
            scfm_enhanced = scfm_enhanced.squeeze(1)
            gnn_enhanced = gnn_enhanced.squeeze(1)
        
        return scfm_enhanced, gnn_enhanced


class IterativeCrossModalFusion(nn.Module):
    """
    Multiple layers of cross-modal attention for deeper fusion
    """
    def __init__(self, scfm_dim, gnn_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, fusion_mode="concat"):
        super(IterativeCrossModalFusion, self).__init__()
        
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode
        
        self.attention_layers = nn.ModuleList([
            CrossModalAttention(scfm_dim, gnn_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final fusion: concatenate, add, or gated
        if self.fusion_mode == "concat":
            self.final_dim = scfm_dim + gnn_dim
        elif self.fusion_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(scfm_dim + gnn_dim, scfm_dim + gnn_dim),
                nn.Sigmoid()
            )
            self.final_dim = scfm_dim + gnn_dim
        else:  # add
            # For add, we need to project to same dimension
            self.scfm_proj = nn.Linear(scfm_dim, max(scfm_dim, gnn_dim))
            self.gnn_proj = nn.Linear(gnn_dim, max(scfm_dim, gnn_dim))
            self.final_dim = max(scfm_dim, gnn_dim)
    
    def forward(self, scfm_embed, gnn_embed):
        """
        Args:
            scfm_embed: Foundation model embeddings
            gnn_embed: GNN embeddings
            
        Returns:
            fused_embed: Deeply fused embeddings
        """
        # Iteratively refine both embeddings through cross-modal attention
        for layer in self.attention_layers:
            scfm_embed, gnn_embed = layer(scfm_embed, gnn_embed)
        
        # Final fusion
        if self.fusion_mode == "concat":
            fused = torch.cat([scfm_embed, gnn_embed], dim=-1)
        elif self.fusion_mode == "gated":
            concat = torch.cat([scfm_embed, gnn_embed], dim=-1)
            gate = self.gate(concat)
            fused = gate * concat
        else:  # add
            scfm_proj = self.scfm_proj(scfm_embed)
            gnn_proj = self.gnn_proj(gnn_embed)
            fused = scfm_proj + gnn_proj
        
        return fused
    
    def get_output_dim(self):
        return self.final_dim
