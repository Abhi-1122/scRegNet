import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from src.cross_modal_attention import IterativeCrossModalFusion
from src.spatial_gnn import SpatialGNN, AttentionPool


class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):
        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T,negative_slope=self.alpha)
        return e

    def forward(self,x,adj):
        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense()>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass

        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data = F.normalize(output_data,p=2,dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data


class scTransNet_GCN(nn.Module):
    def __init__(self,input_dim,args,gene_dim,device):
        super(scTransNet_GCN, self).__init__()
        self.args = args
        self.device = device

        self.convs = torch.nn.ModuleList()
        
        for i in range(self.args.gnn_num_layers):
            hidden_dim = self.args.gnn_hidden_dims[i]
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Store GNN output dimension
        self.gnn_output_dim = input_dim
        
        # Cross-Modal Attention Fusion
        self.use_cross_attention = getattr(self.args, 'use_cross_attention', False)
        
        if self.use_cross_attention:
            # Create cross-modal attention fusion module
            self.cross_modal_fusion = IterativeCrossModalFusion(
                scfm_dim=gene_dim,
                gnn_dim=self.gnn_output_dim,
                hidden_dim=getattr(self.args, 'cross_attention_hidden_dim', 256),
                num_layers=getattr(self.args, 'cross_attention_layers', 2),
                num_heads=getattr(self.args, 'cross_attention_heads', 4),
                dropout=self.args.dropout,
                fusion_mode=getattr(self.args, 'cross_attention_fusion_mode', 'concat')
            )
            fused_dim = self.cross_modal_fusion.get_output_dim()
        else:
            # Original concatenation
            fused_dim = gene_dim + self.gnn_output_dim
        
        # MLP layers for TF and Target embeddings
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]

            if i==0:
                self.layers.append(nn.Linear(fused_dim, hidden_dim_mlp))
            else:
                self.layers.append(nn.Linear(input_dim, hidden_dim_mlp))
            
            input_dim = hidden_dim_mlp

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2*input_dim, 2)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()
        
    def encode(self,x,adj):
        for i, conv in enumerate(self.convs):
            x = conv(x,adj)
            if i < len(self.convs) - 1:
                x = F.relu(x) 
                p = self.args.dropout
                x = F.dropout(x, p, training=self.training)
        return x 
        
    def decode(self,tf_embed,target_embed):
        if self.args.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)
            return prob
        elif self.args.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))
        
    def forward(self, x, adj, train_sample, llm_emb):
        # GNN encoding
        embed = self.encode(x, adj)
        
        # Cross-Modal Fusion: Replace simple concatenation with attention-based fusion
        if self.use_cross_attention:
            # Deep cross-modal attention fusion
            embed = self.cross_modal_fusion(llm_emb, embed)
        else:
            # Original simple concatenation
            embed = torch.cat((llm_emb, embed), dim=1)
        
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                tf_embed = F.dropout(tf_embed, p)
        
        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                target_embed = F.dropout(target_embed, p)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred
    
    def get_embedding(self):
        return self.tf_ouput, self.target_output


class scTransNet_SAGE(nn.Module):
    def __init__(self,input_dim,args,gene_dim,device):
        super(scTransNet_SAGE, self).__init__()
        self.args = args
        self.device = device

        self.convs = torch.nn.ModuleList()
        
        for i in range(self.args.gnn_num_layers):
            hidden_dim = self.args.gnn_hidden_dims[i]
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]

            if i==0:
                self.layers.append(nn.Linear(input_dim+gene_dim,hidden_dim_mlp))
            else:
                self.layers.append(nn.Linear(input_dim,hidden_dim_mlp))
            
            input_dim = hidden_dim_mlp

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2*input_dim, 2)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()

    def encode(self,x,adj):
        for i, conv in enumerate(self.convs):
            x = conv(x,adj)
            if i < len(self.convs) - 1:
                x = F.relu(x) 
                p = self.args.dropout
                x = F.dropout(x, p, training=self.training)
        return x 
        
    def decode(self,tf_embed,target_embed):
        if self.args.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)
            return prob
        elif self.args.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))
        
    def forward(self, x, adj, train_sample, llm_emb):
        embed = self.encode(x,adj)
        embed = torch.cat((llm_emb, embed), dim=1)
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                tf_embed = F.dropout(tf_embed, p)
        
        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                target_embed = F.dropout(target_embed, p)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred
    
    def get_embedding(self):
        return self.tf_ouput, self.target_output


class scTransNet_GAT(nn.Module):
    def __init__(self,input_dim,args,gene_dim,device):
        super(scTransNet_GAT, self).__init__()
        self.args = args
        self.device = device
        self.reduction = self.args.reduction

        self.convs = []
        gnn_num_layers = self.args.gnn_num_layers
        for i in range(gnn_num_layers):
            num_head = self.args.num_heads[i]
            hidden_dim = self.args.gnn_hidden_dims[i]

            conv_layer = [AttentionLayer(input_dim,hidden_dim,self.args.alpha) for _ in range(num_head)]
            self.convs.append(conv_layer)
            for j, attention in enumerate(conv_layer):
                self.add_module(f'ConvLayer{i}_AttentionHead{j}',attention)
            input_dim = hidden_dim

            if self.reduction == 'concate' and i<gnn_num_layers-1:
                input_dim = num_head*hidden_dim
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]

            if i==0:
                self.layers.append(nn.Linear(input_dim+gene_dim,hidden_dim_mlp))
            else:
                self.layers.append(nn.Linear(input_dim,hidden_dim_mlp))
            
            input_dim = hidden_dim_mlp

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2*input_dim, 2)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            for attention in conv:
                attention.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()

    def encode(self,x,adj):
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                out = torch.mean(torch.stack([att(x, adj) for att in conv]),dim=0)
                return out
            elif self.reduction =='concate':
                x = torch.cat([att(x, adj) for att in conv], dim=1)
            elif self.reduction =='mean':
                x = torch.mean(torch.stack([att(x, adj) for att in conv]), dim=0)
            else:
                raise TypeError
            
            x = F.elu(x)
        
    def decode(self,tf_embed,target_embed):
        if self.args.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)
            return prob
        elif self.args.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))
        
    def forward(self, x, adj, train_sample, llm_emb):
        embed = self.encode(x,adj)
        embed = torch.cat((llm_emb, embed), dim=1)
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                tf_embed = F.dropout(tf_embed, p)
        
        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                target_embed = F.dropout(target_embed, p)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred
    
    def get_embedding(self):
        return self.tf_ouput, self.target_output

# ─────────────────────────────────────────────────────────────────────────────
# scRegNet-Spatial
# ─────────────────────────────────────────────────────────────────────────────

class scRegNet_Spatial(nn.Module):
    """
    scRegNet-Spatial: niche-aware GRN inference for spatial transcriptomics.

    Two additions over the base scTransNet_GCN:
      1. SpatialGNN   – GraphSAGE on the spot kNN graph produces spatially
                        informed spot embeddings.
      2. AttentionPool – learnable pooling collapses (N_spots, D) → (N_genes, D)
                        replacing the implicit mean-pooling in the original model
                        (where the raw expression matrix was used as GCN node
                        features, effectively a uniform average over all cells).

    Everything downstream (gene-level GCN, scFM fusion, MLP decoder) is
    identical to scTransNet_GCN.

    Args:
        num_genes (int): Number of genes (= rows of expression matrix).
        num_spots (int): Number of spatial spots/cells.
        args: Parsed argument namespace.
        gene_dim (int): Dimensionality of scFM gene embeddings.
        device: torch.device.
    """

    def __init__(self, num_genes: int, num_spots: int, args, gene_dim: int, device):
        super().__init__()
        self.args   = args
        self.device = device

        # ── 1. SpatialGNN: spot-level spatial message passing ─────────────
        spatial_hidden = getattr(args, "spatial_gnn_hidden", 128)
        spatial_out    = getattr(args, "spatial_gnn_out",    128)

        self.spatial_gnn = SpatialGNN(
            in_channels     = num_genes,       # each spot's feature = its gene expr profile
            hidden_channels = spatial_hidden,
            out_channels    = spatial_out,
            dropout         = args.dropout,
        )

        # ── 2. AttentionPool: spots → genes ───────────────────────────────
        self.attn_pool = AttentionPool(embed_dim=spatial_out)

        # ── 3. Gene-level GCN (same topology as scTransNet_GCN) ───────────
        input_dim = spatial_out
        self.convs = nn.ModuleList()
        for i in range(args.gnn_num_layers):
            hd = args.gnn_hidden_dims[i]
            self.convs.append(GCNConv(input_dim, hd))
            input_dim = hd
        self.gnn_output_dim = input_dim

        # ── 4. Cross-modal fusion (optional) ──────────────────────────────
        self.use_cross_attention = getattr(args, "use_cross_attention", False)
        if self.use_cross_attention:
            self.cross_modal_fusion = IterativeCrossModalFusion(
                scfm_dim   = gene_dim,
                gnn_dim    = self.gnn_output_dim,
                hidden_dim = getattr(args, "cross_attention_hidden_dim", 256),
                num_layers = getattr(args, "cross_attention_layers", 2),
                num_heads  = getattr(args, "cross_attention_heads", 4),
                dropout    = args.dropout,
                fusion_mode= getattr(args, "cross_attention_fusion_mode", "concat"),
            )
            fused_dim = self.cross_modal_fusion.get_output_dim()
        else:
            fused_dim = gene_dim + self.gnn_output_dim

        # ── 5. MLP decoder ────────────────────────────────────────────────
        self.layers = nn.ModuleList()
        for i in range(args.mlp_num_layers):
            hd_mlp = args.mlp_hidden_dims[i]
            if i == 0:
                self.layers.append(nn.Linear(fused_dim, hd_mlp))
            else:
                self.layers.append(nn.Linear(input_dim, hd_mlp))
            input_dim = hd_mlp

        if args.type == "MLP":
            self.linear = nn.Linear(2 * input_dim, 2)

        self._reset_parameters()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _reset_parameters(self):
        self.spatial_gnn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def _encode_spatial(
        self,
        expr_matrix:        torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_weight: torch.Tensor,
        gene_spot_mask:     list,
    ) -> torch.Tensor:
        spot_feat = expr_matrix.t()          # (num_spots, num_genes)
        spot_emb  = self.spatial_gnn(spot_feat, spatial_edge_index, spatial_edge_weight)
        return self.attn_pool(spot_emb, gene_spot_mask)  # (num_genes, spatial_out)

    def _encode_gene(self, x: torch.Tensor, adj) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.args.dropout, training=self.training)
        return x

    def decode(self, tf_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        if self.args.type == "dot":
            return torch.sum(torch.mul(tf_embed, target_embed), dim=1).view(-1, 1)
        elif self.args.type == "cosine":
            return torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1, 1)
        elif self.args.type == "MLP":
            return self.linear(torch.cat([tf_embed, target_embed], dim=1))
        else:
            raise TypeError(f"{self.args.type!r} is not a valid decode type")

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x:                   torch.Tensor,   # (num_genes, num_spots)
        adj,                                  # gene regulatory graph (sparse)
        train_sample:        torch.Tensor,   # (B, 3) – [tf_idx, target_idx, label]
        llm_emb:             torch.Tensor,   # (num_genes, gene_dim)
        spatial_edge_index:  torch.Tensor,   # (2, E)
        spatial_edge_weight: torch.Tensor,   # (E,)
        gene_spot_mask:      list,           # list[num_genes] of LongTensors
    ) -> torch.Tensor:
        # 1. Spatial enrichment
        spatial_gene_emb = self._encode_spatial(
            x, spatial_edge_index, spatial_edge_weight, gene_spot_mask
        )

        # 2. Gene-level GCN on spatially enriched features
        embed = self._encode_gene(spatial_gene_emb, adj)

        # 3. Fuse with scFM gene embeddings
        if self.use_cross_attention:
            embed = self.cross_modal_fusion(llm_emb, embed)
        else:
            embed = torch.cat((llm_emb, embed), dim=1)

        # 4. TF and Target MLP branches
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                tf_embed = F.dropout(tf_embed, self.args.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                target_embed = F.dropout(target_embed, self.args.dropout, training=self.training)

        self.tf_output     = tf_embed
        self.target_output = target_embed

        train_tf     = tf_embed    [train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]

        return self.decode(train_tf, train_target)

    def get_embedding(self):
        return self.tf_output, self.target_output
