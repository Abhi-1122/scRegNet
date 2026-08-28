import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
try:
    from cross_modal_attention import IterativeCrossModalFusion
except ImportError:
    from src.cross_modal_attention import IterativeCrossModalFusion


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


class scDiffTransNet_GCN(nn.Module):
    """
    Siamese, weight-tied differential/contrastive extension of scTransNet_GCN.
    DIFFERENTIAL_GRN_PLAN.md Sec 4.2. Takes two conditions of the same cell type through
    the SAME shared graph adjacency, condition-specific input projections (proj_A/proj_B,
    not weight-tied since n_cells_A != n_cells_B in general), then weight-tied
    GCNConv layers + cross-modal fusion + tf/target MLP (self.layers), followed by a
    4-way differential-edge decoder (diff_head) and an auxiliary static link-prediction
    head reusing the base model's decode().

    args.diff_ablation controls which branches are active:
      - 'full' / 'no_contrastive': complete architecture (no_contrastive only changes the
        loss at the DiffTrainer level, not this module).
      - 'no_scfm': drop the scFM branch -- fused representation is the GNN embedding alone.
      - 'no_grn': drop the GCNConv branch -- "gnn embedding" is proj_A/proj_B's raw linear
        projection of condition-specific expression, with no message passing, still fused
        with the (condition-independent) scFM embedding as usual. Expected near-chance on
        the differential task by construction (plan Sec 4.1) since this leaves almost no
        condition-specific *structural* signal.
    """
    def __init__(self, input_dim_A, input_dim_B, args, gene_dim, device):
        super(scDiffTransNet_GCN, self).__init__()
        self.args = args
        self.device = device
        self.ablation = getattr(args, 'diff_ablation', 'full')

        gnn_hidden0 = args.gnn_dim_hidden
        self.proj_A = nn.Linear(input_dim_A, gnn_hidden0)
        self.proj_B = nn.Linear(input_dim_B, gnn_hidden0)

        self.convs = torch.nn.ModuleList()
        conv_in = gnn_hidden0
        for i in range(self.args.gnn_num_layers):
            hidden_dim = self.args.gnn_hidden_dims[i]
            self.convs.append(GCNConv(conv_in, hidden_dim))
            conv_in = hidden_dim
        self.gnn_output_dim = conv_in if self.ablation != 'no_grn' else gnn_hidden0

        self.use_cross_attention = getattr(self.args, 'use_cross_attention', False)

        if self.ablation == 'no_scfm':
            fused_dim = self.gnn_output_dim
        elif self.use_cross_attention:
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
            fused_dim = gene_dim + self.gnn_output_dim

        self.layers = torch.nn.ModuleList()
        mlp_in = fused_dim
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]
            self.layers.append(nn.Linear(mlp_in, hidden_dim_mlp))
            mlp_in = hidden_dim_mlp
        self.mlp_out_dim = mlp_in

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2 * mlp_in, 2)

        diff_head_hidden = getattr(args, 'diff_head_hidden', 128)
        self.diff_head = nn.Sequential(
            nn.Linear(6 * self.mlp_out_dim, diff_head_hidden),
            nn.LeakyReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(diff_head_hidden, 4),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.proj_A.reset_parameters()
        self.proj_B.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def encode(self, x, adj):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                p = self.args.dropout
                x = F.dropout(x, p, training=self.training)
        return x

    def decode(self, tf_embed, target_embed):
        if self.args.type == 'dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob, dim=1).view(-1, 1)
            return prob
        elif self.args.type == 'cosine':
            prob = torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1, 1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.args.type))

    def _branch_embed(self, x, proj, adj):
        """Condition-specific GNN branch: proj (not tied) -> [GCNConv]* (tied), or just
        proj alone for the no_grn ablation."""
        h = proj(x)
        if self.ablation == 'no_grn':
            return h
        return self.encode(h, adj)

    def _fuse(self, llm_emb, gnn_embed):
        if self.ablation == 'no_scfm':
            return gnn_embed
        if self.use_cross_attention:
            return self.cross_modal_fusion(llm_emb, gnn_embed)
        return torch.cat((llm_emb, gnn_embed), dim=1)

    def _mlp_pass(self, fused):
        h = fused
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.leaky_relu(h)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                h = F.dropout(h, p, training=self.training)
        return h

    def forward_pair(self, x_A, x_B, adj, llm_emb, static_edges, diff_edges):
        """
        x_A: (num_genes, n_cells_A), x_B: (num_genes, n_cells_B) -- condition-specific
            per-gene expression features.
        adj: shared sparse adjacency (same graph structure for both conditions).
        llm_emb: (num_genes, gene_dim) condition-independent scFM embedding.
        static_edges: (num_static, 2) long tensor of (tf_idx, target_idx), same edge set
            reused for both conditions' auxiliary static link-prediction loss.
        diff_edges: (num_diff, 2) long tensor of (tf_idx, target_idx) candidate edges for
            the 4-way differential head.
        Returns: static_pred_A, static_pred_B, diff_logits, r_A, r_B
        """
        h_A = self._branch_embed(x_A, self.proj_A, adj)
        h_B = self._branch_embed(x_B, self.proj_B, adj)

        fused_A = self._fuse(llm_emb, h_A)
        fused_B = self._fuse(llm_emb, h_B)

        tf_A = self._mlp_pass(fused_A)
        tgt_A = self._mlp_pass(fused_A)
        tf_B = self._mlp_pass(fused_B)
        tgt_B = self._mlp_pass(fused_B)

        self.tf_A, self.tgt_A, self.tf_B, self.tgt_B = tf_A, tgt_A, tf_B, tgt_B

        static_pred_A = self.decode(tf_A[static_edges[:, 0]], tgt_A[static_edges[:, 1]])
        static_pred_B = self.decode(tf_B[static_edges[:, 0]], tgt_B[static_edges[:, 1]])

        i, j = diff_edges[:, 0], diff_edges[:, 1]
        diff_feat = torch.cat([
            tf_A[i], tgt_A[j], tf_B[i], tgt_B[j],
            (tf_A[i] - tf_B[i]).abs(), (tgt_A[j] - tgt_B[j]).abs(),
        ], dim=1)
        diff_logits = self.diff_head(diff_feat)

        r_A = tf_A[i] * tgt_A[j]
        r_B = tf_B[i] * tgt_B[j]

        return static_pred_A, static_pred_B, diff_logits, r_A, r_B

    def get_embedding(self):
        return self.tf_A, self.tgt_A, self.tf_B, self.tgt_B


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