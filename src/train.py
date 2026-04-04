import numpy as np
import pandas as pd
import os
import gc
import logging
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import scTransNet_GCN, scTransNet_SAGE, scTransNet_GAT, scRegNet_Spatial
from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
from src.utils import set_logging, set_seed
from src.args import save_args, parse_args
import warnings

try:
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
except ImportError:
    pass

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


class Trainer: 
    def __init__(self, args, **kwargs):
        self.args = args
        self.trial = kwargs.pop("trial", None)

    @property
    def device(self):
        if torch.cuda.is_available():
            gpu_id = self.args.single_gpu if self.args.single_gpu < torch.cuda.device_count() else 0
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")
    
    def _get_embeddings(self, gene_num, data_input):
        if self.args.llm_type == "Geneformer":
            scFM_embs = os.path.join(self.args.scFM_folder, "Geneformer")
            embs = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_gene_embeddings.csv'))
            final_df = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}.csv'))
            
            x = np.zeros((len(embs.columns)-1))
            gene_embeddings = []
            for _, row in final_df.iterrows():
                ensembl_id = row['ensembl_id']
                if ensembl_id in embs['Unnamed: 0'].values:
                    gene_emb = embs[embs['Unnamed: 0']==ensembl_id].values[0][1:]
                    gene_embeddings.append(np.array(gene_emb, dtype=np.float32))
                else:
                    gene_embeddings.append(x)

            gene_embeddings = np.array(gene_embeddings)

        elif self.args.llm_type == "scBERT":
            scFM_embs = os.path.join(self.args.scFM_folder, "scBERT")
            cell_embeddings_arr = np.load(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_cell_embeddings.npy'))
            gene_embeddings = np.mean(cell_embeddings_arr, axis=0)
            
        elif self.args.llm_type == "scFoundation":
            scFM_embs = os.path.join(self.args.scFM_folder, "scFoundation")
            gene_list_df = pd.read_csv(os.path.join(scFM_embs, 'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
            gene_list = list(gene_list_df['gene_name'])
            x = np.zeros(512)

            gene_embeddings = np.load(os.path.join(scFM_embs, f'genemodule_{self.args.cell_type}_{self.args.num_TF}_singlecell_gene_embedding_f2_resolution.npy'))
            pooled_gene_embeddings = np.mean(gene_embeddings, axis=0)
            
            final_gene_embeddings = []
            cnt = 0
            for i in data_input.index:
                try:
                    final_gene_embeddings.append(pooled_gene_embeddings[gene_list.index(i)])
                except:
                    final_gene_embeddings.append(x)
                    cnt=cnt+1
                    
            gene_embeddings = np.array(final_gene_embeddings)
        
        feature1 = torch.from_numpy(gene_embeddings).float()
        return feature1
    
        
    def _prepare_data(self):
        path = os.path.join(self.args.data_folder, f'{self.args.cell_type}/TFs+{self.args.num_TF}/')
        exp_file = os.path.join(path, 'BL--ExpressionData.csv')
        train_file = os.path.join(path, 'Train_set.csv')
        test_file = os.path.join(path, 'Test_set.csv')
        tf_file = os.path.join(path, 'TF.csv')
        target_file = os.path.join(path, 'Target.csv')
        data_input = pd.read_csv(exp_file, index_col=0)
        train_data = pd.read_csv(train_file, index_col=0).values
        test_data = pd.read_csv(test_file, index_col=0).values
        tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
        tf = torch.from_numpy(tf).to(self.device)
        target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
        target = torch.from_numpy(target).to(self.device)

        loader = load_data(data_input)
        feature2 = loader.exp_data()
        feature2 = torch.from_numpy(feature2)
        gene_num = feature2.shape[0]
        feature1 = self._get_embeddings(gene_num, data_input)
        self.input_dim = feature2.size()[1]
        self.gene_dim  = feature1.size()[1]
        self.num_genes = gene_num
        self.scrna_gene_list = list(data_input.index)  # ordered gene names from scRNA data

        data_feature2 = feature2.to(self.device)
        
        if self.args.llm_type == "scBERT":
            feature1 = feature1[:-1]
        
        data_feature1 = feature1.to(self.device)
        train_load = scRNADataset(train_data, gene_num, flag=self.args.flag)
        adj = train_load.Adj_Generate(tf, loop=self.args.loop)
        adj = adj2saprse_tensor(adj)
        adj = adj.to(self.device)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.to(self.device)
        test_data = torch.from_numpy(test_data)
        test_data = test_data.to(self.device)

        return train_load, test_data, adj, data_feature1, data_feature2


    def _prepare_spatial_data(self, scrna_gene_list):
        """
        Load the three spatial artefacts produced by hHEP_spatial/:
          spatial_expression.csv  – genes × spots expression (replaces BL-ExpressionData)
          spatial_knn_graph.pt    – PyG edge_index + edge_weights for spot kNN graph
          gene_spot_mask.pt       – per-gene list of expressing spot indices

        The spatial expression matrix is reindexed to match *scrna_gene_list* so
        that num_genes, the adjacency matrix, and the SpatialGNN input all agree.
        Genes present in scRNA but absent from spatial are zero-filled.

        Returns:
            data_feature2_spatial : (num_genes, num_spots) float tensor on device
            spatial_edge_index    : (2, E) long tensor on device
            spatial_edge_weight   : (E,)  float tensor on device
            gene_spot_mask        : list[num_genes] of long tensors on device
        """
        import os
        import pandas as pd
        sdir = self.args.spatial_data_folder

        # Expression matrix (genes × spots) – align to scRNA gene ordering
        expr = pd.read_csv(os.path.join(sdir, "spatial_expression.csv"), index_col=0)
        # Reindex rows to match scRNA gene list; missing genes become NaN → 0
        expr = expr.reindex(scrna_gene_list).fillna(0.0)
        from src.utils import load_data
        loader = load_data(expr)
        feat = loader.exp_data()                                         # (num_scrna_genes, num_spots)
        data_feature2_spatial = torch.from_numpy(feat).to(self.device)

        # Spatial kNN graph
        knn = torch.load(os.path.join(sdir, "spatial_knn_graph.pt"))
        spatial_edge_index  = knn["edge_index" ].to(self.device)
        spatial_edge_weight = knn["edge_weights"].to(self.device)

        # Gene–spot expression mask – reindex to scRNA gene ordering
        mask_dict      = torch.load(os.path.join(sdir, "gene_spot_mask.pt"))
        spatial_gene_order = mask_dict["gene_order"]          # 933-gene list
        spatial_mask_by_name = dict(zip(spatial_gene_order, mask_dict["gene_spot_mask"]))
        empty = torch.tensor([], dtype=torch.long)
        gene_spot_mask = [
            spatial_mask_by_name.get(g, empty).to(self.device)
            for g in scrna_gene_list                           # 948-gene scRNA ordering
        ]

        return data_feature2_spatial, spatial_edge_index, spatial_edge_weight, gene_spot_mask

    def get_model(self):
        if getattr(self.args, "use_spatial", False):
            model = scRegNet_Spatial(
                num_genes = self.num_genes,
                num_spots = self.num_spots,
                args      = self.args,
                gene_dim  = self.gene_dim,
                device    = self.device,
            ).to(self.device)
        elif self.args.gnn_type == "GCN":
            model = scTransNet_GCN(input_dim=self.input_dim,
                                   args=self.args,
                                   gene_dim=self.gene_dim,
                                   device=self.device
                                   ).to(self.device)
        elif self.args.gnn_type == "GraphSAGE":
            model = scTransNet_SAGE(input_dim=self.input_dim,
                                   args=self.args,
                                   gene_dim=self.gene_dim,
                                   device=self.device
                                   ).to(self.device)
        elif self.args.gnn_type == "GAT":
            model = scTransNet_GAT(input_dim=self.input_dim,
                                   args=self.args,
                                   gene_dim=self.gene_dim,
                                   device=self.device
                                   ).to(self.device)
        return model


    def train(self):
        max_AUC = 0
        accumulate_patience = 0
        train_load, test_data, adj, data_feature1, data_feature2 = self._prepare_data()

        # ── Spatial branch: load extra artefacts ──────────────────────────
        use_spatial = getattr(self.args, "use_spatial", False)
        if use_spatial:
            (
                data_feature2,          # override: (num_genes, num_spots) spatial expr
                spatial_edge_index,
                spatial_edge_weight,
                gene_spot_mask,
            ) = self._prepare_spatial_data(self.scrna_gene_list)
            self.num_spots = data_feature2.shape[1]
        # ─────────────────────────────────────────────────────────────────

        self.model = self.get_model()
        optimizer = getattr(optim, self.args.optimizer_name)(self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

        for epoch in tqdm(range(self.args.gnn_epochs)):
            running_loss = 0.0
            for train_x, train_y in DataLoader(train_load, batch_size=self.args.batch_size, shuffle=True):
                self.model.train()
                optimizer.zero_grad()

                if self.args.flag:
                    train_y = train_y.to(self.device)
                else:
                    train_y = train_y.to(self.device).view(-1, 1)

                if use_spatial:
                    pred = self.model(
                        data_feature2, adj, train_x, data_feature1,
                        spatial_edge_index, spatial_edge_weight, gene_spot_mask,
                    )
                else:
                    pred = self.model(data_feature2, adj, train_x, data_feature1)

                if self.args.flag:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)

                loss_BCE = F.binary_cross_entropy(pred, train_y)
                loss_BCE.backward()
                optimizer.step()

                running_loss += loss_BCE.item()
            
            if (epoch+1) % self.args.gnn_eval_interval == 0:
                self.model.eval()
                if use_spatial:
                    score = self.model(
                        data_feature2, adj, test_data, data_feature1,
                        spatial_edge_index, spatial_edge_weight, gene_spot_mask,
                    )
                else:
                    score = self.model(data_feature2, adj, test_data, data_feature1)

                if self.args.flag:
                    score = torch.softmax(score, dim=1)
                else:
                    score = torch.sigmoid(score)

                AUC, AUPR, _ = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=self.args.flag)

                if AUC > max_AUC:
                    accumulate_patience = 0
                    max_AUC = AUC
                    AUC_AUPR = AUPR
                    self.args.ckpt_name = os.path.join(self.args.ckpt_dir, f"model_seed{self.args.random_seed}.pt")
                    torch.save(
                        self.model.state_dict(),
                        self.args.ckpt_name,
                    )
                     
                    save_args(self.args, self.args.ckpt_dir)
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 10:
                        break

        logger.info(f"best_auroc: {max_AUC:.4f}, auprc: {AUC_AUPR:.4f}")
        return max_AUC, AUC_AUPR
    

def main():
    set_logging()
    
    args = parse_args()
    logger.critical(
        f"Training on {args.dataset}, with {args.llm_type} as scFM backbone and {args.gnn_type} as GNN backbone"
    )
    logger.info(args)
    set_seed(random_seed=args.random_seed)

    trainer = Trainer(args)
    AUROC, AUPRC = trainer.train()
    logger.info(f"Final Results - AUROC: {AUROC}, AUPRC: {AUPRC}")

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    return AUROC, AUPRC


if __name__ == "__main__":
    AUROC, AUPRC = main()
    print(AUROC, AUPRC)