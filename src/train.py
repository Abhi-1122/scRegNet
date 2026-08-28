import numpy as np
import pandas as pd
import os
import glob
import gc
import logging
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import scTransNet_GCN, scTransNet_SAGE, scTransNet_GAT
from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, build_ppi_adjacency
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

        elif self.args.llm_type == "scGPT":
            scFM_embs = os.path.join(self.args.scFM_folder, "scGPT")
            embs = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_gene_embeddings.csv'), index_col=0)
            x = np.zeros(embs.shape[1])
            gene_embeddings = []
            for gene in data_input.index:
                if gene in embs.index:
                    gene_embeddings.append(embs.loc[gene].values.astype(np.float32))
                else:
                    gene_embeddings.append(x)
            gene_embeddings = np.array(gene_embeddings)

        elif self.args.llm_type == "UCE":
            scFM_embs = os.path.join(self.args.scFM_folder, "UCE")
            embs = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_gene_embeddings.csv'), index_col=0)
            x = np.zeros(embs.shape[1])
            gene_embeddings = []
            for gene in data_input.index:
                if gene in embs.index:
                    gene_embeddings.append(embs.loc[gene].values.astype(np.float32))
                else:
                    gene_embeddings.append(x)
            gene_embeddings = np.array(gene_embeddings)

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
        target_df = pd.read_csv(target_file, index_col=0)
        target = target_df['index'].values.astype(np.int64)
        target = torch.from_numpy(target).to(self.device)

        loader = load_data(data_input)
        feature2 = loader.exp_data()
        feature2 = torch.from_numpy(feature2)
        gene_num = feature2.shape[0]

        panda_full_np = None
        panda_full_train_np = None
        node_feature_source = getattr(self.args, 'node_feature_source', 'scfm')
        use_panda_edge = getattr(self.args, 'use_panda_edge', False)
        if node_feature_source == 'panda' or use_panda_edge:
            panda_path = os.path.join(path, 'PANDA', 'panda_full_matrix.npy')
            if not os.path.exists(panda_path):
                raise FileNotFoundError(f"node_feature_source='panda' or --use_panda_edge set but "
                                         f"{panda_path} not found. Run src/build_panda.py first.")
            panda_full_np = np.load(panda_path)
        if use_panda_edge:
            # PANDA's motif prior IS the train-positive edges, so scoring a train edge
            # from panda_full_np would leak its own label (score ~10 vs ~0 for everything
            # else). Use the leakage-safe, k-fold out-of-fold matrix for TRAIN-pair
            # lookups instead; test-pair lookups stay on panda_full_np (already
            # leakage-free, since test edges are never part of any prior).
            oof_path = os.path.join(path, 'PANDA', 'panda_full_matrix_train_oof.npy')
            if not os.path.exists(oof_path):
                raise FileNotFoundError(f"--use_panda_edge set but {oof_path} not found. "
                                         f"Run src/build_panda.py first.")
            panda_full_train_np = np.load(oof_path)

        if node_feature_source == 'panda':
            tf_np = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
            feature1 = torch.from_numpy(panda_full_np[tf_np, :].T).float()
        else:
            feature1 = self._get_embeddings(gene_num, data_input)
        self.input_dim = feature2.size()[1]
        self.gene_dim = feature1.size()[1]

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

        adj_ppi = None
        if getattr(self.args, 'use_ppi', False):
            gene_to_idx = dict(zip(target_df['Gene'], target_df['index']))
            ppi_matches = glob.glob(os.path.join(path, 'STRING', '*.edges.csv'))
            if not ppi_matches:
                raise FileNotFoundError(f"--use_ppi set but no *.edges.csv found under {os.path.join(path, 'STRING')}")
            adj_ppi = build_ppi_adjacency(ppi_matches[0], gene_to_idx, gene_num, loop=self.args.loop)
            adj_ppi = adj2saprse_tensor(adj_ppi)
            adj_ppi = adj_ppi.to(self.device)

        panda_full = None
        if panda_full_np is not None:
            panda_full = torch.from_numpy(panda_full_np).float().to(self.device)
        panda_full_train = None
        if panda_full_train_np is not None:
            panda_full_train = torch.from_numpy(panda_full_train_np).float().to(self.device)

        return train_load, test_data, adj, adj_ppi, data_feature1, data_feature2, panda_full, panda_full_train


    def get_model(self):
        if self.args.gnn_type == "GCN":
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
        train_load, test_data, adj, adj_ppi, data_feature1, data_feature2, panda_full, panda_full_train = self._prepare_data()
        self.model = self.get_model()
        optimizer = getattr(optim, self.args.optimizer_name)(self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

        epoch_iter = range(self.args.gnn_epochs)
        if getattr(self.args, 'show_progress_bar', True):
            epoch_iter = tqdm(epoch_iter)
        for epoch in epoch_iter:
            running_loss = 0.0
            for train_x, train_y in DataLoader(train_load, batch_size=self.args.batch_size, shuffle=True):
                self.model.train()
                optimizer.zero_grad()

                if self.args.flag:
                    train_y = train_y.to(self.device)
                else:
                    train_y = train_y.to(self.device).view(-1, 1)

                panda_edge_feat = panda_full_train[train_x[:, 0], train_x[:, 1]] if panda_full_train is not None else None
                pred = self.model(data_feature2, adj, train_x, data_feature1, adj_ppi=adj_ppi, panda_edge_feat=panda_edge_feat)

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
                panda_edge_feat_test = panda_full[test_data[:, 0], test_data[:, 1]] if panda_full is not None else None
                score = self.model(data_feature2, adj, test_data, data_feature1, adj_ppi=adj_ppi, panda_edge_feat=panda_edge_feat_test)

                if self.args.flag:
                    score = torch.softmax(score, dim=1)
                else:
                    score = torch.sigmoid(score)

                AUC, AUPR, _ = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=self.args.flag)

                if AUC > max_AUC:
                    accumulate_patience = 0
                    max_AUC = AUC
                    AUC_AUPR = AUPR
                    if getattr(self.args, 'save_checkpoint', True):
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