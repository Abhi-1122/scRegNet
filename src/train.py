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
from src.models import scTransNet_GCN, scTransNet_SAGE, scTransNet_GAT
from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
from src.utils import set_logging, set_seed
from src.args import save_args, parse_args
import warnings
from src.load_pretrained_gcn import load_pretrained_gcn

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
        self.gene_dim = feature1.size()[1]

        data_feature2 = feature2.to(self.device)

        sample_weights = np.ones(len(train_data), dtype=np.float32)
        if self.args.use_hard_negatives:
            train_data, sample_weights = self._augment_with_hard_negatives(data_input, train_data, sample_weights)
        
        if self.args.llm_type == "scBERT":
            feature1 = feature1[:-1]
        
        data_feature1 = feature1.to(self.device)
        train_load = scRNADataset(train_data, gene_num, flag=self.args.flag, sample_weights=sample_weights)
        adj = train_load.Adj_Generate(tf, loop=self.args.loop)
        adj = adj2saprse_tensor(adj)
        adj = adj.to(self.device)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.to(self.device)
        test_data = torch.from_numpy(test_data)
        test_data = test_data.to(self.device)

        return train_load, test_data, adj, data_feature1, data_feature2


    def _augment_with_hard_negatives(self, data_input, train_data, sample_weights):
        hard_negative_file = self.args.hard_negative_file
        if not os.path.exists(hard_negative_file):
            logger.warning(f"Hard-negative file not found: {hard_negative_file}. Skipping hard negatives.")
            return train_data, sample_weights

        norman_df = pd.read_csv(hard_negative_file)
        required_cols = {"TF", "Gene", "Log2FC", "PVal_adj"}
        if not required_cols.issubset(norman_df.columns):
            logger.warning(
                f"Hard-negative file missing required columns {required_cols}. Skipping hard negatives."
            )
            return train_data, sample_weights

        hard_negatives = norman_df[
            (norman_df["Log2FC"].abs() < self.args.hard_negative_log2fc_threshold)
            & (norman_df["PVal_adj"] > self.args.hard_negative_padj_threshold)
        ][["TF", "Gene"]]

        gene_to_idx = {str(gene): idx for idx, gene in enumerate(data_input.index.astype(str))}
        existing_pairs = {(int(row[0]), int(row[1])) for row in train_data}
        positive_pairs = {(int(row[0]), int(row[1])) for row in train_data if int(row[-1]) == 1}

        hard_rows = []
        for tf_name, gene_name in hard_negatives.itertuples(index=False):
            tf_idx = gene_to_idx.get(str(tf_name))
            gene_idx = gene_to_idx.get(str(gene_name))
            if tf_idx is None or gene_idx is None:
                continue

            pair = (tf_idx, gene_idx)
            if pair in existing_pairs or pair in positive_pairs:
                continue

            hard_rows.append([tf_idx, gene_idx, 0])
            existing_pairs.add(pair)

        if len(hard_rows) == 0:
            logger.info("No valid hard negatives were added after index overlap and deduplication.")
            return train_data, sample_weights

        hard_negative_array = np.asarray(hard_rows, dtype=train_data.dtype)
        hard_negative_weights = np.full(len(hard_negative_array), self.args.hard_negative_weight, dtype=np.float32)

        augmented_train = np.concatenate([train_data, hard_negative_array], axis=0)
        augmented_weights = np.concatenate([sample_weights, hard_negative_weights], axis=0)

        logger.info(
            f"Added {len(hard_negative_array)} hard negatives from {hard_negative_file}. "
            f"Total train edges: {len(augmented_train)}"
        )
        return augmented_train, augmented_weights


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
        train_load, test_data, adj, data_feature1, data_feature2 = self._prepare_data()
        self.model = self.get_model()
        if getattr(self.args, "pretrained_gcn", False):
            load_pretrained_gcn(self.model, self.args.pretrained_gcn_ckpt, map_location=self.device)
        optimizer = getattr(optim, self.args.optimizer_name)(self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

        for epoch in tqdm(range(self.args.gnn_epochs)):
            running_loss = 0.0
            for train_x, train_y, sample_weight in DataLoader(train_load, batch_size=self.args.batch_size, shuffle=True):
                self.model.train()
                optimizer.zero_grad()

                if self.args.flag:
                    train_y = train_y.to(self.device)
                else:
                    train_y = train_y.to(self.device).view(-1, 1)

                pred = self.model(data_feature2, adj, train_x, data_feature1)

                if self.args.flag:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)

                sample_weight = sample_weight.to(self.device)
                per_elem_loss = F.binary_cross_entropy(pred, train_y, reduction="none")
                if self.args.flag:
                    per_sample_loss = per_elem_loss.mean(dim=1)
                else:
                    per_sample_loss = per_elem_loss.view(-1)

                loss_BCE = (per_sample_loss * sample_weight).mean()
                loss_BCE.backward()
                optimizer.step()

                running_loss += loss_BCE.item()
            
            if (epoch+1) % self.args.gnn_eval_interval == 0:
                self.model.eval()
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