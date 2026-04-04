"""
inference_spatial.py
────────────────────
Mirror of inference.py for scRegNet-Spatial checkpoints.

Usage (from workspace root):
    python -m src.inference_spatial

Or with a custom checkpoint directory:
    python -m src.inference_spatial --best_dir ./ckpt/spatial_hHEP_500/best
"""

import argparse
import gc
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.args import load_args
from src.models import scRegNet_Spatial
from src.utils import scRNADataset, load_data, adj2saprse_tensor, set_logging
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(random_seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def Evaluation(y_true, y_pred, flag=False):
    if flag:
        y_p = y_pred[:, -1].cpu().detach().numpy().flatten()
    else:
        y_p = y_pred.cpu().detach().numpy().flatten()
    y_t = y_true.cpu().numpy().flatten().astype(int)
    return {
        'AUC':  roc_auc_score(y_true=y_t, y_score=y_p),
        'AUPR': average_precision_score(y_true=y_t, y_score=y_p),
    }


# ── Infer class ──────────────────────────────────────────────────────────────

class InferSpatial:
    def __init__(self, args):
        self.args = args

    @property
    def device(self):
        if torch.cuda.is_available():
            gpu_id = self.args.single_gpu if self.args.single_gpu < torch.cuda.device_count() else 0
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")

    # ── scFM embeddings (identical to inference.py) ───────────────────────

    def _get_embeddings(self, gene_num, data_input):
        if self.args.llm_type == "Geneformer":
            scFM_embs = os.path.join(self.args.scFM_folder, "Geneformer")
            embs     = pd.read_csv(os.path.join(scFM_embs,
                           f'{self.args.cell_type}_{self.args.num_TF}_gene_embeddings.csv'))
            final_df = pd.read_csv(os.path.join(scFM_embs,
                           f'{self.args.cell_type}_{self.args.num_TF}.csv'))
            x = np.zeros(len(embs.columns) - 1)
            gene_embeddings = []
            for _, row in final_df.iterrows():
                eid = row['ensembl_id']
                if eid in embs['Unnamed: 0'].values:
                    gene_embeddings.append(
                        embs[embs['Unnamed: 0'] == eid].values[0][1:].astype(np.float32))
                else:
                    gene_embeddings.append(x)
            gene_embeddings = np.array(gene_embeddings)

        elif self.args.llm_type == "scBERT":
            scFM_embs = os.path.join(self.args.scFM_folder, "scBERT")
            cell_embeddings_arr = np.load(
                os.path.join(scFM_embs,
                             f'{self.args.cell_type}_{self.args.num_TF}_cell_embeddings.npy'))
            gene_embeddings = np.mean(cell_embeddings_arr, axis=0)

        elif self.args.llm_type == "scFoundation":
            scFM_embs   = os.path.join(self.args.scFM_folder, "scFoundation")
            gene_list_df = pd.read_csv(
                os.path.join(scFM_embs, 'OS_scRNA_gene_index.19264.tsv'),
                header=0, delimiter='\t')
            gene_list = list(gene_list_df['gene_name'])
            x = np.zeros(512)
            raw = np.load(os.path.join(scFM_embs,
                f'genemodule_{self.args.cell_type}_{self.args.num_TF}'
                '_singlecell_gene_embedding_f2_resolution.npy'))
            pooled = np.mean(raw, axis=0)
            gene_embeddings = []
            for g in data_input.index:
                try:
                    gene_embeddings.append(pooled[gene_list.index(g)])
                except ValueError:
                    gene_embeddings.append(x)
            gene_embeddings = np.array(gene_embeddings)

        return torch.from_numpy(gene_embeddings).float()

    # ── scRNA data (identical to inference.py) ────────────────────────────

    def _prepare_data(self):
        path = os.path.join(self.args.data_folder,
                            f'{self.args.cell_type}/TFs+{self.args.num_TF}/')
        data_input = pd.read_csv(os.path.join(path, 'BL--ExpressionData.csv'), index_col=0)
        train_data = pd.read_csv(os.path.join(path, 'Train_set.csv'),  index_col=0).values
        test_data  = pd.read_csv(os.path.join(path, 'Test_set.csv'),   index_col=0).values
        tf         = pd.read_csv(os.path.join(path, 'TF.csv'),         index_col=0)['index'].values.astype(np.int64)
        target     = pd.read_csv(os.path.join(path, 'Target.csv'),     index_col=0)['index'].values.astype(np.int64)

        tf     = torch.from_numpy(tf).to(self.device)
        target = torch.from_numpy(target).to(self.device)

        loader   = load_data(data_input)
        feature2 = torch.from_numpy(loader.exp_data())
        gene_num = feature2.shape[0]

        feature1 = self._get_embeddings(gene_num, data_input)
        self.input_dim = feature2.size(1)
        self.gene_dim  = feature1.size(1)
        self.num_genes = gene_num
        self.scrna_gene_list = list(data_input.index)   # ordered gene names

        data_feature2 = feature2.to(self.device)
        if self.args.llm_type == "scBERT":
            feature1 = feature1[:-1]
        data_feature1 = feature1.to(self.device)

        train_load = scRNADataset(train_data, gene_num, flag=self.args.flag)
        adj = adj2saprse_tensor(train_load.Adj_Generate(tf, loop=self.args.loop))
        adj = adj.to(self.device)

        train_data = torch.from_numpy(train_data).to(self.device)
        test_data  = torch.from_numpy(test_data).to(self.device)

        return train_data, test_data, adj, data_feature1, data_feature2

    # ── spatial data (mirrors train.py _prepare_spatial_data) ────────────

    def _prepare_spatial_data(self):
        sdir = self.args.spatial_data_folder

        # Expression matrix (genes × spots) – reindex to scRNA gene ordering
        expr = pd.read_csv(os.path.join(sdir, "spatial_expression.csv"), index_col=0)
        expr = expr.reindex(self.scrna_gene_list).fillna(0.0)
        feat = load_data(expr).exp_data()               # (num_scrna_genes, num_spots)
        data_feature2_spatial = torch.from_numpy(feat).to(self.device)

        # Spot kNN graph
        knn = torch.load(os.path.join(sdir, "spatial_knn_graph.pt"))
        spatial_edge_index  = knn["edge_index" ].to(self.device)
        spatial_edge_weight = knn["edge_weights"].to(self.device)

        # Gene–spot mask – reindex to scRNA gene ordering
        mask_dict = torch.load(os.path.join(sdir, "gene_spot_mask.pt"))
        spatial_mask_by_name = dict(zip(mask_dict["gene_order"], mask_dict["gene_spot_mask"]))
        empty = torch.tensor([], dtype=torch.long)
        gene_spot_mask = [
            spatial_mask_by_name.get(g, empty).to(self.device)
            for g in self.scrna_gene_list
        ]

        return data_feature2_spatial, spatial_edge_index, spatial_edge_weight, gene_spot_mask

    # ── model ─────────────────────────────────────────────────────────────

    def get_model(self):
        return scRegNet_Spatial(
            num_genes = self.num_genes,
            num_spots = self.num_spots,
            args      = self.args,
            gene_dim  = self.gene_dim,
            device    = self.device,
        ).to(self.device)

    # ── inference (mirrors inference.py, 50 stochastic forward passes) ────

    def infer(self):
        train_data, test_data, adj, data_feature1, data_feature2 = self._prepare_data()

        # Load and align spatial artefacts
        (
            data_feature2,          # override with spatial expression
            spatial_edge_index,
            spatial_edge_weight,
            gene_spot_mask,
        ) = self._prepare_spatial_data()
        self.num_spots = data_feature2.shape[1]

        self.model = self.get_model()

        # Load checkpoint – spatial trainer saves as model_seed{N}.pt; fall back to model.pt
        ckpt_dir   = os.path.join(self.args.output_dir, "best/ckpt")
        seeded_path = os.path.join(ckpt_dir, f"model_seed{self.args.random_seed}.pt")
        plain_path  = os.path.join(ckpt_dir, "model.pt")
        model_path  = seeded_path if os.path.exists(seeded_path) else plain_path
        logger.info(f"Loading checkpoint: {model_path}")
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model.eval()

        spatial_args = (spatial_edge_index, spatial_edge_weight, gene_spot_mask)

        results_train, results_test = [], []
        for _ in tqdm(range(50)):
            score_test  = self.model(data_feature2, adj, test_data,  data_feature1, *spatial_args)
            score_train = self.model(data_feature2, adj, train_data, data_feature1, *spatial_args)

            if self.args.flag:
                score_train = torch.softmax(score_train, dim=1)
                score_test  = torch.softmax(score_test,  dim=1)
            else:
                score_train = torch.sigmoid(score_train)
                score_test  = torch.sigmoid(score_test)

            results_train.append(Evaluation(y_pred=score_train, y_true=train_data[:, -1], flag=self.args.flag))
            results_test.append( Evaluation(y_pred=score_test,  y_true=test_data[:, -1],  flag=self.args.flag))

        return results_train, results_test


# ── main ─────────────────────────────────────────────────────────────────────

def main(best_dir: str):
    set_logging()
    args = load_args(os.path.join(best_dir, 'ckpt'))
    logger.info(args)
    set_seed(args.random_seed)

    infer = InferSpatial(args)
    results_train, results_test = infer.infer()

    del infer
    torch.cuda.empty_cache()
    gc.collect()
    return results_train, results_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_dir", type=str,
                        default="./output/spatial_hHEP_500/best",
                        help="Path to the 'best/' directory containing ckpt/args.json and ckpt/model.pt")
    cli = parser.parse_args()

    _, results_test = main(cli.best_dir)

    metric_keys = results_test[0].keys()
    metrics      = {k: np.array([r[k] for r in results_test]) for k in metric_keys}
    mean_metrics = {k: np.mean(metrics[k]) for k in metric_keys}
    std_metrics  = {k: np.std( metrics[k]) for k in metric_keys}

    print("\n── Test results (mean ± std over 50 runs) ──")
    for k in metric_keys:
        print(f"  {k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
