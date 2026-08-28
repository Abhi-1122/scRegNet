"""
DiffTrainer -- trains scDiffTransNet_GCN on one (cell_type, condition_a, condition_b) pair.
DIFFERENTIAL_GRN_PLAN.md Sec 4.4. Parallel to src/train.py's Trainer; does not touch
train.py, so the static baseline stays runnable unmodified.
"""

import os
import gc
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import warnings

from src.models import scDiffTransNet_GCN
from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, set_seed, set_logging
from src.args import save_args, parse_args
from src.differential_data import load_condition_subsets
from src.differential_labels import (
    load_or_compute_labels, get_candidate_edges, TRAINABLE_CLASSES, CLASS_TO_IDX, stratified_split,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class DiffTrainer:
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
        """Condition-independent scFM embeddings -- identical logic to Trainer._get_embeddings
        in src/train.py (Geneformer path only; this pass's runs all use llm_type=Geneformer)."""
        if self.args.llm_type == "Geneformer":
            scFM_embs = os.path.join(self.args.scFM_folder, "Geneformer")
            embs = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}_gene_embeddings.csv'))
            final_df = pd.read_csv(os.path.join(scFM_embs, f'{self.args.cell_type}_{self.args.num_TF}.csv'))

            x = np.zeros((len(embs.columns) - 1))
            gene_embeddings = []
            for _, row in final_df.iterrows():
                ensembl_id = row['ensembl_id']
                if ensembl_id in embs['Unnamed: 0'].values:
                    gene_emb = embs[embs['Unnamed: 0'] == ensembl_id].values[0][1:]
                    gene_embeddings.append(np.array(gene_emb, dtype=np.float32))
                else:
                    gene_embeddings.append(x)
            gene_embeddings = np.array(gene_embeddings)
        else:
            raise NotImplementedError(
                f"DiffTrainer._get_embeddings: llm_type='{self.args.llm_type}' not ported "
                f"in this pass (only Geneformer used for the differential-GRN experiments)."
            )
        feature1 = torch.from_numpy(gene_embeddings).float()
        return feature1

    def _prepare_data(self):
        path = os.path.join(self.args.data_folder, f'{self.args.cell_type}/TFs+{self.args.num_TF}/')
        exp_file = os.path.join(path, 'BL--ExpressionData.csv')
        train_file = os.path.join(path, 'Train_set.csv')
        test_file = os.path.join(path, 'Test_set.csv')
        tf_file = os.path.join(path, 'TF.csv')

        data_input = pd.read_csv(exp_file, index_col=0)
        train_data = pd.read_csv(train_file, index_col=0).values
        test_data = pd.read_csv(test_file, index_col=0).values
        tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
        tf_t = torch.from_numpy(tf).to(self.device)

        gene_num = data_input.shape[0]

        feature1 = self._get_embeddings(gene_num, data_input)
        self.gene_dim = feature1.size()[1]
        data_feature1 = feature1.to(self.device)

        cols_a, cols_b = load_condition_subsets(data_input, self.args.cell_type,
                                                 self.args.condition_a, self.args.condition_b)
        x_a = load_data(cols_a).exp_data()
        x_b = load_data(cols_b).exp_data()
        self.input_dim_a = x_a.shape[1]
        self.input_dim_b = x_b.shape[1]
        data_feature2_a = torch.from_numpy(x_a).to(self.device)
        data_feature2_b = torch.from_numpy(x_b).to(self.device)

        train_load = scRNADataset(train_data, gene_num, flag=self.args.flag)
        adj = train_load.Adj_Generate(tf_t, loop=self.args.loop)
        adj = adj2saprse_tensor(adj).to(self.device)

        test_data_t = torch.from_numpy(test_data).to(self.device)

        labels_df = load_or_compute_labels(
            self.args.data_folder, self.args.cell_type, self.args.num_TF,
            self.args.condition_a, self.args.condition_b,
            corr_method=self.args.corr_method, corr_min_r=self.args.corr_min_r, fdr_q=self.args.fdr_q,
        )
        self.labels_df = labels_df
        self.diagnostic_counts = labels_df['label'].value_counts().to_dict()

        trainable = labels_df[labels_df['label'].isin(TRAINABLE_CLASSES)].reset_index(drop=True)
        edges = trainable[['TF_idx', 'Target_idx']].values.astype(np.int64)
        y = trainable['label'].map(CLASS_TO_IDX).values.astype(np.int64)

        edges_train, y_train, edges_test, y_test = stratified_split(
            edges, y, self.args.diff_train_frac, self.args.random_seed)

        self.diff_train_class_counts = {TRAINABLE_CLASSES[c]: int((y_train == c).sum()) for c in range(4)}
        self.diff_test_class_counts = {TRAINABLE_CLASSES[c]: int((y_test == c).sum()) for c in range(4)}

        self.candidate_edges_all = get_candidate_edges(self.args.data_folder, self.args.cell_type, self.args.num_TF)

        return (train_load, test_data_t, adj, data_feature1, data_feature2_a, data_feature2_b,
                edges_train, y_train, edges_test, y_test)

    def get_model(self):
        model = scDiffTransNet_GCN(
            input_dim_A=self.input_dim_a, input_dim_B=self.input_dim_b,
            args=self.args, gene_dim=self.gene_dim, device=self.device,
        ).to(self.device)
        return model

    def _contrastive_loss(self, r_A, r_B, y):
        device = r_A.device
        conserved_idx = torch.nonzero(y == CLASS_TO_IDX['conserved'], as_tuple=True)[0]
        specific_mask = (y == CLASS_TO_IDX['A-specific']) | (y == CLASS_TO_IDX['B-specific'])

        r_A_n = F.normalize(r_A, dim=1, eps=1e-8)
        r_B_n = F.normalize(r_B, dim=1, eps=1e-8)

        loss_infonce = torch.tensor(0.0, device=device)
        if conserved_idx.numel() >= 1:
            sim_all = torch.matmul(r_A_n[conserved_idx], r_B_n.T) / self.args.contrastive_temperature
            loss_infonce = F.cross_entropy(sim_all, conserved_idx)

        loss_margin = torch.tensor(0.0, device=device)
        if specific_mask.sum() >= 1:
            sim_s = (r_A_n[specific_mask] * r_B_n[specific_mask]).sum(dim=1)
            loss_margin = torch.clamp(sim_s - self.args.contrastive_margin, min=0).mean()

        return loss_infonce + loss_margin

    def _diff_metrics(self, y_true, probs, pred):
        floor = self.args.diff_class_count_floor
        class_counts = np.bincount(y_true, minlength=4)
        reports = {}
        f1s, included = [], []
        for c in range(4):
            name = TRAINABLE_CLASSES[c]
            n = int(class_counts[c])
            entry = {'n_test': n}
            if n >= floor:
                y_bin = (y_true == c).astype(int)
                try:
                    entry['auroc'] = float(roc_auc_score(y_bin, probs[:, c]))
                    entry['auprc'] = float(average_precision_score(y_bin, probs[:, c]))
                except Exception:
                    entry['auroc'], entry['auprc'] = None, None
                entry['f1'] = float(f1_score(y_bin, (pred == c).astype(int), zero_division=0))
                f1s.append(entry['f1'])
                included.append(name)
            else:
                entry['auroc'], entry['auprc'], entry['f1'] = None, None, None
            reports[name] = entry
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        reports['_macro_f1_classes_included'] = included
        return macro_f1, reports

    def evaluate(self, feat2_a, feat2_b, adj, feat1, test_static, edges_test_t, y_test_t):
        self.model.eval()
        with torch.no_grad():
            static_pred_A, static_pred_B, diff_logits, r_A, r_B = self.model.forward_pair(
                feat2_a, feat2_b, adj, feat1, test_static, edges_test_t)

            if self.args.flag:
                score_A = torch.softmax(static_pred_A, dim=1)
                score_B = torch.softmax(static_pred_B, dim=1)
            else:
                score_A = torch.sigmoid(static_pred_A)
                score_B = torch.sigmoid(static_pred_B)
            auc_A, aupr_A, _ = Evaluation(y_pred=score_A, y_true=test_static[:, -1], flag=self.args.flag)
            auc_B, aupr_B, _ = Evaluation(y_pred=score_B, y_true=test_static[:, -1], flag=self.args.flag)

            diff_probs = torch.softmax(diff_logits, dim=1).cpu().numpy()
            diff_pred = diff_probs.argmax(axis=1)
            y_true = y_test_t.cpu().numpy()

            macro_f1, per_class = self._diff_metrics(y_true, diff_probs, diff_pred)
            cm = confusion_matrix(y_true, diff_pred, labels=list(range(4))).tolist()

        return {
            'static_auc_A': float(auc_A), 'static_aupr_A': float(aupr_A),
            'static_auc_B': float(auc_B), 'static_aupr_B': float(aupr_B),
            'macro_f1': macro_f1, 'per_class': per_class, 'confusion_matrix': cm,
            'confusion_matrix_labels': TRAINABLE_CLASSES,
        }

    def train(self):
        (train_load, test_static, adj, feat1, feat2_a, feat2_b,
         edges_train, y_train, edges_test, y_test) = self._prepare_data()

        self.model = self.get_model()
        optimizer = getattr(optim, self.args.optimizer_name)(
            self.model.parameters(), lr=self.args.gnn_lr, weight_decay=self.args.gnn_weight_decay)

        edges_train_t = torch.from_numpy(edges_train).to(self.device)
        y_train_t = torch.from_numpy(y_train).to(self.device)
        edges_test_t = torch.from_numpy(edges_test).to(self.device)
        y_test_t = torch.from_numpy(y_test).to(self.device)

        class_counts_train = np.bincount(y_train, minlength=4).astype(np.float64)
        class_counts_safe = np.where(class_counts_train == 0, 1, class_counts_train)
        inv_freq = class_counts_safe.sum() / (4 * class_counts_safe)
        ce_weight = torch.tensor(inv_freq, dtype=torch.float32, device=self.device)

        best_f1 = -1.0
        best_metrics = None
        patience = 0
        has_diff_train = len(y_train) > 0

        for epoch in tqdm(range(self.args.gnn_epochs)):
            self.model.train()
            for train_x, train_y in DataLoader(train_load, batch_size=self.args.batch_size, shuffle=True):
                optimizer.zero_grad()
                if self.args.flag:
                    train_y = train_y.to(self.device)
                else:
                    train_y = train_y.to(self.device).view(-1, 1)

                static_pred_A, static_pred_B, diff_logits, r_A, r_B = self.model.forward_pair(
                    feat2_a, feat2_b, adj, feat1, train_x, edges_train_t if has_diff_train else edges_test_t)

                if self.args.flag:
                    pred_A = torch.softmax(static_pred_A, dim=1)
                    pred_B = torch.softmax(static_pred_B, dim=1)
                else:
                    pred_A = torch.sigmoid(static_pred_A)
                    pred_B = torch.sigmoid(static_pred_B)

                loss_static = F.binary_cross_entropy(pred_A, train_y) + F.binary_cross_entropy(pred_B, train_y)

                if has_diff_train:
                    loss_ce = F.cross_entropy(diff_logits, y_train_t, weight=ce_weight)
                    loss_con = torch.tensor(0.0, device=self.device)
                    if self.args.diff_ablation != 'no_contrastive':
                        loss_con = self._contrastive_loss(r_A, r_B, y_train_t)
                else:
                    loss_ce = torch.tensor(0.0, device=self.device)
                    loss_con = torch.tensor(0.0, device=self.device)

                loss = (self.args.diff_static_weight * loss_static
                        + self.args.diff_ce_weight * loss_ce
                        + self.args.diff_contrastive_weight * loss_con)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % self.args.gnn_eval_interval == 0:
                metrics = self.evaluate(feat2_a, feat2_b, adj, feat1, test_static, edges_test_t, y_test_t)
                if metrics['macro_f1'] >= best_f1:
                    best_f1 = metrics['macro_f1']
                    best_metrics = metrics
                    patience = 0
                    if self.args.ckpt_dir:
                        os.makedirs(self.args.ckpt_dir, exist_ok=True)
                        ckpt_path = os.path.join(self.args.ckpt_dir, f"model_seed{self.args.random_seed}.pt")
                        torch.save(self.model.state_dict(), ckpt_path)
                        save_args(self.args, self.args.ckpt_dir)
                else:
                    patience += 1
                    if patience >= self.args.diff_patience:
                        break

        if best_metrics is None:
            best_metrics = self.evaluate(feat2_a, feat2_b, adj, feat1, test_static, edges_test_t, y_test_t)

        best_metrics['diagnostic_counts'] = self.diagnostic_counts
        best_metrics['diff_train_class_counts'] = self.diff_train_class_counts
        best_metrics['diff_test_class_counts'] = self.diff_test_class_counts
        self._final_tensors = (feat2_a, feat2_b, adj, feat1)
        return best_metrics

    def predict_all_candidates(self):
        """Run the trained model over the FULL candidate edge space (not just the diff-test
        split) -- used for the plan Sec 5.3 named biological checks, which look up specific
        genes' predicted labels/direction rather than aggregate metrics."""
        feat2_a, feat2_b, adj, feat1 = self._final_tensors
        candidate_edges_t = torch.from_numpy(self.candidate_edges_all).to(self.device)
        self.model.eval()
        with torch.no_grad():
            dummy_static = candidate_edges_t  # unused output, but forward_pair needs a static_edges arg
            _, _, diff_logits, _, _ = self.model.forward_pair(
                feat2_a, feat2_b, adj, feat1, dummy_static, candidate_edges_t)
            probs = torch.softmax(diff_logits, dim=1).cpu().numpy()
        pred_idx = probs.argmax(axis=1)
        gene_names = pd.read_csv(
            os.path.join(self.args.data_folder, f'{self.args.cell_type}/TFs+{self.args.num_TF}/BL--ExpressionData.csv'),
            index_col=0).index.to_numpy()
        out = pd.DataFrame({
            'TF': gene_names[self.candidate_edges_all[:, 0]],
            'Target': gene_names[self.candidate_edges_all[:, 1]],
            'pred_label': [TRAINABLE_CLASSES[i] for i in pred_idx],
            'prob_conserved': probs[:, CLASS_TO_IDX['conserved']],
            'prob_A_specific': probs[:, CLASS_TO_IDX['A-specific']],
            'prob_B_specific': probs[:, CLASS_TO_IDX['B-specific']],
            'prob_neither': probs[:, CLASS_TO_IDX['neither']],
        })
        return out


def main():
    set_logging()
    args = parse_args()
    logger.critical(
        f"Differential training: {args.cell_type} {args.condition_a} vs {args.condition_b}, "
        f"ablation={args.diff_ablation}, gnn_type={args.gnn_type}"
    )
    set_seed(random_seed=args.random_seed)

    trainer = DiffTrainer(args)
    metrics = trainer.train()
    logger.info(json.dumps(metrics, indent=2, default=str))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        preds = trainer.predict_all_candidates()
        preds.to_csv(os.path.join(args.output_dir, 'full_candidate_predictions.csv'), index=False)

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    return metrics


if __name__ == "__main__":
    metrics = main()
    print(json.dumps(metrics, indent=2, default=str))
