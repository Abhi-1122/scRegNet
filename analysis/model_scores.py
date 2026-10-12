"""Load a trained summer26 checkpoint and dump its scores for test edges and for every
TF x gene pair, plus the per-node TF/target embeddings the decoder sees.
Usage: python3.12 analysis/model_scores.py <run_dir_name>   (under out/GCN/Geneformer/)"""
import json, os, sys
import numpy as np
import torch

sys.argv = sys.argv[:1] + ['--placeholder_unused'] * 0 + sys.argv[1:]
run = sys.argv.pop(1)
from src.args import parse_args
from src.train import Trainer
from src.utils import set_seed

ckpt_dir = f'out/GCN/Geneformer/{run}/ckpt'
saved = json.load(open(os.path.join(ckpt_dir, 'args.json')))
sys.argv = sys.argv[:1]
args = parse_args()
for k, v in saved.items():
    if v is not None or not hasattr(args, k):
        setattr(args, k, v)
set_seed(args.random_seed)

tr = Trainer(args)
_, test_data, adj, adj_ppi, f1, f2, panda_full, _ = tr._prepare_data()
model = tr.get_model()
pt = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')][0]
model.load_state_dict(torch.load(os.path.join(ckpt_dir, pt), map_location=tr.device))
model.eval()


def score(pairs):
    pairs = torch.as_tensor(pairs, device=tr.device)
    pe = panda_full[pairs[:, 0], pairs[:, 1]] if (panda_full is not None and getattr(args, 'use_panda_edge', False)) else None
    with torch.no_grad():
        logits = model(f2, adj, pairs, f1, adj_ppi=adj_ppi, panda_edge_feat=pe)
    return logits.cpu().numpy()


test_logits = score(test_data[:, :2])
n = f2.shape[0]
tf_idx = np.unique(test_data[:, 0].cpu().numpy())  # the ~34 TFs that appear in the benchmark
allpairs = np.array([(t, g) for t in tf_idx for g in range(n)])
all_logits = np.concatenate([score(allpairs[i:i + 20000]) for i in range(0, len(allpairs), 20000)])
tf_emb, tgt_emb = (x.detach().cpu().numpy() for x in model.get_embedding())
W = model.linear.weight.detach().cpu().numpy()  # (2, 2*dim [+1 panda])

os.makedirs('analysis/out', exist_ok=True)
np.savez(f'analysis/out/scores_{run.replace("/", "_")}.npz', test_pairs=test_data.cpu().numpy(), test_logits=test_logits,
         all_pairs=allpairs, all_logits=all_logits, tf_emb=tf_emb, tgt_emb=tgt_emb, W=W,
         b=model.linear.bias.detach().cpu().numpy())

from sklearn.metrics import roc_auc_score, average_precision_score
y = test_data[:, -1].cpu().numpy()
s = test_logits[:, 1] - test_logits[:, 0]
print(run, 'test AUROC %.4f AUPRC %.4f' % (roc_auc_score(y, s), average_precision_score(y, s)))
