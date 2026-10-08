"""Retrain Exp C with a different PANDA matrix. Everything else is copied from Exp C's saved
args.json, so the variant and the seed are the only things that change.
Usage: python3.12 -m analysis.train_variant <panda_variant | none> <seed>"""
import json, os, sys, time
variant, seed = sys.argv[1], int(sys.argv[2])
sys.argv = sys.argv[:1]
from src.args import parse_args
from src.train import Trainer
from src.utils import set_seed, set_logging

args = parse_args()
src_run = 'tf_500_hESC_baseline_rerun' if variant == 'none' else 'tf_500_hESC_panda_edge'  # 'none' = baseline, no PANDA
for k, v in json.load(open(f'out/GCN/Geneformer/{src_run}/ckpt/args.json')).items():
    if v is not None or not hasattr(args, k):
        setattr(args, k, v)
run = f'tf_500_hESC_pandavar_{variant}_seed{seed}'
if variant != 'none':
    args.panda_subdir = f'PANDA_variants/{variant}'
args.random_seed = seed
args.output_dir = f'out/GCN/Geneformer/{run}'
args.ckpt_dir = f'{args.output_dir}/ckpt'
args.show_progress_bar = False
os.makedirs(args.ckpt_dir, exist_ok=True)
set_seed(seed)
t0 = time.time()
auc, aupr = Trainer(args).train()
line = f'{run}\tAUROC {auc:.4f}\tAUPRC {aupr:.4f}\t{(time.time() - t0) / 60:.1f} min'
print(line)
with open('analysis/out/item2_runs.log', 'a') as f:
    f.write(line + '\n')
