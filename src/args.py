import argparse
from email import parser
import json
import logging
import os

logger = logging.getLogger(__name__)

GNN_LIST = ["GraphSAGE", "GCN", "GAT"]

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--single_gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--suffix", type=str, default="main")
    parser.add_argument("--deepspeed", type=str, default=None)

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--scFM_folder", type=str, default="./scFM")
    parser.add_argument("--task_type", type=str, default="link_pred")
    parser.add_argument("--output_dir", type=str)  # output dir
    parser.add_argument("--ckpt_dir", type=str)  # ckpt path to save
    parser.add_argument(
        "--ckpt_name", type=str, default="model.pt"
    )  # ckpt name to be loaded
   

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accum_interval", type=int, default=1)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.6)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "constant"],
    )

    # module hyperparameters
    # gnn parameters
    parser.add_argument("--gnn_epochs", type=int, default=300)
    parser.add_argument("--gnn_eval_interval", type=int, default=5)
    parser.add_argument("--gnn_label_smoothing", type=float, default=0.1)
    parser.add_argument("--gnn_warmup_ratio", type=float, default=0.25)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn_dim_hidden", type=int, default=256)
    parser.add_argument("--gnn_lr", type=float, default=5e-4)
    parser.add_argument("--gnn_weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--gnn_lr_scheduler_type",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )

    # optuna hyperparameters
    parser.add_argument("--expected_valid_acc", type=float, default=0.6)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--load_study", action="store_true", default=False)

    # grn parameters
    parser.add_argument("--dataset", type=str, default="tf_500")
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--llm_type", type=str, default="Geneformer")
    parser.add_argument("--cell_type", type=str, default="hESC")
    parser.add_argument("--num_TF", type=str, default="500")
    parser.add_argument("--flag", type=bool, default=True) 
    parser.add_argument("--loop", type=bool, default=False)
    parser.add_argument("--type", type=str, default="MLP") # score metric
    parser.add_argument("--mlp_dim_hidden", type=int, default=64) 
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    
    # Hidden dimensions for each layer (for manual training without Optuna)
    parser.add_argument("--gnn_hidden_dims", type=int, nargs='+', default=None,
                       help="Hidden dimensions for each GNN layer")
    parser.add_argument("--mlp_hidden_dims", type=int, nargs='+', default=None,
                       help="Hidden dimensions for each MLP layer")

    #params for GAT
    parser.add_argument("--reduction", type=str, default="concate")
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=0.2)

    # Cross-Modal Attention parameters
    parser.add_argument("--use_cross_attention", action="store_true", default=False,
                    help="Use aligned cross-modal attention instead of concatenation")
    parser.add_argument("--cross_attention_heads", type=int, default=8,
                    help="Attention heads (scfm_dim=896 must be divisible). "
                         "Valid: 1,2,4,7,8,14,16,...")
    parser.add_argument("--cross_attention_output_dim", type=int, default=None,
                    help="Optional: compress 896-dim fused output before MLP. "
                         "None = keep at scfm_dim (896).")
    parser.add_argument("--cross_attention_query", type=str, default="scfm",
                    choices=["scfm", "gnn"],
                    help="Which modality is the Query in cross-attention. "
                         "'scfm' = scFM queries GNN (scFM gets updated). "
                         "'gnn' = GNN queries scFM (GNN gets updated). Ablation flag.")

    # PPI dual-encoder / multi-view parameters
    parser.add_argument("--use_ppi", action="store_true", default=False,
                    help="Add a second GNN encoder branch over the PPI adjacency")
    parser.add_argument("--multi_view", action="store_true", default=False,
                    help="Method 2: interleave GRN/PPI encoders layer-by-layer with cross-view "
                         "guidance instead of Method 1's late fusion. Requires --use_ppi.")
    parser.add_argument("--ppi_fusion_type", type=str, default="concat",
                    choices=["concat", "gate", "attention"],
                    help="How to fuse embed_grn and embed_ppi into embed_graph")
    parser.add_argument("--ppi_fusion_heads", type=int, default=1,
                    help="Attention heads for --ppi_fusion_type attention; must divide the "
                         "GNN output dim")
    parser.add_argument("--view_guidance", type=str, default="symmetric",
                    choices=["symmetric", "grn_guides_ppi", "ppi_guides_grn"],
                    help="Multi-view (Method 2) only: which view guides/weights the other "
                         "during per-layer cross-view interaction")

    # Ablation: drop one modality entirely to measure its contribution
    parser.add_argument("--ablation", type=str, default="full",
                    choices=["full", "no_scfm", "no_grn"],
                    help="'no_scfm' = MLP head sees only the GNN graph embedding. "
                         "'no_grn' = MLP head sees only the raw scFM gene embedding.")

    args = parser.parse_args()
    return args


def save_args(args, dir):
    # if int(os.getenv("RANK", -1)) <= 0:
    FILE_NAME = "args.json"
    with open(os.path.join(dir, FILE_NAME), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logger.info("args saved to {}".format(os.path.join(dir, FILE_NAME)))


def load_args(dir):
    with open(os.path.join(dir, "args.json"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    return args


if __name__ == "__main__":
    args = parse_args()
