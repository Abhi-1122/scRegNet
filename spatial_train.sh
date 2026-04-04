#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# spatial_train.sh
#
# Run scRegNet-Spatial on hHEP Visium data.
#
# Usage:
#   bash spatial_train.sh [--gpu 0]
#
# Before first run, build the gene-spot mask:
#   cd hHEP_spatial && python build_gene_spot_mask.py && cd ..
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Default hyper-parameters (tune with Optuna if desired) ──────────────────
GPU=${1:-0}
SEED=42
CELL_TYPE="hHEP"
NUM_TF="500"
LLM_TYPE="Geneformer"
GNN_TYPE="GCN"

GNN_EPOCHS=300
GNN_EVAL_INTERVAL=5
# ── Best HPs from Optuna trial #190  (AUROC 0.904026) ─────────────────────
BATCH_SIZE=131
GNN_LR=1.36407e-05
GNN_WEIGHT_DECAY=1.6446e-05
DROPOUT=0.110489

# GCN layers: 3 layers  (Optuna-tuned, study tf_500_hHEP_spatial_GCN_Geneformer)
GNN_HIDDEN_DIMS="256 45 187"
# MLP layers: 1 layer
MLP_HIDDEN_DIMS="165"

# Spatial GNN dimensions
SPATIAL_GNN_HIDDEN=44
SPATIAL_GNN_OUT=69

# Paths
DATA_FOLDER="./data"
SCFM_FOLDER="./scFM"
SPATIAL_DATA_FOLDER="./hHEP_spatial"
OUTPUT_DIR="./output/spatial_hHEP_${NUM_TF}"
CKPT_DIR="./ckpt/spatial_hHEP_${NUM_TF}"

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"

echo "════════════════════════════════════════════"
echo "  scRegNet-Spatial  |  hHEP  |  TFs+${NUM_TF}"
echo "════════════════════════════════════════════"

python -m src.train \
    --use_spatial \
    --cell_type        "$CELL_TYPE" \
    --num_TF           "$NUM_TF" \
    --llm_type         "$LLM_TYPE" \
    --gnn_type         "$GNN_TYPE" \
    --data_folder      "$DATA_FOLDER" \
    --scFM_folder      "$SCFM_FOLDER" \
    --spatial_data_folder "$SPATIAL_DATA_FOLDER" \
    --output_dir       "$OUTPUT_DIR" \
    --ckpt_dir         "$CKPT_DIR" \
    --gnn_epochs       "$GNN_EPOCHS" \
    --gnn_eval_interval "$GNN_EVAL_INTERVAL" \
    --batch_size       "$BATCH_SIZE" \
    --gnn_lr           "$GNN_LR" \
    --gnn_weight_decay "$GNN_WEIGHT_DECAY" \
    --dropout          "$DROPOUT" \
    --gnn_num_layers   3 \
    --gnn_hidden_dims  $GNN_HIDDEN_DIMS \
    --mlp_num_layers   1 \
    --mlp_hidden_dims  $MLP_HIDDEN_DIMS \
    --spatial_gnn_hidden "$SPATIAL_GNN_HIDDEN" \
    --spatial_gnn_out    "$SPATIAL_GNN_OUT" \
    --random_seed      "$SEED" \
    --single_gpu       "$GPU" \
    --flag True \
    --type MLP

echo ""
echo "✅  Done. Results in: $OUTPUT_DIR"
