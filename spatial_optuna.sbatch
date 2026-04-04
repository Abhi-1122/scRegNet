#!/usr/bin/env bash
#SBATCH --job-name=scRegNet_spatial_optuna
#SBATCH --output=/home2/garimella.abhishek/scRegNet/logs/spatial_optuna_%j.out
#SBATCH --error=/home2/garimella.abhishek/scRegNet/logs/spatial_optuna_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20

# ── adjust these to match your cluster's partition / account ──────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=garimella.abhishek@research.iiit.ac.in

set -euo pipefail

# ── wandb authentication ───────────────────────────────────────────────────
export WANDB_API_KEY="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"

# ── paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR="/home2/garimella.abhishek/scRegNet"
source /home2/garimella.abhishek/venv/bin/activate
cd "$PROJECT_DIR"

# ── Optuna spatial HP search ───────────────────────────────────────────────
CELL_TYPE="hHEP"
NUM_TF="500"
LLM_TYPE="Geneformer"
GNN_TYPE="GCN"
N_TRIALS=50        # total trials shared across 4 workers → ~12-13 each

OUTPUT_DIR="$PROJECT_DIR/out/spatial_optuna/${GNN_TYPE}/${LLM_TYPE}/tf_${NUM_TF}_${CELL_TYPE}_spatial"
CKPT_DIR="${OUTPUT_DIR}/ckpt"
mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"

echo "════════════════════════════════════════════"
echo "  scRegNet-Spatial Optuna  |  hHEP  |  TFs+${NUM_TF}"
echo "  n_trials=${N_TRIALS}  workers=4"
echo "════════════════════════════════════════════"

# Common args for all workers
COMMON_ARGS="
    --use_spatial
    --cell_type          $CELL_TYPE
    --num_TF             $NUM_TF
    --llm_type           $LLM_TYPE
    --gnn_type           $GNN_TYPE
    --data_folder        $PROJECT_DIR/data
    --scFM_folder        $PROJECT_DIR/scFM
    --spatial_data_folder $PROJECT_DIR/hHEP_spatial
    --output_dir         $OUTPUT_DIR
    --ckpt_dir           $CKPT_DIR
    --dataset            tf_${NUM_TF}_${CELL_TYPE}_spatial
    --gnn_epochs         200
    --gnn_eval_interval  5
    --n_trials           $N_TRIALS
    --start_seed         42
    --flag               True
    --type               MLP
    --loop               False
"

# Launch 4 workers in parallel, each on a different GPU, sharing the SQLite study
for GPU_ID in 0 1 2 3; do
    python run_optuna.py $COMMON_ARGS \
        --single_gpu   $GPU_ID \
        --random_seed  $((42 + GPU_ID)) \
        > "${OUTPUT_DIR}/optuna_worker${GPU_ID}.log" 2>&1 &
done

# Wait for all workers to finish
wait

echo "✅  Done. Best trial saved to: ${OUTPUT_DIR}/best"
