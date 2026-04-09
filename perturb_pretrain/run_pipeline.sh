#!/usr/bin/env bash
# Full pipeline runner — run from the directory containing your data files.
# Expects: Label.csv, BL-network.csv, (optionally fm_embeddings.pt)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# echo "=== Step 0: Build FM embeddings ==="
# python3 00_build_fm_embeddings.py

# echo "=== Step 1: Prepare Perturb-seq triples ==="
# python3 01_prepare_perturb_data.py

# echo "=== Step 0b: Build edge index ==="
# python3 00b_build_edge_index.py

echo "=== Step 2: Pretrain encoder on Norman data ==="
python3 03_pretrain.py

echo "=== Step 3: Fine-tune baseline + pretrained on BEELINE ==="
python3 04_finetune.py

echo "=== Step 4: Evaluate and visualise ==="
python3 05_evaluate.py

echo "Done. Results in ./results/"
