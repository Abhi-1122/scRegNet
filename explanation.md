# scRegNet - Project Documentation

## Overview
**scRegNet** is a deep learning framework for predicting gene regulatory connections using single-cell RNA-seq data. It combines single-cell foundation models (scFMs) with graph neural networks (GNNs) to predict regulatory links between transcription factors (TFs) and target genes.

---

## Project Structure

### Root Files

#### `README.md`
- Project overview and installation instructions
- Citation information
- Links to download pre-trained Geneformer embeddings
- Basic usage examples

#### `run_optuna.py`
Main entry point for hyperparameter optimization using Optuna:
- Loads arguments from command line
- Selects appropriate HP search class based on GNN type (GAT or GCN/GraphSAGE)
- Runs Optuna trials or loads existing studies
- Supports both single-GPU and distributed training

#### `gnn_hp.sh`
Wrapper script for running hyperparameter search:
- **Usage**: `bash gnn_hp.sh <dataset> <gnn_type> <cell_type> <num_TF> <scFM_type>`
- **Example**: `bash gnn_hp.sh tf_500_mDC GCN mDC 500 Geneformer`
- Takes 5 arguments and passes them to `optuna.sh`

#### `optuna.sh`
Shell script that:
- Sets up output directories for storing model checkpoints and logs
- Calls `run_optuna.py` with appropriate arguments
- Redirects output to log file

---

## Source Code (`src/`)

### `src/__init__.py`
Empty initialization file for Python package structure.

### `src/args.py`
Command-line argument parser and configuration:
- **Environment settings**: GPU selection, random seed
- **Data paths**: data folder, scFM embeddings folder, output/checkpoint directories
- **Training hyperparameters**: batch size, learning rate, epochs, optimizer
- **GNN hyperparameters**: number of layers, hidden dimensions, dropout
- **Optuna settings**: number of trials, expected validation accuracy
- **Task-specific parameters**: cell type, dataset name, GNN type (GCN/GraphSAGE/GAT), scFM type (Geneformer/scBERT/scFoundation)
- **Functions**: `parse_args()`, `save_args()`, `load_args()`

### `src/models.py`
Neural network architectures:

#### `AttentionLayer`
Custom GAT attention layer with:
- Learnable weight matrices
- Leaky ReLU activation
- Attention mechanism for neighborhood aggregation

#### `scTransNet_GCN`
GCN-based model:
- **Encoder**: Multiple GCN layers for graph feature extraction
- **Decoder**: MLP/dot product/cosine similarity for link prediction
- **Forward pass**: Combines scFM embeddings with GCN output, generates TF and target embeddings

#### `scTransNet_SAGE`
GraphSAGE-based model:
- Similar architecture to GCN but uses SAGE convolutions
- Better for large-scale graphs with sampling

#### `scTransNet_GAT`
GAT-based model:
- Uses attention mechanism with multiple heads
- Supports concatenation or mean reduction of attention heads
- Customizable attention parameters (alpha)

### `src/train.py`
Training pipeline:

#### `Trainer` class
- **`_get_embeddings()`**: Loads gene embeddings from scFMs (Geneformer, scBERT, or scFoundation)
- **`_prepare_data()`**: Loads expression data, train/test splits, TF/target indices, creates adjacency matrix
- **`get_model()`**: Instantiates appropriate model (GCN/SAGE/GAT)
- **`train()`**: 
  - Main training loop with early stopping
  - Evaluates every N epochs
  - Saves best model based on AUC
  - Returns best AUROC and AUPRC scores

### `src/inference.py`
Evaluation and inference:

#### `Infer` class
- Similar data preparation to Trainer
- **`infer()`**: 
  - Loads trained model from checkpoint
  - Runs 50 inference iterations for robustness
  - Computes AUC and AUPR metrics
  - Returns results for train and test sets

### `src/utils.py`
Utility functions and classes:

#### `scRNADataset`
PyTorch Dataset class:
- Handles TF-Target pairs with labels
- **`Adj_Generate()`**: Creates adjacency matrix from training edges
- Supports directional and self-loop options

#### `load_data`
Data loading and preprocessing:
- StandardScaler normalization of expression data
- Returns normalized expression matrix

#### Helper functions
- **`adj2saprse_tensor()`**: Converts scipy sparse matrix to PyTorch sparse tensor
- **`Evaluation()`**: Computes AUC, AUPR, and normalized AUPR
- **`set_seed()`**: Sets random seeds for reproducibility
- **`set_logging()`**: Configures colored console logging
- **`store_results()`**: Saves evaluation results to JSON

### `src/optuna/HP_search.py`
Hyperparameter optimization base classes:

#### `HP_search` (Abstract)
Base class defining:
- `train()`: Wrapper for training with trial parameters
- `setup_search_space()`: Abstract method for defining search space
- `load_study()`: Loads existing Optuna study

#### `Single_HP_search`
- Single-GPU hyperparameter search
- Uses Optuna's SuccessiveHalvingPruner
- Integrates with Weights & Biases for logging
- Saves best trial results

#### `Dist_HP_search`
- Distributed hyperparameter search (for multi-GPU setups)
- Broadcasts arguments across processes

### `src/optuna/search_space.py`
Defines hyperparameter search spaces:

#### `GNN_HP_search`
For GCN and GraphSAGE:
- Learning rate: 1e-5 to 1e-2 (log scale)
- Weight decay: 1e-7 to 1e-4 (log scale)
- Dropout: 0.1 to 0.8
- Number of GNN layers: 1 to 6
- Number of MLP layers: 1 to 6
- Hidden dimensions per layer: 4 to 256
- Batch size: 32 to 256
- Optimizer: Adam/RMSprop/SGD

#### `GAT_HP_search`
Additional GAT-specific parameters:
- Number of attention heads per layer: 1 to 8
- Reduction method: concatenate or mean
- Alpha (LeakyReLU slope): 0.01 to 0.5

---

## Data (`data/`)

The project includes benchmark datasets from BEELINE for 4 cell types, each with 2 network sizes:

### Cell Types
1. **hESC** - Human Embryonic Stem Cells
2. **hHEP** - Human Hepatocytes
3. **mESC** - Mouse Embryonic Stem Cells
4. **mHSC-E** - Mouse Hematopoietic Stem Cells

### Network Sizes
- **TFs+500**: 500 target genes + transcription factors
- **TFs+1000**: 1000 target genes + transcription factors

### Data Files (per dataset)

#### `BL--ExpressionData.csv`
- Single-cell RNA-seq expression matrix
- Rows: Genes (TFs + targets)
- Columns: Single cells
- Values: Normalized gene expression levels
- **Format**: CSV with gene names as row index, cell IDs as column headers

#### `BL--network.csv`
- Ground truth gene regulatory network
- **Columns**: `Gene1` (TF), `Gene2` (Target)
- Contains known regulatory interactions from literature/databases
- Used as gold standard for evaluation

#### `Train_set.csv`
- Training data for link prediction
- **Columns**: `TF`, `Target`, `Label`
- TF/Target are integer indices mapping to genes
- Label: 1 (regulatory link exists) or 0 (no link)
- Typically contains both positive and negative samples

#### `Test_set.csv`
- Test data for evaluation
- Same format as Train_set.csv
- Held-out edges for performance evaluation

#### `TF.csv`
- List of transcription factors
- **Columns**: `TF` (gene name), `index` (integer position)
- Maps TF names to their row indices in expression matrix

#### `Target.csv`
- List of all genes (TFs + targets)
- **Columns**: `Gene` (gene name), `index` (integer position)
- Maps gene names to row indices in expression matrix

#### `Label.csv`
- Known regulatory interactions
- **Columns**: `TF`, `Target` (both as indices)
- Subset of positive interactions from ground truth network

---

## Output Structure (`out/`)

Not included in repository but created during training:
```
out/
├── <GNN_TYPE>/           # e.g., GCN, GraphSAGE, GAT
│   └── <scFM_TYPE>/      # e.g., Geneformer, scBERT
│       └── <DATASET>/    # e.g., tf_500_hESC
│           ├── ckpt/     # Model checkpoints
│           │   ├── model_seed*.pt
│           │   └── args.json
│           ├── best/     # Best trial results
│           └── log.txt   # Training logs
```

---

## Single-Cell Foundation Model Embeddings (`scFM/`)

**Not included in repository** - Download from [Google Drive](https://drive.google.com/drive/folders/1xnh4ixJwx1kzmO98FmGUvy5S7uqLW-yR?usp=sharing)

Expected structure:
```
scFM/
├── Geneformer/
│   ├── hESC_500_gene_embeddings.csv
│   ├── hESC_500.csv
│   └── ... (for other datasets)
├── scBERT/
│   └── <cell_type>_<num_TF>_cell_embeddings.npy
└── scFoundation/
    ├── OS_scRNA_gene_index.19264.tsv
    └── genemodule_<cell_type>_<num_TF>_singlecell_gene_embedding_f2_resolution.npy
```

### Embedding Types

1. **Geneformer**: Gene-level embeddings from transformer-based foundation model
2. **scBERT**: Cell-level embeddings averaged to gene level
3. **scFoundation**: Gene module embeddings from foundation model

---

## How to Run

### Prerequisites
```bash
# Python 3.10
pip install torch==2.4.1
pip install scikit-learn==1.5.2
pip install numpy==1.20.3
pip install optuna==4.0.0
pip install wandb  # Optional for logging
```

### Quick Demo (Inference)
```bash
python src/inference.py
```
This runs inference on pre-trained model for hESC dataset with Geneformer embeddings.

### Train New Model
```bash
bash gnn_hp.sh <dataset> <gnn_type> <cell_type> <num_TF> <scFM_type>
```

**Example:**
```bash
bash gnn_hp.sh tf_500_hESC GCN hESC 500 Geneformer
```

**Parameters:**
- `dataset`: Dataset name (e.g., tf_500_hESC)
- `gnn_type`: GNN architecture (GCN, GraphSAGE, or GAT)
- `cell_type`: Cell type (hESC, hHEP, mESC, mHSC-E)
- `num_TF`: Network size (500 or 1000)
- `scFM_type`: Foundation model (Geneformer, scBERT, scFoundation)

### Advanced Options

Direct Python execution:
```bash
python run_optuna.py \
    --dataset tf_500_hESC \
    --gnn_type GCN \
    --llm_type Geneformer \
    --cell_type hESC \
    --num_TF 500 \
    --n_trials 50 \
    --gnn_epochs 300 \
    --batch_size 256 \
    --single_gpu 0
```

---

## Model Architecture

### Overall Pipeline
1. **Input**: Single-cell expression data + scFM embeddings
2. **Graph Construction**: Build adjacency matrix from training edges
3. **GNN Encoding**: Extract graph-based features
4. **Feature Fusion**: Concatenate scFM embeddings with GNN features
5. **MLP Projection**: Project to TF/Target embedding spaces
6. **Link Prediction**: Decode TF-Target pairs using MLP/dot/cosine

### Key Components
- **scFM embeddings**: Pre-computed gene representations from foundation models
- **GNN layers**: Aggregate neighborhood information
- **MLP layers**: Transform combined features
- **Decoder**: Predicts regulatory link probability

---

## Evaluation Metrics

- **AUROC (AUC)**: Area Under ROC Curve - measures overall classification performance
- **AUPRC (AUPR)**: Area Under Precision-Recall Curve - better for imbalanced data
- **Normalized AUPR**: AUPR normalized by positive class ratio

---

## Key Features

1. **Multi-scale GNN support**: GCN, GraphSAGE, GAT
2. **Multi-foundation model support**: Geneformer, scBERT, scFoundation
3. **Automated hyperparameter tuning**: Optuna with pruning
4. **Early stopping**: Prevents overfitting
5. **Multiple datasets**: 4 cell types × 2 network sizes = 8 benchmarks
6. **Reproducible**: Seed control and deterministic training

---

## Important Notes

1. **GPU strongly recommended** for training (CUDA 12.4 used in development)
2. **Download scFM embeddings** before running (not included in repo)
3. **Data format**: Expression data is normalized using StandardScaler
4. **Adjacency matrix**: Built from positive training samples only
5. **Evaluation**: 50 inference runs for robust metrics

---

## Citation

If using this code, please cite:
```
@article {Kommu2024.12.16.628715,
    author = {Kommu, Sindhura and Wang, Yizhi and Wang, Yue and Wang, Xuan},
    title = {Prediction of Gene Regulatory Connections with Joint Single-Cell 
             Foundation Models and Graph-Based Learning},
    year = {2025},
    doi = {10.1101/2024.12.16.628715},
    journal = {bioRxiv}
}
```

---

## Troubleshooting

**Common Issues:**

1. **Missing scFM embeddings**: Download from Google Drive link in README
2. **GPU memory error**: Reduce batch size or use smaller network (500 vs 1000)
3. **CUDA version mismatch**: Install PyTorch compatible with your CUDA version
4. **File not found**: Ensure data folders follow exact structure shown above

**Contact**: Refer to paper for author contact information
