# Perturbation-Informed Pretraining for scRegNet

Improve scRegNet's GRN inference by pretraining the gene encoder
on Norman 2019 Perturb-seq causal response data before fine-tuning
on BEELINE hESC/hHEP benchmarks.

## Idea in one sentence
Use TF-knockdown → target-gene-response triples as a weak supervision
signal to teach the encoder what "causal regulation looks like",
then fine-tune on your noisy ChIP/curated GRN labels.

## Pipeline

```
Step 0   00_build_fm_embeddings.py  ← one-time: generate FM embeddings for all genes
Step 0b  00b_build_edge_index.py    ← one-time: convert BL-network.csv to edge_index.pt
Step 1   01_prepare_perturb_data.py ← load Norman via GEARS, compute DE, save triples
Step 2   03_pretrain.py             ← pretrain FusedGeneEncoder on triples
Step 3   04_finetune.py             ← fine-tune baseline vs pretrained on BEELINE (5-fold CV)
Step 4   05_evaluate.py             ← compare, bar chart, overlap ablation
```

Run all: `bash run_pipeline.sh`

## File layout expected
```
your_project/
├── Label.csv              ← BEELINE ground truth (TF, Target, Label)
├── BL-network.csv         ← TF-target prior network
├── fm_embeddings.pt       ← {gene: tensor(fm_dim,)} — from your scFM
├── TF.csv                 ← your TF list
└── perturb_pretrain_pipeline/   ← this folder
```

## Key files

| File | Purpose |
|---|---|
| `02_encoder.py` | Shared FusedGeneEncoder (FM + GCN), also CrossAttentionFusedEncoder |
| `03_pretrain.py` | Pretraining loop on perturbation triples |
| `04_finetune.py` | 5-fold CV fine-tuning, baseline vs pretrained |
| `05_evaluate.py` | Bar charts + overlap ablation (the key result for paper) |

## Ablation story (for paper)
The overlap ablation in Step 4 is the strongest experiment:
- Overlap TFs (in Norman): does pretraining help MORE for TFs that were actually perturbed?
- Non-overlap TFs (not in Norman): does pretraining hurt / neutral for unseen TFs?
- Expected result: Δ AUROC should be larger for overlap TFs → pretraining is causally learning
  TF-specific patterns, not just memorizing data.

## Requirements
```
pip install gears torch torch-geometric transformers scanpy statsmodels scikit-learn matplotlib pandas
```
