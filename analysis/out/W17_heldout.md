# W17: the curated pipeline on held-out genes and held-out TFs, all four cells

Date 2026-11-13. Code: `analysis/w17_jobs.py` (job files), `analysis/w17_report.py` (all tables, `analysis/out/w17_heldout.txt`), `analysis/w16_dcval.py` and `analysis/w12_stat.py` (new `--cell` for held-out genes, new `--skip_lsm`; both default to the old behaviour). Runs `analysis/out/pair/w17_*`, jobs `analysis/jobs/w17_*.txt`. 180 new GNN runs (3 cells x 2 protocols x 3 arms x 5 folds x seeds 42, 43) plus 10 reruns of the hESC pipeline on held-out TFs with the validation grid saved.

## Verdict

- **Pipeline against the scRegNet baseline, 12 cell/protocol combinations (within-TF AUROC, paired per-TF Wilcoxon, p < 0.05 counts as a difference):** beats it in 9, ties in 2 (hESC and mESC held-out TFs), loses in 1 (hHEP held-out genes, -0.0125, p = 0.012).
- **Held-out genes:** the DC-SBM blend pipeline beats the Adam-tuned scRegNet design on hESC (+0.062), mHSC-E (+0.043) and mESC (+0.068) and loses on hHEP (-0.013). It beats the hero on all four cells (p = 0.11 on hESC, otherwise p < 0.05).
- **Held-out TFs:** the pipeline now scores above the Adam-tuned scRegNet design on hHEP (+0.034, p = 0.016) and mHSC-E (+0.032, p = 0.001) and ties on hESC (+0.008, p = 0.053) and mESC (+0.005, p = 0.96). All of this comes from the popularity score, not the GNN: popularity alone is at least as good as the blend on every cell (blend minus popularity +0.0004, -0.0037, -0.0059, -0.0134; significant on hHEP and mESC) and beats the scRegNet design on all four cells (p = 0.04, 3e-4, 2e-4, 1e-5). The GNN adds nothing on unseen TFs.
- **Measured, not guessed:** every number above. **Guesses** are marked as such below.

## Part 1: popularity blend, hESC held-out TFs

Method. For an unseen TF the pipeline GNN's within-TF rank is averaged with the within-TF rank of gene popularity (the number of training TFs that have the gene as a positive target, from training positives only, log1p; the monotone transform does not change ranks). The weight w on popularity is in {0, 0.25, 0.5, 0.75, 1}. It was chosen per fold on the fold's 3 validation TFs (the TF protocol's validation split, `prepare` in `src/pair_gnn.py`): GNN scores from the best-epoch model before the refit (`--save_sel_grid 1`, validation TFs' edges not in its graph), popularity computed from the training rows minus the validation TFs' edges, criterion mean within-TF AUROC over the two seeds. The refit test scores are blended with popularity computed from all training rows. The existing `w16_ho_tf*` runs had no saved grid, so the pipeline GNN (GCN 1 layer, Hadamard, Geneformer, decoder wd 0) was rerun with the flag as `w17_hESC_ho_tf*_pipe`; its refit test within-TF is 0.6860 against 0.6850 for the W16 run. The saved grid reproduces each run's recorded validation within-TF (checked in the script).

hESC held-out TFs, 34 TFs, 5 folds x seeds 42, 43. AUROC / AUPRC / within-TF / within>=8:

| Model | AUROC / AUPRC / within-TF / within>=8 | Paired per-TF difference, within-TF (p), pipeline minus the row |
|---|---|---|
| **Pipeline: GNN + popularity, w chosen per fold on validation** | 0.6685 / 0.2332 / 0.7248 / 0.7130 | |
| w chosen on validation pooled over the 5 folds (w = 1) | 0.6689 / 0.2316 / 0.7244 / 0.7120 | +0.0004 (1.0) |
| GNN alone (w = 0) | 0.5585 / 0.1921 / 0.6860 / 0.6665 | +0.0395 (3e-5) |
| Popularity alone (w = 1) | 0.6689 / 0.2316 / 0.7244 / 0.7120 | +0.0004 (1.0) |
| Sensitivity: fixed w = 0.5 (not validation-chosen) | 0.6261 / 0.2179 / 0.7193 / 0.7034 | +0.0058 (0.018) |
| scRegNet design, Adam-tuned | 0.6009 / 0.2303 / 0.7163 / 0.7045 | +0.0084 (0.053); within>=8 +0.0084 (0.072) |
| Hero (SAGE, no Geneformer) | 0.6304 / 0.2314 / 0.6855 / 0.6766 | +0.0391 (1e-6) |

Weights chosen per fold: 0.75, 1, 1, 1, 1 (validation within-TF of the blend rises from w = 0 to w = 1 on 4 of 5 folds: mean over folds 0.6968, 0.7227, 0.7457, 0.7594, 0.7626). With 3 validation TFs the choice is noisy, but here it is nearly unanimous and agrees with the test result: the GNN scores only hurt. The blend with w chosen on validation is the popularity ranking in all but one fold. Result: the pipeline moves from -0.0320 (GNN alone, W16) to +0.0084 against the Adam-tuned design, a tie by the p < 0.05 rule, and from -0.0014 to +0.0391 against the hero. Within the >= 8 positives subset: 0.7130 against 0.7045.

## Part 2: every cell, both held-out protocols

Setup. Seeds 42, 43 x 5 folds per cell and protocol. **Pipeline arm** (per cell, validation-chosen in W16, `analysis/out/w16_choose.txt`): hESC and hHEP GCN 1 layer, Hadamard, batch split, Geneformer, decoder wd 0; mHSC-E neural PANDA 2 layers, batch split; mESC neural PANDA 2 layers, no split; all Hadamard, Geneformer, Adam lr 1e-4, wd 1e-3, refit, `--save_sel_grid 1`. **Held-out genes:** rank blend with the fixed-cluster DC-SBM with Geneformer soft blocks (`analysis/w12_stat.py --protocol gene --skip_lsm 1`; K, covariance, smoothing chosen on the validation genes per fold; K picked: hESC 10/10/20/10/5, hHEP 5/20/5/40/40, mHSC-E 20/5/20/10/20, mESC 10/5/20/40/5). Blend weight w in {0, 0.25, 0.5, 0.75} chosen on the held-out validation genes pooled over the 10 runs of the cell, with the DC-SBM refit without the validation rows (`analysis/w16_dcval.py --kind hog`); the same recipe as W16 hESC. All four cells chose w = 0.5 (validation within-TF of the blend, w = 0 to 0.75: hHEP 0.6297 / 0.6469 / 0.6580 / 0.6537, mHSC-E 0.7070 / 0.7202 / 0.7243 / 0.7100, mESC 0.6840 / 0.6969 / 0.7059 / 0.7043, hESC 0.6142 / 0.6399 / 0.6635 / 0.6610). **Held-out TFs:** popularity blend as in Part 1. **Baselines:** the scRegNet design, Adam-tuned (the arguments of `ho_*_f01_A_tuned`: GCN, add decoder, 3 layers 227/29/34, Geneformer, no split, Adam lr 4.29e-5, wd 0.002, dropout 0.481, batch 512, only cell, protocol, fold and seed changed) and the hero (`w5_ho_*_hero` arguments, SAGE, no Geneformer). hESC reuses the existing baseline and W16 runs; the three other cells were run new. The published SGD recipe has no held-out runs, so "scRegNet design" means the Adam-tuned version everywhere on held-out data.

Compute. All GNN runs took under 1 minute on the laptop or ada except mESC (Adam-tuned 0.6 min on genes and 2.2 on TFs, pipeline 2.9 on TFs, hero 7 to 13 min on TFs). The DC-SBM fits were about 1 minute per fold on the CPU. Total wall time of Part 2 about 35 minutes. mHSC-E and the mESC tuned and pipeline arms ran on ada; hHEP and the hESC reruns on the laptop; the mESC hero and the end of the mESC pipeline arm were split across both machines (laptop in reverse order), so a few mESC runs come from the laptop. Earlier checks found no ada-versus-laptop bias; epoch selection noise is about 0.02 per run.

## Part 2 tables

**Held-out genes.** AUROC / AUPRC / within-TF / within>=8, mean over 5 folds x seeds 42, 43. The last two columns are paired per-TF differences (pipeline minus the arm, Wilcoxon p).

| Cell | Arm | AUROC / AUPRC / within-TF / within>=8 | pipeline - scRegNet design (p) | pipeline - hero (p) |
|---|---|---|---|---|
| hESC | **pipeline** (GNN + DC-SBM, w=0.5) | 0.8598 / 0.5517 / 0.5783 / 0.5944 | +0.0620 (0.0021) | +0.0161 (0.11) |
| hESC | GNN alone | 0.8380 / 0.5180 / 0.5576 / 0.5832 | | |
| hESC | DC-SBM Geneformer soft alone | 0.8573 / 0.5379 / 0.5753 / 0.5782 | | |
| hESC | scRegNet design, Adam-tuned | 0.8328 / 0.5275 / 0.5241 / 0.5591 | | |
| hESC | hero (SAGE, no Geneformer) | 0.8467 / 0.5213 / 0.5461 / 0.5617 | | |
| hHEP | **pipeline** (GNN + DC-SBM, w=0.5) | 0.8426 / 0.7205 / 0.6377 / 0.6377 | -0.0125 (0.012) | +0.0464 (0.00095) |
| hHEP | GNN alone | 0.8356 / 0.7112 / 0.6314 / 0.6414 | | |
| hHEP | DC-SBM Geneformer soft alone | 0.8263 / 0.6810 / 0.5981 / 0.5893 | | |
| hHEP | scRegNet design, Adam-tuned | 0.8447 / 0.7226 / 0.6477 / 0.6600 | | |
| hHEP | hero (SAGE, no Geneformer) | 0.8246 / 0.6850 / 0.5908 / 0.6039 | | |
| mHSC-E | **pipeline** (GNN + DC-SBM, w=0.5) | 0.8874 / 0.9068 / 0.6881 / 0.6939 | +0.0434 (0.023) | +0.0118 (0.036) |
| mHSC-E | GNN alone | 0.8861 / 0.9060 / 0.6905 / 0.6996 | | |
| mHSC-E | DC-SBM Geneformer soft alone | 0.8757 / 0.8890 / 0.6359 / 0.6360 | | |
| mHSC-E | scRegNet design, Adam-tuned | 0.8831 / 0.8977 / 0.6461 / 0.6681 | | |
| mHSC-E | hero (SAGE, no Geneformer) | 0.8796 / 0.9003 / 0.6688 / 0.6758 | | |
| mESC | **pipeline** (GNN + DC-SBM, w=0.5) | 0.8961 / 0.7966 / 0.6724 / 0.6864 | +0.0682 (1.6e-12) | +0.0441 (1.5e-07) |
| mESC | GNN alone | 0.8635 / 0.6998 / 0.6559 / 0.6701 | | |
| mESC | DC-SBM Geneformer soft alone | 0.8968 / 0.8000 / 0.6597 / 0.6707 | | |
| mESC | scRegNet design, Adam-tuned | 0.8692 / 0.7468 / 0.6094 / 0.6266 | | |
| mESC | hero (SAGE, no Geneformer) | 0.8515 / 0.6876 / 0.6247 / 0.6326 | | |

**Held-out TFs.** AUROC / AUPRC / within-TF / within>=8, mean over 5 folds x seeds 42, 43. The last two columns are paired per-TF differences (pipeline minus the arm, Wilcoxon p).

| Cell | Arm | AUROC / AUPRC / within-TF / within>=8 | pipeline - scRegNet design (p) | pipeline - hero (p) |
|---|---|---|---|---|
| hESC | **pipeline** (GNN + popularity, w chosen per fold on validation) | 0.6685 / 0.2332 / 0.7248 / 0.7130 | +0.0084 (0.053) | +0.0391 (1e-06) |
| hESC | pipeline, w chosen on validation pooled over folds (w=1.0) | 0.6689 / 0.2316 / 0.7244 / 0.7120 | | |
| hESC | GNN alone (w=0) | 0.5585 / 0.1921 / 0.6860 / 0.6665 | | |
| hESC | popularity alone (w=1) | 0.6689 / 0.2316 / 0.7244 / 0.7120 | | |
| hESC | sensitivity, fixed w=0.5 (NOT validation-chosen) | 0.6261 / 0.2179 / 0.7193 / 0.7034 | | |
| hESC | scRegNet design, Adam-tuned | 0.6009 / 0.2303 / 0.7163 / 0.7045 | | |
| hESC | hero (SAGE, no Geneformer) | 0.6304 / 0.2314 / 0.6855 / 0.6766 | | |
| hHEP | **pipeline** (GNN + popularity, w chosen per fold on validation) | 0.7189 / 0.5083 / 0.7748 / 0.7731 | +0.0343 (0.015) | +0.0288 (1.4e-06) |
| hHEP | pipeline, w chosen on validation pooled over folds (w=0.75) | 0.7251 / 0.5203 / 0.7760 / 0.7743 | | |
| hHEP | GNN alone (w=0) | 0.6367 / 0.4492 / 0.7246 / 0.7211 | | |
| hHEP | popularity alone (w=1) | 0.7349 / 0.5269 / 0.7785 / 0.7771 | | |
| hHEP | sensitivity, fixed w=0.5 (NOT validation-chosen) | 0.7024 / 0.5005 / 0.7650 / 0.7629 | | |
| hHEP | scRegNet design, Adam-tuned | 0.6187 / 0.4332 / 0.7405 / 0.7393 | | |
| hHEP | hero (SAGE, no Geneformer) | 0.6789 / 0.4912 / 0.7460 / 0.7473 | | |
| mHSC-E | **pipeline** (GNN + popularity, w chosen per fold on validation) | 0.6678 / 0.6535 / 0.7554 / 0.7722 | +0.0315 (0.00097) | +0.0354 (0.0032) |
| mHSC-E | pipeline, w chosen on validation pooled over folds (w=0.5) | 0.6620 / 0.6516 / 0.7551 / 0.7720 | | |
| mHSC-E | GNN alone (w=0) | 0.6228 / 0.6276 / 0.7336 / 0.7490 | | |
| mHSC-E | popularity alone (w=1) | 0.6828 / 0.6710 / 0.7609 / 0.7765 | | |
| mHSC-E | sensitivity, fixed w=0.5 (NOT validation-chosen) | 0.6620 / 0.6516 / 0.7551 / 0.7720 | | |
| mHSC-E | scRegNet design, Adam-tuned | 0.6406 / 0.6533 / 0.7254 / 0.7430 | | |
| mHSC-E | hero (SAGE, no Geneformer) | 0.6425 / 0.6634 / 0.7187 / 0.7296 | | |
| mESC | **pipeline** (GNN + popularity, w chosen per fold on validation) | 0.6493 / 0.3994 / 0.7555 / 0.7466 | +0.0045 (0.96) | +0.0293 (0.00012) |
| mESC | pipeline, w chosen on validation pooled over folds (w=0.5) | 0.6789 / 0.4265 / 0.7658 / 0.7580 | | |
| mESC | GNN alone (w=0) | 0.6527 / 0.4057 / 0.7364 / 0.7283 | | |
| mESC | popularity alone (w=1) | 0.6759 / 0.4210 / 0.7690 / 0.7609 | | |
| mESC | sensitivity, fixed w=0.5 (NOT validation-chosen) | 0.6789 / 0.4265 / 0.7658 / 0.7580 | | |
| mESC | scRegNet design, Adam-tuned | 0.6433 / 0.4272 / 0.7506 / 0.7444 | | |
| mESC | hero (SAGE, no Geneformer) | 0.6313 / 0.3919 / 0.7269 / 0.7170 | | |

Weights chosen per fold on the held-out TFs (3 validation TFs per fold; the noise is large):

| Cell | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | Pooled over folds | Within-TF, per-fold choice | Within-TF, pooled choice | Within-TF, fixed w = 0.5 (not validation-chosen) |
|---|---|---|---|---|---|---|---|---|---|
| hESC | 0.75 | 1 | 1 | 1 | 1 | 1 | 0.7248 | 0.7244 | 0.7193 |
| hHEP | 1 | 0.75 | 0.5 | 1 | 0.75 | 0.75 | 0.7748 | 0.7760 | 0.7650 |
| mHSC-E | 0.5 | 0 | 0.75 | 1 | 0.5 | 0.5 | 0.7554 | 0.7551 | 0.7551 |
| mESC | 0.75 | 0.5 | 0 | 1 | 0 | 0.5 | 0.7555 | 0.7658 | 0.7658 |

The per-fold choice is the headline number. It picks w = 0 on two mESC folds and one mHSC-E fold, where the validation TFs happened to prefer the GNN, and that costs 0.010 on mESC against the pooled choice (p = 0.003). Popularity alone is better still on mESC (0.7690).

## Summary: pipeline against the scRegNet baseline on every cell and protocol

Within-TF AUROC. Standard split: W16, seeds 42-44, baseline = the published scRegNet recipe (SGD). Held-out: seeds 42, 43 x 5 folds, baseline = the scRegNet design, Adam-tuned. Verdict rule: beats or loses when p < 0.05, otherwise ties. The last column compares with the hero.

| Cell | Protocol | Pipeline | scRegNet baseline | Difference | p | Verdict | Against the hero (p) |
|---|---|---|---|---|---|---|---|
| hESC | standard split | 0.7460 | 0.7020 | +0.0440 | 7e-4 | beats | |
| hESC | held-out genes | 0.5783 | 0.5241 | +0.0620 | 0.002 | beats | +0.0161 (0.11) ties |
| hESC | held-out TFs | 0.7248 | 0.7163 | +0.0084 | 0.053 | ties | +0.0391 (1e-6) beats |
| hHEP | standard split | 0.8075 | 0.7873 | +0.0203 | 0.012 | beats | |
| hHEP | held-out genes | 0.6377 | 0.6477 | -0.0125 | 0.012 | **loses** | +0.0464 (0.001) beats |
| hHEP | held-out TFs | 0.7748 | 0.7405 | +0.0343 | 0.016 | beats | +0.0288 (1e-6) beats |
| mHSC-E | standard split | 0.7700 | 0.7210 | +0.0490 | 6e-6 | beats | |
| mHSC-E | held-out genes | 0.6881 | 0.6461 | +0.0434 | 0.023 | beats | +0.0118 (0.037) beats |
| mHSC-E | held-out TFs | 0.7554 | 0.7254 | +0.0315 | 0.001 | beats | +0.0354 (0.003) beats |
| mESC | standard split | 0.8403 | 0.7658 | +0.0745 | 4e-11 | beats | |
| mESC | held-out genes | 0.6724 | 0.6094 | +0.0682 | 2e-12 | beats | +0.0441 (2e-7) beats |
| mESC | held-out TFs | 0.7555 | 0.7506 | +0.0045 | 0.96 | ties | +0.0293 (1e-4) beats |

Counts: 9 beats, 2 ties, 1 loss. Against the hero on the 8 held-out combinations: 7 beats and 1 tie (hESC genes).

## What the numbers say

Measured.

- **Held-out TFs: the pipeline's advantage over the scRegNet design is popularity.** The within-TF AUROC of the popularity ranking alone is 0.7244, 0.7785, 0.7609, 0.7690 (hESC, hHEP, mHSC-E, mESC). The GNN alone scores 0.6860, 0.7246, 0.7336, 0.7364, below popularity on every cell, and the blend with a validation-chosen weight is never significantly above popularity (differences +0.0004, -0.0037, -0.0059, -0.0134; p = 1, 0.022, 0.30, 0.002). So the earlier hESC failure (-0.032 against the scRegNet design) was the GNN scoring below popularity, and the scRegNet design (which has no TF-specific term) is closest to popularity, as W16 said. On the other cells the same holds: the Adam-tuned design scores 0.7405, 0.7254, 0.7506, below popularity in every cell (popularity minus design +0.008, +0.038, +0.037, +0.018, p = 0.04, 3e-4, 2e-4, 1e-5).
- **Held-out genes: the DC-SBM blend helps where the DC-SBM is as good as the GNN.** Blend minus GNN alone: hESC +0.0189 (p = 0.004), mESC +0.0223 (3e-5), hHEP +0.0111 (p = 0.94, 17/30 TFs better), mHSC-E -0.0036 (p = 0.41). The DC-SBM alone is 0.5753 / 0.5981 / 0.6359 / 0.6597 (hESC, hHEP, mHSC-E, mESC), against GNN alone 0.5576 / 0.6314 / 0.6905 / 0.6559. On mHSC-E and hHEP the DC-SBM is weaker than the GNN and the blend adds nothing, in line with W16's standard-split finding (mHSC-E).
- **The single loss, hHEP held-out genes.** The Adam-tuned scRegNet design (GCN, additive decoder, three layers, Geneformer) is the best model on hHEP held-out genes: 0.6477 against 0.6314 for the pipeline's GNN and 0.6377 for the pipeline. Validation raised the blend from 0.6297 to 0.6580 on the held-out validation genes, but on the test genes the gain over the GNN is smaller (+0.0063 as the run mean, +0.0111 as the paired per-TF mean, which weights TFs that occur in few folds differently) and not significant (p = 0.94 across TFs). Guess, untested: the additive decoder with deeper propagation transfers better to unseen genes in hHEP than the Hadamard decoder with one layer.
- **Weight choice on 3 validation TFs is noisy** (per-fold weights run from 0 to 1 on mHSC-E and mESC) and cost 0.010 on mESC against the pooled choice, and nothing on hESC, hHEP, mHSC-E. Using a fixed w = 0.5 would have given 0.7193, 0.7650, 0.7551, 0.7658; this is a sensitivity row, not a validation-chosen result.

Guesses, untested.

- A better route for unseen TFs would be a model fitted on top of popularity (the GNN predicting the residual of the gene's popularity) or a blend weight fitted on many more validation TFs (leave-one-TF-out rather than 3 TFs per fold). Neither was run.
- The 2 ties on held-out TFs (hESC, mESC) are not ties of the GNN: they are the point at which popularity alone matches the scRegNet design.

## Dropped or uncertain

- The published SGD recipe has no held-out runs, so every held-out comparison is against its Adam-tuned version. The standard-split rows use the published recipe, so the summary table mixes two baselines.
- Two seeds per fold; the tests are paired over TFs (each TF appears once per protocol for TFs and in all 5 folds for genes, per-TF AUROC averaged over folds and seeds). No multiplicity correction over the 12 comparisons; the hHEP genes loss (p = 0.012) and the two ties near 0.05 could move.
- The per-fold blend weights are chosen on 3 validation TFs per fold (about 15 TFs in total), so single folds choose w = 0 or 1 on small differences.
- The pooled AUROC and AUPRC of a blend come from a global rank average; the within-TF numbers from a within-TF rank average. For popularity alone the pooled AUROC (0.6689 hESC, 0.7349 hHEP) is a ranking of genes across TFs, so it carries no TF base rate; do not compare it with the GNN's pooled AUROC as a model of absolute probability.
- Seeds were not ensembled in Part 2 (per-run metrics averaged). The mESC pipeline and hero runs are split over two machines.
