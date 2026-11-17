# W22: fair baselines for complete network inference

## Question

The task is complete network inference: predict a cell type's whole TF -> gene network from
label-free inputs (motif prior, STRING PPI, the cell's expression, Geneformer). ChIP-seq labels are
only a training loss on source cells and the score on the target. The published scRegNet used the
same cell's ChIP training edges as its graph, so it is not a fair comparison here. How strong are
standard methods that obey the same rule: PANDA, GENIE3-style regression with motif pruning (the
SCENIC recipe), and scRegNet with its ChIP graph replaced by the motif graph?

## Method

All code is in `src/xcell_base.py`. Data loading, node features, the motif graph, the evaluation and
the label guard are imported from `src/xcell_gnn.py` (W20), so every baseline is scored on the same
target pairs and TF sets as W20.

- **PANDA** (unsupervised; motif + PPI + expression). netZooPy is not installed on ada and nothing
  may be installed, so the netZooPy CPU loop was ported to numpy (about 40 lines): `normalize_network`
  (column and row z-scores, the same NaN rules), continuous Tanimoto updates of the network, the PPI
  and the co-expression matrices, the diagonal update, alpha 0.1, stop at hamming < 0.001. Prior W0 =
  binary `motif_hits > 0`; P0 = STRING TF-TF edges with unit diagonal (identity for "no PPI");
  C0 = Pearson correlation of the cell's expression. Rows = every TF in `TF.csv`.
- **GENIE3-style regression** (unsupervised; expression only, then motif). arboreto is not installed,
  so scikit-learn was used. For each gene, the expression of all `TF.csv` TFs (the gene itself
  excluded) predicts its expression.
  - GENIE3: ExtraTrees, 500 trees, sqrt features, output scaled to unit variance, unnormalised
    importances averaged over trees.
  - GRNBoost2-style: gradient boosting with arboreto's settings (learning rate 0.01, at most 5000
    trees, max_features 0.1, subsample 0.9, stop when the mean OOB improvement over 25 rounds is
    below 0), importance times the number of trees, raw expression.
  - Motif pruning: hard = pairs with `motif_hits > 0` ranked above all others, importance rank inside
    each group; soft = mean of the within-TF ranks of importance and `motif_max` (no motif = lowest).
  - All settings were fixed before any run. Nothing was tuned.
- **scRegNet on a motif graph** (supervised transfer). The published model: GCN 227/29/34 with ReLU
  and dropout between layers, Geneformer joined after the GCN, Linear 198 + LeakyReLU, additive
  decoder Linear([u_TF, u_gene]) (as `src/pair_gnn.py --encoder gcn --decoder add --scfm 1`).
  Changes, and only these: (i) graph = `motif_hits > 0`, made undirected; (ii) GCN input = the 12
  expression statistics W20 uses (the raw expression has one column per cell and differs between
  datasets), Geneformer = W20's standardised vectors; (iii) trained on all labelled pairs of the
  source cell(s), 20% of source TFs held out for validation with W20's split, tested on the target.
  Two recipes: published SGD (lr 1.44e-3, wd 1.04e-5, dropout 0.175, batch 193, evaluation every 5
  epochs, at most 600) and Adam-tuned (lr 4.29e-5, wd 2e-3, dropout 0.481, batch 512, every epoch, at
  most 300). Patience 10 evaluations. The epoch and the recipe were chosen on source-validation
  within-TF AUROC (mean over seeds 42-44). Model at the best epoch, no refit.
- Settings: hESC <-> hHEP, mESC <-> mHSC-E, and leave one cell out. Unsupervised baselines run on
  each cell. Metrics on every labelled pair of the target: within-TF AUROC (primary), pooled AUROC,
  AUPRC. Seed runs: per-TF AUROC averaged over seeds. Paired per-TF Wilcoxon, two-sided, no
  multiplicity correction.
- PANDA port check: on hESC, with the 332 TFs netZooPy keeps (motif or PPI TFs), the port gives the
  netZooPy network of W20 (`analysis/out/xcell/base_hESC.npz`): Pearson r = 1.000000, max |difference|
  2.4e-7 (float32 storage), 49 steps (`python3.12 -m src.xcell_base --panda_check hESC`).

## Results

### Unsupervised baselines on each cell

Within-TF AUROC over all labelled TFs (paired per-TF Wilcoxon p against motif alone); pooled AUROC / AUPRC.

| Method | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| motif alone (`motif_max`) | **0.602** ; 0.656 / 0.221 | **0.552** ; 0.659 / 0.482 | **0.535** ; 0.540 / 0.342 | **0.555** ; 0.523 / 0.588 |
| PANDA (motif + PPI + expression) | 0.541 (7e-6) ; 0.649 / 0.219 | 0.543 (0.73) ; 0.620 / 0.451 | 0.515 (0.024) ; 0.520 / 0.331 | 0.541 (0.93) ; 0.531 / 0.587 |
| PANDA without PPI | 0.544 (7e-5) | 0.537 (0.39) | 0.510 (0.014) | 0.539 (0.86) |
| GENIE3 (ExtraTrees), expression only | 0.464 (3e-6) ; 0.520 / 0.151 | 0.517 (0.055) ; 0.532 / 0.381 | 0.498 (5e-4) ; 0.502 / 0.317 | 0.510 (0.041) ; 0.530 / 0.567 |
| GENIE3 + hard motif pruning | 0.517 (1e-5) | 0.542 (0.52) | 0.512 (0.001) | 0.552 (0.28) |
| GENIE3 + soft motif (rank average) | 0.539 (4e-4) | 0.546 (0.79) | 0.521 (0.008) | 0.551 (0.27) |
| GRNBoost2-style (GBM), expression only | 0.494 (6e-6) | 0.509 (0.003) | 0.505 (0.042) | 0.497 (0.004) |
| GBM + hard motif pruning | 0.540 (3e-5) | 0.540 (0.033) | 0.521 (0.13) | 0.537 (0.08) |
| GBM + soft motif (rank average) | 0.574 (0.002) | 0.550 (0.26) | 0.533 (0.70) | 0.539 (0.09) |

TFs with a motif / without a motif (motif alone is 0.5 by construction without a motif):

| Method | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| motif alone | 0.619 / 0.500 | 0.571 / 0.500 | 0.591 / 0.500 | 0.576 / 0.500 |
| PANDA | 0.547 / 0.504 | 0.574 / 0.460 | 0.576 / 0.476 | 0.580 / 0.437 |
| PANDA without PPI | 0.556 / 0.477 | 0.566 / 0.457 | 0.571 / 0.472 | 0.578 / 0.437 |
| GENIE3 / + hard / + soft | 0.466 / 0.528 / 0.554 ; 0.452 | 0.527 / 0.561 / 0.567 ; 0.489 | 0.513 / 0.549 / 0.571 ; 0.489 | 0.485 / 0.543 / 0.542 ; 0.574 |

(For GENIE3 the TFs without a motif have the same score in all three variants: 0.452, 0.489, 0.489, 0.574.)

### scRegNet on the motif graph (supervised transfer)

Within-TF AUROC over all labelled target TFs, seed mean (p against motif alone). The chosen recipe
is the one with the higher source-validation within-TF.

| Setting | motif | PANDA | GENIE3 soft | scRegNet SGD | scRegNet Adam | chosen | W20 full | W20 MLP |
|---|---|---|---|---|---|---|---|---|
| hESC -> hHEP | 0.552 | 0.543 | 0.546 | 0.600 (0.005) | 0.598 (0.005) | Adam **0.598** (0.005) | 0.562 (0.37) | 0.579 (0.020) |
| hHEP -> hESC | **0.602** | 0.541 | 0.539 | 0.524 (0.004) | 0.526 (0.007) | Adam 0.526 (0.007) | 0.512 (0.002) | 0.529 (0.002) |
| mESC -> mHSC-E | 0.555 | 0.541 | 0.551 | 0.627 (0.014) | 0.634 (0.005) | Adam **0.634** (0.005) | 0.588 (0.023) | 0.575 (0.11) |
| mHSC-E -> mESC | 0.535 | 0.515 | 0.521 | 0.605 (1e-5) | 0.598 (8e-5) | Adam **0.598** (8e-5) | 0.596 (6e-5) | 0.598 (3e-5) |
| others -> hESC | **0.602** | 0.541 | 0.539 | 0.599 (0.46) | 0.592 (0.43) | SGD 0.599 (0.46) | 0.582 (0.32) | 0.581 (0.23) |
| others -> hHEP | 0.552 | 0.543 | 0.546 | 0.615 (7e-4) | 0.620 (3e-4) | SGD **0.615** (7e-4) | 0.596 (0.002) | 0.606 (0.002) |
| others -> mESC | 0.535 | 0.515 | 0.521 | 0.629 (5e-10) | 0.629 (1e-9) | SGD **0.629** (5e-10) | 0.592 (1e-7) | 0.595 (8e-7) |
| others -> mHSC-E | 0.555 | 0.541 | 0.551 | 0.644 (0.003) | 0.654 (0.001) | SGD **0.644** (0.003) | 0.630 (0.001) | 0.621 (0.012) |

- Recipe choice: the two recipes differ by at most 0.003 in source validation (Adam chosen on all
  four same-species settings, SGD on all four leave-one-out settings). On the target they differ by
  at most 0.010.
- Chosen scRegNet-motif minus W20 full: +0.036 (p 6e-5), +0.014 (0.42), +0.047 (0.006), +0.003
  (0.19), +0.017 (0.58), +0.019 (0.31), +0.037 (1e-4), +0.013 (0.51) in the order of the table. It is
  never worse and is significantly better in 3 of 8 settings. Against the W20 MLP it is higher in 6
  of 8, significantly in 3 (hESC -> hHEP, mESC -> mHSC-E, others -> mESC); the MLP is ahead by less
  than 0.004 in the other two.
- Pooled AUROC is weak: 0.50-0.62 for scRegNet-motif. It is below motif alone on both human targets
  (hHEP 0.50 / 0.51 vs 0.66; hESC 0.56 / 0.62 vs 0.66) and above it on the mouse targets.
- Breakdown (chosen recipe vs motif alone): on TFs with a motif it is better in 3 of 8 settings
  (others -> hHEP, mESC, mHSC-E; p 0.011-0.026), not significantly different in 4 (mESC -> mHSC-E
  +0.054 at p 0.065) and worse in hHEP -> hESC (0.545 vs 0.619, p 0.024). On TFs without a motif it reaches 0.59-0.65
  except on hESC (0.41 same species, 0.58 leave-one-out). With the TFs labelled in any source cell
  removed, the picture is unchanged (for example others -> mESC 0.625 vs 0.534, p 1e-7; hHEP -> hESC
  0.527 vs 0.603, p 0.013).

## Why

- Measured: scRegNet's additive decoder scores a pair as a TF term plus a gene term, so every TF gets
  the same gene ranking. Across all 48 runs, the within-TF AUROC equals that of the grid averaged over
  TF rows (max difference 1.2e-4, ties). scRegNet-motif is therefore a learned gene prior: "which genes
  are frequent ChIP targets", predicted from expression statistics, Geneformer and the motif graph.
  It wins on TFs without a motif for that reason, and it cannot use the TF-specific motif signal.
- Measured: that gene prior transfers better than W20's learned PANDA (never worse, 3 of 8
  significant). W20 already showed that its own gains came from a gene prior that a feature-only MLP
  matched; a simpler additive model learns the same thing with less overfitting.
- Measured: hESC is the exception. Its motif prior is the strongest of the four cells (0.619 on TFs
  with a motif), and no transferred gene prior beats it there.
- Measured: expression-only regression is at or below chance (GENIE3 0.46-0.52, GBM 0.49-0.51).
  Motif pruning recovers part of the gap but never beats motif alone on TFs with a motif; the soft
  rank average is the best variant and still sits at or below motif alone. Co-expression within a
  few hundred cells does not identify ChIP targets here.
- Measured: PANDA lowers the motif signal on hESC and mESC and leaves it unchanged on hHEP and mHSC-E.
  The PPI matters little (with and without PPI within 0.007). TFs without a motif, which get edges
  only through PPI, score 0.44-0.50, at or below chance.
- Guess, not tested: as in W20, target frequency reflects each dataset's ChIP protocol and peak
  calling; the mouse sources share that more with each other than human cells do with hESC.

## Caveats

- No refit and selection on within-TF instead of pooled AUROC in scRegNet-motif (W20's protocol,
  not the published one). Its input is 12 expression statistics, not the expression profile.
- The validation gap between recipes is at noise level, so the recipe choice is close to arbitrary;
  both are reported.
- GENIE3 uses ExtraTrees with 500 trees (GENIE3 allows ExtraTrees; the default is a random forest
  with 1000). The GBM follows arboreto's settings with scikit-learn, not arboreto itself. Motif
  pruning uses the project's per-pair JASPAR promoter scans, not SCENIC's cisTarget enrichment over
  gene modules, so it is a SCENIC-style recipe and not SCENIC.
- PANDA uses a binary motif prior and all TF.csv rows. netZooPy drops TFs absent from both priors;
  the netZooPy network (W20) gives 0.535 / 0.543 / 0.514 / 0.530, within 0.011 of the port.
- No multiplicity correction. W21 is not in the table (its final report did not exist).

## Verdict

Of the three fair baselines, only scRegNet on the motif graph beats motif alone, in 6 of 8 transfer
settings (+0.05 to +0.09 within-TF), tied in others -> hESC and worse in hHEP -> hESC. It does so as
a pure gene prior, and it is at least as good as W20's learned PANDA everywhere, so it is the
baseline to beat. PANDA and GENIE3-style regression, with or without motif pruning, do not beat the
motif prior on any cell; expression-only regression is at chance.

Report: `python3.12 analysis/w22_report.py` -> `analysis/out/w22_baselines.txt`. Code
`src/xcell_base.py` (`--selfcheck`, `--panda_check`), runs `analysis/out/xcell_base/`, jobs
`analysis/jobs/w22_gpu.txt`, `analysis/jobs/w22_cpu.txt`, ada wrapper `analysis/ada_w22.sh`.
Compute: 1.45 GPU-hours on one 2080 Ti (job 86487 84 min, smoke 1 min, a cancelled first attempt
2 min), 29 allocated CPU-hours (20 CPUs).
