# W21: two-view model, motif + PPI graph and co-expression graph

Task: complete network inference. A model trained on source cell types predicts the whole TF -> gene
network of a different target cell from label-free inputs. ChIP-seq labels serve only as the source
training loss and to score the target. Data, features, source-validation split and evaluation are
those of W20 (`src/xcell_gnn.py`, imported, not copied).

## Question

W20's single-graph model scores 0.70-0.79 on source-validation TFs but only near motif alone on the
target. Does a second, independent view of each gene, the target cell's own co-expression graph,
narrow that gap and beat the single-graph model, motif alone and a feature-only MLP?

## Method

- **View A**: one graph per cell with motif edges TF -> gene (and reverse) and STRING TF-TF edges.
  Motif graph = `motif_hits > 0` or the top 50 genes per TF by `motif_max`, chosen on source
  validation. 2-layer relational encoder (self term + one weighted-mean aggregation per relation),
  hidden 256.
- **View B**: mutual kNN on the cell's Pearson correlation, k in {5, 10, 20}, positive r only or
  |r| (signed weights normalised by the sum of |w|), chosen on source validation. Edge weight = r.
  2-layer weighted SAGE with its own weights. Co-expression never scores a pair directly.
- **Node features**: as W20 (12 expression statistics, Geneformer 896, is-TF, has-motif).
- **Fusion**: concat (z = ReLU(W [zA; zB])), per-gene gate (z = g zA + (1-g) zB), early (co-expression
  as a fourth relation inside encoder A). Single-view arms: view A only, view B only, MLP (self term
  only, no graph).
- **Decoder**: MLP on [z_t * z_g ; motif_max, log1p motif_hits, missing flag].
- **Loss**: TF-balanced BCE. Every source TF weighs the same, and its positives and negatives weigh
  the same. View dropout 0.2 (one view zeroed for a whole batch). Adam, weight decay 1e-3, dropout
  0.175, patience 10, at most 150 epochs.
- **Selection**: 20% of source TFs held out (W20's seeded split). lr in {1e-4, 3e-4, 1e-3}, motif
  graph, co-expression graph and epoch chosen on source-validation within-TF AUROC (mean over seeds
  42-44). The best fusion for the ablations (gate, 0.7430 vs concat 0.7389) was also chosen on source
  validation. Nothing was chosen on the target.
- **Ablations on the gate model at its chosen config**: plain BCE; no view dropout; InfoNCE
  consistency between zA and zB of the same gene, lambda 0.1 or 0.5, on the source graphs only or
  also on the target graph (target expression, motif and PPI only, no target labels).
- **Settings**: hESC <-> hHEP, mESC <-> mHSC-E, and leave one cell out (train on three, test on the
  fourth). Seeds 42-44. 3384 runs in total on one 2080 Ti on ada.
- **Statistics**: paired per-TF Wilcoxon on seed-averaged per-TF AUROC, two-sided. No multiplicity
  correction.

## Results

Within-TF AUROC over all labelled target TFs. W20 = the W20 full model chosen by its own report.

| Setting | nTF | motif | MLP | view A | view B | concat | gate | early | W20 |
|---|---|---|---|---|---|---|---|---|---|
| hESC -> hHEP | 30 | 0.552 | 0.554 | **0.578** | 0.560 | 0.540 | 0.550 | 0.558 | 0.562 |
| hHEP -> hESC | 34 | **0.602** | 0.527 | 0.523 | 0.533 | 0.546 | 0.530 | 0.528 | 0.512 |
| mESC -> mHSC-E | 29 | 0.555 | 0.614 | 0.616 | 0.604 | **0.622** | 0.614 | 0.614 | 0.588 |
| mHSC-E -> mESC | 88 | 0.535 | 0.598 | 0.597 | **0.603** | 0.598 | 0.595 | 0.583 | 0.596 |
| others -> hESC | 34 | 0.602 | 0.608 | 0.585 | 0.595 | **0.612** | 0.608 | 0.600 | 0.582 |
| others -> hHEP | 30 | 0.552 | **0.623** | 0.605 | 0.616 | 0.607 | 0.603 | 0.610 | 0.596 |
| others -> mESC | 88 | 0.535 | 0.606 | 0.580 | **0.627** | 0.608 | 0.608 | 0.577 | 0.592 |
| others -> mHSC-E | 29 | 0.555 | **0.661** | 0.657 | 0.644 | 0.641 | 0.649 | 0.631 | 0.630 |
| mean of 8 | | 0.561 | **0.599** | 0.593 | 0.598 | 0.597 | 0.595 | 0.588 | 0.582 |

Paired per-TF Wilcoxon p, fusion against motif / view A / MLP (sign = fusion minus the other):

| Setting | concat | gate | early |
|---|---|---|---|
| hESC -> hHEP | −0.30 / −7e-6 / −0.21 | −0.97 / −5e-4 / −0.45 | +0.95 / −0.007 / +0.82 |
| hHEP -> hESC | −0.033 / +0.003 / +0.003 | −0.005 / +0.65 / +0.52 | −0.005 / +0.79 / +0.99 |
| mESC -> mHSC-E | +0.004 / +0.69 / +0.28 | +0.003 / −0.38 / −0.65 | +0.009 / −0.81 / +0.61 |
| mHSC-E -> mESC | +2e-4 / +0.13 / −0.75 | +7e-5 / −0.31 / −0.029 | +9e-4 / −1e-4 / −8e-5 |
| others -> hESC | +0.87 / +0.036 / +0.69 | +0.89 / +0.19 / +0.33 | −0.89 / +0.70 / −0.11 |
| others -> hHEP | +0.004 / +0.86 / −0.004 | +0.003 / −0.79 / −0.002 | +7e-4 / +0.38 / −0.003 |
| others -> mESC | +2e-8 / +0.002 / +0.35 | +1e-8 / +9e-6 / +0.10 | +2e-5 / −0.17 / −4e-5 |
| others -> mHSC-E | +6e-4 / −0.013 / −0.016 | +7e-4 / −0.056 / −0.069 | +0.002 / −0.003 / −0.006 |

- Against view A only: concat is significantly better in 3 settings and worse in 2 (mean +0.004).
  Gate is better in 1 and worse in 1 (mean +0.002). Early fusion is worse in 3 and never
  significantly better (mean −0.005).
- Against motif alone: concat and gate are significantly better in 5 of 8 settings and
  significantly worse on hHEP -> hESC. Both tie on hESC -> hHEP and others -> hESC.
- Against the MLP: no fusion is significantly better anywhere except concat on hHEP -> hESC. The MLP
  has the highest mean (0.599).
- Pooled AUROC on the human targets stays below motif alone for every arm (hHEP 0.52-0.58 vs 0.66,
  hESC 0.52-0.65 vs 0.66). Full pooled AUROC / AUPRC are in `analysis/out/w21_2view.txt`.

Breakdown, within-TF (motif / MLP / view A / gate; p gate vs view A):

| Setting | TFs with motif | TFs without motif | TFs not labelled in any source |
|---|---|---|---|
| hESC -> hHEP | n=22 .571/.555/.582/.564 p=0.008 | n=8 .500/.549/.567/.512 p=0.055 | n=27 .551/.548/.576/.548 p=5e-4 |
| hHEP -> hESC | n=29 .619/.534/.527/.528 p=0.86 | n=5 .500/.488/.500/.542 p=0.62 | n=31 .603/.514/.510/.518 p=0.61 |
| mESC -> mHSC-E | n=21 .576/.618/.618/.612 p=0.45 | n=8 .500/.603/.612/.616 p=0.74 | n=23 .563/.607/.605/.601 p=0.34 |
| mHSC-E -> mESC | n=34 .591/.585/.597/.594 p=0.85 | n=54 .500/.607/.597/.595 p=0.28 | n=82 .535/.598/.598/.596 p=0.31 |
| others -> hESC | n=29 .619/.608/.584/.603 p=0.29 | n=5 .500/.607/.591/.638 p=0.44 | n=20 .599/.613/.575/.615 p=0.11 |
| others -> hHEP | n=22 .571/.620/.607/.608 p=0.97 | n=8 .500/.631/.601/.589 p=0.55 | n=21 .549/.629/.606/.605 p=0.89 |
| others -> mESC | n=34 .591/.626/.608/.631 p=6e-4 | n=54 .500/.593/.562/.593 p=0.002 | n=69 .534/.599/.572/.601 p=1e-4 |
| others -> mHSC-E | n=21 .576/.659/.659/.654 p=0.24 | n=8 .500/.666/.652/.634 p=0.078 | n=21 .568/.652/.642/.637 p=0.20 |

Source validation vs target (mean of 8 settings, within-TF):

| Arm | source-val | target | gap |
|---|---|---|---|
| MLP | 0.717 | 0.599 | 0.118 |
| view A | 0.742 | 0.593 | 0.149 |
| view B | 0.721 | 0.598 | 0.124 |
| concat | 0.739 | 0.597 | 0.142 |
| gate | 0.743 | 0.595 | 0.148 |
| early | 0.746 | 0.588 | 0.159 |
| W20 full | 0.749 | 0.582 | 0.167 |

Ablations of the gate model (change in within-TF against gate, mean of 8 settings; significant
settings by Wilcoxon):

| Ablation | mean change | settings p < 0.05 | source-val change |
|---|---|---|---|
| plain BCE instead of TF-balanced | −0.005 | 1 better (+0.002), 1 worse (−0.016) | +0.005 |
| no view dropout | −0.001 | 2 better, 1 worse | −0.005 |
| consistency 0.1, source only | −0.006 | none | −0.034 |
| consistency 0.5, source only | −0.013 | 1 better, 3 worse | −0.061 |
| consistency 0.1, source + target | −0.023 | 3 worse | −0.041 |
| consistency 0.5, source + target | −0.032 | 5 worse | −0.078 |

The learned gate puts 0.51-0.88 of each gene's weight on view A (mean over target genes).

**Loss and the W20 comparison.** The MLP, view A and view B arms here use the TF-balanced loss. W20
used plain BCE, a wider motif-graph choice (top 50/100/200 or hits) and no view dropout. View A only
(0.593) is 0.010 above W20's full model (0.582) on the mean of 8. The plain-BCE ablation, run on the
gate model, changes within-TF by −0.005 on average (range −0.016 to +0.009). So the loss accounts
for about half of the difference between view A and W20, and the rest is within run-to-run spread.
The two-view effect (concat or gate against view A, +0.002 to +0.004) is smaller than the loss
effect.

**Gene-prior check.** When each chosen run's grid is averaged over TFs, so that every TF gets one
gene ranking, within-TF drops by at most 0.043. On the mESC targets the averaged grid scores the same or
higher (for example MLP 0.598 -> 0.622 on mHSC-E -> mESC). The two-view models, like W20's, are
mostly gene priors.

## Why

- Measured: co-expression does not close the gap between source and target. The gap is 0.142-0.159 for
  the fusions, 0.149 for view A and 0.118 for the MLP. The arms with the best source validation
  (early, gate) do not transfer better.
- Measured: the gate leans on view A (weight 0.67-0.88 in 7 of 8 settings), and view B alone is as
  good as view A on the target (0.598 vs 0.593). The two views are close substitutes, not
  complements, for the gene prior the decoder learns.
- Measured: the label-free consistency loss lowers both source validation and target. It pulls zA
  and zB together, which removes view-specific signal. Computing it on the target graph hurts more
  (−0.023 and −0.032), so aligning the target's views does not adapt the gene prior to the target.
- Measured: the TF-balanced loss helps a little (+0.005 against plain BCE on average) and view
  dropout changes almost nothing.
- Measured: removing TFs labelled in a source cell does not change any conclusion. TFs without a
  motif gain most over motif alone. On TFs with a motif the gate model beats motif_max in the mouse
  settings and in others -> hHEP, and loses on both hESC targets (hHEP -> hESC 0.528 vs 0.619).
- Guess, not tested: the co-expression neighbourhood mostly encodes expression level and
  cell-cycle or housekeeping programs, which the 12 expression statistics already carry. That would
  explain why view B alone matches view A and adds little on top.

## Caveats

- Source validation is small (6-7 TFs for a single human source), and selecting over 36
  configurations per fusion is noisier than over 6 for view A. That favours view A in the comparison
  only slightly; the fusions' source-validation scores are not lower.
- No refit on all source TFs. In leave-one-out, cells are not weighted explicitly. Each batch comes
  from one cell, so each cell's share follows its number of pairs, which is roughly equal per TF.
- The plain-BCE ablation was run on the gate model only, so the loss effect on view A and the MLP
  is inferred, not measured.
- No multiplicity correction across 8 settings × several comparisons. Single p-values near 0.01-0.05
  could move.
- W22 (section 38) reports scRegNet on the motif graph at a mean of 0.605 over the same 8 settings,
  above every W21 arm. That run is not part of this experiment and uses a different decoder.

## Verdict

Fail. The two-view model does not beat view A only in a consistent way (concat +0.004, gate +0.002
mean within-TF; significant wins and losses roughly balance). It never beats the feature-only MLP
overall, and it narrows the gap between source and target by nothing measurable. It beats motif alone in 5 of 8
settings, as every learned arm here does, through the gene prior. The label-free consistency
loss, including adaptation on the target graph, hurts.

## Leak audit

Source labels are loaded first. All graphs, features, training (including the target consistency
term) and the target score grid then run inside `xcell_gnn.no_labels()`, which raises on any read of
`Train_set.csv`, `Test_set.csv`, `Label.csv` or `BL--network.csv`. `evaluate()` then reads the target
labels. `python -m src.xcell_2view --selfcheck` passed on ada. It asserts that the guard fires, all graphs
and features of every cell build under the guard, mutual kNN is symmetric with at most k neighbours,
a perfect grid scores within-TF and pooled AUROC 1, TF weights sum to 1 per TF, and a 2-epoch
source+target consistency run completes under the guard.

## Files

- Code: `src/xcell_2view.py` (`--selfcheck`), `analysis/ada_w21.sh` (ada wrapper, one GPU).
- Report: `python3.12 analysis/w21_report.py` -> `analysis/out/w21_2view.txt`.
- Job lists `analysis/jobs/w21_p1.txt` (MLP, view A, view B), `w21_p2.txt` (fusions),
  `w21_p3.txt` (ablations, written on ada from the finished runs).
- Runs `analysis/out/xcell2/` (json + score grid per run), logs `analysis/out/ada/w21_*.log`.
- Compute: 3384 runs, 8.8 GPU-hours wall time on one 2080 Ti (2698 process-minutes, 6 processes in
  parallel).
