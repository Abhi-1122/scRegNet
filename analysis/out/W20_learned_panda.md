# W20: learned PANDA, trained on one cell type and applied to another

## Question

The task is now complete network inference. A cell type's whole TF -> gene network has to be predicted
from label-free inputs only: the JASPAR motif prior, STRING PPI, that cell's expression and Geneformer.
ChIP-seq labels serve only as the training loss on source cell types and for scoring the target.
Can a PANDA-like GNN trained on source cells beat the motif prior, original PANDA and a feature-only
MLP on a target cell it has never seen labels for?

## Method

- One graph per cell, built only from label-free files (`build_cell` in `src/xcell_gnn.py`).
  - Node features: 12 per-gene expression statistics (the `--feat stats` code of `pair_gnn.py`),
    the Geneformer vector (896 wide in all four cells), an is-TF flag and a has-motif flag. All are
    standardised within the dataset. No gene-identity embedding.
  - Relations: Mf (TF -> gene with `motif_hits > 0`, or the top-k genes per TF by `motif_max`,
    k in {50, 100, 200}), Mr (the reverse direction as its own relation), P (STRING edges among TFs).
    C (co-expression kNN) only as an ablation.
  - Pair features: `motif_max` and `log1p(motif_hits)`, each z-scored over the whole dataset, NaN set
    to 0, plus a missing flag. TF-gene Pearson correlation only as an ablation.
- Encoder: 2 NPanda layers (self term plus one mean-SAGE per relation), hidden 256, dropout 0.175.
  Decoder: an MLP (hidden 128) on [Hadamard of TF and gene embeddings, pair features].
- Feature-only MLP: the same network with no relations, so only the self terms remain.
- Loss: BCE on all labelled pairs (Train_set + Test_set) of the source cells. Adam, weight decay 1e-3.
- Selection: 20% of the source TFs (whole TFs) were held out as validation. The epoch and the
  config were chosen on source-validation within-TF AUROC. Full model: lr {1e-4, 3e-4, 1e-3} x motif
  graph {hits, top-50, top-100, top-200}, chosen on the mean over seeds 42-44. MLP: lr only.
  Nothing was chosen on the target. The model at the best validation epoch was used (no refit).
- Transfer settings: same species both ways (hESC <-> hHEP, mESC <-> mHSC-E) and leave one cell out
  (train on three, test on the fourth, across species).
- Upper bound: 5-fold TF-disjoint CV inside each cell with the same model. The config was chosen per
  fold on seed-42 validation, then seeds 43 and 44 were run for the chosen config.
- Baselines on the same target pairs: `motif_max`, |Pearson|, netZooPy PANDA (prior = binary
  `motif_hits > 0`, PPI = STRING TF-TF edges, the cell's expression). TFs without a motif enter PANDA
  through PPI; 33/34, 30/30, 86/88 and 29/29 labelled TFs have a PANDA row.
- Test: every labelled pair of the target. Within-TF AUROC is averaged over TFs with both classes.
  Model per-TF AUROC is averaged over seeds 42-44 before the paired per-TF Wilcoxon test. No
  multiplicity correction.

## Results

Within-TF AUROC, all labelled target TFs (p: paired per-TF Wilcoxon, full model against each):

| Setting | TFs | motif | |corr| | PANDA | MLP | full | p motif | p PANDA | p MLP |
|---|---|---|---|---|---|---|---|---|---|
| hESC -> hHEP | 30 | 0.552 | 0.504 | 0.543 | **0.579** | 0.562 | 0.37 | 0.58 | 0.013 (MLP better) |
| hHEP -> hESC | 34 | **0.602** | 0.464 | 0.535 | 0.529 | 0.512 | 0.002 (motif better) | 0.30 | 0.10 |
| mESC -> mHSC-E | 29 | 0.555 | 0.487 | 0.530 | 0.575 | **0.588** | 0.023 | 0.048 | 0.18 |
| mHSC-E -> mESC | 88 | 0.535 | 0.538 | 0.514 | **0.598** | 0.596 | 6e-5 | 4e-4 | 0.044 (MLP better) |
| others -> hESC | 34 | **0.602** | 0.464 | 0.535 | 0.581 | 0.582 | 0.32 | 0.20 | 0.58 |
| others -> hHEP | 30 | 0.552 | 0.504 | 0.543 | **0.606** | 0.596 | 0.002 | 2e-4 | 0.72 |
| others -> mESC | 88 | 0.535 | 0.538 | 0.514 | **0.595** | 0.592 | 1e-7 | 4e-5 | 0.32 |
| others -> mHSC-E | 29 | 0.555 | 0.487 | 0.530 | 0.621 | **0.630** | 0.001 | 7e-4 | 0.36 |

Pooled AUROC on the human targets is far below motif alone (hHEP: full 0.536 vs motif 0.659;
hESC: 0.531 vs 0.656 same species, 0.627 vs 0.656 leave-one-out). On the mouse targets all methods
are near 0.52-0.56.

Split by motif (within-TF; motif / PANDA / MLP / full; pm = p full vs motif):

| Setting | TFs with a motif | TFs without a motif | TFs not labelled in any source |
|---|---|---|---|
| hESC -> hHEP | n=22: 0.571 / 0.573 / 0.584 / 0.566, pm 0.73 | n=8: 0.500 / 0.460 / 0.567 / 0.551 | n=27: 0.551 / 0.539 / 0.580 / 0.561 |
| hHEP -> hESC | n=29: 0.619 / 0.540 / 0.538 / 0.524, pm 0.001 | n=5: 0.500 / 0.500 / 0.477 / 0.444 | n=31: 0.603 / 0.534 / 0.519 / 0.501 |
| mESC -> mHSC-E | n=21: 0.576 / 0.566 / 0.575 / 0.587, pm 0.12 | n=8: 0.500 / 0.436 / 0.573 / 0.589 | n=23: 0.563 / 0.534 / 0.557 / 0.574 |
| mHSC-E -> mESC | n=34: 0.591 / 0.569 / 0.592 / 0.588, pm 0.93 | n=54: 0.500 / 0.480 / 0.603 / 0.601 | n=82: 0.535 / 0.512 / 0.597 / 0.596 |
| others -> hESC | n=29: 0.619 / 0.540 / 0.585 / 0.584, pm 0.08 | n=5: 0.500 / 0.500 / 0.554 / 0.570 | n=20: 0.599 / 0.514 / 0.578 / 0.581 |
| others -> hHEP | n=22: 0.571 / 0.573 / 0.603 / 0.612, pm 0.016 | n=8: 0.500 / 0.460 / 0.615 / 0.551 | n=21: 0.549 / 0.543 / 0.606 / 0.589 |
| others -> mESC | n=34: 0.591 / 0.569 / 0.609 / 0.610, pm 0.15 | n=54: 0.500 / 0.480 / 0.586 / 0.581 | n=69: 0.534 / 0.514 / 0.588 / 0.589 |
| others -> mHSC-E | n=21: 0.576 / 0.566 / 0.616 / 0.625, pm 0.013 | n=8: 0.500 / 0.436 / 0.633 / 0.644 | n=21: 0.568 / 0.537 / 0.609 / 0.619 |

Gene-only diagnostic: each run's scores averaged over all TF rows, so every TF gets the same gene
ranking. On the hESC and hHEP targets the full model then loses 0.014-0.024 within-TF (p 0.0005 to
0.11). On the mHSC-E targets it loses 0.009-0.010 (p 0.27, 0.59). On the mESC targets the gene-only
ranking is slightly better (0.601 vs 0.596, 0.596 vs 0.592).

Upper bound, within-cell TF-disjoint CV (within-TF):

| Cell | motif | PANDA | CV popularity* | CV MLP | CV full | best transfer (full) | p CV full vs CV MLP | p CV full vs same-species |
|---|---|---|---|---|---|---|---|---|
| hESC | 0.602 | 0.535 | 0.729 | 0.699 | 0.694 | 0.582 (LOO) | 0.75 | 7e-7 |
| hHEP | 0.552 | 0.543 | 0.779 | 0.733 | 0.756 | 0.596 (LOO) | 0.022 | 6e-7 |
| mESC | 0.535 | 0.514 | 0.770 | 0.698 | 0.741 | 0.596 (same species) | 1e-9 | 2e-14 |
| mHSC-E | 0.555 | 0.530 | 0.765 | 0.713 | 0.741 | 0.630 (LOO) | 0.007 | 7e-6 |

*Not label-free: the fraction of the training-fold TFs of the same cell that target the gene.

Ablations of the full model at its chosen config, within-TF on the target (delta vs full, p):

| Setting | full | no P | no M (P only) | no motif pair feats | + C | + corr pair feat | ranking loss | no Geneformer |
|---|---|---|---|---|---|---|---|---|
| hESC -> hHEP | 0.562 | +0.009 (0.04) | -0.003 (0.97) | -0.007 (0.15) | +0.004 (0.28) | -0.001 (0.87) | +0.007 (0.19) | -0.026 (0.07) |
| hHEP -> hESC | 0.512 | +0.004 (0.16) | +0.006 (0.21) | -0.006 (0.08) | +0.032 (0.001) | +0.007 (0.17) | +0.029 (8e-4) | +0.042 (0.03) |
| mESC -> mHSC-E | 0.588 | +0.001 (0.98) | +0.009 (0.21) | -0.011 (0.10) | +0.021 (3e-4) | +0.002 (0.33) | +0.035 (2e-4) | -0.047 (0.002) |
| mHSC-E -> mESC | 0.596 | +0.006 (0.003) | +0.006 (0.002) | -0.001 (0.86) | -0.029 (1e-9) | +0.010 (1e-6) | +0.021 (2e-7) | +0.009 (0.99) |

Ablation numbers are measured on the target. They describe the model and were not used to choose it.

Leak audit:
- `build_cell()` runs inside `no_labels()`, which makes `pd.read_csv`, `np.load` and `open` raise on
  `Train_set.csv`, `Test_set.csv`, `Label.csv` and `BL--network.csv`. All graphs, node features and
  pair features are built there. `--selfcheck` asserts that the guard fires and that all four cells
  build under it. No gene popularity or other label statistic is computed for any cell.
- `load_labels()` is the only label reader. `run()` calls it for source cells (training loss; in CV the
  test-fold TFs are removed first, asserted). For the target it is called only inside `evaluate()`.
  Transfer runs assert that the target is not a source.
- Shared labelled TFs: hESC/hHEP 3 (EZH2, JUND, TEAD4); mESC/mHSC-E 6 (CBX7, HCFC1, HDAC2, JUN,
  MYC, NR4A1); leave-one-out 14 / 9 / 19 / 8 for hESC / hHEP / mESC / mHSC-E. Shared genes 130-445.
  With the shared TFs removed (last column of the motif table) every conclusion holds.

Compute: 824 runs, 371 GPU-process minutes, 4 processes in parallel on the laptop GPU.

## Why

- The model learned mostly a gene prior, not TF-specific regulation (measured). Averaging its scores
  over TFs costs at most 0.024 within-TF, and nothing on the mESC targets. Most of its gain over
  motif alone sits on TFs without a motif, where motif alone is fixed at 0.5 (mESC: 54 such TFs).
  Only two leave-one-out settings (hHEP, mHSC-E) also beat `motif_max` on TFs with a motif.
- Message passing over M and P adds nothing across cells (measured). The full model never beats the
  feature-only MLP and is below it in two settings. Removing M or P does not hurt and often helps
  slightly.
- Within a cell, message passing helps (CV full beats CV MLP on three cells), but in-cell gene
  popularity beats both (measured). The within-cell bound therefore mostly reflects which genes are
  frequent targets in that cell. That property does not carry over to another cell's genes and
  ChIP experiments. Guess, not tested: target frequency depends on the ChIP protocol and peak calling
  of each dataset more than on gene properties visible in expression or Geneformer.
- Motif TFs: the motif pair features alone do not let the model reach `motif_max` on hESC. The
  model trained on hHEP ranks hESC motif TFs at 0.524 against 0.619 for the raw motif score. Guess:
  the decoder learns a weight for motif features that fits the source cell's positive rate and
  peak profile, and the gene-prior part swamps the motif part on a new cell.
- Original PANDA with the motif prior is below the raw motif score in within-TF AUROC on all four
  cells (measured; pooled AUROC is 0.007 higher on hESC only). Expression smoothing does not help
  in these data.

## Caveats

- Validation sets are small (7 TFs for hESC as source, 6 for hHEP). The chosen config is noisy.
- The model at the best validation epoch was trained on 80% of the source TFs; no refit on all of them.
- In leave-one-out, batches are drawn in proportion to the number of labelled pairs, so mESC (88
  TFs) dominates the loss.
- The node features have 12 statistics, not 14: the existing `--feat stats` code produces 12.
- The first 117 runs computed layer 1's neighbour means inside each SAGE call; later runs cache
  them. The two are the same function (asserted in `--selfcheck` to 1e-4), so runs differ only by
  floating-point rounding.
- Ranking loss, the correlation pair feature and the C relation help on some targets. Adopting them
  now would be a choice made on the target, so they stay ablations. A fresh test would need its own
  source-validation selection.
- No multiplicity correction over 8 settings x 3 comparisons.

## Verdict

The learned PANDA does not achieve complete network inference across cell types. It beats the motif
prior in 5 of 8 settings, but only through a gene prior that the feature-only MLP learns equally
well. On TFs with a motif it beats `motif_max` in 2 of 8 settings and loses badly on hHEP -> hESC.
Message passing over the motif and PPI graphs adds nothing in transfer. Pooled AUROC on the human
targets stays below the raw motif score. The gap to the within-cell bound is 0.11-0.16, and most of
that bound is in-cell gene popularity.

Report: `python3.12 analysis/w20_xcell_report.py` -> `analysis/out/w20_xcell.txt`. Runs:
`analysis/out/xcell/w20_*.{json,npz}`, baselines `analysis/out/xcell/base_<cell>.npz`
(`python3.12 -m src.xcell_gnn --baselines <cell>`). Jobs `analysis/jobs/w20_p1.txt`,
`analysis/jobs/w20_p2.txt` (`python3.12 analysis/w20_xcell_report.py --jobs 1|2`). Self-check
`python3.12 -m src.xcell_gnn --selfcheck`.
