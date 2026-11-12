# W14: gene-level properties or co-expression across cells?

Hero settings (1-layer SAGE, `--split batch`, `--decoder had`, no Geneformer, epoch chosen on validation AUROC, refit), seeds 42-44 in random-split tables, 5 folds x seeds 42, 43 for held-out genes. Paired tests: per-TF test AUROC averaged over seeds, Wilcoxon over TFs. Baselines are the existing runs (`w3_hero`, `w4_<cell>_hero`, `w5_ho_gene<f>_hero`). One hero rerun with the new code reproduced `w3_hero_s42` to GPU rounding (AUROC 0.8889 against 0.8890, within-TF 0.7505 against 0.7499). Reproduce: `python3.12 analysis/w14_report.py` (writes `analysis/out/w14_feat.txt`). Runs: `analysis/jobs/w14_feat.txt`, `w14_feat_ho_rand.txt`; all on the laptop (ada was unreachable), so each variant's seeds share one machine.

## 1. Question

Every graph idea failed, and section 32 showed the GNN does not get its signal mainly from the co-variation of a gene with a TF's activity. Two readings remain. The model may read gene-level properties of the expression profile (mean level, detection rate, spread), which predict whether a gene is a target at all and so act as a refined popularity. Or it may read the co-expression pattern across cells.

## 2. Method in short

New flag `--feat {full, cellperm, stats, rand}` in `src/pair_gnn.py`, applied in `prepare()` to the node features (default `full` is the old behaviour). The standard pipeline z-scores every gene across cells, so `full` already carries no mean or spread; only the shape of each profile and the cell-by-cell pattern remain.
- `cellperm`: each gene's values are permuted across cells with a fixed RandomState per gene (same matrix for all seeds). Every gene keeps its marginal distribution; all co-expression is destroyed.
- `stats`: the cells dimension is replaced by 12 per-gene statistics of the raw expression: mean, std, coefficient of variation, detection rate, mean over nonzero cells, skewness, percentiles 10/25/50/75/90/99, each standardised across genes.
- `rand`: i.i.d. Gaussian features of the width of `full` (fixed seed). Only the training-graph structure and a memorised gene identity are available.
No leak is possible from these changes: the features are label-free and the message-passing graph is untouched.

## 3. Results

Random split, within-TF AUROC (mean ± std over 3 seeds); p is the paired per-TF Wilcoxon against full. Kept share = (variant - rand) / (full - rand), on within-TF.

| Cell | full | cellperm | stats | rand | full - rand (p) | kept: cellperm / stats |
|---|---|---|---|---|---|---|
| hESC | 0.749 ± 0.002 | 0.714 ± 0.006 (p = 0.017) | 0.629 ± 0.005 (p = 2e-7) | 0.721 ± 0.007 (p = 0.04) | +0.028 (p = 0.04) | -0.26 / -3.3 |
| hHEP | 0.759 ± 0.002 | 0.773 ± 0.002 (p = 0.98) | 0.738 ± 0.004 (p = 0.003) | 0.768 ± 0.002 (p = 0.97) | -0.009 (p = 0.97) | -0.58 / 3.3 |
| mESC | 0.824 ± 0.001 | 0.806 ± 0.002 (p = 4e-4) | 0.734 ± 0.002 (p = 1e-12) | 0.805 ± 0.003 (p = 3e-4) | +0.019 (p = 3e-4) | 0.05 / -3.7 |
| mHSC-E | 0.758 ± 0.006 | 0.754 ± 0.006 (p = 0.95) | 0.695 ± 0.002 (p = 1e-4) | 0.760 ± 0.011 (p = 0.85) | -0.003 (p = 0.85) | 2.3 / 23 |

The kept share is only meaningful where full beats rand (hESC, mESC). In hHEP and mHSC-E the denominator is zero or negative, so the ratio has no meaning there and is reported only for completeness; the direct reading is that rand matches full. AUROC and AUPRC follow the same order (full 0.889 / 0.625 on hESC against 0.879 / 0.614 for rand, 0.877 / 0.605 for cellperm, 0.862 / 0.575 for stats); the full tables with the paired tests are in `w14_feat.txt`.

Held-out genes (hESC, 10 runs per row, within-TF; training in-degree is 0 for every test gene, so popularity gives 0.5):

| full | cellperm | stats | rand |
|---|---|---|---|
| 0.546 ± 0.031 | 0.484 ± 0.034 (p = 6e-4) | 0.560 ± 0.025 (p = 0.055) | 0.491 ± 0.017 (p = 0.016) |

Kept share of the gain over rand: cellperm -0.13, stats 1.25.

Single raw statistic as a score (within-TF AUROC of ranking each TF's test genes by the statistic alone, signed):

| Statistic | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| detection rate | 0.546 | 0.610 | 0.607 | 0.586 |
| mean | 0.519 | 0.588 | 0.537 | 0.571 |
| best of the 12 | detection rate | detection rate | detection rate | p25 (0.587) |
| popularity (training in-degree) | 0.705 | 0.784 | 0.760 | 0.714 |

Spearman of a statistic with popularity is small: detection rate 0.14 (hESC), 0.26, 0.40 (mESC), 0.18; coefficient of variation and skewness are negative (down to -0.33).

Spearman of test scores with popularity (within TF, mean over TFs):

| | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| full | 0.721 | 0.830 | 0.717 | 0.638 |
| cellperm | 0.753 | 0.862 | 0.707 | 0.654 |
| stats | 0.505 | 0.718 | 0.588 | 0.550 |
| rand | 0.763 | 0.867 | 0.718 | 0.634 |

## 4. Why

Measured:
- Random features match the real profile. Within-TF, rand equals full in hHEP (+0.009 for rand) and mHSC-E (+0.003), and trails it by 0.028 in hESC and 0.019 in mESC. Almost everything the model scores on a random split comes from the training graph and a memorised identity of each gene. The features add little.
- The scores of cellperm and rand correlate with popularity as strongly as, or more strongly than, those of full (0.76 to 0.87 for rand in three cells). They are popularity estimators: a gene's identity vector, learned from its degree in the message-passing graph, is enough.
- Gene-level summary statistics are poor features on a random split. Stats loses 0.02 to 0.09 within-TF against full and falls below rand in every cell. Twelve numbers per gene cannot identify 700 to 1100 genes, so the memorised-identity route that rand uses is closed, and the statistics alone do not replace it. The best single statistic (detection rate, 0.55 to 0.61) is far below popularity (0.70 to 0.78) and correlates only weakly with it.
- On new genes the picture reverses. The permuted profile and the random vector sit at chance (0.48 to 0.49), while stats reaches 0.560, above full (0.546, p = 0.055). The summary statistics carry as much as the whole profile for a gene the graph has not seen, and they do so without any co-expression.
- Where full beats rand (hESC, mESC) cellperm falls back to rand level (kept share -0.26 and 0.05). The 2 to 3 points that the real profile adds therefore need the cell-by-cell alignment, not the per-gene marginal.

Guess: cellperm cannot show that marginals are irrelevant. A first linear layer reads fixed coordinates, so shuffled coordinates do not expose the order-free statistics of a profile, which is why `stats` is a separate test. The hHEP and mHSC-E result for rand being at or above full may reflect that the extra input dimensions add noise at these sizes. The 0.02 differences are close to the run-to-run swings noted in section 32.

## 5. Verdict

On a random split the model does not read gene-level properties: stats keeps none of the gain and falls below rand, and rand alone reproduces most of full (all of it in hHEP and mHSC-E). The signal is training-graph structure with a memorised gene identity, which behaves as popularity. The co-expression pattern adds a small gain over that (+0.028 hESC, +0.019 mESC, none elsewhere) and it needs the real cell alignment, since cellperm loses it. On held-out genes the answer flips: per-gene statistics match or beat the full profile and permuted profiles do not, so the signal for unseen genes is gene-level (a refined popularity from expression level and detection rate), with no co-expression needed.

Implication: feature engineering has little left to give on a random split (full over rand is at most 0.03), and richer gene statistics do not help there. For unseen genes, gene-level summaries are a cheap and sufficient input, and estimating TF activity from co-expression is not what drives the score.
