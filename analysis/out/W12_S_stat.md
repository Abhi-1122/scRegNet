# W12 S: statistical network models (DC-SBM, Poisson factorisation, latent space) against gene popularity and the GNN

Tables regenerate with `python3.12 analysis/w12_stat_report.py` from the saved scores in `analysis/out/stat/`; the full output is `analysis/out/w12_stat.txt`. Models in `analysis/w12_stat.py`. CPU only, no GPU, no ada.

## 1. Where the idea comes from

- A bipartite TF x gene label matrix with a strong gene-popularity effect is the textbook case for the degree-corrected stochastic block model (DC-SBM; Karrer and Newman 2011): the edge rate is `theta_g * omega(t, block(g))`, a per-gene degree factor times a TF-by-block preference. Gene blocks here come from a Gaussian mixture (GMM) on gene features, or are learned. The soft version is a mixed-membership block model (MMSB; Airoldi et al. 2008). Hierarchical Poisson factorisation (HPF; Gopalan et al. 2015) is the matching low-rank model for sparse count matrices with per-row and per-column activity. The latent space model (Hoff, Raftery and Handcock 2002) places TFs and genes in R^d and lets the edge probability fall with distance.
- Project findings that motivated it: gene popularity (training in-degree) alone reaches within-TF AUROC 0.707 (hESC), 0.784 (hHEP), 0.713 (mHSC-E), 0.761 (mESC), and a five-feature stacker without a neural network reaches 0.738 / 0.759 / 0.742 / 0.814. The preprint of Stock et al. (bioRxiv 2025.12.19.695616) reports that degree heuristics match GNNs on BEELINE. The question tested: does a principled "popularity x TF preference for gene programs" model match the GNN, and does the program term add over popularity?

## 2. Method in short

Every (TF, gene) pair of the TF x gene grid is scored by a model fitted on the training labelled pairs only (unlabelled pairs are missing, not zero). Splits, validation slice (val_frac 0.1, split_seed 0) and test rows are the ones of the GNN runs (`prepare` from `src/pair_gnn.py`). Each knob was chosen on the validation slice (mean over the seeds of the within-TF AUROC), the chosen config was refit on all training rows, and the test rows were scored once. Seeds 42-44 change GMM, EM and gradient initialisation. No validation or test label enters any fit. Gene features are used without labels.

| Model | Score | Knobs searched on validation |
|---|---|---|
| popularity (control) | log of the smoothed positive rate of the gene, `(P_g + a m) / (L_g + a)`; raw in-degree as a second control | a in {0.3 ... 30} |
| 1. DC-SBM, fixed clusters | `log theta_g + log sum_k p(k given g) omega(t, k)`; omega = Beta-smoothed positive rate of TF t among its labelled pairs in cluster k; optional division of omega by the mean theta of the cluster (DC normalisation) | GMM on (a) Geneformer (30 PCs) or (b) expression (40 PCs); diag or full covariance; K in {5, 10, 20, 40}; soft or hard assignment; theta strength {1, 3, 10}; theta prior global or cluster rate; omega strength {1, 3, 10, 30}; DC normalisation on or off |
| 2. DC-SBM, learned clusters | logit = `a_t + b_g + log sum_k softmax(u_g)_k exp(W_tk)`, Bernoulli likelihood on observed pairs, Adam full batch | K in {5, 10, 20}; L2 {1 ... 100} on b and W (0.1 on u); steps up to 650 |
| 3. Poisson factorisation | masked Poisson rate `exp(a_t + b_g) (1 + theta_t . phi_g)`, theta and phi non-negative (softplus), MAP | rank {4, 8, 16}; L2 {0.03 ... 1}; steps up to 1000 |
| 4. Latent space | logit = `a_t + b_g - ||z_t - z_g||`, one z per node (a TF uses the z of its own gene), L2 on b and z | d in {2, 4, 8}; L2 {1 ... 100}; steps up to 650 |
| main-effects control | model 2 with K = 1: logit = `a_t + b_g` (Bernoulli ridge popularity), no program term | L2, steps |

Ablations of model 1: `omega only` (drop log theta) and `theta only` (drop the program term, keep the selected theta settings and the TF rate). Model 2 is a gradient-descent soft-membership relaxation, not a variational EM. Held-out genes: model 1(a) with theta set from the soft-cluster mean rate (prior of the shrinkage when a gene has no labels) and model 4 with z_g and b_g of an unseen gene predicted by ridge regression from the Geneformer PCs. Held-out TFs were skipped: an unseen TF has no omega row and no latent position.

Comparison runs, same test rows and seeds 42-44: hESC hero `w3_hero`; hHEP, mESC, mHSC-E hero `m_hHEP_e_sage_d256`, `w4_mESC_hero`, `w4_mHSC-E_hero` (SAGE, Hadamard decoder, batch split, no Geneformer; the files `w11_grid_<cell>_sage_had_batch_gf0` do not exist, these runs have the same configuration). Best GNN per cell: `w8_hESC_tuned`, `w8_hHEP_gcngf`, `w8_mESC_tuned`, `w8_mHSC-E_gcngf` (PROGRESS section 21).

## 3. Results

### 3.1 Random split, within-TF AUROC (test, mean over seeds 42-44; std in the output file)

| Model | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| popularity, in-degree | 0.7068 | 0.7838 | 0.7612 | 0.7132 |
| popularity, smoothed rate | 0.7176 | 0.7833 | 0.7647 | 0.7194 |
| main-effects control (K = 1) | 0.7184 | 0.7871 | 0.7668 | 0.7312 |
| 1. DC-SBM fixed clusters, best on validation | 0.7014 (expr, hard) | 0.7680 (expr, soft) | 0.7731 (expr, hard) | 0.7444 (expr, soft) |
| 1. same family, Geneformer clusters, soft | 0.7252 | 0.7738 | 0.7669 | 0.7500 |
| 1. omega only (program term alone) | 0.548 | 0.519 | 0.636 | 0.653 |
| 2. DC-SBM learned clusters | **0.7295** | **0.8006** | **0.8332** | 0.7531 |
| 3. Poisson factorisation | 0.7033 | 0.7803 | 0.8250 | 0.7218 |
| 4. latent space | 0.7234 | 0.7998 | 0.8229 | 0.7457 |
| hero | 0.7490 | 0.7583 | 0.8241 | 0.7575 |
| best GNN | 0.7614 | 0.7898 | 0.8281 | 0.8076 |

Best statistical model by validation within-TF: learned DC-SBM on hESC (val 0.7387), hHEP (0.7772) and mESC (0.8312); fixed-cluster DC-SBM on expression clusters on mHSC-E (0.8200; the learned DC-SBM, val 0.8156, has test 0.7531). Pooled test AUROC / AUPRC of smoothed popularity with the TF rate: 0.881 / 0.603 (hESC, hero 0.889 / 0.625), 0.897 / 0.833 (hHEP, 0.899 / 0.838), 0.924 / 0.850 (mESC, 0.939 / 0.881), 0.917 / 0.935 (mHSC-E, 0.923 / 0.943).

### 3.2 Paired per-TF tests (per-TF test AUROC averaged over seeds, difference, Wilcoxon p)

| Cell | Best statistical model | minus popularity | minus hero | minus best GNN |
|---|---|---|---|---|
| hESC | learned DC-SBM | +0.0119 (p = 0.32) | -0.0195 (p = 0.54) | -0.0319 (p = 0.019) |
| hHEP | learned DC-SBM | +0.0174 (p = 6e-4) | +0.0423 (p = 0.0016) | +0.0109 (p = 0.48) |
| mESC | learned DC-SBM | +0.0685 (p = 6e-11) | +0.0091 (p = 0.0019) | +0.0051 (p = 0.059) |
| mHSC-E | fixed DC-SBM (expr, soft) | +0.0250 (p = 0.44) | -0.0131 (p = 0.13) | -0.0632 (p = 5e-4) |

Further pairs: hero minus popularity is +0.031 (p = 0.21) on hESC, -0.025 (p = 0.26) on hHEP, +0.060 (p = 1e-7) on mESC and +0.038 (p = 0.06) on mHSC-E. The best GNN minus popularity is +0.044 (p = 0.010), +0.0065 (p = 0.12), +0.063 (p = 7e-8) and +0.088 (p = 6e-4). The learned DC-SBM on mHSC-E gains +0.034 over popularity (p = 0.001) and the latent space model +0.026 (p = 8e-6) but both stay below the best GNN.

### 3.3 Held-out genes (hESC, 5 folds x seeds 42-43, mean ± std over the 10 fold-seed pairs)

| Model | AUROC | AUPRC | within-TF | within >= 8 |
|---|---|---|---|---|
| hero (`w5_ho_gene*_hero`) | 0.8467 ± 0.0064 | 0.5213 ± 0.0111 | 0.5461 ± 0.0305 | 0.5617 |
| DC-SBM, Geneformer clusters, soft | 0.8573 ± 0.0057 | 0.5379 ± 0.0191 | **0.5753 ± 0.0277** | 0.5782 |
| same, hard assignment | 0.8518 | 0.5224 | 0.5564 | 0.5656 |
| same, theta only (cluster mean degree) | 0.8564 | 0.5392 | 0.5214 | 0.5331 |
| same, omega only | 0.8526 | 0.5016 | 0.5603 | 0.5579 |
| latent space (ridge-placed genes) | 0.8538 | 0.5360 | 0.5567 | 0.5846 |

Paired over (fold, TF) pairs (155 pairs): DC-SBM soft minus hero +0.0289 (p = 0.008); hard minus hero +0.0099 (p = 0.36); latent space minus hero +0.0104 (p = 0.14); DC-SBM soft minus its own theta-only version +0.0534 (p = 8e-6). Per fold the DC-SBM beats the hero in folds 0 to 3 and loses in fold 4 (0.539 against 0.568). Validation on unseen genes is noisy (about 70 validation genes per fold), and the chosen config changes from fold to fold.

## 4. Why

Measured:

- **The program term from fixed clusters adds nothing reliable.** Dropping log theta (`omega only`) leaves within-TF 0.52 to 0.55 on hESC and hHEP and 0.64 to 0.65 on mESC and mHSC-E. The best fixed-cluster DC-SBM is within 0.02 of popularity on every cell, never significant against popularity (p >= 0.13) and not better than its own theta-only version (p >= 0.13 in all comparisons). The K sweep is flat: per K the test within-TF varies by about 0.02 around the popularity value with no trend, and the config that wins on validation is often not the one that wins on test (hESC Geneformer soft: K = 5 chosen on validation 0.725, K = 10 0.738 on test).
- **Learned blocks and latent positions add something, large on mESC.** Against the main-effects control (same Bernoulli fit, no program term) the learned DC-SBM gains +0.011 (hESC, p = 0.38), +0.014 (hHEP, p = 0.04), +0.066 (mESC, p = 1e-11) and +0.022 (mHSC-E, p = 0.056); the latent space model +0.005 (p = 0.50), +0.013 (p = 0.001), +0.056 (p = 2e-11), +0.015 (p = 0.001). The main-effects control itself is close to smoothed popularity (hESC +0.001, mESC +0.002, not significant; hHEP +0.004, p = 0.007; mHSC-E +0.012, p = 3e-4). Poisson factorisation helps only on mESC (+0.058 over the control). Where the learned program adds, it is not the gene clusters defined by expression or Geneformer: the same K and the same likelihood with GMM clusters stay at popularity level. A guess, not tested: the useful structure is co-targeting (groups of genes that the same TFs hit), which unsupervised feature clusters do not recover.
- **Gain versus TF size and omega concentration.** For the fixed-cluster DC-SBM the per-TF gain over popularity rises with the number of training targets on hESC (Spearman +0.46, p = 0.008; TFs with at most 38 targets -0.047, with more +0.013) and is not related to it on the other cells (hHEP +0.32, p = 0.08; mESC +0.21, p = 0.05; mHSC-E -0.13, p = 0.5). The gain is not larger for TFs with a more concentrated omega profile (Spearman with normalised entropy +0.35, +0.14, +0.12, -0.24; with the spread of log lift -0.34, -0.16, -0.11, +0.24). The omega profiles are close to flat everywhere: normalised entropy 0.95 (hESC), 0.95 (hHEP), 0.98 (mESC), 0.96 (mHSC-E). The learned models give their gain mostly to small TFs: mESC TFs with at most 156 targets gain +0.086 over popularity, larger TFs +0.050; mHSC-E +0.052 and +0.015 (Spearman -0.35, p = 0.06).
- **hESC key TFs** (Geneformer K = 10, soft, seed 42; preference is the TF-normalised lift, 1 = the TF's own average). The lifts are modest (1.0 to 1.3 for the three large TFs, 1.3 to 1.8 for the two small ones where 14 or 15 training targets make the estimate noisy). NANOG prefers cluster 3 (80 genes, lift 1.19; RGPD2, CER1, HESX1, ASNS), cluster 7 (49; ZNF589, TDRD7, DEPDC1B, SALL2) and cluster 9 (48; SKA3, CCNA2, EXO1, CDC7, NUSAP1). SOX2 prefers cluster 9 (48 genes, lift 1.18; MCM4, CCNA2, CDC7, NUSAP1, MCM7, E2F7, a replication and cell-cycle group), then cluster 4 (233 genes, 1.07) and cluster 1 (93; OTX2, ZNF219, SNAI2, LMO2). POU5F1 prefers cluster 3 (1.33) and cluster 2 (1.30); SOX17 cluster 8 (113 genes, 1.64; CSDE1, PHB2, ILF2) and 6; GATA6 clusters 3, 1, 2 (1.7 to 1.8, based on 14 targets); EOMES cluster 5 (39 genes, 1.16; RBFOX2, TCF7L1), cluster 1 (OTX2, SNAI2, MAFF, RXRG) and cluster 8. NANOG and SOX2 sharing the cell-cycle cluster 9 is the one clearly interpretable pattern; the others are small differences on few targets. Cluster labels are not checked against gene-set annotation.
- **Why hHEP is popularity-dominated** (table below). On hHEP the program term carries no TF-specific signal at fixed clusters (omega only 0.519, chance level, against 0.636 and 0.653 on mESC and mHSC-E), the learned program adds the least among the cells except hESC (+0.014 over the control), and every model sits near popularity (0.783): hero 0.758, best GNN 0.790 (+0.0065 over popularity, p = 0.12), learned DC-SBM 0.801 (+0.017, p = 6e-4). The hero GNN scores closest to popularity of the four cells (per-TF Spearman of score with gene popularity on the test rows 0.835 on hHEP against 0.721 on hESC, 0.717 on mESC and 0.638 on mHSC-E). Measured correlates: TFs in hHEP share targets (mean pairwise cosine of the TFs' training target vectors 0.25 against 0.09 on hESC and 0.19 on mESC), the omega profiles are flatter than on hESC (spread of log lift 0.30 against 0.41). The mean cosine does not explain the ranking alone, since mHSC-E has the highest value (0.36) and a clear gain from non-popularity structure; the explanation offered is therefore a guess: in hHEP the targets of different TFs overlap strongly and what separates the TFs is not carried by expression-defined gene groups.

| Cell | pop within-TF | omega only | best fixed DC-SBM minus pop | learned DC-SBM minus pop | learned minus main effects | entropy of omega | spread of log lift | targets per TF | TF-TF cosine | hero score vs popularity (Spearman) |
|---|---|---|---|---|---|---|---|---|---|---|
| hESC | 0.7176 | 0.548 | -0.016 | +0.012 | +0.011 | 0.950 | 0.409 | 89 | 0.094 | 0.721 |
| hHEP | 0.7833 | 0.519 | -0.015 | +0.017 | +0.014 | 0.947 | 0.300 | 221 | 0.252 | 0.835 |
| mESC | 0.7647 | 0.636 | +0.009 | +0.069 | +0.066 | 0.981 | 0.251 | 225 | 0.190 | 0.717 |
| mHSC-E | 0.7194 | 0.653 | +0.025 | +0.034 | +0.022 | 0.957 | 0.251 | 267 | 0.356 | 0.638 |

- **Score correlation with popularity** (per-TF Spearman on the test rows, mean over TFs and seeds; hESC / hHEP / mESC / mHSC-E): smoothed popularity 0.98 / 0.98 / 0.97 / 0.87; fixed-cluster DC-SBM (Geneformer soft) 0.88 / 0.89 / 0.81 / 0.71; learned DC-SBM 0.89 / 0.90 / 0.80 / 0.73; latent space 0.91 / 0.91 / 0.85 / 0.80; hero 0.72 / 0.84 / 0.72 / 0.64; best GNN 0.75 / 0.82 / 0.72 / 0.63. The statistical models that beat popularity keep a higher correlation with popularity than the GNN. Their scores correlate with the hero logit at 0.67 to 0.85. Popularity correlates with it at 0.72 / 0.85 / 0.73 / 0.71; the learned DC-SBM at 0.73 / 0.83 / 0.80 / 0.77, so it is closer to the hero than popularity is on mESC and mHSC-E and equally close on hESC and hHEP.
- **Held-out genes.** With no in-degree available, the program term is what carries the score: DC-SBM soft is +0.053 over its theta-only version (p = 8e-6), and +0.029 over the hero (p = 0.008). The hero is near chance on unseen genes (0.546); the cluster posterior of a gene from Geneformer already places an unseen gene in a program with some TF preference. Hard assignment keeps about two thirds of the gain over theta-only (0.556 against 0.575 and 0.521), so the soft posterior carries part of the information.

## 5. Verdict

- A statistical network model matches the GNN on the cells where popularity is a large share of the signal: the learned DC-SBM beats the hero on hHEP (+0.042, p = 0.002) and mESC (+0.009, p = 0.002) and is within noise of the best GNN there (+0.011, p = 0.48; +0.005, p = 0.06). It does not match on hESC (-0.032 against the tuned model, p = 0.02) or on mHSC-E (-0.063, p = 5e-4). Pooled AUROC of popularity plus the TF rate is within 0.002 to 0.015 of the hero, so the claim of Stock et al. is reproduced for pooled metrics.
- The program term from GMM clusters on features does not add over popularity on seen genes; a learned program (blocks, latent space) adds +0.01 to +0.07 within-TF, most on mESC. The gain on unseen genes (+0.03 over the hero, +0.05 over cluster-degree alone) is the one place where feature-defined clusters help.
- Limits: the learned models are fitted by gradient descent as a soft-membership relaxation; knobs were picked on a validation slice of about 2000 rows, which is noisy (the mHSC-E selection picked a fixed-cluster model whose test value 0.744 is 0.075 below its validation value); hero runs of hHEP, mESC and mHSC-E are the older equal-configuration runs.
- Next, if pursued: use a learned block or latent score as one input of the late-fusion ensemble (its scores are only partly redundant with the GNN, correlation 0.67 to 0.85), and test fusion on mESC and hHEP where it matches the GNN alone.

Compute: one process at a time, `ulimit -v 6000000` (4 GB failed at the torch import), peak resident memory 0.78 to 0.86 GB per run; wall time per cell 5 to 13 min, per held-out fold 2 min.
