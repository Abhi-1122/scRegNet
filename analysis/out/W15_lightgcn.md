# W15 part 2: LightGCN, a collaborative-filtering GNN, for GRN link prediction

## 1. Where the idea comes from
LightGCN (He et al., SIGIR 2020) is a standard recommender-system model: a free embedding per node, propagated with
parameter-free symmetric-normalised neighbour sums (no weights, no non-linearity), with the layer outputs averaged. To our
knowledge it is unused for GRN inference. Section 33 found that the hero is collaborative filtering over known edges, with a unique
vector per gene acting as an identity (random features nearly match expression). LightGCN is that mechanism without the
feature transform: the identity vector is learned directly. For new genes the free embedding never receives a gradient, so a hybrid that adds a
linear projection of node features to E_0 was planned for them.

## 2. Method in short
- `src/pair_gnn.py`: `--encoder lightgcn --lgc_layers K --emb_dim d`; E_0 = `nn.Embedding(n, d)` (normal, std 0.1, `--emb_std`) plus, with
  `--emb_feat {expr, gf, stats}`, a linear projection of standardised expression, Geneformer or the 14 summary statistics; E_{k+1} = D^-1/2 A D^-1/2 E_k on the current
  message-passing graph; output = mean of E_0..E_K. Decoders: the existing Hadamard decoder on a linear + leaky-ReLU map of the output (`had`), or a plain dot
  product plus one bias (`--decoder dot`, LightGCN's scorer). `--emb_l2` is an Adam weight decay on the free embedding only; `--emb_free 0` drops the free
  embedding (feature-only ablation). All flags default off; a `w3_hero` rerun (seed 42) gives within-TF 0.7500 against 0.7499 and the same epoch.
- Leaks: the pair's own edge leaves the graph under `--split batch`, exactly as for SAGE, so a training pair is never scored through its own edge. The free embedding is
  trained by the other pairs of the gene. Features use no labels. Validation and test labels enter no graph.
- Sweep (hESC, seed 42, 144 runs, chosen on validation AUROC): K in {1, 2, 3}, d in {64, 128}, decoder {had, dot}, embedding L2 in {1e-4, 1e-3, 1e-2}, lr in {1e-3, 1e-4},
  emb_feat {none, expr}; decoder weight decay 1e-3. The sweep ran on ada and on the laptop (only validation values were compared); every comparison run is a laptop run, like the baselines.
  Because the default init stalls the dot product, 16 extra dot runs used init std 0.5 or 1.0 and patience 30.
- Chosen on validation: pure LightGCN = K 2, d 128, had, L2 1e-4, lr 1e-3 (val AUROC 0.8864). Hybrid (expr) = K 2, d 64, had, L2 1e-4, lr 1e-4 (0.8854). Dot: std 0.5, lr 1e-2 (0.8265).
  Geneformer and statistics hybrids reuse the pure setting (d 128, lr 1e-3). Feature-only ablations reuse the setting of the matching hybrid.
- Final: seeds 42-44 on all four cells for pure, hybrid-expr, hybrid-Geneformer; hybrid-statistics on hESC, hHEP, mESC; feature-only ablations on hESC and hHEP only (mESC runs take 7-30 minutes; the remaining
  ablations were dropped for time); dot on hESC. Held-out genes and TFs on hESC: 5 folds x seeds 42, 43 for pure and the three hybrids.
- Report `analysis/w15_lgc_report.py` -> `analysis/out/w15_lgc.txt`; jobs `analysis/jobs/w15_sw_*.txt`, `w15_swdot.txt`, `w15_lgc.txt`, `w15_lgc_ho.txt`, `w15_lgc_diag.txt`.

## 3. Results
Sweep. Mean validation AUROC by level: K 2 (0.809) above K 3 (0.779) and K 1 (0.759); had (0.881) far above dot (0.684); embedding L2 1e-4 best (0.806, then 0.781, 0.759); lr 1e-3
slightly better than 1e-4; d and emb_feat make no difference (expr 0.781, none 0.784). The dot scorer reaches 0.826 on validation at best and several of its refits fell far
below their validation score (test AUROC 0.50-0.67 against 0.65-0.78), so it is unstable as well as weaker.

Standard split, within-TF AUROC, test, mean over seeds 42-44 (paired per-TF Wilcoxon p in brackets; "best GNN" = the best grid configuration of section 24, run with Geneformer):

| Model | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| LightGCN, free embedding only | 0.724 | 0.776 | 0.766 | 0.744 |
| Hybrid, + expression | 0.705 | 0.774 | 0.798 | 0.744 |
| Hybrid, + Geneformer | 0.722 | 0.772 | 0.800 | 0.779 |
| Hybrid, + statistics | 0.703 | 0.773 | 0.797 | not run |
| Hero (SAGE, expression) | 0.749 | 0.759 | 0.824 | 0.758 |
| SAGE, random features | 0.721 | 0.768 | 0.805 | 0.760 |
| Best grid GNN | 0.750 | 0.792 | 0.828 | 0.808 |
| Learned DC-SBM | 0.730 | 0.801 | 0.833 | 0.753 |

Pure LightGCN against each: hESC hero -0.025 (p = 0.83), random-feature SAGE +0.003 (0.50), best GNN -0.026 (0.32), DC-SBM -0.005 (0.84). hHEP hero +0.017 (0.09),
random SAGE +0.008 (0.08), best GNN -0.016 (0.17), DC-SBM -0.025 (0.02). mESC hero -0.058 (1e-8), random SAGE -0.039 (9e-5), best GNN -0.062, DC-SBM -0.067 (both < 1e-8).
mHSC-E hero -0.014 (0.70), best GNN -0.065 (0.21), DC-SBM -0.010 (0.78).
Hybrids: expression hybrid on hHEP beats the hero by +0.016 (p = 0.005) and random SAGE by +0.007 (p = 0.02), and trails the best GNN by 0.018 (p = 0.24); on hESC it is below the hero by 0.044 (p = 2e-5) and below the pure model by
0.019 (p = 0.009); on mESC it adds +0.032 to the pure model (p = 5e-5) and is still 0.026 below the hero. Geneformer hybrid on mHSC-E: +0.035 over the pure model (p = 0.33), +0.021 over the hero (p = 0.39),
0.030 below the best GNN (p = 0.13). AUROC and AUPRC move in the same direction (full table in `w15_lgc.txt`).

hESC held-out (5 folds x seeds 42, 43, within-TF; difference to the hero in brackets):

| Model | Held-out genes | Held-out TFs |
|---|---|---|
| Hero | 0.546 | 0.686 |
| LightGCN, free embedding only | 0.502 (-0.041, p = 0.015) | 0.670 (-0.018, p = 0.91) |
| Hybrid, + expression | 0.564 (+0.014, p = 0.09) | 0.481 (-0.206, p = 4e-9) |
| Hybrid, + Geneformer | 0.567 (+0.016, p = 0.13) | 0.673 (-0.013, p = 0.53) |
| Hybrid, + statistics | 0.546 (+0.001, p = 0.09) | 0.690 (+0.004, p = 0.47) |

## 4. Why
- Which part carries the signal (hESC ablation): the free embedding. Free only 0.724; features only 0.701 (expression), 0.717 (Geneformer); both 0.705 (expression), 0.722 (Geneformer). Adding expression to the free
  embedding lowers the result by 0.019 (p = 0.009), adding Geneformer changes it by -0.002 (p = 0.90). On mESC the opposite holds: features add +0.032, probably because the free-embedding-only run is held back by the decoder weight decay (below).
  The Spearman of test scores with gene popularity within a TF is 0.92 for the pure model (hero 0.72, random-feature SAGE 0.76, expression only 0.60, Geneformer only 0.69): the pure model is
  mostly a learned popularity plus a low-rank interaction, which is the same reading as section 33, taken further.
- Gain per TF (hESC): pure minus random-feature SAGE is larger for TFs with many training targets (+0.020 above the median of 44 targets, -0.014 below; Spearman of the gain with the number of targets +0.33, p = 0.06).
  Against the hero, the pure model loses most on TFs with few targets (-0.043 against -0.006). A TF's embedding is the mean of its few targets' embeddings, so a TF with few targets has little to average.
- Why LightGCN does not beat the hero on hESC: it matches the random-feature SAGE (+0.003, ns) and the hero's lead over random features (+0.028) is the co-expression part that a free embedding cannot supply. The hybrid with expression
  was meant to supply it but does not (-0.019), presumably because the projected 758-dimensional expression adds a weight-heavy path that overfits at the chosen lr 1e-4 (a guess; not tested further).
- mESC diagnostic (one seed, about 10 minutes, nothing chosen on it): widening the embedding (d 256, 512) changes nothing (0.768); one layer gives 0.803; setting the decoder weight decay to 0 with d 256 gives within-TF 0.823 and validation AUROC 0.9418
  (against 0.9265), level with the hero (0.824). So the mESC loss comes largely from the weight decay of 1e-3 that was kept from the hero setting and not from the model class. The sweep did not vary the decoder weight decay (an omission).
  This one-seed value was not run at three seeds, so the mESC table above stays the protocol result.
- Held-out genes: the pure model sits at the random-feature level (0.502 against 0.491) because new genes have an untrained embedding; the expression and Geneformer hybrids lift it to 0.564 and 0.567 (hero 0.546), but with 10 runs
  and p near 0.1 this is a lead at best. Held-out TFs: the expression hybrid collapses (0.481, below 0.5): its projection of expression is fitted on TFs seen in training and reads a new TF's profile as a gene's. The statistics hybrid (0.690) matches the hero.

## 5. Verdict
LightGCN works as expected for what it is, a learned-popularity collaborative filter. It ties random-feature SAGE on hESC and mHSC-E, beats the hero on hHEP (+0.016 to +0.017, p = 0.005 for the expression hybrid, 0.09 for the pure model) without
reaching the best grid GNN or the learned DC-SBM there, and trails everywhere else under the protocol settings. It does not set a new best on any cell. The hybrids give a small, non-significant lead on held-out genes (+0.014 to +0.016) and match the hero on held-out
TFs only with the statistics features. Worth trying next: decoder weight decay 0 (it recovers the hero on mESC in one seed), plus a re-sweep of that knob on hESC; the Geneformer hybrid on mHSC-E (+0.021 over the hero) with more seeds.
