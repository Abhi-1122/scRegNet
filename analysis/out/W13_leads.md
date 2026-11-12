# W13 leads: a second TF summary and the block-model score on each cell's best configuration

Best grid configuration of each cell (section 24; Hadamard decoder, Geneformer on, Adam lr 1e-4, wd 1e-3, epoch chosen on validation AUROC, refit on all training rows): hESC and hHEP neural PANDA (2 layers, RCP, batch split), mHSC-E GCN (1 layer, batch split), mESC SAGE (1 layer, no split). Seeds 42-44. Baselines are the grid runs of that configuration (`w5_nphad_gf`, `w11_grid_hHEP_npanda_had_batch_gf1`, `w8_mHSC-E_gcngf`, `w11_grid_mESC_sage_had_none_gf1`). Paired tests: per-TF test AUROC averaged over the seeds, Wilcoxon over TFs. Reproduce: `python3.12 analysis/w13_report.py` (writes `analysis/out/w13_leads.txt`). Jobs: `analysis/jobs/w13_*.txt`.

## 1. Where the leads come from

- **Lead A.** Section 25 found that a plain mean of a TF's known targets' post-GNN embeddings, compared with the gene (`--decoder qatt --qatt_uniform 1`), helps hHEP (+0.011), mHSC-E (+0.011) and held-out genes (+0.018). The comparison was against the hero without Geneformer. The open question was whether it survives on the best configuration of each cell.
- **Lead B.** Section 31 found that the learned DC-SBM (logit = a_t + b_g + log sum_k softmax(u_g)_k exp(W_tk)) ties the best GNN on hHEP and mESC while its test scores differ from the GNN's (Spearman 0.86-0.93). Its score was therefore given to the GNN decoder as an input.

## 2. Method in short

- **Lead A.** No new code. Same flags as section 25 (1 head, post-GNN embeddings, uniform weights, no top-k), `--decoder qatt` instead of `--decoder had`. Code audit of `PairGNN.qctx` and `fit`:
  1. The known-target mask `ctx.A` is built in `full_graph` from `OT[keep], OG[keep]`, the TF-to-gene orientation of the same unique edges that form the R relation. It does not depend on the encoder, so with the neural PANDA encoder only R edges enter it (the C and P relations are separate tensors), and with GCN/SAGE it is the same graph. Under `--split batch` the batch's own edges are already outside `keep`.
  2. Under `--split none` (mESC) the pair's own gene is masked out of its TF's target set (`mask[arange, g] = False`), and the reverse edge cannot enter: the set is A[t], and a reverse row (g, t) with g a TF has orientation (t, g) or (g, t) stored once per undirected edge, so A[t, g] is either masked or absent. No code change was needed. As an empirical check, `--own_edge_diag 1` on mESC (seed 42) gives a positive-edge logit gap (edge present minus removed) of +0.132 for the baseline and +0.127 with the summary, and AUROC present/removed 0.959/0.951 for both. The known no-split cue of the baseline is unchanged by the summary and not enlarged.
- **Lead B.** New scalar kind `dcsbm` in `analysis/scalars.py`, used through the existing `--scalar` mechanism (`get` returns the evaluation matrix and the out-of-fold matrix). The learned DC-SBM of `w12_stat.py` (`fit_torch('sbm')`) is fitted on all labelled training rows (positives and negatives; unlabelled pairs missing) with the hyper-parameters chosen on validation there: hESC k=5, l2=10, 300 steps; hHEP k=5, l2=10, 150; mHSC-E k=5, l2=10, 250; mESC k=10, l2=3, 150. Evaluation matrix = fit on the training rows (validation rows excluded in the selection stage, all training rows in the refit stage; the cache key holds `val_frac` and `split_seed`, as for the other kinds). Out-of-fold matrix = for each of the 5 folds (same fold split as `panda`) a refit without that fold's rows, scored on that fold, so no training pair is scored with its own label. The fit seed is fixed at 42; the matrix is divided by its standard deviation. Check: the evaluation matrix of the refit stage reproduces the `w12_stat.py` test within-TF exactly (hESC 0.7319, hHEP 0.8006 at seed 42), and the out-of-fold training AUROC (0.873 hESC, 0.890 hHEP) matches the test AUROC (0.875, 0.895). A fit takes about a second. `netZooPy` is now imported lazily (the cluster has no netZooPy); nothing else in `scalars.py` changed. `src/pair_gnn.py` was not edited.
- **Knobs.** Concatenation into the decoder (`dc`) against a separate linear term at 10x learning rate (`dclr10`), chosen per cell on seed-42 validation AUROC: dclr10 in all four cells (hESC .8965 vs .8950, hHEP .9050 vs .9028, mHSC-E .9258 vs .9248, mESC .9396 vs .9389). The choice was first made from incomplete seed-42 results for hHEP and mHSC-E (picked `dc`), found wrong once all runs finished, and corrected; the `dc` runs at 3 seeds stay in the tables as "not chosen".
- Both ideas together were also run (`qs + dclr10`). The runs made on the cluster (held-out genes, `qs + dclr10` on hESC and mESC) were discarded and rerun on the laptop: three cluster runs of hESC `qs + dclr10` had a seed spread of 0.0001 AUROC and within-TF 0.734, whereas the same runs on the laptop gave 0.739, 0.754, 0.754. The cause was not found. All tables use laptop runs only.

## 3. Results

Within-TF test AUROC, mean ± sd over seeds 42-44 (AUROC and AUPRC are in the report file). Paired difference from the baseline in brackets: mean over TFs, p.

| Cell | Baseline | + summary (qs) | + DC-SBM scalar (dclr10) | + both |
|---|---|---|---|---|
| hESC | 0.7502 ± 0.0089 | 0.7513 [+0.0011, p=0.48] | 0.7516 [+0.0015, p=0.50] | 0.7476 [-0.0025, p=0.24] |
| hHEP | 0.7919 ± 0.0036 | 0.7885 [-0.0034, p=0.61] | 0.7937 [+0.0018, p=0.23] | 0.7903 [-0.0016, p=0.35] |
| mHSC-E | 0.8076 ± 0.0016 | 0.7930 [-0.0147, p=0.51] | 0.8016 [-0.0061, p=0.29] | 0.7924 [-0.0153, p=0.98] |
| mESC | 0.8284 ± 0.0007 | 0.8287 [+0.0003, p=0.60] | **0.8335 [+0.0051, 57/85 TFs, p=2e-4]** | **0.8352 [+0.0068, 62/85, p=4e-6]** |

AUROC / AUPRC, mESC: baseline 0.9399 / 0.8830; scalar 0.9420 / 0.8875; both 0.9425 / 0.8885. hHEP: baseline 0.9041 / 0.8467; scalar 0.9048 / 0.8476. The `dc` variant (concatenation) on hHEP gives 0.7917 (-0.0002, p=0.33) and on mHSC-E 0.8035 (-0.0042, p=0.05).

- **hHEP does not beat 0.789 significantly.** The 3-seed baseline is 0.7919 (the 0.789 of section 24 is the 5-seed mean). The best variant, the scalar, reaches 0.7937 (+0.0018, 20/30 TFs, p=0.23).
- **hHEP held-out genes** (best hHEP configuration, 5 folds x seeds 42-43): within-TF 0.6563 baseline, 0.6503 with the summary (-0.0068, 8/30 TFs better, p=0.007). AUROC 0.8375 vs 0.8385.

Diagnostic (not a method), 50/50 rank average of the baseline GNN and the DC-SBM test scores (global ranks), within-TF; difference from the GNN baseline:

| Cell | GNN | DC-SBM | Rank average | vs GNN |
|---|---|---|---|---|
| hESC | 0.7502 | 0.7295 | 0.7538 | +0.0036 (p=0.03) |
| hHEP | 0.7919 | 0.8006 | 0.7975 | +0.0056 (24/30, p=0.001) |
| mHSC-E | 0.8076 | 0.7531 | 0.7986 | -0.0091 (p=0.03) |
| mESC | 0.8284 | 0.8332 | 0.8428 | +0.0144 (72/85, p=4e-11) |

The rank average is also above the DC-SBM alone on mESC (+0.0096, p=5e-4) and mHSC-E (+0.0454, p=0.02). Its mESC AUROC is 0.9436 / AUPRC 0.8919, against 0.9425 / 0.8885 for the best in-network variant (both ideas).

## 4. Why

**Lead A (loss, 10 minutes of diagnostics).**
- The own-edge check above rules out a leak or a new shortcut under `--split none`.
- The gain of section 25 shrinks as the baseline gets stronger. On hHEP, `qs` minus baseline within-TF: sage without Geneformer (section 25 setting) +0.0107 (21/30, p=0.014); sage with Geneformer +0.0064 (p=0.14); neural PANDA without Geneformer +0.0057 (20/30, p=0.010); neural PANDA with Geneformer (the best configuration) -0.0034 (p=0.61). Each added source (Geneformer, a second message-passing layer with the C and P relations) lowers the gain, and with both it is gone. This is a measured pattern over four configurations. Why it happens is a guess: the second summary is a two-step view of the TF (TF to its targets to their regulators), and a 2-layer encoder with Geneformer features may already hold that information in the TF embedding.
- The per-TF gain does not follow the number of training targets (Spearman between -0.12 and +0.31, all p > 0.09 on the four cells). On held-out hHEP genes the loss is larger on TFs with few targets (-0.0108 below the median of 225, -0.0029 above; Spearman +0.44, p=0.015), the reverse of the hESC held-out gain in section 25.
- mHSC-E loses -0.015 on average but only 13/29 TFs go down (p=0.51): TFs below the median of 314 targets lose 0.031, above 0.000.

**Lead B (gain on mESC only).**
- Information added: a label-trained TF-by-gene-block interaction, fitted on positives and negatives and scored out of fold, so the GNN receives a gene-and-TF score that its message passing does not compute (its graph holds positives only). Section 31 showed this term is what gives the DC-SBM its edge over popularity on mESC (+0.066 over the one-block control).
- Where it lands: the mESC gain is spread over TFs (+0.0050 on TFs below the median of 156 targets, +0.0052 above; Spearman with the number of targets -0.07, p=0.5). It correlates with the DC-SBM's own advantage over the baseline (Spearman +0.61, p=7e-10), but that statistic shares the baseline's per-TF noise with the gain; the cleaner comparison, with the DC-SBM's per-TF AUROC alone, is +0.15 (p=0.17). The relation is therefore suggestive, not shown.
- mESC is the one cell with no batch split, so the baseline reads each training edge through its own edge; a score computed out of fold gives the decoder an edge-free signal. This is a guess and was not tested.
- Elsewhere the scalar is neutral (hESC, hHEP) or negative (mHSC-E, -0.006, p=0.29): on mHSC-E the DC-SBM is far below the GNN (0.753 vs 0.808), and the decoder cannot be told to trust it only where it helps.
- The rank average is a stronger combiner than the in-network input. It gains on mESC (+0.0144 vs +0.0051), hHEP (+0.0056, p=0.001 where the scalar's +0.0018 is not significant) and hESC (+0.0036), and loses on mHSC-E where the DC-SBM is much weaker. The two models' errors therefore complement each other on hHEP and mESC, as section 31 suggested, but the learned-input route captures only part of it.

## 5. Verdict

- Lead A is negative: on the best configuration of each cell the second TF summary adds nothing (hESC +0.001, hHEP -0.003, mHSC-E -0.015, mESC +0.000, none significant) and costs 0.007 on held-out hHEP genes. Not worth keeping for the best configurations.
- Lead B gives one real gain: mESC +0.005 within-TF (p=2e-4), +0.007 with both ideas (p=4e-6). hHEP stays at its tie (+0.002, p=0.23), and hESC and mHSC-E gain nothing. The late rank average (+0.014 mESC, +0.006 hHEP) is the larger and more consistent effect; if either route goes into the curated pipeline, a per-cell late fusion with the DC-SBM on hHEP and mESC is the one to test with the choice made on validation.
- Selection: the dclr10 variant was chosen on validation per cell; the cell list and configurations are the section 24 test-picked bests, so baselines are slightly optimistic.
