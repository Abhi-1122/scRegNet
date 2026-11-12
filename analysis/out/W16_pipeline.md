# W16: the curated pipeline (all choices on validation)

**Verdict: FAIL.** Rule: PASS if the pipeline beats the scRegNet recipe on within-TF in all 4 cells and on hESC held-out genes and TFs, and is not below the best single model anywhere. Within-TF AUROC, seeds 42-44, seed mean:

- Standard split: the pipeline beats the recipe in all 4 cells (hESC +0.044, hHEP +0.020, mHSC-E +0.049, mESC +0.075; p between 4e-11 and 0.012). Same result as a seed ensemble against an equally ensembled recipe.
- hESC held-out genes: passes against the recipe baseline (+0.062, p = 0.002) and against the hero (+0.016, p = 0.11).
- **hESC held-out TFs: fails.** Pipeline 0.6850 against 0.7163 for the recipe baseline (-0.0320, 6/34 TFs better, p = 1e-4), and 0.6855 for the hero (-0.0014, p = 0.40). Within the >= 8 positives subset 0.6657 against 0.7045.
- **mHSC-E: below the best single model** (the validation-chosen configuration alone): -0.0007 as a per-seed mean, -0.0016 as a seed ensemble (p = 0.09 and 0.21, so within noise, but not at or above it).

The recipe baseline on held-out data is the "scRegNet design, Adam-tuned" runs (`ho_*_f01_A_tuned`: GCN, add decoder, three layers 227/29/34, Geneformer, tuned Adam), not the published SGD recipe. No held-out runs of the published SGD recipe exist and none were launched. Popularity alone scores 0.7244 on held-out TFs, above every GNN.

## What went in and how each choice was made

Everything except the final tables used validation numbers only (the 10% validation slice of the training rows, split_seed 0; best-epoch metrics). Criterion everywhere: validation within-TF AUROC, mean over seeds; validation AUROC is shown beside it in `analysis/out/w16_pipeline.txt`.

1. **GNN configuration.** Among the 24 grid configurations (seeds 42-46, `analysis/w16_choose.py`, `analysis/out/w16_choose.txt`). Validation picked a different set than the test-picked grid best of section 24 in three cells:

| Cell | Validation-chosen | Val within-TF | Runner-up on validation | Test-picked grid best (section 24) |
|---|---|---|---|---|
| hESC | GCN 1 layer, Hadamard, batch, Geneformer | 0.7437 | neural PANDA (4th, 0.7269) | neural PANDA |
| hHEP | GCN 1 layer, Hadamard, batch, Geneformer | 0.7842 | neural PANDA 0.7841, GCN no split 0.7838 | neural PANDA |
| mHSC-E | neural PANDA 2 layers, Hadamard, batch, Geneformer | 0.8397 | SAGE 0.8387 | GCN |
| mESC | neural PANDA 2 layers, Hadamard, no split, Geneformer | 0.8334 | neural PANDA batch 0.8329 | SAGE no split |

   The top two on validation are within 0.0004 (hHEP, top three), 0.0010 (mHSC-E) and 0.0005 (mESC), so these are near ties. Hero hyper-parameters (Adam, lr 1e-4, wd 1e-3, refit) everywhere.
2. **Decoder weight decay.** `--wd` in `src/pair_gnn.py` goes to every parameter (GNN convolutions, node projection, decoder) through one optimiser group. The new `--dec_wd` (default off) sets the decay of everything after the GNN encoder (node projection, decoder, query-summary weights) and leaves `--wd` for the convolutions. Validation chose decoder wd 0 for hESC (0.7447 against 0.7441) and hHEP (0.7846 against 0.7835), wd 1e-3 for mHSC-E (0.8381 against 0.8367) and mESC (0.8335 against 0.8287).
3. **Uniform second TF summary** (hHEP and mHSC-E only). Validation skipped it in both: hHEP 0.7766 with against 0.7846 without, mHSC-E 0.8367 against 0.8381.
4. **DC-SBM rank blend.** Blend = (1 - w) x within-TF rank of the GNN + w x within-TF rank of the learned DC-SBM, w in {0, 0.25, 0.5, 0.75}. Validation scores of the GNN: the best-epoch model before the refit (`--save_sel_grid 1`, its TF x gene grid; trained without the validation rows, validation edges not in its graph). Validation scores of the DC-SBM: refit without the validation rows (`analysis/w16_dcval.py`, same hyper-parameters as the test-time fit, per seed). The grid check reproduces each run's recorded validation within-TF exactly. Chosen: hESC 0.5, hHEP 0.5, mHSC-E 0.25, mESC 0.5. Test scores: refit GNN and the DC-SBM fitted on all training rows (`analysis/out/stat/<cell>_dcsbm_learned_s4*.npz`). Pooled AUROC and AUPRC of a blend use a global rank average with the same w (a within-TF rank would drop each TF's base rate); within-TF numbers use the within-TF rank blend.
5. **Seed averaging.** Mean of the three seeds' logits (GNN) and DC-SBM logits, then the blend. Reported both ways. The recipe has saved per-pair scores (`f02_A_paper`, `w4_*_A_paper`, seeds 42-44), so it was ensembled the same way.

Held-out genes (hESC, 5 folds x seeds 42, 43): the chosen hESC GCN configuration (decoder wd 0 from the standard-split choice; the other value was run too and is shown as a control), blended with the fixed-cluster DC-SBM with Geneformer soft blocks (`analysis/out/stat/hESC_hog<f>_dcsbm_gf_soft_s*.npz`). w chosen on the held-out validation genes of the gene protocol over the 10 runs: 0 gives 0.6142, 0.25 gives 0.6399, 0.5 gives 0.6635, 0.75 gives 0.6610, so w = 0.5. Held-out TFs: GNN alone (an unseen TF has no DC-SBM row).

A rerun with `--save_sel_grid 1` reproduces the existing run: hESC seeds 42-44 against `w8_hESC_gcngf` give the same best epoch and within-TF to 0.0001. The mESC arms ran on ada against laptop grid runs and differ by -0.0012 on average (epochs 19/20/25 against 19/29/25).

## Results

Standard split, within-TF is the verdict metric. AUROC / AUPRC / within-TF, mean over seeds 42-44:

| Cell | scRegNet recipe | Best single model (chosen config alone) | Pipeline | Pipeline - recipe (p) | Pipeline - best single (p) |
|---|---|---|---|---|---|
| hESC | 0.8829 / 0.6147 / 0.7020 | 0.8892 / 0.6237 / 0.7360 | 0.8890 / 0.6336 / 0.7460 | +0.0440 (7e-4) | +0.0100 (0.037) |
| hHEP | 0.9022 / 0.8410 / 0.7873 | 0.9031 / 0.8450 / 0.7900 | 0.9063 / 0.8499 / 0.8075 | +0.0203 (0.012) | +0.0175 (0.013) |
| mHSC-E | 0.9196 / 0.9392 / 0.7210 | 0.9240 / 0.9432 / 0.7707 | 0.9271 / 0.9461 / 0.7700 | +0.0490 (6e-6) | -0.0007 (0.088) |
| mESC | 0.9254 / 0.8587 / 0.7658 | 0.9412 / 0.8864 / 0.8252 | 0.9445 / 0.8933 / 0.8403 | +0.0745 (4e-11) | +0.0151 (2e-9) |

Seed ensemble (one score from seeds 42-44), within-TF, recipe ensembled the same way:

| Cell | Recipe | Best single model | Pipeline | Pipeline - recipe (p) | Pipeline - best single (p) |
|---|---|---|---|---|---|
| hESC | 0.7034 | 0.7421 | 0.7478 | +0.0445 (0.001) | +0.0057 (0.082) |
| hHEP | 0.7870 | 0.7925 | 0.8097 | +0.0227 (0.009) | +0.0172 (0.019) |
| mHSC-E | 0.7208 | 0.7729 | 0.7713 | +0.0505 (3e-5) | -0.0016 (0.21) |
| mESC | 0.7659 | 0.8291 | 0.8428 | +0.0769 (2e-11) | +0.0137 (3e-8) |

The paired tests use per-TF AUROC averaged over seeds (per-seed table) or the per-TF AUROC of the ensemble score. For reference, the test-picked grid best of section 24 (optimistic, 5-seed choice made on test) has seed-mean within-TF 0.7502 / 0.7919 / 0.8076 / 0.8284 in hESC / hHEP / mHSC-E / mESC, so the pipeline sits above it on hHEP (+0.0156, p = 0.03) and mESC (+0.0119) and at or below it on hESC (-0.0042) and mHSC-E (-0.0376, p = 0.5).

hESC held-out (within-TF; AUROC / AUPRC / within-TF / within >= 8; 10 runs):

| Model | Held-out genes | Held-out TFs |
|---|---|---|
| Pipeline | 0.8598 / 0.5517 / 0.5783 / 0.5944 | 0.5576 / 0.1918 / 0.6850 / 0.6657 |
| GNN alone (same configuration) | 0.8380 / 0.5180 / 0.5576 / 0.5832 | same as the pipeline |
| DC-SBM Geneformer soft alone | 0.8573 / 0.5379 / 0.5753 / 0.5782 | none |
| scRegNet design, Adam-tuned (recipe baseline) | 0.8328 / 0.5275 / 0.5241 / 0.5591 | 0.6009 / 0.2303 / 0.7163 / 0.7045 |
| Hero (SAGE, no Geneformer) | 0.8467 / 0.5213 / 0.5461 / 0.5617 | 0.6304 / 0.2314 / 0.6855 / 0.6766 |

Held-out genes, pipeline minus: GNN alone +0.0189 (23/34 TFs, p = 0.004), DC-SBM alone -0.0070 (p = 0.37), recipe baseline +0.0620 (p = 0.002), hero +0.0161 (p = 0.11). As a 2-seed ensemble per fold: pipeline 0.5802, recipe baseline 0.5244 (+0.0642, p = 0.002). Held-out TFs, pipeline minus recipe baseline -0.0320 (6/34, p = 1e-4), minus hero -0.0014 (14/34, p = 0.40).

## Which pieces added something

Within-TF change inside the final pipeline, seed mean (seed ensemble in brackets where it differs in sign or size); paired per-TF Wilcoxon p:

| Piece | hESC | hHEP | mHSC-E | mESC |
|---|---|---|---|---|
| DC-SBM blend (chosen w against 0) | +0.0094 (p = 0.04) [+0.0058] | +0.0200 (p = 0.008) [+0.0174] | -0.0007 (p = 0.09) [-0.0016] | +0.0151 (p = 2e-9) [+0.0137] |
| Decoder wd 0 minus wd 1e-3 | -0.0007 (0.30) | -0.0003 (0.82) | -0.0008 (0.09) | -0.0019 (0.07) |
| Second summary (with minus without) | not run | -0.0069 (0.09) | +0.0002 (0.57) | not run |
| Seed ensemble minus per-seed mean | +0.0018 (0.008) | +0.0021 (8e-5) | +0.0013 (0.05) | +0.0025 (4e-11) |
| Same, for the recipe | +0.0014 | -0.0003 | -0.0002 | +0.0001 |
| Validation-chosen config minus the test-picked config, each with its own blend | -0.0047 (0.35) | +0.0036 (0.56) | -0.0167 (0.14) | -0.0022 (0.79) |

The last row comes from the sensitivity runs (`w16_<cell>_npbase|gcnbase|sagebase`: the section 24 test-picked configuration, wd 1e-3, no second summary, w chosen on validation). With the test-picked configurations the pipeline still beats the recipe in all 4 cells (+0.049, +0.017, +0.066, +0.077), so the verdict does not turn on the validation choice. The sensitivity arms for mHSC-E and mESC ran on ada and those for hESC and hHEP on the laptop, so those differences are not same-machine.

- **DC-SBM blend: the only piece with a consistent gain**, +0.009 to +0.020 on three cells and about zero on mHSC-E. Measured: the blend helps where the DC-SBM alone is as strong as the GNN (hHEP 0.8006 against 0.7875, mESC 0.8332 against 0.8252, hESC 0.7295 against 0.7366) and not where it is clearly weaker (mHSC-E 0.7531 against 0.7707). The two models agree only moderately within a TF (mean within-TF Spearman 0.76 hESC, 0.82 hHEP, 0.82 mHSC-E, 0.81 mESC). Guess, untested: the blend adds a popularity-and-co-targeting signal that the GNN under-weights, in line with section 31. Validation chose w = 0.25 on mHSC-E because the validation gain there was +0.0026, within noise.
- **Decoder wd 0 does nothing here.** All four effects are within 0.002 and none is significant, and on mESC wd 1e-3 is slightly better. The LightGCN gain at wd 0 (section 35) was seen on a model with free node embeddings; the GNNs here show no such gain. Guess, untested: the decoder decay only hurts models whose node vectors carry no input features.
- **The second summary does nothing or hurts**, as sections 25 and 32 suggested for strong baselines. Validation correctly skipped it.
- **Seed ensembling adds 0.001 to 0.003** to the pipeline. The recipe gains at most 0.0014 from it, so the ensemble comparison is slightly more favourable to the pipeline than the per-seed one, and the verdict is the same under both.
- **Validation choice of the configuration is noisy.** The validation slice is about a tenth of the training rows, and its within-TF differences between top configurations (0.0004 to 0.003) are smaller than the test noise, so the choice costs up to 0.037 (mHSC-E, compared with the test-picked grid best) without being a mistake of method.
- **Why held-out TFs fail.** Measured: every GNN in the project, the pipeline included, sits near or below popularity (0.7244) on unseen TFs, and the recipe baseline, which has no TF-specific interaction term, is closest to popularity (0.7163). The pipeline's Hadamard decoder and Geneformer give gains only for TFs with training targets. The DC-SBM cannot help because an unseen TF has no row. Guess: the interaction term overfits TF identity, which does not transfer to new TFs.

## Dropped or uncertain

- The published SGD recipe has no held-out runs, so the held-out recipe baseline is the Adam-tuned scRegNet design; the hero is listed too.
- The DC-SBM validation refits ran on the laptop CPU (seconds each), not on ada: ada's four GPU slots were all in use by one-GPU jobs. `analysis/w16_dcval.py` also runs through `analysis/ada.sh run FILE P analysis.w16_dcval`.
- hESC and hHEP configurations chosen by validation are near ties with the neural PANDA config, and the decoder-wd and second-summary choices rest on validation gaps of 0.0006 to 0.008.
- Pooled AUROC and AUPRC of the pipeline come from a global rank blend, within-TF numbers from a within-TF rank blend; the two are not one score.
- Decoder wd 0 was also run on held-out TFs and genes as a control (`w16_ho_*_dwd0_*`): within-TF 0.6816 (TFs) and 0.5570 (genes) against 0.6850 and 0.5576 for the chosen wd.
- Held-out TF runs were done without `--save_sel_grid`, since no blend applies.

Scripts: `analysis/w16_choose.py`, `analysis/w16_dcval.py`, `analysis/w16_pipeline.py`. Outputs: `analysis/out/w16_choose.txt`, `analysis/out/w16_pipeline.txt`. Jobs: `analysis/jobs/w16_*.txt`. Runs: `analysis/out/pair/w16_*`. New flag: `--dec_wd` in `src/pair_gnn.py` (default off); `analysis/w12_stat.py` `load()` now passes `emb_feat` and `feat` so it runs with the current `prepare`.
