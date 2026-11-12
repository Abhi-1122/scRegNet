# W12 idea 2: NBFNet (path-based link prediction)

Run 2026-11-12. Code `src/nbf.py`, report `python3.12 analysis/w12_nbf_report.py` (output `analysis/out/w12_nbf_report.txt`), runs `analysis/out/pair/w12_nbf_*` and `w12_ho_*_nbf_*`, job files `analysis/jobs/ada_inbox/w12_nbf_*.txt`. hESC, TFs+500, seeds 42-44, hero = `w3_hero`.

## 1. Where the idea comes from
Neural Bellman-Ford Networks (Zhu et al., NeurIPS 2021) score a query (source, relation, target) by running a generalised Bellman-Ford iteration from the source node: the source carries a learned query vector, every other node starts at zero, and each layer sends relation-specific messages along the edges and aggregates them at the receiving node. The state at the target summarises all short paths from the source to it. The method is state of the art on knowledge-graph completion.
Motivation here: popularity (gene in-degree) dominates the scores, and in an earlier week the stacker ranked 3-path counts (TF -> target <- regulator -> gene) among its strongest features. NBFNet is per-TF by construction (one run per source TF) and learns a soft, weighted generalisation of those path counts.

## 2. Method in short
- Graph: training TF->target edges with two relations (TF->target and the reverse target->TF). Optional extra relations, co-expression kNN (`C`) and STRING (`P`), reused from `pair_gnn.coexp_graph` and `ppi_graph`.
- For each source TF t in the batch (all of them at once as a tensor TFs x genes x d): boundary h0 = q at t, 0 elsewhere (learned q); L layers of `h_v <- h_v + ReLU(LayerNorm(W [h_v, AGG_{u->v}(h_u * w_r) + boundary_v]))` with a DistMult message, aggregation sum, mean or PNA (mean, max, min, std with identity, amplification and attenuation scalers), short-cut and layer norm as in the paper.
- Score(t, g) = MLP([h_t^1(g), ..., h_t^L(g), proj(x_g), proj(x_t) * proj(x_g)]) where x is the standardised expression and proj a linear map. `--feat final` puts the gene projection only into the final MLP, `--feat layer` also into every layer, `--feat none` is pure paths. `--tf_inter 1` adds the TF-gene expression product (the term the hero's Hadamard decoder has). `--layers 0` removes the paths (features only).
- Training: Adam lr 1e-3, wd 1e-4, dropout 0.2, batch 512 (a batch costs about the same whatever its size, so larger batches are faster), epoch chosen on validation AUROC (evaluated every epoch, patience 10), refit on all training rows for the chosen epoch count. Same validation slice (`prepare`, split_seed 0) as the hero.
- Leak control: as `--split batch`, the batch's own positive edges and their reverses are removed from the graph in every training step; validation and test use the graph of all training-positive edges (the same graph the hero uses). The boundary condition starts at t, and a trivial path t -> g through the removed edge does not exist. The `--deg` diagnostic reads degrees from the current batch graph, so the own edge is not counted.
- Knobs tried (seed 42 only, chosen on validation AUROC, never on test): layers {3, 4, 6}, aggregation {sum, mean, pna}, features {final, layer, none}, TF query {shared, + TF expression}, relations {R, RCP}, lr {1e-3, 3e-4}, with and without `--tf_inter`. Chosen: 6 layers, PNA, `--feat final`, `--tf_inter 1`, relations R (val AUROC 0.8852, the highest of the tf_inter variants; the 3-layer, 4-layer and sum variants were 0.883 to 0.884).

Sweep result (seed 42, validation AUROC): without `tf_inter` 0.862 to 0.870 for every layer count and aggregation (layers 3 / 4 / 6 with PNA: 0.870 / 0.869 / 0.868; sum 0.863, mean 0.863; `--feat layer` 0.849); with `tf_inter` 0.878 to 0.885 (hero on the same slice: 0.889).

## 3. Results
Stage 1, hESC, random split, 3 seeds (42-44), refit on all training rows (mean ± std over seeds). Paired per-TF tests: per-TF test AUROC averaged over the 3 seeds, Wilcoxon over the 33 test TFs.

| model | AUROC | AUPRC | within-TF | within-TF diff vs hero (TFs better, p) |
|---|---|---|---|---|
| hero | 0.8885 ± 0.0014 | 0.6254 ± 0.0015 | 0.7490 ± 0.0018 | |
| NBFNet L6 PNA + tf_inter (chosen) | 0.8848 ± 0.0033 | 0.6074 ± 0.0071 | 0.6971 ± 0.0117 | -0.0519 (9/33, p=0.0013) |
| NBFNet pure paths (no features) | 0.8729 ± 0.0022 | 0.6139 ± 0.0033 | 0.6576 ± 0.0060 | -0.0914 (8/33, p=6e-5) |
| NBFNet no paths (0 layers, features only) | 0.8670 ± 0.0017 | 0.5830 ± 0.0046 | 0.6839 ± 0.0034 | -0.0651 (5/33, p=5e-5) |
| chosen + gene degree (diagnostic) | 0.8863 ± 0.0013 | 0.6117 ± 0.0045 | 0.7032 ± 0.0085 | -0.0458 (10/33, p=0.021) |
| no paths, hidden 256 (diagnostic) | 0.8626 ± 0.0048 | 0.5737 ± 0.0082 | 0.6760 ± 0.0069 | -0.0730 (3/33, p=3e-6) |
| L3 sum, hidden 128 + tf_inter (diagnostic) | 0.8802 ± 0.0020 | 0.6057 ± 0.0049 | 0.6972 ± 0.0102 | -0.0518 (8/33, p=0.0017) |

The chosen model is 0.004 below the hero in AUROC (better than the hero in 1 of 3 seeds) and 0.052 below in within-TF AUROC (p=0.0013). Stage 1 is a regression, so stage 3 (other cell types, Geneformer) was not run.

Stage 2 (run because the idea claims to handle new nodes; 5 folds x seeds 42, 43; mean over folds, hero = `w5_ho_*_hero`):

| protocol | model | AUROC | AUPRC | within-TF | within-TF >= 8 positives | diff vs hero (per fold-TF, p) |
|---|---|---|---|---|---|---|
| held-out TFs | hero | 0.6304 | 0.2314 | 0.6855 | 0.6766 | |
| held-out TFs | NBFNet | 0.5975 | 0.2159 | 0.6016 | 0.5905 | -0.0840 (4/34, p=8e-7) |
| held-out genes | hero | 0.8467 | 0.5213 | 0.5461 | 0.5617 | |
| held-out genes | NBFNet | 0.7370 | 0.3594 | 0.5433 | 0.5610 | -0.0028 (74/155, p=0.62) |

On held-out TFs the source has no training edges, so no path leaves it: every layer sees only the boundary at t and zeros elsewhere, the path part of the score is the same for all genes of that TF, and what remains is the expression term (gene projection and the TF-gene product). On held-out genes the new gene has no edges, so it is reached by no path either; the within-TF ranking matches the hero but the global AUROC is 0.11 lower (a guess: the scores of unreachable genes are not on the same scale as those of reachable ones in the training graph, which hurts pooled ranking but not within-TF ranking).

## 4. Why (it regressed)
Measured:
- Features carry most of the signal: the model without any path layers already reaches within-TF 0.684, the pure-path model 0.658, the combination 0.697. Adding paths to features is worth +0.013 within-TF (p=0.035 paired, CI [-0.036, +0.052]), small and not consistent.
- The TF-gene expression product is what lifts the sweep (3-layer PNA validation AUROC 0.870 to 0.883, seed 42). Paths alone, even with 6 layers, add little over a linear feature model.
- Split by training-graph 3-paths (80% of test pairs have at least one): the hero's within-TF AUROC is 0.815 on pairs with none and 0.691 on pairs with some; NBFNet is 0.724 and 0.648. The gap is largest where no path exists (-0.09), so the loss is not a failure to use paths. The raw 3-path count alone scores within-TF 0.670, so a path-only model was never going to match 0.749.
- Popularity: the within-TF Spearman between score and gene in-degree is 0.72 for the hero, 0.62 for NBFNet, 0.49 for pure paths. NBFNet tracks popularity less well. Giving it the degree explicitly raised within-TF only to 0.703 (+0.006), so missing popularity is not the main gap. More capacity does not close it either (hidden 256 features only: 0.676; sum aggregation with hidden 128: 0.697).
- Training stops at epoch 3 to 7: the model overfits the few thousand training edges quickly, and lr 3e-4 did not change that (validation AUROC 0.8827).

Guess (not tested): the hero's 1-layer SAGE averages the expression of a node's neighbours, so a gene's embedding carries the expression profile of its known regulators and a TF's embedding that of its known targets. NBFNet's messages carry learned states that start at zero, so the neighbour expression does not reach them except through `--feat layer`, which did worse (validation 0.876 vs 0.883). The hero keeps a full 758-dim linear read of the expression plus neighbour averages; NBFNet projects to 64 dimensions without neighbour averaging.

## 5. Verdict
No gain: within-TF 0.697 vs 0.749 (p=0.0013), AUROC -0.004, and clearly worse on held-out TFs (-0.084 within-TF), where the idea has nothing to start from. Paths add about +0.013 within-TF over the same model without paths, so path information that is not already in the neighbour-averaged features is small on this graph. If tuned further, the one untested change is feeding neighbour-averaged expression (SAGE-style) into the boundary or the layer update; the evidence here does not make it promising.
