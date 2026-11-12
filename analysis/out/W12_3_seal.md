# W12 idea 3: SEAL (subgraph classification)

Run 2026-11-12. Code `src/seal.py`, report `python3.12 analysis/w12_seal_report.py` (output `analysis/out/w12_seal_report.txt`), runs `analysis/out/pair/w12_seal_*` and `w12_ho_*_seal_*`, job files `analysis/jobs/ada_inbox/w12_seal_*.txt`. hESC, TFs+500, seeds 42-44, hero = `w3_hero`.

## 1. Where the idea comes from
SEAL (Zhang and Chen, NeurIPS 2018, "Link prediction based on graph neural networks") turns link prediction into graph classification: for every candidate pair the h-hop enclosing subgraph is extracted with the target edge removed, nodes are marked with a double-radius node label (DRNL, a function of the node's distances to the two endpoints), and a GNN with a graph-level pooling classifies the subgraph. The labelling trick lets the model see the joint structure of the pair (common neighbours, short paths), which node-wise embeddings cannot, and the pair's own edge cannot leak because it is deleted from every subgraph by design.
Motivation here: the own-edge problem of `--split none` (a training pair is scored through its own edge) and the strength of 3-path counts in the stacker. Both are handled by construction.

## 2. Method in short
- Subgraph of pair (t, g): t, g, their neighbours in the undirected training-positive graph (at most `--cap` per node, a fixed random subset, 30 by default) and, for `--hops 2`, the neighbours of those. Induced edges, then the edge t-g (both directions) is deleted for every pair of every split (train, validation, test alike), so the train/test distributions match. The subgraph is extracted once per fit (graph fixed across batches, since each pair only removes its own edge); about 28 nodes and 116 directed edges per pair, 17 s for the training rows on 3 CPU workers.
- Node input: DRNL label (embedded), log in-degree and out-degree in the training graph (for t and g the own edge is subtracted, so a positive and a negative pair see the same distribution), PCA(32) of the standardised expression. Three SAGE (or GCN) layers of width 64, dropout 0.2, then pooling and an MLP: `mm` mean + max; `tt` mean + max + the representations of t, g and their product; `sort` SortPooling (k=20).
- Training: Adam lr 1e-3, wd 1e-4, batch 256, epoch chosen on validation AUROC every epoch (patience 10), refit on all training rows (the subgraphs are re-extracted from the graph that now contains the validation positives, the same graph the hero uses at test time). All training negatives are used (no subsample was needed; a run takes about 2 minutes on a 2080 Ti).
- Leak control: the pair's own edge and its reverse are absent from its subgraph, from DRNL distances and from the degree features (checked on 300 training-positive pairs: none keeps a t-g edge in its subgraph and the gene in-degree feature is exactly 1 lower than the graph in-degree); validation and test labels never enter the graph. The held-out-TF and held-out-gene protocols use the same `prepare` as the hero.
- Knobs tried (seed 42, validation AUROC): hops/cap {1/30, 2/10, 2/20}, pooling {mm, tt, sort}, convolution {SAGE, GCN}. Validation AUROC: 1-hop tt 0.8875, sort 0.8876, mm 0.8818, GCN 0.8864, 2-hop cap 10 0.8868, 2-hop cap 20 0.8869; all inside 0.006. 1-hop with `tt` was chosen (tied with `sort` on validation AUROC, higher validation within-TF 0.717 vs 0.686, and cheaper than 2 hops, which add nothing).

## 3. Results
Stage 1, hESC, random split, seeds 42-44, refit (mean ± std). Paired per-TF tests: per-TF AUROC averaged over the 3 seeds, Wilcoxon over 33 TFs.

| model | AUROC | AUPRC | within-TF | within-TF diff vs hero (TFs better, p) |
|---|---|---|---|---|
| hero | 0.8885 ± 0.0014 | 0.6254 ± 0.0015 | 0.7490 ± 0.0018 | |
| SEAL 1-hop cap 30, tt (chosen) | 0.8864 ± 0.0011 | 0.6195 ± 0.0036 | 0.7075 ± 0.0081 | -0.0415 (11/33, p=0.027) |
| without DRNL labels | 0.8833 ± 0.0004 | 0.6077 ± 0.0008 | 0.6896 ± 0.0012 | -0.0593 (10/33, p=0.0074) |
| without expression (structure only) | 0.8816 ± 0.0031 | 0.6023 ± 0.0169 | 0.6990 ± 0.0118 | -0.0499 (12/33, p=0.0058) |
| without degree features | 0.8857 ± 0.0008 | 0.6190 ± 0.0024 | 0.7077 ± 0.0046 | -0.0412 (11/33, p=0.021) |
| expression PCA 128 (diagnostic) | 0.8850 ± 0.0005 | 0.6190 ± 0.0020 | 0.7143 ± 0.0048 | -0.0347 (12/33, p=0.059) |
| expression PCA 256 (diagnostic) | 0.8839 ± 0.0011 | 0.6172 ± 0.0025 | 0.7142 ± 0.0015 | -0.0348 (12/33, p=0.10) |

Same-seed AUROC difference of the chosen SEAL vs the hero: -0.0021 (better in 1 of 3 seeds); AUPRC -0.0059 (0 of 3). Stage 1 regressed on within-TF AUROC, so stage 3 (other cells, Geneformer) was not run.

Stage 2 (5 folds x seeds 42, 43; hero = `w5_ho_*_hero`):

| protocol | model | AUROC | AUPRC | within-TF | within-TF >= 8 positives | diff vs hero (per fold-TF, p) |
|---|---|---|---|---|---|---|
| held-out TFs | hero | 0.6304 | 0.2314 | 0.6855 | 0.6766 | |
| held-out TFs | SEAL | 0.6450 | 0.2338 | 0.6552 | 0.6535 | -0.0310 (13/34, p=0.12) |
| held-out genes | hero | 0.8467 | 0.5213 | 0.5461 | 0.5617 | |
| held-out genes | SEAL | 0.8226 | 0.4517 | 0.5064 | 0.5231 | -0.0399 (46/155, p=2e-6) |

For a held-out TF the subgraph of (t, g) contains no neighbours of t, so the model sees g's regulators and the expression of t and g; the global AUROC rises slightly above the hero (0.645 vs 0.630) while the within-TF ranking is lower.

## 4. Why (it regressed)
Measured:
- Structure helps a little, not enough. DRNL labels are worth +0.018 within-TF (24/33 TFs, p=0.014 paired, chosen vs no DRNL) and expression +0.009 (22/33, p=0.080) on top of structure. The structure-only SEAL already reaches 0.699 within-TF and AUROC 0.882, close to the full model, so most of what it knows is the training graph around the pair.
- Popularity: the within-TF Spearman between score and gene in-degree is 0.85 for SEAL (0.84 to 0.87 for every ablation) against 0.72 for the hero. SEAL ranks genes within a TF mostly by how connected they are; the hero mixes in more expression-specific information.
- Where the hero wins: split by training-graph 3-paths (80% of test pairs have some), the hero's within-TF AUROC is 0.815 on pairs without any and 0.691 with some; SEAL is 0.702 and 0.641. The gap (-0.11 and -0.05) is largest on poorly connected pairs, where subgraphs are small and DRNL has little to say. Structure-only SEAL drops to 0.578 on the no-path pairs (expression is what carries them), so these pairs depend on expression read at full resolution.
- Expression resolution is not the cause: PCA 128 or 256 instead of 32 gave within-TF 0.714 (+0.007, inside noise) and the gap to the hero stayed at 0.035.
- Neighbour cap and hops: 2 hops (cap 10 or 20) did not change validation AUROC (0.8868, 0.8869 vs 0.8875), test within-TF was lower (0.676, 0.694 vs 0.696 on seed 42).

Guess (not tested): the hero is a 1-layer SAGE over the whole graph trained with the batch's edges removed, so its gene embedding carries the mean expression of all of the gene's known regulators and its TF embedding the mean expression of all targets, and a bilinear head compares them. SEAL sees at most 30 sampled neighbours with a capped view, and its pairwise structure features are dominated by popularity, which the test split (random pairs from the same TFs) rewards in AUROC but not in the within-TF ranking.

## 5. Verdict
No gain: within-TF 0.7075 vs 0.7490 (p=0.027), AUROC -0.002, AUPRC -0.006; worse on held-out genes (-0.040 within-TF, p=2e-6), not distinguishable on held-out TFs (-0.031, p=0.12; higher pooled AUROC). The pair-structure signal (DRNL) is real but small (+0.018 within-TF). Not promising to tune: neither hops, pooling, convolution nor expression resolution moved the result.
