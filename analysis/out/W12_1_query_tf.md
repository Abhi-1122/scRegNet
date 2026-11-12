# W12 idea 1: query-aware TF summary

hESC unless stated, hero settings (1-layer SAGE, `--split batch`, epoch chosen on validation AUROC, refit), 3 seeds (42-44) in random-split tables, 5 folds x seeds 42, 43 in held-out tables. Paired tests: per-TF test AUROC averaged over the runs, Wilcoxon over TFs. Reproduce: `python3.12 analysis/w12_1_report.py` (writes `analysis/out/w12_1_query_tf.txt`). Runs: `analysis/jobs/w12_1_screen.txt`, `w12_conf_a.txt`, `w12_ho_a.txt`, `ada_sent/w12_1_*.txt.orig`.

## 1. Where the idea comes from

Item-based collaborative filtering scores a candidate item by its similarity to the items a user already liked, and DIN (Zhou et al., KDD 2018) weights the user's history by the candidate item (target attention), so that a user with several interests is not blurred into one average vector. In the hero a TF is the mean of its known targets' features. A TF that regulates several gene programmes gets one blurred vector, while the question asked at test time is "does this gene resemble any of this TF's known targets?".

## 2. Method in short

`--decoder qatt` (hero `had` decoder plus a query block). For pair (t,g), N(t) is the set of TF->target edges of the current message-passing graph with t as source, g itself removed. Under `--split batch` the batch's own edges (both orientations of the same undirected edge) are already out of the graph, so a pair cannot attend to its own edge. Single dense masked attention over all genes: alpha_j = softmax_j(<Wq u_g, Wk u_j> / sqrt(dk)), context c = Wc sum_j alpha_j Wv u_j. Decoder input: [u_t, u_g, u_t * u_g, c, c * u_g, masked max cosine between u_g and u_j over N(t)]. At evaluation N(t) is every training target of t. If N(t) is empty (unseen TF), c = 0 and the max cosine is 0.

Knobs (flags `--qatt_heads`, `--qatt_src {emb,raw}`, `--qatt_topk`, `--qatt_dim`, `--qatt_uniform`): heads {1, 4}; vectors compared = post-GNN embeddings (`emb`) or raw expression (`raw`); all targets (k = 0) or the 5 best-matching (k = 5). `--qatt_uniform 1` replaces alpha by 1/|N(t)| (control: a plain mean, no query-awareness). Knob choice: seed-42 sweep, picked by validation (`h1_emb_k0`: best validation within-TF, 0.7219 against at most 0.7187 for the other seven rows, with a validation AUROC 0.0002 below the best). `--tf_drop 0.2` (existing flag) was added as the follow-up that teaches the model about TFs without known targets.

## 3. Results

Stage 1 (random split, seed-42 sweep, within-TF; hero 0.7499): emb: h1 k0 0.7556, h4 k0 0.7558, h1 k5 0.7448, h4 k5 0.7482. raw: h1 k0 0.7339, h4 k0 0.7173, h1 k5 0.6880, h4 k5 0.7195. Raw expression vectors are clearly worse than embeddings.

Confirmation (3 seeds, hero from `w3_hero`):

| model | AUROC | AUPRC | within-TF | diff within vs hero | p |
|---|---|---|---|---|---|
| hero | 0.8885 | 0.6254 | 0.7490 | | |
| qatt h1 emb k0 | 0.8893 | 0.6256 | 0.7506 | +0.0017 | 0.58 |
| qatt h4 emb k0 | 0.8891 | 0.6253 | 0.7493 | +0.0003 | 0.52 |
| uniform attention (mean) | 0.8890 | 0.6254 | 0.7492 | +0.0002 | 0.45 |
| qatt + tf_drop 0.2 | 0.8892 | 0.6278 | 0.7440 | -0.0049 | 0.92 |

AUROC is above the hero in 3 of 3 same-seed pairs for h1 (+0.0009), within-TF in 2 of 3. Not a result at this size.

Stage 2, held-out genes (hero = `w5_ho_gene*_hero`):

| model | AUROC | within-TF | diff within | p |
|---|---|---|---|---|
| hero | 0.8467 | 0.5461 | | |
| qatt h1 emb k0 | 0.8470 | 0.5580 | +0.0182 | 0.003 |
| uniform attention | 0.8473 | 0.5572 | +0.0175 | 0.005 |
| qatt + tf_drop 0.2 | 0.8491 | 0.5588 | +0.0154 (+0.0236 against hero + tf_drop 0.3) | 0.002 |

Stage 2, held-out TFs: hero 0.6855 within-TF; qatt 0.6333 (-0.0531, p < 0.001); qatt + tf_drop 0.2 0.6613 (-0.0244, p = 0.018) and equal to the matched hero + tf_drop 0.3 control (0.6598; +0.0018, p = 0.99).

Stage 3 (3 seeds; baseline `w4_<cell>_hero`, same configuration; Geneformer baseline `w3_herogf`):

| cell | hero within-TF | qatt | diff (p) | uniform attention | diff (p) |
|---|---|---|---|---|---|
| hHEP | 0.7587 | 0.7700 | +0.0114 (0.003) | 0.7694 | +0.0107 (0.014) |
| mHSC-E | 0.7575 | 0.7654 | +0.0080 (0.033) | 0.7689 | +0.0114 (0.002) |
| mESC | 0.8241 | 0.8093 | -0.0148 (0.12) | 0.8245 | +0.0003 (0.12) |
| hESC + Geneformer | 0.7495 | 0.7537 | +0.0042 (0.074) | | |

## 4. Why

Measured:
- The gain is not query-awareness. Uniform attention, a plain mean over the TF's known targets, gives the same held-out-gene gain (+0.0175 against +0.0182) and the same gains on hHEP and mHSC-E, and avoids the mESC loss (+0.0003 against -0.0148). What carries the gain is the added similarity term between the gene and the TF's known-target set (c * u_g and the max-cosine feature), a prototype/kNN score that is not a function of the TF embedding alone because it is computed against the gene.
- The attention is almost flat: normalised attention entropy 0.97 on positive and 0.85 on negative test pairs (1.0 = uniform).
- Where the held-out-gene gain sits: TFs with few known targets (Spearman between per-TF gain and the TF's number of targets -0.46, p = 0.006; TFs below the median of 38 targets gain +0.035, those above +0.001). TFs with more multi-modal target sets (low mean pairwise cosine) gain more in the expected direction in the random split and the held-out-gene runs (Spearman -0.29, p = 0.13 and -0.29, p = 0.13), but the evidence is weak. The number of target clusters has no relation (+0.07).
- Held-out TFs: the loss is -0.053. Training with hidden TFs (`--tf_drop`) cuts it to -0.024, and the result is then equal to the same drop applied to the hero (+0.002). Guess: with no known targets the query block is an all-zero input that never occurred in training, and the decoder reads it as "unlikely"; hiding TFs during training exposes the model to that case.
- Raw expression similarity is not informative: the maximum cosine of a gene's expression profile to the TF's known targets ranks genes below chance inside a TF (within-TF AUROC 0.46; mean of the top three cosines 0.47), and the learned embeddings' maximum cosine reaches a pooled (not within-TF) AUROC of 0.75. Gene popularity alone has within-TF AUROC 0.707. Raw inputs to the attention (`raw`) are correspondingly bad.

Guess, not tested: in hESC the hero's one SAGE layer already contains the information that the prototype term brings for genes seen in training; for unseen genes (held-out gene protocol) and for the cells with more targets per TF (hHEP, mHSC-E) the explicit comparison of the gene with the target set adds information that the TF mean cannot express. mESC has hubs with up to 45 TFs per gene, which probably lets the learned attention fit popularity patterns and overfit.

## 5. Verdict

Neutral on the benchmark split in hESC, positive for unseen genes and for two of the three other cells, but the useful part is the plain mean-similarity feature, not the attention. Next: make it an always-on, parameter-light feature (cosine of u_g to the mean of N(t) and max over N(t), no attention), with `--tf_drop` for held-out TFs; check on held-out genes and the cells with larger TFs. Not run: the Geneformer comparison for the uniform variant, clustering-based multi-modality beyond the three summary numbers above.
