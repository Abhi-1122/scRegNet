# W12 idea 6: hub control

hESC unless stated; hero settings (1-layer SAGE, `had`, `--split batch`, no Geneformer), epoch chosen on validation AUROC, refit. 3 seeds (42-44) with paired per-TF tests against `w3_hero` (within-TF 0.7490 +- 0.0018). Reproduce: `python3.12 analysis/w12_6_report.py` (writes `analysis/out/w12_6_hubs.txt`). Runs: `analysis/jobs/w12_6_screen.txt`, `w12_conf_a.txt`, `w12_ho_a.txt`, `ada_sent/w12_6_cells.txt.orig`.

## 1. Where the idea comes from

Over-squashing (Topping et al., ICLR 2022) shows that information from many nodes is squeezed through a few high-degree bottlenecks, and SDRF rewires edges around negatively curved bottlenecks. Recommender systems have a parallel problem: popularity bias, handled by weighting interactions by inverse item frequency (IDF) or by down-sampling popular items. In the project the dominant finding is that gene popularity (number of TFs that target a gene) drives the predictions, especially for unseen TFs, and that hub targets appear in every TF's neighbourhood mean, which makes TF embeddings alike. Hub control was meant to push the model from popularity toward TF-specific signal.

## 2. Method in short

Flags (default off): `--hub_cap K`, `--hub_eval {cap,all}`, `--idf_beta b`, `--deg_feat 1`.
- `--hub_cap K`: each gene keeps at most K message-passing edges (both directions of an undirected edge drop together), redrawn at random every training batch after the batch's own edges are removed. Evaluation uses one fixed-seed draw (`--hub_eval cap`, default) or all edges (`all`).
- `--idf_beta b` (SAGE only, own `WSage` layer): the edge (t,g) gets weight deg(g)^-b, deg counted in the current message-passing graph, so a batch's own edges do not enter the degree; a TF's embedding is the weighted mean of its targets. Checked equal to `SAGEConv` for unit weights.
- `--deg_feat 1`: log(1 + current degree) as an extra node feature (degree taken after the batch edges are removed).
- Not done: curvature-based rewiring (SDRF-style) and the "hub edges removed + degree feature" variant beyond `hub_cap10_degfeat`; time was spent on the cells with real hubs instead.
- Leak: every degree and every cap is computed on the message-passing graph after the batch's own edges are removed, so a training pair never sees its own edge.

Important about the data: hubs in hESC are mild. Largest in-degree 14, 8 genes with more than 10 TFs, 3030 positive training edges in all. Cap 10 removes 16 edges, cap 20 removes none (cap 20 and cap 40 are therefore null controls: same graph, different random-number stream). Elsewhere hubs are heavy: hHEP max 19 (220 genes above 10), mESC max 45 (902 genes above 10, cap 10 removes 10383 of 19805 edges), mHSC-E max 20 (431 genes above 10).

## 3. Results

hESC, 3 seeds (within-TF; diff and p against `w3_hero`):

| variant | AUROC | within-TF | diff | p |
|---|---|---|---|---|
| hero | 0.8885 | 0.7490 | | |
| cap 20 (null control, no edge removed) | 0.8874 | 0.7456 | -0.0034 | 0.37 |
| IDF beta 0.5 | 0.8875 | 0.7453 | -0.0037 | 0.22 |
| IDF beta 1 | 0.8877 | 0.7446 | -0.0044 | 0.14 |
| IDF 0.5 + cap 20 | 0.8874 | 0.7449 | -0.0040 | 0.81 |

Seed-42 screen (within-TF; hero 0.7499): cap 3 0.7315, cap 5 0.7317 (p = 0.005), cap 10 0.7385, cap 20 0.7390, cap 40 0.7390, cap 10 with all edges at evaluation 0.7387, degree feature 0.7485, cap 10 + degree feature 0.7501, IDF 0.5 0.7526, IDF 1 0.7511, IDF 0.5 + cap 20 0.7537. The null controls (cap 20, 40) sit 0.011 below the hero at this seed, so at seed 42 the cap results cannot be separated from random-stream noise except for caps 3 and 5 (0.007 lower again); over 3 seeds the null control is -0.0034.

Stage 2 (IDF 0.5; 5 folds x seeds 42, 43): held-out genes 0.5488 vs 0.5461 (+0.0013, p = 0.80); held-out TFs 0.6787 vs 0.6855 (-0.0065, p = 0.26).

Stage 3, cells with heavy hubs (3 seeds, baseline `w4_<cell>_hero`; within-TF diff vs hero, p):

| cell | IDF 0.5 | IDF 1 | cap 10 | cap 20 |
|---|---|---|---|---|
| hHEP (hero 0.7587) | +0.0012 (0.82) | +0.0040 (0.89) | -0.0006 (0.44) | +0.0000 (0.86) |
| mESC (hero 0.8241) | +0.0007 (0.29) | +0.0012 (0.41) | -0.0092 (0.006) | -0.0022 (0.32) |
| mHSC-E (hero 0.7575) | -0.0049 (0.12) | -0.0061 (0.09) | +0.0015 (0.59) | +0.0023 (0.29) |

## 4. Why

Measured:
- Score/popularity: Spearman between the score and a gene's training in-degree (all genes, mean over TFs) is 0.731 for the hero and 0.720 to 0.734 for every hESC variant. IDF changes it by at most 0.005 in every cell. Capping lowers it somewhat where hubs are heavy (mESC 0.724 -> 0.706 at cap 20, 0.687 at cap 10; hHEP 0.835 -> 0.823; mHSC-E 0.651 -> 0.636), and within-TF falls with it in mESC (cap 10: -0.009). Less popularity in the scores did not make the TF-specific ranking better; popularity is a real signal inside a TF (a ranking by popularity alone has within-TF AUROC 0.707 in hESC).
- Hub share of the TF's targets (fraction of targets in the top quartile of in-degree) does not explain the per-TF gain: Spearman +0.28 (IDF 0.5, p = 0.13), +0.20 (IDF 1, p = 0.28), +0.02 (cap 20) over 31 TFs. The sign is the expected one for IDF, the evidence is weak.
- No gain in the held-out-TF protocol (-0.0065), where popularity is strongest.

Guess, not tested: popularity enters through the gene-side terms of the decoder and through the number of messages a gene receives, not through the neighbourhood mean that IDF and caps modify, so these knobs act on the part of the model that is not the cause. The hubs in the benchmark are also mild compared with the graphs for which over-squashing was designed.

## 5. Verdict

Neutral to slightly negative in four cells and both held-out protocols; the only significant effect is a loss (mESC cap 10). Not worth tuning further. Curvature rewiring was not tried because the diagnostic (score/popularity correlation barely moves under IDF) indicates that the neighbourhood-mean pathway is not the route by which popularity is used.
