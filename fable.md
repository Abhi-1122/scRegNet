# fable.md: where this project stands and what to try next

Written 17 Nov 2026, after sections 0-37 of `PROGRESS.md`. Two experiments were still running
(W21 two-view model, W22 baselines); section 9 says how their results change the plan.
Goal: a novel research paper on GRN inference from single-cell data.

Citations marked (unverified) are from memory and must be checked before they are used anywhere.

---

## 1. The one-paragraph picture

Every score in this project decomposes as

    score(TF t, gene g) = a(g) + b(t) + c(t, g)

- `a(g)`: how bindable the gene is in this cell type (open, CpG-rich, active promoter).
- `b(t)`: how broadly the TF binds in its ChIP-seq experiment. Within-TF AUROC ignores it.
- `c(t, g)`: the TF-specific part. This is the only part that is a regulatory network.

Seven weeks of experiments say the same thing from different sides: **the benchmark is almost all
`a(g)`, every model so far learns `a(g)`, and nothing has yet learned a `c(t, g)` that transfers.**
The paper has to be built on that fact, not around it.

## 2. The evidence, in numbers

| Finding | Number | Section |
|---|---|---|
| Popularity alone, standard split | 0.72-0.78 within-TF | 15, 31 |
| Random gene vectors instead of expression | ties on hHEP, mHSC-E; −0.028 hESC, −0.019 mESC | 33 |
| Edge-only block model vs best GNN | ties on hHEP, mESC | 31 |
| New TFs: best model vs popularity | nothing beats popularity | 15, 36 |
| Motif alone | 0.57-0.62 | 22, W19 |
| Cross-cell model vs the same model without a graph | no difference; 2 of 8 worse | 37 |
| Cross-cell model replaced by one gene ranking for all TFs | costs at most 0.024 | 37 |
| In-cell, TF-disjoint: learned model vs in-cell popularity | 0.69-0.76 vs 0.73-0.78 | 37 |

Three new measurements (17 Nov, logged in `results.txt` as W23):

| Gene-only score, no labels of the target used | hESC | hHEP | mHSC-E | mESC |
|---|---|---|---|---|
| Detection rate of the gene | 0.520 | 0.603 | 0.583 | 0.620 |
| Total motif hits of the gene, summed over all TFs | 0.603 | 0.515 | 0.511 | 0.557 |
| Pair-specific motif score (all labelled TFs) | 0.602 | 0.552 | 0.555 | 0.535 |
| W20 best transfer model | 0.582 | 0.596 | 0.630 | 0.596 |
| **In-cell popularity, leave-one-TF-out (uses target labels: the ceiling for `a(g)`)** | **0.739** | **0.783** | **0.773** | **0.770** |

Popularity copied from another cell type, on the genes the two datasets share, scores 0.55-0.72
(best: mESC -> mHSC-E 0.721, hESC -> hHEP 0.676). The target's own popularity on the same genes
scores 0.64-0.79.

What these three add:

1. **The `a(g)` ceiling is 0.74-0.78 and the best label-free estimate reaches 0.58-0.63.** That gap of
   0.13-0.16 is the largest prize on this benchmark, and it is a gene-level prediction problem.
2. **Half of the motif signal is not TF-specific.** On hESC the gene's total motif count over all TFs
   scores the same as the TF's own motif (0.603 vs 0.602). Section 22's GC control left 0.554. So
   the TF-specific motif signal is about 0.55, not 0.60.
3. **Bindability is partly generic and partly cell-type-specific.** Another cell's popularity
   recovers about half of the gap between chance and the target's own popularity. The rest has to
   come from the target's expression.

## 3. Consequences

- **More graph architecture will not help.** Sections 26-30, 35 and 37 all failed for the same
  reason: there is no transferable `c(t, g)` in the inputs as currently built, and `a(g)` needs no
  graph.
- **Within-TF AUROC cannot tell `a` from `c`.** A paper claiming a better network needs a metric
  that removes `a(g)`. `analysis/w18_pop_matched.py` already does this (popularity-matched
  within-TF AUROC); it should become a headline metric.
- **PANDA, co-expression and correlation are weak here** (PANDA below motif alone in all four
  cells, correlation near 0.5). Beating them is necessary and not sufficient.
- **ChIP-seq binding is the label, and binding is mostly promoter occupancy.** A method that is
  honest about this (predict occupancy first, specificity second) fits the data better than one
  that pretends to read regulation from co-expression.

## 4. Fixes to make before any new model

These are errors or weak points in the current setup. Each is cheap.

| # | Problem | Fix |
|---|---|---|
| F1 | Settings are chosen on held-out TFs of the source cell. Source validation is 0.70-0.79 while the target is 0.57-0.63, so the criterion rewards fitting the source cell's own `a(g)`. | With three source cells, validate on a held-out source **cell** (inner leave-one-cell-out). Never on source TFs alone. |
| F2 | Leave-one-out batches are proportional to pair counts, so mESC (88,633 pairs) dominates. | Equal weight per cell, per TF, and per class within a TF. |
| F3 | BCE learns the source's positive rate (0.15 to 0.57). The ranking loss gained +0.007 to +0.035 but was only seen on targets. | Make a within-TF ranking or TF-balanced loss the default; confirm on a held-out source cell. W21 tests the balanced form. |
| F4 | Expression statistics depend on platform and depth. | Rank-transform every statistic within a dataset (quantiles), not z-scores. |
| F5 | The motif score mixes gene-level GC/CpG with TF specificity. | Double-centre the motif matrix (remove each gene's mean over TFs, then each TF's mean over genes) and feed both parts separately: gene mean to the `a(g)` model, residual to the `c(t, g)` model. |
| F6 | The gene set is 500 variable genes plus TFs (700-1,100 genes). Bindability is learned from under 1,000 examples per cell and the set is biased towards variable genes. | Train `a(g)` on all expressed genes: BEELINE's full ChIP-seq networks and full expression matrices cover the genome. 10-20 times more training examples at no modelling cost. Keep TFs+500 as the reporting set for comparability. |
| F7 | JASPAR covers 22-34 labelled TFs per cell (34 of 88 in mESC). | Add HOCOMOCO and CIS-BP (CIS-BP infers motifs for TFs from DNA-binding-domain similarity). Report coverage before and after. |
| F8 | Promoter window is −1000/+200 with a max-score scan. | Try the SCENIC cisTarget gene-based rankings (precomputed, cross-species conserved, 500 bp and 10 kb windows). They are a general public resource built from sequence only (unverified: confirm they contain no ChIP-derived tracks in the motif-only version). |
| F9 | No multiplicity correction anywhere; several "best" models were picked on test. | For the paper: fix the comparison list in advance, report Holm-corrected p-values, and never report a test-picked configuration as a result. |
| F10 | mHSC-E, mHSC-GM and mHSC-L in BEELINE are believed to share one ChIP-seq network (unverified). | If true, never train on one and test on another. It is also a free control: same labels, different expression, so any difference between their predictions is expression-driven. |

## 5. Ideas, ranked by expected benefit per day

Each idea has a test that can kill it. Run the kill test first.

### Tier 1: do these first

**I1. Bindability model: predict `a(g)` for a new cell type.** *(largest measurable gain)*
- **What:** a regression model from label-free gene features to the gene's popularity (fraction of
  ChIP-profiled TFs that bind it), trained on source cells, applied to the target.
- **Features:** promoter sequence (CpG observed/expected, GC, CpG island, TATA, total motif density
  from F5), expression in the target cell (detection rate, mean, rank, burst statistics, all
  rank-transformed), Geneformer embedding, and expression breadth across tissues from a public
  atlas (housekeeping genes have open promoters everywhere).
- **Why it should work:** the ceiling is 0.74-0.78 and the gap to the current 0.58-0.63 is a
  dense, low-noise, gene-level target. With F6 there are about 15,000 training genes per cell.
- **Kill test:** gradient-boosted trees on these features, leave-one-cell-out, scored as within-TF
  AUROC of the predicted `a(g)`. If it stays under 0.65 with F6 in place, expression plus sequence
  does not determine promoter occupancy and I7 (accessibility distillation) becomes the route.
- **Novelty:** the decomposition and an explicit transferable occupancy prior. Closest: HOT-region
  and promoter-openness literature; no GRN method was found that separates it (not searched
  exhaustively).
- **Cost:** 1-2 days after F6.

**I2. Residual specificity model: learn `c(t, g)` with `a(g)` removed.**
- **What:** final score = `a(g)` from I1 + `c(t, g)`. Train `c` so that it cannot use gene-level
  information: for each positive (t, g) sample negatives (t, g') with matched `a(g')`; or constrain
  `c` to have zero mean over TFs for every gene.
- **Inputs for `c`:** residual motif score (F5), TF expression and activity in the target cell,
  TF protein features (I5), partner TFs through PPI weighted by partner expression, gene-TF
  covariation with the TF's activity (I6).
- **Metric:** popularity-matched within-TF AUROC.
- **Kill test, to run before building anything:** in-cell TF-disjoint cross-validation (the easiest
  setting) with matched negatives. If matched AUROC is at or below 0.53, these inputs carry no
  learnable TF specificity and the method paper must rest on I1 plus new inputs (I7, I8).
- **Novelty:** matched-negative training and evaluation for GRN inference. Closest: popularity
  deconfounding in recommender systems (IDEAS.md L2); Stock et al. diagnose the bias without a fix.

**I3. More source cell types.**
- Three sources beat one in all four targets (section 37). Cheapest: the three remaining BEELINE
  datasets (subject to F10). Larger: ENCODE cell lines that have both scRNA-seq and hundreds of
  ChIP-seq experiments (K562, HepG2, GM12878, MCF-7, A549).
- **A baseline this makes mandatory:** "copy the TF's targets from other cell types". Same-TF binding
  is shared across cell types, so a reviewer will ask for it. Report every result three ways: TFs
  never seen in any source cell, TFs seen in a source cell, and all.
- **Kill test:** learning curve of target score against number of source cells (1, 2, 3, then more).
  If it is flat from 2 to 3, more cells will not help and the effort goes to features.

**I4. A fair-protocol table (needed for any paper).**
- motif alone, PANDA, GENIE3 + motif pruning, scRegNet on a motif graph (W22), "copy from other
  cells" (I3), predicted bindability alone (I1), the full model. Two metrics: within-TF and
  popularity-matched within-TF. Three TF groups as in I3.

### Tier 2: TF-specific signal, where the science is

**I5. TF protein features, so unseen TFs are not anonymous.**
- A TF currently has no features except its expression and its motif row. Add a protein language
  model embedding of the TF (full sequence and DNA-binding domain) and its TF family.
- **Why:** section 22 found that borrowing targets from TFs with similar motifs scored 0.649-0.711
  on new TFs, far above the motif itself. Similar TFs bind similar genes; protein embeddings give
  that similarity for every TF, including the ones without a motif (54 of 88 in mESC).
- **Kill test:** nearest-neighbour transfer. For a held-out TF, average the ChIP targets of its 3
  most similar source TFs by protein embedding. Compare with section 22's motif-similarity transfer.
- **Novelty:** not found in the GRN methods reviewed (GRNPT uses text descriptions of regulators,
  unverified).

**I6. TF activity from motif regulons, then covariation controlled for bindability.**
- Section 32: correlation with TF activity beats correlation with TF mRNA in all four cells but
  scored only 0.52-0.62, and activity there came from ChIP targets.
- **Change:** activity per cell from the TF's motif targets (leave the scored gene out), then the
  partial correlation of the gene with that activity given the gene's detection rate. On the
  time-course datasets (hESC, hHEP, mESC) add the lagged version along pseudotime.
- **Kill test:** matched within-TF AUROC of this score alone. Under 0.52 on all four cells: drop.

**I7. Predicted accessibility distilled from multiome data.** *(the most novel route, higher risk)*
- **What:** the missing input is chromatin accessibility. Train a teacher on public paired
  RNA + ATAC single-cell data to predict promoter and enhancer accessibility of each gene from the
  cell population's expression. Apply it to an RNA-only dataset to get a predicted accessibility
  per gene and per regulatory element. Use it as `a(g)` and to restrict the motif scan to elements
  predicted open.
- **Why:** methods with ATAC (SCENIC+, CellOracle, LINGER, Dictys; all unverified here) are
  much stronger than RNA-only ones. An RNA-only method that recovers most of that gap is a clear,
  useful claim: "what ATAC would have told you, from RNA alone".
- **Kill test:** on a held-out multiome dataset, correlation between predicted and measured
  promoter accessibility across genes. If under about 0.6, stop.
- **Leak rule:** the teacher's training data must not include the four BEELINE cell types.

**I8. Foundation-model probes as pair features.**
- Geneformer helps through content (section 34), but it enters only as a per-gene vector.
- **Pair-level probes:** (a) in-silico deletion: remove the TF from each cell's token list and measure
  the shift in the gene's embedding; (b) attention from the gene's token to the TF's token,
  averaged over cells and heads.
- **Kill test:** matched within-TF AUROC of each probe alone on hESC. Cheap: a few GPU-hours.
- **Leak check:** Geneformer was pretrained on expression only, so no ChIP leak.

**I9. Biophysical decoder.**
- Replace the free MLP by `P(bound) = sigmoid(log open(g) + log(conc(t)) + w * energy(t, g) + coop(t, g))`,
  with `open` from I1 or I7, `conc` from TF expression or activity, `energy` from the residual
  motif score, and `coop` from PPI partners that have a motif hit in the same promoter.
- **Why:** it has a handful of parameters, so it cannot memorise a source cell, and each term is
  interpretable. Low-capacity models are the ones that transfer in section 37.
- **Do after** I1 and I2 show that the terms carry signal.

**I10. Label-free adaptation to the target.**
- W21 tests one form (view consistency on the target). Two others: self-training on the model's
  most confident target predictions; and matching the distribution of predicted `a(g)` on the target
  to the source by quantiles.
- Only worth doing once F1 is in place, since the source-target gap is partly a selection artefact.

### Tier 3: other directions in this domain

**D1. A different ground truth: perturbation data.**
- ChIP-seq measures binding, not regulation. Large TF perturbation datasets with single-cell
  readout exist: genome-wide Perturb-seq in K562 and RPE1 (Replogle et al. 2022, unverified) and a
  TF overexpression atlas in human embryonic stem cells (Joung et al. 2023, unverified; the same
  cell type as the hESC dataset).
- **Use 1:** a second evaluation: does `c(t, g)` predict which genes change when the TF is perturbed?
  A model that wins on both binding and perturbation is a much stronger paper.
- **Use 2:** training signal for `c(t, g)` that popularity cannot explain (perturbation effects are
  not dominated by promoter openness; this assumption needs checking).

**D2. Differential networks.**
- Predict what changes between two cell types: for a TF with ChIP-seq in both, which genes gain or
  lose binding, from the change in expression. Generic binding and `a(g)` cancel in the difference,
  so the task isolates the cell-type-specific signal. The `differential-grn` branch and
  `data/*/TFs+500/differential/` already hold a start.
- **Limit:** needs TFs profiled in two cell types; within BEELINE that is 2-10 TFs per pair. I3's
  ENCODE cell lines fix this.

**D3. The benchmark paper** (section 6, option A). Most of its evidence already exists.

**D4. Leave for later:** cell-state-specific networks (section 23: untestable without state-specific
ChIP-seq), causal discovery from observational expression, any new encoder on the completion task.

## 6. Paper options

**Option A: "What supervised GRN link prediction benchmarks measure".**
- **Claims:** (1) on the standard split, random features, popularity and an edge-only block model
  match published GNNs; (2) the graph is the training half of the test network, so the task is
  completion; (3) under TF-disjoint, gene-disjoint and cross-dataset evaluation the advantage
  disappears; (4) a corrected protocol and a popularity-matched metric.
- **Have:** sections 15, 24, 31, 33-37, W18. **Missing:** the same controls (random features,
  popularity, block model) rerun inside two or three other published methods' own code (GENELink,
  GNNLink, GCLink), so the claim is about the field and not one model.
- **Risk:** Stock et al. (bioRxiv 2025.12.19.695616, unread) report that degree heuristics match
  GNNs. Read it first. The random-feature test, the block model, the identity control, the held-out
  protocols and the matched metric are the likely differences.
- **Effort:** 2-3 weeks. Low risk.

**Option B: "Occupancy first: transferable GRN inference from scRNA-seq and sequence".**
- **Claims:** (1) the `a + c` decomposition; (2) a bindability model that transfers across cell types
  and species (I1); (3) a specificity model trained and scored with popularity removed (I2, I5);
  (4) beats PANDA, GENIE3 + motif pruning and scRegNet-on-motif under the fair protocol.
- **Minimum result that makes it a paper:** predicted bindability reaches at least 0.68 within-TF on
  all four targets (from 0.58-0.63), and `c` is significantly above 0.5 on the matched metric for
  unseen TFs in at least three targets.
- **Risk:** I2's kill test fails. Then B becomes "bindability explains ChIP-seq benchmarks, and here
  is how to predict it", which merges into A.
- **Effort:** 4-6 weeks.

**Recommended: A and B as one paper.** Part 1 is the diagnosis (already done). Part 2 is the method
the diagnosis calls for. The diagnosis explains why the method is built the way it is, and the
method shows the diagnosis is useful. Neither half depends on beating a moving state of the art on
a flawed benchmark.

**Option C (larger, later):** I7 as its own paper, with D1 as the evaluation.

## 7. What not to spend time on

| Do not | Because |
|---|---|
| New encoders, deeper GNNs, path or subgraph methods | Sections 26-30, 35, 37: no gain, same cause each time |
| Co-expression edges as regulatory evidence | Correlation scores about 0.5; adding the C relation moved results by −0.029 to +0.032 with no pattern |
| PANDA as a component | Below motif alone in all four cells; its Tanimoto decoder lost 0.03 |
| Tuning on source-TF validation | F1 |
| Comparing against scRegNet's published numbers | Different task; only the motif-graph version is fair |
| Reporting pooled AUROC as a headline | It rewards `b(t)`, an experimental artefact (section 1 of PROGRESS.md) |
| Motif scanning with more windows of the same kind | Section 22 tried three windows; the limit is specificity, not the window |

## 8. Order of work

| Step | Work | Days | Decides |
|---|---|---|---|
| 1 | F1-F5 in `src/xcell_gnn.py`; rerun the no-graph model | 1 | How much of the transfer gap was selection and loss |
| 2 | I2 kill test: in-cell matched AUROC of residual motif, activity covariation (I6), protein-similarity transfer (I5) | 1-2 | Whether any TF-specific signal exists in current inputs |
| 3 | F6 + I1 bindability model, leave-one-cell-out | 2-3 | Whether Option B's first claim holds |
| 4 | I3 extra BEELINE cells and the "copy from other cells" baseline | 1-2 | Learning curve; mandatory baseline |
| 5 | I4 table with W22 baselines, both metrics | 1 | First complete results table |
| 6 | Read Stock et al.; rerun controls in two other methods' code | 3-5 | Option A's novelty |
| 7 | I5 full model, I8, then I9 | 5+ | Option B's second claim |
| 8 | D1 perturbation evaluation | 3-5 | Strength of the final paper |

Steps 1-3 take about a week and settle which paper this is.

## 9. How the two running experiments change this

- **W21 (two-view model): done, section 39. The prediction held.** Mean within-TF over the 8
  settings: no-graph MLP 0.599, co-expression view 0.598, concat 0.597, gate 0.595, motif + PPI
  view 0.593, early fusion 0.588, motif alone 0.561. What it changes:
  - **Fusion adds nothing.** No fusion beats the MLP (concat 1 better / 2 worse of 8 at p < 0.05,
    early 0 / 4). Graph structure of any kind is not where the signal is; stop adding graphs.
  - **These models are gene priors too.** Averaging a run's score grid over TFs costs at most 0.043
    and raises within-TF on the mESC targets. The source-validation to target gap stays 0.12-0.16.
  - **F3 is half confirmed.** TF-balanced BCE is worth about +0.005 over plain BCE (1 better / 1
    worse); the MLP here beats W20's by 0.01-0.02 mostly for that reason. Keep it as default, but
    it is not the fix for the gap.
  - **I10 in its consistency form is dead.** InfoNCE between views lowers the target by 0.006-0.013
    on source only and by 0.023-0.032 when computed on the target graph (0 better / 5 worse at
    weight 0.5). Target adaptation stays in Tier 2 and only as self-training or importance
    weighting.
  - **hESC stays the hard target.** On TFs with a motif, every learned arm loses to motif_max there
    (gate 0.528 and 0.603 against 0.619). Any gene prior has to be checked against it.
  - scRegNet-motif (mean 0.605) is above every W21 arm, so the bar in the next bullet stands.
- **W22 (baselines): done, section 38.** What it changes:
  - **scRegNet on a motif graph is the strongest method so far** (0.60-0.64 on six of eight
    transfers, +0.003 to +0.047 over the W20 model), and its additive decoder gives every TF the
    same gene ranking (within-TF equals the TF-averaged grid to 1.2e-4 in all 48 runs). The best
    cross-cell method is, exactly, a gene prior. "Every architecture learns `a(g)`" now includes the
    starting paper. This is a headline line for Option A and the direct motivation for I1.
  - **The bar for I1 is now 0.60-0.64**, not 0.58-0.63, and the baseline to beat is scRegNet-motif.
  - **Expression importance has no TF-specific value here.** GENIE3-style regression is at chance
    (0.46-0.52) and pruning it with motifs never beats the motif alone. Drop it from I2's inputs.
  - **PPI does not rescue motif-less TFs.** PANDA scores them at or below chance (0.44-0.50). I5
    (protein features) and the bindability prior are the routes for those TFs, not PPI propagation.
  - **Pooled AUROC of the learned gene priors is below motif alone on human targets.** Report both
    metrics; the calibration across TFs (`b(t)`) is lost in transfer and may need a per-TF term.

## 10. Open questions to settle early

1. Is a gene-level prior from **other cell types' ChIP-seq** an allowed input? It is general and
   public, and it is what "copy from other cells" uses. It would lift `a(g)` a lot (0.55-0.72 on
   shared genes from a single other cell). The stricter rule (no ChIP-derived input at all) is
   cleaner; the looser one is closer to how a practitioner would work. Report both.
2. Is the evaluation set TFs+500 or all expressed genes? All genes is the honest "complete network".
3. Which claim matters more: predicting binding (ChIP-seq) or predicting regulation (perturbation)?
   D1 depends on this.
