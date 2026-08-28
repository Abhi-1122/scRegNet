# Differential/Contrastive GRN — Project Explainer

Branch: `differential-grn`.

This document is **fully self-contained**. Reading only this file is enough to understand
the whole project — the main claim, the biology, the method, the literature it builds on,
the honest results, and what to do next. No need to open `DIFFERENTIAL_GRN_PLAN.md`,
`LIT_REVIEW.md`, `work.md`, or anything else in this repo.

Those files still exist as the detailed process record and exact technical spec, and §6 at
the end maps what lives where if you ever want to dig further — but nothing in that map is
required reading to understand what's written here.

---

## 0. The core claim, in one paragraph

This project extends `scRegNet` — a model that fuses a graph neural network over a curated
gene-regulatory reference network with a single-cell foundation-model (scFM) gene embedding
to predict whether a TF→target regulatory edge exists — from predicting **static** edges in
one condition to predicting **differential** edges across two conditions.

Concretely: does the regulatory relationship between transcription factor X and target gene
Y exist, strengthen, weaken, flip, or stay the same, when a cell population moves from one
biological state to another (an earlier time point to a later one, or a less-differentiated
population to a more-differentiated one)?

- The core technical contribution is a single **Siamese (weight-shared) network** that
  ingests **both conditions in one joint training pass** and outputs a discrete **4-way
  label** per candidate edge: conserved / condition-A-specific / condition-B-specific /
  neither.
- Training uses a statistically constructed **"silver" label**, since no ground-truth
  differential label exists in this data.
- The model is validated three ways: against a naive gene-level-statistics baseline;
  against an adaptation of how the field currently does this (train two models separately,
  compare after the fact); and against a synthetic system with a truly known answer.

**The result, worked through in full below, is positive but mixed, and closer to positive
than negative:**

- The joint model beats the naive statistical baseline by a wide margin on 7 of 9 real
  condition pairs.
- It clears its synthetic sanity-check bar comfortably.
- Most notably, it correctly recovers one specific, textbook-documented piece of regulatory
  biology — the GATA2→GATA1 "GATA switch" in blood-cell erythroid commitment — at maximum
  model confidence, in exactly the biological context where that switch is expected to be
  sharpest.
- But several other predicted biological patterns did not resolve as expected, and one of
  the three cell types studied (human embryonic stem cells, hESC) turned out to have too
  little detectable differential signal in its data to fairly test the method at all.

The rest of this document explains all of that in detail: the literature this builds on,
what "condition A vs B" concretely means for each biological system studied, how the model
and its labels are built, exactly what every number below means and why, and what to do
next.

---

## 1. What we did and why

### 1.1 The motivation

The base project on this repo, `scRegNet` (see the `summer26` branch and this branch's
earlier history), predicts **static** gene regulatory network (GRN) edges:

- Given a scRNA-seq dataset for one cell type, and a candidate TF→target edge from a
  curated reference network, predict whether that edge is real.
- It fuses two signals: a GNN over the reference-network graph structure, and a single-cell
  foundation model (scFM, e.g. Geneformer) embedding of each gene's identity.
- Edges are scored by a small decoder on top of the fused representation.

That line of work — hyperparameter tuning, scFM-swap experiments, PPI/PANDA fusion (see
`summer26`'s `work.md`) — converged on a performance ceiling: static single-condition edge
prediction was saturated, and further tuning wasn't moving the needle.

This branch pivots to a different, less-picked-over question:

- **Not** "does this edge exist" — **but** "does this edge's regulatory activity change
  between two biological conditions" (e.g. two time points in a differentiation time
  course, or two stages of a differentiation-potential gradient).
- This is a more biologically interesting question: differential regulation is what
  actually drives cell-fate decisions, not just the static presence of a wiring diagram.
- Per the literature review (§1.4 below, restated in full from `LIT_REVIEW.md`), it is also
  a genuinely less crowded research question than static GRN inference.

### 1.2 The three biological systems, and what "condition A vs condition B" means for each

All data was already in this repo (BEELINE benchmark datasets) — no new data acquisition
was needed. Three cell types are used, each a genuinely different *kind* of biological
transition — deliberately not "just three replicates of the same idea."

**hESC — human embryonic stem cells exiting pluripotency toward definitive endoderm**
(Chu et al. 2016 time course):

- This is a *directed lineage decision*: a pluripotent stem cell population is pushed, over
  time, through mesendoderm induction into definitive endoderm — a specific,
  textbook-characterized fate.
- Samples are labeled by hours after the differentiation cue: `00h` (pluripotent) through
  `96h` (committed definitive endoderm), with `12h`/`24h`/`36h`/`72h` in between.
- "Condition A vs condition B" here literally means "cells sampled at hour X vs cells
  sampled at hour Y" — the *same* differentiation protocol, two snapshots in time.
- Three pairs were chosen:
  - `00h_vs_96h` — biggest contrast, start vs end (the primary/bookend pair).
  - `00h_vs_12h` — the fast initial exit from pluripotency.
  - `36h_vs_72h` — a mid-trajectory transition that deliberately avoids the two
    most-different-looking endpoints, to test whether the model is finding real biology or
    just "distance from t=0."

**mESC — mouse embryonic stem cells exiting *naive* pluripotency** (Hayashi et al. 2018,
2i/LIF-withdrawal time course):

- This is a different *kind* of process from hESC's directed lineage decision.
  Withdrawing 2i/LIF from naive mESCs doesn't push them down one specific lineage — it
  causes the *naive pluripotency circuit itself* to collapse.
- That circuit is a specific, named set of transcription factors: Klf4, Klf2, Esrrb, Tbx3,
  Zfp42, Prdm14 — the genes that actively maintain the naive state.
- After the circuit collapses, cells drift toward a more general "formative" state.
- Expected biological signature: not "gene X turns on, gene Y turns off in sequence" but
  "the whole naive circuit loses its internal coherence together."
- Time points: `0h` through `72h`. Three pairs were chosen:
  - `0h_vs_72h` — bookend (primary pair).
  - `0h_vs_12h` — the fast onset of circuit collapse.
  - `24h_vs_48h` — mid-transition, where the *formative*-state marker Otx2 should be
    rising.

**mHSC-E — mouse hematopoietic stem/progenitor cells committing to the erythroid lineage**
(Nestorowa et al., flow-sorted):

- This is **not** a time course at all — it's a differentiation-*potential* gradient
  captured by flow-sorting cells into three discrete populations.
- `LT_HSC` (long-term hematopoietic stem cells, most primitive) → `HSPC` (a
  stem/progenitor intermediate) → `Prog` (erythroid-committed progenitors, most
  differentiated).
- "Condition A vs condition B" here means "cells sorted into population X vs population Y,"
  not "cells sampled at time X vs time Y."
- Three pairs were chosen:
  - `LT_HSC_vs_Prog` — bookend, biggest contrast (primary pair).
  - `LT_HSC_vs_HSPC` — early loss of stemness.
  - `HSPC_vs_Prog` — the erythroid-commitment step itself, where the textbook
    GATA2→GATA1 "GATA switch" and the PU.1/GATA and GFI1/GFI1B antagonisms are expected to
    be sharpest (these are specifically erythroid-vs-myeloid decision factors).

**What defines the candidate edge space**: each cell type's reference network (the same
curated ChIP-seq/database source `scRegNet`'s static task already uses) defines which
TF→target pairs are even eligible for a differential label.

- A TF→target pair is only ever a candidate if it's already a validated edge in that cell
  type's static network.
- This means the model is never asked to invent an edge from nothing — it only ever
  resolves *whether a known-plausible edge's activity changed* between two conditions.
- Candidate-edge counts (same set reused across all 3 pairs of a given cell type):
  hESC 4,545; mESC 29,613; mHSC-E 11,557.

### 1.3 What "novel" means here, precisely

- The core mechanism — a weight-tied paired-snapshot encoder + a discrete per-edge
  state-transition classifier + a contrastive loss — is a known *pattern* in the general
  dynamic-graph-learning literature (§1.4.C below). This project does **not** claim to have
  invented that shape.
- What's new is the specific **combination**:
  - Applied to GRN inference at all — no dynamic-graph paper does this.
  - Fused with scFM gene-identity embeddings — no dynamic-graph paper does this either.
  - Jointly trained rather than post-hoc-compared — the field's current practice
    (represented by RegGAIN, the closest prior work — §1.4.D) trains one static model per
    condition and runs a statistical test after the fact.
  - Trained against a discrete 4-way supervised target, validated with named,
    pre-registered biological checks.
- This is an honest, narrower novelty claim than "first-ever differential-GRN model," and
  it's the one any resulting paper should make.

### 1.4 The literature this idea builds on and compares against

This subsection restates, in full, the literature landscape found during the design
process (originally `LIT_REVIEW.md`, produced by a dedicated literature-review pass before
any code was written).

#### 1.4.A — GRN-specific contrastive-learning / GNN deep-learning methods

These are the methods closest in *mechanism* (graph neural networks, contrastive learning,
applied to GRN inference):

| Method | Venue / date | What it does | Cross-condition element? |
|---|---|---|---|
| **RegGAIN** | *Advanced Science*, 2025/2026 | Dual-role (regulator/target) graph contrastive learning: two augmented views of one graph (edge-drop+mask vs. topology-aware hub-preserving perturbation), InfoNCE-style contrastive loss, direction-aware edge decoding | Applies the trained *static* model separately to each of 6 hESC time points (the exact Chu et al. data this project uses) and to MM-vs-normal bone marrow; ranks TFs by differential out-degree via a **Wilcoxon rank-sum test comparing two independently-inferred static networks** — post-hoc, not jointly trained. **The single closest prior-work overlap found — see §1.4.D for a full breakdown.** |
| **GP-DHT** | arXiv 2601.10995 (2026) | Heterogeneous gene/cell graph + multi-relational graph attention + dual-head Transformer; gene-pair-level supervised contrastive learning | **Cross-species** (human vs. other species), not within-species time/condition. Conserved-vs-species-specific regulation is an analysis *output*, not a jointly-trained 4-way label. |
| **BRIDGE** | arXiv 2606.14734 (2026) | Gene+cell dual representations from the expression matrix and its transpose; contrastive learning between a co-expression-refined view and the original graph; **cross-cell-type transfer** (pretrain on mESC, fine-tune on 5% of a target cell type's data) | None — static network per cell type; the "transfer" axis is cell-type identity, not paired differential prediction. |
| **BRIDGE-GRN** (a distinct, similarly-named paper) | bioRxiv 2026.05.12.724562 | Role-aware bi-tower graph learning; cross-view contrast between identity- and edge-perturbed graph views, for **directed** GRN inference | None — a robustness-to-noise contrastive objective, not cross-condition. |
| **SupGCL** | arXiv 2505.17786 / ICML 2026 | Probabilistic generalization of graph contrastive learning that substitutes **real gene-knockdown perturbations** for artificial graph augmentations as the contrastive positive/negative signal | None on the time/condition axis — applied to patient-level cancer-subtype GRNs, not paired biological conditions of one system. Uses the strongest ground-truth signal in this table (real knockdown data), but for a different task. |
| **Meta-TGLink** | *Genome Biology*, 2025 | Meta-learning + Transformer-GNN hybrid for few-shot cross-domain GRN link prediction | Cross-cell-type/cross-domain generalization, not paired differential prediction. |
| **KEGNI** | *Genome Biology*, 2025 | Knowledge-graph-enhanced GNN for GRN inference | None described as cross-condition. |
| **GCLink**, **DGCRL**, **DeepMCL**, hierarchical-graph-transformer-with-CL (IEEE) | Various, cited as baselines in the above papers | Graph contrastive learning / directed-augmentation / multi-view fusion, all for static single-condition GRN inference | None — static only. |
| **scRegNet** (this project's own base method) | bioRxiv 2024.12.16.628715 | Joint GNN + scFM representation — exactly the dual-branch architecture this project extends | Static, single condition — the paper this repo already extends. |
| **DeepFGRN** | *Briefings in Bioinformatics*, 2024 | Directed graph embedding + regulation-*type* classification (activator/repressor) | Not condition-differential — a reminder that "extra label dimensions beyond binary edge/no-edge" isn't itself new, just a different axis (edge sign, not condition). |
| **CroCoNet** | bioRxiv 2025.11.18.689002 | Not a GNN — quantitative network-module comparison (topology variability within/between species), centered on known TF modules | **Cross-species** rewiring (human/gorilla/macaque neural differentiation) — analogous in spirit to "differential," but on the species axis, and classical/statistical rather than deep learning. Notably validated its top finding (POU5F1/OCT4 module divergence) with **real cross-species CRISPRi + scRNA-seq perturbation** as a follow-up confirmation — the only entry in this whole review that does new wet-lab perturbation work. |

#### 1.4.B — Classical/statistical differential-network-biology methods

This field substantially predates the deep-learning wave (some methods from the early
2010s). These are the direct methodological ancestors of this project's silver-labeling
approach (§2.2):

| Method | Venue / date | What it does |
|---|---|---|
| **DiffCorr** | Fukushima, *BMC Systems Biology*, 2013 (R package, still widely used) | Differential *correlation* network analysis — Fisher z-test on correlation coefficients between two conditions. **This is the exact statistical lineage this project's silver-labeling pipeline (§2.2) is built on.** |
| **DINGO** | Ha, Baladandayuthapani & Do, *Bioinformatics* 31(21), 2015 | Bayesian pathway-based differential network model; borrows statistical strength across groups; can integrate multiple omics layers. |
| **CFGL** | bioRxiv 290346 | Condition-adaptive fused graphical lasso — condition-specific co-expression networks that share structure across conditions when the data supports it. |
| **jointGHS** | Camiling et al., *Annals of Applied Statistics* 18(3), 2023 | Joint graphical horseshoe Bayesian model for multiple Gaussian graphical models — explicitly estimates **shared edges across conditions in addition to condition-specific edges**, via a continuous per-edge latent shrinkage variable. **Conceptually the closest classical analogue to this project's "conserved vs. specific" label** — but continuous/probabilistic, not a hard 4-way trained classifier. |
| **CSN / c-CSN** | Dai et al., *Nucleic Acids Research*, 2019 and follow-ups | Statistical test yielding a network *per single cell* (conditional variant adds conditional-independence tests to reduce false positives). |
| **Sc-compReg** | Duren et al., *Nature Communications*, 2021 | Joint scRNA+scATAC clustering across two conditions; a likelihood-ratio-style statistic for testing differential regulatory relations between matched subpopulations. |
| **BoostDiff** | Galindez et al., *Bioinformatics Advances* 4(1), 2024 | AdaBoost ensemble of "differential trees" with a novel differential-variance-improvement splitting criterion, converting variable importance into a directed differential network. |
| **DCI** | Belyaeva, Squires & Uhler, *Bioinformatics*, 2021 | Causal-DAG-based method that directly estimates the *causal difference* between two related GRNs (not each network separately then a diff), with finite-sample statistical guarantees. |
| **scTenifoldNet** | bioRxiv 2020.02.12.931469 | Regression + manifold alignment + low-rank tensor denoising, for constructing and comparing transcriptome-wide GRNs from two scRNA-seq conditions. |
| **DiCoLo** | RECOMB 2026 (bioRxiv 2025.11.23.689932) | Integration-free, cluster-free detection of localized differential gene co-expression via Optimal-Transport-based per-condition gene graphs. |

**The key structural fact this table establishes**:

- Several of these classical methods (jointGHS, Sc-compReg, DCI) already formalize the
  statistical notion of "conserved vs. condition-specific edge" that this project's 4-way
  label approximates with weak supervision.
- **Critically, every single one of them validates purely against
  simulation-with-known-truth and/or literature/pathway plausibility when applied to real
  biological data with no perturbation available** — none claim or need real perturbation
  data for real-data validation.
- That is exactly the validation pattern this project follows: §2.2's silver label + §3.4's
  synthetic calibration check + §3.5's named biological checks mirror this
  field-standard three-part approach.

#### 1.4.C — The general (non-biological) dynamic-graph-learning check

Before finalizing the novelty claim, the *mechanism shape itself* — a paired-snapshot
Siamese encoder, discrete per-edge appear/disappear/persist/absent classification,
contrastive learning across two graph states — was checked against the general
(non-biological) dynamic-graph representation-learning literature, which is large and
predates essentially all the GRN-specific papers above:

| Method | Venue / date | What it does | Why it's not the same as this project |
|---|---|---|---|
| **EvolveGCN, DySAT, TGN, ROLAND** | Various, standard architectures for dynamic-graph node/edge classification and link prediction | Support "edge appears/disappears" framing in principle; next-step link forecasting is their dominant benchmark task | Built for **many-snapshot or continuous-time streaming graphs** (dozens to thousands of timestamps — social interactions, transactions, recommender events), trained to forecast the *next* state of an ever-evolving graph — not to jointly ingest exactly **two designated conditions** and emit one discrete 4-way transition label per edge. None frame their output as conserved/A-specific/B-specific/neither between two named states. |
| **DyGnROLE** | arXiv 2602.23135 (Bonnet & Rei) | Edge classification on dynamic graphs with a role-aware (source/destination-asymmetric) architecture and a directional contrastive pretraining objective | The closest general-ML match found — edge classification + contrastive pretraining on a graph with directional source/target roles, much like TF→target edges. But it classifies edges *as observed in a continuous interaction stream*, not a discrete transition-type label comparing two designated snapshots, and has no biological or foundation-model application. |
| **Graph-snapshot change-point detection via Siamese GNNs** | arXiv 2203.15470 and the related Laplacian change-point line of work | A Siamese, weight-tied GNN encoder compares **pairs of graph snapshots** directly | **The structurally closest match found overall** — a paired-snapshot Siamese encoder, exactly like this project's two-condition encoder. But the output is a whole-graph similarity/change-point *score*, not a per-edge discrete label, and this line of work uses no real node features (foundation-model or otherwise) and no biological data. |

**Conclusion from this check, stated plainly**:

- The core mechanism shape (weight-tied paired-snapshot encoder + discrete per-edge
  state-transition classifier + contrastive objective) is a **recognized general pattern
  in dynamic-graph learning, not a novel invention of this project**.
- What was not found anywhere — in either the GRN-specific literature or this general
  dynamic-graph literature — is the specific *combination* actually used here: applied to
  gene regulatory networks at all, fused with scFM gene-identity embeddings, jointly
  trained (not post-hoc) for GRN data specifically, with a discrete 4-way supervised
  target and named biological validation.
- This is the honest, narrower novelty claim from §1.3.

#### 1.4.D — RegGAIN: the critical overlap case, examined closely

RegGAIN is the single closest prior-work match found — same architecture family
(contrastive GNN), same benchmark data (BEELINE hESC/mESC/mHSC-E), and same real time
course (the exact Chu et al. hESC 6-time-point dataset this project uses). It gets its own
detailed breakdown rather than being buried in a table row.

**What RegGAIN actually does**:

1. Trains one contrastive GNN with dual-role (regulator/target) gene embeddings, using two
   augmented graph views and an InfoNCE-style contrastive loss between the two views'
   embeddings of the same gene.
2. This is trained and benchmarked as a **single static GRN inference model** — on
   BEELINE (hESC, mESC, mHSC-E among 7 datasets), against one static reference network per
   cell type, exactly like `scRegNet`'s own baseline task.
3. Separately, as an *application/case study* (not the main benchmark), the same
   trained-model architecture is run **independently** on: (a) the Chu et al. hESC
   time course, inferring "a GRN at the individual sample level," and (b) multiple myeloma
   vs. normal bone marrow.
4. For each such application, a **Wilcoxon rank-sum test compares edge-score
   distributions between two independently-inferred static networks** to rank TFs by
   differential out-degree. There is no single model that ingests both conditions and
   outputs one 4-way (or any) differential label per edge in one trained/joint step.
5. Validation of the "dynamic regulon"/rewiring claims is **entirely qualitative**: GO
   enrichment of stage-specific TF modules, Jaccard similarity of top-TF sets between
   consecutive stages, manual inspection of specific feed-forward loops (e.g.
   GATA2-GATA3-STAT3). No quantitative differential-edge metric against any ground truth is
   reported anywhere.
6. Ground truth: one static reference network (NicheNet) reused identically across all six
   time points — RegGAIN faces **exactly the same "no time-resolved reference network"
   problem** this project solves with a silver label (§2.2), and does **not** solve it —
   it sidesteps the problem by never requiring the reference network to be
   condition-specific in the first place.

**How this project's approach differs from RegGAIN's, point by point**:

| Dimension | RegGAIN | This project |
|---|---|---|
| Training | Two static models trained independently, one per condition | **One** Siamese model trained jointly on both conditions at once |
| Comparison mechanism | Wilcoxon rank-sum test on edge scores, applied *after* training, as a separate step | A trained 4-way classification head, part of the model itself |
| Output granularity | Per-TF differential-*ranking* (out-degree shift) | Per-*edge* discrete label (conserved/A-specific/B-specific/neither) |
| Contrastive axis | Two *augmented views* of one static graph (edge-drop/hub-perturbation) — the augmentation is artificial | Two *real biological conditions* (e.g. 0h vs 96h expression) — the positive/negative pairing is supervised by the differential label itself |
| Node features | Expression-only | Expression (GNN branch) **+ scFM gene-identity embedding** (fused) |
| Real-data validation | Entirely qualitative (GO enrichment, Jaccard similarity, manual loop inspection) | Named, pre-registered biological checks (§3.5) **plus** a quantitative silver-label metric (§2.2) **plus** a synthetic ground-truth check (§3.4) |

**Precise verdict — what RegGAIN establishes**:

- Contrastive GNNs for GRN inference on this exact data are already published.
- Post-hoc comparison of two independently-run static networks for "rewiring" analysis is
  already published on this exact hESC time course.
- Purely qualitative validation of stage-specific regulatory claims is already the field's
  practice on this exact dataset.

**What RegGAIN does *not* establish** — exactly what this project adds:

- Joint simultaneous training of both conditions in one model.
- A discrete 4-way per-edge label.
- Any dedicated differential-edge quantitative metric.

This project's own `posthoc_baseline` (§2.3, §3.3) is a direct, same-codebase adaptation of
RegGAIN's Wilcoxon-based procedure, specifically so the "joint beats post-hoc" claim above
could be tested with real numbers rather than just architectural description.

#### 1.4.E — What can our numbers actually be compared to?

**Honest answer: almost nothing outside this project's own codebase, and that limitation
should be stated plainly rather than glossed over.**

- **None of the papers in §1.4.A/§1.4.B report a directly comparable metric.** RegGAIN
  reports no quantitative differential-edge metric at all (§1.4.D) — its validation is
  qualitative (GO enrichment, Jaccard similarity). GP-DHT, BRIDGE, SupGCL, and the classical
  methods in §1.4.B each report numbers, but for different tasks entirely (cross-species
  conservation, cross-cell-type transfer, knockdown-supervised patient classification,
  simulation-only differential-network recovery) on different data, with different metrics.
  No specific AUROC/AUPRC/F1 number from any of these papers was captured during the
  literature-review process as being on a task close enough to report side-by-side with
  this project's numbers — stating that honestly here rather than inventing or
  approximating a number that was never actually verified.
- **What IS directly, apples-to-apples comparable**: this project's own `posthoc_baseline`
  and `naive_expression_baseline` (§2.3), run on the *exact same data and codebase* as
  `full`. These are the only genuinely fair comparisons available, and are the basis of
  this project's central quantitative claim (§3.3):

  | Comparison | Basis |
  |---|---|
  | `full` vs `posthoc_baseline` | Same data, same architecture family, `posthoc_baseline` is a direct adaptation of RegGAIN's own statistical procedure (§1.4.D) |
  | `full` vs `naive_expression_baseline` | Same data, tests whether graph+scFM structure adds anything over gene-level statistics alone |

- **The pre-existing static scRegNet baseline is a reference point, not a benchmark for
  this project's task** — it's a *different task* (static single-condition edge existence,
  not differential change), so it's not a "better/worse" comparison, only a sanity check
  that the shared architecture still works:
  - Paper-reported (`scregnet.pdf`, Table 2, hESC/TFs+500, scRegNet w/Geneformer):
    **AUROC 0.88, AUPRC 0.62**.
  - Independent rerun on the `summer26` branch before this project's changes: **AUROC
    0.8874, AUPRC 0.6225**.
  - This project's own Phase 1 check (§3.2): the differential model's auxiliary static
    task, on hESC `00h_vs_96h`, scored **AUROC 0.873/0.874, AUPRC 0.613/0.616** for the two
    conditions — close to the reference numbers above, confirming the shared-adjacency
    Siamese design didn't break the underlying static task. This is a sanity check, not a
    differential-task result.

---

## 2. How it works

### 2.1 Architecture — conceptual picture first

The base `scRegNet` static model has two branches fused together per gene:

- A **GNN branch** — message-passing over the reference-network graph, using each gene's
  raw per-cell expression as its input feature.
- An **scFM branch** — a fixed, pre-computed foundation-model embedding of the gene's
  identity (e.g. Geneformer), the same vector regardless of which cell or condition you're
  looking at.

The fused per-gene representation is then turned into a TF embedding and a target
embedding via a shared MLP, and an edge score is the dot product (or a small MLP) of the
two.

**The differential model, `scDiffTransNet_GCN` (`src/models.py`), extends this to two
conditions at once**, in a single forward pass, with almost everything shared:

- **The graph structure is shared.** Both conditions' expression is passed over the *same*
  adjacency matrix, built once from the static reference network exactly as the base model
  does.
  - This was a deliberate choice, not an oversight: a condition-specific graph would have
    to be built from the differential-correlation signal we're trying to predict —
    circular.
  - Sharing structure and varying only node features keeps this a real extension of the
    static model, not a different model wearing its clothes.
- **The scFM branch is shared and condition-independent** — same limitation as the base
  model: a gene's foundation-model embedding doesn't know what condition it's in.
  - This is intentional, and turned into an explicit ablation (`no_grn`) — see §4.
- **Only the first projection layer is condition-specific.** Condition A and condition B
  have different numbers of cells (e.g. hESC `00h` has 92 cells, `96h` has 188), so each
  condition needs its own linear layer to project its raw per-cell expression vector down
  to a common hidden size.
  - Everything after that — the graph convolutions, the scFM/GNN fusion, the TF/target
    MLP, the edge decoder — uses the exact same weights for both conditions (a "Siamese"
    network).
  - This is what makes it a genuinely joint model: the same learned function is applied to
    both conditions, not two independently-trained copies.
- **A new 4-way differential head** sits on top. For each candidate edge, it looks at the
  TF and target embeddings *from both conditions at once* (concatenated, plus their
  absolute difference) and outputs a 4-way classification: **conserved / A-specific /
  B-specific / neither**.
- **A contrastive loss** pushes the model to make embedding *differences* between
  conditions meaningful:
  - For edges the silver label calls "conserved," the model is pushed to make the edge's
    representation similar across conditions.
  - For edges called condition-specific, it's pushed to make them *dissimilar*.
  - This is trained jointly with an auxiliary static link-prediction loss (reusing the base
    model's existing train/test edge split, so the model doesn't forget how to do the
    static task) and the 4-way cross-entropy.

**The exact loss, in full** — a weighted sum of three terms, computed every training step:

```
L = λ_static · [ BCE(static_pred_A, y) + BCE(static_pred_B, y) ]
  + λ_ce     · CrossEntropy(diff_logits, diff_label, weight = inverse_class_frequency)
  + λ_con    · [ L_infonce + L_margin ]
```

- **`λ_static · [BCE_A + BCE_B]`** — the auxiliary static task, computed independently for
  each condition against the *same* static edge labels (the graph structure and which
  edges are "real" doesn't change between conditions). This checks: does this condition's
  GNN branch still produce a usable embedding for the base task?
- **`λ_ce · CrossEntropy(...)`** — the 4-way classification loss.
  - Each of the 4 classes is weighted by the *inverse* of how often it appears in the
    training data.
  - "Neither" outnumbers the other three classes by anywhere from ~13x to essentially
    infinity (§3.1) — an unweighted loss would just learn to always predict "neither."
  - Inverse-frequency weighting forces the rare classes (often just a handful of real
    examples) to matter proportionally in the gradient.
- **`λ_con · [L_infonce + L_margin]`** — the contrastive term. It only ever touches edges'
  *embeddings*, not the raw classification logits, and only applies to edges the silver
  label calls "conserved" or condition-specific ("neither" edges carry no informative
  embedding-similarity signal by definition, so they're excluded entirely):
  - `L_infonce` (conserved edges only) — a standard in-batch InfoNCE loss. For a conserved
    edge *e*, the model computes a representation under condition A (`r_A[e]`, the
    elementwise product of the TF and target embeddings — the same combination rule the
    base model's "dot" decoder already uses) and under condition B (`r_B[e]`), and is
    trained to make `r_A[e]` and `r_B[e]` more cosine-similar to each other than `r_A[e]`
    is to *every other edge's* `r_B` in the same training batch. This is what "conserved"
    means at the representation level: the same edge should look like the same edge across
    conditions, and unlike everything else.
  - `L_margin` (A-specific and B-specific edges) — a hinge loss. `r_A[e]` and `r_B[e]` are
    pushed to be *dissimilar*, with a penalty only kicking in if their cosine similarity
    exceeds a margin (default 0.3) — the model isn't punished for making them arbitrarily
    different, only for making them too similar.
- **Weights used in every run reported below**: `λ_static = 1.0`, `λ_ce = 1.0`,
  `λ_con = 0.5`. Like all other hyperparameters in this project, these were **not tuned**
  — they're the values established as reasonable defaults during the design process, and
  no hyperparameter search was run this pass (see §4.2, item 7).

**Where this is implemented**:

- `src/models.py`'s `scDiffTransNet_GCN.forward_pair()` — produces `static_pred_A`,
  `static_pred_B`, `diff_logits`, `r_A`, `r_B`.
- `src/train_differential.py`'s `DiffTrainer.train()` — assembles and backpropagates the
  loss above every training step.

**One implementation detail worth knowing if re-running this code**:

- The differential-edge terms (`L_ce`, `L_infonce`, `L_margin`) are computed over a
  randomly-sampled mini-batch of candidate edges (capped at 4,096 by default,
  `--diff_batch_cap`) rather than the full candidate set every step.
- This was added partway through this project's Phase 2 run, after mESC (29,613
  candidates, ~6.5x hESC's 4,545) turned out to make every training step far more
  expensive than hESC's.
- The fix bounds that cost without changing what the model is being trained to do — every
  candidate edge is still visited over the course of an epoch, just not all at once.

### 2.2 There is no ground-truth differential label — so how do we train and evaluate at all?

This is the central methodological problem the whole project is built around solving.

- The reference networks give a static "this edge exists" label.
- Nothing in this data says "this edge's activity changed between 00h and 96h."
- So a **weak/silver label** has to be constructed statistically, and its own accuracy has
  to be separately checked (that's what Phase 3, §3.4, is for).

The method (`src/differential_labels.py`) is a classical **differential correlation test**
— the DiffCorr/Fisher-z lineage (§1.4.B), a standard, decades-old approach in this exact
subfield, not something invented for this project:

1. For each candidate edge (TF, target), compute the Spearman correlation between the TF's
   and target's expression **separately within each condition's cells** — `r_A` using only
   condition A's cells, `r_B` using only condition B's cells.
2. Fisher-transform each correlation (`arctanh`) and test two things:
   - Is `r_A` significantly different from zero (edge "active" in condition A)? Same for
     `r_B`.
   - Is the *difference* between `r_A` and `r_B` itself statistically significant (a proper
     two-sample test, not just "the numbers look different")?
   - This second test correctly catches a genuinely interesting biological pattern: a
     **sign flip** (an edge that looks activator-like in one condition and repressor-like
     in the other). A naive "`|r_A - r_B|` is big" threshold would miss this if the
     magnitudes happen to be similar.
3. All three sets of p-values get their own separate false-discovery-rate correction
   (Benjamini-Hochberg, q=0.05) — three independent corrections, because they test three
   different null hypotheses, not one pooled family.
4. A minimum effect-size floor (`|r| >= 0.3`) is required on top of statistical
   significance, so a technically-significant-but-tiny correlation (which happens easily
   with enough cells) doesn't get called "real."

**The decision table** — this is what actually assigns the 4-way label per edge:

| Significant in A? | Significant in B? | Significantly different (A vs B)? | Label |
|---|---|---|---|
| yes | yes | no | **conserved** — real correlation in both conditions, not meaningfully different |
| yes | no | yes | **A-specific** — real, active only in condition A |
| no | yes | yes | **B-specific** — real, active only in condition B |
| no | no | (either) | **neither** — no detectable signal in either condition |
| yes | yes | yes | *ambiguous* — significant in both, but the two correlations differ significantly (e.g. weak-positive-in-A vs strong-positive-in-B). Excluded from training, counted separately. |
| yes/no | no/yes | no | *inconsistent* — significant in one condition alone, but the direct A-vs-B comparison isn't significant (can happen near a significance boundary, especially with unequal sample sizes). Excluded, counted separately. |

- Only **conserved / A-specific / B-specific / neither** are trainable/evaluable classes.
- *Ambiguous*/*inconsistent* are excluded from training and reported as their own
  diagnostic counts (see the Phase 0 table in §3.1 — this matters a lot for hESC).

**What this means biologically, in plain terms**:

- An edge is called "A-specific" not because someone decided the gene looks different, but
  because there's a statistically defensible, effect-size-gated signal that the TF and
  target's expression move together in condition A and don't in condition B (or vice versa
  for B-specific).
- "Conserved" means the co-expression relationship is real and essentially unchanged.
- "Neither" is the default. Most candidate edges, most of the time, show no detectable
  co-expression signal at all in either condition, given realistically small single-cell
  sample sizes (tens to a few hundred cells per condition).
- This is expected, and is exactly why "neither" dominates the class counts almost
  everywhere (§3.1).

### 2.3 The three configurations being compared

| Config | What it is |
|---|---|
| **`full`** | The complete model described in §2.1. |
| **`posthoc_baseline`** (`src/posthoc_baseline.py`) | An adaptation of how the field currently does this — RegGAIN's procedure (§1.4.D). Train two **independent** static GNN models, one per condition (no sharing at all). Score every candidate edge under each. Run a Wilcoxon rank-sum test **per TF**, comparing that TF's edge scores across conditions. Collapse the result into a per-edge call. **Has no "neither" option by construction** — RegGAIN's real procedure never tests "is this edge active at all," only "did this TF's activity change" — so every edge gets called conserved, A-specific, or B-specific, never neither. This is a real, documented structural limitation, not a bug, and matters a lot for reading the results (§3.3). |
| **`naive_expression_baseline`** (`src/naive_baseline.py`) | No GNN, no scFM, no learned embeddings at all — a logistic regression on four numbers per edge (`r_A`, `r_B`, and the TF's and target's own mean-expression log-fold-change between conditions). Tests whether the graph+scFM machinery adds anything over gene-level statistics alone. |

---

## 3. How to interpret the numbers

### 3.1 Phase 0 — the labeling-yield dry run (does each pair even have data to learn from?)

Before building any training code, the silver-labeling pipeline (§2.2) was run on all 9
pairs, CPU-only, just to count how many edges land in each class:

| cell_type | pair | priority | candidates | conserved | A-spec | B-spec | neither | ambiguous | inconsistent |
|---|---|---|---|---|---|---|---|---|---|
| hESC | 00h_vs_96h | primary | 4,545 | 0 | 0 | 9 | 4,468 | 0 | 68 |
| hESC | 00h_vs_12h | secondary | 4,545 | 0 | 0 | 0 | 4,545 | 0 | 0 |
| hESC | 36h_vs_72h | secondary | 4,545 | 5 | 1 | 5 | 4,484 | 0 | 50 |
| mESC | 0h_vs_72h | primary | 29,613 | 32 | 71 | 175 | 28,359 | 6 | 970 |
| mESC | 0h_vs_12h | secondary | 29,613 | 46 | 20 | 129 | 28,333 | 2 | 1,083 |
| mESC | 24h_vs_48h | secondary | 29,613 | 222 | 35 | 11 | 27,281 | 2 | 2,062 |
| mHSC-E | LT_HSC_vs_Prog | primary | 11,557 | 279 | 15 | 810 | 9,277 | 18 | 1,158 |
| mHSC-E | LT_HSC_vs_HSPC | secondary | 11,557 | 9 | 16 | 0 | 11,153 | 0 | 379 |
| mHSC-E | HSPC_vs_Prog | secondary | 11,557 | 7 | 0 | 2,134 | 9,373 | 5 | 38 |

**The single most important number in this table**:

- hESC's `00h_vs_96h` — the pair designated as the *primary, largest-contrast* pair — has
  **zero** conserved and **zero** A-specific edges, and only 9 B-specific edges, out of
  4,545 candidates.
- `00h_vs_12h` has **literally nothing** in any of the four trainable classes.
- Root cause (verified directly, not assumed): at `00h`, the pluripotent starting
  population, essentially no candidate TF-target pair shows a detectable correlation at
  all. Only 25/4,545 edges reach `|r|>=0.3`, and the fraction passing the raw significance
  test is only ~4.75% — barely above the 5% you'd expect from pure chance.
- This means the statistical test correctly, honestly finds "no signal" in condition A for
  this pair — **it is not a bug in the labeling code.**
- Biological plausibility: a young, homogeneous, tightly-regulated pluripotent population
  can have flatter, less differentiated pairwise co-expression structure than a population
  that's actively differentiating (96h).
- mESC and mHSC-E don't have this problem nearly as badly — mESC's `0h_vs_72h` is the
  best-balanced pair of all nine.

**Why this matters for everything downstream**: any biological check performed against
hESC's `00h_vs_96h` or `00h_vs_12h` model is fundamentally limited by this — the model
cannot learn to call something "A-specific" if it has *zero* real training examples of that
class. Every hESC negative result later in this document traces back to this table, not to
a separate architecture failure.

### 3.2 Phase 1 — architecture verification (hESC 00h vs 96h)

Two separate questions get two separate verdicts here:

**Is the architecture itself sound? Yes.**

- The model trained end-to-end without errors.
- It converged, via early stopping, at epoch 64/300.
- Its auxiliary static link-prediction task — reusing exactly the base model's train/test
  split — scored AUROC 0.873/0.874 and AUPRC 0.613/0.616 for the two conditions.
- This is close to the existing single-condition baseline (AUROC 0.8874, AUPRC 0.6225 —
  §1.4.E), confirming the shared-adjacency, weight-tied Siamese design doesn't break the
  underlying static task it's built on top of.

**Does it recover the expected hESC biology? No** — and this is exactly what §3.1
predicts.

- POU5F1, NANOG, SOX2 (expected to skew 00h-specific) and SOX17, GATA4, GATA6 (expected to
  skew 96h-specific) all get predicted "neither" for essentially 100% of their out-edges.
- Given the training data had 0 A-specific and only 6 B-specific examples total for this
  pair, there was no signal to learn from.
- This is reported as a genuine failure, not explained away — but its root cause is a
  labeling-yield problem for this specific pair, not evidence the architecture can't learn
  4-way differential structure. Phase 3 (§3.4) directly tests that question with clean
  labels and gets a clearly positive answer.

### 3.3 Phase 2 — the core comparison (all 9 pairs)

**What "macro-F1" means, and why it's the headline metric**:

- Macro-F1 averages the F1 score of *each class equally*, regardless of how many examples
  that class has — unlike raw accuracy, which is dominated by whichever class has the most
  examples.
- Since "neither" dominates class counts almost everywhere (§3.1), a model that just
  predicts "neither" for everything would score high on accuracy but should score
  near-zero on the classes that actually matter.
- Macro-F1 is specifically designed to expose that — it's the plan's own designated
  headline metric for exactly this reason.

**What the class-count floor means**:

- A per-class AUROC/AUPRC is only reported if that class has at least 20 test examples.
- A class with e.g. 3 test examples gets its raw count reported instead of a statistically
  meaningless AUROC computed on 3 points.

**The core matrix**:

| cell_type | pair | `full` | `posthoc_baseline` | `naive_expression_baseline` |
|---|---|---|---|---|
| hESC | 00h_vs_96h | 0.9974 | 0.0000 | 0.9966 |
| hESC | 00h_vs_12h | 1.0000 | 0.0000 | 1.0000 |
| hESC | 36h_vs_72h | 0.9985 | 0.0000 | 0.8176 |
| mESC | 0h_vs_72h | 0.4392 | 0.0055 | 0.3541 |
| mESC | 0h_vs_12h | 0.7062 | 0.0039 | 0.3917 |
| mESC | 24h_vs_48h | 0.6259 | 0.0044 | 0.3884 |
| mHSC-E | LT_HSC_vs_Prog | 0.7708 | 0.0636 | 0.3701 |
| mHSC-E | LT_HSC_vs_HSPC | 0.9982 | 0.0000 | 0.8451 |
| mHSC-E | HSPC_vs_Prog | 0.8750 | 0.0000 | 0.6087 |

**Reading `full` vs `naive_expression_baseline` first** (the more informative comparison —
does graph+scFM structure add anything over gene-level statistics alone?):

- `full` beats naive by ≥5–10 macro-F1 points — the plan's own bar for a "positive
  result" — on **7 of 9 pairs**.
- All three mESC pairs: +31.5, +8.5, +23.7 points.
- All three mHSC-E pairs: +40.1, +15.3, +26.6 points.
- It *ties* naive on hESC's two nearly-all-"neither" pairs, simply because both methods
  are already near the ceiling (1.0) with no room to separate.
- hESC's `36h_vs_72h` — the one hESC pair with a *little* real signal (Phase 0: 5
  conserved, 1 A-specific, 5 B-specific) — does show `full` clearly ahead (+18.1 points).
- **Read together, this is a genuinely positive result**: the GNN+scFM machinery is
  adding real information over gene-level correlation/fold-change statistics alone, not
  just re-deriving the same statistic through a more expensive path (the specific negative
  scenario the plan explicitly warned about).

**Reading `full` vs `posthoc_baseline`**:

- `full` wins everywhere, often overwhelmingly.
- **Be careful reading this one.** `posthoc_baseline` structurally cannot predict
  "neither" at all (§2.3) — so on every single pair here, where "neither" is the dominant
  true class, it is *guaranteed* to score near-zero macro-F1, regardless of how good its
  actual condition-specific calls are.
- The honest reading: **"the jointly-trained model correctly handles the dominant class
  that the field's current post-hoc-testing practice structurally cannot."** That's a real
  and legitimate advantage (a per-edge tool that can't say "no change here" is much less
  useful in practice) — but it's a narrower claim than "the joint model is ~100x better at
  detecting condition-specific edges," which these numbers alone don't establish.

### 3.4 Phase 3 — the synthetic sanity check (is any of this real, or is it silver-label noise?)

Because the training/evaluation labels above are all statistically-derived (silver, not
ground truth), Phase 3 asks a sharper question: if we build a system where we *actually
know* the true 4-way label (by literally constructing it), does the trained model recover
it, and does the silver-labeling procedure agree with the truth?

**What was used, and why not GRouNdGAN**:

- GRouNdGAN (a causal-GAN simulator that generates realistic scRNA-seq data under a
  specified reference GRN — see §1.4.B/D) was the originally-proposed synthetic simulator
  in the design process.
- It was out of scope for this pass — a multi-day separate-codebase integration, as
  anticipated during planning.
- The documented fallback was used instead: a **parametric negative-binomial expression
  simulator**, seeded with real per-gene mean/dispersion from hESC's actual data and real
  TF/gene identities.
- One condition (`G_A`) equals a real reference sub-network. The other (`G_B`) is a
  deliberately, precisely rewired version — one TF loses 15% of its regulatory edges,
  another TF gains that many new ones — so the true label of every affected edge is known
  exactly by construction.

**Two real bugs were found and fixed while building this** (worth knowing if extending
Phase 3):

1. The expression data in this repo is `log2(count+1)`-transformed, not raw counts —
   fitting a count-distribution model directly to log-scale values badly distorts it.
2. Using the *real* hESC network's full topology to drive the simulated co-expression made
   "conserved" edges statistically indistinguishable from random unconnected gene pairs.
   Root cause: the real network has one mega-hub TF touching ~79% of the gene panel, and
   most targets have several independent regulators — both wash out any single edge's
   marginal correlation signal. The fix: a simplified, single-regulator-per-target
   synthetic network for the *injected* correlation structure specifically, while keeping
   real gene/TF identities and real per-gene mean/dispersion.

**Results, one severity (moderate, ~15%) × one cell type (hESC)**:

- **The trained model against the TRUE labels**:
  - Macro-AUROC (one-vs-rest, macro-averaged) = **0.880**.
  - Macro-F1 = 0.759, macro-AUPRC = 0.740.
  - This clears the plan's own stop-condition bar ("AUROC near chance would mean the
    architecture is not learning the differential signal at all" — 0.88 is nowhere near
    chance, 0.5).
  - **This is the cleanest evidence in the whole project that the architecture genuinely
    can learn 4-way differential structure**, when given labels that aren't pathologically
    thin.
  - Per-class breakdown: conserved 0.879, B-specific 1.000, neither 1.000, A-specific
    0.641 (the one weak spot — the confusion matrix shows most true A-specific test edges
    get misclassified as conserved, the hardest of the four distinctions since both
    require real signal in condition A, differing only in whether B *also* has it).
- **The silver-labeling procedure itself, against the TRUE labels**:
  - **97.2%** accuracy restricted to the 4 trainable classes.
  - 92.7% including the ambiguous/inconsistent rows (which have no analogue in the
    true-label space, so they count as automatic mismatches).
  - This is a strong calibration result — it means the DiffCorr/Fisher-z statistic, run
    completely unmodified, is a *trustworthy* proxy for ground truth on data structured
    like this synthetic system.

**Important caveat**:

- This synthetic system was deliberately built with a simplified
  (single-regulator-per-target) structure specifically to make the check clean and
  informative — the *real* hESC network's topology couldn't be used directly (see the bug
  fix above).
- So "97.2% calibration accuracy" is a statement about the labeling procedure's soundness
  under *favorable* conditions, not a guaranteed number for every real pair.
- Phase 0's hESC finding (near-zero signal in the `00h` condition specifically, §3.1)
  shows real data can be much less favorable than this synthetic system.

### 3.5 Named biological checks — pass/fail table, with the biological "why," not just the label

These are pre-registered, named checks — chosen and specified *before* looking at model
output, based on textbook/literature-documented biology for each system — run against
`full`'s predictions over the **entire** candidate edge space (not just the held-out test
split), so every named gene's full out-edge set could be inspected.

**hESC — pluripotency-to-definitive-endoderm handoff**:

| Check | Result | Biological interpretation |
|---|---|---|
| POU5F1/NANOG/SOX2 → 00h-specific | **FAIL** (0% A-specific, all 3 genes) | Directly explained by §3.1: this pair has zero A-specific training examples for the model to learn from. Not evidence these genes aren't really 00h-specific biologically — evidence the label supply for this pair is too thin to test the claim at all. |
| SOX17/GATA4/GATA6 → 96h/72h-specific | **FAIL** (0–4.5% B-specific) | Same root cause — very few B-specific training examples (9 for `00h_vs_96h`), and none of them happened to involve these specific genes. |
| EOMES → transiently 12h-specific | **FAIL** (0%) | `00h_vs_12h` has *zero* examples in any trainable class — this check was never learnable for this pair. |

**mESC — naive-pluripotency-circuit collapse**:

| Check | Result | Biological interpretation |
|---|---|---|
| NANOG/POU5F1/SOX2/KLF4/KLF2/ESRRB/TBX3/ZFP42/PRDM14 → 0h-specific | **FAIL** — a *real* failure, not a labeling-yield artifact (`0h_vs_72h` has 32/71/175 conserved/A-spec/B-spec examples, comfortably above the floor). All genes show <3% A-specific out-edges. Three genes (POU5F1, ESRRB, TBX3) show *more* B-specific than A-specific edges — the opposite of the textbook-predicted direction, though small counts relative to each gene's total out-edges. | The naive-pluripotency-circuit hypothesis (a coordinated, early, "0h-specific" signature) is not what the trained model recovered for these specific genes on this data — worth investigating further (§4.2, item 2), not treating as settled. |
| OTX2 → 72h-specific (formative marker) | **FAIL** (0%) | Same pattern — no signal recovered in the predicted direction. |

**mHSC-E — erythroid commitment** (the best-resolved cell type — one clear positive
finding alongside several inconclusive/negative ones):

| Check | Result | Biological interpretation |
|---|---|---|
| **GATA2 → GATA1** ("the GATA switch") | **PARTIAL PASS.** "neither" in `LT_HSC_vs_Prog` (fail). "**B-specific, 100% model confidence**" in `HSPC_vs_Prog` (pass). | **The strongest positive biological finding in the project.** `HSPC_vs_Prog` is *exactly* the pair predicted to show this classic erythroid-commitment switch most sharply (it's the specific step where progenitors commit to the erythroid fate) — and the model found it, at maximum confidence, without ever being told to look for it specifically. It *not* resolving in `LT_HSC_vs_Prog` (the bigger-contrast bookend pair) is itself informative: it suggests the GATA switch is a signature of the *specific commitment step*, not something that shows up gradually across the whole LT_HSC→Prog trajectory — a biologically sensible pattern, not a random miss. |
| GATA2's other out-edges (567), SPI1/PU.1 out-edges (672), GFI1 vs GFI1B (604/257), EPOR in-edges (11) | **Mostly FAIL** to confirm the predicted direction. Predicted: activity concentrated in LT_HSC/HSPC, fading toward Prog. Observed: mostly the opposite — heavy B-specific/"Prog-specific" skew. | Confounded by a structural fact from Phase 0: `HSPC_vs_Prog` has **zero** A-specific silver-training examples at all — the model cannot learn to predict A-specific for this pair under any circumstances, so its near-uniform "B-specific" calls for these genes reflect that label gap at least as much as anything gene-specific. GFI1B's 39%-"conserved" pattern in `LT_HSC_vs_Prog` is the one loose partial confirmation (roughly consistent with the predicted "conserved-active" pattern). |

---

## 4. What this means for the paper, and prioritized next steps

### 4.1 Where this stands, using the plan's own two-claim framing

A paper from this project would need to make two separable claims — a methodological one
and a biological one — and be scored against each honestly:

**Claim 1 — Methodological** (`full` beats current practice on this exact
data/codebase):

- A real pass against `naive_expression_baseline` — 7/9 pairs, often by a wide margin.
  This is the comparison that most cleanly answers "is the graph+scFM machinery doing
  something naive statistics can't."
- The comparison against `posthoc_baseline` technically passes everywhere but should
  **not** be oversold — the margin is substantially inflated by a structural feature (no
  "neither" call) of the adapted baseline, not purely differential-detection quality. Any
  paper draft should say so plainly, as this document does.

**Claim 2 — Biological** (recovers named, independently-documented biology):

- A genuine partial pass.
- One flagship, textbook, maximum-confidence, mechanistically-correctly-located finding
  (GATA2→GATA1 in `HSPC_vs_Prog`) is exactly the kind of result that would make this
  clearly paper-worthy.
- But it stands alongside honest, non-cherry-picked failures: all of hESC, mESC's
  naive-circuit check, most of mHSC-E's secondary checks.

**Overall verdict: positive-but-mixed, closer to positive than negative.**

- This is **not** a stop-condition result — that would require Phase 3's synthetic AUROC
  to be near chance (it's 0.88, clearly not).
- This is also **not** a weak "marginal win over naive" result — the margins on 6 of 7
  non-degenerate pairs are well above the 5–10-point bar for a positive result.
- The honest framing for a paper draft: *"the architecture is sound and adds real signal
  over gene-level statistics on well-powered pairs, and recovers one specific, well-known,
  citable piece of regulatory biology in exactly the predicted context — but real-data
  performance is capped by silver-label yield (severely for hESC, less so for
  mESC/mHSC-E), and several secondary biological checks did not confirm their predicted
  direction and should be investigated, not asserted."*

**Target venue**:

- "Contrastive GNN for GRN inference" is now a crowded mechanism (§1.4.A — at least 7
  papers 2025–2026).
- The closest comparable work (RegGAIN, BoostDiff, DiCoLo, Meta-TGLink, KEGNI) publishes
  in venues like *Advanced Science*, *Bioinformatics Advances*, *Genome Biology*, and
  RECOMB/ISMB proceedings — not top-tier ML venues.
- The realistic target for this work is a bioinformatics methods venue in that same tier,
  not a general ML venue, where the contrastive-GNN mechanism alone would not read as
  novel (see §1.4.C's honest novelty-scoping).

### 4.2 Concrete, prioritized next steps

1. **Run the deferred ablations now** — `no_scfm`, `no_grn`, `no_contrastive`, 9 runs on
   the 3 primary/bookend pairs. Deferred this pass per an explicit mid-task scope change,
   once the core matrix was already looking promising.
   - This is the highest-priority next step: the paper's ablation table is currently
     empty, and the core numbers above are good enough to justify the compute.
   - `run_phase2.sh ablations` already exists and is ready to run — see §6.
2. **Investigate the mESC naive-circuit-check failure specifically** (§3.5) — this is a
   well-powered pair (unlike hESC), so the failure is real and worth understanding, not
   just reporting. Concrete angles:
   - Check whether the *raw* differential-correlation signal (before the 4-way decision
     table) shows any directional trend for these genes even if it doesn't clear the
     significance+effect-size gate.
   - Check whether the naive-circuit genes' high out-degree (KLF4: 687, NANOG: 922) means
     their signal is diluted the same way Phase 3 found real-network multi-regulator
     convergence dilutes single-edge correlation (§3.4). If so, this may be a
     labeling-sensitivity issue rather than a model issue — testable by loosening
     `--corr_min_r` as a controlled ablation, not a silent change to the default.
3. **Extend Phase 3** to the deferred severities (mild, severe — currently only moderate
   ~15% was run) and to mESC/mHSC-E as seed systems (currently only hESC).
   - Directly strengthens the "is the architecture sound" claim with more evidence points.
   - Comparatively cheap — Phase 3 runs in tens of minutes, not hours.
4. **Port SAGE/GAT backbones** (the plan's designated robustness spot-check, 6 runs on
   the 3 primary pairs) — currently entirely unimplemented on this branch (only the GCN
   variant of `scDiffTransNet_*` exists). Needed before claiming the result isn't a
   GCN-specific artifact.
5. **Re-examine the `posthoc_baseline` comparison's framing** before writing it into a
   paper draft — either:
   - Explicitly split the reported metric into "neither-class handling" vs
     "condition-specific-class accuracy restricted to non-neither edges," so the
     comparison isn't dominated by a structural difference; or
   - Extend the baseline itself with a simple within-condition significance test
     (mirroring the `full` model's own sig_A/sig_B logic) so it *can* call "neither" and
     the comparison becomes more apples-to-apples.
   - Right now the numbers are correct but easy to over-read.
6. **If pursuing a real submission, acquire mHSC-GM and mHSC-L** — the other two branches
   of the hematopoietic lineage, an open data gap noted since the very start of this
   branch's work. These would let the GATA2→GATA1 finding be checked against the
   myeloid/lymphoid and megakaryocyte branches too — exactly the kind of internal-
   consistency evidence that would turn a "candidate finding" into a stronger one.
7. **Lower priority but easy: port `src/tuning/` from `summer26`.**
   - Not needed for this pass's fixed-hyperparameter runs.
   - Would let a real per-cell-type hyperparameter search happen before any of the above
     become a paper's final numbers.
   - This project has so far deliberately used one fixed hyperparameter set borrowed from
     the static baseline — permitted for this pass, but a paper would likely want it
     tuned properly.

---

## 5. Where the numbers above came from — reproducing or extending them

Quick reference for re-running or extending any part of this project (full detail is in
§6's file map):

- **Re-run Phase 0** (labeling yield, CPU-only, ~5 seconds for all 9 pairs):
  `python3.12 -m src.differential_labels`
- **Re-run one `full` training run**:
  `python3.12 -m src.train_differential --cell_type <hESC|mESC|mHSC-E> --condition_a <A> --condition_b <B> --diff_ablation full ...`
  (see `run_phase2.sh` for the exact fixed hyperparameters used throughout this project).
- **Run the whole Phase 2 matrix** (skip-logic means it's safe to re-run — it won't redo
  a pair/config that already has a `metrics.json`):
  `bash run_phase2.sh {full|posthoc|naive|ablations|all}`
- **Re-run Phase 3** (synthetic check, one severity/cell type takes tens of minutes):
  `python3.12 -m src.phase3_synthetic`
- **Aggregate results into a table**: `python3.12 -m src.aggregate_phase2`
- **Re-run the named biological checks** against any completed `full` run:
  `python3.12 -m src.biological_checks`

---

## 6. Map of the deliverables — where to look for what

| File / directory | What it is |
|---|---|
| `DIFFERENTIAL_GRN_BRIEF.md` | The original user-facing brief that kicked off the design process. |
| `LIT_REVIEW.md` | The literature review restated in full in §1.4 above. |
| `DIFFERENTIAL_GRN_PLAN.md` | The technical spec: exact formulas, the full evaluation protocol, the full experiment matrix, feasibility/risk notes, and the paper-worthiness framing. This document explains and interprets the plan in full; the plan is the authority on exact technical minutiae if anything here is ambiguous. |
| `work.md` | The terse, dated, chronological methodology log — every decision, bug, and finding as it happened. Read this for the *process*; read `explanation.md` for the *synthesis*. |
| `results.txt` | Terse, appendable, numbers-only log — every metric reported in this document, in the order it was produced. |
| `PROGRESS.md` | A live/backfilled run-status tracker generated by `src/progress_watch.py` during Phase 2's 27-run matrix. Static now that Phase 2 is complete; regenerate with `python3.12 src/progress_watch.py` if resuming more runs. |
| `src/differential_data.py` | The condition/barcode-prefix registry (§1.2) and the 9 chosen condition pairs with their stated rationale. |
| `src/differential_labels.py` | The DiffCorr/Fisher-z silver-labeling pipeline (§2.2) — `compute_differential_labels()`, the exact decision table, BH-FDR correction, and the deterministic `stratified_split()` used by every downstream script so all three configs are evaluated on the *identical* held-out test edges. Also the Phase 0 dry-run entry point. |
| `src/models.py` | `scDiffTransNet_GCN` — the architecture (§2.1), including the ablation dispatch (`full`/`no_scfm`/`no_grn`/`no_contrastive`). |
| `src/train_differential.py` | `DiffTrainer` — the training loop, evaluation (macro-F1 + class-count-floor-gated per-class AUROC/AUPRC), checkpointing, and `predict_all_candidates()` (used for the biological checks in §3.5). |
| `src/posthoc_baseline.py` | The RegGAIN-adapted per-TF Wilcoxon baseline (§2.3, §1.4.D). |
| `src/naive_baseline.py` | The logistic-regression baseline (§2.3) — standalone, CPU-only, no GNN. |
| `src/phase3_synthetic.py` | The parametric negative-binomial synthetic simulator, rewiring construction, and the calibration check (§3.4). |
| `src/biological_checks.py` | Runs the named checks in §3.5 against any completed `full` run's `full_candidate_predictions.csv`. |
| `src/aggregate_phase2.py` | Walks `out/differential/*/*/*/metrics.json` and prints the core-matrix summary table (§3.3). |
| `run_phase2.sh` | The driver script for the Phase 2 matrix. |
| `out/differential/{cell_type}/{pair}/{config}/` | Per-run outputs: `metrics.json` (the numbers), `ckpt/` (model checkpoint + args, `full`/`posthoc_baseline` only), `log.txt` (training log), `full_candidate_predictions.csv` (per-edge predictions over the *entire* candidate space, `full` only — gitignored like all `*.csv`, regeneratable from the checkpoint). |
| `data/{cell_type}/TFs+500/differential/{A}_vs_{B}_labels.csv` | Cached silver labels per pair (gitignored, deterministic, regenerated automatically if missing). |
