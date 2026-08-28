# Differential/Contrastive GRN — Technical Plan (Agent B)

Input: `DIFFERENTIAL_GRN_BRIEF.md`, `LIT_REVIEW.md` (Agent A), and a direct read of
`src/models.py`, `src/train.py`, `src/args.py`, `src/utils.py`, `src/cross_modal_attention.py`
on the `differential-grn` branch, plus `src/build_panda.py` / `src/tuning/*` read via
`git show summer26:...` since **this branch predates the PPI/PANDA/tuning work and does not
contain those files** (confirmed: `differential-grn` branches from `summer26@8a3f58a`; the
tuning infra and PANDA script were added afterward, on commits `e9e4544`..`b77d0ce`). Numbers
below (cell counts, gene counts, reference-network sizes, marker-gene presence and out-degree)
were checked directly against the data in this repo rather than assumed from the brief —
**with one correction**: the mHSC-E cell counts in an earlier draft of this plan were wrong (an
apparent transposition toward the original Nestorowa et al. published population sizes rather
than this repo's actual `TFs+500`/`TFs+1000` data files) and have been recomputed directly from
the CSV headers for this revision (§2, §7). The lesson generalizes: every number in this
document should be treated as re-verifiable against the raw files, not as already infallible
because it was "checked" once.

---

## 1. Core idea & precise novelty claim

**Idea.** A single Siamese, weight-tied extension of `scTransNet_{GCN,SAGE,GAT}` that takes
**two conditions of the same cell type at once** (two time points or two differentiation
stages), encodes each condition through the *same* GNN + scFM dual-branch encoder over a
*shared* graph structure but *condition-specific* node features, and outputs a **jointly
trained, discrete 4-way label per candidate TF→target edge** — conserved / A-specific /
B-specific / neither — via a dedicated differential-edge decoder trained end-to-end with a
combined static-link + 4-way cross-entropy + condition-contrastive objective, where the
contrastive positive/negative pairing is supervised by the differential label itself (edges
called "conserved" are pulled together across conditions in embedding-difference space; edges
called condition-specific are pushed apart).

**Structural differences vs. the closest prior work** (per `LIT_REVIEW.md` §1.A, §3):

1. **Joint vs. post-hoc.** RegGAIN (the one paper training a contrastive GNN on this exact
   BEELINE hESC/mESC/mHSC-E data and this exact Chu et al. hESC time course) trains **one
   static model, applies it independently to each time point, then runs a Wilcoxon
   rank-sum test on edge scores post-hoc** to call differential TFs. Our model ingests both
   conditions in one forward pass and is trained against a differential objective directly —
   there is no independently-trained-then-compared step anywhere in the pipeline. This is
   the single biggest mechanistic difference and is validated head-to-head in §6 (the
   `posthoc_baseline` ablation adapts RegGAIN's differential-testing procedure — a Wilcoxon
   rank-sum test on independently-inferred static edge scores — to produce per-edge calls
   inside this codebase for a fair, same-data, same-architecture comparison; RegGAIN's own
   paper never poses a per-edge question, only a per-TF ranking, so this is an adaptation of
   its statistical procedure, not a literal reimplementation of its reported output — see
   §5.1 for the exact adaptation).
2. **Discrete 4-way trained label vs. qualitative/continuous.** RegGAIN's rewiring claim is
   a p-value on out-degree distributions with no discrete conserved/specific/neither
   classification and no accuracy metric of its own. jointGHS (closest classical analogue)
   estimates a continuous shared-vs-specific shrinkage variable via a Bayesian graphical
   model, not a supervised/contrastive classifier with a discrete training target. Nothing
   found in the GRN-inference literature trains a 4-class edge classifier end-to-end for this
   task — though §1.1 below checks this claim against the broader, non-biological dynamic-graph
   literature too, and narrows it accordingly.
3. **Contrastive axis is condition, not augmentation.** RegGAIN's InfoNCE loss pairs two
   *augmented views of one static graph* (edge-drop / hub-aware perturbation) as its
   positive/negative signal — the augmentation is artificial. GP-DHT, BRIDGE, BRIDGE-GRN,
   SupGCL, GCLink, DGCRL are the same pattern: contrast is over augmented/perturbed views of
   a single condition. Here, the positive/negative pair for the *same TF-target edge* is
   **two real biological conditions** (e.g. 0h vs. 96h expression), and the sign of the pair
   (pulled together vs. pushed apart) is supervised by the weak/silver differential label,
   not by an augmentation scheme. This is a materially different notion of what "contrastive"
   means in this literature.
4. **scFM+GNN dual-branch fusion has never been used in a differential/cross-condition task.**
   scRegNet's own architecture (this repo's base method) is the only scFM+GNN fusion method
   in the review; every other contrastive-GNN GRN paper uses expression-only or
   expression+prior-network features, never a foundation-model gene-identity embedding. We
   keep the scFM branch condition-*independent* (see §4) specifically so that any observed
   differential signal is attributable to the GNN branch's condition-specific structure, and
   make that an explicit, reported ablation rather than an incidental design choice.

### 1.1 Cross-field check: is this mechanism shape novel even outside GRN inference?

The structural differences above were checked against GRN-inference-specific literature only
(`LIT_REVIEW.md`'s scope). Before finalizing the novelty claim, this revision checked the
mechanism shape — paired-snapshot Siamese/weight-tied encoders, discrete per-edge
appear/disappear/persist/absent classification, contrastive learning across two graph states —
against the general (non-biological) **temporal/dynamic graph representation learning**
literature, which is large and predates essentially all of the GRN-specific papers found in
round 1:

- **EvolveGCN, DySAT, TGN, ROLAND** are the standard architectures for dynamic-graph node/edge
  classification and link prediction, and all support an "edge appears/disappears" framing in
  principle (next-step link existence forecasting is the dominant benchmark task for this
  family). However, all four are built for **many-snapshot or continuous-time streaming
  graphs** (dozens to thousands of timestamps — social interactions, transactions,
  recommender events) and are trained/evaluated to forecast the *next* state of an
  ever-evolving graph, not to jointly ingest exactly **two designated conditions** and emit a
  single discrete 4-way transition-type label per edge. None of the four frame their output
  as a conserved/A-specific/B-specific/neither-style typed classification between two named
  states.
- **DyGnROLE** (arXiv 2602.23135, Bonnet & Rei) is the closest general-ML paper found doing
  "edge classification on dynamic graphs" with a role-aware (source/destination-asymmetric)
  architecture and a directional contrastive pretraining objective — structurally close in
  spirit (edge classification + contrastive pretraining on a graph with directional
  source/target roles, much like TF→target edges). It still targets classification of edges
  *as they are observed in a continuous interaction stream* (e.g., what type of interaction is
  this, at the time it happens), not a discrete transition-type label comparing two designated
  snapshots, and it has no biological or foundation-model-embedding application.
- **Graph-snapshot change-point detection via Siamese GNNs** (e.g. "Graph similarity learning
  for change-point detection in dynamic networks," arXiv 2203.15470, and the related Laplacian
  change-point line of work) is the structurally closest match of everything found: a Siamese,
  weight-tied GNN encoder compares **pairs of graph snapshots** directly, exactly like this
  plan's two-condition encoder. The output, however, is a **whole-graph similarity/change-point
  score**, not a per-edge discrete label, and none of this line uses real node features
  (foundation-model or otherwise) or biological data.

**Conclusion, stated explicitly rather than left implicit**: the core mechanism shape — a
weight-tied paired-snapshot encoder plus a discrete per-edge state-transition classifier plus a
contrastive objective — is a **recognized general pattern in dynamic-graph learning, not a
novel invention of this plan**. What has not been found anywhere, in either the GRN-specific
literature (round 1) or this general dynamic-graph-learning literature (this pass), is the
*combination* actually proposed here: (a) applied to gene regulatory network inference at all
— every dynamic-graph paper found operates on generic interaction/social/transaction graphs, none
on TF-target regulatory graphs; (b) fused with scFM gene-identity embeddings — no dynamic-graph
paper found uses a foundation-model node feature of any kind; (c) validated against named,
literature-documented regulatory biology. This narrows the honest novelty claim: **this paper's
contribution is an application-plus-fusion contribution (a known general dynamic-graph
mechanism class, newly combined with scFM fusion and applied — jointly-trained, for the first
time in the GRN literature — to differential GRN inference, with biological grounding), not a
claim to have invented a new mechanism class in graph machine learning.** This is a materially
narrower and more defensible framing than the original draft of this section implied, and
should be stated this way in any resulting paper's related-work section — a reviewer familiar
with EvolveGCN/DySAT/TGN/ROLAND or graph change-point detection would otherwise reasonably
object to an unqualified "first jointly-trained differential edge classifier" claim.

Not claimed as novel: contrastive learning for GRN inference in general (crowded, ≥7 papers
2025–2026 per the lit review); "shared vs. specific edges" as a *concept* (jointGHS, DINGO,
CroCoNet); differential-network statistics as a construct (DiffCorr/DINGO/BoostDiff/DCI
lineage, decades old); **and, per §1.1 above, the paired-snapshot Siamese + discrete
edge-transition-classification + contrastive-loss mechanism shape itself**, which is a known
pattern in the general dynamic-graph-learning literature. The novelty is the **combination**:
that general mechanism shape, fused with scFM gene embeddings, jointly trained (not post-hoc)
for the first time on GRN data specifically, with a discrete 4-way supervised target and
biological validation, applied to within-species time/stage pairs — none of which co-occur in
any paper found in either literature.

---

## 2. Data & condition pairs

All three usable cell types are **already in the repo**, need no new acquisition, and encode
condition membership directly in the cell barcode (verified by direct inspection of
`BL--ExpressionData.csv` headers, not just the brief's claim):

| Cell type | Source | Stages (cell counts, `TFs+500`) | Genes / TFs (`TFs+500`) | Reference net edges / unique source-TFs |
|---|---|---|---|---|
| **hESC** | Chu et al. 2016, H9 definitive-endoderm time course | `00h`(92) `12h`(102) `24h`(66) `36h`(172) `72h`(138) `96h`(188) | 910 genes / 410 TFs | 4,545 / **34** |
| **mESC** | Hayashi et al. 2018 RamDA-seq, 2i/LIF-withdrawal time course | `0h`(90) `12h`(68) `24h`(90) `48h`(82) `72h`(91) | 1,120 genes / 627 TFs | 29,613 / **88** |
| **mHSC-E** | Nestorowa et al., flow-sorted hematopoietic stages, erythroid branch | `LT_HSC`(141) `HSPC`(384) `Prog`(546) | 704 genes / 205 TFs | 11,557 / **29** |

Note the reference network is **TF-concentrated**: only 34/410, 88/627, 29/205 TFs ever appear
as a source in the curated reference network per cell type. This directly bounds which TFs can
have a differential-edge claim at all (see §3, §7) — stated up front rather than discovered
late.

**Condition pairs chosen (3 per cell type, not arbitrary):**

- **hESC** — definitive endoderm differentiation is a directed, well-characterized process
  (pluripotency exit → mesendoderm induction → definitive endoderm commitment):
  - `00h` vs `96h` (**primary/bookend**): pluripotent hESC vs. committed definitive endoderm —
    largest biological contrast, largest two stage-pools (92, 188 cells) for statistical power.
  - `00h` vs `12h` (early transition): captures the fast initial exit from pluripotency /
    onset of mesendoderm induction — tests whether the model detects transient early rewiring,
    not just any-vs-96h drift.
  - `36h` vs `72h` (late commitment): mesendoderm/mesoderm progenitor → definitive endoderm —
    a mid-trajectory transition, deliberately *not* including the two most different-looking
    endpoints, to test that the model isn't just learning "distance from t=0."
- **mESC** — exit from **naive** pluripotency toward a primed/formative state under 2i/LIF
  withdrawal (this is *not* a directed-lineage system like hESC; it is a general naive-circuit
  collapse, see §5 for why this changes the marker list):
  - `0h` vs `72h` (**primary/bookend**): naive ESC vs. most-differentiated time point sampled.
  - `0h` vs `12h`: immediate LIF/2i-withdrawal response — the naive circuit (Klf4/Klf2/Esrrb/
    Tbx3/Zfp42/Prdm14) begins collapsing within hours, this pair targets that fast onset.
  - `24h` vs `48h`: mid-transition, formative markers (Otx2) becoming dominant.
- **mHSC-E** — LT_HSC/HSPC/Prog is **not** a time course but a flow-sorted
  differentiation-potential gradient within the erythroid branch of Nestorowa et al.'s data
  (long-term HSC → stem/progenitor → erythroid-committed progenitor). All three stages are
  comfortably powered (141/384/546 cells — in fact the best-powered cell type of the three
  once the correct counts are used, see §7):
  - `LT_HSC` vs `Prog` (**primary/bookend**): most primitive vs. most committed state sampled
    (141 vs. 546 cells).
  - `LT_HSC` vs `HSPC`: early loss-of-stemness step (141 vs. 384 cells).
  - `HSPC` vs `Prog`: the erythroid-commitment step itself (384 vs. 546 cells) — where the
    GATA2/GATA1 and PU.1/GATA antagonisms (§5) are expected to be sharpest.

9 pairs total, each with a stated biological rationale for *why that pair* over any other
adjacent pair, not "all pairwise combinations because they exist."

---

## 3. Ground-truth / weak-supervision strategy for the differential label

No time/stage-resolved reference network exists for any of the three cell types (confirmed:
`BL--network.csv` has one static edge list per cell type, no condition column). Per
`LIT_REVIEW.md` §4–5, the field's standard answer when this is true is a **three-tier
combination** (synthetic ground truth + statistical silver label + biological enrichment), not
any single tier alone. This plan adopts all three; this section specifies Tier 2 (the training
label) in full technical detail, since it is the part the brief flagged as "must be designed."

### 3.1 Candidate edge space (the reference-network gate)

A TF→target pair `(i, j)` is only eligible for a differential label if `Gene1=i, Gene2=j` is a
positive edge in that cell type's `BL--network.csv`. This is not a convenience filter — it is
the defensible choice because (a) it restricts the ~410×910 (hESC) or ~627×1120 (mESC) or
~205×704 (mHSC-E) full TF-target product space down to edges with independent
ChIP-seq/curated-DB support, so a "differential" call is never manufactured purely from noise
in an implausible pair, and (b) it is exactly the same candidate-edge convention
`scRegNet`'s own `Train_set.csv`/`Test_set.csv` already use for the static task, so the
differential task's positive-label space is a strict subset of what the base model was already
validated against. Candidate counts: 4,545 (hESC), 29,613 (mESC), 11,557 (mHSC-E) per cell
type (same edges reused across all 3 condition pairs of that cell type).

### 3.2 The statistic: DiffCorr-style Fisher z-test, condition-restricted

For each candidate edge `(TF=i, target=j)` and a given condition pair `(A, B)`:

1. Using the **z-scored** expression matrix (`load_data.data_normalize`, already computed
   per-gene by the existing pipeline — no new normalization code needed), restrict cells to
   condition A's barcodes and compute the Spearman correlation `r_A = corr(x_i, x_j)` over
   condition A's `n_A` cells; independently compute `r_B` over condition B's `n_B` cells.
   Spearman (not Pearson) is chosen because scRNA-seq expression is zero-inflated/non-normal
   even after z-scoring, and Spearman is the standard robust choice in this exact
   classical-differential-network literature (DiffCorr supports both; Spearman is the safer
   default given the modest, sometimes <100-cell, per-condition sample sizes here — see §7).
2. Fisher z-transform: `z_A = arctanh(r_A)`, `z_B = arctanh(r_B)`.
3. **Within-condition correlation significance** (is this edge even active in condition X):
   standard test for `r_X != 0` via `z_X * sqrt(n_X - 3) ~ N(0,1)` under H0, giving p-values
   `p_A`, `p_B`.
4. **Differential test** (Fisher's classical two-sample z-test, the exact DiffCorr statistic):
   `Z_diff = (z_A - z_B) / sqrt(1/(n_A-3) + 1/(n_B-3))`, two-tailed p-value `p_diff` from the
   standard normal. This correctly captures both magnitude changes and **sign flips**
   (activator-like → repressor-like correlation, or vice versa) as strongly differential even
   when `|r_A| ≈ |r_B|` — a subtlety worth stating explicitly since it is a real biological
   mode (co-expression sign changes are reported in the differential-network literature) and a
   naive `|r_A - r_B|` threshold would miss it.
5. **FDR control**: Benjamini-Hochberg at `q=0.05`, applied *separately* to each of the three
   p-value vectors (`p_A` across all candidate edges, `p_B` across all candidate edges,
   `p_diff` across all candidate edges) per `(cell_type, condition pair)` — three independent
   BH corrections, not one pooled correction, since they test different null hypotheses.
6. **Minimum-effect-size gate**: in addition to BH significance, require `|r_X| >= 0.3`
   (a documented, ablatable hyperparameter — `--corr_min_r`, default 0.3) for "significant
   correlation in condition X," to avoid calling a `p<0.05` but `|r|=0.05` relationship
   biologically meaningful given the sample sizes involved.

### 3.3 4-way label construction (exact decision table)

Let `sig_A = (BH_p_A < 0.05) AND (|r_A| >= 0.3)`, `sig_B` symmetric, `sig_diff = (BH_p_diff <
0.05)`.

| `sig_A` | `sig_B` | `sig_diff` | Label |
|---|---|---|---|
| T | T | F | **conserved** |
| T | F | T | **A-specific** |
| F | T | T | **B-specific** |
| F | F | any | **neither** |
| T | T | T | *ambiguous* — both conditions show significant, individually-real correlation, but the magnitude/sign differs significantly. Excluded from training and reported separately as its own diagnostic count (expected to be a small fraction; forcing it into A- or B-specific by "larger `\|r\|`" would be an unjustified tie-break, so it is dropped rather than guessed). |
| T | F | F | *inconsistent* — significant in A only but the direct A-vs-B test did not reach significance (can happen near the significance boundary with unequal `n_A`/`n_B`). Excluded from training, reported as a diagnostic count; not forced into a class. |
| F | T | F | *inconsistent* — symmetric case, excluded. |

This is a hard, auditable decision table (not "use a stats test" hand-waved) with two
explicitly named excluded categories reported as counts rather than silently discarded or
force-assigned — this is the honesty check the brief and lit review both call for.

### 3.4 Why this is defensible, not hand-wavy

- It is not an invented method: DiffCorr (Fukushima 2013) is the standard, still-maintained
  tool for exactly this class of two-condition differential correlation test, and DINGO,
  Sc-compReg, BoostDiff, DCI all validate on simulated ground truth then apply an analogous
  statistic to real data with FDR control as their real-data label — this plan follows that
  exact precedent (`LIT_REVIEW.md` §1.B, §4.3).
- The reference-network gate (§3.1) means the silver label is never asked to invent an edge
  from nothing — it only ever *resolves the condition-specificity* of an edge whose existence
  already has independent (ChIP-seq/DB) support.
- The label's own confidence is stated (BH q=0.05, effect-size floor) rather than presented as
  ground truth — every trained/reported number downstream should be understood as "trained
  against a labeled proxy with q=0.05 FDR control," and that caveat is carried into §5/§8.
- **Tier 1 (§5, §6)** — a fully synthetic GRouNdGAN experiment with *actually known* 4-way
  labels — exists specifically to calibrate how much to trust this silver-labeling procedure
  before trusting it on real data, exactly mirroring how BoostDiff/DCI/Sc-compReg validate
  their statistics on simulated data first.

---

## 4. Architecture

### 4.1 What stays the same

- The GNN branch's **graph structure is shared, not condition-specific.** Both conditions of
  a pair are encoded over the *same* adjacency, built exactly as today
  (`scRNADataset.Adj_Generate` over `Train_set.csv` positive edges from the single static
  reference network). This is a deliberate resolution of the open question the brief raises
  in §5: a condition-specific adjacency would have to be built from the same weak
  differential-correlation signal we are trying to predict, which is circular (you cannot use
  the label-construction statistic to build the model's own input structure and then claim the
  model independently recovered it). Sharing the structure and varying only node features
  keeps the differential model comparable to, and a genuine extension of, the base scRegNet
  static task, and keeps the label leakage story clean.
- The scFM branch stays **condition-independent** — same per-gene embedding regardless of
  time point/stage, exactly as today (Geneformer/scGPT/UCE embeddings are gene-identity, not
  per-condition). This is intentional (§1, point 4) and is turned into an explicit ablation
  (`no_grn`, §6) that is *expected* to perform near chance on the differential task by
  construction — stated up front so it reads as a designed sanity check, not an oversight.
- `decode()` (dot/cosine/MLP), the `IterativeCrossModalFusion` module, and the per-condition
  MLP tf/target projection layers are all **weight-tied (Siamese)** across the two conditions
  — the same learned parameters process condition A's and condition B's embeddings. This is
  what makes it a real joint/shared model rather than two independent models with a shared
  initialization.

### 4.2 What changes — new module, concrete shapes

Today, `scTransNet_GCN.forward(x, adj, train_sample, llm_emb)` takes one expression feature
matrix `x` of shape `(num_genes, input_dim)`, where `input_dim = num_cells` **pooled across all
time points/stages of that cell type** (`feature2 = load_data(data_input).exp_data()` — this
was already confirmed by reading `src/utils.py`: `data_input` is genes × cells, so each gene's
raw "feature" fed to the first `GCNConv` is literally its per-cell expression vector, one
column per cell, standardized per-gene). This is the key structural fact that makes the
extension straightforward: **condition-specific node features already exist in the raw data**
— they only need to be sliced by condition instead of pooled.

New class, e.g. `scDiffTransNet_GCN` in `src/models.py` (SAGE/GAT variants follow identically
since they reuse the same `encode()`/`decode()` pattern):

```
__init__(self, input_dim_A, input_dim_B, args, gene_dim, device):
    # input_dim_A = n_cells_A, input_dim_B = n_cells_B  (different per condition, in general)
    self.proj_A = nn.Linear(input_dim_A, args.gnn_dim_hidden)   # condition-specific
    self.proj_B = nn.Linear(input_dim_B, args.gnn_dim_hidden)   # condition-specific
    self.convs  = [GCNConv(args.gnn_dim_hidden, ...), ...]      # SHARED / weight-tied
    self.cross_modal_fusion = IterativeCrossModalFusion(...)    # SHARED (unchanged)
    self.layers = [...]                                         # SHARED tf/target MLP (unchanged)
    self.diff_head = nn.Sequential(
        nn.Linear(6 * mlp_out_dim, args.diff_head_hidden), nn.LeakyReLU(),
        nn.Dropout(args.dropout), nn.Linear(args.diff_head_hidden, 4))
```

`x_A: (num_genes, n_cells_A)`, `x_B: (num_genes, n_cells_B)` — only the **two condition-specific
input projections `proj_A`/`proj_B` are not weight-tied** (they cannot be: the raw cell counts
differ per condition by construction, so there is no shared-shape input to a single tied
layer). Everything from the first `GCNConv` onward is weight-tied.

```
forward_pair(self, x_A, x_B, adj, llm_emb, static_edges, diff_edges):
    h_A = self.encode(self.proj_A(x_A), adj)          # (num_genes, gnn_out_dim)
    h_B = self.encode(self.proj_B(x_B), adj)          # SAME conv weights as h_A
    fused_A = self.cross_modal_fusion(llm_emb, h_A)   # (num_genes, fused_dim), llm_emb SHARED
    fused_B = self.cross_modal_fusion(llm_emb, h_B)
    tf_A, tgt_A = self._mlp(fused_A), self._mlp(fused_A)   # SAME self.layers, as today
    tf_B, tgt_B = self._mlp(fused_B), self._mlp(fused_B)

    # Auxiliary static task, reusing the SAME Train_set/Test_set edge indices as the
    # existing baseline task, scored once per condition (no new sampling logic needed):
    static_pred_A = self.decode(tf_A[static_edges[:,0]], tgt_A[static_edges[:,1]])
    static_pred_B = self.decode(tf_B[static_edges[:,0]], tgt_B[static_edges[:,1]])

    # Differential 4-way head, over the candidate edges from §3:
    i, j = diff_edges[:,0], diff_edges[:,1]
    diff_feat = torch.cat([
        tf_A[i], tgt_A[j], tf_B[i], tgt_B[j],
        (tf_A[i] - tf_B[i]).abs(), (tgt_A[j] - tgt_B[j]).abs(),
    ], dim=1)                                          # (num_diff_edges, 6*mlp_out_dim)
    diff_logits = self.diff_head(diff_feat)             # (num_diff_edges, 4)

    # Edge representations for the contrastive term (elementwise product, same
    # combination rule as the existing 'dot' decode mode):
    r_A = tf_A[i] * tgt_A[j]                             # (num_diff_edges, mlp_out_dim)
    r_B = tf_B[i] * tgt_B[j]
    return static_pred_A, static_pred_B, diff_logits, r_A, r_B
```

### 4.3 Loss

```
L = λ_static * (BCE(static_pred_A, y_A) + BCE(static_pred_B, y_B))
  + λ_ce     * CrossEntropy(diff_logits, diff_label, weight=inverse_class_freq)
  + λ_con    * (L_infonce + L_margin)
```

- `L_infonce`: standard in-batch InfoNCE over `conserved`-labeled edges only:
  `-log[ exp(sim(r_A[e], r_B[e]) / τ) / Σ_e' exp(sim(r_A[e], r_B[e']) / τ) ]`, `sim` = cosine,
  `τ = args.contrastive_temperature` (default 0.1). In-batch negatives are other edges' B-side
  representations — this naturally includes both true `neither` edges (easy negatives) and
  `A-specific`/`B-specific` edges (hard negatives), which is exactly the intended contrastive
  structure: conserved edges should look similar across conditions, everything else should not.
- `L_margin`: `Σ_e∈{A-specific,B-specific} max(0, sim(r_A[e], r_B[e]) - margin)`, explicit hinge
  pushing apart pairs the silver label says are condition-specific (`margin` default 0.3).
  `neither`-labeled edges are excluded from both contrastive terms (their signal is by
  definition weak/absent in both conditions, so pairing them is not informative either way).
- `λ_static, λ_ce, λ_con` are new args (`--diff_static_weight`, `--diff_ce_weight`,
  `--diff_contrastive_weight`; defaults 1.0/1.0/0.5), tunable via the ported HP-search infra
  (§7).

### 4.4 `train.py` / `args.py` changes

- New `src/train_differential.py` (`DiffTrainer`, parallel to `Trainer`, does not touch
  `train.py` — the static baseline stays runnable unmodified for direct before/after
  comparison). `_prepare_data()` loads `BL--ExpressionData.csv` once, splits columns by a
  per-cell-type barcode-prefix registry (`src/differential_data.py`, e.g.
  `{"hESC": {"00h": "H9_00hb4s", "12h": "H9_12h", ..., "96h": "H9_96h"}, "mESC": {"0h":
  "RamDA_mESC_00h", ...}, "mHSC-E": {"LT_HSC": "LT_HSC", "HSPC": "HSPC", "Prog": "Prog"}}`,
  matched against the exact prefixes verified in §2), builds `X_A`/`X_B` via the existing
  `load_data(...).exp_data()` called independently per condition subset (so each condition
  gets its own per-gene z-scoring, not one pooled scaler), builds the shared `adj` exactly as
  `Trainer._prepare_data()` does today, and calls a new `src/differential_labels.py` module
  implementing §3's pipeline, caching results to
  `data/{cell_type}/TFs+{num_TF}/differential/{A}_vs_{B}_labels.csv` (columns: `TF, Target,
  r_A, r_B, p_A, p_B, p_diff, label`) so the (deterministic, CPU-only) statistical step is
  never recomputed per training run.
- New args in `src/args.py`: `--condition_a`, `--condition_b` (string keys into the registry),
  `--diff_static_weight`, `--diff_ce_weight`, `--diff_contrastive_weight`,
  `--contrastive_temperature`, `--contrastive_margin`, `--corr_method {pearson,spearman}`
  (default `spearman`), `--corr_min_r` (default 0.3), `--fdr_q` (default 0.05),
  `--diff_label_mode {silver, groundgan}` (§5), `--diff_ablation {full, no_scfm, no_grn,
  no_contrastive, posthoc_baseline}` (§6).
- Evaluation loop mirrors `Trainer.train()`'s existing checkpoint-on-improvement / patience
  pattern, but tracks macro-F1 over the diff-test split (not AUROC alone) as the
  early-stopping criterion, since raw accuracy would be dominated by the `neither` majority
  class (§7).

---

## 5. Evaluation protocol

### 5.1 Quantitative differential metric

- Candidate edges labeled by §3 (excluding `ambiguous`/`inconsistent`) are split
  train/test (e.g. 70/30, **stratified by class** so the rare `A-specific`/`B-specific`
  classes are represented in both splits — with only 4,545–29,613 candidate edges and 3–4
  active classes, unstratified random splitting risks a test fold with near-zero minority-class
  examples).
- Report **per-class one-vs-rest AUROC/AUPRC** (4 classes), **macro-F1**, and the full 4×4
  confusion matrix, per condition pair, per cell type. Macro-F1 (not accuracy) is the primary
  headline number specifically because `neither` will dominate class counts (§7).
  **Minimum class-count floor for reporting**: a class's one-vs-rest AUROC/AUPRC is only
  reported for a given condition pair if that class has **≥20 test examples** in that pair's
  diff-test split; below that floor, report the raw count (e.g. "A-specific: 6 test edges,
  AUROC not reported") rather than a per-class number computed on too few examples to be
  meaningful. Given the candidate-edge counts (4,545–11,557–29,613) and the strict BH+
  effect-size gating in §3.3, some non-bookend pairs are expected to produce single-digit
  `A-specific`/`B-specific` test counts — this floor prevents a noisy few-example AUROC from
  being read as a real number in the results table. Macro-F1 for a pair is computed only over
  classes that clear the floor for that pair, with the excluded classes and their counts stated
  alongside it.
- Compare against two baselines run on this exact codebase and exact data, not just cited from
  other papers: (a) **`posthoc_baseline`** — two independently trained *static*
  `scTransNet_GCN` models (today's unmodified `train.py`, one run per condition). RegGAIN's own
  reported procedure is a Wilcoxon rank-sum test on **per-TF** out-degree/edge-score
  distributions, used to rank TFs by differential activity — it never produces a per-edge
  label. This baseline **adapts RegGAIN's differential-testing procedure to a per-edge question
  RegGAIN's paper never poses**: run the same Wilcoxon rank-sum test per-edge (edge score under
  condition A vs. condition B, treating each TF's edge scores as the paired distribution), then
  threshold the per-edge rank-sum p-value to collapse into the same 4-way scheme (significant +
  higher-under-A → A-specific, etc.), so it is scoreable against the same metric as `full`. This
  is an honest adaptation, not a reproduction of RegGAIN's reported output, and should be
  described that way in any writeup; (b) **`naive_expression_baseline`** — logistic
  regression on `(r_A, r_B, mean-expression fold-change of TF, mean-expression fold-change of
  target)` only, no GNN, no scFM, no learned embeddings at all — this operationalizes the
  lit review's point (`LIT_REVIEW.md` §5, Tier 3) that a GRN-level differential claim must beat
  gene-level differential-expression alone, not just beat "no model."

### 5.2 Synthetic ground-truth sanity check (GRouNdGAN)

Simulate condition A from the real hESC `TFs+500` reference network `G_A`; construct `G_B` as a
programmatically rewired variant of `G_A` with a **known** edge-level diff (e.g. remove 15% of
one TF's out-edges, add 15% new out-edges for a different TF, at 3 severities:
mild/moderate/severe rewiring fraction); run GRouNdGAN (causal GAN, simulates realistic
scRNA-seq under a specified GRN, preserves trajectory/noise structure, can simulate
knockouts) to generate matched synthetic expression for both `G_A` and `G_B`, at cell counts
comparable to the real hESC bookend pair (~90–190 cells/condition) to keep statistical power
realistic; run the **entire pipeline unchanged** (§3 silver-labeling + §4 model) against this
synthetic pair, but now score against the **actually-known** 4-way label from the construction
of `G_A`/`G_B` rather than the silver label. Report two numbers: (a) differential
AUROC/AUPRC of the *trained model* against true labels — the headline "is the architecture
sound at all" number; (b) accuracy of the §3 *silver-labeling procedure itself* against the
same true labels — a calibration check on how much noise the silver label injects before
trusting it on real data, directly mirroring how DiffCorr/DINGO/BoostDiff/DCI validate their
own statistics on simulated data before real-data case studies (`LIT_REVIEW.md` §4.1, §4.3).
GRouNdGAN is not currently installed in this environment (verified — only `netZooPy`,
`scanpy`, `torch-geometric` are present) and is a separate PyTorch codebase to integrate; see
§7 for the fallback if this proves too costly to stand up on the project timeline.

### 5.3 Biological validation — checked, not assumed, against the actual gene panels

The brief's suggested marker lists were checked against what is *actually present* in each
`TFs+500` gene panel and, critically, against **out-degree in the reference network** (a gene
can be in the expression panel but have zero candidate out-edges, in which case no edge-level
differential claim is possible for it at all — this was directly verified and changes the
mHSC-E list substantially from the brief's guess).

**hESC** (definitive endoderm; genes confirmed present with nonzero reference-net out-degree):

| Gene | Out-degree | Expected pattern |
|---|---|---|
| POU5F1 (OCT4) | 87 | out-edges skew `00h`-specific across all 3 pairs (pluripotency factor, down by 96h) |
| NANOG | 346 | out-edges skew `00h`-specific across all 3 pairs |
| SOX2 | 254 | out-edges skew `00h`-specific, though SOX2 persists longer than NANOG/POU5F1 in some protocols — expect a weaker effect than NANOG |
| EOMES | 300 | **transient** mesendoderm marker — expect `12h`-specific (up) for the `00h`-vs-`12h` pair specifically; for `00h`-vs-`96h` its edges may look only weakly differential or `neither` if it has already waned by 96h — this pair is *not* expected to show the cleanest EOMES signal, and the plan should not claim it does |
| SOX17 | 23 | out-edges skew `96h`/`72h`-specific (definitive endoderm) for `00h`-vs-`96h` and `36h`-vs-`72h` |
| GATA4 | 45 | same expectation as SOX17 |
| GATA6 | 22 | same expectation as SOX17 |

`FOXA2`, `CXCR4`, `MIXL1`, `TBXT`, `NODAL`, `HNF1B` — canonical DE/mesendoderm markers named in
the brief — are **not present** in the `TFs+500` (or `TFs+1000`) gene panel at all (checked
directly) and are dropped from the validation list rather than kept as an unverifiable claim.
`GSC`, `LHX1`, `CER1` are present in the gene panel but have **zero out-degree** in the
reference network (never a TF-source row) so no edge-level claim is possible for them either;
they are dropped from the edge-level check (could still be reported as expression-level context
only, not as a GRN-edge validation).

**mESC** (naive-pluripotency-circuit collapse — **not** the DE system, deliberately different
markers from hESC since this is a different biological process; genes confirmed present with
nonzero out-degree):

| Gene | Out-degree | Expected pattern |
|---|---|---|
| NANOG | 922 | `0h`-specific across all 3 pairs |
| POU5F1 | 854 | `0h`-specific, more gradual decline expected than the naive-circuit genes below |
| SOX2 | 512 | `0h`-specific |
| KLF4 | 687 | `0h`-specific — core naive-circuit gene |
| KLF2 | 28 | `0h`-specific — naive-circuit gene, low out-degree so a weaker/noisier signal expected |
| ESRRB | 618 | `0h`-specific — naive-circuit gene |
| TBX3 | 290 | `0h`-specific — naive-circuit gene |
| ZFP42 (REX1) | 581 | `0h`-specific — naive-circuit gene |
| PRDM14 | 223 | `0h`-specific — naive-circuit gene, extinguished early, best signal expected in `0h`-vs-`12h` |
| OTX2 | 642 | `72h`-specific (formative-state marker, rises later) — expect it to look `neither`/weakly-`0h`-conserved-off at `0h`-vs-`12h` and only clearly `72h`-specific by the bookend pair, with `24h`-vs-`48h` as the transition zone to watch |

`DNMT3B`, `FGF5`, `SOX3`, `PIM2`, `KLF5` from a naive-circuit reading are **not present** in the
panel. `TFCP2L1` and `DNMT3A` are present but have **zero reference-net out-degree** and are
dropped from the edge-level list for the same reason as `GSC`/`LHX1` above.

**mHSC-E** (erythroid commitment — **the brief's GATA1/KLF1/TAL1 list mostly does not work as
an edge-level check and is corrected here**, verified directly): `TAL1` is not in the `TFs+500`
gene panel at all. `GATA1`, `KLF1`, `NFE2`, `MECOM`, `HLF`, `PROCR`, `MPL` are present but every
one of them has **zero out-degree** in the reference network — none of them are ever a
TF-source row, so no edge-level differential claim is possible for GATA1's own regulatory
targets, KLF1's targets, etc. Only **`GATA2`** (out-degree 567) has usable candidate out-edges.
The corrected, verified validation set:

| Check | Detail |
|---|---|
| **`GATA2 → GATA1`** (flagship edge) | This specific edge **is present** in the mHSC-E reference network (score 1618) — the textbook "GATA switch" (GATA2 activates GATA1, which represses GATA2, driving the HSC/progenitor → erythroid transition; Bresnick-lab-documented biology). Directly testable: check the model's predicted 4-way label for this one edge across `LT_HSC`-vs-`Prog` and `HSPC`-vs-`Prog`, expecting it to be flagged differential (A-specific or a state-dependent flip) rather than `neither` or flatly `conserved`. This is the single most citable, most specific check available in this dataset and should be a headline biological result, not a footnote. |
| `GATA2`'s other out-edges (567 candidates) | Expect enrichment among `LT_HSC`/`HSPC`-specific and `conserved` classes relative to a random TF of similar out-degree, consistent with GATA2's known (gradual, not step-function) decline through erythroid commitment — reported as an enrichment comparison, not a hard binary claim, since GATA2's decline is graded in the literature, not a clean on/off switch. |
| `SPI1` (PU.1) out-edges (672 candidates) | PU.1 is the classic myeloid/HSC-progenitor master regulator; the PU.1–GATA1 mutual-antagonism axis (Nerlov & Graf and later work) predicts PU.1's regulatory activity should be `LT_HSC`/`HSPC`-skewed and fade toward `Prog`. No single named SPI1→target edge is proposed as a flagship (no direct SPI1–GATA1 edge in this reference network — checked, absent), so this is reported as an out-edge-set-level enrichment check, same caveat as GATA2 above. |
| `GFI1` vs `GFI1B` out-edges (604 / 257 candidates) | A second, independently documented antagonistic pair: GFI1 is myeloid/lymphoid-associated, GFI1B is pro-erythroid/megakaryocytic — expect GFI1's out-edges to skew `LT_HSC`/`HSPC`-specific and GFI1B's to skew `Prog`-specific/conserved-active, an internal-consistency check (two independently known antagonistic TF pairs should show the *same direction* of stage-specificity if the model is really recovering this biology, not overfitting one hand-picked edge). Note GFI1B is not expected to be a clean step function: it shows real activity earlier in the trajectory too (it also participates in early megakaryocyte-erythroid progenitor priming, not purely a `Prog`-stage-onset factor), so the expected pattern is a *skew* toward `Prog`/conserved-active rather than a hard `LT_HSC`-absent-then-`Prog`-present switch. |
| `EPOR` in-edges (12 candidates: `CBFB, ETS1, GATA2, GFI1, GFI1B, HCFC1, HDAC2, IKZF1, MYC, SPI1, STAT3`) | EPOR (erythropoietin receptor) is a canonical erythroid target gene, confirmed present with 12 real in-edges from legitimate hematopoietic regulators — used as a convergence check: do multiple erythroid-associated regulators' edges into EPOR trend `Prog`-specific together, vs. myeloid-associated regulators' edges into EPOR trending `LT_HSC`/`HSPC`-specific. `GATA1 → EPOR` (a very well-documented direct GATA1 target in real biology) is **not** in this reference network and cannot be used as a candidate edge under the §3.1 gating rule — noted explicitly rather than silently claimed as testable. |

### 5.4 What "grounded in biology" means operationally here

Per §3.4/§5.3, every biological check above is reported as a **named, pre-registered
expectation with a stated out-degree/candidate-edge count**, checked against actual model
output, not chosen post-hoc from whatever the model happens to produce. Any check that fails
(e.g., GATA2's out-edges do *not* skew `LT_HSC`-specific) is reported as a failure, not
dropped from the writeup — this is what distinguishes this from a metrics-only or
cherry-picked-anecdote paper (explicit user requirement).

---

## 6. Experiment plan

**Core matrix**: 9 condition pairs (§2) × GCN backbone (matches the existing scRegNet baseline
numbers, 0.8874/0.6225 AUROC/AUPRC on hESC/500 static task, the reference point for "how good
is static prediction here") × 5 configurations:

1. `full` — the complete model (§4): shared adjacency, condition-specific input projections,
   weight-tied GNN+fusion+MLP, 4-way head, static+CE+contrastive loss.
2. `no_scfm` — drop the scFM branch (GNN-only differential model), analogous to the existing
   `--ablation no_scfm` pattern from the base project (not present on this branch, must be
   ported — see §7).
3. `no_grn` — drop the GNN branch (scFM-embedding-only differential model via `proj_A`/`proj_B`
   feeding straight into the MLP head, no `GCNConv` at all). **Expected to perform near-chance
   on the differential task by construction** (§4.1) — this is a designed sanity check, and
   the plan explicitly predicts this result rather than treating a weak `no_grn` score as a
   surprise finding.
4. `no_contrastive` — drop `L_infonce`/`L_margin`, keep static BCE + 4-way CE only, to isolate
   the contrastive term's specific contribution from the 4-way classification head's.
5. `posthoc_baseline` — RegGAIN's differential-testing procedure, adapted to produce per-edge
   calls on this codebase (§5.1(a)).

Plus, cell-type-and-pair-independent:

6. `naive_expression_baseline` — logistic regression on correlation/fold-change features only
   (§5.1(b)), run once per condition pair (not per GNN ablation, since it has no GNN branch).

That is `9 pairs × 5 configs + 9 pairs × 1 naive baseline ≈ 54` GCN training runs plus 9 naive
baselines. **Robustness spot-check**: repeat the `full` configuration with `SAGE` and `GAT`
backbones on the 3 primary/bookend pairs only (not the full 9×5 grid, to keep compute bounded)
— 6 additional runs. **Synthetic validation** (§5.2): 3 rewiring severities × `full` config = 3
runs, plus the silver-label-calibration check (no additional training, reuses the GRouNdGAN
true labels against the already-cached silver labels). Total: **~63 training runs**, each on
graphs of ≤1,120 nodes and ≤~30k candidate edges — small by GNN standards, and the existing
`gnn_epochs=300` baseline already trains in well under an hour per run on this project's
hardware, so the full matrix is a multi-GPU-day, not multi-GPU-week, undertaking (see §7 for
the tuning-infra porting needed to run it efficiently).

**What compares against what**:
- `full` vs. `posthoc_baseline`: the central novelty claim (§1, point 1) — must show a
  measurable macro-F1 / per-class-AUROC advantage, specifically on the `A-specific`/
  `B-specific` classes (not overall accuracy, which `neither` would dominate either way), to
  defend "joint training beats post-hoc comparison" with actual numbers, not just architecture
  description.
- `full` vs. `naive_expression_baseline`: is graph+scFM structure adding anything over
  gene-level differential statistics alone (the tradeSeq-style objection from the lit review).
- `full` vs. `no_scfm`/`no_grn`/`no_contrastive`: which components of the architecture and
  loss are actually doing the work — required for any credible ablation table.
- `full` on real data vs. `full` on GRouNdGAN synthetic data with known labels: is the
  method's absolute performance level on real data limited by the architecture or by the
  silver label's own noise ceiling.

**Positive result**: `full` beats `posthoc_baseline` by a material margin (target: ≥5–10
macro-F1 points) on the differential task specifically for the specific classes, `full` also
beats `naive_expression_baseline` by a comparable material margin (target: ≥5–10 macro-F1
points) — not just a small edge — GRouNdGAN synthetic AUROC is well above chance (target ≥0.75)
confirming basic soundness, and the named biological checks (§5.3) resolve correctly in a
majority of cases across all three cell types (not just the easiest one).

**A specific failure mode worth naming explicitly, not hiding**: if `full` only marginally beats
`naive_expression_baseline` (the correlation/fold-change-only logistic regression), that is a
meaningful negative signal in its own right — it would mean the GNN+scFM machinery is mostly
**re-deriving the §3 label-generating statistic through a more expensive, harder-to-interpret
path**, rather than adding genuinely new structural information. That is a legitimate result to
report (the architecture is at least not actively hurting), but it is a substantially weaker
paper than one where `full` clears the naive baseline by a real margin, and the writeup should
say so plainly rather than reframing a marginal win as a strong one.

**Negative/still-informative result**: `full` does not clearly beat `posthoc_baseline`
numerically but *does* recover the named biological patterns (especially the `GATA2→GATA1`
edge and the PU.1/GATA1-GFI1/GFI1B convergence check) more consistently or more directly
(single trained model, no post-hoc test) than the baseline — per the user's explicit
requirement that biological grounding counts on its own, this is reported honestly as "the
joint model is not a clear quantitative win but is a methodologically cleaner and equally
biologically grounded alternative," not silently reframed as a win. A genuinely negative
result — GRouNdGAN synthetic AUROC near chance — is a stop condition: it would mean the
architecture is not learning the differential signal at all, and no real-data biological claim
should be trusted until that is fixed.

---

## 7. Feasibility & risks

**Branch/infra gap (must fix first, low risk, pure engineering)**: `differential-grn` predates
`src/tuning/`, `src/build_panda.py`, and the `--ablation`/`--node_feature_source` flags in
`args.py` — none of that exists on this branch today (verified: `git ls-tree` on
`differential-grn` returns nothing for `src/tuning`; only `__pycache__` is present). The tuning
infra (`manifest.py`/`worker.py`/`report_best.py`) is architecture-agnostic and would work for
`DiffTrainer` with minimal changes (swap `Trainer` for `DiffTrainer`, add a `differential_grn`
entry to `EXPERIMENTS`), but it needs to be ported from `summer26` (merge or cherry-pick)
before it can be used — do this early, since running ~63 training configs by hand is
impractical.

**Statistical power is limited for exactly one condition, not the pattern originally claimed
here.** An earlier draft of this plan asserted `LT_HSC`=59 cells and treated mHSC-E as the
weakest-powered cell type; that was wrong — grepped directly from the `TFs+500`/`TFs+1000`
CSV headers, `LT_HSC`=141, `HSPC`=384, `Prog`=546 cells, all comfortably above 100, and mHSC-E
is in fact the **best-powered** cell type of the three, not the worst. The corrected picture:
across all 9 chosen pairs, the one genuinely low-count condition is mESC's `12h` (68 cells),
which affects the `0h`-vs-`12h` mESC pair specifically — expect the `sig_A`/`sig_B` gates in
§3.3 to under-call significance there, producing a thinner `A-specific` class and more
`neither`/`ambiguous`/`inconsistent` labels for that one pair. hESC's `24h` (66 cells) is
similarly low but is not used in any of the three chosen hESC pairs (00h/12h/36h/72h/96h all
have ≥92 cells) — one more reason, alongside the biological rationale in §2, that `24h` was not
picked as a pair endpoint. mHSC-E's three pairs need no power caveat at all. **Net effect on
pair priority**: the bookend-pairs-as-primary framing in §2 stands for biological-contrast
reasons (largest state change), but the "early/mid pairs are secondary because underpowered"
justification only actually applies to mESC's `0h`-vs-`12h` pair — it should not be read as a
blanket statement about every non-bookend pair, and mHSC-E's `LT_HSC`-vs-`HSPC` and
`HSPC`-vs-`Prog` pairs in particular should be treated as equally reliable as its bookend pair
from a power standpoint.

**Class imbalance is severe by construction**: with modest cell counts and a strict BH+effect-
size gate, `neither` will be the majority class for most pairs (most of a 4,545–29,613-edge
candidate set will not show detectable condition-specific correlation given these sample
sizes). Macro-F1 (not accuracy) is mandated as the headline metric for exactly this reason
(§5.1), and the loss uses inverse-class-frequency weighting (§4.3).

**GRouNdGAN integration is the single highest-effort, highest-uncertainty item.** It is not
installed in this environment and is a separate PyTorch causal-GAN codebase requiring its own
training per simulated condition — realistically a multi-day integration + compute cost, not a
drop-in call. This is flagged as **core but time-boxed**: if it proves too costly to stand up
cleanly within the project's timeline, the documented fallback is a much simpler parametric
simulator (e.g. a negative-binomial model seeded directly from the real reference network's
edge structure and per-gene mean/dispersion from the real data) that gives a weaker but still
genuine known-ground-truth sanity check, at the cost of realism (no learned noise/trajectory
model). The plan should attempt GRouNdGAN first and fall back explicitly, not silently drop
Tier 1 if it turns out hard.

**The candidate edge space is TF-concentrated** (34/29/88 unique source TFs per cell type,
§2) — this is a limitation inherited from the base reference networks, not introduced by this
plan, but it does mean the differential-edge claim only ever covers a minority of each panel's
TFs. This should be stated as an explicit scope limitation in the paper, not discovered by a
reviewer.

**The `no_grn` ablation performing near chance is predicted, not a risk** — worth restating
here because a naive reading of a weak `no_grn` result could look like a broken experiment
rather than the expected, designed outcome given the scFM branch is condition-independent by
construction (§4.1).

**The `posthoc_baseline` comparison is non-negotiable for the paper's central claim** to
survive review — it must be run on this exact codebase/data as an adaptation of RegGAIN's
differential-testing procedure to a per-edge question (not just cited from RegGAIN's own
reported per-TF numbers, which use a different reference network — NicheNet — different
train/test conventions, and a different question entirely), or the "joint beats post-hoc" claim
is not actually demonstrated.

**Compute is not the bottleneck** — graphs here are small (≤1,120 nodes), and the existing
static baseline already trains fast; the ~63-run matrix is feasible on the project's 4-GPU HPC
setup once the tuning infra is ported.

**Recommended phasing** given "soon, not just theorized": **Phase 0 — labeling-yield dry run,
all 9 pairs, before any training infra is built.** Run just the §3 silver-labeling pipeline
(CPU-only, no model training) across all 9 condition pairs and record, per pair, the raw class
counts (`conserved`/`A-specific`/`B-specific`/`neither`/`ambiguous`/`inconsistent`). This is
cheap (no GPU, no training loop) and directly tests the assumption the rest of the plan is built
on — that every pair yields a non-trivial number of `A-specific`/`B-specific` examples to learn
from and evaluate against the §5.1 class-count floor. If a pair comes back with, say, single-
digit specific-class counts even after cross-checking §7's corrected power picture, that pair
should be dropped or flagged low-priority *before* building the training/ablation matrix around
it, not discovered after Phase 2 is already underway. Phase 1 — get the full architecture
working correctly end-to-end on exactly one pair (hESC `00h` vs `96h`, largest contrast,
cleanest markers, and — pending Phase 0's confirmation — good labeling yield) including the
biological check, before generalizing. Phase 2 — extend to the remaining pairs (as informed by
Phase 0) and the ablation grid (§6). Phase 3 — GRouNdGAN synthetic validation, which can proceed
in parallel with Phase 2 once Phase 1 confirms the architecture is sound.

---

## 8. What would make this paper-worthy

**Target venue**: given the lit review's finding that "contrastive GNN for GRN inference" is
now a crowded mechanism (≥7 papers 2025–2026) and that the closest comparable work (RegGAIN,
BoostDiff, DiCoLo, Meta-TGLink, KEGNI) publishes in *Advanced Science*, *Bioinformatics
Advances*, *Genome Biology*, and RECOMB/ISMB proceedings rather than top-tier ML venues, the
realistic target is a bioinformatics methods venue in that same tier (Bioinformatics/
Bioinformatics Advances, Genome Biology, or a RECOMB/ISMB track), not a general ML venue where
the contrastive-GNN mechanism alone would not read as novel.

**Two separable claims, not one conflated claim** (this section directly follows the user's
explicit requirement that biology grounding matters as much as metrics):

1. **Methodological claim**: a jointly-trained, weight-tied Siamese GNN+scFM architecture with
   an explicit 4-way differential-edge decoder measurably outperforms the field's current
   practice for this exact task on this exact data — a same-codebase, same-data,
   apples-to-apples comparison against `posthoc_baseline` (RegGAIN's differential-testing
   procedure, adapted to produce per-edge calls, §5.1/§6 — not a reproduction of RegGAIN's own
   reported per-TF output, which is described accurately as an adaptation throughout), backed
   by a metric with real ground truth
   (GRouNdGAN synthetic, §5.2) and a defensible, FDR-controlled silver-label proxy on real
   data (§3). This claim stands or falls on the numbers in §6 and must be reported honestly
   even if the margin is small.
2. **Biological claim**: the model's predicted differential edges on real hESC/mESC/mHSC-E data
   recover specific, named, independently-documented regulatory transitions — the GATA2→GATA1
   switch and PU.1/GATA1-GFI1/GFI1B convergence in erythropoiesis, the naive-pluripotency-
   circuit (NANOG/KLF4/ESRRB/TBX3/ZFP42/PRDM14) collapse in mESC, and the pluripotency-to-
   definitive-endoderm handoff (POU5F1/NANOG/SOX2 down, EOMES transiently up, then
   SOX17/GATA4/GATA6 up) in hESC — checked against the model's actual output, not asserted.

**Honesty condition on claim 2, explicitly stated for the paper**: recovering well-known
textbook regulatory biology is a **validation** result (it establishes the model's outputs are
biologically sane, matching how RegGAIN/SCENIC+/BoostDiff/DINGO all validate — per the lit
review, this is the field's dominant practice, not a weak substitute for it), not by itself a
**novel-biology discovery** claim. Before any finding is framed as new, it must be checked
against RegGAIN's own published hESC time-course findings (its GATA2-GATA3-STAT3 loop and any
other reported hESC results) specifically to confirm the paper is not accidentally re-deriving
RegGAIN's answer through a differential label instead of a Wilcoxon test — a *substantive*, not
merely mechanistic, differentiation is required before claiming new biology. Any genuinely
unexpected differential edge among the small set of reference-network-covered TFs (34/29/88 per
cell type) should be reported as a hedged "candidate finding for future validation," not a
headline result — this project has no perturbation data of its own, and per the lit review, no
compute-only GRN paper in this space treats an unconfirmed computational finding as a
validated discovery on its own.

**What would make it clearly paper-worthy, concretely**: a positive result on claim 1 (§6) *and*
a clean recovery of the `GATA2→GATA1` flagship edge plus at least one of the two
pluripotency-circuit collapses (hESC or mESC) on claim 2, reported together with the explicit
methodological-vs-biological framing above, and with the `posthoc_baseline`/GRouNdGAN
comparisons included as evidence the numbers are not just an artifact of an easy silver label.
A result with only claim 1 (numbers beat baseline, no biological check) or only claim 2
(biology recovered, no controlled comparison against post-hoc practice or synthetic ground
truth) would each, on their own, be a weaker paper than the combination — this is the
concrete operational meaning of the user's "not a metrics-only paper" requirement.
