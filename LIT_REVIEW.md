# Literature Review — Differential/Contrastive GRN Idea

Agent A output, produced against `DIFFERENTIAL_GRN_BRIEF.md`. Search performed via arXiv
(direct + web-indexed), bioRxiv, Nature Methods/Communications, Genome Biology,
Bioinformatics/Bioinformatics Advances, Cell Systems-adjacent venues, ISMB/RECOMB
proceedings, and general web search — not arXiv-only. Read at least an abstract-level
extraction of every paper cited; several were read in fuller detail (RegGAIN, BoostDiff,
Sc-compReg, BRIDGE, GP-DHT, SupGCL, CellOracle) via full-text or structured-abstract fetch.

**Bottom line up front**: the brief's earlier claim ("none do joint, simultaneously-trained
differential-edge prediction across paired conditions with a dedicated metric") **survives**,
but by a narrower margin than the earlier arXiv-only pass suggested. One paper —
**RegGAIN** (Guan et al., *Advanced Science*, 2025/2026) — trains a contrastive GNN on the
*exact same* BEELINE hESC/mESC/mHSC-E benchmarks *and* the *exact same* Chu et al. hESC
differentiation time-course (6 time points, same barcodes) that this project uses, and
explicitly frames part of its analysis as "regulatory rewiring" / "dynamic regulons" across
those time points. It does **not**, however, jointly train on a differential-edge task or
use a dedicated differential metric — it trains one static contrastive GNN and compares two
post-hoc network instantiations with a Wilcoxon rank-sum test on edge scores, validated only
qualitatively (GO enrichment, Jaccard similarity of TF sets, visual inspection of known
loops). This is the single most important prior-work finding in this review and is
discussed in detail in §3.

---

## 1. Taxonomy of prior work

### 1.A. Modern contrastive-learning / GNN deep-learning methods for GRN inference

| Method | Venue/date | Core mechanism | Cross-condition element | Ground truth used |
|---|---|---|---|---|
| **GP-DHT** | arXiv 2601.10995 (2026) | Heterogeneous gene/cell graph + multi-relational graph attention; dual-head Transformer; gene-pair-level *supervised* contrastive learning | **Cross-species** (human vs. other species), not within-species time/condition. Distinguishes conserved vs. species-specific regulation as an analysis output, not a jointly-trained 4-way label | BEELINE reference networks (7 datasets), supervised contrastive labels from known edges |
| **BRIDGE** | arXiv 2606.14734 (2026) | Gene+cell dual representations from expression matrix and its transpose; contrastive learning between "self and neighbors" across a co-expression-refined view and the original graph; **cross-cell-type transfer** (pretrain on mESC, fine-tune 5% data on target cell type) | None — static network per cell type; the "transfer" axis is cell type identity, not paired differential prediction | Curated TF-target DBs (e.g. ChIPBase) + BEELINE references |
| **BRIDGE-GRN** (distinct paper, similarly named) | bioRxiv 10.64898/2026.05.12.724562 (2026) | Role-aware bi-tower graph learning; cross-view contrast between identity- and edge-perturbed graph views for **directed** GRN inference | None — robustness-to-noise contrastive objective, not cross-condition | Reference GRN, train-positive support graph |
| **SupGCL** | arXiv 2505.17786, ICML 2026 (OpenReview YRh4epoqIS) | Probabilistic generalization of GCL that substitutes real gene-knockdown perturbations for artificial graph augmentations as the contrastive "positive/negative" signal | None on time/condition axis — applied to patient-level cancer-subtype GRNs, not paired biological conditions of one system | **Real knockdown/knockout perturbation data** (gold standard, confirmed) |
| **RegGAIN** | *Advanced Science*, 2025/2026 (doi 10.1002/advs.202518277, PMC12904006) | Dual-role (regulator/target) graph contrastive learning: two augmented views (random edge-drop+feature-mask vs. topology-aware hub-preserving perturbation), InfoNCE-style anchor/positive over gene embeddings, direction-aware decoding | **Applies the same static model separately to each of 6 hESC time points** (Chu et al., identical barcodes to this repo's data) and to MM-vs-normal bone marrow; ranks TFs by *differential out-degree* via **Wilcoxon rank-sum test on edge scores between two independently-inferred static networks** — a post-hoc statistical comparison, not a jointly-trained differential-edge classifier, and has no 4-way conserved/A-specific/B-specific/neither label or dedicated differential metric | One static reference network (NicheNet) reused across all 6 time points; no per-time-point ground truth; validation of "dynamic regulon" claims is **qualitative only** (GO enrichment, Jaccard similarity of TF sets across stages, feed-forward-loop inspection — e.g. GATA2-GATA3-STAT3) |
| **Meta-TGLink** / "Structure-enhanced graph meta-learning" | *Genome Biology*, 2025 (doi 10.1186/s13059-025-03860-8) | Meta-learning + Transformer-GNN hybrid for **few-shot cross-domain** GRN link prediction | Cross-cell-type/cross-domain generalization, not paired differential prediction | Small labeled support sets per target domain |
| **KEGNI** | *Genome Biology*, 2025 (doi 10.1186/s13059-025-03780-7) | Knowledge-graph-enhanced GNN framework for GRN inference | None described as cross-condition | External KG + reference edges |
| **GCLink** | (cited as baseline in BRIDGE/RegGAIN papers) | Graph contrastive learning extended to GRN link prediction | Static, single condition | Reference network |
| **DGCRL** (Directed Graph Contrastive Representation Learning) | — | Augmentation reverses edge direction to build source/target contrastive views | Static, single condition; direction-aware not condition-aware | Reference network |
| **DeepMCL** | — | Multi-view contrastive learning combining data from diverse platforms/time points into one embedding | Time points are pooled as multi-view *inputs* to improve one static inference, not predicted as a differential output | Reference network |
| **Hierarchical Graph Transformer w/ Contrastive Learning** | IEEE (doi via pubmed 39401117) | Hierarchical graph transformer + contrastive pretraining for GRN inference | Static | Reference network |
| **scRegNet** (this project's own base method) | bioRxiv 2024.12.16.628715 | Joint GNN + scFM representation, exactly the dual-branch architecture in `src/models.py` | Static, single condition — the paper this repo already extends | BEELINE reference networks |
| **DeepFGRN** | *Briefings in Bioinformatics*, 2024 | Directed graph embedding, regulation-*type* (activator/repressor) classification | Not condition-differential — a different 3rd axis (edge sign) than our 4-way conserved/specific/neither axis, but a reminder that "extra label dimensions beyond binary edge/no-edge" is itself not new | Reference network with signed edges |
| **CroCoNet** | bioRxiv 2025.11.18.689002 (2025) | Not a GNN — quantitative **network-module comparison** (topology variability within/between species) centered on known TF modules; R package | **Cross-species** rewiring (human/gorilla/macaque neural differentiation), analogous in spirit to a "differential" framing but on the species axis, classical/statistical not deep learning | Validated the flagged POU5F1/OCT4 module divergence with **independent cross-species CRISPRi + scRNA-seq perturbation as a follow-up confirmation experiment** — i.e., the paper explicitly obtained real perturbation data to validate its top computational finding rather than relying on the computational result alone |

**Survey/taxonomy sources**: Hegde, Nguyen & Cheng, "Machine learning methods for gene
regulatory network inference," *Briefings in Bioinformatics* 26(5), 2025 (bbaf470) — general
ML taxonomy, does not single out differential/cross-condition prediction as a distinct
established sub-area, which is itself informative (the survey treats "dynamic"/time-series
GRN inference as reconstructing *one* evolving network, not classifying edges into
conserved/specific/neither categories).

**Re-assessment of §2's three anchor papers from the brief**:
- **GP-DHT**: confirmed as described — cross-*species*, not cross-condition/time; supervised
  contrastive but single jointly-trained network per species pair, then post-hoc conserved-
  vs-specific *interpretation*, not a trained 4-way edge classifier.
- **BRIDGE**: confirmed as described — self-supervised contrastive pretraining for
  cross-cell-type transfer of a *static* target network; no differential-edge output at all.
  Note there are now **two distantly-related "BRIDGE" GRN papers** in 2026 (BRIDGE and
  BRIDGE-GRN) — worth double-checking exact citation when writing the paper's related work,
  to avoid conflating them.
- **SupGCL**: confirmed — real knockdown perturbation as supervision, strongest ground-truth
  signal in the space, but applied to disease-subtype embedding/classification tasks on
  patient GRNs, not to paired time-point/trajectory differential-edge prediction.

### 1.B. Classical/statistical differential-network-biology methods

This field substantially predates the deep-learning wave (some methods from the early 2010s)
and is large. Representative, relevant tools:

| Method | Venue/date | What it does | Validation without perturbation |
|---|---|---|---|
| **DiffCorr** | Fukushima, *BMC Systems Biology* 2013 (R package, still widely used) | Differential *correlation* network analysis (Fisher z-test on correlation coefficients between two conditions) | Statistical significance (permutation/FDR) + downstream pathway/GO annotation of the differential edges |
| **DINGO** | Ha, Baladandayuthapani & Do, *Bioinformatics* 31(21), 2015 (PMC4751246) | Bayesian pathway-based differential network model; borrows statistical strength across groups via a common/group-specific decomposition; can integrate multiple omics layers (mRNA + CNV + methylation) and overlay edges present in all of them for higher confidence | Simulation studies with known ground-truth precision matrices **+** known-hub-gene biology (recovered known c-Myc-network hubs in glioblastoma) **+** multi-omics-layer concordance as an internal cross-check (no perturbation data used at all) |
| **CFGL** (Condition-adaptive Fused Graphical Lasso) | bioRxiv 290346 / PMC6173447 | Adaptive fused graphical lasso for condition-specific co-expression networks, sharing structure across conditions when supported by the data | Simulation + biological case studies |
| **jointGHS** | Camiling et al., *Annals of Applied Statistics* 18(3) 2023 (arXiv 2206.11820) | Joint graphical horseshoe Bayesian model for multiple Gaussian graphical models; explicitly estimates **shared edges across conditions in addition to condition-specific edges** via an edge-specific latent shrinkage variable — conceptually the closest classical analogue to a "conserved vs. specific edge" label, though continuous/probabilistic rather than a hard 4-way class | Simulation (varying network similarity) + applied to immune-stimulation monocyte data, interpreted via known immune pathway biology |
| **CSN / c-CSN** (Cell-Specific Networks) | Dai et al., *Nucleic Acids Research* 2019 and follow-ups | Statistical test (usually based on comparing conditional/marginal dependence with vs. without one cell) yielding a **network per single cell**; conditional variant (c-CSN) adds conditional-independence tests to reduce false positives | Downstream clustering/enrichment; no perturbation ground truth used |
| **NeighbourNet** | bioRxiv 2025.03.27.645629 (2025) | Scalable cell-specific co-expression networks for granular regulatory pattern discovery (CSN-family extension) | Same class of validation as CSN |
| **CSCN** (Cell-Specific Causal Networks) | bioRxiv 2025.10.09.681381 (2025) | Cell-specific *causal* network inference from scRNA-seq | — |
| **Sc-compReg** | Duren et al./collaborators, *Nature Communications* 2021 (PMC8346476) | Joint scRNA + scATAC clustering/embedding across two conditions; a novel likelihood-ratio-style statistic (non-standard null distribution, estimated) for **testing differential regulatory relations** between matched subpopulations across two conditions | **Simulation-based validation with ROC curves and FDR control** for the differential-detection statistic itself, then a real biological case study (CLL vs. healthy) where the top novel finding (TOX2 as a tumor-subpopulation regulator) is treated as a *biologically plausible, literature-consistent* discovery, not re-validated experimentally in the paper |
| **BoostDiff** | Galindez et al., *Bioinformatics Advances* 4(1) 2024 (bbae034 / PMC10948285, bioRxiv 2022.09.26.509450) | AdaBoost ensemble of "differential trees" with a novel differential-variance-improvement splitting criterion; converts variable importance into a directed **differential network** between two conditions | **Simulated data (two complexity regimes) with known ground truth** to validate method correctness, **then** real transcriptomics (COVID-19, Crohn's, breast/prostate cancer, *B. subtilis* stress) validated purely via **KEGG/pathway enrichment consistency with known disease biology** — no perturbation data anywhere in the paper |
| **DCI** | Belyaeva, Squires & Uhler, *Bioinformatics* 2021 (PMC9991896) | Causal-DAG-based method that directly estimates the *causal difference* between two related GRNs (not each network separately then a diff) with finite-sample statistical guarantees | Simulation with known causal ground truth; real-data application to leukemia/T-cell data validated by literature-known relevant genes, not perturbation |
| **scTenifoldNet** | bioRxiv 2020.02.12.931469 | Machine-learning workflow (regression + manifold alignment + low-rank tensor denoising) for constructing and *comparing* transcriptome-wide GRNs from two scRNA-seq conditions | Simulation + literature-known biology in case studies |
| **DiCoLo** | RECOMB 2026 (bioRxiv 2025.11.23.689932, PMC12704003) | Integration-free, cluster-free detection of **localized differential gene co-expression** using Optimal-Transport-based per-condition gene graphs, without requiring cross-condition cell alignment | Benchmarked against existing methods on multiple datasets + applied to mouse hair-follicle development, validated via known morphogen-signaling/dermal-condensate biology |

**Key structural point for our plan**: several of these classical methods (jointGHS,
Sc-compReg, DCI) already formalize the *statistical* notion of "conserved vs. condition-
specific edge" that our brief's 4-way label wants to approximate with weak supervision. None
of them are deep/GNN-based, none use scFM node features, and — critically for the "what
counts as ground truth" question — **every one of them validates purely against
simulation-with-known-truth and/or literature/pathway plausibility when applied to real
biological data with no perturbation available.** None claim or need real perturbation data
for real-data validation; SupGCL and CroCoNet's confirmatory follow-up are the only entries
in this whole review that use real perturbation ground truth, and both are add-ons/follow-up
confirmations rather than the primary validation route.

### 1.C. Trajectory/pseudotime-based dynamic GRN and branch-specific regulon methods

| Method | Venue/date | Mechanism | Cross-condition/branch handling | Validation |
|---|---|---|---|---|
| **dynGENIE3** | Huynh-Thu & Geurts, *Scientific Reports* 2018 (PMC5821733) | Random-forest variable importance extended from GENIE3 to handle time-series (ODE-based extension: predicts derivative of expression) | Single evolving network, not branch/condition differential | DREAM4 synthetic benchmarks (known ground truth) + 3 real time-series datasets scored by AUC against curated references |
| **SCODE** | Matsumoto et al., bioRxiv 088856 / *Bioinformatics* 2017 | Linear ODE model fit via pseudotime as a proxy for real time; treats pseudotime-ordered cells as a time-course | Single trajectory, not multi-branch differential | TF-target motif databases (e.g. TF2DNA) used as external validation, not perturbation |
| **tradeSeq** | Van den Berge et al., *Nature Communications* 2020 (PMC7058077) | GAM (negative-binomial) smoothers of expression along pseudotime per lineage; `associationTest`/`patternTest` for within- and **between-lineage** differential expression testing | This is the most directly relevant *classical statistical* precedent for "compare gene behavior across two branches of one trajectory" — but it operates at the single-gene expression level, not the TF→target regulatory-edge level; would need to be paired with a GRN step to produce edge-level differential calls | Benchmarked against Monocle3/other DE tools on simulated + real trajectory data; no perturbation ground truth needed since it's a statistical test on observed expression, not a predictive model |
| **SCENIC+** | Bravo González-Blas et al., *Nature Methods* 2023 (doi 10.1038/s41592-023-01938-4) | Multiome-based enhancer-driven GRN (eRegulon) inference; for trajectory/branch analysis, fits a GAM of TF expression and target-gene regulon enrichment along pseudotime per branch, defines a "differentiation force" metric (distance between the TF-expression GAM curve and the future-cell regulon-enrichment GAM curve) to find TFs whose activity precedes/lags target activation | Directly analogous conceptually to a "branch-specific regulon activation" differential claim, applied e.g. to OPC→oligodendrocyte differentiation, recovering known TFs (Olig2, Sox10, etc.) | **Entirely biological-plausibility-based**: known marker/regulator TFs recovered, GAM-based delay/precedence patterns interpreted qualitatively; no dedicated differential-edge ground truth or metric used |
| **CellOracle** | Kamimoto et al., *Nature* 2023 (doi 10.1038/s41586-022-05688-9) | Infers cluster/state-specific GRNs (regularized linear regression, multiome) then performs **in-silico TF knockout/overexpression simulation** using the inferred GRN + pseudotime vector field to predict downstream cell-identity shifts — all from wild-type (unperturbed) data alone | Produces cell-state-specific GRN configurations across a trajectory and validates that in-silico perturbation of these state-specific networks reproduces *already-published* real perturbation phenotypes (mouse/human hematopoiesis, zebrafish embryogenesis) | This is the strongest and most transferable **"validate without your own perturbation data"** precedent found in this review: it doesn't have or need new perturbation data for the system under study, it validates by checking whether in-silico-perturbing its inferred network reproduces *literature-documented* phenotypes for well-studied systems — hematopoiesis (relevant to our mHSC-E data) and general embryogenesis (relevant to our hESC/mESC data) |
| **DELAY**, **scTIGER2.0**, **PseudoGRN** | various, 2022–2025 | CNN / time-lagged divergence / partial-correlation pipelines that use pseudotime-lag structure as a supervisory signal for directionality | Single evolving trajectory | Simulated + curated reference networks |

### 1.D. Ground-truth/synthetic-benchmark generators

- **GRouNdGAN** (Yousefi & ..., *Nature Communications* 2024, PMC11525796; toolkit update
  bioRxiv 2025.08.14.670294): causal-GAN that simulates realistic scRNA-seq data *under a
  user-specified reference GRN*, preserving pseudotime/trajectory structure and technical
  noise, and can simulate in-silico TF knockouts. This offers a route to a fully-synthetic,
  known-ground-truth differential benchmark (simulate two conditions from two related GRNs,
  one a perturbed/rewired version of the other) to validate *the differential-prediction
  method itself* before trusting it on the real hESC/mESC/mHSC-E data — analogous to how
  BoostDiff and DCI validate on simulated data with known differential ground truth before
  the real-data case study.
- **In-silico perturbation via scFMs** (Geneformer/scGPT zero-shot gene-deletion embedding
  shift): recent evaluation work (e.g. bioRxiv 2025.05.11.653338, 2025.09.22.677811) shows
  zero-shot scFM in-silico perturbation prediction is currently **weak** (~14-15% top-k
  accuracy on K562 Perturb-seq, well below GEARS which is trained on real perturbation data)
  — this is a caution against treating scFM in-silico-knockout output as a reliable
  pseudo-ground-truth differential signal without substantial validation of its own.

---

## 2. Direct comparison against the brief's proposed idea

The proposed idea: a jointly-trained contrastive/differential GNN that, given paired
conditions, predicts a **4-way label per TF→target edge** (conserved / A-specific /
B-specific / neither), trained end-to-end with a dedicated differential metric, applied
within-species to time-course/stage data (hESC time points, mESC time points, mHSC-E
stage gradient), using existing scFM+GNN dual-branch architecture (`scTransNet_*`),
supervised by a **weak/silver label** derived from statistical differential-co-expression
analysis (since no time-resolved reference network exists), with biological-plausibility
validation against known stage-specific regulators.

| Dimension | Our idea | Closest match found | Verdict |
|---|---|---|---|
| Jointly-trained (single model, single forward pass conditions on both networks simultaneously and outputs a differential label) | Yes | None found does this for GRNs. RegGAIN, GP-DHT, CroCoNet, Sc-compReg, DINGO, jointGHS, BoostDiff, DCI all **train/estimate per-condition networks (or paired precision matrices) then compare post-hoc** via a separate statistical/comparison step, not a single jointly-trained differential classifier | **Novel** — genuinely a gap, not just an arXiv-search artifact |
| 4-way (conserved/A-specific/B-specific/neither) discrete edge label as the *direct training target* | Yes | jointGHS is the closest conceptually (shared vs. condition-specific edges via a continuous latent variable) but it's a Bayesian graphical-model estimator over precision matrices, not a supervised/contrastive classifier with discrete 4-way training labels | **Novel in this specific formulation**, though "shared vs. specific" as a *concept* is old (jointGHS, DINGO's multi-platform overlay, CroCoNet's conserved/diverged modules) |
| Dedicated differential-edge evaluation metric | Yes (to be designed) | Sc-compReg has a dedicated differential-detection test statistic with simulation-validated ROC/FDR properties — closest classical precedent. BoostDiff/DCI validate differential-network *recovery* on simulated ground truth. No deep-learning GRN paper in this review reports a dedicated differential-edge AUROC/AUPRC | **Partially precedented** in classical stats (Sc-compReg, BoostDiff, DCI simulation protocols) but **not in the deep-learning GRN literature** — worth explicitly building the metric analogous to Sc-compReg's simulation-based validation of the statistic itself |
| Contrastive learning as the mechanism | Yes | GP-DHT, BRIDGE, BRIDGE-GRN, SupGCL, RegGAIN, DGCRL, GCLink all use contrastive learning for GRN inference | **Mechanism itself is not novel** — contrastive GNNs for GRNs is now a crowded sub-area (at least 7 papers in 2025-2026 alone). Novelty must rest on the *differential/joint-condition task formulation*, not on "we used contrastive learning for GRNs" |
| scFM node embeddings (Geneformer/scGPT/UCE) as condition-independent features fused with a graph branch | Yes (already scRegNet's baseline) | scRegNet itself (this repo's own prior paper) is the only method combining scFM + GNN found for GRN inference; no other reviewed paper fuses scFM embeddings into a *differential* prediction task | **Novel combination** in the differential context, though the scFM+GNN fusion mechanism itself is this project's own established baseline, not new |
| Application to hESC/mESC/mHSC-E time-course/stage data specifically | Yes | **RegGAIN uses the identical Chu et al. hESC 6-time-point dataset and the identical BEELINE hESC/mESC/mHSC-E benchmarks**, and explicitly performs "regulatory rewiring"/dynamic-regulon analysis across those same time points | **This is the closest data overlap found.** Any paper draft must cite and differentiate from RegGAIN explicitly and precisely (see §3) |
| Weak/silver differential label from statistical differential-co-expression, gated by the static reference network as candidate-edge filter | Yes (proposed) | DiffCorr/DINGO/CFGL/jointGHS/BoostDiff/DCI are all exactly this class of statistical differential-network method and are the natural sources for constructing the silver label; **none of them have previously been used as a training-label generator for a downstream GNN** in the reviewed literature — the pattern "classical differential-network statistic → weak label → train a deep model" appears field-standard in the *adjacent* area of scFM in-silico-perturbation validation (PSGRN: correlation-derived pseudo-ground-truth + self-training) but not, in this review, for cross-condition differential GRN edges specifically | **Novel pairing** — statistically well-grounded (methods exist and are mature) but the specific use as a differential-label generator for a jointly-trained GNN is not precedented |

---

## 3. RegGAIN — the critical overlap case, examined closely

Because RegGAIN is materially the closest paper found (same architecture family —
contrastive GNN — same exact benchmark data — same exact real time-course dataset), it
deserves explicit, precise treatment rather than being buried in the table.

**What RegGAIN does, precisely** (from full-text extraction of PMC12904006):
1. Trains one contrastive GNN with dual-role (regulator/target) gene embeddings, using two
   augmented graph views (uniform random edge-drop + feature-mask vs. a topology-aware
   variant that preferentially perturbs hub edges) and an InfoNCE-style anchor/positive
   contrastive loss between the two views' embeddings of the same gene.
2. This is trained as a **single static GRN inference model**, benchmarked on BEELINE
   (hESC, mESC, mHSC-E among 7 datasets) against a single static reference network per cell
   type, exactly like scRegNet's own baseline task.
3. Separately, as an **application/case study** (not part of the main benchmark), the same
   trained-model architecture is run independently on: (a) the Chu et al. hESC differentiation
   time course (0/12/24/36/72/96h, same barcodes this repo uses), inferring "a GRN at the
   individual sample level," and (b) multiple myeloma vs. normal bone marrow samples.
4. For each such application, a **Wilcoxon rank-sum test compares edge-score distributions
   between two independently-inferred static networks** (e.g., MM vs. NBM) to rank TFs by
   differential out-degree. There is no single model that ingests both conditions and
   outputs one 4-way (or any) differential label per edge in one trained/joint step.
5. Validation of the "dynamic regulon"/rewiring claims is **entirely qualitative**: GO
   enrichment of stage-specific TF modules, Jaccard similarity of top-TF sets between
   consecutive stages, and manual inspection of specific feed-forward loops (e.g.
   GATA2-GATA3-STAT3). No quantitative differential-edge metric against any ground truth is
   reported anywhere in the paper.
6. Ground truth: one static reference network (NicheNet) reused identically across all six
   time points — i.e., RegGAIN faces **exactly the same "no time-resolved reference network"
   problem** this project's brief identifies (§4 of the brief), and **does not solve it** —
   it sidesteps the problem by never requiring the reference network to be
   condition-specific in the first place (reference is only used for the static single-
   condition benchmark; the time-course analysis is reference-free and purely qualitative).

**Precise verdict**: RegGAIN establishes that (a) contrastive GNNs for GRN inference on
this exact data are already published, (b) post-hoc comparison of two independently-run
static networks for "rewiring" analysis is already published on this exact hESC time
course, and (c) purely qualitative/biological-plausibility validation of stage-specific
regulatory claims (no dedicated differential metric, no time-resolved ground truth) is
already the field's practice on this exact dataset. It does **not** establish (a) joint
simultaneous training of both conditions in one model with a shared differential objective,
(b) a discrete conserved/A-specific/B-specific/neither label, or (c) any dedicated
differential-edge quantitative metric. **Novelty claims in any resulting paper must
explicitly cite RegGAIN and state precisely this distinction** — "post-hoc statistical
comparison of two independently-trained static networks" vs. "jointly-trained differential
classifier" — rather than claiming to be first to look at cross-time-point regulatory
change in this data at all, which would be incorrect and easily challenged by a reviewer
who knows this paper.

---

## 4. How the field validates differential/dynamic GRN claims without perturbation data

This is the central methodological question posed in the brief. Synthesizing across every
method reviewed above (both classical and deep-learning), the field uses a **small, well-
established set of substitute validation strategies**, almost always used in combination
rather than alone:

1. **Simulation with known synthetic ground truth, as a first-pass validation of the
   method's statistical/computational correctness** — used by essentially every classical
   differential-network method reviewed (DINGO, DiffCorr-family, CFGL, jointGHS, Sc-compReg,
   BoostDiff, DCI, scTenifoldNet) and by the standard GRN-benchmarking generators
   (GRouNdGAN, DREAM4 for dynGENIE3). The logic: prove the method recovers a *known*
   differential structure in synthetic data (where a perturbed/rewired ground-truth network
   is used to *generate* the second condition's data) before trusting it on real data where
   no such ground truth exists. **This is directly actionable for us via GRouNdGAN**, which
   can generate two related synthetic datasets from two related reference GRNs (one a
   deliberately rewired version of the other) — giving a genuine quantitative differential-
   edge AUROC/AUPRC benchmark with known truth, something no real-data-only design in this
   review has.

2. **Known-regulator / marker-gene enrichment as the primary real-data validation** — by far
   the most common approach for the exact hESC/mESC/mHSC-E-style time-course/trajectory
   setting: RegGAIN (GO enrichment + known TF modules), SCENIC+ (recovers Olig2/Sox10-type
   known lineage TFs), SCODE (TF2DNA motif-database cross-check), BoostDiff (KEGG pathway
   enrichment consistent with known disease biology), DINGO (known c-Myc network hubs),
   CroCoNet (known POU5F1 biology, later perturbation-confirmed). This is precisely what the
   brief already proposes in §4 (SOX17/FOXA2/EOMES turning on, POU5F1/NANOG/SOX2 turning off
   for hESC; GATA1/KLF1/TAL1 activating for mHSC-E) — **this is not a weaker or improvised
   substitute for ground truth in this field, it is the field's dominant real-data
   validation strategy when perturbation data is absent**, used essentially universally
   across both the classical and deep-learning literatures reviewed here.

3. **Statistical differential-network tests as a "silver label," with explicit FDR/
   significance control** — DiffCorr, DINGO, Sc-compReg, BoostDiff, DCI all produce
   statistically-principled (not deep-learning) differential-edge/differential-network
   calls with p-values or FDR control. These are the most defensible non-perturbation
   "ground truth" available for real data, precisely because they come with statistical
   guarantees rather than being another opaque model's output. This is exactly the class of
   method the brief proposes using to build the weak/silver label (§4) — the review confirms
   this is methodologically sound and field-precedented, provided the specific statistic and
   its FDR control are made explicit (e.g., DiffCorr's Fisher z-test, or Sc-compReg's
   likelihood-ratio-style statistic with simulation-calibrated null).

4. **In-silico perturbation simulation from the inferred network itself, checked against
   literature-documented (not newly-collected) real perturbation phenotypes** — CellOracle's
   approach: don't collect new perturbation data for your system, instead simulate TF
   knockout in silico from your inferred GRN + trajectory and check whether the *direction*
   and *qualitative outcome* match already-published knockout phenotypes for that system.
   This is directly transferable to mHSC-E (hematopoiesis is CellOracle's own primary
   validated use case) and plausible for hESC (definitive endoderm differentiation, where
   FOXA2/SOX17/EOMES knockout phenotypes are well documented in the literature). Caution:
   scFM-based (Geneformer/scGPT) zero-shot in-silico perturbation is currently reported as
   weak (~14-15% top-k accuracy vs. real Perturb-seq) — if in-silico perturbation is used as
   a validation layer, it should be done CellOracle-style (simulate via the *inferred GRN's
   own structure/weights*, which is what the model being validated actually predicts) rather
   than by leaning on a scFM's own perturbation-prediction ability, which is a separate and
   currently under-validated capability.

5. **Orthogonal-omics or orthogonal-database concordance** — SCODE/motif databases
   (TF2DNA), DINGO's multi-platform overlay (mRNA+CNV+methylation concordance as internal
   cross-validation), BRIDGE's ChIPBase cross-check for novel predictions. Not directly
   available for the BEELINE datasets in this repo (no matched ATAC/ChIP per time point),
   but worth flagging as a stronger-tier option if any of hESC/mESC/mHSC-E turn out to have
   public matched chromatin accessibility data at the same time points (would be worth a
   quick check before ruling it out).

6. **Real perturbation data as confirmatory follow-up on a small subset of top findings**,
   not as the primary training/validation signal — CroCoNet's model. This is the
   "gold-standard partial credit" option the brief's §4 asks about: the honest answer from
   this review is that **no method in the deep-learning GRN space obtains new perturbation
   data to validate a differential/dynamic claim**; the one example of real perturbation
   validation found (CroCoNet) is a classical network-comparison paper from a wet lab with
   in-house CRISPRi capability, confirming an already-published cross-species finding, not a
   general validation recipe portable to a compute-only project. SupGCL uses *existing*
   public knockdown data as training supervision, which is the closer-to-portable option if
   a public perturbation dataset exists for hESC/mESC/hematopoiesis systems and were sourced
   as a *held-out validation subset* rather than full training data — this remains the
   "would strengthen but is not required" tier, consistent with the brief's own framing.

---

## 5. Recommendation for this project's exact datasets

Given hESC (Chu et al. time course), mESC (RamDA-seq time course), and mHSC-E (Nestorowa
stage gradient) each have **real, biologically meaningful within-lineage condition pairs but
only a single static per-cell-type reference network and no perturbation data**, the most
defensible validation strategy — directly justified by field precedent above, not
improvised — is a **three-tier combination**, matching what the strongest classical papers
in this review (BoostDiff, DCI, Sc-compReg) already do, adapted to the deep-learning/GNN
setting:

**Tier 1 (required, quantitative, dedicated metric): synthetic validation via GRouNdGAN.**
Generate two related synthetic single-cell datasets from two deliberately-related reference
GRNs (e.g., the real hESC BEELINE network and a programmatically-rewired variant with a
known set of removed/added/flipped edges), train and evaluate the differential model against
this **known ground-truth 4-way label**, and report a genuine differential-edge
AUROC/AUPRC. This directly answers "is the joint differential classifier statistically
sound at all" independent of any real-data ambiguity, exactly as BoostDiff/DCI/Sc-compReg
validate their statistics before applying them to real disease/cancer data. This is the
single most defensible way to report a headline quantitative number for the differential
task, since it is the only tier with actual ground truth.

**Tier 2 (required, silver-label training signal on the real data): classical
differential-co-expression/differential-network statistic, restricted to the static
reference network's candidate edge space.** Use DiffCorr-style Fisher z-test differential
correlation (simplest, most standard) or DINGO/jointGHS-style shared-vs-specific
decomposition, computed between the paired time points/stages, as the weak/silver training
label for the real hESC/mESC/mHSC-E data — exactly as the brief proposes in §4, and directly
precedented by this entire branch of the classical literature. Report the label's own FDR
control (e.g., Benjamini-Hochberg on the differential-correlation p-values) so the "ground
truth" being trained against has a stated, defensible statistical confidence level rather
than being presented as if it were a real label.

**Tier 3 (required, the field's dominant real-data validation and the brief's explicit
priority — "grounded in biology, not just numbers"): known-regulator enrichment, matching
what RegGAIN, SCENIC+, DINGO, BoostDiff, DINGO, and CroCoNet all do as their primary
real-data check.** For hESC: check that top-ranked A-vs-B-specific edges (e.g., 0h vs. 72h)
recover SOX17/FOXA2/EOMES turning on and POU5F1/NANOG/SOX2 turning off as *hub changes* in
the predicted differential edges, not just marginal expression change (this is a stronger
claim than differential expression alone — it means the *regulatory role*, not just the
expression level, of these genes should shift in the predicted network, which is the
distinguishing claim of a GRN-level method over the trajectory-DE methods like tradeSeq).
For mHSC-E: same test with GATA1/KLF1/TAL1 along the LT-HSC→Prog gradient. Report this as a
qualitative/enrichment table (as every field precedent does), not as a fabricated
quantitative "biological accuracy" score.

**Explicitly out of scope / not recommended as primary strategy**: sourcing new perturbation
data. No deep-learning method in this review does this for a differential/dynamic-GRN claim;
it is a wet-lab-only move (CroCoNet) that isn't portable to a compute-only project on this
timeline. If a *public* perturbation dataset already exists for one of these exact systems
(worth a targeted follow-up search — e.g., published hESC endoderm-differentiation TF
knockdown screens, or hematopoietic Perturb-seq data covering GATA1/KLF1/TAL1), it should be
used the SupGCL way — as a **held-out validation subset**, not baked into training — which
would meaningfully strengthen the paper's rigor claim over Tier 1-3 alone without requiring
new wet-lab work. This is worth a short, explicit follow-up literature check by Agent B or
the plan before ruling it out entirely, but should not block the plan's core design.

**One CellOracle-inspired optional Tier 4** worth the plan considering as a differentiator
from RegGAIN specifically: simulate the *predicted* differential edges' downstream effect
(e.g., in-silico "remove this A-specific edge" and check whether it moves the predicted
target's expression in the direction consistent with known knockout phenotypes reported in
the literature for that system) — this would be a genuinely novel validation angle none of
the reviewed GRN-contrastive-learning papers use, and squarely addresses the brief's
"biology, not just numbers" requirement while staying compute-only.

---

## 6. Sources

GP-DHT: https://arxiv.org/abs/2601.10995 · BRIDGE: https://arxiv.org/abs/2606.14734 ·
BRIDGE-GRN: https://www.biorxiv.org/content/10.64898/2026.05.12.724562v1 ·
SupGCL: https://arxiv.org/abs/2505.17786 , https://openreview.net/forum?id=YRh4epoqIS ·
RegGAIN: https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202518277 ,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12904006/ ·
Meta-TGLink: https://link.springer.com/article/10.1186/s13059-025-03860-8 ·
KEGNI: https://link.springer.com/article/10.1186/s13059-025-03780-7 ·
CroCoNet: https://www.biorxiv.org/content/10.1101/2025.11.18.689002v1 ·
scRegNet (base paper): https://www.biorxiv.org/content/10.1101/2024.12.16.628715v1 ·
DeepFGRN: https://academic.oup.com/bib/article/25/3/bbae143/7640736 ·
Briefings survey: https://academic.oup.com/bib/article/26/5/bbaf470/8257610 ·
DiffCorr: (Fukushima 2013, BMC Systems Biology) ·
DINGO: https://academic.oup.com/bioinformatics/article/31/21/3413/195238 ,
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4751246/ ·
CFGL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173447/ ·
jointGHS: https://arxiv.org/pdf/2206.11820 , https://github.com/Camiling/jointGHS ·
CSN/c-CSN: NAR 2019 (Dai et al.) ·
NeighbourNet: https://www.biorxiv.org/content/10.1101/2025.03.27.645629 ·
CSCN: https://www.biorxiv.org/content/10.1101/2025.10.09.681381 ·
Sc-compReg: https://www.nature.com/articles/s41467-021-25089-2 ,
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8346476/ ·
BoostDiff: https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae034/7616994 ,
https://www.biorxiv.org/content/10.1101/2022.09.26.509450v1 ·
DCI: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9991896 ,
https://par.nsf.gov/servlets/purl/10232169 ·
scTenifoldNet: https://www.biorxiv.org/content/10.1101/2020.02.12.931469 ·
DiCoLo: https://www.biorxiv.org/content/10.1101/2025.11.23.689932v2 ,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12704003/ ·
dynGENIE3: https://pmc.ncbi.nlm.nih.gov/articles/PMC5821733/ ·
SCODE: https://www.biorxiv.org/content/10.1101/088856.full.pdf ·
tradeSeq: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7058077/ ·
SCENIC+: https://www.nature.com/articles/s41592-023-01938-4 ·
CellOracle: https://www.nature.com/articles/s41586-022-05688-9 ·
GRouNdGAN: https://www.nature.com/articles/s41467-024-48516-6 ,
https://pmc.ncbi.nlm.nih.gov/articles/PMC11525796/ ·
PSGRN (self-training synthetic gold standard): https://www.science.org/doi/10.1126/sciadv.aeb3376 ·
scFM in-silico perturbation evaluation: https://www.biorxiv.org/content/10.1101/2025.05.11.653338 ,
https://www.biorxiv.org/content/10.1101/2025.09.22.677811 ·
RECOMB 2026 accepted papers: https://recomb.org/recomb2026/accepted_papers
