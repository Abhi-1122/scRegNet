# Differential/Contrastive GRN — Idea Brief (input to lit-review + plan agents)

Branch: `differential-grn` (cut from `summer26`@`8a3f58a`, predates PPI/PANDA work).
This file is the shared context handoff for the multi-agent research pipeline building
this idea into a paper-worthy, implemented, run project. Do not duplicate `work.md`
history — read that too, it has the terse dated log.

## 1. Origin idea (as pitched to the user, condensed)

Starting point: GEARS-style perturbation models use a fixed, static PPI graph as prior
structure and never model that regulatory wiring itself changes across biological state.
Proposed direction: a **contrastive/differential GNN** that, given two conditions
(e.g. two time points or two branches of a differentiation trajectory), predicts which
regulatory edges are *conserved* across both, *specific* to condition A, *specific* to
condition B, or in *neither* — a 4-way differential edge label — instead of just
predicting "is this edge real" in one static condition. Motivation: static single-condition
link prediction (what `summer26`'s scRegNet work did) looks saturated; the biological
variance across conditions is a much less picked-over target and is closer to what a
biologist actually wants to know (what rewires during differentiation), not just "does
this edge exist."

Working name so far: differential-GRN / contrastive-GRN.

## 2. Literature already scoped (arXiv only, earlier pass — treat as a *floor*, not complete)

- **GP-DHT**: cross-*species* conserved-vs-specific regulatory edges via supervised
  contrastive learning, evaluated on BEELINE-style benchmarks. Closest in mechanism
  (contrastive + conserved/specific edge typing) but the axis of comparison is species,
  not within-species biological condition/time.
- **BRIDGE**: self-supervised contrastive GNN pretraining, used for cross-cell-type
  *few-shot transfer* of GRN inference (i.e. learn on cell type A, transfer to predict
  static network of cell type B). Not differential — never jointly predicts how edges
  change between two conditions, no 4-way differential label, no differential metric.
- **SupGCL** (ICML 2026): supervised contrastive learning using *real experimental
  knockdown/knockout perturbations* as ground truth, applied to patient/cancer GRNs.
  Real perturbation ground truth is the strongest supervision signal seen so far in this
  space — worth understanding closely since our datasets don't have this.

None of these three do joint, simultaneously-trained, differential-edge prediction across
paired conditions with a dedicated differential-edge metric. That was the claimed gap.
**This claim needs re-verification with a wider, non-arXiv-only search (see task for
lit-review agent below) before it's trusted for a paper-novelty claim.**

## 3. IMPORTANT correction to the earlier "data gap" conclusion

An earlier pass concluded mHSC-GM/mHSC-L (other HSC lineage branches) were needed and
absent, and that acquiring them would require redoing BEELINE's raw preprocessing from
scratch. **That undersold what's actually already in the repo.** Just checked the raw
`BL--ExpressionData.csv` cell-barcode headers directly:

- **hESC** (`data/hESC/TFs+{500,1000}`): cell barcodes are `H9_00hb4s_*`, `H9_12h_*`,
  `H9_24h_*`, `H9_36h_*`, `H9_72h_*`, `H9_96h_*` — this is Chu et al. 2016's H9 hESC
  **definitive-endoderm differentiation time course**, 6 explicit time points, encoded
  directly in the cell names.
- **mESC** (`data/mESC/TFs+{500,1000}`): barcodes are `RamDA_mESC_00h_*`,
  `_12h_*`, `_24h_*`, `_48h_*`, `_72h_*` — a RamDA-seq mESC differentiation time course,
  5 explicit time points.
- **mHSC-E** (`data/mHSC-E/TFs+{500,1000}`): barcodes are `LT_HSC_*` (long-term HSC,
  the progenitor state), `HSPC_*`, `Prog_*` — Nestorowa et al. hematopoiesis data,
  an explicit **differentiation-stage gradient** (stem → progenitor) within the erythroid
  branch, not raw noise.
- **hHEP** (`data/hHEP/TFs+{500,1000}`): barcodes are plain 96-well-plate coordinates
  (`A1`..`H12`), no time/stage encoded in the name itself — would need external metadata
  to recover any staging; treat as *not* usable for within-lineage differential pairing
  unless/until such metadata is found.

**So 3 of the 4 already-available cell types have genuine, real, biologically meaningful
within-lineage condition pairs sitting in data already in this repo** (time point A vs
time point B within hESC; within mESC; stage A vs stage B within mHSC-E). This is very
likely a *better* framing than the original mHSC-GM/mHSC-L cross-branch idea anyway —
temporal differential regulation during differentiation is a cleaner, more standard
biological question than comparing across terminal branches, and requires zero new data
acquisition or preprocessing. The mHSC-GM/mHSC-L gap is no longer a blocker; it's now
just a *possible* future extension, not a dependency.

## 4. IMPORTANT: no ground-truth differential-edge labels exist — must be designed

Checked `BL--network.csv` (the reference regulatory network BEELINE/scRegNet trains
against) per cell type: it is a **single static reference network per cell type**
(4,546 edges for hESC; 29,614 for mESC; 11,558 for mHSC-E), built from ChIP-seq/curated
DB evidence, **not time-point- or stage-resolved**. There is no experimentally-derived
"this edge is present at 12h but not 72h" label anywhere in this repo's data.

This is the central open methodological problem the plan must solve: what supervises
the "conserved / A-specific / B-specific / neither" 4-way label, given the reference
network doesn't distinguish conditions? Candidates to investigate (not prescriptive —
this is exactly what the plan agent + lit review need to resolve and defend biologically):
- Statistical differential-co-expression/differential-network methods over the raw
  expression data, binned by time point/stage (e.g. differential correlation, mutual
  information, or existing tools like DiffCorr/DINGO/CSN), used as a **weak/silver**
  differential label, with the static reference network still used to restrict which
  TF-target pairs are even candidates (a differential edge must first be a real edge in
  the reference network in *at least one* condition, or in the general TF-target
  candidate space scRegNet already trains on).
- Evaluating differential predictions not by AUROC/AUPRC against a differential ground
  truth alone but partly through **biological plausibility checks**: do top-ranked
  differential edges recover known stage-specific regulators for these specific systems
  (e.g. hESC definitive-endoderm TFs like SOX17, FOXA2, EOMES turning on; pluripotency
  factors POU5F1/NANOG/SOX2 turning off; known HSC erythroid regulators like GATA1,
  KLF1, TAL1 activating along the LT_HSC→Prog gradient). This is what "grounded in
  biology, not just numbers" should concretely mean here — the user explicitly does not
  want a metrics-only paper.
- SupGCL's real-perturbation-ground-truth approach is the gold standard but we don't have
  perturbation data for these systems — the plan should explicitly state whether/how it
  gets partial credit for methodological rigor without that data, or whether a public
  perturbation dataset should be sourced for a validation subset.

## 5. Existing codebase capabilities to reuse (don't reinvent)

- `src/models.py`: `scTransNet_{GCN,GraphSAGE,GAT}` — dual-branch encoder: a graph branch
  (GNN over the *training* TF→target adjacency, built from **train-positive edges only**,
  transductive) and an scFM node-embedding branch (Geneformer/scGPT/UCE gene embeddings,
  condition-*independent* per gene — same embedding regardless of time point except
  Geneformer which is pooled per-dataset), fused via `IterativeCrossModalFusion`
  (`src/cross_modal_attention.py`) or plain concat depending on config, decoded via a
  learned bilinear/MLP `decode(tf_embed, target_embed)` into an edge score.
  **Important wrinkle for the differential design**: the adjacency fed to the GNN branch
  is itself built from the train-split positive edges of the (single, static) reference
  network — there's no notion of a condition-specific adjacency today. A differential
  model needs to decide whether both conditions share one adjacency (structure fixed,
  only node features/embeddings vary by condition) or get condition-specific adjacencies
  (requires the weak/silver differential labels from §4 to even construct).
  Ablation flags already exist: `--ablation {full,no_scfm,no_grn}` isolates each branch's
  contribution — reuse this pattern for a differential ablation study.
  `--node_feature_source {scfm,panda}`, `--use_panda_edge` — PANDA integration from the
  `summer26` line, present but not required for this idea.
- `src/build_panda.py` — classical network-integration prior; possibly relevant as one
  candidate source of a differential prior across time-binned expression, since PANDA
  already knows how to build a regulatory prior from expression + PPI + motif evidence.
- Baseline numbers already established for hESC/TFs+500, static single-condition task:
  paper 0.88/0.62 AUROC/AUPRC, `summer26` rerun 0.8874/0.6225 — **not directly comparable**
  to a differential-edge task (different task, different metric), but is the reference
  point for "how good is static prediction on this exact data" that any differential
  approach should be understood relative to.
- `src/tuning/` — modular, race-condition-safe, multi-GPU HP tuning infra built for this
  project's HPC (4 GPUs), reusable for the new model with a new `EXPERIMENTS` entry.

## 6. What the two agents are being asked to do

**Agent A (literature review)**: thorough, *not* arXiv-only survey. Must cover both the
modern contrastive/GNN deep-learning literature (extend §2, verify/refute the novelty
claim) AND the classical/statistical differential-network-biology literature (this is an
establshed field older than deep learning — differential co-expression networks,
differential GRN inference from time-course/trajectory data, pseudotime-branch regulon
methods like SCENIC+, tools like DINGO/DiffCorr/CSN, dynGENIE3, SCODE, tradeSeq). Also
specifically research: how do existing differential/dynamic GRN methods obtain ground
truth or validation when no perturbation data exists, since that's exactly our situation.
Cover established academic venues beyond arXiv (bioRxiv, Bioinformatics, Genome Biology,
Nature Methods, Cell Systems, ISMB/RECOMB proceedings), via WebSearch/WebFetch in addition
to the arXiv MCP tools already available. Output a written report to a file.

**Agent B (plan)**: given Agent A's report + this brief + a read of the actual codebase
(`src/models.py`, `src/train.py`, `src/args.py`, `src/build_panda.py`), produce a concrete,
implementable technical plan: exact architecture, exact differential-label construction
method (with explicit biological justification, not just "use a stats test"), exact
evaluation protocol (both a quantitative differential metric AND a biological-plausibility
validation using known regulators per §4), exact experiments/ablations, and an explicit
statement of what would make this paper-worthy vs. what's just an incremental tweak.

Then Agent A and Agent B go back and forth (orchestrated by the main session, not by the
agents calling each other directly): Agent A critiques Agent B's plan specifically for
(a) biological inconsistency or hand-waving, (b) novelty gaps against the literature it
found, (c) infeasibility given this repo's actual data. Agent B revises. Repeat until they
reasonably converge. Only then does implementation start.
