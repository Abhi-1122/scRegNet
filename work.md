# Work Log (branch: differential-grn)

Terse, dated log of methodology ideas tried on this branch — just enough to understand
each idea. Numbers go in `results.txt`. This branch is a clean cut from `summer26`
commit `8a3f58a` (predates the PPI/PANDA/scFM-swap/tuning line of work, which lives on
`summer26` — see that branch's `work.md`/`results.txt`/`PPI_EXPERIMENTS_SUMMARY.md` for
the full history that led here).

## 2026-08-28

**Why this branch exists:** systematic tuning/fusion/scFM-swap work on `summer26`
converged on a saturation ceiling for static single-condition edge prediction (see that
branch's closing entry). Pivoting to a differential/contrastive framing instead: predict
*which* regulatory edges change between two conditions (e.g. cell-differentiation
branches), not just whether an edge exists in one condition. Literature check (arXiv):
closest competitors are GP-DHT (cross-*species* conserved-vs-specific edges, supervised
contrastive learning, same BEELINE benchmark), BRIDGE (self-supervised contrastive GNN,
cross-cell-type few-shot *transfer*, not differential prediction), and SupGCL (ICML 2026,
supervised contrastive learning from real knockdown perturbations, patient/cancer GRNs).
None do joint, cross-condition, *simultaneously-trained* differential-edge prediction
with its own benchmark metric — that's the gap this branch targets.

**Bug found and fixed while setting up this branch:** `src/models.py` at commit `8a3f58a`
has a flat import (`from cross_modal_attention import ...`) that only resolves when
`src/` itself is on `sys.path`, but `src/train.py` uses package-style imports
(`from src.models import ...`) requiring repo-root on `sys.path` — the two are
incompatible in one invocation, so `python3.12 -m src.train` failed outright on this
commit. Fixed with the same try/except pattern already used elsewhere in the later
`summer26` history (try flat, fall back to `src.`-prefixed). One line, no other changes.

**Baseline established -- no rerun needed:** paper's Table 2 already reports hESC/TFs+500
scRegNet(w/Geneformer) = AUROC 0.88, AUPRC 0.62 (2-decimal precision). `summer26`'s own
rerun from earlier this project (before any PPI/PANDA additions) independently
reproduced this: 0.8874/0.6225. Briefly started a third rerun on this branch specifically
(reasoning: "confirm the code this branch inherited, post-import-fix, still reproduces
it") and then stopped it -- that reasoning didn't hold up, since the import fix above is
a pure module-path resolution change with no effect on computation, so it cannot change
the result. Re-running to verify a no-op fix is redundant. The paper number + the
existing `summer26` rerun are the baseline; using those as-is.

**Data gap (unresolved):** this repo only ships 4 of BEELINE's 7 cell types (hESC, hHEP,
mESC, mHSC-E) -- `data/mHSC-GM` and `data/mHSC-L` are absent, as are their Geneformer
embeddings, and no embedding-generation pipeline exists in this repo for any scFM
(the shipped Geneformer embeddings were downloaded pre-made). Acquiring mHSC-GM/mHSC-L
-- the other two branches of the HSC differentiation lineage, the strongest biological
motivation for the differential idea -- would mean redoing BEELINE's raw preprocessing
pipeline from scratch, a real side-task not yet started.

## 2026-08-28 (cont'd) -- Execution phase begins, scoping decision

`DIFFERENTIAL_GRN_PLAN.md` (Agent B's technical plan, adversarially reviewed) is now the
spec for this branch. Full plan calls for ~63 training runs (9 pairs x 5 GCN configs + 9
naive baselines + 6 SAGE/GAT spot-checks + 3 synthetic severities). **This pass explicitly
reduces scope** to keep the work tractable in one execution session:

- Phase 0: labeling-yield dry run, all 9 pairs, CPU-only (plan Sec 7 Phase 0). Done first,
  before any training code, exactly as the plan recommends.
- Phase 1: architecture + trainer built once, verified end-to-end on exactly one pair
  (hESC 00h vs 96h), GCN backbone only (SAGE/GAT skipped -- deferred robustness spot-check
  per plan's own Sec 6 framing).
- Phase 2: reduced core matrix -- `full` (9 pairs) + `posthoc_baseline` (9 pairs) +
  `naive_expression_baseline` (9 pairs) + `no_scfm`/`no_grn`/`no_contrastive` (3
  primary/bookend pairs only: hESC 00h-vs-96h, mESC 0h-vs-72h, mHSC-E LT_HSC-vs-Prog).
  That's 9+9+9+9 = 36 runs, not 63. No hyperparameter search -- reusing this codebase's
  existing static-baseline GNN hyperparameters unmodified (gnn_epochs=300, gnn_dim_hidden=
  256, dropout=0.2, gnn_lr=5e-4, batch_size=256, mlp_num_layers=2 etc, from `src/args.py`
  defaults) per the task's explicit instruction not to tune.
- Phase 3: GRouNdGAN itself out of scope (plan's own Sec 7 flags it "multi-day
  integration"); uses the plan's documented fallback -- a parametric negative-binomial
  simulator seeded from hESC/TFs+500's real reference network + real per-gene mean/
  dispersion, one rewiring severity (moderate, ~15%), one cell type. Other severities/cell
  types deferred.
- SAGE/GAT robustness spot-check (6 runs) and the full 3-severity synthetic sweep are
  **deferred**, not run this pass -- both are explicitly named as reducible/deferrable in
  the plan itself (Sec 6, Sec 7), not corners cut against the plan's spirit.
- `src/tuning/` porting from `summer26` (plan Sec 7) is **not done** -- unnecessary since
  this pass runs no HP search, only fixed-hyperparameter runs launched directly.

### Phase 0 results (labeling-yield dry run, all 9 pairs)

Ran `python3.12 -m src.differential_labels` (CPU-only, ~5s total for all 9 pairs). Full
counts also cached per-pair at `data/{cell_type}/TFs+{num_TF}/differential/{A}_vs_{B}_labels.csv`
and summarized at `data/phase0_yield_summary.csv`.

| cell_type | pair | priority | candidates | conserved | A-spec | B-spec | neither | ambiguous | inconsistent |
|---|---|---|---|---|---|---|---|---|---|
| hESC | 00h_vs_96h | primary | 4545 | 0 | 0 | 9 | 4468 | 0 | 68 |
| hESC | 00h_vs_12h | secondary | 4545 | 0 | 0 | 0 | 4545 | 0 | 0 |
| hESC | 36h_vs_72h | secondary | 4545 | 5 | 1 | 5 | 4484 | 0 | 50 |
| mESC | 0h_vs_72h | primary | 29613 | 32 | 71 | 175 | 28359 | 6 | 970 |
| mESC | 0h_vs_12h | secondary | 29613 | 46 | 20 | 129 | 28333 | 2 | 1083 |
| mESC | 24h_vs_48h | secondary | 29613 | 222 | 35 | 11 | 27281 | 2 | 2062 |
| mHSC-E | LT_HSC_vs_Prog | primary | 11557 | 279 | 15 | 810 | 9277 | 18 | 1158 |
| mHSC-E | LT_HSC_vs_HSPC | secondary | 11557 | 9 | 16 | 0 | 11153 | 0 | 379 |
| mHSC-E | HSPC_vs_Prog | secondary | 11557 | 7 | 0 | 2134 | 9373 | 5 | 38 |

**Honest finding, worse than the plan anticipated for hESC specifically**: hESC's yield is
much thinner than mESC/mHSC-E's. `00h_vs_96h` (the designated hESC primary/bookend pair)
has **zero** conserved and zero A-specific edges, and only 9 B-specific -- all far below the
plan's own Sec 5.1 class-count floor (>=20 test examples) even before any train/test split.
`00h_vs_12h` yields literally nothing in any of the 4 trainable classes (100% neither).
Root-caused (not a code bug): condition-A (`00h`, pluripotent, n=92) shows almost no
detectable pairwise correlation among candidate edges at all -- only 25/4545 edges reach
`|r_A|>=0.3` and raw (pre-BH) `p_A<0.05` hits only 4.75% of edges, i.e. barely above the 5%
expected by chance alone, so BH-FDR correctly zeroes out `sig_A` entirely for this pair.
`96h` (n=188, post-differentiation) shows more signal (77/4545 pass `|r_B|>=0.3`, `sig_B`
count 77). This is a real biological/statistical fact about this dataset (a young,
homogeneous pluripotent population has flatter pairwise co-expression than a
differentiating one), not a defect in the labeling code -- verified by inspecting the raw
r/p distributions directly. mHSC-E's `LT_HSC_vs_Prog` A-specific count (15) is also under
the floor. mESC's `0h_vs_72h` is the best-balanced pair of all 9 (32/71/175 across
conserved/A-spec/B-spec) and is the pair most likely to produce a fully-reportable 4-class
result.

**Decision per task instructions ("deprioritize, don't drop")**: no pair is dropped. All 9
proceed to Phase 2, but hESC's pairs (especially `00h_vs_12h`, which literally cannot train
a 4-way head since 3 of 4 classes have zero examples) are flagged low-yield up front. This
finding is carried into Phase 1's biological-check interpretation below: the hESC
POU5F1/NANOG/SOX2/SOX17/GATA4/GATA6 check can only ever be evaluated against a
handful of the model's most-confident predictions, not a well-powered A-specific/B-specific
test split, and any resolution should be read with that caveat front and center.

### Phase 1 results (architecture + trainer, hESC 00h vs 96h)

Implemented `scDiffTransNet_GCN` in `src/models.py` (Sec 4.2, GCN backbone only per this
pass's scoping) and `src/train_differential.py` (`DiffTrainer`, Sec 4.4), plus the new args
in `src/args.py` (Sec 4.4's full list). Fixed hyperparameters used throughout Phase
1/2/3 -- the SAME hyperparameters that produced this codebase's existing hESC/TFs+500
static baseline (0.8874/0.6225 AUROC/AUPRC), found by the pre-existing Optuna search
(`out/GCN/Geneformer/tf_500_hESC/best/ckpt/args.json`), reused as-is per the "no new HP
search" instruction: `gnn_hidden_dims=[227,29,34]`, `mlp_hidden_dims=[198]`,
`gnn_num_layers=3`, `mlp_num_layers=1`, `dropout=0.1748`, `gnn_lr=0.00144`,
`gnn_weight_decay=1.04e-5`, `gnn_dim_hidden=256`, `batch_size=193`, `type=MLP`,
`gnn_epochs=300` (early-stopped on macro-F1, patience=10 evals).

**Architecture-sound verdict: YES.** Trained end-to-end without errors; converged in 64
epochs (~2.5 min on the project GPU). The auxiliary static link-prediction heads (one per
condition, sharing all GNN/fusion/MLP weights) reached AUROC 0.873/0.874, AUPRC
0.613/0.616 on the shared static Test_set -- close to the existing single-condition
baseline (0.8874/0.6225), confirming the Siamese shared-adjacency design does not break
the base static task it is built on top of.

**Biological check verdict: FAILS to resolve, and this is a direct, expected consequence
of Phase 0's finding, not a separate architecture failure.** Ran
`model.predict_all_candidates()` over the full 4,545-edge hESC candidate space.
POU5F1 (87 out-edges), NANOG (346), SOX2 (254), SOX17 (23), GATA4 (45), GATA6 (22) are
predicted **100% "neither"** -- zero A-specific, zero B-specific, zero conserved calls for
any of the 6 named genes' out-edges. The model collapsed to predicting the majority
("neither") class almost everywhere (4527/4545 candidate edges), because the 00h_vs_96h
diff-train split had literally 0 conserved and 0 A-specific training examples and only 6
B-specific examples out of 3134 trainable edges (Phase 0's finding) -- there was no
A-specific training signal at all for the model to learn to call POU5F1/NANOG/SOX2's
edges 00h-specific, and only 6 real B-specific examples anywhere for SOX17/GATA4/GATA6's
expected 96h-specific pattern. Reported honestly per task instructions (Sec 5.4): this is
a genuine negative result for hESC specifically, root-caused to labeling yield, not to the
architecture -- Phase 3's synthetic check (below) independently confirms the architecture
itself CAN learn real 4-way differential structure when given a less pathologically thin
label distribution.

### Phase 3 results (synthetic sanity check, parametric NB simulator, moderate/hESC only)

Plan Sec 5.2's own documented fallback (GRouNdGAN out of scope, Sec 7). Implemented
`src/phase3_synthetic.py`. First attempt (reusing the REAL hESC reference network's full
multi-regulator topology to drive simulated co-expression, mean/dispersion NB-fit directly
to `BL--ExpressionData.csv`'s values) produced a **degenerate, uninformative synthetic
system** on inspection -- two bugs found and fixed before trusting any downstream number:

1. `BL--ExpressionData.csv` is BEELINE-preprocessed **log2(count+1)** data (verified: floats
   in [0, ~19.3], ~28% exact zeros), not raw counts. Fitting an NB distribution's
   mean/variance directly to log-scale values collapses almost all real per-gene variance,
   giving a near-degenerate dispersion fit. Fixed by inverting the log2(x+1) transform
   before the NB method-of-moments fit.
2. Even after fixing (1), simulating co-expression from the REAL network's full topology
   gave **no detectable separation** between true "conserved" edges and true random
   "neither" pairs (mean |r| 0.13 vs 0.14 at beta=6) -- root-caused to two real properties
   of this specific reference network: (a) mean in-degree ~5.6 (most targets have several
   independent regulators), which dilutes any *single* TF-target pair's marginal
   correlation well below detectability at n~92-188 cells -- a real, honest echo of Phase
   0's hESC finding, but one that makes a synthetic *calibration* check uninformative if
   left in; (b) one mega-hub TF (TFAP2A, out-degree 722 = touches 79% of the 910-gene
   panel) means most "random" gene pairs are not actually an independent negative control,
   since they are likely to share it as a common regulator.
   **Fix (documented simplification, not hidden):** replaced the real network's topology
   with a simplified, single-regulator-per-target synthetic network for the *injected
   correlation structure* only -- built from real TF identities (drawn from the real
   `TF.csv`) and real target-gene identities (drawn from the real gene panel), one hub TF
   (250 private targets) + 15 regular TFs (6 private targets each), with severity=0.15
   rewiring applied to the hub TF (loses 38 out-edges) and a regular TF (gains 38 new
   out-edges) -- and "neither" ground-truth pairs drawn only from genes never used as a
   regulator or target anywhere in the construction, guaranteeing a clean negative control.
   This trades away the real network's combinatorial-regulation realism specifically to
   keep the calibration check meaningful, and is reported as such rather than silently
   presented as "the real network's structure."

**Result, one severity (moderate, ~15%) x one cell type (hESC), as scoped:**
- **(a) Trained `full` model vs TRUE (by-construction) labels**: macro-F1 0.759,
  macro AUROC (one-vs-rest, macro-averaged) **0.880**, macro AUPRC 0.740. Per-class AUROC:
  conserved 0.879, A-specific 0.641, B-specific 1.000, neither 1.000. Clears the plan's own
  Sec 6 stop-condition bar (target >=0.75) with room to spare -- the architecture DOES learn
  genuine 4-way differential structure when trained against clean, non-degenerate labels.
  Confusion matrix shows the one real weak spot: 10/11 true A-specific test edges were
  misclassified as conserved (A-specific vs conserved is the harder distinction, as
  expected -- both require significant within-condition correlation, differing only in
  whether it's shared or not).
- **(b) Sec 3 silver-labeling procedure vs TRUE labels (calibration check)**: 92.7%
  accuracy across all candidate rows (counting the silver procedure's own
  ambiguous/inconsistent calls as automatic non-matches, since the true-label space has no
  such categories), **97.2% accuracy restricted to the trainable 4-class subset**. This is
  a strong calibration result: the DiffCorr/Fisher-z silver-labeling statistic, run
  completely unmodified through this synthetic pair, recovers the true construction labels
  correctly the large majority of the time.
- n_removed=n_added=38, n_conserved=302, n_neither=120 (before any train/test split);
  A-specific/B-specific test sets are small (11 each after the 70/30 split) since this is
  one moderate-severity run at modest edge counts -- reported as-is, not padded.
- **Deferred**: mild/severe severities and mESC/mHSC-E synthetic variants, per work.md's
  scoping entry above.
