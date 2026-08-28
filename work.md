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

## 2026-08-28 (cont'd) -- mid-Phase-2 scope change (from coordinator) + perf fix

**Perf fix**: found mid-Phase-2 that mESC's `full` runs were taking 40+ min each (vs
hESC's ~2.5 min) -- root-caused to the training loop recomputing the diff-head forward
pass over the ENTIRE diff-train edge set (up to ~20K edges for mESC) on every one of the
~341 static mini-batch steps/epoch (mESC's `Train_set.csv` is 65,898 rows vs hESC's
20,677, so ~3.2x more static steps/epoch too, compounding). Killed the in-flight mESC run
(no completed metrics.json lost) and added `--diff_batch_cap` (default 4096, new arg in
`src/args.py`) to `DiffTrainer.train()`: the diff edges are now mini-batched too (a fresh
random subset up to the cap, resampled every step), same spirit as the existing static
mini-batching. This changes ONLY the stochastic mini-batching granularity of an already-
full-batch-equivalent training signal (every diff edge is still visited over the course of
an epoch, just not all-at-once every single step) -- it does not touch any of the fixed
architecture/loss hyperparameters carried over from the baseline, so it isn't a
hyperparameter-search change. Re-launched the `full` stage from scratch (skip-logic
correctly skipped hESC's 3 already-completed pairs).

**Scope change from the user (relayed by coordinator), recorded verbatim as instructed**:
skip the `no_scfm`/`no_grn`/`no_contrastive` ablation runs for this pass entirely -- to be
run later only if the core results look promising. None of the 9 planned ablation runs (3
primary pairs x 3 ablations) had been started yet at the time of this message, so nothing
is reverted -- they simply do not run this pass. `run_phase2.sh ablations` stage is left
in place in the script (for a later pass) but is not invoked.

Remaining scope for this pass, per the coordinator's message: `full` (9 pairs) +
`posthoc_baseline` (9 pairs) + `naive_expression_baseline` (9 pairs, already done) + the
plan's Sec 5.3 named biological checks against `full`'s output for hESC, mESC, and
mHSC-E -- prioritizing mHSC-E's GATA2->GATA1 flagship check and mESC's naive-circuit
collapse check, since Phase 0's yield numbers suggest both are far better-powered than
hESC's checks.

### Phase 2 execution notes: concurrency, a perf fix, and a mid-run diagnostic

Running all 27 GPU jobs (9 full + 9 posthoc_baseline; naive_expression_baseline is CPU-only
and finished in 18s for all 9) took substantially longer wall-clock than Phase 1's single
hESC run suggested, entirely because of `Train_set.csv` size differences across cell types
(mESC: 65,898 rows / ~341 static mini-batches per epoch vs hESC's 20,677 rows / ~107
batches) and, separately, GPU-contention effects from running too many concurrent
processes on one 8GB GPU:

- Fixed a real inefficiency: `DiffTrainer.train()` was recomputing the full diff-edge
  forward pass (`forward_pair` over ALL training diff-edges, up to ~20K for mESC) on every
  one of the ~341 static mini-batch steps per epoch. Added `--diff_batch_cap` (default
  4096, new arg) to mini-batch the diff edges too (fresh random subset per step, same
  spirit as the pre-existing static mini-batching) -- committed before any mESC/mHSC-E
  `full` runs were (re-)launched.
- Ran 5 mESC jobs concurrently at one point (2 `full` + 3 `posthoc_baseline`) to use spare
  GPU headroom; this caused severe contention (observed 108s/epoch vs ~17.5s/epoch
  uncontended, a ~6.2x slowdown) that more than erased the concurrency benefit. Coordinator
  killed the 5 stuck processes and asked for a profiling check to rule out an O(n^2)
  scaling bug in the loss (the 47x observed vs. 2.3s/epoch hESC baseline lined up
  suspiciously with mESC's ~6.5x larger candidate-edge count, squared).
- **Profiled directly** (isolated, uncontended reruns of mESC `0h_vs_12h`, 3-epoch and
  10-epoch): consistently ~17.5-18.5s/epoch regardless of epoch count -- ruled out any
  O(n^2)/growing-cost bug (`diff_batch_cap` was already correctly bounding the CE +
  contrastive loss to <=4096 rows per step). The observed 47x factor decomposes as ~7.6x
  legitimate (mESC's 3.2x larger `Train_set.csv` -> more static batches/epoch, plus some
  residual overhead) x ~6.2x from the since-abandoned 5-way GPU contention -- not a single
  bug, a combination of a real (now understood, not "fixed" since it isn't a bug) scaling
  factor and an over-aggressive concurrency choice.
- Relaunched the remaining mESC jobs at 2-way concurrency (per coordinator's guidance),
  which gave a real, if modest, throughput gain (~1.9x slowdown per process vs ~6.2x at
  5-way) with no further code changes. All 27 Phase 2 runs completed at that concurrency
  level. `src/progress_watch.py` (backfilled + live-updated `PROGRESS.md`) was added
  mid-run per the coordinator's request and used throughout to track status without
  digging through logs.

### Phase 2 results: core matrix (full vs posthoc_baseline vs naive_expression_baseline, macro-F1, all 9 pairs)

| cell_type | pair | full | posthoc_baseline | naive_expression_baseline |
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

**`full` vs `posthoc_baseline`** (the central novelty claim, plan Sec 6): `full` wins on
every single pair, usually by a huge margin. This is real but partly an artifact of
`posthoc_baseline`'s documented structural limitation (no "neither" call at all -- see
`src/posthoc_baseline.py`'s docstring -- so it can never score above ~0 macro-F1 whenever
"neither" dominates the test set, which is true for all 9 pairs given the severe class
imbalance Sec 7 predicted). This should be read as "the joint model handles the dominant
`neither` class, which the post-hoc adaptation structurally cannot" rather than "the joint
model is uniformly ~100x better at the differential task per se" -- an honest reading, not
a rhetorical one, per the plan's own instruction not to reframe a mechanistic difference
as a bigger win than it is.

**`full` vs `naive_expression_baseline`** (is graph+scFM structure adding anything over
gene-level statistics alone): `full` beats naive by >=5-10 macro-F1 points (the plan's own
target) on **7 of 9 pairs** -- mESC's all three pairs (+31.5, +8.5, +23.7 pts) and
mHSC-E's all three pairs (+40.1, +15.3, +26.6 pts) -- and ties naive on hESC's two
~100%-"neither" pairs (00h_vs_96h, 00h_vs_12h) where there is no room to improve (both
already near-ceiling). hESC's 36h_vs_72h pair is the one case where `full` clearly beats
naive on a hESC pair (0.9985 vs 0.8176, +18.1 pts). This is a genuinely positive result
against the plan's Sec 6 framing for `full` vs `naive_expression_baseline` specifically --
the architecture is not just re-deriving the label statistic through a more expensive path.

### Phase 2 results: named biological checks (plan Sec 5.3), `full` model, all 3 cell types

**hESC** (00h_vs_96h, 00h_vs_12h, 36h_vs_72h): **ALL FAIL TO RESOLVE.** POU5F1 (87
out-edges), NANOG (346), SOX2 (254) show 0% A-specific in 00h_vs_96h. SOX17 (23), GATA4
(45), GATA6 (22) show 0% B-specific in 00h_vs_96h, and only 0-4.5% B-specific in
36h_vs_72h (GATA4: 1/45, GATA6: 1/22, SOX17: 0/23). EOMES shows 0% B-specific in
00h_vs_12h. The model predicts ~100% "neither" for essentially every named gene's
out-edges across all 3 hESC pairs. This is a direct, expected consequence of Phase
0/1's already-established finding (hESC's condition-A correlation signal is
near-absent at n=92, so the silver labels themselves have almost no A-specific/conserved
examples to train against) -- not a new or separate failure mode discovered here.

**mESC naive-pluripotency-circuit collapse** (0h_vs_72h, 0h_vs_12h): **FAILS to cleanly
resolve.** Every naive-circuit gene (NANOG, POU5F1, SOX2, KLF4, KLF2, ESRRB, TBX3, ZFP42,
PRDM14) shows <3% A-specific out-edges in both pairs -- overwhelmingly "neither". Three
genes (POU5F1: 65 B-specific vs 2 A-specific; ESRRB: 57 B-specific vs 16 A-specific; TBX3:
7 B-specific vs 0 A-specific, all in 0h_vs_72h) show *more* B-specific (72h-specific) than
A-specific (0h-specific) edges -- the opposite of the predicted direction, though absolute
counts are a small fraction of each gene's total out-edges. OTX2 (predicted 72h-specific)
shows 0% B-specific. Unlike hESC, mESC's silver-label yield for this pair is not thin
(Phase 0: 32 conserved / 71 A-specific / 175 B-specific for 0h_vs_72h) -- so this is a
genuine negative finding about what the trained model recovered for these specific named
genes, not a labeling-yield artifact. Reported honestly as a failure, per plan Sec 5.4.

**mHSC-E erythroid-commitment checks** -- the best-resolved cell type, with one clear
positive result and several inconclusive/negative ones:

| Check | Result |
|---|---|
| **GATA2->GATA1 flagship edge** | **PARTIAL PASS.** `LT_HSC_vs_Prog`: predicted "neither" (fail). `HSPC_vs_Prog`: predicted "**B-specific**" with 100% model confidence (pass) -- correctly flagged as strongly differential in exactly the pair the plan itself named as where the GATA-switch effect should be sharpest (the erythroid-commitment step). **This is the single most citable positive biological finding of Phase 2** -- a single trained model recovering the textbook GATA2->GATA1 switch direction with full confidence, in the mechanistically correct pair. |
| GATA2 other out-edges (567) | FAIL to confirm predicted LT_HSC/HSPC+conserved enrichment. `LT_HSC_vs_Prog`: 40% B-specific, only 0.5% A-specific+conserved (opposite of predicted direction). `HSPC_vs_Prog`: 96% B-specific, 0% A-specific+conserved. |
| SPI1 (PU.1) out-edges (672) | FAIL to confirm predicted LT_HSC/HSPC skew. `LT_HSC_vs_Prog`: 0% A-specific (30% B-specific, 69% neither). `HSPC_vs_Prog`: 96% B-specific. |
| GFI1 vs GFI1B (604/257) | Mixed. `LT_HSC_vs_Prog`: GFI1 100% "neither" (uninformative, no signal at all); GFI1B 39% "conserved" -- loosely consistent with the predicted "conserved-active" pattern, the one partial confirmation in this row. `HSPC_vs_Prog`: both GFI1 (96%) and GFI1B (93%) predicted B-specific -- GFI1B's direction is plausible, GFI1's is not (predicted LT_HSC/HSPC-specific, i.e. NOT Prog-specific). |
| EPOR in-edges (11 found of 12 named) | No clean myeloid-vs-erythroid convergence pattern. `LT_HSC_vs_Prog`: mixed B-specific/neither/conserved calls not obviously split by regulator identity (SPI1, a myeloid regulator predicted LT_HSC/HSPC-specific, is instead called B-specific). `HSPC_vs_Prog`: **all 11** in-edges called B-specific uniformly -- completely uninformative, reflects the pair's class skew rather than target-specific biology. |

**Important confound for all `HSPC_vs_Prog` checks above**: Phase 0 found this pair has
**zero** A-specific silver-training examples (conserved=7, A-specific=0, B-specific=2134,
neither=9373) -- the model literally cannot learn to predict A-specific for this pair at
all (no gradient signal ever pushes toward that class), so its near-uniform B-specific
calls for GATA2/SPI1/GFI1/GFI1B/EPOR-regulator edges in this pair reflect that structural
label gap at least as much as gene-specific biology. This should be stated explicitly in
any writeup, not silently absorbed into "the model recovered X" or "the model failed to
recover X" framing.
