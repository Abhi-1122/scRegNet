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
