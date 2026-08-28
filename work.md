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
