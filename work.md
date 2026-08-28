# Work Log

Terse, dated log of methodology ideas tried — just enough to understand each idea.
Numbers go in `results.txt`; detailed writeups of pre-2026-08-14 experiments (PPI
Methods 1/2, modality ablation, PANDA Exp B/C/D, scGPT, UCE) are in
`PPI_EXPERIMENTS_SUMMARY.md`.

## 2026-08-14

**PANDA Exp C hyperparameter tuning** — 50-trial QMC-sampled search (`src/tuning/`,
4 GPUs on HPC) over the GRN+scFM+PANDA-edge architecture. Best trial (AUROC
0.8890/AUPRC 0.6255) did not beat the untuned reused-hyperparameter result
(0.8902/0.6333) — reused hyperparameters (from the original adaptive TPE search)
were already near-optimal; a 50-trial non-adaptive QMC search over a ~15-dim space
couldn't beat that. Notable: top trials favored shallow (1-2 layer) GNNs vs
baseline's 3 layers.

**Next planned**: tune PPI Method 1 (dual encoder, attention fusion) — most
parameter-mismatched vs its reused hyperparameters (added PPI branch roughly
doubles GNN capacity), so most theoretical headroom, and simpler search space than
Method 2 (no per-layer multi-view gate coupling).

## 2026-08-28

**Closing the "improve scRegNet" line.** Decided against chasing PPI Method 1
tuning or a concat/gate fusion sweep — the PANDA tuning result above already
showed the search process (QMC, 50 trials) can't beat well-chosen reused
hyperparameters on this small dataset, so more tuning runs have low expected
value. Instead ran the one thing still worth confirming: **PANDA Exp C
multi-seed validation** (5 seeds, same reused hyperparameters, hESC TFs+500) —
all 5 seeds beat baseline on both AUROC and AUPRC, tight spread (AUROC
0.8899±0.0005, AUPRC 0.6307±0.0018 vs. baseline 0.8874/0.6225). The PANDA Exp C
gain is confirmed real, not a single-seed fluke. This closes out the
PPI/ablation/PANDA/scFM-swap line of work on `summer26` — see
`PPI_EXPERIMENTS_SUMMARY.md` for the full narrative and results.txt for every
number.

**Pivoting to the differential/contrastive GRN idea** (see chat history for the
full literature scoping — closest competitors are GP-DHT, BRIDGE, SupGCL, none
of which do joint cross-condition differential-edge prediction specifically).
New work happens on a fresh branch cut from commit `8a3f58a` (predates all
PPI/PANDA/tuning additions) to keep the new line's history clean. First step
there: nail down a verified baseline number to build against (paper's Table 2
hESC number, 0.88/0.62, already corroborated by our own `summer26` baseline
rerun — no new rerun needed for hESC). Data gap found: this repo only ships
4 of BEELINE's 7 cell types (hESC, hHEP, mESC, mHSC-E) — mHSC-GM and mHSC-L
(the other two branches of the HSC differentiation lineage, the strongest
biological motivation for the idea) are not present and would need BEELINE's
raw preprocessing pipeline plus a new Geneformer-embedding-generation step
neither of which exist in this repo, to acquire.
