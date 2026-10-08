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

## 2026-10-08 (branch in-depth-analysis)

**Item 1: single signals.** Scored each information source alone on the test edges, plus a
within-TF AUROC (AUROC inside each TF's pairs, averaged) that removes credit for knowing
which TFs have many targets. All edges come from 34 TFs whose positive rates range from
0.1% to 79% (TFAP2A), so TF identity alone gives 0.844 AUROC, and a 2-feature degree LR
gives 0.872 vs. scRegNet's 0.887. Within-TF: target degree 0.707, scRegNet 0.720, Exp C
0.737. Raw and guilt-by-association co-expression carry ~no signal; the one-step PPI term
carries most of PANDA's. Writeup in PROGRESS.md §1.

**Item 2: PANDA input ablation.** Reimplemented PANDA's loop on a fixed gene/TF universe
(reproduces the original matrix exactly). Beyond node degree, co-expression carries PANDA's
information (+0.010 within-TF, CI excludes 0); PPI adds ~nothing once degree is known, since the
PPI term mostly tracks target popularity. One normalized step equals the iterated result: the
useful part reduces to "z-scored correlation with the TF's targets minus with its non-targets".
Retraining of Exp C per variant (3 seeds each) in progress.

**Item 3: where the gain lands.** Graph distance can't explain anything (TFAP2A hub puts ~all
pairs within 2-3 hops). Exp C helps most on moderately popular targets (baseline 0.528 within-TF)
and on EZH2 (0.478 -> 0.592; Polycomb targets are the opposite of the popular genes); it hurts on
targets with no training edges.

**Item 4: redundancy.** Baseline scRegNet (decoder type=MLP, the untuned default) is exactly
additive: it ranks genes in the same order for every TF (R2 = 1, between-TF Spearman = 1). PANDA
is the only TF-specific input in Exp C, and post-hoc stacking of baseline + PANDA reproduces Exp C.
Ceiling for any TF-agnostic ranking: within-TF 0.843.

**Item 5: cooperativity.** TF-TF cooperativity holds (STRING-linked TFs share targets 1.56x vs
1.27x chance-adjusted; top pairs CTNNB1-LEF1, FOXH1-SMAD2, OCT4-SOX2), but most overlap is
generic pile-up on popular genes. Gene-gene graphs (STRING, co-expression kNN) carry ~no
regulator homophily beyond degree, which explains the failed PPI Methods 1/2.

**Item 6: predicted networks.** Model networks cover 14/33 TFs and give different TFs 3.7x more
shared targets than reality; scRegNet's network is 72% the degree-only network. EZH2's true
targets are a significant developmental program (GO); no model's EZH2 predictions are. HOT-region
explanation only weakly supported (popularity vs expression rho ~0.1-0.16).

**Item 2 Result C: retraining + leakage-fix flaw.** Retrained Exp C with each PANDA variant (3 seeds).
Found that build_panda.py's leakage fix re-scores only train POSITIVES out-of-fold and leaves train
negatives on the full-prior matrix, so the PANDA scalar is much weaker on training rows than on test
rows (within-TF AUROC 0.58 vs 0.66; for no_ppi 0.51 vs 0.62) and the network learns too small a
weight. A symmetric OOF (every train row scored out-of-fold) fixes this: Exp C within-TF AUROC
0.7350 -> 0.7471, AUROC 0.8899 -> 0.8915 (3 seeds each), beating post-hoc stacking. Co-expression
is the input that matters most in the trained model too (full 0.747 > no_ppi 0.743 > no_coexpr 0.736).
