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

## 2026-10-12 (branch in-depth-analysis, week 2)

**Harness.** New `src/pair_gnn.py` (one file: data, model, training) replaces `src/train.py` for
these studies (about 20x faster; the old code converted a sparse matrix every forward pass).
Switches: final layer (add / had = linear on [u,v,u*v] / mlp), per-batch removal of the batch's own
positive edges from the message-passing graph (split), Geneformer on/off, encoder (gcn, sage, gat,
gin-mean, directed sage), PANDA / co-expression-contrast scalar or edge weights (always scored
out-of-fold, symmetric), validation-based epoch choice, refit on all training rows, held-out-gene
and held-out-TF protocols. Safe queue: `analysis/queue.sh` (cap 4 jobs, RAM guard; 20 parallel
jobs froze the 15 GB machine once).

**Findings.** (1) per-batch edge split helps additive models (+0.0045 AUROC, +0.026 withinTF) and
is mandatory for interacting final layers (without it SAGE+had collapses to 0.8486 / 0.635).
(2) Depth hurts: GCN withinTF 0.724 / 0.673 / 0.569 for 1 / 2 / 3 layers. (3) SAGE is the best
encoder (1 layer: 0.8884 / 0.6252 / 0.7489); GAT, GIN(mean), directed SAGE, GCN are lower.
(4) With SAGE the interacting final layer adds +0.033 withinTF over additive (0.7489 vs 0.7163).
(5) PANDA / co-expression contrast numbers add nothing to the SAGE hero (0.7488), small gains on
weaker GCN models; PANDA as GCN edge weights does not help. (6) A no-neural-network stacker
(popularity + CF + 3-paths + contrast + PANDA, logistic regression) reaches 0.8867 / 0.6102 / 0.7399.
(7) popularity-only models nearly match scRegNet on hHEP and mHSC-E too.

**Week 2 final (2026-10-12).** Equal-budget validation tuning for three families (scRegNet design,
SAGE hero, SAGE hero + Geneformer; 7-13 trials), 5-seed finals with paired tests over 33 TFs,
held-out-gene and held-out-TF protocols, shuffled-Geneformer control, TF-level edge dropout,
replication on hHEP / mHSC-E / mESC, and no-neural-network stackers (popularity + CF + 3-paths +
co-expression number + PANDA).
Results: edge split +0.078 withinTF (p=3e-6); hero beats scRegNet design/recipe by +0.04-0.05 withinTF
(p<0.04); hero + Geneformer (default settings) 0.8910/0.6346/0.7517 ~ last week's PANDA model;
PANDA/co-expression number adds nothing to the hero (branch can be removed); replicates on mESC
(0.9389/0.8814/0.8249 vs 0.9260/0.8593/0.7660) and mHSC-E, not hHEP; SAGE-vs-GCN is dataset dependent;
no model beats popularity on held-out TFs (hero -0.042, p=7e-4) and none ranks held-out genes well;
TF dropout does not fix it; five-feature stacker statistically tied with hero on hESC.
Validation AUROC is a poor tuning criterion for withinTF ranking. Not run: SEAL / NBFNet / NCN.
Details and tables: PROGRESS.md sections 10-16.

## 2026-10-16 (branch in-depth-analysis)

**Direct check of the PANDA-in-the-GNN mechanism.** `analysis/hero_vs_contrast.py`: per TF (33 TFs, >=8 test rows,
both classes), Spearman and partial Spearman (popularity rank regressed out of both) between the model's test
logit and the co-expression contrast / PANDA number; alignment of npz tf/tgt/y with Test_set.csv asserted.
Mean ± std over 5 seeds. Hero partial rho with contrast 0.309 ± 0.045, additive-final-layer hero 0.074,
scRegNet design 0.07; paired hero minus additive +0.234 (CI [0.147, 0.327], 26/33 TFs, p=2e-5). Moderate agreement,
specific to interacting layers, also present with GCN (0.303). Correlation only; not proof of the mechanism.
Contrast alone has within-TF AUROC 0.574 (PANDA 0.664). Details: PROGRESS.md section 12.

**Within-TF validation criterion for tuning (hESC pilot).** Added `--select_metric {auc,within}` to `src/pair_gnn.py` (default auc, old runs unchanged);
`analysis/tune2.py` (jobs/finals/report). 16 random trials per family (C_hero, B_hero_sf; encoder sage/gcn, Adam, one GNN layer) on the validation
slice, seed 0, hyperparameters and epoch both chosen by validation within-TF AUROC; winners refit with 5 seeds, beside the untuned scRegNet recipe
(same epoch rule). Test within-TF: scRegNet 0.7240, hero 0.7101, hero+Geneformer 0.7393. Pre-specified verdict (tuned hero > 0.7471, the hand-set
default) failed: tuning on within-TF did not help. Validation within-TF winners beat the default trial on validation (0.743 vs 0.705) but not on test.
Epoch selection on within-TF raised the published recipe from 0.7029 to 0.7240. Report: analysis/out/t2_report_hESC.txt.

## 2026-10-23 (branch in-depth-analysis)

**Week 3, hESC: PANDA as the graph, neural PANDA, stacker on other graphs.** `src/pair_gnn.py`: new flags `--mp_graph {grn,panda,union}`, `--panda_k`, `--panda_safe`
(fold-pure training batches: fold-k PANDA top-K graph for fold-k batches, full graph for val/test; naive = full graph everywhere), encoder `npanda` (R known targets,
C co-expression kNN k=10 label-free, `--coexp meta` metacells, P STRING; `--relations`), decoders `tani`, `tani_pure`. Defaults reproduce old runs
(w3_hero_s42 GPU 0.8890 vs d01 0.8890). `analysis/scalars.py` now also saves the 5 per-fold PANDA matrices (`get_folds`; old ev/tr caches reproduced exactly).
New: `analysis/w3_graph_recall.py`, `analysis/stacker_graphs.py`, `analysis/w3_report.py` (report analysis/out/w3_report_hESC.txt). Runs: 125 GPU runs, 0.3 to 0.9 min each.
Results (AUROC/AUPRC/within-TF, 5 seeds): hero 0.8884/0.6255/0.7470; safe PANDA graph K=100 0.8822/0.6117/0.7231 (p=0.009), union 0.8836/0.6151/0.7298 (p=0.009), K=200 0.8838 (p=0.08);
naive K=100 0.8659/0.5847/0.6918. Leak: 55.6% of training positives inside the full K=100 graph vs 3.3% in the own-fold graph (test 3.2%; 1685 of the 3400 edges are training positives); naive minus safe -0.016 AUROC, -0.031 within-TF (p=0.004).
Neural PANDA RCP with Tanimoto 0.8852/0.6210/0.7180 (p=0.042 vs hero); with Hadamard decoder 0.8898/0.6315/0.7410 (within-TF diff -0.006, p=0.86; AUPRC +0.006 in 5/5 seeds);
two-layer hero 0.8866/0.6282/0.7333; Tanimoto decoder alone on the hero 0.8837/.6205/.7174; tani_pure 0.8337. Stacker: best new `all` LR 0.8893/0.6194/0.7423 vs old best 0.8867-0.8871, within-TF
difference not significant; PANDA/co-expression graphs alone far below the hero within-TF. Verdict: no variant beats the hero on within-TF. Caveat: 7 of 1061 test positives have the reverse
(gene, TF) edge in the undirected training graph (shared by all rows). Part 5 (other cell types) not run. Details: PROGRESS.md section 17.

**Week 3 part 5 (2026-10-23): hHEP, mESC, mHSC-E.** `make_ppi.py` now takes `CELL SPECIES` (default hESC 9606) and reports edges under the dataset symbol (needed for mouse; STRING v12 API, score 700, functional,
as hESC). Built STRING edge files: hHEP 4486 edges, mESC 3862, mHSC-E 1504 (hESC 2296; the hESC files were re-created with the original pipeline after an accidental overwrite and verified to reproduce
the cached PANDA matrices exactly). `analysis/scalars.py <cell> 500` run for all three. 90 GPU runs `w4_<cell>_{hero,herogf,pgU_k100,np_had,np_tani,A_paper}_s42-46`
(jobs `analysis/jobs/w4_*.txt`; mESC runs 4-14 min, mESC published recipe 70-85 min), `analysis/stacker_graphs.py <cell>` for each cell, report `analysis/w4_report.py`.
Within-TF difference to the cell's hero (p): neural PANDA RCP Hadamard hHEP +0.0145 (0.0004), mESC +0.0015 (0.006), mHSC-E -0.0044 (0.88), hESC -0.006 (0.86), AUROC/AUPRC gain in all four (5/5 seeds in three);
safe GRN+PANDA graph -0.017 / -0.005 / -0.010 / -0.002 (hESC/hHEP/mESC/mHSC-E); Tanimoto decoder -0.029 / +0.015 / -0.062 / -0.041; Geneformer +0.004 / +0.020 / +0.001 / +0.020;
published recipe (rule A) -0.044 / +0.027 (p=0.08) / -0.058 / -0.040. No graph-PANDA variant beats the hero within-TF except neural PANDA in hHEP. Details: PROGRESS.md section 18.

**2026-10-23 W5: new ideas, coordinator's own line.** `src/pair_gnn.py`: `--rank_w` (within-TF BPR, each batch positive vs 4 same-TF training negatives), `--aux_panda`
(aux MSE head on out-of-fold PANDA over 512 random (TF, gene) pairs per batch), `--scalar a,b` (several scalars) and `--scalar_lr` (scalars as a separate linear term, own lr, no wd).
Defaults unchanged. New: `analysis/w5_report.py`, `analysis/w5_ensembles.py` (z-scored logit averaging, seed-matched, with hero x2/x3/x4 seed controls), `analysis/w5_heldout_tf_transfer.py` (CPU).
hESC, 5 seeds, GPU: ranking loss and PANDA teacher -0.005 to -0.008 within-TF; free scalar weight hurts (contrast -0.012, p=3e-4; both -0.016, p=0.006).
Held-out genes (5 folds x 5 seeds): np had +0.021 within-TF (p=0.041, AUPRC lower), gone without PPI. Held-out TFs: no gain; popularity still best, TF-similarity transfer 0.701 vs 0.724.
Late fusion (hero + Geneformer hero + np had + stacker) vs hero x4 seeds within-TF: hESC +0.010 (p=0.007), hHEP +0.013 (2e-5), mESC +0.011 (5e-8), mHSC-E +0.009 (0.07).
One run (w5_nphad_gf_s44) started in the same second the hESC STRING files were restored; re-run, identical to 4 decimals. Write-up: PROGRESS.md section 19.

**2026-10-23 W6: hard negatives, neural PANDA for unseen genes.** `src/pair_gnn.py`: `--rank_neg {rand,pop,model}` (1 of 16 random same-TF training negatives:
most popular gene or highest current score), `--rank_loss {bpr,bce}`; rand path reproduces w5_rank1_s42 within GPU noise. 45 hard-negative runs: all tie or hurt (popular-gene
negatives -0.04 to -0.10 within-TF; model-hard BPR w=0.1 -0.002, p=0.55). Held-out genes 5x5: np RP +0.003, hero+GF +0.026 (p=0.004), np+GF vs hero+GF +0.004;
4-model late fusion +0.013 vs same-cost seed blend (p=0.004; `analysis/w6_heldout_fusion.py`). PROGRESS.md section 20; section 19 PPI sentence corrected.

**2026-10-23 W7: late-fusion attribution.** `analysis/w7_attribution.py`: hero + w*ingredient (w scanned on test, diagnostic only) vs hero x2 seeds, and the
4-model blend vs hero+Geneformer x4 seeds. Geneformer is the only ingredient that helps on all 4 cells; raw PANDA/contrast do not explain the gain. Blend vs
GF hero x4: hESC +0.006 (p=0.12), hHEP -0.008, mESC +0.010 (p=1e-5), mHSC-E -0.011. Blend dropped from slides, summary and PROGRESS conclusions
(PROGRESS s19 Result 3 rewritten as the attribution; s20 held-out fusion paragraph removed).

**2026-10-26 W8 part A: best model per cell, then tuning.** `analysis/w8_report.py`; `analysis/tune2.py` gained `wjobs`/`wfinals <cell> <encoder>` (16 random trials,
fixed encoder + Geneformer, score = validation AUROC over seeds 0,1, `default` scored but not selectable; finals `w8_<cell>_tuned_s42..46`, `--select_metric auc`, refit).
Part 1: `w8_<cell>_gcngf_s42..46` = SAGE+GF args with `--encoder gcn`; last week's g11 GCN+GF used the identical args, so no separate `_lw` run (s42 reproduces g11).
Winners (5-seed within-TF): hESC SAGE, hHEP GCN, mHSC-E GCN, mESC SAGE; GCN vs SAGE significant only on mESC (SAGE, p=2e-4). Tuning: hESC +0.009 (p=0.03), mESC +0.005
(p=0.003), hHEP -0.001 (p=0.75), mHSC-E -0.008 (p=0.05); validation gains of the best trial were 0.001-0.004 AUROC (noise level). PROGRESS.md section 21.

**2026-10-26 W9 item 7 round 1: motif-only TF->gene prior (hESC).** `analysis/w9_motif_build.py` (JASPAR 2024 CORE vertebrates non-redundant PFMs from data/external/jaspar2024, mapped to TFs by symbol, `::`
dimers split, `(var.N)` stripped; promoter TSS -1000/+200 of the Ensembl canonical transcript via REST, raw responses cached in data/external/promoters_hg38/; own numpy PWM scan on both strands,
pseudocount 0.8, background = pooled promoter ACGT made strand-symmetric, p<1e-4 threshold by DP; writes data/hESC/TFs+500/motif/motif_{max,hits}.npy). Coverage: 849/910 promoters, 29/34 labelled TFs (143/410 all).
`analysis/w9_motif_signal.py` -> `analysis/out/w9_motif_r1.txt`. Motif alone has signal (within-TF 0.59 on 29 held-out TFs, p=2e-5 vs chance) but is far below popularity (0.71, p=8e-6);
popularity+motif LR/rank-average is +0.013/+0.007 within-TF, p=0.35/0.95: no detectable gain on held-out TFs. Ensembl REST was slow (expand=1 lookups timed out; switched to
non-expanded gene + transcript lookups); build is ~90 min mostly waiting on the API. Run with `python3.12 -P` (user-site numpy/Bio unavailable under -I).

**2026-10-26 W9 item 7 round 2: motif prior diagnostics.** `analysis/w9_motif_r2.py` -> `analysis/out/w9_motif_r2.txt`. A: window -2000/+500 fetched once (17 batches, cached seqw_*.json; the -1000/+200 slice equals the round-1 sequences,
0 mismatches), integer PWM scan on the DP grid, 6 variants saved as data/hESC/TFs+500/motif/r2_{logp,hitsmax}_*.npy; chosen on Train_set (logp -500/+100, tied with -1000/+200). No variant beats popularity on held-out TFs
(LR +0.004, p=0.37). B: residual motif signal stays at 0.55 within-TF (mostly not CpG/GC/popularity). C: motif-similarity transfer does not beat popularity (k=all 0.711 vs 0.724); rank average +0.005 to +0.008, p>=0.08.
D: adding z(motif) at w=0.25 to the trained model changes within-TF by +0.002 (p=0.40) and 0.000 (p=0.70). Fetch took ~20 min, analysis ~1 min.

**2026-10-26 W10 item 8 round 1: per-time-point co-expression vs ChIP labels (hESC).** `analysis/w10_states_r1.py` -> `analysis/out/w10_states_r1.txt`. |Spearman| TF-gene on each of the 6 time points, pooled, and pooled after centring each gene within its
time point; Test_set and all labelled rows; CPU, ~1 min. Early time points (00h/12h/24h) are better than late (36h/72h/96h) and than pooled (24h vs pooled +0.096, p=0.001; early vs late +0.049, p=0.009; 20/20 subsamples at 66 cells), but all within-TF values are 0.42-0.54
(pooled 0.42 is below chance), nothing beats popularity (0.707) and fusion with popularity lowers it. Per-TF expression-vs-AUROC check does not support "TFs match best where expressed" (mean rho -0.09, p=0.27).

**2026-10-26 W10 item 8 round 2: hero on one cell state.** `analysis/w10_states_prep.py` (data/states/<cond>/TFs+500: expression subset + symlinked labels; late/rand redrawn per seed with RandomState(seed)), jobs `analysis/jobs/w10.txt`, `analysis/w10_report.py` ->
`analysis/out/w10_states_r2.txt`. Hero args as w3_hero including `--scfm 0` (pair_gnn defaults to --scfm 1, so it must be passed to get no Geneformer); the hero path needs nothing else under the cell_type folder; `all` reuses w3_hero (identical args).
Result: early 0.728 / late 0.712 / rand 0.741 / all 0.747 within-TF. Late is worse than the random control (-0.029, p=0.005); early is not better than random (-0.013, p=0.21), though early beats late (+0.016, p=0.16, 5/5 same-seed pairs).
Per-TF early-vs-late AUROC does not follow expression timing (rho -0.20, p=0.27). The early-state hypothesis is not supported; mixed-state training columns are as good or better.
