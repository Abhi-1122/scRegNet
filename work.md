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

**2026-11-12 W11 A: reverse-edge leak.** `analysis/w11_reverse_leak.py` -> `analysis/out/w11_reverse_leak.txt`. Test pair (t,g) is leaky if (g,t) is a training positive (undirected message-passing edge, so the pair is scored through its own edge); counted against Train_set positives (refit graph), TFs+500 index space shared by TF and Target columns.
hESC 30 pairs / 7 positives (matches), hHEP 57/18, mHSC-E 59/36, mESC 429/167. Best model and published recipe re-scored from the saved logits on the cleaned set (no retraining, 5 seeds), paired per-TF test repeated: scores and gaps are unchanged within +0.003 within-TF (cleaning slightly raises them); the leak does not drive any reported result.

**2026-11-12 W11 B: edge-fix checks (hESC, hero without Geneformer, 5 seeds).** `src/pair_gnn.py` gained `--split fixed --mp_frac 0.7` (fixed random 70% of the undirected training edges is the training graph, supervision = the other edges + all negatives; validation/test graph = all training edges, as for batch)
and `--own_edge_diag 1` (after the refit, 10% of training-positive edges + as many negatives scored with and without those edges in the graph); both default off, `w11_own_batch` reproduces `w3_hero` to 4 decimals. Scripts `analysis/w11_edge_audit.py` (leak audit, asserts), `analysis/w11_edge_report.py`; jobs `analysis/jobs/w11_edge.txt`; output `analysis/out/w11_edge_checks.txt`.
Own-edge: positive logit gap present-removed +0.39 ± 0.27 for split none vs -0.06 ± 0.04 for batch (held-out AUROC 0.899 vs 0.852 for none, 0.930 vs 0.927 for batch). Fixed split within-TF 0.671 vs batch 0.747 (p=5e-7), better than none 0.620. Audit: validation/test pairs are not in any training graph through their own direction; through the reverse row 6/2068 validation and 28-30/7142 test pairs are; Geneformer loader takes no labels; DEG/NC/scalars use training rows only.

**2026-11-12 W11 C: consolidated-grid inventory (no runs).** `analysis/w11_grid_inventory.py` -> `analysis/out/w11_grid_inventory.txt`, job file `analysis/jobs/w11_grid.txt` (not submitted). Grid encoder {gcn, sage, npanda} x decoder {add, had} x split {none, batch} x Geneformer {0, 1}, 1 layer (256), hero hyper-parameters, 4 cells x seeds 42-46 = 96 configs / 480 runs;
13 configs already complete, 391 runs missing, about 954 run-minutes (~4 h wall at 4 slots, factor-2 uncertainty). Existing npanda runs are 2-layer, so none count at 1 layer.

**2026-11-12 W12 B: NBFNet and SEAL (hESC, seeds 42-44, hero w3_hero).** `src/nbf.py`: Bellman-Ford path model over the training TF->target graph (relations TF->target, target->TF, optional co-expression/STRING), per-TF boundary query, DistMult messages, sum/mean/PNA aggregation with layer norm, score = MLP of the layer states plus gene and TF-gene expression projections; the batch's own positive edges leave the graph as in `--split batch`. `src/seal.py`: 1-hop enclosing subgraph per pair (<=30 sampled neighbours per node), pair edge deleted for every pair of every split, DRNL + degrees (own edge subtracted) + PCA-32 expression, 3 SAGE layers, mean/max/target-node pooling. Both write the pair_gnn json/npz format; knobs picked on validation AUROC at seed 42.
NBFNet chosen L6 PNA + TF-gene product: within-TF 0.697 vs hero 0.749 (p=0.0013); paths add only +0.013 over the same model without path layers. SEAL: 0.7075 (p=0.027); DRNL worth +0.018. Held-out TFs: NBFNet 0.602 vs 0.686, SEAL 0.655 vs 0.686. Held-out genes: NBFNet 0.543 vs 0.546, SEAL 0.506 vs 0.546. SEAL scores track gene in-degree more (Spearman 0.85 vs hero 0.72). Not carried to other cells (stage 1 regressed). `analysis/w12_nbf_report.py`, `analysis/w12_seal_report.py`, helpers `analysis/w12b_lib.py`, `analysis/w12_sweep_table.py`.

**2026-11-12 W11 C: consolidated grid run.** Inventory `analysis/w11_grid_inventory.py` (a seed counts only if json and npz exist; older g*/h* runs have no logits), jobs `analysis/jobs/w11_grid.txt` (union, 381 runs) + `w11_grid_fix.txt` (hESC gcn/sage gf0 seeds 42-44 rerun for logits), run names `w11_grid_<cell>_<enc>_<dec>_<split>_gf<g>_s<seed>`,
CAP=4 queue, hESC/hHEP/mHSC-E first, mESC last, ~4.3 h wall, no failures. Report `analysis/w11_grid_report.py` -> `analysis/out/w11_grid.txt`. gcn/sage at 1 layer (256), npanda at 2 layers (256 256, `--relations RCP`; existing `*np_had*` runs reused), decoder add = TF + gene terms without interaction.
Main effects (paired per-TF, pooled over the other factors): Geneformer helps in all four cells (+0.006 to +0.014, p<=0.012); the Hadamard decoder matters on mHSC-E (+0.043) and mESC (+0.055) and not on hESC/hHEP; per-batch removal matters on hESC (+0.039) and is neutral or slightly negative elsewhere; no encoder wins consistently (sage > gcn only on mESC).
Best grid config vs the recipe: hESC +0.052 (p=0.002), hHEP +0.001 (p=0.53), mHSC-E +0.084 (p=5e-5), mESC +0.063 (p=2e-8); the best config is picked on test within-TF, so these are optimistic.

**2026-11-07 W12 S: statistical network models (CPU).** `analysis/w12_stat.py` (models), `analysis/w12_stat_report.py` (tables), scores in `analysis/out/stat/`, output `analysis/out/w12_stat.txt`, write-up `analysis/out/W12_S_stat.md`. Every (TF, gene) pair scored from the training labelled pairs only (unlabelled = missing); data via `prepare` so splits, validation slice (0.1, split_seed 0) and test rows equal the GNN runs. Models: popularity (smoothed rate, raw in-degree); DC-SBM with GMM gene clusters (Geneformer 30 PCs or expression 40 PCs, diag/full, K 5-40, soft/hard, Beta-smoothed omega, optional DC normalisation) with omega-only and theta-only ablations; DC-SBM with learned soft blocks (Bernoulli, `a_t + b_g + log sum_k pi_gk exp(W_tk)`, Adam); masked Poisson factorisation (rank 4-16); latent space model (`a_t + b_g - ||z_t - z_g||`, d 2-8); K=1 main-effects control. All knobs and the number of gradient steps chosen on validation within-TF (mean over seeds 42-44), refit on all training rows, test scored once. Held-out genes: Geneformer-cluster DC-SBM (theta from cluster rate) and latent space (z, b by ridge from features), 5 folds x seeds 42-43; held-out TFs skipped.
Result: popularity alone is within 0.002-0.015 of the hero in pooled AUROC. Learned DC-SBM matches the GNN on hHEP and mESC (beats hero +0.042 p=0.002, +0.009 p=0.002; vs best GNN +0.011 p=0.48, +0.005 p=0.06) and falls short on hESC (-0.032 p=0.02) and mHSC-E (-0.063 p=5e-4). GMM-cluster program adds nothing over popularity (omega-only 0.52-0.65; p>=0.13); learned blocks add +0.011 to +0.066 over a Bernoulli main-effects control. Held-out genes: soft DC-SBM within-TF 0.575 vs hero 0.546 (p=0.008), +0.053 over its theta-only version. Peak RSS 0.85 GB (`ulimit -v 6000000`; 4 GB fails at the torch import), one process at a time.

**2026-11-07 W12 A: graph ideas 1, 4, 6, 9 (hESC, 3 seeds; held-out 5 folds x 2 seeds).** Code in `src/pair_gnn.py` (new flags, all default off: `--decoder qatt --qatt_heads --qatt_src --qatt_topk --qatt_uniform`; `--hub_cap --hub_eval --idf_beta --deg_feat`; `--pe --pe_dim --pe_graph --pe_use --pe_fold --pe_flip --pe_rand --fold_batches`; `--save_sel_grid`), `analysis/w12_common.py` (seed-filtered report helpers, held-out report), `analysis/w12_{1,6,9}_report.py`, `analysis/w12_cs.py` + `w12_4_report.py`; write-ups `analysis/out/W12_{1_query_tf,4_cs,6_hubs,9_pe}.md`. Epochs chosen on validation AUROC, refit; hero reproduces within GPU noise (<=0.001). Leaks: qatt attends only over the current graph's targets of t (batch edges and g itself removed); hub degrees/caps computed after batch-edge removal; PE from label-free graphs, GRN PE in a leak and a fold-safe version (5 folds, PE without the fold's edges) plus a fold-pure-batching control; C&S uses out-of-fold training scores (20 folds, own edge removed) and hyperparameters chosen on the validation slice of the stage-1 model, test scored once.
Results: PE and hub control neutral or negative everywhere (hESC hubs are mild, max in-degree 14). C&S lowers within-TF by 0.014 (hero) and 0.022 (Geneformer hero): smooth step carries the loss. qatt: no gain on hESC random split, +0.018 on held-out genes (p=0.003), -0.053 on held-out TFs (-0.024 with tf_drop, equal to hero+tf_drop), +0.011 / +0.008 on hHEP / mHSC-E, -0.015 on mESC; a uniform-weight control gives the same gains, so the benefit comes from the gene-vs-known-targets similarity feature, not the attention. Baselines for the other cells are `w4_<cell>_hero` (the w11 grid has no sage/had/batch/no-Geneformer row). Ada job 74423 appeared stalled (6 processes at 130% CPU for >35 min, node GPU idle); the affected confirmation and C&S Geneformer runs were done on the laptop queue.
- 2026-11-12 Idea 0: single correlations across cells between candidate gene and TF mRNA, known-target mean (activity) and least-squares TF activity (X ~ P A); test rows, no training; analysis/w13_activity.py.

**2026-11-12 W13: second TF summary (Lead A) and DC-SBM score as decoder input (Lead B), best grid configuration of each cell.** Report `analysis/w13_report.py` -> `analysis/out/w13_leads.txt`, write-up `analysis/out/W13_leads.md`, jobs `analysis/jobs/w13_*.txt`, status `analysis/out/W13_STATUS.md`. Seeds 42-44, baselines = the grid runs of each cell's best configuration, epoch on validation AUROC, refit.
Code: `analysis/scalars.py` gets kind `dcsbm` (learned DC-SBM of w12_stat.py fitted on all labelled training rows, per-cell k/l2/steps from w12_stat validation; out-of-fold training matrix over the same 5 folds as the other kinds; fit seed 42; reproduces the w12_stat test within-TF exactly) and a lazy netZooPy import. `src/pair_gnn.py` unchanged. qatt audit: the known-target mask comes from the R edges of the current graph only (npanda and gcn alike), own gene masked under `--split none`; own_edge_diag on mESC shows no added own-edge cue (gap +0.132 vs +0.127).
Result: second summary adds nothing on the best configurations (hESC +0.001, hHEP -0.003, mHSC-E -0.015, mESC +0.000; held-out hHEP genes -0.007, p=0.007); its section-25 gain on hHEP shrinks from +0.011 (sage, no GF) to +0.006 (GF or npanda) to -0.003 (both). DC-SBM scalar: mESC +0.005 (p=2e-4), others neutral/negative; hHEP 0.7937 vs 0.7919 (p=0.23). Rank average of GNN and DC-SBM scores (diagnostic) +0.014 mESC, +0.006 hHEP. Cluster (ada) runs of the held-out and qs+dcsbm jobs were discarded (hESC qs+dcsbm differed from laptop reruns by 0.015 within-TF, cause unknown); all numbers are laptop runs. The dcsbm variant was first picked from incomplete seed-42 results for hHEP and mHSC-E, corrected to scalar_lr 10 after all runs finished.

## 2026-11-12 W14: gene-level properties or co-expression?
New flag `--feat {full,cellperm,stats,rand}` in `src/pair_gnn.py` (default full; hero rerun reproduces w3_hero_s42 to GPU rounding). cellperm = per-gene permutation across cells (fixed per-gene RandomState); stats = 12 per-gene statistics of the raw expression, standardised across genes; rand = fixed Gaussian of the same width. Hero config, 4 cells x 3 variants x seeds 42-44, hESC held-out genes (5 folds x seeds 42, 43) for cellperm, stats and (extra) rand; all on the laptop (ada unreachable). Finding: rand matches full within-TF in hHEP and mHSC-E and trails by 0.028 (hESC) and 0.019 (mESC); stats falls below rand everywhere (-0.02 to -0.09 against full); cellperm is at rand level. On held-out genes stats (0.560) matches or beats full (0.546) while cellperm and rand sit at chance. Report analysis/w14_report.py -> analysis/out/w14_feat.txt; write-up analysis/out/W14_feat.md.

## 2026-11-13 W15 part 3: Geneformer through biology or identity?
No code change (`--scfm 2` already shuffled the Geneformer vectors across genes with a fixed RandomState(0) permutation). Best grid configuration per cell (hESC/hHEP npanda 2L RCP batch, mHSC-E gcn batch, mESC sage none; had, Adam lr 1e-4, wd 1e-3, refit), seeds 42-44. none and real = existing grid runs (found by matching all args, as in w11_grid_report.py), shuffled = 12 new runs (`analysis/jobs/w15_gf.txt`); hESC held-out genes (hero SAGE, 5 folds x seeds 42, 43): none = w5_ho_gene*_hero, real = w6_ho_gene*_herogf, shuffled = 10 new runs. All laptop. Finding: shuffled is at or below none everywhere (within-TF -0.007 hESC, +0.003 hHEP, -0.006 mHSC-E, -0.010 mESC, -0.012 held-out genes); real GF gain is +0.007/+0.020/+0.033/+0.004 and +0.018 held-out, significant only on mHSC-E at 3 seeds. So the identity hypothesis is rejected; the vectors help through content. Report analysis/w15_gf_report.py -> analysis/out/w15_gf.txt; write-up analysis/out/W15_gf.md.

## 2026-11-13 W15 part 2: LightGCN encoder
New encoder `--encoder lightgcn` in `src/pair_gnn.py` (flags `--lgc_layers --emb_dim --emb_std --emb_l2 --emb_feat {none,expr,gf,stats} --emb_free`, decoder `dot`; default off, `w3_hero_s42` rerun reproduces to 7e-5 within-TF with the same epoch). E_0 = nn.Embedding (+ linear projection of features), E_{k+1} = D^-1/2 A D^-1/2 E_k on the current message-passing graph (batch edges removed under `--split batch`), output = mean of layers, then the existing Hadamard decoder (or dot + bias). hESC seed-42 sweep of 144 runs (K, d, decoder, embedding L2, lr, emb_feat; validation AUROC; run on ada and laptop, only validation compared) plus 16 dot runs with larger init; final runs seeds 42-44 on the laptop; held-out genes/TFs on hESC for pure and three hybrids. Dropped for time: feature-only ablations on mESC and mHSC-E, statistics hybrid on mHSC-E. Finding: pure LightGCN ties random-feature SAGE on hESC (0.724 vs 0.721), beats the hero on hHEP (+0.017, p=0.09; expression hybrid +0.016, p=0.005), loses on mESC (-0.058; one-seed diagnostic shows decoder wd 0 recovers 0.823), ties on mHSC-E; no cell beats its best GNN or DC-SBM. Free embedding carries the signal (score/popularity Spearman 0.92). Hybrids are +0.014 to +0.016 on held-out genes (p 0.09-0.13); expression hybrid collapses on held-out TFs. The sweep did not vary decoder weight decay. Report analysis/w15_lgc_report.py -> analysis/out/w15_lgc.txt; write-up analysis/out/W15_lightgcn.md.

## 2026-11-13 W16: curated pipeline
New flag `--dec_wd` in `src/pair_gnn.py` (default -1 = `--wd` for everything; >= 0 gives the node projection, decoder and query-summary weights their own decay, GNN convolutions keep `--wd`). `analysis/w16_choose.py`: validation-only choice among the 24 grid configs (mean `val` within-TF over seeds 42-46). `analysis/w16_dcval.py`: DC-SBM refit without the validation rows (learned DC-SBM per cell and seed with the stored k/l2/steps; fixed-cluster Geneformer-soft DC-SBM per held-out gene fold), writes `analysis/out/pair/w16_dcv_*`. `analysis/w12_stat.py` `load()` now passes `emb_feat`/`feat` (needed by the current `prepare`). Runs with `--save_sel_grid 1` (saves the best-epoch TF x gene grid, used as validation scores): standard arms base / dwd0 (all cells), qs / qsdwd0 (hHEP, mHSC-E), seeds 42-44 (laptop: hESC, hHEP, mHSC-E; ada: mESC); hESC held-out genes (base, dwd0, 5 folds x seeds 42,43) and held-out TFs on ada; sensitivity arms with the test-picked grid configs (npanda hESC/hHEP laptop, mHSC-E gcn and mESC sage ada). A rerun with the flag reproduces `w8_hESC_gcngf` (same epochs, within-TF to 1e-4). Blend = (1-w) * within-TF rank(GNN) + w * within-TF rank(DC-SBM), w in {0,.25,.5,.75} on validation within-TF; pooled AUROC/AUPRC from a global-rank blend with the same w. Seed ensemble = mean logit over seeds 42-44 (recipe ensembled the same way from `f02_A_paper`, `w4_*_A_paper`). `analysis/w16_pipeline.py` -> `analysis/out/w16_pipeline.txt`; write-up `analysis/out/W16_pipeline.md`; jobs `analysis/jobs/w16_*.txt`. Result: FAIL, held-out TFs -0.0320 vs the Adam-tuned scRegNet design (p=1e-4) and mHSC-E -0.0007 vs its own config alone; all 4 standard cells and held-out genes beat the recipe. Only the DC-SBM blend adds (+0.009 to +0.020 on hESC, hHEP, mESC); decoder wd 0, the second summary and the validation config choice add nothing measurable.

## 2026-11-13 W17: pipeline on held-out genes and TFs, all cells
`analysis/w17_jobs.py` writes `analysis/jobs/w17_*.txt`; arms per cell, protocol (gene|tf), fold 0-4, seeds 42, 43: `pipe` (W16 validation-chosen GNN, `--save_sel_grid 1`), `tuned` (args of `ho_*_f01_A_tuned` with `--cell_type`), `hero` (args of `w5_ho_*_hero` with `--cell_type`); names `w17_<cell>_ho_{gene,tf}<f>_<arm>_s<seed>`. hESC reuses `ho_*_f01_A_tuned`, `w5_ho_*_hero`, `w16_ho_gene*_dwd0`; hESC held-out TFs pipeline rerun with the grid saved (`w17_hESC_ho_tf*_pipe`). Held-out genes: fixed-cluster DC-SBM Geneformer soft per cell and fold via `analysis/w12_stat.py --protocol gene --skip_lsm 1` (new flag, default off; K/cov/smoothing on validation genes), validation-mode refits via `analysis/w16_dcval.py --kind hog --cell C` (new `--cell`, default hESC), `analysis/jobs/w17_dcsbm.sh`; w in {0,.25,.5,.75} on validation genes pooled over the 10 runs (all 0.5). Held-out TFs: rank blend GNN with gene popularity (log1p of the number of training-positive TFs per gene); val scores from the best-epoch grid, popularity without the 3 validation TFs' edges (`prepare` protocol tf, split_seed 0), test popularity from all training rows of the fold; w in {0,.25,.5,.75,1} per fold on the 3 validation TFs (grid reproduces the recorded val within-TF, asserted); pooled-choice and fixed w=0.5 reported as sensitivity. `analysis/w17_report.py` does the tables and paired per-TF Wilcoxon (all TFs and TFs with >= 8 test positives). Result: pipeline beats the scRegNet baseline in 9/12 cell-protocol combinations, ties 2 (hESC and mESC held-out TFs), loses 1 (hHEP held-out genes, -0.0125, p=0.012); on unseen TFs popularity alone is as good as the blend. mESC hero/pipe runs split between ada and the laptop (reverse order). Total compute about 35 min.

## 2026-11-13 W18: popularity-matched within-TF AUROC
`analysis/w18_pop_matched.py` (CPU, reads saved `analysis/out/pair/*.npz` and `analysis/out/stat/*_dcsbm_learned_*.npz`). Per TF, concordance is counted only over positive-negative pairs whose genes share the exact training in-degree (from `Train_set.csv` positives), then pooled over degree strata; TFs with fewer than 30 such pairs are dropped. Additive runs are the floor because a gene-only term can still refine the noisy training degree. Output `analysis/out/w18_pop_matched.txt`.

## 2026-11-14 W19: motif priors for all cells
`analysis/w9_motif_build.py` takes a cell argument (default hESC, unchanged output); cells starting with 'm' use Ensembl mus_musculus (GRCm39) promoters, chromosomes 1-19, X, Y. Promoter caches per cell in `data/external/promoters_{hg38,mm39}/<cell>/` (hESC keeps `promoters_hg38/`). User-Agent no longer carries an email. Same JASPAR 2024 CORE vertebrates non-redundant PFMs, TSS -1000/+200, p < 1e-4 hits; mouse TF symbols matched upper-cased. Writes `data/<cell>/TFs+500/motif/motif_{max,hits}.npy`; logs `analysis/out/w19_motif_<cell>.log`. Coverage: labelled TFs with a motif hHEP 22/30, mHSC-E 21/29, mESC 34/88 (most mESC labelled TFs are chromatin regulators/cofactors); genes with a promoter hHEP 917/948, mHSC-E 687/704, mESC 1051/1120. Motif-only within-TF AUROC over all labelled pairs in results.txt.

## 2026-11-16 W20: learned PANDA, complete network inference across cell types
New `src/xcell_gnn.py` (imports NPanda, coexp_graph, ppi_graph, feat_variant from `src/pair_gnn.py`, which is unchanged). `build_cell` builds each cell from label-free files inside `no_labels()` (pd.read_csv/np.load/open raise on Train_set, Test_set, Label, BL--network): node features 12 expression stats + Geneformer 896 (z-scored per dataset) + is-TF + has-motif; relations Mf/Mr (motif_hits > 0 or top-k by motif_max, k 50/100/200), P (STRING among TF.csv), optional C; pair features z(motif_max), z(log1p hits), missing flag, optional z(Pearson). Encoder 2 NPanda layers hidden 256 (layer-1 neighbour means cached; equal to NPanda, asserted in `--selfcheck`), decoder MLP on [Hadamard, pair features]; `--model mlp` = no relations. Loss BCE (ablation: within-TF pairwise softplus over all same-TF pos/neg pairs in the batch), Adam wd 1e-3, dropout 0.175, batch 512, patience 10. 20% of source TFs held out (RandomState(seed)); epoch and config (lr x motif graph, mean over seeds 42-44) on source-validation within-TF; best-epoch model, no refit. Settings: 4 same-species, 4 leave-one-out, within-cell 5-fold TF-disjoint CV (folds as pair_gnn heldout_split; config on seed 42 per fold, then seeds 43-44). Baselines `--baselines <cell>`: motif_max, |Pearson|, netZooPy Panda (union mode, prior binary motif_hits > 0, PPI STRING TF-TF). Runs in `--jobs` shards (4 processes; queue.sh is tied to pair_gnn); after two phase-1 shards hung at 100% CPU (cause not found; likely thread oversubscription), each job runs in its own subprocess with a 20 min timeout and 2 threads. Report `analysis/w20_xcell_report.py` (`--jobs 1|2` writes `analysis/jobs/w20_p{1,2}.txt`) -> `analysis/out/w20_xcell.txt`, with a gene-only diagnostic and an in-cell popularity reference. 824 runs, 371 GPU-process minutes. Result: FAIL (see results.txt).

## 2026-11-17 W23: gene-only ceilings
One-off inline script (not saved). Labels = Train_set + Test_set per cell. Gene-only scores used for every TF: detection rate (fraction of cells with expression > 0), column sum of `motif_hits` over all TFs, and in-cell popularity = mean label of the gene over all other labelled TFs (leave-one-TF-out). Cross-cell popularity: source gene popularity over all source labelled TFs, mapped by upper-cased symbol, scored on the target's labelled pairs restricted to shared genes. Within-TF AUROC averaged over TFs with both classes and at least 10 pairs. Ideas document `fable.md` at the repo root uses these numbers.

## 2026-11-17 W22: fair baselines for complete network inference
New `src/xcell_base.py` (imports cell/evaluate/load_labels/motif_edges/no_labels/per_tf_auc from `src/xcell_gnn.py`, sym_edges from `src/pair_gnn.py`; both unchanged). (1) PANDA: numpy port of netZooPy's CPU loop (normalize_network, Tanimoto, diagonal update, alpha 0.1, hamming < 1e-3); prior binary motif_hits > 0, PPI STRING TF-TF + diagonal or identity (`--noppi`), Pearson of raw expression; rows = all TF.csv TFs. Port vs W20's netZooPy run on hESC (same 332 TFs): r = 1.000000, max |diff| 2.4e-7 (`--panda_check`, laptop). (2) GENIE3 ExtraTrees 500 trees sqrt, unit-variance output, unnormalised importances; GBM lr 0.01, <= 5000 trees, max_features 0.1, subsample 0.9, arboreto OOB early stop (window 25), importance x n_trees, raw expression; regressors = all TF.csv TFs minus the target; random_state 0; hard pruning (motif_hits > 0 first, importance rank within) and soft (within-TF rank mean of importance and motif_max). arboreto/netZooPy not on ada, nothing installed. (3) scRegNet-motif: GCN 227/29/34 + Geneformer after GCN + Linear 198 + additive decoder; graph motif_hits > 0 undirected (dense D^-1/2(A+I)D^-1/2, equal to GCNConv, asserted in `--selfcheck`); GCN input 12 expr stats; W20's 20% source-TF validation split, epoch and recipe (published SGD vs Adam-tuned args) on source-val within-TF; no refit. `leak_audit()` parses the module: load_labels only in run_scr, no label file named elsewhere. ada via `analysis/ada_w22.sh` (one 1-GPU job, 20 CPUs; claim files make parallel workers share one job list; finished runs skipped). Jobs `analysis/jobs/w22_{gpu,cpu}.txt` (48 + 16 lines; smoke `w22_smoke_*`). First main job cancelled after 2 min (PyG GCNConv per-step normalisation too slow with 4 processes on one GPU); rerun 84 min. 1.45 GPU-h, 29 allocated CPU-h. Report `analysis/w22_report.py` -> `analysis/out/w22_baselines.txt` (adds W20 full and MLP; W21 left out, no final report yet). Result in results.txt.

## 2026-11-17 W21: two-view model (motif + PPI graph, co-expression graph)
New `src/xcell_2view.py` (imports cell/evaluate/load_labels/motif_edges/no_labels/per_tf_auc from `src/xcell_gnn.py`, unchanged; copies W20's source-validation split line as `val_tfs`). View A: Mf/Mr motif edges (hits > 0 or top-50 per TF) + STRING TF-TF, 2-layer relational encoder (self + weighted mean per relation), hidden 256. View B: mutual kNN on Pearson r (k 5/10/20, r > 0 or |r|, weight r, normalised by sum |w|), 2-layer weighted SAGE. Fusions concat / per-gene gate / early (C as 4th relation); arms A, B, mlp. Decoder MLP on [z_t*z_g ; z motif_max, z log1p hits, missing]. TF-balanced BCE (each TF sums to 1, half positives half negatives), view dropout 0.2; ablations plain BCE, vdrop 0, InfoNCE zA-zB (tau 0.2, lambda 0.1/0.5) on source or source+target graphs. Grid lr {1e-4,3e-4,1e-3} x mk {0,50} x coexp graph (6), config and epoch on mean source-val within-TF over seeds 42-44; best fusion for ablations (gate) on mean source-val over the 8 settings. Leak guard: source labels loaded first, all graphs/features/training/target grid inside no_labels(); `--selfcheck` (passed on ada). Run on ada, one 2080 Ti, 6 processes (`analysis/ada_w21.sh`; phase 3 written on ada from finished runs via a dependent job): 3384 runs, 8.8 GPU-hours. `analysis/w21_report.py` (`--jobs 1|2|3`, report incl. gene-prior check = TF-averaged grid) -> `analysis/out/w21_2view.txt`; write-up `analysis/out/W21_2view.md`. Result: FAIL, two-view = view A within noise (concat +0.004, gate +0.002 mean), never above the MLP overall, gap source-target unchanged; consistency loss hurts (target version most).
