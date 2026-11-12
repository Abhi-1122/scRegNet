[18:34] A: started ada pip install (srun), ideas 9/6 implementation begins
[18:38] A: ada rules updated (use analysis/jobs/ada_inbox only); pair_gnn flags for ideas 1/4/6/9 being added
[18:41:30] worker B: src/nbf.py written, smoke OK on laptop, sweep w12_nbf_sweep submitted to ada_inbox
[18:43:50] worker B: src/seal.py written, laptop smoke OK (2 epochs val AUROC 0.88), sweep w12_seal_sweep in ada_inbox
[18:47:18] worker B: nbf sweep1 (seed 42): pure paths val AUC .868/within .658, +tf_inter .884/.716 (hero val .889). sweep2 with tf_inter submitted
[18:50] A: flags for ideas 1/4/6/9 implemented+smoked (hero reproduces within GPU noise: s42 within 0.7507/0.7500 vs 0.7499); 37 screen runs (s42) + CS hero runs queued in ada_inbox
[18:50:42] worker B: seal sweep seed42 h1tt/mm/sort/h2c10 val auc .8875/.8818/.8876/.8868, within .72/.71/.69/.69 (hero val within .70); confirm+ablation (3 seeds) submitted; h1tt chosen (tie on val AUC, best val within)
[18:58:11] worker B: nbf sweep2 val AUC (tf_inter): L3pna .8833, L4 .8831, L6 .8852, L3sum .8842, mean .8776, tfq .8807, RCP .8818, layer .8763, lr3e-4 .8827 -> chosen L6 pna tf_inter 1 (best val AUC); confirm 3 seeds + paths-only + no-path submitted
[19:03] [S] w12_stat.py written, smoke on hESC starting
[19:08] A: stage-1 screen (s42) done for 9/6/1; CS hero runs done; submitted seeds 43,44 confirmations of 15 families (w12_conf_a)
[19:10:11] worker B: nbf stage1 (2 seeds in) within .699 vs hero .749, regress; diag: gene-degree feature run (deg) submitted; ho_tf/gene runs for nbf+seal submitted
[19:13] [S] hESC smoke OK (8 min, peak RSS 0.8 GB); running all 4 cells then held-out genes serially
[19:15:22] worker B: seal stage1 3 seeds in: within .7075 vs hero .749 (p=.027), AUROC .8864 vs .8885; diag: pca128/256 runs submitted (hypothesis: PCA32 expression bottleneck); ho runs pending
[19:15:50] worker B: nbf stage1 3 seeds: within .6971 vs hero .749 (p=.001). diag runs: nopath_h256 and L3sum_h128 (capacity), deg
[19:26] [S] hESC done
[19:31] A: submitted held-out gene/tf (5 folds x s42,43) for qatt_h1_emb_k0, hub_idf05, pe_lap_ppi_k16 (w12_ho_a); C&S hero done (regressed)
[19:31] [S] hHEP done
[19:41:57] worker B: DONE. nbf within .697 vs hero .749 (p=.0013); seal .7075 (p=.027); write-ups W12_2_nbfnet.md, W12_3_seal.md; results.txt/work.md appended
[19:44] [S] mESC done
[19:49] [S] mHSC-E done
[19:51] [S] held-out genes fold 0 done
[19:51] A: ada job 74423 (rerun_herogf, includes my 30 confirmation runs + 3 CS herogf) shows 6 procs at 130% CPU for 37 min and GPU 0% on gnode066: likely CPU-only, stuck. Queued the 30 confirmation runs on the laptop queue too (names identical, whichever finishes first).
[19:52] A: stage 2 (ho_a done): qatt_h1_emb_k0 held-out genes within +0.018 (p=.003) but held-out TFs -0.053 (p<.001); hub_idf05 and pe_lap_ppi ~0. Submitted qatt + tf_drop 0.2 follow-up (w12_1_drop)
[19:55] [S] held-out genes fold 1 done
[19:57] [S] held-out genes fold 2 done
[19:58] A: random-split confirmations done for 9/6/1 (all within noise or worse); submitted qatt uniform-attention control (w12_1_uni)
[19:59] A: qatt stage 3 submitted (3 other cells + Geneformer, 3 seeds)
[19:59] [S] held-out genes fold 3 done
[20:01] [S] held-out genes fold 4 done
[20:01] A: hESC hubs are mild (max in-degree 14; cap 20 removes 0 edges = null control); mESC/mHSC-E/hHEP have heavy hubs -> submitted hub idf/cap on those 3 cells (w12_6_cells, 36 runs)
[20:04] [S] maineff control (K=1 Bernoulli main effects) done for all cells
[20:07] A: note: w11_grid has no sage_had_batch_gf0 for other cells; baselines taken from w4_<cell>_hero_s42..44 (same config)
[20:08] [S] all runs and report done: learned DC-SBM matches GNN on hHEP/mESC (within-TF .801/.833), GMM-cluster program adds nothing over popularity, held-out genes DC-SBM .575 vs hero .546 (p=.008); write-up analysis/out/W12_S_stat.md, results/work appended
[20:09] A: write-ups W12_9_pe.md and W12_4_cs.md written
[20:12] A: qatt finding: uniform-attention control matches learned attention on held-out genes (+0.0175 vs +0.0182), so gain is the prototype similarity term, not query-awareness; stage 3 hHEP +0.011 (p=.003), mHSC-E +0.008 (p=.03), mESC -0.015; submitted uniform control on the 3 cells
[20:31] A: write-ups W12_1, W12_4, W12_6, W12_9 written; now: results.txt / work.md appends, final checks
