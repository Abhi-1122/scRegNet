[22:38] W15 start: Part 3 jobs (w15_gf.txt, 22 shuffled-Geneformer runs) queued on laptop; baselines reused from grid/w3/w6
[22:44] W15: lightgcn encoder added to src/pair_gnn.py (hero reproduces: w15_chk_hero_s42 within 7e-5 of w3_hero_s42, same epoch); sweep 144 runs at seed 42 (none on ada, expr on laptop)
[22:45] W15 part 3: all 22 shuffled runs done, report analysis/w15_gf_report.py -> analysis/out/w15_gf.txt (shuffled keeps none of the gain; real GF helps via biology); write-up pending
[22:52] W15: dot decoder stalls at default init (val 0.57-0.79); extra dot sweep with larger init std, patience 30 (w15_swdot.txt, 16 runs)
[23:08] W15: ada slow; remaining 'none' sweep runs also started on the laptop in reverse order (w15_sw_none_rev.txt)
[23:16] W15: sweep done (144 runs). Chosen on val AUROC: pure = K2 d128 had L2 1e-4 lr 1e-3; hybrid = expr K2 d64 L2 1e-4 lr 1e-4; dot best = std0.5 lr 1e-2 (val 0.8265, far below had 0.886). Running w15_lgc.txt (4 cells x 6 variants x 3 seeds + dot) and w15_lgc_ho.txt (held-out)
[23:32] W15 part 3 appended to results.txt and work.md; part 2 final runs in progress
[00:00] W15: mESC runs take 7-20 min each; to finish in time the feature-only ablations (featexpr/featgf) on mESC and mHSC-E and stats on mHSC-E were dropped (stub json placeholders, removed after the queue passes them); ablations stay on hESC and hHEP, stats on hESC, hHEP, mESC
[00:17] W15: standard runs done (stubs removed); held-out runs in progress; 4 diagnostic mESC runs (larger d) queued since pure LightGCN trails the hero by 0.058 on mESC
[00:41] W15 done: part 2 report analysis/out/w15_lgc.txt, write-up W15_lightgcn.md; results.txt/work.md appended
