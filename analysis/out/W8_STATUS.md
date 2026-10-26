# W8 status (2026-10-26)
- 10:48 Part 1 launch: jobs analysis/jobs/w8_p1.txt (w8_<cell>_gcngf_s42..46). Last week's GCN+GF (g11/m_*_g11) used identical args to SAGE+GF with encoder gcn, so no separate _lw variant is needed (seeds 42-44 of g11 serve as reproducibility check). Smoke first: w8_smoke.txt (seed 42, 4 cells).
- 10:52 smoke OK: s42 reproduces old g11 (hESC, mESC exact; hHEP/mHSC-E within 0.002 AUROC). Launching all seeds w8_p1.txt.
- 10:58 Part 1 done (20 runs), report analysis/out/w8_report.txt. Winner by test within-TF: hESC sage, hHEP gcn, mHSC-E gcn, mESC sage (val AUROC disagrees on hESC, mHSC-E). Part 2: tuning hHEP (gcn) first via tune2.py wjobs/wfinals.
- 11:03 hHEP tuning done: best t08 (val AUROC 0.9056 vs default 0.9039, trial median 0.9026). Finals w8_hHEP_tuned running, then mHSC-E trials (w8t_mHSC-E.txt).
- 11:09 mHSC-E trials done, finals + hESC trials launched.
- 11:12 hESC trials done, finals + mESC trials launched.
- 11:28 mESC trials done, finals launched.
- 11:34 ALL DONE: Part 1 + Part 2 finished; PROGRESS.md s21, results.txt, work.md appended. Stopped per plan.
