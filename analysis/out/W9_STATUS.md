# W9 status (item 7 round 1, hESC motif prior)
- 11:36 start. Plan: analysis/w9_motif_build.py (JASPAR 2024 + Ensembl promoters -> data/hESC/TFs+500/motif/*.npy), analysis/w9_motif_signal.py -> analysis/out/w9_motif_r1.txt
- 11:37 wrote analysis/w9_motif_build.py, running (Ensembl fetch ~60 requests, cached)
- 11:45 Ensembl expand=1 lookups are slow (~1 min per 10 genes, 91 batches, est. ~1.5 h); build running in background (analysis/out/w9_build.log). Could not stop it (kill denied by the sandbox classifier), so waiting. Signal script analysis/w9_motif_signal.py written, untested.
- 12:34 switched to non-expanded lookups (sym_*, tr_* caches); at transcripts step, then sequence fetch (19 batches) and scan
- 13:00 build done: 849/910 promoters, 143/410 TFs and 29/34 labelled TFs with a motif; npy written. Running w9_motif_signal.py
- 13:01 ALL DONE: signal check finished, results.txt and work.md appended. Stopped.
- 13:04 round 2 started: wrote analysis/w9_motif_r2.py; first run fetches -2000/+500 sequences (19 Ensembl batches, slow) then scans
- 13:25 round 2 DONE: analysis/w9_motif_r2.py -> analysis/out/w9_motif_r2.txt; results.txt and work.md appended. Stopped.
