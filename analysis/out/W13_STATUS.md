# W13 status
[21:00] read protocol; qatt code path audited (see below); implementing dcsbm scalar in analysis/scalars.py
[21:02] launched w13_a (qs x4 cells x3 seeds) and w13_b1 (dcsbm scalar, concat vs lr10, seed 42) on the laptop; dcsbm scalar reproduces w12_stat test within-TF exactly (hESC .7319, hHEP .8006)
[21:02] submitted hHEP held-out genes (20 runs, base vs qs) to ada, 1 GPU, P=5
[21:09] hHEP held-out (20 runs) finished on ada and pulled
[21:09] dcsbm variant chosen on seed-42 val AUROC: hESC dclr10, hHEP dc, mHSC-E dc, mESC dclr10. launched w13_b2 (laptop, seeds 43-44) and w13_both (ada, qs + dcsbm, 12 runs). Early read: summary on top of best configs ~0 or negative (hHEP -0.003, mHSC-E -0.015), held-out hHEP genes -0.006 (p=.07)
[21:16] own_edge_diag mESC: positive logit gap base +0.132, qs +0.127 (AUROC present/removed .959/.951 both): the summary adds no own-edge cue. Fixed scalars.py to import netZooPy lazily (ada has no netZooPy; the 'both' runs crashed there), resubmitted w13_both
[21:22] correction: seed-42 comparison for hHEP/mHSC-E was incomplete when b2/both were launched; full val AUROC picks dclr10 for hHEP (.9050 vs .9028) and mHSC-E (.9258 vs .9248). Running w13_fix (dclr10 seeds 43-44 + qs+dclr10, 10 runs); the dc runs already made stay as an extra row
[21:29] all runs done (w13_fix included); running report
[21:32] ada vs laptop mismatch: hESC qs+dclr10 on ada gave within-TF .7350/.7342/.7340 (AUROC .8866, seed sd 0.0001) but the same runs on the laptop gave .7390/.7538/.7544; ada results for the held-out runs and qs+dcsbm cells moved to analysis/out/ada_w13_discarded and rerun on the laptop (w13_redo.txt, 26 runs). Cause not found (torch unknown on ada: venv path differs)
[21:35] diagnostic for the qs null on hHEP: which factor removes the section-25 gain (Geneformer or the 2-layer npanda encoder): w13_diag2.txt, 6 runs
[21:46] report, W13_leads.md, results.txt and work.md done
