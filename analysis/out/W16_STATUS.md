# W16 status (curated pipeline), newest last
[02:50] W16 start. Validation choice (analysis/w16_choose.py): hESC gcn had batch GF; hHEP gcn had batch GF; mHSC-E npanda had batch GF; mESC npanda had none GF. Added flag --dec_wd. Launching: laptop w16_std.txt (hESC/hHEP/mHSC-E arms, 30 runs); ada w16_mESC_base, w16_mESC_dwd0, w16_ho_gene, w16_ho_tf
[02:53] launched w16_sens.txt on laptop (sensitivity arm: npanda 2-layer config on hESC/hHEP, test-informed, not validation-chosen). DC-SBM validation refits (analysis/w16_dcval.py) running on laptop CPU. ada: mESC base/dwd0, ho_gene running; ho_tf pending
[03:01] all W16 runs done (laptop std+sens, ada mESC, ho_gene, ho_tf); DC-SBM validation refits done; running analysis/w16_pipeline.py
[03:02] main pipeline run done (FAIL on held-out TFs vs recipe; mHSC-E pipeline -0.0007 vs single). Launched ada w16_sens2.txt (test-picked configs mHSC-E gcn, mESC sage as sensitivity)
[03:07] W16 DONE: write-up analysis/out/W16_pipeline.md, report analysis/out/w16_pipeline.txt; results.txt and work.md appended. Verdict FAIL (held-out TFs; mHSC-E vs single)
