# Week 4 plan, part A (written 2026-10-26): best model per cell type, then per-cell tuning

Status log: `analysis/out/W8_STATUS.md` (timestamped line per step). Stop after Part 2 and report.
Do NOT git commit; do NOT start any other experiments.

## Rules
- GPU, `CAP=4 analysis/queue.sh <jobfile>` only (shared 4 slots, 3.5 GB RAM gate unchanged). Never `pkill -f`.
- Rule A everywhere: epoch chosen on validation AUROC (`--select_metric auc`, the default), `--refit 1`,
  never anything chosen on test. 5 seeds (42-46) for every reported row. Never report a best seed.
- Paired test = per-TF test AUROC averaged over seeds, Wilcoxon over TFs. Reuse `analysis/w3_report.py`
  / `analysis/w4_report.py` helpers rather than new statistics code.
- Write-up goes in PROGRESS.md as a new section BELOW the last line `****** 25 Oct ******`; touch
  nothing above it. Simple language, short paragraphs, small tables, impersonal voice, and present
  everything as the work done (never "the advisor asked"). Run the `stop-slop` skill on the new text.
  Append numbers to results.txt and a dated (2026-10-26) entry to work.md, same style as before.

## Part 1: GCN + Geneformer vs SAGE + Geneformer, same setup, 4 cell types
Existing SAGE + Geneformer runs: `w3_herogf_s*` (hESC), `w4_<cell>_herogf_s*` (hHEP, mESC, mHSC-E);
args: `--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1 --encoder sage --split batch
--decoder had --gnn_dims 256 --scfm 1`. Last week a GCN 1-layer + Geneformer looked best on hHEP and
mHSC-E (PROGRESS.md "replication" section, 3 seeds); find its exact job lines in `analysis/jobs/` (e.g. grid*/master*/extra*).
- Run `w8_<cell>_gcngf_s42..46` = the SAGE+Geneformer args above with ONLY `--encoder gcn` changed
  (the controlled comparison), on all 4 cells.
- If last week's GCN+Geneformer job used different settings (dims, lr, etc.), also run it unchanged as
  `w8_<cell>_gcngf_lw_s42..46`, so last week's 3-seed number is checked at 5 seeds.
- Table per cell: SAGE+GF, GCN+GF (both variants), hero (no GF), published scRegNet recipe
  (`w4_<cell>_A_paper`, hESC `f02_A_paper`), with AUROC/AUPRC/within-TF mean ± std and paired tests:
  GCN+GF vs SAGE+GF, and best of them vs the published recipe. Name one best model per cell type.

## Part 2: tune the best model per cell type (hHEP first)
For each cell type, take the Part-1 winner (encoder + Geneformer) and tune it with `analysis/tune2.py`
(jobs / finals / report), selecting by validation AUROC (NOT `--select_metric within`; that failed last time).
- Order: hHEP, mHSC-E, hESC, mESC (mESC runs are slow; do it last, cut it if time runs out).
- Search: about 16 trials, over lr, wd, gnn_dims (64-512), mlp_dim, dropout, batch_size. Score each trial
  on validation AUROC averaged over 2 seeds (the validation slice is noisy).
- Finals: the chosen config at 5 seeds, `w8_<cell>_tuned_s*`. Paired tests vs the untuned winner and vs the
  published recipe. Report honestly if tuning does not help or validation picks badly.

## Report back
Files changed/created; Part-1 table and the best model per cell type; Part-2 tuned vs untuned vs recipe
per cell type with p-values; anything that went wrong or that you were unsure about.
