# W12 protocol (7 Nov graph ideas 1, 2, 3, 4, 6, 9)

Shared rules for both workers. Worker A: ideas 1, 4, 6, 9 (inside `src/pair_gnn.py` + one post-hoc script).
Worker B: ideas 2 (NBFNet) and 3 (SEAL) as new modules `src/nbf.py`, `src/seal.py`.

## Baseline and comparison
- Hero = `python3.12 -m src.pair_gnn --optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1 --encoder sage --split batch --decoder had --gnn_dims 256 --scfm 0` (+ `--cell_type`, `--seed`, `--protocol`, `--fold`).
- Existing baseline runs (do not rerun): hESC random split `w3_hero_s42..46`; hESC + Geneformer `w3_herogf_s42..46`;
  held-out genes `w5_ho_gene{0..4}_hero_s42..46`; held-out TFs `w5_ho_tf{0..4}_hero_s42,s43`;
  other cells (no Geneformer) `w11_grid_{hHEP,mESC,mHSC-E}_sage_had_batch_gf0_s42..46` (grid, may still be finishing).
- SEEDS: 3 seeds (42, 43, 44) everywhere (user decision, 2026-11-12, for speed). Compare against the baseline runs at the
  same 3 seeds only (pass the seed list to the report helpers). Held-out protocols stay at 2 seeds (42, 43) per fold.
- Stage 1 (screen): hESC, random split, seeds 42-44, same epoch selection (validation AUROC) and refit as the hero.
  A small sweep of the idea's main knob (2-4 values) at seed 42 (or 42-43) is allowed first; run the chosen value at 3 seeds.
  The value must be chosen on validation AUROC/within-TF (the `val` field), never on test.
- Stage 2 (only if stage 1 is within 0.005 of the hero or better on within-TF, or the idea is aimed at unseen genes/TFs):
  held-out genes (5 folds x seeds 42,43) and held-out TFs (5 folds x seeds 42,43), protocols `--protocol gene|tf --fold f`.
- Stage 3 (only if it beats the hero somewhere): the other 3 cells at 3 seeds, and +Geneformer (`--scfm 1`) on hESC.
- Report test AUROC, AUPRC, within-TF (mean +- std over seeds), and paired per-TF tests vs the hero: per-TF test AUROC
  averaged over seeds, difference, TFs better, Wilcoxon p. Reuse `analysis/w3_report.py` (`runs`, `row`, `tf_mean`, `paired`, `HDR`);
  see `analysis/w10_report.py` for the pattern. For held-out TFs also report `within8`. Never report the best seed.
- Leak rule: a training pair must never be scored through its own edge (or its reverse edge) in any graph, feature or
  propagation it uses. Validation/test labels never enter any graph, feature or propagation. If a new feature is computed from
  the training graph, say how the pair's own edge is kept out, or run the `--own_edge_diag 1` style check to show it does not matter.

## Compute
- Laptop: always through `CAP=4 analysis/queue.sh analysis/jobs/<file>.txt` (lines: `<name> <args>`; shared 4-slot lock with
  other jobs, includes a running grid). Never `pkill -f`; kill by PID only. 15 GB RAM, 8 GB GPU.
- ada (SLURM cluster) is owned by the ada-handler agent; workers never submit there themselves. Drop a job file in
  `analysis/jobs/ada_inbox/<name>.txt` (lines `<name> <args>`, optional first line `#mod <module> [P]`, default
  `src.pair_gnn 4`). The handler pushes code only (`src/ analysis/*.py analysis/*.sh analysis/jobs/`; data and scFM are
  already on ada), submits on RTX 2080 Ti (max 4 one-GPU jobs), pulls results into `analysis/out/pair/` every ~5 min, and
  logs status and tracebacks in `analysis/out/ADA_STATUS.md`. Venv `~/venv312`. Keep `src/pair_gnn.py` importable at every save.
- Use both machines at once: e.g. screens on one, held-out protocols on the other.

## Naming and files
- Run names: `w12_<idea>_<variant>_s<seed>` (random split), `w12_ho_gene<f>_<idea>_<variant>_s<seed>`, `w12_ho_tf<f>_...`,
  other cells `w12_<cell>_<idea>_<variant>_s<seed>`. Job files `analysis/jobs/w12_<idea>*.txt`.
- New flags in `src/pair_gnn.py` default to off, so every old run reproduces (check: one `w3_hero` seed reproduces exactly).
- Scripts `analysis/w12_<idea>_*.py`, outputs `analysis/out/w12_<idea>*.txt`.
- Live status: append to `analysis/out/W12_STATUS.md` (one line per milestone, with time) as work happens.
- Append final numbers to `results.txt` and a terse dated (2026-11-07) methodology entry to `work.md` (append with `cat >>`, other worker writes too).

## Write-up per idea (`analysis/out/W12_<n>_<idea>.md`, the planner moves it into PROGRESS.md)
Impersonal report voice (no I/we/you), plain words, no em dashes, no adverbs-as-filler. Sections:
1. **Where the idea comes from**: the home-domain method (paper) and which project finding motivated trying it here.
2. **Method in short**: what was built, flags, the knob values tried, how leaks are prevented.
3. **Results**: tables (stage 1, then 2/3 if run) with paired tests.
4. **Why**: the explanation. If it improved: what extra information the model gained, shown with a diagnostic test
   (e.g. per-TF gain vs a TF property, score correlation with gene popularity, ablation of the new part, gain on
   genes/TFs where the new information exists vs where it does not). If it regressed: at most ~10 minutes of diagnostic
   tests, then state the most likely reason with the evidence. Say plainly what is measured and what is a guess.
5. **Verdict** in one or two lines, and what to tune next if the config looks promising.
