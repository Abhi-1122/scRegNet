# Week 3 plan (written 2026-10-23 05:10 IST): PANDA as a graph, neural PANDA, context-specific stacker

Order: everything on hESC first (Parts 0-4). Only when hESC is complete and checked, move to
the other cell types (Part 5). Hard stop: all work finished and written up by 10:00 IST.
Live status log: `analysis/out/W3_STATUS.md` (append a timestamped line at every step).

## UPDATE 2026-10-23: the other process was stopped by the user; the machine is ours.
Use the GPU (no `CUDA_VISIBLE_DEVICES=`), `CAP=4` (the queue's 3.5 GB RAM gate stays). All rows in a
table must still come from one device: rows already finished on CPU get re-run on GPU. The rules
below about PID 169625 and CPU-only no longer apply; the RAM rules still do.

## Resource rules (another heavy process shares this machine)
- PID 169625 (`scripts/train_gnn.py`, about 5 GB RAM, GPU 7.1/8 GB at 100%) belongs to another
  project. Never touch, kill, renice or wait on it. Never `pkill -f`; kill only our own PIDs.
- All our torch jobs run on CPU: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CAP=2 analysis/queue.sh <jobfile>`.
  Never use the GPU. CAP=2, never more. Keep the queue's 3.5 GB free-RAM gate as is.
- GPU sharing: once PID 169625 has exited AND `nvidia-smi` shows under 2 GB used, new queue runs may
  use the GPU (drop `CUDA_VISIBLE_DEVICES=`) with CAP up to 4. Re-check before every launch; if a new
  GPU job from another project appears, go back to CPU. Rows inside one table must all come from the
  same device (re-run the CPU hero on GPU if a table switches device).
- If the queue waits more than 20 min for RAM, do not lower the gate; log it in W3_STATUS.md and
  keep doing CPU-light work (stacker, write-up).
- Check `free -g` before launching anything outside the queue (PANDA builds, stackers): need >3 GB available.

## Shared definitions
- Hero = `--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1 --encoder sage --split batch --decoder had --gnn_dims 256 --scfm 0`
  (`--scfm 1` = hero + Geneformer). Epoch chosen on validation AUROC (rule A), then refit.
- Seeds 42-46 (5 seeds) for every hESC row.
- Re-run the hero and hero+Geneformer on CPU as `w3_hero_s*` / `w3_herogf_s*` so all rows in the
  new tables come from the same device (the old d01/d02 runs were on GPU).
- Every comparison: mean +- std over seeds for AUROC, AUPRC, within-TF; paired per-TF test against
  the CPU hero (per-TF AUROC averaged over seeds, Wilcoxon over the 33 test TFs). Reuse the existing
  per-TF/paired code in `analysis/` (agg.py / final_report.py / tune2.py report) instead of new code.
- PANDA fold scheme must match `analysis/scalars.py`: folds = `np.array_split(RandomState(42).permutation(len(trn)), 5)`
  over the training rows `trn = tr.iloc[tri]`; fold-k PANDA = PANDA with fold k's positives removed
  from the prior. "Full" PANDA = prior from all `tri` positives (validation slice excluded; with
  `val_frac 0` for refit, all training positives). Test edges are never in any prior.

## Part 0: setup
- `CUDA_VISIBLE_DEVICES=` smoke test: one hero run finishes on CPU; note its minutes and peak RSS.
- Add the per-fold PANDA matrices to the cache: extend `analysis/scalars.py` so `get()` also saves
  `panda_v{vf}_s{seed}_fold{k}.npy` (the fold-k matrix, same scale as full). Small diff, same style.

## Part 1: PANDA network as the message-passing graph (advisor item: "put PANDA in place of the GRN")
New `src/pair_gnn.py` option `--mp_graph {grn,panda,union}` (default `grn` = unchanged behaviour),
`--panda_k K` (top-K genes per TF by PANDA score, over the 34 TFs that have labels), and
`--panda_safe {1,0}`.
- `panda`: message-passing graph = PANDA top-K edges only (undirected, as the GRN graph is now).
- `union`: GRN training-positive edges (with the usual per-batch split) plus PANDA top-K edges.
- Leak control (`--panda_safe 1`, default): the PANDA prior contains the training positives, so a
  batch's own pair could pass messages through PANDA edges that exist because of that pair. Make
  training batches fold-pure: each epoch, shuffle the folds and the rows inside each fold, batch
  within a fold, and use the fold-k PANDA top-K graph for fold-k batches. Evaluation (val/test)
  uses the full PANDA top-K graph.
- `--panda_safe 0` = naive: full PANDA graph for all training batches. This run exists only to
  measure the leak (advisor item "why is having a prior a leak").
Runs (hero settings otherwise, 5 seeds each): `w3_pg_k50`, `w3_pg_k100`, `w3_pg_k200`,
`w3_pgU_k100` (union), `w3_pg_k100_naive`, and the best of the safe ones again with `--scfm 1`.
Also report, as a diagnostic, how many test positives are in each PANDA top-K set (recall of the
graph itself) - computed with a few lines, no training.

## Part 2: neural PANDA (advisor items: graph-based PANDA on single-cell data, PANDA scoring in the GNN)
New encoder `--encoder npanda` in `src/pair_gnn.py` with three relations, summed per layer:
`h' = lin_self(h) + SAGE_R(h, R) + SAGE_C(h, C) + SAGE_P(h, P)` (SAGEConv with `root_weight=False`
for the three relations), ReLU + dropout between layers.
- R (TF programme): the GRN training-positive graph with the usual per-batch split.
- C (co-expression neighbours): label-free gene-gene kNN graph, k=10, top positive Pearson
  correlations of each gene across cells, symmetrised. `--coexp meta` builds it on metacells
  instead: KMeans (k=75, seed 0) on the first 30 PCs of the cells, average expression per cluster,
  then the same kNN.
- P (PPI cooperativity): STRING TF-TF edges among the 910 genes from
  `data/hESC/TFs+500/STRING/*.symmetric_edges.csv` (or `.edges.csv` symmetrised). Label-free.
- `--relations` picks a subset, e.g. `RCP`, `RC`, `RP`, `CP` (ablations).
- Default 2 layers (`--gnn_dims 256 256`): layer 1 gives each TF its targets (programme), layer 2
  passes a partner TF's programme over P, which is PANDA's "responsibility" (P x W); the decoder
  then compares the TF programme with the gene's co-expression neighbourhood ("availability", W x C).
New decoders in `PairGNN`:
- `tani`: H=6 head Tanimoto. Split the D-dim vectors a, b into 6 chunks; per chunk
  `T = a.b / (|a|^2 + |b|^2 - a.b + 1e-6)`; logit = Linear([a, b, T_1..T_6]) (keeps the per-node
  terms that carry popularity).
- `tani_pure`: logit = Linear([T_1..T_6]) only (PANDA-like, no popularity terms).
  `--mlp_dim 198` splits into 6 x 33.
Runs (hero optimiser settings, 5 seeds each):
`w3_np_full` (RCP, tani, 2 layers), `w3_np_noC` (RP), `w3_np_noP` (RC), `w3_np_noR` (CP),
`w3_np_had` (RCP, had), `w3_np_1L` (RCP, tani, `--gnn_dims 256`), `w3_np_meta` (RCP, tani, metacells),
`w3_np_pure` (RCP, tani_pure), `w3_np_gf` (full + `--scfm 1`), and `w3_hero_tani` (hero with the
tani decoder, to separate the decoder from the graph).
Sanity checks before launching seeds: shapes, no test pair in R, C and P built without labels,
a 1-seed run of each variant finishes.

## Part 3: context-specific stacker (advisor items: PANDA graph in the stacker; which graph popularity runs on)
New CPU script `analysis/stacker_graphs.py` (reuse `feature_mats`, folds and stacking code from
`analysis/cs_baselines.py` by importing or copying the small parts; same 5 folds, val_frac 0).
Today popularity and the stacker run on the training ChIP-seq label graph only. Add:
- PANDA-graph features: `feature_mats` on the binarised PANDA top-100 graph (34 label TFs x 910),
  plus the raw PANDA score. Test rows use full PANDA; training rows use the fold-k PANDA of their
  fold (no row sees its own edge).
- Co-expression-graph features (label-free, from the kNN graph of Part 2): Pearson corr(t, g),
  common neighbours and Adamic-Adar of t and g, degree of g and of t ("context popularity"), and
  item-CF on the co-expression graph (mean kNN adjacency between g and the TF's known targets,
  out-of-fold for training rows).
Stacks (LR and GBM, as in cs_baselines): existing best label stack (baseline), context-only
(PANDA-graph + co-expression features, no label-graph features), label + PANDA graph,
label + co-expression graph, all. Also each single context heuristic alone.
Report test AUROC/AUPRC/within-TF and paired per-TF tests vs the existing best stacker and vs the
CPU hero.

## Part 4: hESC write-up
- `analysis/out/w3_report_hESC.txt`: all tables (Part 1, 2, 3), paired tests, graph recall diagnostic,
  leak measurement (naive vs safe), seconds/run.
- PROGRESS.md: new sections BELOW the line `****** 18 Oct ******` (append at end; never edit text
  above it). Simple language, complete sentences, short paragraphs, small tables, impersonal voice
  (no I/we/you). Then run the `stop-slop` skill on the new text only.
- Append numbers to `results.txt` and a terse dated entry to `work.md` (repo root), same style as
  existing entries.
- No git commits.

## Part 5: other cell types (only after Parts 0-4 are done; stop at 10:00 IST whatever the state)
Data download is approved. hHEP = human, mESC and mHSC-E = mouse.
- Download STRING v12 for human (9606) and mouse (10090) into `data/external/STRING/` (new dir),
  build `data/<cell>/TFs+500/STRING/<cell>_TFs500_STRING_PPI.edges.csv` in the SAME format and with
  the SAME score threshold as the hESC file (find how the hESC file was made first: git log, old
  scripts, file header/scores). Mouse gene symbols: use the mouse STRING aliases.
- Then: scalars cache (`python3.12 analysis/scalars.py <cell> 500`), and run only the headline rows,
  3 seeds: CPU hero, best safe PANDA-graph variant, `w3_np_full`, best neural-PANDA ablation if it beat
  full; plus the stacker script per cell type.
- Add these as a short cross-cell-type table to the report and PROGRESS.md.
