# W5: new ideas on hESC (coordinator's own line, 2026-10-23)

Ideas, from earlier observations and advisor remarks:
1. Within-TF ranking loss (BPR, positive vs 4 same-TF training negatives): the metric that matters is within-TF, BCE optimises the pooled one. `--rank_w`.
2. PANDA as a teacher: auxiliary head regresses the out-of-fold PANDA score on random (TF, gene) pairs, so unlabelled pairs give a signal. `--aux_panda`.
3. Neural PANDA (had) where labels are missing: held-out TFs / held-out genes, since C and P are label-free.
4. Late fusion: z-averaged logits of hero + stacker / Geneformer hero / np had (post hoc, `analysis/w5_ensembles.py`).
5. Early fusion done right: PANDA/contrast numbers entered into the hero failed before; hypothesis = the scalar's weight
   cannot move in ~10 epochs at lr 1e-4. `--scalar panda,contrast --scalar_lr 100|1000` (separate term, own lr, no wd).

Results (`analysis/out/w5_report_hESC.txt`, `w5_ensembles_hESC.txt`):
- 1: no gain (−0.005 to −0.008 within-TF). 2: no gain (−0.005 to −0.007).
- 3: held-out genes +0.012 within-TF (24/34 TFs, p=0.057); held-out TFs no change.
- 4: hero + stacker 0.7594 within-TF (+0.012, p=0.011) beats hero x4 seeds (0.7529). 4-way 0.7626.
  Equal-weight averaging with the raw PANDA/contrast numbers hurts; the stacker's weighting matters.
- 5: running (`analysis/jobs/w5_b.txt`).
