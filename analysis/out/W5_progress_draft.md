## 19. New ideas tested on hESC (late fusion, ranking loss, PANDA teacher, scalar weight, cold start)

### Why
Sections 17 and 18 showed that PANDA does not help when it is put inside the GNN: not as the
message-passing graph, not as a neural encoder with a Tanimoto decoder. Five further ideas were
tested on hESC, each aimed at a specific earlier observation. Every GNN row uses the hero settings,
5 seeds (42-46), the GPU, epoch chosen on validation AUROC, then refit. Paired tests are per-TF
AUROC averaged over seeds, Wilcoxon over the 33 test TFs, against the hero (0.7470 within-TF).

| Idea | Observation behind it | Flag / script |
|---|---|---|
| Within-TF ranking loss | The metric that matters is within-TF, but training uses pooled BCE | `--rank_w` |
| PANDA as a teacher | PANDA scores all 31k TF-gene pairs, labels cover 10k | `--aux_panda` |
| A free weight for the PANDA / contrast number | These numbers never helped inside the GNN; their weight may not move in ~10 epochs at lr 1e-4 | `--scalar panda,contrast --scalar_lr` |
| Late fusion | The stacker and the GNN may make different errors | `analysis/w5_ensembles.py` |
| Neural PANDA without labels | Its co-expression and PPI edges need no labels, so they may help unseen genes and TFs | `--protocol gene/tf` |

### Result 1: only late fusion helps
| Model (5 seeds) | AUROC | AUPRC | Within-TF | Diff vs hero (p) |
|---|---|---|---|---|
| Hero | 0.8884 | 0.6255 | 0.7470 | |
| Hero + ranking loss (w = 1) | 0.8860 | 0.6165 | 0.7391 | −0.008 (0.046) |
| Hero + PANDA teacher (w = 1) | 0.8882 | 0.6280 | 0.7411 | −0.006 (0.17) |
| Hero + PANDA number, fixed lr | 0.8886 | 0.6261 | 0.7475 | +0.001 (0.22) |
| Hero + contrast number, free weight (lr × 1000) | 0.8854 | 0.6210 | 0.7352 | −0.012 (0.0003) |
| Hero + both numbers, free weight (lr × 1000) | 0.8864 | 0.6236 | 0.7307 | −0.016 (0.006) |
| Hero, 4 seeds averaged (compute control) | 0.8912 | 0.6330 | 0.7529 | +0.006 (3e-6) |
| **Hero + stacker (late fusion)** | 0.8927 | 0.6364 | **0.7594** | +0.012 (0.011) |
| Hero + Geneformer hero + neural PANDA + stacker | 0.8945 | 0.6423 | 0.7626 | +0.016 (0.003) |

Late fusion means averaging the z-scored test scores of separately trained models, seed by seed.
The stacker is the logistic-regression stack with all features (label graph, contrast, PANDA,
PANDA graph, co-expression graph); it is trained on out-of-fold features, so no row sees its own label.

- The ranking loss and the PANDA teacher both lower within-TF slightly.
- Giving the PANDA or contrast number a free weight makes things worse, not better. The GNN leans
  on the number and loses within-TF ranking. So the old null result was not a learning-rate artefact.
- The same numbers help when they stay in a separate model. Two models, hero + stacker, beat four
  hero seeds averaged (0.7594 against 0.7529), so the gain is not just from averaging.
- Averaging the hero with the raw PANDA number at equal weight does not help (0.7446), and with the
  raw contrast number it hurts (0.6924). The stacker's learned weighting matters.

### Result 2: unseen genes and TFs
| Held-out genes (5 folds × 5 seeds) | AUROC | AUPRC | Within-TF | Diff vs hero (p) |
|---|---|---|---|---|
| Hero | 0.8461 | 0.5220 | 0.5464 | |
| Neural PANDA, had decoder (R + C + P) | 0.8461 | 0.5082 | 0.5624 | +0.021 (0.041), 24/34 TFs |
| Neural PANDA without PPI (R + C) | 0.8445 | 0.5018 | 0.5444 | −0.014 (0.81) |

| Held-out TFs (5 folds × 2 seeds) | AUROC | AUPRC | Within-TF | Diff vs hero (p) |
|---|---|---|---|---|
| Hero | 0.6304 | 0.2314 | 0.6855 | |
| Neural PANDA, had decoder | 0.6518 | 0.2347 | 0.6897 | +0.004 (0.95) |

- For genes never seen in training, neural PANDA ranks targets a little better within each TF
  (+0.021, p = 0.041), but its AUPRC is lower. The gain needs the PPI edges: without them it vanishes.
  With about ten comparisons in this section, p = 0.041 is weak evidence.
- For TFs never seen in training, neural PANDA changes nothing.

A CPU check for held-out TFs (`analysis/w5_heldout_tf_transfer.py`) scored a TF with no training
targets by borrowing the targets of similar TFs, which is PANDA's "responsibility" step:

| Score for a held-out TF | AUROC | Within-TF |
|---|---|---|
| Popularity (gene's training degree) | 0.669 | 0.724 |
| Borrow targets of co-expressed TFs | 0.646 | 0.701 |
| Borrow targets of STRING partners | 0.594 | 0.640 |
| Direct co-expression of TF and gene | 0.475 | 0.504 |
| Popularity + co-expression borrowing | 0.666 | 0.721 |

Popularity still wins for TFs with no known targets. The direct co-expression of a TF with a gene is
at chance in hESC (0.504). Co-expression carries signal only as the contrast against the TF's known
targets (Section 12), which needs labels.

### What this means
- PANDA and the co-expression contrast add information, but only as a separate model whose score is
  combined at the end. Every attempt to feed them into the GNN (as a graph, an encoder, an input
  number, a teacher) failed or hurt.
- The best hESC result so far is the four-way late fusion, 0.7626 within-TF, +0.016 over the hero.

### Result 3: late fusion holds on all four cell types
The fair control for a blend of four models is the hero averaged over four seeds, which costs the
same. Within-TF AUROC, 5 seeds, paired per-TF test against that control:

| Cell type | Hero | Hero × 4 seeds | Hero + Geneformer hero | Four-model blend | Blend vs hero × 4 (p) |
|---|---|---|---|---|---|
| hESC | 0.7470 | 0.7529 | 0.7540 | 0.7626 | +0.010 (0.007) |
| hHEP | 0.7599 | 0.7626 | 0.7742 | 0.7754 | +0.013 (2e-5) |
| mESC | 0.8235 | 0.8283 | 0.8321 | 0.8389 | +0.011 (5e-8) |
| mHSC-E | 0.7616 | 0.7641 | 0.7739 | 0.7726 | +0.009 (0.07) |

The four-model blend is the hero, the Geneformer hero, neural PANDA (had decoder) and the stacker.
AUROC and AUPRC also rise on every cell type (`analysis/out/w5_ensembles_<cell>.txt`).

- The blend beats the same-cost seed average on all four cell types, significantly on three.
- Which member helps most depends on the cell type. The Geneformer hero is the most reliable partner:
  it beats the four-seed average on its own in hHEP, mESC and mHSC-E. The stacker helps in hESC and
  mESC but not in mHSC-E.
- The members differ in input (Geneformer or not), encoder (GRN only, or GRN + co-expression + PPI)
  and model family (GNN or logistic regression on hand-made features). Their errors differ enough
  that averaging gains more than extra seeds of one model.
