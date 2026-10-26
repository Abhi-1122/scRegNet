# scRegNet: week 3 summary (17–24 October)

As of 2026-10-23.

PANDA did not help in any form tested inside the GNN: not as the message-passing graph, not as a neural encoder, not as an input number or a teacher. Neural PANDA ties the hero within-TF and helps a little on genes never seen in training.

## Setup

The task is to predict which genes each transcription factor (TF) regulates, from single-cell RNA-seq. Ground truth is cell-type-specific ChIP-seq (BEELINE, TFs + 500 genes).

- **Data:** hESC first (910 genes, 34 TFs, 3,030 training and 1,061 test positives), then hHEP (human liver), mESC (mouse embryonic stem cells) and mHSC-E (mouse erythroid progenitors).
- **Metrics:** AUROC, AUPRC, and **within-TF AUROC**: AUROC computed separately for each TF, then averaged. Pooled AUROC mostly rewards gene popularity: a gene that many TFs regulate scores high for every TF. Within-TF AUROC asks the biological question: for one TF, does the model rank its real targets above other genes?
- **Reference model ("hero"):** a one-layer GraphSAGE over the training regulatory graph, plus a decoder that multiplies TF and gene embeddings (biased matrix factorisation). During training, each batch's own edges are removed from the graph, so a pair is never scored through its own label.
- **Protocol:** 5 seeds per model. The epoch is chosen on a held-out validation slice by AUROC, never on the test set. Models are compared with a paired Wilcoxon test over TFs on per-TF AUROC. Reported numbers are seed means, never the best seed.

## This week's work

Six of the eight items are done; the edge-fix checks and the full consolidated grid are still open.

| Work | Status | Result in one line |
| --- | --- | --- |
| Used the final PANDA network in place of the GRN, for message passing as a prior | Done, 4 cell types | Never beats the hero; half the PANDA graph is the training edges themselves |
| Built a graph-based PANDA for single-cell data (neural PANDA) | Done, 4 cell types | Ties the hero within-TF; small AUROC/AUPRC gain |
| Measured why a prior leaks | Done | A prior built from training labels lets the model echo its own labels; it must be built out-of-fold |
| Further checks of the edge fix | Open | Three planned tests not yet run |
| Added the PANDA knowledge graph to the stacker | Done, 4 cell types | No significant gain |
| Tested which graph popularity and the stacker should run on, including context-specific ones | Done, 4 cell types | They run on the training ChIP-seq labels; PANDA and co-expression graphs do not replace them |
| Adapted PANDA scoring to the GNN | Done | A Tanimoto decoder (PANDA's similarity) hurts within-TF |
| Collected all results in one place, every combination | Partly | Per-week and cross-cell tables exist; the full factorial grid is not run |

## PANDA as the message-passing graph

Replacing the regulatory graph with the PANDA network lowered within-TF AUROC on every cell type (hESC −0.024 at the top 100 targets per TF, p = 0.009).

PANDA combines three networks: a prior (which TF binds which gene; here, the training ChIP-seq edges), TF–TF protein interactions (STRING) and gene–gene co-expression. It iterates until TFs that interact share targets and co-expressed genes share regulators. Its top-K targets per TF (K = 50, 100, 200) were used as the GNN graph, alone or added to the regulatory graph.

| hESC, 5 seeds | AUROC | AUPRC | Within-TF | vs hero (p) |
| --- | --- | --- | --- | --- |
| Hero (regulatory graph) | 0.8884 | 0.6255 | 0.7470 | |
| PANDA graph, K = 100 | 0.8822 | 0.6117 | 0.7231 | −0.024 (0.009) |
| Regulatory graph + PANDA K = 100 | 0.8836 | 0.6151 | 0.7298 | −0.017 (0.009) |
| PANDA graph K = 100, built naively (leaks) | 0.8659 | 0.5847 | 0.6918 | −0.055 (2e-6) |

**Why the PANDA graph adds little.** 1,685 of its 3,400 edges (K = 100) are simply training edges, because PANDA's prior is the training labels. The ~1,700 new edges are depleted of true targets: test pairs inside the graph are positive 8% of the time, against a 15% base rate.

**Why a prior can leak.** If the PANDA prior contains a training pair's own edge, PANDA scores that pair high because it was told so. Inside the full PANDA graph sit 56% of training positives, but only 3% of test positives. A model trained on that graph learns "in the graph means target". Test pairs never have that cue, so test scores drop (the naive row above), and training stops after 2–3 epochs. The fix is out-of-fold construction: each training batch sees a PANDA network built without its own fold's edges. Test pairs are never in any prior, so they are safe either way.

## Neural PANDA

A GNN built from PANDA's three ingredients ties the hero within-TF and adds a small AUROC/AUPRC gain; PANDA's own similarity score, used as the decoder, hurts.

Each layer passes messages over three edge types, learned with separate weights:

- **R, TF programme:** the training regulatory edges (each batch's own edges removed, as in the hero).
- **C, co-expression:** each gene's 10 most correlated genes across cells. No labels used.
- **P, cooperativity:** STRING protein–protein interactions between TFs. No labels used.

Two layers mirror PANDA's two steps. Layer 1 gives each TF a summary of its targets. Layer 2 passes a partner TF's programme over P (PANDA's "responsibility"). The decoder then compares a TF's programme with a gene's co-expression neighbourhood (PANDA's "availability").

| hESC, 5 seeds | AUROC | AUPRC | Within-TF | vs hero (p) |
| --- | --- | --- | --- | --- |
| Hero | 0.8884 | 0.6255 | 0.7470 | |
| Neural PANDA, Tanimoto decoder (PANDA's score) | 0.8852 | 0.6210 | 0.7180 | −0.029 (0.04) |
| Neural PANDA, product decoder | 0.8898 | 0.6315 | 0.7410 | −0.006 (0.86) |
| Hero with only the decoder swapped to Tanimoto | 0.8837 | 0.6205 | 0.7174 | −0.030 (0.10) |

- **The decoder decides.** Swapping in the Tanimoto score costs about 0.03 within-TF whether or not the encoder is PANDA-like. Tanimoto squeezes the 198-dimensional TF × gene interaction into 6 similarity numbers, so the model loses most of its capacity to tell one TF's targets from its non-targets.
- **With the product decoder,** neural PANDA beats the hero on AUROC and AUPRC on all four cell types (on hESC in 4 of 5 and 5 of 5 seeds), but the gains are small (+0.001 to +0.003 AUROC). Within-TF it is level, except hHEP (+0.015, p = 0.0004).
- **Ablations:** removing any one of R, C or P, or building C on metacells (averages of ~10 similar cells), changes within-TF by less than the seed noise.

## Popularity and the stacker on context-specific graphs

Popularity and the stacker run on the training ChIP-seq edges; neither the PANDA network nor a co-expression graph can replace them.

The stacker is a logistic regression (or gradient boosting) on hand-made link-prediction features: gene and TF popularity (how many training edges they have), collaborative filtering ("TFs like this one target this gene"), path counts and resource allocation. The same features were recomputed on two context-specific graphs: the PANDA network (built out-of-fold) and the co-expression neighbour graph of each cell type.

| hESC stacker (single fit) | AUROC | AUPRC | Within-TF |
| --- | --- | --- | --- |
| Training-label graph + contrast + PANDA score (old best) | 0.8871 | 0.6171 | 0.7279 |
| All features, both new graphs added | 0.8893 | 0.6194 | 0.7423 |
| PANDA + co-expression graphs only (no label graph) | 0.8586 | 0.5611 | 0.6673 |
| Co-expression graph only | 0.7995 | 0.4342 | 0.5324 |

- Adding the new graphs gives a small, non-significant gain (p = 0.07). Without the label graph, the stacker falls 0.08 below the hero within-TF.
- On its own, a gene's co-expression with a TF says almost nothing about whether the TF regulates it (within-TF 0.50, chance). Co-expression carries signal only as a **contrast**: is the gene more co-expressed with the TF's known targets than with its non-targets? That needs labels.
- The same holds on hHEP, mESC and mHSC-E: no stacker variant beats the hero within-TF.

## Other ideas that did not help

Five further ways of putting more information into one GNN all tied with the hero or hurt it.

| Idea (hESC, 5 seeds) | Why it was tried | Within-TF vs hero (p) |
| --- | --- | --- |
| Within-TF ranking loss: a positive must outscore a random non-target of the same TF | Train on the metric that matters | −0.008 (0.05) |
| PANDA as a teacher: an extra head predicts PANDA's score on unlabelled TF–gene pairs | PANDA covers 31k pairs, labels only 10k | −0.006 (0.17) |
| PANDA and contrast numbers fed to the decoder with their own fast learning rate | Earlier, these numbers never helped; maybe their weight could not move | −0.016 (0.006) |
| Hard negatives: the most popular non-target of the same TF | Teach the model what is not a target | −0.037 (2e-5) |
| Hard negatives: the non-target the model currently scores highest | Same, chosen by the model | −0.002 (0.55) at best |

- **Feeding PANDA in, any way, fails.** Graph, encoder, input number, teacher: none helps. With a free weight, the GNN leans on the number and ranks targets worse within each TF.
- **Popular non-targets are not mistakes to correct.** A gene regulated by many TFs is more likely to be a target of the next TF as well. Pushing such genes down removes real signal, and within-TF falls by up to 0.10.

## Unseen genes and unseen TFs

For genes never seen in training, Geneformer helps most and neural PANDA a little; for TFs never seen in training, nothing beats popularity.

The benchmark split mixes pairs whose TF and gene both appear in training. Two harder tests (hESC, 5 folds) remove whole genes or whole TFs from training and from the graph:

| Held-out genes (5 folds × 5 seeds) | Within-TF | vs hero (p) |
| --- | --- | --- |
| Hero | 0.546 | |
| Neural PANDA | 0.562 | +0.021 (0.04) |
| Neural PANDA without co-expression edges | 0.545 | +0.003 (0.40) |
| Hero + Geneformer | 0.570 | +0.026 (0.004) |
| Neural PANDA + Geneformer | 0.577 | +0.004 vs hero + Geneformer (0.72) |

- A gene with no known regulators has no edges in the regulatory graph, so the model must describe it from other information. Geneformer, which learned gene context from millions of cells, is the best source.
- Neural PANDA needs both its co-expression and its PPI edges to help, and its gain mostly overlaps with Geneformer's.
- Within-TF on unseen genes stays near 0.57, against 0.75 on the normal split. Finding regulators of a gene with no known regulators remains largely unsolved.

**Held-out TFs.** Gene popularity alone reaches 0.724 within-TF. The best GNN reaches 0.690. Scoring a new TF by borrowing the targets of TFs it is co-expressed with (PANDA's "responsibility" step) reaches 0.701, and borrowing from its STRING partners 0.640. Popularity still wins.

## All four cell types

The hero beats the published scRegNet recipe within-TF on three of four cell types; hHEP is the exception, where the hero was never tuned.

| Within-TF AUROC, 5 seeds (diff vs that cell's hero) | hESC | hHEP | mESC | mHSC-E |
| --- | --- | --- | --- | --- |
| Hero | 0.747 | 0.760 | 0.824 | 0.762 |
| Neural PANDA, product decoder | 0.741 (−0.006) | 0.774 (+0.015) | 0.825 (+0.002) | 0.757 (−0.004) |
| Regulatory + PANDA graph | 0.730 (−0.017) | 0.755 (−0.005) | 0.813 (−0.010) | 0.760 (−0.002) |
| Neural PANDA, Tanimoto decoder | 0.718 (−0.029) | 0.775 (+0.015) | 0.762 (−0.062) | 0.720 (−0.041) |
| Published scRegNet recipe | 0.703 (−0.044) | 0.787 (+0.027) | 0.766 (−0.058) | 0.722 (−0.040) |
| Stacker, all features | 0.742 (−0.005) | 0.759 (−0.001) | 0.804 (−0.019) | 0.734 (−0.027) |

- No model beats the hero on every cell type; neural PANDA gains only on hHEP.
- The hero's settings were chosen on hESC and reused unchanged. On hHEP the published recipe is ahead (0.787 against 0.760, p = 0.08), most likely a tuning gap.
- The number of test TFs differs by cell type (33, 30, 85, 29), so p-values are not comparable across cell types.

## What it means biologically

The results say more about how regulation shows up in this data than about any one architecture.

- **Hub targets are real.** Some genes are bound by many TFs (popular targets). They are more likely to be bound by the next TF too, which is why popularity alone nearly matches scRegNet on pooled AUROC, and why teaching a model to distrust popular genes backfires.
- **Co-expression is relative, not direct.** In hESC a TF's mRNA level barely tracks its targets' (within-TF 0.50). What carries signal is whether a gene moves with the TF's other targets: a shared programme, not a one-to-one dose response. TF activity is set largely after transcription (protein level, partners, chromatin), so this fits.
- **PANDA's ingredients add little to a label-trained GNN.** Co-expression and PPI carry information, but every way of feeding them into the GNN tied with the hero or lowered within-TF.
- **Gene identity transfers across contexts.** Geneformer's pretrained gene knowledge is what rescues genes with no known regulators. Neural PANDA's label-free edges add a little on top.
- **New TFs remain the hard case.** A TF with no known targets cannot be placed from its co-expression or its protein partners better than by guessing popular genes. Binding specificity (motifs) or chromatin accessibility would be the missing input.

## Open items and next steps

- [ ] **Edge-fix checks.** Score training positives with and without their own edge; compare a fixed 70/30 message-passing/supervision split with the per-batch removal; audit validation edges, Geneformer and popularity counts for leaks. About 30 minutes of GPU.
- [ ] **Consolidated grid.** Encoder × decoder × edge split × Geneformer × PANDA × depth on four cell types, about 1,000 runs (5–8 hours), plus one master table.
- [ ] **Tune the hero per cell type,** starting with hHEP.
- [ ] **Try scGPT and UCE embeddings in place of Geneformer;** embedding code already exists in the repository.
- [ ] **Small known leak shared by all models:** 7 of 1,061 hESC test positives have their reverse (gene → TF) edge in the training graph. At 0.7% of test positives it should barely move any comparison, but it should be removed before a paper.
