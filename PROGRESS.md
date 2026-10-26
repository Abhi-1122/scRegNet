# In-depth analysis: why does PANDA help scRegNet?

********* Oct 4th ********

Branch: `in-depth-analysis` (cut from `summer26`). Dataset throughout: **hESC, TFs+500**
(910 genes, 410 TFs; 3,030 train-positive edges, 1,061 test-positive edges, the rest
negatives). Model: GCN + Geneformer, with the same hyperparameters as every `summer26` run.

I add to this file top to bottom as the work happens. Each section covers **what** I did,
**why**, **the result**, and **what we can infer**, both computationally and biologically.
New sections go at the bottom.

**Status:** All done. Section 7 at the bottom summarizes everything; start there if you want
the short version.

---

## 0. Starting point: how does scRegNet's graph differ from PANDA's graph?

This section answers the question you asked in chat. While writing it I read PANDA's
source code and found two places where my chat answers were imprecise. I correct both
below.

### 0.1 What scRegNet's graph layer does

**The input:**
- **Nodes:** all 910 genes.
- **Node features:** each gene's expression across all 758 hESC cells, standardized, one
  number per cell. The GNN sees expression: each gene starts as a 758-long vector.
- **Edges:** an undirected graph built from the **3,030 training-positive TF→target edges
  and nothing else** (`Adj_Generate` in `src/utils.py`). Two genes share an edge only if
  they form a known training pair. Edges carry no weights and no other evidence.

**The GCN step:** each layer replaces a gene's vector with an average of its own vector
and its graph neighbours' vectors, then applies a learned transformation. After 3 layers,
a gene's embedding summarizes the expression of every gene within 3 steps of it in the
training graph.

**After the GCN:**
- The model concatenates the GCN output with the Geneformer embedding.
- One small MLP turns that into a "TF embedding" and a "target embedding". The model
  applies the same MLP to both.
- One linear layer on `[TF embedding, target embedding]` produces the edge score
  (`decode()` with `type=MLP`).

That last step matters. A linear layer on a concatenation computes a sum, so the edge
score has the form

`score(TF, target) = (a term that depends only on the TF) + (a term that depends only on the target)`

Nothing in that formula lets the TF and the target interact. The model can learn "how
regulator-like is this TF" and "how target-like is this gene", and it adds the two. It has
no way to encode "this TF regulates this specific gene" beyond that sum. I treat this as a
hypothesis about the trained model's behaviour. **Item 4 tests it**: if I'm right, a
per-TF plus per-gene additive fit will reproduce the baseline model's scores with almost
no error.

### 0.2 What PANDA does (corrected)

PANDA takes three networks and iterates until they agree:
- **Prior (motif) network**, TF×gene: here, the 3,030 train-positive edges with weight 1.
- **Co-expression network**, gene×gene: the Pearson correlation of each gene with each
  other gene, over all 758 cells.
- **PPI network**, TF×TF: which TFs physically interact, from STRING.

On each iteration PANDA scores TF *i* → gene *j* as the average of two terms:

1. **Do your partners regulate it?** PANDA calls this *responsibility*. The score rises
   when the TFs that physically interact with TF *i* also target gene *j*.
2. **Does it behave like your other targets?** PANDA calls this *availability*. The score
   rises when gene *j*'s expression correlates with the genes TF *i* already targets.

PANDA then moves all three networks a step (size 0.1) toward each other and repeats until
the scores stop changing. The output is a dense 410×910 matrix of continuous scores.

**Correction 1.** In chat I said PANDA brings "the correlation between the TF and the
target". PANDA's co-expression term never compares the TF's own expression with the
target's. It asks whether the target co-expresses with the TF's *other known targets*:
guilt by association. This signal tends to hold up better than raw TF-target correlation,
because a TF's mRNA level tracks its protein activity poorly.

**Correction 2.** In chat I said the GNN "never sees co-expression". It sees expression as
node features. It can only compare two genes when the training graph already links them,
or links them through shared neighbours. Section 0.3 spells this out.

### 0.3 Side by side

| | scRegNet's graph layer (GCN) | PANDA |
|---|---|---|
| What the graph is | Binary, sparse: the 3,030 known train edges | Dense, weighted: a continuous score for each TF×gene pair |
| Edge weights | None (connected or not) | Yes, refined over iterations |
| Uses expression? | Yes, as each node's starting vector | Yes, as a full gene×gene correlation matrix |
| How it compares expression | Genes mix vectors only with graph neighbours | Each gene against each gene, then "is the target like the TF's other targets?" |
| Can relate two genes the training graph doesn't link? | Only through shared neighbours within 3 hops | Yes, co-expression covers all pairs |
| Uses TF-TF physical interaction? | No | Yes ("do your partners regulate it?") |
| Learned or fixed? | Learned end-to-end by gradient descent | Fixed algorithm, no training |
| What reaches the edge score | TF-only term + target-only term, added | One number for the specific (TF, target) pair |

### 0.4 What extra information PANDA brings

Three things. The first is computational; the other two are biological evidence.

1. **A pairwise number.** If Section 0.1 holds, the baseline's edge score contains no
   TF×target interaction term. In Experiment C the PANDA scalar is the **only** decoder
   input that depends on the specific pair. That may be the main reason it helps; Item 4
   tests this.
2. **Guilt-by-association co-expression** for every pair: is this gene co-expressed with
   the TF's other targets? The GCN gets a weak version of this, and only for genes within
   3 hops of each other in the training graph.
3. **TF cooperativity from physical interaction** (STRING PPI): do this TF's binding
   partners regulate this gene? No other part of scRegNet sees protein-protein interaction
   data. **The biology:** TFs often work in complexes. In hESC the core pluripotency factors
   OCT4 (POU5F1), SOX2 and NANOG bind each other and co-occupy many of the same enhancers,
   so a partner's targets make a reasonable prior for your own. Item 5 checks whether that
   holds in this dataset.

### 0.5 What the existing numbers hint at

| Setup | AUROC | AUPRC |
|---|---|---|
| Geneformer only, no GNN, no PANDA | 0.8774 | 0.6064 |
| Geneformer only + PANDA scalar (Exp D) | 0.8829 | 0.6175 |
| Full GNN + Geneformer, no PANDA (baseline) | 0.8874 | 0.6225 |
| Full GNN + Geneformer + PANDA scalar (Exp C, 5-seed mean) | 0.8899 | 0.6307 |

- PANDA adds more without the GNN (+0.55 / +1.11) than with it (+0.25 / +0.82). Some of
  PANDA's signal overlaps with what the GNN extracts; some doesn't.
- All four rows sit between 0.877 and 0.890. A model with no graph at all nearly matches
  the full one, which looks suspicious. Items 1 and 4 check whether "how regulator-like is
  this TF, how target-like is this gene" alone can solve most of this benchmark. If it can,
  you have to read every number above differently.

---

## 1. Score each signal on its own

**Script:** `analysis/item1_single_signals.py` (output: `analysis/out/item1.txt`).
Model scores come from `analysis/model_scores.py`, which reloads each saved checkpoint and
reproduces its reported test numbers exactly (baseline 0.8874/0.6225, Exp C 0.8902/0.6333).

### Why

Before asking why PANDA helps, we need to know how much each ingredient knows by itself.
If a trivial signal already scores 0.87, the gap between models (0.877 to 0.890) looks
much smaller.

### What I measured

For each test edge I computed one number per signal and scored it against the label,
with no training except where noted:

- **TF identity only:** the fraction of that TF's training pairs that are positive.
- **Target identity only:** how many training positives the target gene has (its degree).
- **Raw TF-target correlation:** Pearson r between the TF's and the target's expression.
- **Guilt by association:** PANDA's "does it behave like your other targets?" term,
  computed once without iterating.
- **PPI partners:** PANDA's "do your partners regulate it?" term, computed once: how many
  of the TF's STRING partners have this gene as a training target.
- **PANDA score:** the full iterated PANDA output.
- **Degree-only model:** logistic regression on just two numbers, the TF's positive rate
  and the target's degree. No expression, no graph learning, no foundation model.
- **Degree + PANDA:** the same, plus the PANDA score as a third number.
- **The trained models**, from their checkpoints.

I report two kinds of score:
- **AUROC / AUPRC**, as in all earlier results.
- **Within-TF AUROC**: I compute AUROC separately inside each TF's own candidate pairs,
  then average over the 33 TFs that have both positives and negatives in the test set.
  This answers the biological question: *for a given TF, can you tell its real targets from
  its non-targets?* It gives no credit for knowing which TFs have many targets.

### Result

| Signal | AUROC | AUPRC | Within-TF AUROC |
|---|---|---|---|
| TF identity only (TF's positive rate) | 0.844 | 0.482 | 0.500 |
| Target identity only (target's degree) | 0.642 | 0.210 | 0.707 |
| Raw corr(TF, target) | 0.457 | 0.135 | 0.491 |
| \|corr(TF, target)\| | 0.510 | 0.152 | 0.424 |
| Guilt by association (one step) | 0.521 | 0.140 | 0.523 |
| PPI partners that target it (one step) | 0.588 | 0.200 | 0.642 |
| PANDA score (full, iterated) | 0.744 | 0.357 | 0.664 |
| **Degree-only model (2 numbers, LR)** | **0.872** | **0.597** | **0.707** |
| **Degree + PANDA (3 numbers, LR)** | **0.880** | **0.612** | **0.724** |
| Model: Geneformer only (no GNN) | 0.877 | 0.606 | 0.703 |
| Model: GNN only (no Geneformer) | 0.875 | 0.592 | 0.652 |
| Model: full scRegNet (baseline) | 0.887 | 0.623 | 0.720 |
| Model: Geneformer + PANDA (Exp D) | 0.883 | 0.618 | 0.725 |
| Model: GNN + PANDA profile, no Geneformer (Exp B) | 0.880 | 0.610 | 0.712 |
| Model: full + PANDA scalar (Exp C) | 0.890 | 0.633 | 0.737 |

The test set has a 14.9% positive rate, so a random guess scores 0.5 AUROC and 0.149 AUPRC.

### Finding 1: most of the benchmark comes down to "which TF is it"

Knowing only the TF, with nothing about the target, gives 0.844 AUROC. The data shows why:

- Every edge in train and test comes from just 34 TFs. Each TF gets about 818 candidate
  pairs, which covers nearly the whole 910-gene panel.
- Those TFs' positive rates range from 0.1% (AR, TCF4, MLLT3) to **79% (TFAP2A)**.
  TFAP2A alone supplies 650 of the ~4,100 positives. JUND (45%), NANOG (38%), OTX2 (37%)
  and TEAD4 (35%) come next.
- So any model that learns "TFAP2A pairs are usually positive, AR pairs almost never" earns
  most of its AUROC without ever looking at the target.

**Biologically:** this ground truth comes from ChIP-seq. A TF's positive rate mostly
reflects how broadly it binds in its ChIP-seq experiment, which depends a lot on the
antibody, sequencing depth and peak-calling threshold. TFAP2A binding 79% of the panel says
more about that experiment than about TFAP2A regulating 79% of these genes. AUROC pooled
across TFs rewards learning this experimental property.

### Finding 2: two numbers get you 98% of the way

The degree-only model uses no expression data, no graph neural network and no foundation
model. It scores 0.872 AUROC, against 0.887 for full scRegNet. Everything scRegNet adds
beyond "this TF has many targets, this gene is a frequent target" amounts to **+0.015
AUROC and +0.026 AUPRC**.

### Finding 3: within a TF, the target's popularity does most of the work

Within-TF AUROC is the score that matters biologically. On it:

- **Target degree alone scores 0.707.** Some genes get bound by many TFs: RGPD2 is a
  target of 18 of the 34 TFs, and MIA3, RBPJ, HNRNPD, ZNF202 and MECOM of 14–15 each.
  When you rank a TF's candidate targets by "how many other TFs bind this gene", you already
  do well.
- **Geneformer only scores 0.703.** That is about the same as target degree, which suggests
  the Geneformer embedding mostly tells the model "is this a gene that many TFs bind".
- **GNN only scores 0.652**, below the degree baseline.
- **Full scRegNet scores 0.720**: +0.013 over target degree.
- **Degree + PANDA, a 3-number logistic regression, scores 0.724** and beats full
  scRegNet on this measure.
- **Exp C scores 0.737**: +0.017 over the baseline. That's the largest within-TF gain of any
  change we've made, so PANDA helps most when ranking a TF's own targets.

**Biologically:** genes bound by many TFs are a known phenomenon. ChIP-seq studies find
"highly occupied target" (HOT) regions: open, active promoters and enhancers where dozens
of unrelated TFs show binding peaks. These regions usually mark accessible, highly active
chromatin rather than specific regulation by each of those TFs. So a large part of
"target-likeness" may track which genes sit in open, busy chromatin.

*Update from Section 6:* I tested this. The support is weak. The most-bound quarter of genes has
a little higher expression than the least-bound quarter (mean 4.47 vs. 3.97 log2 units; expressed
in 76% vs. 69% of cells). But the correlation is small (Spearman +0.11 with expression level),
and GO finds no shared function among the most-bound genes. HOT regions may explain part of
the pattern, but I haven't shown it. To measure chromatin accessibility itself, we'd need
ATAC-seq data for these cells.

### Finding 4: raw co-expression carries no signal here

Raw TF-target correlation scores 0.457 AUROC, worse than chance, and 0.491 within-TF.
Absolute correlation scores 0.424 within-TF, meaning a TF's real targets correlate
*less* with it than its non-targets do. Guilt by association, computed once, barely clears
chance at 0.52.

**Biologically:** three things explain this.
- A TF's mRNA level is a weak proxy for its activity. Phosphorylation, nuclear
  localization and partner availability all control activity at the protein level.
- ChIP-seq binding does not equal functional regulation; many bound sites have no
  measurable effect on the gene.
- The 758 hESC cells span a differentiation time course from 0h to 96h. Correlation across
  that pool mixes changes driven by the time course with any regulatory link, which can
  mask or invert true relationships.

**Computationally:** used on its own, raw co-expression contains almost no signal, and neither
does the simple one-step guilt-by-association form.

*Update from Section 2:* at this point I concluded that co-expression couldn't explain PANDA's
gain, and I was wrong. Co-expression carries most of PANDA's extra information once two things
happen: the score compares the TF's targets against its non-targets, and a degree feature
handles gene popularity. Section 2, Finding 3 has the details.

### Finding 5: PPI is the informative part of PANDA

The one-step PPI term ("how many of the TF's binding partners target this gene") scores
0.642 within-TF, far above the co-expression terms. The full PANDA score reaches 0.664.

**Biologically:** this fits TF cooperativity. TFs that physically interact tend to bind the
same regulatory regions. In hESC, OCT4 (POU5F1), SOX2 and NANOG form a core complex and
co-occupy many enhancers, so if NANOG's partners bind a gene, NANOG probably does too.
Item 5 tests this assumption directly.

*Update from Section 2:* the PPI term is strong *on its own* mostly because it tracks gene
popularity. Once a model knows target degree, PPI adds little, and co-expression adds more.
See Section 2, Findings 1–2.

### What this means for the rest of the analysis

- The pooled AUROC numbers we've used throughout mostly measure two properties of the
  ChIP-seq ground truth: which TFs bind broadly, and which genes many TFs bind. Model
  differences of ±0.01 sit on top of a 0.872 floor that the degree-only model reaches
  without any expression data.
- From here on I report within-TF AUROC alongside pooled AUROC. It shows PANDA's
  contribution more clearly: +0.017 within-TF, versus +0.003 pooled.
- **Prediction for Item 2:** removing the co-expression input from PANDA will barely
  matter; removing the PPI input will remove most of the gain.

**Caveat on the model numbers:** `src/train.py` keeps the checkpoint from the epoch with the
best *test* AUROC, so the reported model scores carry a small optimistic bias. The degree
and LR baselines don't get that advantage, which makes Finding 2 a conservative comparison.

---

## 2. Break PANDA into its three inputs

**Scripts:** `analysis/item2_panda_variants.py` (builds the variants),
`analysis/item2_standalone.py`, `analysis/item2_bootstrap.py`, `analysis/item2_onestep.py`
(outputs in `analysis/out/item2_*.txt`), `analysis/train_variant.py` + `analysis/run_item2.sh`
(retraining).

### Why

PANDA mixes three inputs: the training edges (prior), co-expression and TF-TF PPI. Exp C
only tells us that the mix helps. To understand it, we need to know which input carries the
useful information. If raw co-expression alone gave the same gain, PANDA's machinery would be
unnecessary.

### How I built the variants

When you leave an input out of the netZooPy `Panda` class, it shrinks its own gene list, so
the variants would not be comparable. Instead I called PANDA's own normalization and update
loop directly on a fixed set of 910 genes and the same TFs as the original run. A missing input
becomes an identity matrix, which is what PANDA itself substitutes. To check the driver, I
rebuilt the original full PANDA matrix: it matches `src/build_panda.py`'s output
(maximum difference 0), and so does the out-of-fold matrix used for training edges.

Variants:
- **full:** prior + co-expression + PPI (the original).
- **no_coexpr:** prior + PPI.
- **no_ppi:** prior + co-expression.
- **prior_only:** PANDA run on the training edges alone.
- **raw_corr (control):** no PANDA at all, just the raw TF-target correlation, used as the
  same kind of scalar. It tests whether *any* extra pairwise number helps.

Every variant keeps the leakage fix: training edges get out-of-fold scores computed with
their own fold left out of the prior, using the same 5 folds as the original.

### Result A: each variant alone, and added to the degree-only model

"+ degree" means a logistic regression on the two degree numbers from Item 1 plus the variant's
score. The degree-only model scores 0.872 / 0.597 / within-TF 0.707.

| Variant | Alone: AUROC | AUPRC | Within-TF | + degree: AUROC | AUPRC | Within-TF |
|---|---|---|---|---|---|---|
| full | 0.744 | 0.357 | 0.664 | 0.880 | 0.612 | **0.724** |
| no_coexpr | 0.773 | 0.382 | 0.671 | 0.873 | 0.602 | 0.714 |
| no_ppi | 0.742 | 0.360 | 0.621 | 0.878 | 0.609 | 0.721 |
| prior_only | 0.806 | 0.490 | 0.632 | 0.870 | 0.596 | 0.704 |
| raw_corr (control) | 0.457 | 0.135 | 0.491 | 0.871 | 0.600 | 0.706 |

### Result B: are these differences real? (bootstrap over TFs)

The differences are small, so I resampled the 33 test TFs 5,000 times and recomputed the
within-TF AUROC difference each time (95% confidence interval in brackets):

| Comparison (within-TF AUROC) | Difference | 95% CI | TFs that improved |
|---|---|---|---|
| full PANDA + degree vs. degree | +0.017 | [+0.010, +0.026] | 27 / 33 |
| no_coexpr + degree vs. degree | +0.007 | [+0.004, +0.011] | 24 / 33 |
| no_ppi + degree vs. degree | +0.014 | [+0.007, +0.021] | 27 / 33 |
| prior_only + degree vs. degree | −0.003 | [−0.008, +0.002] | 12 / 33 |
| raw_corr + degree vs. degree | −0.000 | [−0.004, +0.004] | 16 / 33 |
| **full vs. no_coexpr (what co-expression adds)** | **+0.010** | **[+0.004, +0.017]** | 22 / 33 |
| **full vs. no_ppi (what PPI adds)** | **+0.003** | **[−0.002, +0.009]** | 17 / 33 |

### Finding 1: my prediction from Item 1 was wrong

I predicted PPI would carry PANDA's gain. Once you know node degree, **co-expression carries
it**. Removing co-expression cuts PANDA's gain over degree from +0.017 to +0.007, a
significant drop. Removing PPI lowers it by only 0.003, which is within noise.

### Finding 2: PPI looks strong alone because it is mostly target popularity

The one-step PPI term counts how many of the TF's binding partners target the gene. A gene
that many TFs target will also be targeted by many of any TF's partners. Within each TF, that
term correlates with the gene's degree at a median Spearman of **0.63**.

The PANDA variants show the same pattern. The table gives each variant's correlation with
target degree inside a TF (median over TFs):

| Variant | Correlation with target degree |
|---|---|
| no_coexpr | 0.72 |
| prior_only | 0.71 |
| full | 0.54 |
| no_ppi | 0.38 |

Without co-expression, PANDA mostly rediscovers target popularity, which the degree feature
already provides. Co-expression pulls the score away from popularity, toward information the
degree model lacks. That explains Result A: no_coexpr and prior_only score well alone but add
little next to degree, because the degree feature already gives the model what they measure.

**Prior-only PANDA adds nothing.** Run on the training edges alone, PANDA spreads those edges
around the network and recovers node popularity, which the degree model already knows.

### Finding 3: co-expression helps, but only when you compare it the right way

Raw co-expression scored worse than chance in Item 1, yet PANDA's version adds real
information. So the way PANDA processes co-expression must matter. I tested which part of
that processing does the work, in two steps.

**Iteration adds almost nothing.** One step of PANDA's co-expression term (its "availability"),
computed with PANDA's normalization and no iteration, plus degree, gives within-TF **0.723**.
The fully iterated PANDA gives 0.724.

**Two details of the comparison matter.** I scored "does gene *j* co-express with this TF's
known targets?" four ways, each added to the degree model:

| How the co-expression score is computed | Alone, within-TF | + degree, within-TF |
|---|---|---|
| Raw correlation, mean over the TF's targets | 0.523 | 0.717 |
| Raw correlation, mean over targets **minus** mean over non-targets | 0.565 | 0.723 |
| PANDA-normalized correlation, mean over the TF's targets | 0.546 | 0.723 |
| **PANDA-normalized correlation, targets minus non-targets** | **0.574** | **0.728** |
| (full iterated PANDA, for comparison) | 0.664 | 0.724 |

Each of the two changes adds about +0.006, and together they edge past the full PANDA
algorithm on this measure (0.728 vs. 0.724). So you can reproduce all the extra information
PANDA gives scRegNet with one line of arithmetic: **z-score the correlation matrix, then ask whether gene *j* correlates with
the TF's known targets more than with the TF's non-targets.**

**Why both details matter, biologically.** These 758 cells span a differentiation time course,
so thousands of genes rise or fall together with time. Some genes (cell-cycle genes, highly
expressed housekeeping genes, time-course-driven genes) correlate with almost everything.
A raw "mean correlation with the TF's targets" rewards those genes for every TF.

- PANDA's z-scoring measures each correlation relative to that gene's usual correlation
  level, so a gene that correlates with everything stops looking special.
- The contrast against non-targets asks "does this gene keep the company of *this TF's*
  targets in particular?". A gene co-expressed with lots of genes scores near zero, because
  it correlates with the non-targets too.

The method is guilt by association, with the non-targets as a control group. The same logic
explains why raw co-expression
is useless alone yet helps next to degree: alone, the time-course trend and gene popularity
swamp it. Once the degree feature handles popularity, the co-expression score can separate
genes of similar popularity.

### Result C: retraining Exp C with each variant

**Scripts:** `analysis/train_variant.py`, `analysis/run_item2*.sh`,
`analysis/item2_retrain_summary.py` (output: `analysis/out/item2_retrain_summary.txt`),
`analysis/item2_why_training_underuses.py`, `analysis/item2_symmetric_oof.py`.

Results A and B used logistic regressions. Here I retrained Exp C itself with each PANDA
variant: same architecture, same hyperparameters, only the PANDA matrix changes, 3 seeds each.
I also trained the baseline (no PANDA) on 2 extra seeds, so every row has a seed-to-seed spread
to compare against.

**First round: the original leakage fix.**

| Retrained model (3 seeds) | AUROC | AUPRC | Within-TF AUROC | Within-TF gain over baseline |
|---|---|---|---|---|
| Baseline, no PANDA | 0.8870 ± 0.0009 | 0.6245 ± 0.0018 | 0.7215 ± 0.0017 | — |
| full (Exp C) | 0.8899 ± 0.0006 | 0.6320 ± 0.0012 | 0.7350 ± 0.0020 | +0.014 |
| no_ppi (co-expression kept) | 0.8886 ± 0.0006 | 0.6288 ± 0.0013 | 0.7302 ± 0.0016 | +0.009 |
| no_coexpr (PPI kept) | 0.8874 ± 0.0006 | 0.6275 ± 0.0017 | 0.7255 ± 0.0018 | +0.004 |
| raw_corr (control) | 0.8864 ± 0.0004 | 0.6252 ± 0.0010 | 0.7242 ± 0.0011 | +0.003 |
| prior_only | 0.8866 ± 0.0005 | 0.6242 ± 0.0015 | 0.7228 ± 0.0011 | +0.001 |

The trained network agrees with Results A and B. Co-expression carries most of the gain, PPI
adds a smaller part on top, and raw correlation and prior-only PANDA sit within seed noise of
the baseline.

**A problem in how the original leakage fix scores training edges.** The retrained models got
less out of each variant than the post-hoc stacking in Section 4 did. I traced this to
`src/build_panda.py`:
- PANDA's prior *is* the training positives, so scoring a training positive with the full
  prior would leak its own label. The fix re-scores each training **positive** out-of-fold,
  with its fold removed from the prior.
- Training **negatives** keep their full-prior score.
- So during training, positives and negatives are scored under different conditions. At test
  time, both use the full prior.

That mismatch weakens the feature on the rows the network learns from. The table shows how well
the PANDA scalar alone separates targets from non-targets inside a TF:

| Variant | On training rows (original fix) | On training rows (symmetric fix) | On test rows |
|---|---|---|---|
| full | 0.580 | 0.648 | 0.664 |
| no_coexpr | 0.563 | 0.654 | 0.671 |
| no_ppi | **0.510** (chance) | 0.589 | 0.621 |

The **symmetric fix** scores *every* training row, positive and negative, from a PANDA run whose
prior excludes that row's fold. It stays leakage-free: no row's own label, nor any label from its
fold, enters its score. With this fix, training rows look like test rows.

**Second round: the symmetric fix.**

| Retrained model (3 seeds) | AUROC | AUPRC | Within-TF AUROC | Within-TF gain over baseline |
|---|---|---|---|---|
| Baseline, no PANDA | 0.8870 ± 0.0009 | 0.6245 ± 0.0018 | 0.7215 ± 0.0017 | — |
| full (Exp C), original fix | 0.8899 ± 0.0006 | 0.6320 ± 0.0012 | 0.7350 ± 0.0020 | +0.014 |
| **full, symmetric fix** | **0.8915 ± 0.0004** | **0.6337 ± 0.0009** | **0.7471 ± 0.0009** | **+0.026** |
| no_ppi, symmetric fix | 0.8904 ± 0.0005 | 0.6301 ± 0.0011 | 0.7425 ± 0.0011 | +0.021 |
| no_coexpr, symmetric fix | 0.8892 ± 0.0006 | 0.6346 ± 0.0012 | 0.7357 ± 0.0007 | +0.014 |

The symmetric fix also changes how the network uses PANDA:

| | Original fix | Symmetric fix |
|---|---|---|
| Decoder weight on the PANDA scalar | 0.27 | 0.68 |
| PANDA's share of each TF's score variation (median) | 0.5% | 2.7% |
| Gene-ranking correlation between TFs (min / median) | 0.42 / 0.87 | 0.14 / 0.81 |

### Finding 4: the original leakage fix hid about half of PANDA's benefit

- With the symmetric fix, PANDA's within-TF gain nearly doubles: +0.014 → **+0.026**. Pooled
  AUROC rises to 0.8915, the best of any model in this project.
- The trained model now beats the post-hoc ensemble from Section 4 (within-TF 0.747 vs. 0.738).
  When training rows get the same kind of PANDA score as test rows, the network gives it more
  weight (0.27 → 0.68) and ranks genes differently for different TFs.
- Co-expression still matters most. Removing it costs 0.011 within-TF; removing PPI costs 0.005.
  Both inputs now contribute beyond seed noise.

**Computationally:** this lesson reaches beyond PANDA. When you build an
out-of-fold feature to avoid leakage, give every training row (positives *and* negatives) the
same out-of-fold treatment. Otherwise the feature behaves differently in training than at test
time, and the model learns to give it too little weight. `src/build_panda.py` should switch to the symmetric version.

**Biologically:** with the symmetric fix, the model's rankings for different TFs move further
apart. That is closer to real cells, where different TFs regulate different programs (Section 6).
The model is still additive apart from the PANDA term, so PANDA remains its only source of
TF-specific knowledge.

---

## 3. Where does PANDA's gain land?

**Script:** `analysis/item3_where_gain_lands.py` (output: `analysis/out/item3.txt`).

### Why

Section 0 argued that the GCN can only relate two genes through the training graph, so PANDA
should help most on pairs the GCN can't reach. I tested this by comparing the baseline and
Exp C edge by edge.

### What I measured

For each test edge I took its position within its own TF's list (percentile: 0 = bottom,
1 = top) under the baseline and under Exp C. A real target that moves up counts as a good
move; so does a non-target that moves down. I then grouped edges by:
- graph distance between the TF and the target in the GCN's training graph
- how many training edges the target has (target degree)
- how many training targets the TF has
- how many of the TF's STRING partners target the gene
- the co-expression contrast score from Section 2 (does the gene co-express with the TF's
  targets more than with its non-targets?)

Within each group I also computed the within-TF AUROC for both models.

### Finding 1: "reachable in the graph" doesn't separate anything here

| TF → target distance in the training graph | Test pairs |
|---|---|
| 1 hop | 30 |
| 2 hops | 4,630 |
| 3 hops | 1,444 |
| 4–5 hops | 74 |
| Unreachable | 964 |

Nearly every test pair sits 2 or 3 hops apart. TFAP2A alone connects to most of the panel, so
the graph is "small world": through TFAP2A, almost any gene reaches almost any other in two
steps. The 964 unreachable pairs are exactly the pairs whose target has no training edge at
all. My Section 0 hypothesis (PANDA helps where the GCN can't reach) has nothing to work with,
because the GCN can reach nearly everything. Section 4 shows a deeper reason the hypothesis
fails: the baseline's ranking doesn't depend on the TF at all.

### Finding 2: PANDA helps most on the genes the baseline handles worst

| Target's training degree | Pos | Neg | Within-TF AUROC: baseline → Exp C |
|---|---|---|---|
| 0 (no training edges) | 26 | 938 | 0.788 → **0.679** |
| 1–2 | 223 | 1,716 | 0.642 → 0.649 |
| 3–5 | 468 | 2,258 | **0.528 → 0.571** |
| 6+ | 326 | 1,106 | 0.640 → 0.677 |

- On moderately popular targets (3–5 training edges), the baseline scores **0.528, barely
  above a coin flip.** These genes are real targets for some TFs and not for others. A model
  that gives each gene one score for every TF can't separate those cases. PANDA's
  TF-specific score helps most here (+0.043).
- On targets with no training edges, Exp C does **worse** (0.788 → 0.679). For these genes
  PANDA's prior is empty and no partner TF targets them, so their PANDA score rests on
  co-expression alone. My guess, which I haven't tested: this weak signal reorders genes the baseline (through Geneformer) had ranked well. This stratum has only 26
  positives over 10 TFs, so the drop itself is uncertain.

### Finding 3: gains concentrate where a partner TF or co-expression vouches for the edge

| Group | Within-TF AUROC: baseline → Exp C |
|---|---|
| 0 STRING partners target the gene | 0.728 → 0.738 (+0.010) |
| 1 partner targets it | 0.620 → 0.631 (+0.011) |
| 2+ partners target it | 0.592 → **0.634 (+0.042)** |
| Co-expression contrast below −0.02 | 0.739 → 0.750 |
| Co-expression contrast between −0.02 and +0.02 | 0.726 → **0.784 (+0.057)** |
| Co-expression contrast above +0.02 | 0.692 → 0.706 |

Real targets with a high co-expression contrast move up the list (average +0.032 percentile),
and non-targets with a low contrast move down (+0.023). PANDA pushes edges in the direction
the co-expression evidence points.

Read the PPI rows with care. Genes that 2+ partners target are mostly popular genes, the same
group where the baseline does worst (Finding 2). Section 2 showed that, averaged over all
TFs, PPI adds little once you know degree. So this row mostly shows *where* the baseline
needed help, and says less about whether PPI supplied that help.

### Finding 4: which TFs gain

Showing TFs with at least 8 test positives (TFs with 1 positive, such as TCF4, DNMT3B, HAND2
and MLLT3, can swing by a large amount on a single edge, so they tell us nothing):

| TF | Test positives | Within-TF AUROC: baseline → Exp C |
|---|---|---|
| **EZH2** | 27 | **0.478 → 0.592 (+0.114)** |
| POU5F1 (OCT4) | 21 | 0.749 → 0.795 (+0.046) |
| TRIM28 | 18 | 0.728 → 0.774 (+0.045) |
| FOXH1 | 25 | 0.701 → 0.736 (+0.036) |
| SOX2 | 59 | 0.760 → 0.783 (+0.023) |
| NANOG | 80 | 0.793 → 0.798 (+0.005) |
| TFAP2A | 167 | 0.835 → 0.843 (+0.009) |
| SMAD2 | 36 | 0.828 → 0.819 (−0.009) |
| KLF5 | 10 | 0.761 → 0.746 (−0.015) |
| ZFP42 | 8 | 0.673 → 0.644 (−0.029) |

**EZH2 tells the clearest biological story.** The baseline ranks EZH2's real targets *below*
its non-targets (0.478, worse than chance). EZH2 is the enzyme in the Polycomb complex PRC2.
In hESC it silences developmental genes that are poised for later activation (the "bivalent"
genes). Those targets are the opposite of the popular, open, highly active genes that many
TFs bind (the HOT regions from Item 1). A model that gives every TF the same "popular genes
first" ranking will rank EZH2's silent targets low, and it does. PANDA's TF-specific score
moves EZH2 to 0.592, which is better but still weak.

**POU5F1, TRIM28 and SOX2 fit cooperativity.** OCT4 (POU5F1) and SOX2 form a complex and
co-bind enhancers with NANOG. TRIM28 (KAP1) works with large families of zinc-finger TFs.
These TFs have many STRING partners (33, 23 and 40), and each gains (+0.046, +0.045 and
+0.023).

**The hub TFs move little.** For TFAP2A, JUND, OTX2 and TEAD4, much of the target list overlaps
with the generally popular genes, so the global ranking already suits them.

Across all TFs, the size of the gain has a weak correlation with the TF's number of training
targets (Spearman −0.14) and with its number of STRING partners (+0.13), so neither property
predicts which TFs gain.

---

## 4. How much of PANDA's signal does the baseline already have?

**Scripts:** analysis inline in `analysis/out/item4_additivity.txt`,
`analysis/item4_redundancy.py`.

### Why

The Exp C vs. Exp D numbers suggested partial overlap between PANDA and the GNN. I wanted
to measure it, and to test the Section 0 claim that the baseline's decoder can't represent
TF-specific predictions.

### Finding 1: baseline scRegNet ranks genes in the same order for every TF

I scored all 34 TFs × 910 genes with each trained model. Then I checked two things: whether a
"TF number + gene number" additive fit reproduces the scores, and whether two different TFs
get their genes ranked in the same order.

| Model | Additive fit R² | Gene-ranking correlation between TFs (min / median) |
|---|---|---|
| Baseline scRegNet | **1.000000** | **1.0000 / 1.0000** |
| Geneformer only | 1.000000 | 1.0000 / 1.0000 |
| GNN only | 1.000000 | 1.0000 / 1.0000 |
| Exp C (+ PANDA scalar) | 0.932 | 0.416 / 0.867 |

The baseline gives every gene one "target-likeness" score and every TF one "regulator
activity" score, and adds them. For TFAP2A, NANOG and EZH2 alike, it ranks the 910 genes in
exactly the same order. It cannot predict that NANOG's targets differ from EZH2's.

**Why:** the decoder applies one linear layer to `[TF embedding, target embedding]`, and the
TF and target embeddings are each computed from that node alone. A linear layer on a
concatenation is a sum, so no TF×target interaction can arise. `type=MLP` is the default in
`src/args.py`, and the Optuna search never varied it (it isn't in `src/optuna/search_space.py`).
So this decoder came in with the defaults, and no tuning run ever tested an alternative. The
code also offers `dot` and `cosine` decoders, which multiply the TF and target embeddings
together and can express interaction. Item 1 shows why no one noticed the problem: pooled AUROC on this benchmark
mostly rewards "which TF is it" and "which gene is popular", and an additive decoder captures
both cheaply.

**What this explains:**
- Geneformer-only matches target degree within TF (Item 1): it learned one global gene ranking.
- The baseline's weakness on moderately popular genes (Section 3, Finding 2): one ranking
  can't be right for TFs that disagree about the same gene.
- Why "reachable within 3 hops" didn't matter (Section 3, Finding 1): the baseline's score
  never depends on the relationship between the specific TF and gene.

**The ceiling for this kind of model:** I built the best possible single gene ordering from the
test labels themselves (cheating, as an upper bound). Even that scores within-TF **0.843**. The
baseline reaches 0.720. A model that ranks genes the same way for every TF can't exceed 0.843
on this test set, no matter how well you train it.

### Finding 2: PANDA is almost entirely TF-specific information

- Only **7%** of the PANDA score's within-TF variation is a single gene effect shared by all
  TFs. The other **93%** differs from TF to TF.
- That makes PANDA the **only** TF-specific input in Exp C. Every other part of the model
  produces the same ranking for every TF.

### Finding 3: the model gives PANDA little weight

- Exp C's decoder puts weight 0.28 (in logit units per standard deviation of PANDA) on the
  PANDA scalar.
- Within a TF, the PANDA term accounts for a median of **0.5%** of the variation in Exp C's
  score. The node terms still dominate; PANDA nudges the order among genes with similar
  global scores. That nudge moves within-TF AUROC from 0.720 to 0.737.

### Finding 4: PANDA's information is mostly new to the model

Within each TF, ranking edges by each score:
- label vs. PANDA: correlation 0.181
- label vs. baseline: 0.215
- PANDA vs. baseline: 0.311 (the overlap, mostly gene popularity)
- **label vs. PANDA after removing what the baseline explains: 0.123**

PANDA keeps about two-thirds of its predictive correlation after controlling for the
baseline. The overlap that the Exp C vs. Exp D gap hinted at is real but small.

### Finding 5: an ensemble gets the same gain as Exp C

I added the PANDA score to the baseline's output with a two-feature logistic regression,
cross-validated across TFs so each TF gets predictions from a model that never saw it:

| Model | AUROC | AUPRC | Within-TF AUROC |
|---|---|---|---|
| Baseline | 0.8874 | 0.6225 | 0.7202 |
| Baseline score + PANDA score, combined afterward | 0.8899 | 0.6290 | 0.7380 |
| Exp C (PANDA fed into the network) | 0.8902 | 0.6333 | 0.7374 |

Combining the two scores after the fact matches Exp C. Feeding PANDA into the network did not
let the model learn anything about how PANDA interacts with the expression or Geneformer
features. Exp C works as an ensemble of two independent predictors: a TF-agnostic gene
ranking, and PANDA's TF-specific co-expression score.

### What this means

- **Computationally:** scRegNet as configured cannot model TF-specific regulation; the
  decoder design rules it out. PANDA helps because it smuggles in the only TF-specific
  number, and you get the same benefit by adding it afterward. The most promising next change
  is the decoder: a `dot`, `cosine` or bilinear decoder (or adding a `TF × target` product term)
  would let the model learn TF-specific targets itself. Such a model has no 0.843 ceiling.
- **Biologically:** the baseline predicts "which genes are targets in general", meaning
  open, active, heavily bound genes. Regulation is specific: EZH2 silences developmental
  genes, OCT4 and SOX2 co-bind pluripotency enhancers. Only Exp C's PANDA term carries any of
  that specificity, and so far it carries only a little.

---

## 5. Test PANDA's biological assumption directly

**Script:** `analysis/item5_cooperativity.py` (output: `analysis/out/item5.txt`). This item
studies the data itself, so it uses all labelled positives from train and test together.

### Why

Two earlier choices rest on biological assumptions we never tested:
- PANDA's PPI term assumes **cooperativity**: TFs that physically interact regulate the same
  genes.
- PPI Methods 1 and 2 on `summer26` passed messages over the STRING gene-gene network. That
  only helps if **genes linked in STRING share regulators**. Graph people call this
  homophily. Both methods made scores slightly worse.

I measured both properties in the data.

### Test A: do TFs that interact share targets?

For every pair of the 34 benchmark TFs, I counted how many targets they share and compared
that with the overlap you'd expect by chance given how many targets each TF has. A TF with 280
targets and a TF with 200 targets will share some targets by accident. Test A asks whether
they share more than that. I report the ratio of observed to expected overlap.

| TF pairs | Number of pairs | Typical (median) overlap vs. chance |
|---|---|---|
| Interacting in STRING | 114 | **1.56×** |
| Not interacting | 430 | 1.27× |

The difference is significant (Mann-Whitney, p = 4×10⁻⁵). Interacting TFs share more targets.

The most enriched interacting pairs are well-known partners in development:

| Pair | Shared targets (observed vs. expected) | What the pair is |
|---|---|---|
| CTNNB1–LEF1 | 17 vs. 2.5 | β-catenin and LEF1: the transcription complex that carries out Wnt signalling |
| CTNNB1–POU5F1 | 30 vs. 6.4 | β-catenin binds OCT4 in embryonic stem cells |
| LEF1–POU5F1 | 12 vs. 2.4 | the same Wnt–pluripotency link, from the LEF1 side |
| FOXH1–SMAD2 | 39 vs. 14.1 | FOXH1 is the DNA-binding partner of SMAD2 in Nodal/Activin signalling, the pathway that drives hESC toward definitive endoderm |
| GATA4–SOX17, SOX17–GATA6 | 4 vs. 0.9, 2 vs. 0.4 | endoderm regulators that work together (small counts) |
| GATA2–GATA3 | 8 vs. 0.6 | paralogous GATA factors (small counts) |

The core pluripotency trio also co-regulates, as expected:

| Pair | Shared targets (observed vs. expected) | Ratio |
|---|---|---|
| SOX2–POU5F1 (OCT4) | 46 vs. 20.2 | 2.3× |
| NANOG–POU5F1 | 54 vs. 26.7 | 2.0× |
| SOX2–NANOG | 120 vs. 77.0 | 1.6× |

### Finding 1: cooperativity is real, and most of it is non-specific

- Interacting TFs share more targets (1.56×), so PANDA's assumption holds.
- **72% of all TF pairs**, interacting or not, share more targets than chance. Most TFs in this
  ChIP-seq ground truth pile onto the same popular genes (the HOT regions from Item 1).

The specific part of cooperativity, interacting pairs over non-interacting pairs (1.56× vs.
1.27×), is real but modest. The bulk is the general pile-up onto popular genes, which the degree
model already captures. This explains Item 2's result: the PPI term looks strong alone, adds
little next to degree, and helps specific TFs. In Item 3, POU5F1 (OCT4) gained +0.046. Its
partners SOX2 and NANOG share its targets at 2–2.3× chance.

How confident STRING is about an interaction barely predicts how many targets the pair shares
(Spearman +0.17). STRING's score measures evidence for a protein interaction. It doesn't
measure whether two TFs bind DNA together.

### Test B: do genes linked in a graph share regulators?

For each gene I wrote down which of the 34 TFs regulate it, as a 34-long 0/1 vector. Then for
pairs of genes linked in a graph, I measured how similar those vectors are (cosine similarity).
Popular genes share regulators with everything, so I compared against random gene pairs
matched on how many regulators each gene has.

| Graph linking the genes | Gene pairs | Regulator similarity | Matched random pairs | Excess |
|---|---|---|---|---|
| STRING (score ≥ 0.7, what PPI Methods 1/2 used) | 2,256 | 0.421 | 0.379 | **+0.042** |
| Top-10 co-expression neighbours (raw correlation) | 6,418 | 0.355 | 0.357 | −0.002 |
| Top-10 co-expression neighbours (z-scored correlation) | 5,962 | 0.361 | 0.361 | 0.000 |

### Finding 2: gene-gene graphs carry almost no regulator information

- **STRING:** genes whose proteins interact share regulators only a little more than
  matched random genes (+0.042 on a baseline of 0.379, about 11% more).
- **Co-expression neighbours:** genes with the most similar expression share **no more**
  regulators than random genes of the same popularity.

**Why PPI Methods 1 and 2 hurt:** a GNN over STRING averages each gene's features with its
STRING neighbours. STRING neighbours barely share regulators, so the averaging mostly blurs
genes together. Section 4 also showed that whatever the GNN learns can only change
one TF-agnostic gene ranking.

**The biology:** protein-protein interaction between *target* genes (two proteins in the same
complex or pathway) and co-regulation of those genes are different relationships. TF-TF
interaction concerns the regulators working together on DNA, which ties directly to shared
targets. That's why the TF-TF part of STRING showed a real effect in Test A, and the gene-gene
part barely did in Test B.

### Finding 3: co-expression works in aggregate, not pair by pair

This seems to contradict Section 2, where co-expression carried PANDA's information. The two
tests measure different things:
- **Test B** asks whether one gene's 10 closest co-expression partners share its regulators.
  They don't, beyond popularity.
- **Section 2's useful score** compares a gene against a TF's *entire* target set: hundreds
  of genes, contrasted with the TF's non-targets. Each single correlation is weak and noisy,
  but averaging over hundreds of targets lets a weak signal accumulate.

A GNN that mixes each gene with a handful of neighbours can't do this aggregation. PANDA's
co-expression term does it by construction. **In computational terms:** the useful
co-expression signal comes from comparing a gene against a TF's whole program, and
nearest-neighbour graphs discard that comparison.

---

## 6. Compare the networks each method predicts

**Script:** `analysis/item6_networks.py` (output: `analysis/out/item6.txt`). GO enrichment uses
the GO Biological Process 2023 library (via `gseapy`), with the **910-gene panel as the
background**. The panel is 45% TFs, so using all human genes as the background would make every
list look enriched for "regulation of transcription".

### Why

Sections 1–5 compared scores. A biologist gets a network from each method, so here I compare
the networks: how they differ in shape, how much they overlap, and whether their predicted
targets make biological sense.

### How I built each network

Each method's network is its **top 1,061 test pairs**, the same number as the real test edges.
That way every network has the same size as the truth, and precision means the fraction of
predicted edges that are real.

### Result: network shape

| Network | Precision | TFs with any predicted edge | Target-set overlap between TFs | Edges per TF, correlation with truth |
|---|---|---|---|---|
| Real test edges | 1.000 | 33 | 0.024 | 1.000 |
| Baseline scRegNet | 0.576 | 14 | 0.088 | 0.881 |
| Exp C (+ PANDA) | 0.580 | 14 | 0.086 | 0.897 |
| Degree-only LR | 0.567 | 12 | 0.083 | 0.861 |
| PANDA alone | 0.403 | 28 | 0.031 | 0.796 |
| Co-expression contrast alone | 0.099 | 20 | 0.033 | −0.346 |

"Target-set overlap between TFs" is the average Jaccard similarity between the predicted target
lists of two different TFs. Low means each TF gets its own targets.

### Finding 1: the model networks only cover 14 of 33 TFs

The real network has edges for 33 TFs. Baseline scRegNet and Exp C put all 1,061 predicted edges
on 14 TFs, the broad binders such as TFAP2A, JUND, NANOG and OTX2. The other 19 TFs get no
predicted edge. Section 1 explains why: the TF's base rate dominates the score, so a
narrow-binding TF's best candidate never outranks a broad-binding TF's average candidate.

### Finding 2: the models give different TFs the same targets

Real TFs have distinct target lists: two TFs share 2.4% of their targets on average.
The baseline's predicted lists overlap 8.8%, **3.7× more than reality**. This is Section 4's
"same ranking for every TF" seen at the network level: each TF gets the top of the same
popularity list. PANDA's network has realistic specificity (3.1%) and covers 28 TFs, but only
40% of its edges are real.

### Finding 3: scRegNet's network is mostly the degree network

| Pair of networks | Edge overlap (Jaccard) |
|---|---|
| Baseline vs. degree-only LR | **0.72** |
| Baseline vs. Exp C | 0.85 |
| Baseline vs. PANDA alone | 0.31 |
| Baseline vs. real edges | 0.40 |

Full scRegNet and the two-number degree model share most of their predicted edges (Jaccard 0.72). Exp C swaps 84
of the baseline's 1,061 edges: 39 of the added edges are real and 35 of the removed ones were
real, a net gain of **4 correct edges**. Most added edges belong to EOMES, NANOG, OTX2,
FOXP1 and TEAD4.

Co-expression contrast alone gets 9.9% precision, below the 14.9% base rate, and its per-TF edge
counts anti-correlate with the truth. On its own it can't build a network. It helps only
next to degree, as a tie-breaker between genes of similar popularity (Section 2).

### Result: biology of the predicted targets (GO enrichment)

**EZH2's real targets form a developmental program.** Of EZH2's 104 known targets in the panel:

| GO term | Genes in EZH2's targets / in panel | Adjusted p |
|---|---|---|
| Endoderm development | 6 / 7 | 0.004 |
| Aortic valve development | 8 / 13 | 0.004 |
| Aortic valve morphogenesis | 8 / 12 | 0.004 |
| Regulation of ERK1 and ERK2 cascade | 11 / 24 | 0.004 |
| Heart development | 12 / 31 | 0.011 |

This fits Polycomb biology. In embryonic stem cells, PRC2 (of which EZH2 is the catalytic
subunit) holds developmental regulators silent but ready to switch on later.

**Neither model's EZH2 predictions show that program.**

| Source of EZH2's top 60 new predicted targets | Held-out EZH2 targets recovered | Top GO terms |
|---|---|---|
| Baseline | 4 / 27 | generic (organic substance response, protein localization); nothing significant |
| Exp C | 4 / 27 | generic (cell migration, GTPase activity); nothing significant |
| PANDA alone | — | developmental (anterior/posterior patterning, heart development, left/right asymmetry); not significant (adjusted p 0.33) |

PANDA leans toward the right biology, but Exp C gives PANDA little weight (Section 4: 0.5% of the
within-TF score variation), so Exp C's EZH2 list differs little from the baseline's.

**The baseline's generic "target-likeness" has no shared function.** The baseline's 100 most
target-like genes, its 100 least target-like genes and the 100 genes bound by the most TFs all
show no significant GO enrichment (best adjusted p ≈ 0.4). Popularity in this ground truth
doesn't correspond to a biological program.

### What this means

- **Computationally:** scRegNet's predicted network is mostly "popular genes × broad-binding
  TFs". Exp C changes about 8% of edges, for a net gain of 4. PANDA alone builds
  a network with the right *shape*, TF-specific and covering most TFs, but with lower precision.
- **Biologically:** EZH2's real targets share a coherent developmental function and the models'
  predictions don't. The pooled AUROC hides this gap: on this benchmark a model can score
  0.887 and still predict nearly the same targets for a Polycomb repressor and a pluripotency
  activator.

---

## 7. Summary: what we now know

### The three questions you asked

**1. How do scRegNet's graph and PANDA's graph differ?**
- scRegNet's GCN runs on a binary graph made of the 3,030 training edges. It mixes each gene's
  expression vector with its neighbours', then its decoder adds a TF-only term to a gene-only
  term. As built, it ranks the 910 genes **in exactly the same order for every TF** (Section 4).
- PANDA builds a dense, continuous TF×gene score from three inputs (training edges,
  co-expression, TF-TF PPI). That score differs from TF to TF: 93% of its within-TF variation is
  TF-specific (Section 4).

**2. Why does adding PANDA improve the score?**
- PANDA is the **only TF-specific number** that reaches scRegNet's decoder. Everything
  else in the model produces one TF-agnostic gene ranking.
- Adding PANDA's score to the baseline's output afterward gives the same gain as Exp C
  (Section 4), so with the original setup Exp C works as an ensemble.
- With the symmetric fix to how training edges get their PANDA score, the gain nearly doubles
  and the model beats the ensemble (Section 2, Result C).

**3. What extra information does PANDA bring?**
- **Mostly co-expression, as a contrast score:** "does this gene correlate with this TF's
  known targets more than with its non-targets, after z-scoring correlations?" Raw TF-target
  correlation carries nothing (Sections 1, 2). This contrast score is weak alone and useful next
  to gene popularity, because popularity and the time-course trend otherwise swamp it.
- **Some TF cooperativity:** TFs that physically interact share more targets (β-catenin–LEF1,
  FOXH1–SMAD2, OCT4–SOX2; Section 5). But most of what the PPI term measures duplicates gene
  popularity, so it adds less than co-expression.
- **PANDA's iteration adds nothing:** one normalized step gives the same information as the converged
  result (Section 2).

### Headline numbers (hESC, TFs+500, 3 seeds each)

| Model | AUROC | AUPRC | Within-TF AUROC |
|---|---|---|---|
| Degree-only (2 numbers, no deep learning) | 0.8723 | 0.5970 | 0.7068 |
| Baseline scRegNet | 0.8870 ± 0.0009 | 0.6245 ± 0.0018 | 0.7215 ± 0.0017 |
| Exp C, original leakage fix | 0.8899 ± 0.0006 | 0.6320 ± 0.0012 | 0.7350 ± 0.0020 |
| **Exp C, symmetric leakage fix** | **0.8915 ± 0.0004** | **0.6337 ± 0.0009** | **0.7471 ± 0.0009** |
| Ceiling for any model that ranks genes the same for every TF | — | — | 0.843 |

### Things this analysis found that matter beyond PANDA

1. **The benchmark mostly measures ChIP-seq properties.** "Which TF binds broadly" alone gives
   0.844 AUROC; adding "which gene many TFs bind" gives 0.872. Report within-TF AUROC alongside
   pooled AUROC from now on, since it tests whether a model can tell one TF's targets from its
   non-targets.
2. **The decoder can't express TF-specific regulation.** `type=MLP` is the untuned default and
   makes the score additive. Trying a `dot`, `cosine` or bilinear decoder, or adding a TF×target
   product term, is the most direct next experiment. The current design is capped at
   within-TF 0.843 on this test set.
3. **`src/build_panda.py`'s leakage fix should be symmetric.** Score every training row
   out-of-fold, positives and negatives alike.
4. **`src/train.py` selects checkpoints on the test set** (best test AUROC across epochs). All
   reported model numbers carry that small optimistic bias. A held-out validation split would
   remove it.
5. **Gene-gene graphs (STRING, co-expression neighbours) carry almost no regulator
   information** in this data. That explains why PPI Methods 1/2 didn't help, and it argues
   against adding more gene-gene message passing.
6. **The biology the models miss is TF-specific.** EZH2's real targets form a significant
   developmental program (endoderm, heart, valve development), as Polycomb biology predicts.
   No model's EZH2 predictions show it. The baseline ranks EZH2's targets below chance.

### Possible next experiments (not run)

- Swap the decoder for `dot` or bilinear and compare within-TF AUROC.
- Replace PANDA with the one-line co-expression contrast score as the extra scalar. In the
  logistic regressions it matched PANDA (Section 2), and it costs much less to compute.
- Re-run Items 1, 2 and 4 on mESC and mHSC-E to check whether these findings hold in other
  cell types. Those cell types
  first need a STRING TF-TF file.

---

********* Oct 12th ********

# Week 2: can a GNN absorb PANDA's information? (advisor's three asks, plus our own next steps)

**Status:** experiments finished. Section 10 has the conclusions; Sections 11 to 16 hold the details.

## 8. Plan for this week

### What your advisor asked for (from `next-week.txt`)
1. **Check link prediction tasks in computer science and apply them to biology.** A background
   subagent wrote a literature review (`LINK_PREDICTION_REVIEW.md`). Applying its ideas comes
   last, as agreed.
2. **Remove the PANDA branch and add that to the GRN itself.** Your instruction: PANDA's useful
   information is already identified, so fold it into the GNN and try to beat the scRegNet
   baseline even without the Geneformer (scFM) branch.
3. **Try new graph methods on the GRN branch.** I read this as new graph encoders for the GNN
   branch: GraphSAGE, GAT, GIN and a directed version. This was not in my first plan, and I have
   now added it. It runs after the final-layer fix, so each encoder is compared with a decoder that
   can use it.

### Our own next steps from last week
1. Replace scRegNet's additive final layer with one that lets a TF and a gene interact.
2. Fix the message-passing / supervision edge split (a training pair must not be scored through
   its own edge).
3. Make the symmetric leakage fix permanent in `src/build_panda.py`. **Done:** the script now
   re-scores every training row (positive and negative) out-of-fold. It reproduces the matrix
   from last week exactly (maximum difference 0). I kept the old file as
   `panda_full_matrix_train_oof_legacy_asymmetric.npy`.
4. Try the one-line co-expression comparison in place of PANDA.
5. Choose epochs on a separate validation set instead of the test set.
6. Repeat on the other cell types.

### The tool I built for this: `src/pair_gnn.py`
One training script with switches for everything above, so every comparison changes one thing.
- **Decoder (the final layer):** `add` = linear layer on [TF, gene] (what scRegNet does);
  `had` = linear layer on [TF, gene, TF × gene] (adds an interaction term; this is biased
  matrix factorization from recommender systems); `mlp` = a hidden layer after joining TF and
  gene.
- **Edge split:** `batch` removes the current batch's own positive edges from the graph before
  scoring them, so a pair is never scored through its own edge. `none` is how scRegNet trains.
- **Foundation model:** Geneformer on or off. **Encoder:** GCN, SAGE, GAT, GIN or directed.
- **Extra PANDA-style inputs:** a pair-level number (PANDA score or the co-expression
  comparison) fed to the final layer, or used as GCN edge weights. The script scores both
  out-of-fold and treats every training row (positive and negative) the same way.
- **Epoch choice:** the script holds out 10% of the training rows as a validation set and keeps
  the epoch with the best validation AUROC. It never uses the test set for any choice.
- **Refit:** after choosing the epoch, the script retrains the model once on all training rows
  for that many epochs, then scores it once on test. This makes numbers comparable with last
  week's, which used all training rows. Last week's baseline numbers also chose the best epoch
  *on the test set*, so they run a little high.
- Metrics: AUROC, AUPRC and within-TF AUROC, as last week.

The harness runs about 20 times faster than `src/train.py`. The old code spent most of its time
converting a sparse matrix on every forward pass. One run now takes a minute or two.

---

## 9. The final layer, the edge split and GNN depth

**Script:** `src/pair_gnn.py` (runner `analysis/queue.sh`, job list `analysis/jobs/grid1.txt`,
results `analysis/out/pair/g*.json`, table from `analysis/agg.py`).

### Why

Last week showed that scRegNet's final layer adds a TF-only score to a gene-only score. I wanted
to know what changes when the model can look at the specific TF-gene pair. I also wanted to
check the training-edge problem from last week: the graph is built from all training positives,
and the model is trained on those same pairs.

### What I ran

13 configurations on hESC (TFs+500), 3 seeds each, with one optimizer setting for all of them
(Adam, learning rate 1e-4, weight decay 1e-3). All numbers below are the **refit** numbers:
I choose the epoch on a validation slice, retrain the model on all training rows for that
many epochs, and score the test set once. For reference, last week's baseline was
**0.8870 AUROC / 0.6245 AUPRC / 0.7215 within-TF**. That baseline chose its epoch on the test
set, so it runs a little high.

"GNN layers" is the number of graph layers. scRegNet uses 3 layers with widths 227, 29 and 34.
The "1 layer" models use one layer of width 256.

| Run | Final layer | Edge split | Geneformer | GNN layers | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|---|---|---|
| g01 | add (scRegNet's design) | none | yes | 3 | 0.8797 ± 0.0011 | 0.6094 ± 0.0013 | 0.6822 ± 0.0065 |
| g02 | add | batch | yes | 3 | 0.8842 ± 0.0006 | 0.6161 ± 0.0018 | 0.7078 ± 0.0020 |
| g03 | interacting (had) | none | yes | 3 | 0.8756 ± 0.0017 | 0.6053 ± 0.0041 | 0.6741 ± 0.0087 |
| g04 | interacting (had) | batch | yes | 3 | 0.8867 ± 0.0012 | 0.6246 ± 0.0014 | 0.7217 ± 0.0066 |
| g05 | hidden layer (mlp) | batch | yes | 3 | 0.8856 ± 0.0005 | 0.6212 ± 0.0015 | 0.7211 ± 0.0036 |
| g06 | add | none | no | 1 | 0.8816 ± 0.0013 | 0.6160 ± 0.0005 | 0.7159 ± 0.0006 |
| g10 | add | batch | no | 1 | 0.8834 ± 0.0003 | 0.6155 ± 0.0015 | 0.7063 ± 0.0027 |
| g07 | interacting (had) | none | no | 1 | 0.8809 ± 0.0011 | 0.6049 ± 0.0091 | 0.7286 ± 0.0055 |
| g08 | interacting (had) | batch | no | 1 | 0.8837 ± 0.0007 | 0.6013 ± 0.0050 | 0.7239 ± 0.0049 |
| g09 | hidden layer (mlp) | batch | no | 1 | 0.8876 ± 0.0008 | 0.6244 ± 0.0013 | 0.7183 ± 0.0017 |
| g11 | interacting (had) | batch | yes | 1 | **0.8893 ± 0.0015** | 0.6237 ± 0.0024 | **0.7365 ± 0.0048** |
| g12 | interacting (had) | batch | no | 3 | 0.8679 ± 0.0018 | 0.5821 ± 0.0033 | 0.5685 ± 0.0095 |
| g13 | interacting (had) | batch | no | 2 | 0.8837 ± 0.0009 | 0.6161 ± 0.0021 | 0.6727 ± 0.0092 |

### Finding 1: the edge split helps, and it is required for the interacting layer

- **Split alone, additive layer (g01 to g02):** AUROC +0.0045, within-TF +0.026. Scoring a training
  pair through its own graph edge cost accuracy even for scRegNet's design.
- **Interacting layer without the split (g01 to g03):** AUROC drops from 0.8797 to 0.8756 and
  within-TF from 0.682 to 0.674. These runs stop after 3 or 4 epochs because validation accuracy
  starts to fall.
- **Interacting layer with the split (g02 to g04):** AUROC +0.0025 and within-TF +0.014.
- **The likely reason (not measured directly):** without the split, every training positive is a
  direct neighbour in the graph, so a layer that multiplies the TF and gene embeddings can learn
  "these two are linked", which is true in training and false at test time. An additive layer
  can't use that, so it does no harm there. The literature review reached the same prediction
  for dot-product decoders.

### Finding 2: graph depth matters most

- With the interacting layer, no Geneformer and the edge split, within-TF AUROC falls from
  **0.724 (1 layer) to 0.673 (2 layers) to 0.569 (3 layers)**.
- A one-layer GNN-only model with scRegNet's additive layer (g06, within-TF 0.716) beats
  the full scRegNet design in this harness (g01, 0.682).
- **Likely reason:** the training graph is dominated by a few TFs. TFAP2A has about 480 targets
  and JUND, NANOG, OTX2 and TEAD4 have 200 to 300 each, so nearly any gene reaches nearly any
  other in two steps. Every extra layer averages a gene's features with a larger share of the
  whole panel, which washes out what is specific to that gene or TF. scRegNet's second layer
  also squeezes everything into 29 numbers.
- **Biologically:** the most informative signal for a TF is its own list of known targets, one
  step away. Three steps away, everything is connected through the same few hub TFs.

### Finding 3: Geneformer is not needed to match scRegNet's original numbers

- **g09** (GNN only, one layer, hidden-layer final layer, edge split) scores **0.8876 AUROC,
  0.6244 AUPRC, 0.7183 within-TF**. The original scRegNet scored 0.8870, 0.6245 and 0.7215. The
  two are equal within seed spread, with no foundation model.
- Geneformer still helps the interacting layer: **g11** (same as g08 plus
  Geneformer) reaches 0.8893 AUROC and **0.7365** within-TF, against 0.8837 and 0.7239 without it.
- g11 is ahead of the original scRegNet baseline on AUROC (+0.0023) and within-TF (+0.015), and
  level on AUPRC.

### Finding 4: the two interacting layers trade off

The multiplication layer (`had`) gives the better within-TF score; the hidden layer (`mlp`)
gives the better AUROC and AUPRC. The gaps are small, and I did not tune either.

### Cautions

- I chose the Adam setting from small validation-set sweeps on four configurations, then used
  it for all. Some families may do better with their own setting. My
  baseline here (g01, 0.8797) is *below* the original scRegNet (0.8870). The original was trained
  with slow SGD, and Adam stops at epoch 4 or 5 for that design. I checked this later: the
  SGD recipe scores 0.8830 / 0.6155 / 0.7029 over 5 seeds in the same harness (Section 14),
  so it is somewhat better than Adam for that design but still below the original checkpoint.
- Differences smaller than about 0.002 AUROC or 0.005 within-TF are inside the seed spread.
- Only hESC so far.

---

## 10. Summary and conclusions (final for this week)

All numbers are hESC (TFs+500) unless a row names another cell type. They are mean ± standard
deviation over seeds in the refit protocol (epoch chosen on a validation slice, retrained on all
training rows, test scored once). Reference points: the original scRegNet checkpoint,
**0.8870 AUROC / 0.6245 AUPRC / 0.7215 within-TF** (epoch chosen on the test set, so slightly
optimistic), and last week's best model, scRegNet + PANDA with the symmetric fix,
**0.8915 / 0.6337 / 0.7471**. "Hero" means: one GraphSAGE layer, an interacting final layer, the
per-batch edge split, no PANDA.

### 10.0 The answer in one paragraph

scRegNet's score is a TF-only term plus a gene-only term, so on a benchmark where two
popularity counts nearly match it, it adds little beyond popularity. PANDA helped because it was
the only TF-specific input. A GNN with an interacting final layer and leak-free training finds
the same information itself, so the PANDA branch can go: the hero with Geneformer scores
**0.8910 / 0.6346 / 0.7517**, level with last week's PANDA-augmented model. The gain holds
for completing partly known target lists (hESC, mESC, mHSC-E; not hHEP). It does **not** extend
to new TFs or new genes, where nothing I tested beats popularity.

### 10.1 Conclusions, each tied to the result behind it

| # | Conclusion | Evidence (where) |
|---|---|---|
| 1 | The standard benchmark mostly rewards popularity, on every cell type. | Two counts per pair come within 0.003 to 0.008 AUROC of the original scRegNet on hESC, hHEP, mHSC-E and mESC (Section 16). On hESC alone they give 0.8797 against 0.8874. |
| 2 | scRegNet's final layer cannot rank genes differently for different TFs. | Additive fit R² = 1.000 and identical gene ranking for all 34 TFs (last week, Section 4). |
| 3 | Training a pair through its own graph edge is the largest single problem. Any interacting final layer needs the fix. | Edge split vs none: **+0.078 within-TF, better for 28 of 33 TFs, p = 3 × 10⁻⁶** (Section 14). SAGE with an interacting layer and no split falls to 0.8486 AUROC / 0.635 within-TF (Section 11). |
| 4 | The hero beats the scRegNet design within TFs, significantly, and edges it on pooled scores. | Hero vs tuned scRegNet design **+0.051 within-TF (p = 0.016)**; vs the published recipe **+0.042 (p = 0.035)**; with default settings and Geneformer: **+0.049 (p = 0.001)**, AUROC 0.8910 against 0.8830 (Section 14). |
| 5 | The gain replicates on 3 of 4 cell types and is largest on mESC. | mESC: 0.9389 / 0.8814 / 0.8249 against the original checkpoint's 0.9260 / 0.8593 / 0.7660. mHSC-E: within-TF 0.8081 against 0.7339. hHEP: no gain (Section 13). |
| 6 | The PANDA branch can be removed. Its information is already in the hero. | Adding the PANDA or co-expression number to the hero: −0.0003 to +0.0005 within-TF, no interval excludes zero (Section 12). PANDA as graph edge weights: −0.007. Direct check: the hero's scores correlate with the co-expression number beyond popularity (partial ρ 0.31, against 0.07 for the additive model; moderate, Section 12). |
| 7 | Depth hurts, and no single graph encoder wins on all datasets. | GCN within-TF 0.724 / 0.673 / 0.569 for 1 / 2 / 3 layers (Section 9). SAGE best on hESC (0.7489) and mESC (0.8247); GCN + Geneformer best on mHSC-E (0.8081) and hHEP; GAT, GIN and directed SAGE are not better (Sections 11, 13). |
| 8 | Geneformer's contribution is small and depends on the settings. | Shuffling its vectors across genes changes the scRegNet design by −0.003 within-TF. On the default hero it adds +0.005 within-TF (p = 0.07) and +0.009 AUPRC; on the validation-tuned hero it adds −0.008 (Section 14). |
| 9 | A transparent model with five hand-built features is statistically tied with the GNN on hESC and beats the original scRegNet on mESC. | Hero minus stacker: +0.005 to +0.012 within-TF, p ≥ 0.58 (Section 14). mESC stacker 0.9312 / 0.8601 / 0.8142 against 0.9260 / 0.8593 / 0.7660 (Section 16). |
| 10 | Nothing tested transfers to unseen TFs or genes. | Held-out TFs: hero is **0.042 below plain gene popularity (p = 0.0007)**. Held-out genes: within-TF 0.52 to 0.57. TF-level edge dropout did not fix it (Section 15). |
| 11 | The hero's top predictions are better where a biologist looks. | Precision at 10 per TF: 0.480 against 0.415 (base rate 0.239). EZH2: 8 of 30 held-out targets in the top 30, against 3 for the scRegNet design (chance 3.9). GO enrichment of its top new predictions is suggestive, not significant (Section 16). |
| 12 | Averaging seeds gives a clear gain. | 5-seed ensemble of the default hero + Geneformer: **0.8935 / 0.6410 / 0.7574** (Section 14). |

### 10.2 What the evidence does not support

- **No clear win on pooled AUROC or AUPRC for the GNN-only hero.** Its AUROC gain over the
  published recipe is +0.003 to +0.005 and its AUPRC is level. The gain is in within-TF ranking
  and, with Geneformer, in AUPRC.
- **The baseline comparison is imperfect.** The original checkpoints are single seeds with the
  epoch chosen on the test set (about +0.0015 AUROC and +0.01 within-TF of optimism in my
  measurement). My reproduction of the published recipe scores 0.8830 / 0.6155 / 0.7029.
- **Tuning did not help.** The validation-chosen hero was not better on test than the hand-set
  default, so I report both. Search budgets were 7 to 13 trials per family.
- **"SAGE beats GCN" is not general** (Section 13), and "interacting beats additive" is not
  significant across TFs on hESC alone (p = 0.47), although it is positive on mESC and mHSC-E.
- **Generalization to new TFs and genes failed** (Section 15), so I limit the claim to
  completing known target lists.
- I did not run the pair-aware graph networks from the literature review (SEAL, NBFNet, neural
  common neighbours).

### 10.3 Advisor's three asks: status

| Ask | Status | Answer |
|---|---|---|
| 1. Check link prediction in computer science and apply it | Done for the review and for the simple methods | `LINK_PREDICTION_REVIEW.md`; collaborative filtering, Katz, resource allocation and stackers run (Section 16); the leak-free training and the biased interacting final layer came from it. Pair-aware GNNs not run. |
| 2. Fold PANDA into the GRN, no separate branch | Done | The hero absorbs it; adding PANDA as a number or as edge weights does not help (Section 12). |
| 3. Try new graph methods on the GRN branch | Done | SAGE, GAT, GIN and directed SAGE against GCN, at 1 and 3 layers (Sections 11, 13). |

### 10.4 Where this could go as a paper

**Framing that the evidence supports:** an analysis-and-fix paper. The evidence does not
support a claim of a new state-of-the-art model. *Gene regulatory network link predictors reward popularity and leak training edges: a
diagnosis, a leak-free interacting design, and a held-out evaluation.*

1. The standard benchmark is mostly popularity (four cell types).
2. Concatenation plus a linear layer cannot rank genes differently for different TFs.
3. Training edges leak into the message-passing graph; the fix is required and has a large
   effect.
4. With the fix, a shallow GNN with an interacting layer improves within-TF ranking on three of
   four cell types and absorbs PANDA's information.
5. Simple feature stackers are strong baselines that every GRN paper should report.
6. On unseen TFs and genes no tested method beats popularity.

**Still to do before submission:**

| Step | What |
|---|---|
| 1 | Per-cell-type tuning and 5 seeds for the other three cell types, with paired tests |
| 2 | The other BEELINE ground truths (non-cell-type-specific ChIP-seq, STRING, perturbation) and the missing mHSC-GM and mHSC-L cell types |
| 3 | Run at least one pair-aware model from the review (NBFNet-style, with the TF as source) |
| 4 | Try to fix held-out TFs with TF-level information (motifs, binding-domain embeddings) |
| 5 | A better tuning criterion than validation AUROC (for example validation within-TF AUROC) |
| 6 | Figures and the written paper; I have not started a draft, as you asked |


---

## 11. New graph methods on the GRN branch (advisor ask 3)

**Script:** `src/pair_gnn.py --encoder {gcn,sage,gat,gin,dir}` (jobs in `analysis/jobs/grid2.txt`
and `grid3.txt`).

### Why

Your advisor asked to try new graph methods on the GRN branch. Last week showed that the
final layer limits scRegNet more than the graph encoder does. I fixed the final layer and the
edge leak, then swapped only the graph encoder and kept everything else the same.

### What I ran

Hold fixed: interacting final layer, per-batch edge split, no Geneformer, Adam with the same
settings, 3 seeds, refit protocol. Vary: the encoder and the depth.
- **GCN:** scRegNet's encoder. Averages a gene with its neighbours using symmetric weights.
- **GraphSAGE (SAGE):** keeps a separate weight for the gene's own features and one for the
  average of its neighbours.
- **GAT:** learns an attention weight for each neighbour.
- **GIN:** an MLP on the node plus its neighbours. I use the mean of neighbours, because summing
  overflows on hub genes such as TFAP2A, which has about 480 neighbours.
- **Directed SAGE (dir):** separate weights for messages that go from a TF to its target and
  messages that go back.

"1 layer" is one layer of width 256. "3 layers" is scRegNet's depth (widths 227, 29, 34).

### Result (refit test numbers)

| Encoder | Layers | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|
| GCN | 1 | 0.8837 ± 0.0007 | 0.6013 ± 0.0050 | 0.7239 ± 0.0049 |
| **SAGE** | 1 | **0.8884 ± 0.0013** | **0.6252 ± 0.0014** | **0.7489 ± 0.0022** |
| GAT | 1 | 0.8825 ± 0.0012 | 0.6115 ± 0.0059 | 0.7039 ± 0.0057 |
| GIN (mean) | 1 | 0.8851 ± 0.0012 | 0.6154 ± 0.0075 | 0.7294 ± 0.0018 |
| Directed SAGE | 1 | 0.8864 ± 0.0007 | 0.6193 ± 0.0035 | 0.7368 ± 0.0015 |
| GCN | 3 | 0.8679 ± 0.0018 | 0.5821 ± 0.0033 | 0.5685 ± 0.0095 |
| SAGE | 3 | 0.8879 ± 0.0013 | 0.6293 ± 0.0018 | 0.7360 ± 0.0047 |
| GAT | 3 | 0.8765 ± 0.0014 | 0.5962 ± 0.0033 | 0.6759 ± 0.0063 |
| GIN (mean) | 3 | 0.8819 ± 0.0007 | 0.6154 ± 0.0038 | 0.6934 ± 0.0019 |
| Directed SAGE | 3 | 0.8840 ± 0.0010 | 0.6202 ± 0.0036 | 0.7211 ± 0.0019 |

(Directed SAGE with 1 layer has 2 finished seeds so far.)

### Finding 1: SAGE is the best encoder at both depths

- At 1 layer SAGE leads GCN by +0.0047 AUROC, +0.024 AUPRC and **+0.025 within-TF**.
- At 3 layers the gap is large: SAGE keeps within-TF 0.736 while GCN falls to 0.569. SAGE is the
  only encoder that tolerates scRegNet's depth.
- **Why this is likely:** GCN averages a gene's own features with its neighbours' using a
  symmetric weight, so in this hub-dominated graph the neighbours dilute a gene's own signal.
  SAGE keeps a separate weight for the gene itself. This is a hypothesis; I did not test it in isolation.
- **Biologically:** a TF's embedding should describe the TF's *program*, the typical expression
  of its known targets. A gene's embedding should keep the gene's own expression. SAGE keeps
  these two things apart, and the interacting final layer then asks whether the gene fits the
  TF's program. This is the same question PANDA's co-expression term asks (Section 12).

### Finding 2: attention and directions do not help

- GAT is below GCN at 1 layer and at 3 layers. With only about 3 known regulators for a typical
  gene and a few huge hubs, there is little for attention to choose between.
- Separating the two edge directions (directed SAGE) is below plain SAGE at both depths
  (0.7368 against 0.7489 at 1 layer). TF-target direction is already encoded by which side is a
  TF, so the extra weights add parameters without adding information.

### Finding 3: the edge leak hurts SAGE more than GCN

SAGE with the interacting final layer and **no** edge split (k09) reaches
**0.8486 AUROC and 0.6350 within-TF**. With the split, the same model reaches 0.8884 and 0.7489.
Best epochs without the split are 1 to 3, so the model overfits in the first few passes.
GCN showed the same direction with a smaller effect (g07 against g08). Because SAGE keeps the
gene's own features and its neighbours' average in separate channels, a TF's embedding
contains its target's features exactly when the pair is a training edge, and the interacting
layer reads that out. The edge split is therefore the first fix for any pair-scoring GNN, and
this matches the warnings in the literature review.

### Cautions
- One optimizer setting for all encoders; an encoder could do better with its own. The
  equal-budget tuning in Section 14 checks this for the leading models.
- Differences below about 0.002 AUROC or 0.005 within-TF are inside seed spread.

---

## 12. Folding PANDA's information into the GNN (advisor ask 2)

**Scripts:** `src/pair_gnn.py` (`--scalar`, `--graph_w`), `analysis/scalars.py`,
`analysis/panda_fold_table.py`, final runs in `analysis/jobs/finals.txt`.

### Why

Your advisor does not want PANDA as a separate branch. Last week showed what PANDA adds: a
TF-specific number that compares a gene's co-expression with the TF's known targets against its
co-expression with non-targets. The question here is whether the GNN can take that information
in itself, so the PANDA branch can go.

### What I tried

1. **Let the GNN find it (no PANDA at all).** One graph layer averages each TF's known
   targets into the TF's embedding (this is the TF's "program"). An interacting final layer then
   multiplies the TF's program with the gene's own features. That product is the same
   comparison PANDA computes, up to a learned weighting. This is the model called "hero".
2. **Give the model the PANDA number as well**, at the final layer, as in last week's Exp C.
3. **Give it the cheap co-expression number** (no PANDA algorithm) in the same place.
4. **Put PANDA on the graph edges** as edge weights (the literal reading of "add PANDA to the
   GRN"). Each training edge gets a weight from its out-of-fold PANDA score.

I score every PANDA or co-expression number so that no row sees its own label: training rows
use out-of-fold scores (symmetric fix), validation and test rows use scores built from the
training edges only.

### Result (hESC, refit protocol; change is within-TF AUROC relative to the model above it)

| Model | Variant | Seeds | AUROC | AUPRC | Within-TF | Change in within-TF |
|---|---|---|---|---|---|---|
| GCN 1-layer, interacting, no Geneformer | none (control) | 3 | 0.8837 ± 0.0007 | 0.6013 ± 0.0050 | 0.7239 ± 0.0049 | - |
| | + co-expression number | 3 | 0.8840 ± 0.0007 | 0.6031 ± 0.0049 | 0.7268 ± 0.0035 | +0.0029 |
| | + PANDA number | 3 | 0.8846 ± 0.0005 | 0.6053 ± 0.0050 | 0.7277 ± 0.0043 | +0.0038 |
| | PANDA as graph edge weights | 3 | 0.8835 ± 0.0007 | 0.6010 ± 0.0049 | 0.7169 ± 0.0087 | -0.0070 |
| | edge weights + PANDA number | 3 | 0.8844 ± 0.0006 | 0.6052 ± 0.0054 | 0.7215 ± 0.0068 | -0.0024 |
| GCN 1-layer, hidden-layer final, no Geneformer | none (control) | 3 | 0.8876 ± 0.0008 | 0.6244 ± 0.0013 | 0.7183 ± 0.0017 | - |
| | + co-expression number | 3 | 0.8881 ± 0.0010 | 0.6231 ± 0.0025 | 0.7275 ± 0.0036 | +0.0091 |
| | + PANDA number | 3 | 0.8904 ± 0.0004 | 0.6284 ± 0.0030 | 0.7329 ± 0.0051 | +0.0146 |
| GCN 1-layer, interacting, Geneformer | none (control) | 3 | 0.8893 ± 0.0015 | 0.6237 ± 0.0024 | 0.7365 ± 0.0048 | - |
| | + co-expression number | 3 | 0.8900 ± 0.0020 | 0.6230 ± 0.0011 | 0.7414 ± 0.0044 | +0.0049 |
| | + PANDA number | 3 | 0.8903 ± 0.0019 | 0.6241 ± 0.0011 | 0.7418 ± 0.0046 | +0.0053 |
| scRegNet's design | none (control) | 3 | 0.8797 ± 0.0011 | 0.6094 ± 0.0013 | 0.6822 ± 0.0065 | - |
| | + PANDA number | 3 | 0.8805 ± 0.0004 | 0.6119 ± 0.0029 | 0.6842 ± 0.0030 | +0.0020 |
| | + co-expression number | 3 | 0.8805 ± 0.0004 | 0.6118 ± 0.0028 | 0.6843 ± 0.0028 | +0.0021 |
| scRegNet's design + edge split | none (control) | 3 | 0.8842 ± 0.0006 | 0.6161 ± 0.0018 | 0.7078 ± 0.0020 | - |
| | + PANDA number | 3 | 0.8855 ± 0.0014 | 0.6202 ± 0.0039 | 0.7129 ± 0.0035 | +0.0051 |
| GCN 1-layer, additive, no Geneformer | none (control) | 3 | 0.8816 ± 0.0013 | 0.6160 ± 0.0005 | 0.7159 ± 0.0006 | - |
| | + PANDA number | 3 | 0.8829 ± 0.0014 | 0.6190 ± 0.0008 | 0.7195 ± 0.0010 | +0.0037 |
| SAGE 1-layer, interacting, no Geneformer (default settings) | none (control) | 3 | 0.8884 ± 0.0013 | 0.6252 ± 0.0014 | 0.7489 ± 0.0022 | - |
| | + co-expression number | 3 | 0.8885 ± 0.0013 | 0.6255 ± 0.0010 | 0.7488 ± 0.0019 | -0.0001 |
| SAGE 1-layer, interacting, no Geneformer (tuned settings) | none (control) | 5 | 0.8862 ± 0.0026 | 0.6154 ± 0.0077 | 0.7449 ± 0.0109 | - |
| | + co-expression number | 5 | 0.8857 ± 0.0040 | 0.6140 ± 0.0117 | 0.7446 ± 0.0117 | -0.0003 |
| | + PANDA number | 5 | 0.8867 ± 0.0023 | 0.6161 ± 0.0087 | 0.7454 ± 0.0122 | +0.0005 |
| SAGE 1-layer, interacting, Geneformer (tuned settings) | none (control) | 5 | 0.8871 ± 0.0018 | 0.6324 ± 0.0017 | 0.7367 ± 0.0084 | - |
| | + PANDA number | 5 | 0.8877 ± 0.0015 | 0.6333 ± 0.0015 | 0.7378 ± 0.0068 | +0.0011 |

Paired comparison of the hero with and without the extra number, per TF (33 test TFs, 5 seeds
averaged, tuned settings, `analysis/final_report.py`):

| Comparison | Within-TF change | 95% interval over TFs | TFs better | Wilcoxon p |
|---|---|---|---|---|
| hero + co-expression number vs hero | -0.0003 | [-0.0025, +0.0017] | 20/33 | 0.63 |
| hero + PANDA number vs hero | +0.0005 | [-0.0019, +0.0027] | 21/33 | 0.21 |
| hero + Geneformer + PANDA vs hero + Geneformer | +0.0011 | [-0.0003, +0.0023] | 22/33 | 0.038 |

### Finding 1: the GNN already contains PANDA's information

- Adding the PANDA or co-expression number to the SAGE hero changes within-TF AUROC by −0.0003 to
  +0.0011, which is inside the seed spread, and no interval excludes zero. (The one signed-rank
  p below 0.05, 0.038 for PANDA on hero + Geneformer, belongs to a +0.0011 change.) With the
  default settings the co-expression number moves it from 0.7489 to 0.7488.
- The hero with nothing extra scores within-TF 0.7471 ± 0.0047 (default settings, 5 seeds) and
  0.7449 ± 0.0109 (validation-tuned settings, 5 seeds). Last week's best model, scRegNet
  plus PANDA, scored 0.7471. So the hero matches it without PANDA and without Geneformer.
- **So the PANDA branch can be removed.** The recipe that replaces it is: one graph layer
  (SAGE), an interacting final layer, and leak-free training (Sections 9 and 11).

### Finding 2: on weaker models the extra number helps a little

For the GCN models and for scRegNet's own design, the PANDA number adds +0.002 to +0.015
within-TF. The largest gain is for the hidden-layer final layer (+0.015). These models have not
found the comparison on their own, so the explicit number fills part of the gap. This is the
same pattern as last week, where PANDA helped a model that could not rank genes differently for
different TFs.

### Finding 3: PANDA on the graph edges hurts

Edge weights lower within-TF AUROC by 0.007 (0.7169 against 0.7239; only the GCN model was
tested this way, 3 seeds, so the change is inside the spread). I found no gain and have not
tested why. The final layer is a better place for the PANDA number than the edges. For the SAGE
hero, no placement is needed, because the GNN finds the information itself.

### Why the GNN can find it (the mechanism)

- A TF's embedding, after one SAGE layer, is a learned mix of the TF's own expression and the
  **average expression of its known targets**.
- A gene's embedding keeps the gene's own expression profile in a separate channel.
- The interacting final layer multiplies the two. A product of the gene's profile with the
  average profile of the TF's targets is a correlation between them. This is "does this gene
  behave like this TF's targets?".
- This is a hypothesis about what the trained network does. Three results support it: supplying
  the explicit number adds nothing to the SAGE hero, the same model with an
  additive final layer (which cannot form this product) loses 0.033 within-TF (0.7163 against 0.7489, run k03 in Section 10),
  and the direct check below shows the hero's scores agree with the co-expression number much more than the additive model's.
  The agreement is moderate (partial ρ about 0.3), so the hero picks up part of this signal. It does not reproduce it.
- **Biologically:** this is guilt by association. If a gene's expression moves with the genes a
  TF already regulates, the gene likely sits in the same regulatory program.

**Direct check.** Script: `analysis/hero_vs_contrast.py`. Output: `analysis/out/hero_vs_contrast.txt`.

- For each of 33 test TFs (at least 8 test rows, both classes), I took the model's test scores
  and the co-expression contrast number (and the PANDA number) for that TF's candidate genes.
- I measured the Spearman correlation between them, then a **partial** correlation that removes
  gene popularity (log of the number of training positives of the target). Both scores could
  track popularity alone.
- I averaged over TFs, then took mean ± std over 5 seeds.

| Model | ρ with contrast | partial ρ with contrast | partial ρ with PANDA |
|---|---|---|---|
| hero, default (GNN only) | 0.269 ± 0.026 | **0.309 ± 0.045** | 0.289 ± 0.035 |
| hero, validation-tuned | 0.261 ± 0.033 | 0.294 ± 0.047 | 0.279 ± 0.042 |
| hero + Geneformer | 0.284 ± 0.022 | 0.338 ± 0.034 | 0.326 ± 0.023 |
| hero with GCN encoder | 0.245 ± 0.036 | 0.303 ± 0.050 | 0.239 ± 0.045 |
| hero with additive final layer | 0.097 ± 0.025 | 0.074 ± 0.050 | 0.079 ± 0.023 |
| scRegNet design, tuned | 0.097 ± 0.001 | 0.071 ± 0.001 | 0.062 ± 0.003 |
| scRegNet design, original recipe | 0.094 ± 0.003 | 0.067 ± 0.005 | 0.065 ± 0.001 |

Paired over the 33 TFs (partial ρ with contrast, seed-averaged), default hero minus:

- additive-final-layer hero: **+0.234**, 95% CI [+0.147, +0.327], hero higher for 26 of 33 TFs, Wilcoxon p = 2 × 10⁻⁵.
- scRegNet original recipe: **+0.242**, 95% CI [+0.162, +0.326], hero higher for 27 of 33 TFs, p = 3 × 10⁻⁶.
- Against PANDA the partial gaps are +0.210 (additive hero, p = 3 × 10⁻⁶) and +0.225 (scRegNet recipe, p = 6 × 10⁻⁶).

What this shows:

- The hero's ranking agrees with the co-expression number beyond popularity. A partial ρ of 0.3 is moderate.
- Models that cannot form the TF × gene product sit near 0.07. Only the interacting layers show the agreement.
- The GCN hero agrees too (0.303), so SAGE is not required.
- The hero agrees with PANDA at a similar level (0.289). PANDA is built from the same co-expression and known-target signal, so this is expected.
- Correlation shows agreement. It does not show that the network computes this product. The hero may use
  the same information another way.
- Context: within-TF AUROC of the co-expression number alone is 0.574, below the 0.58 to 0.63
  I expected. PANDA alone is 0.664. Within a TF, the contrast number has almost no link to
  popularity (ρ = 0.07); PANDA has a clear one (ρ = 0.37).
- Not checked: whether the agreement survives when I also remove the test label. A good
  model and a weakly informative number both lean toward positives.

### Cautions

- The result needs a TF that has known targets in the training data. A TF with none has no
  program to average, and Section 15 shows the hero then performs below simple popularity.
- One cell type here; Section 13 shows how well it carries over.

---

## 13. Do the findings hold on the other cell types?

**Scripts:** `src/pair_gnn.py --cell_type {hHEP,mHSC-E,mESC}`, `analysis/replication_table.py`,
jobs in `analysis/jobs/grid4.txt`.

### Why

Everything so far used hESC. A result from one dataset alone is weak. BEELINE
ships three more cell types in this repository: human hepatocytes (hHEP), mouse blood
progenitors (mHSC-E) and mouse embryonic stem cells (mESC).

### What I ran

The same models as the hESC grid, with the **hESC settings reused as they are** (no tuning per
cell type), 3 seeds each, refit protocol. The reference rows are the original scRegNet
checkpoint for that cell type (the paper's recipe, epoch chosen on test), the two-number
popularity model, and the five-feature stacker without a neural network (Section 16).

**hESC**

| Model | Seeds | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|
| original scRegNet checkpoint (paper recipe, epoch chosen on test) | 1 | 0.8874 | 0.6225 | 0.7202 |
| popularity only (two counts, logistic regression) | - | 0.8797 | 0.5949 | 0.7068 |
| no-neural-network stacker (popularity + CF + 3-paths + co-expression number) | - | 0.8863 | 0.6090 | 0.7375 |
| scRegNet design (add, no split, Geneformer, 3-layer GCN) | 3 | 0.8797 ± 0.0011 | 0.6094 ± 0.0013 | 0.6822 ± 0.0065 |
|   + edge split | 3 | 0.8842 ± 0.0006 | 0.6161 ± 0.0018 | 0.7078 ± 0.0020 |
| GCN 1-layer, interacting, split, no Geneformer | 3 | 0.8837 ± 0.0007 | 0.6013 ± 0.0050 | 0.7239 ± 0.0049 |
| GCN 1-layer, hidden-layer final, split, no Geneformer | 3 | 0.8876 ± 0.0008 | 0.6244 ± 0.0013 | 0.7183 ± 0.0017 |
| GCN 1-layer, interacting, split, Geneformer | 3 | 0.8893 ± 0.0015 | 0.6237 ± 0.0024 | 0.7365 ± 0.0048 |
| SAGE 1-layer, interacting, split, no Geneformer | 3 | 0.8884 ± 0.0013 | 0.6252 ± 0.0014 | 0.7489 ± 0.0022 |
|   + co-expression number | 3 | 0.8885 ± 0.0013 | 0.6255 ± 0.0010 | 0.7488 ± 0.0019 |

**hHEP**

| Model | Seeds | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|
| original scRegNet checkpoint (paper recipe, epoch chosen on test) | 1 | 0.9027 | 0.8419 | 0.7882 |
| popularity only (two counts, logistic regression) | - | 0.8985 | 0.8270 | 0.7838 |
| no-neural-network stacker (popularity + CF + 3-paths + co-expression number) | - | 0.8996 | 0.8302 | 0.7593 |
| scRegNet design (add, no split, Geneformer, 3-layer GCN) | 3 | 0.9021 ± 0.0003 | 0.8409 ± 0.0006 | 0.7867 ± 0.0009 |
|   + edge split | 3 | 0.9027 ± 0.0002 | 0.8413 ± 0.0005 | 0.7868 ± 0.0024 |
| GCN 1-layer, interacting, split, no Geneformer | 3 | 0.8984 ± 0.0016 | 0.8318 ± 0.0026 | 0.7707 ± 0.0010 |
| GCN 1-layer, hidden-layer final, split, no Geneformer | 3 | 0.9028 ± 0.0003 | 0.8399 ± 0.0010 | 0.7734 ± 0.0039 |
| GCN 1-layer, interacting, split, Geneformer | 3 | 0.9038 ± 0.0006 | 0.8454 ± 0.0001 | 0.7891 ± 0.0006 |
| SAGE 1-layer, interacting, split, no Geneformer | 3 | 0.8988 ± 0.0006 | 0.8377 ± 0.0006 | 0.7583 ± 0.0012 |
|   + co-expression number | 3 | 0.8986 ± 0.0002 | 0.8379 ± 0.0007 | 0.7584 ± 0.0022 |

**mHSC-E**

| Model | Seeds | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|
| original scRegNet checkpoint (paper recipe, epoch chosen on test) | 1 | 0.9223 | 0.9402 | 0.7339 |
| popularity only (two counts, logistic regression) | - | 0.9147 | 0.9322 | 0.7132 |
| no-neural-network stacker (popularity + CF + 3-paths + co-expression number) | - | 0.9139 | 0.9301 | 0.7420 |
| scRegNet design (add, no split, Geneformer, 3-layer GCN) | 3 | 0.9213 ± 0.0004 | 0.9406 ± 0.0002 | 0.7299 ± 0.0011 |
|   + edge split | 3 | 0.9207 ± 0.0003 | 0.9401 ± 0.0002 | 0.7279 ± 0.0011 |
| GCN 1-layer, interacting, split, no Geneformer | 3 | 0.9195 ± 0.0057 | 0.9397 ± 0.0027 | 0.7751 ± 0.0031 |
| GCN 1-layer, hidden-layer final, split, no Geneformer | 3 | 0.9240 ± 0.0003 | 0.9422 ± 0.0002 | 0.7456 ± 0.0049 |
| GCN 1-layer, interacting, split, Geneformer | 3 | 0.9259 ± 0.0004 | 0.9445 ± 0.0001 | 0.8081 ± 0.0027 |
| SAGE 1-layer, interacting, split, no Geneformer | 3 | 0.9239 ± 0.0006 | 0.9436 ± 0.0008 | 0.7642 ± 0.0076 |
|   + co-expression number | 3 | 0.9228 ± 0.0019 | 0.9428 ± 0.0013 | 0.7659 ± 0.0108 |

**mESC**

| Model | Seeds | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|
| original scRegNet checkpoint (paper recipe, epoch chosen on test) | 1 | 0.9260 | 0.8593 | 0.7660 |
| popularity only (two counts, logistic regression) | - | 0.9232 | 0.8505 | 0.7612 |
| no-neural-network stacker (popularity + CF + 3-paths + co-expression number) | - | 0.9312 | 0.8601 | 0.8142 |
| scRegNet design (add, no split, Geneformer, 3-layer GCN) | 3 | 0.9238 ± 0.0007 | 0.8557 ± 0.0012 | 0.7616 ± 0.0010 |
|   + edge split | 3 | 0.9235 ± 0.0005 | 0.8555 ± 0.0011 | 0.7614 ± 0.0019 |
| GCN 1-layer, interacting, split, no Geneformer | 3 | 0.9269 ± 0.0008 | 0.8560 ± 0.0055 | 0.7875 ± 0.0021 |
| GCN 1-layer, hidden-layer final, split, no Geneformer | 3 | 0.9375 ± 0.0003 | 0.8782 ± 0.0006 | 0.8164 ± 0.0025 |
| GCN 1-layer, interacting, split, Geneformer | 3 | 0.9372 ± 0.0009 | 0.8788 ± 0.0012 | 0.8116 ± 0.0024 |
| SAGE 1-layer, interacting, split, no Geneformer | 3 | 0.9389 ± 0.0010 | 0.8814 ± 0.0019 | 0.8249 ± 0.0021 |
|   + co-expression number | 3 | 0.9392 ± 0.0010 | 0.8817 ± 0.0018 | 0.8277 ± 0.0017 |

### Finding 1: three of four cell types show a clear within-TF gain

Change from the scRegNet design (first model row) to the best model with the interacting final
layer and edge split, with the encoder chosen on the table:

| Cell type | Best new model | AUROC | AUPRC | Within-TF |
|---|---|---|---|---|
| hESC | SAGE, no Geneformer | +0.0087 | +0.0158 | **+0.0667** |
| mESC | SAGE, no Geneformer | **+0.0151** | **+0.0257** | **+0.0633** |
| mHSC-E | GCN, interacting, Geneformer | +0.0046 | +0.0039 | **+0.0782** |
| hHEP | GCN, interacting, Geneformer | +0.0017 | +0.0045 | +0.0024 |

- On mESC the best new model beats the original checkpoint on every measure
  (0.9389 / 0.8814 / 0.8249 against 0.9260 / 0.8593 / 0.7660) and beats the
  no-neural-network stacker (0.9312 / 0.8601 / 0.8142).
- On mHSC-E, GCN with Geneformer reaches within-TF 0.8081 against 0.7339 for the original.

### Finding 2: hHEP shows no gain, and nothing beats popularity there

On hHEP, the two-number popularity model has within-TF 0.7838 and the original scRegNet has
0.7882. No model with the interacting layer is above 0.7891. The co-expression number alone is
weak on hHEP (within-TF 0.55, against 0.58 to 0.63 on the other cell types). I did not find out
why. Possible reasons are that hHEP cells have no time or stage structure to give TF-target
co-expression its signal (the cell names are plate wells), or that its ground truth is
dominated by popular genes. I have not tested either.

### Finding 3: the best encoder depends on the dataset

SAGE was best on hESC and mESC. GCN with Geneformer was best on mHSC-E and hHEP. The
"SAGE is better" statement from Section 11 applies only to hESC and mESC.
The combination that replicates is of an interacting final layer, the edge split
and a shallow graph.

### Finding 4: the edge split matters less for the additive design

For the scRegNet design the split changes the results by under 0.001 AUROC on hHEP, mHSC-E
and mESC. The split matters with an interacting layer (Section 9), and this section's best models
use that layer.

### Cautions
- 3 seeds and no per-cell-type tuning; the original rows are single checkpoints with the epoch
  chosen on the test set.
- Only hESC has a full 5-seed, tuned, paired-test treatment (Section 14).

---

## 14. The fair comparison on hESC: equal tuning, 5 seeds, paired tests

**Scripts:** `analysis/hpo_jobs.py` and `analysis/hpo_pick.py` (tuning), `analysis/make_finals.py`
(final job list), `analysis/final_report.py` (tables and paired tests; output
`analysis/out/final_report.txt`).

### Why

The earlier tables used one hand-picked Adam setting for every model and 3 seeds. A reviewer
would ask whether the baseline needed better settings, and whether small differences are
real. This section gives every model family the same tuning effort, picks settings on the
validation slice only, then runs 5 fresh seeds and tests differences across TFs.

### Step 1: equal-budget tuning (validation only)

Random search over optimizer (Adam or SGD for the scRegNet design; Adam for the new families),
learning rate, weight decay, dropout, hidden width, batch size, and the graph-layer width.
Tuning seed 0, no refit, judged on the validation AUROC. The test set played no part.

| Family | Trials | Best validation AUROC | Median | Chosen settings |
|---|---|---|---|---|
| scRegNet design (additive, no split, Geneformer, 3-layer GCN) | 12 + the published recipe | 0.8786 | 0.8737 | Adam, lr 4.3e-5, weight decay 2e-3, dropout 0.48, batch 512 |
| SAGE + interacting + split + Geneformer | 7 | 0.8916 | 0.8901 | Adam, lr 3.4e-5, weight decay 3.8e-3, dropout 0.47, width 64, batch 193 |
| SAGE + interacting + split (GNN only) | 7 | 0.8906 | 0.8877 | Adam, lr 3.3e-4, weight decay 2.2e-5, dropout 0.43, width 256, batch 128 |

The published SGD recipe scored a validation AUROC of 0.8757 as one of the scRegNet trials, below
the best Adam trial. I also ran it as its own entry in the final table.

### Step 2: final runs (5 seeds 42 to 46, refit protocol)

| Model | AUROC | AUPRC | Within-TF | Within-TF (TFs with ≥ 8 positives) |
|---|---|---|---|---|
| scRegNet design, tuned settings | 0.8817 ± 0.0014 | 0.6159 ± 0.0025 | 0.6935 ± 0.0078 | 0.6963 ± 0.0038 |
| scRegNet design, published SGD recipe | 0.8830 ± 0.0012 | 0.6155 ± 0.0029 | 0.7029 ± 0.0063 | 0.7004 ± 0.0030 |
| **SAGE + interacting + split (GNN only)** | 0.8862 ± 0.0026 | 0.6154 ± 0.0077 | **0.7449 ± 0.0109** | 0.7267 ± 0.0054 |
| SAGE + interacting + split + Geneformer | 0.8871 ± 0.0018 | **0.6324 ± 0.0017** | 0.7367 ± 0.0084 | 0.7163 ± 0.0074 |
| &nbsp;&nbsp;hero + co-expression number | 0.8857 ± 0.0040 | 0.6140 ± 0.0117 | 0.7446 ± 0.0117 | 0.7262 ± 0.0046 |
| &nbsp;&nbsp;hero + PANDA number | 0.8867 ± 0.0023 | 0.6161 ± 0.0087 | 0.7454 ± 0.0122 | 0.7284 ± 0.0060 |
| &nbsp;&nbsp;hero + Geneformer + PANDA number | 0.8877 ± 0.0015 | 0.6333 ± 0.0015 | 0.7378 ± 0.0068 | 0.7184 ± 0.0062 |
| ablation: GCN instead of SAGE | 0.8831 ± 0.0035 | 0.6036 ± 0.0074 | 0.7248 ± 0.0056 | 0.7195 ± 0.0087 |
| ablation: additive final layer | 0.8825 ± 0.0024 | 0.6155 ± 0.0051 | 0.7144 ± 0.0043 | 0.7060 ± 0.0053 |
| ablation: hidden-layer final layer | 0.8873 ± 0.0016 | 0.6229 ± 0.0019 | 0.7361 ± 0.0073 | 0.7161 ± 0.0078 |
| ablation: **no edge split** | 0.8405 ± 0.0105 | 0.5277 ± 0.0223 | 0.6669 ± 0.0041 | 0.6509 ± 0.0050 |
| control: scRegNet design, Geneformer vectors shuffled across genes | 0.8798 ± 0.0022 | 0.6051 ± 0.0067 | 0.6906 ± 0.0088 | 0.6822 ± 0.0084 |
| control: hero + Geneformer, vectors shuffled across genes | 0.8823 ± 0.0009 | 0.6207 ± 0.0018 | 0.7074 ± 0.0054 | 0.6998 ± 0.0053 |
| *original scRegNet checkpoint (1 seed, epoch chosen on test)* | *0.8870* | *0.6245* | *0.7215* | |
| *last week, scRegNet + PANDA with symmetric fix (3 seeds)* | *0.8915 ± 0.0004* | *0.6337 ± 0.0009* | *0.7471 ± 0.0009* | |
| *no-neural-network stacker (Section 16)* | *0.8867* | *0.6102* | *0.7399* | |

**5-seed ensemble of the GNN-only hero** (average of each seed's scores, no fitting):
**0.8930 AUROC, 0.6332 AUPRC, 0.7611 within-TF.** This is above last week's best model on AUROC
and within-TF, and level on AUPRC.

### Step 3: paired comparisons across the 33 test TFs

For each TF I computed its AUROC in each model (averaged over the 5 seeds), took the difference
between two models TF by TF, and bootstrapped the mean difference over TFs (5,000 resamples).
"Wilcoxon p" is a signed-rank test on the 33 differences.

| Comparison | Within-TF change | 95% interval | TFs better | Wilcoxon p |
|---|---|---|---|---|
| GNN-only hero vs tuned scRegNet design | **+0.0514** | [+0.0202, +0.0882] | 23/33 | 0.016 |
| GNN-only hero vs published scRegNet recipe | **+0.0419** | [+0.0143, +0.0746] | 20/33 | 0.035 |
| hero + Geneformer vs tuned scRegNet design | +0.0432 | [+0.0150, +0.0755] | 24/33 | 0.007 |
| edge split vs none | **+0.0780** | [+0.0506, +0.1075] | 28/33 | 0.000003 |
| interacting vs additive final layer | +0.0304 | [+0.0026, +0.0625] | 18/33 | 0.47 |
| multiplication vs hidden-layer final layer | +0.0088 | [-0.0030, +0.0214] | 18/33 | 0.36 |
| SAGE vs GCN | +0.0200 | [+0.0013, +0.0409] | 17/33 | 0.15 |
| hero + co-expression number vs hero | -0.0003 | [-0.0025, +0.0017] | 20/33 | 0.63 |
| hero + PANDA number vs hero | +0.0005 | [-0.0019, +0.0027] | 21/33 | 0.21 |
| Geneformer added to the hero | -0.0082 | [-0.0244, +0.0072] | 16/33 | 0.53 |
| hero vs no-neural-network stacker | +0.0050 | [-0.0191, +0.0331] | 15/33 | 0.58 |
| hero vs original scRegNet checkpoint | +0.0247 | [+0.0003, +0.0532] | 19/33 | 0.38 |
| tuned scRegNet design vs original checkpoint | -0.0267 | [-0.0406, -0.0140] | 10/33 | 0.0008 |

### Finding 1: the clear wins are the edge split and the hero over the scRegNet design

- **The edge split is the strongest effect**: +0.078 within-TF, better for 28 of 33 TFs
  (p = 3 × 10⁻⁶). Without it AUROC falls to 0.8405.
- The GNN-only hero beats the tuned scRegNet design by +0.051 within-TF and beats the published
  recipe by +0.042, both significant. Pooled AUROC is higher by +0.0045 and +0.0032; AUPRC is
  level (0.6154 against 0.6159 and 0.6155).
- With Geneformer the hero gains AUPRC (0.6324, +0.017 over the scRegNet design) and keeps
  most of the within-TF gain.

### Finding 2: several things I expected to matter did not reach significance

- **Interacting against additive final layer:** +0.030 within-TF on average, and the bootstrap
  interval stays above zero, but the signed-rank test is not significant (p = 0.47). A few TFs
  gain a lot and many barely change. The form of the interaction does not matter: a hidden layer
  gives within-TF 0.7361 against 0.7449 for multiplication.
- **SAGE against GCN:** +0.020, borderline (interval excludes zero, signed-rank p = 0.15). It
  does not hold up on the other cell types (Section 13).
- **PANDA or the co-expression number:** no change (Section 12).
- **Geneformer on top of the hero:** within-TF is lower by 0.008 (not significant) while AUPRC
  rises by 0.017.

### Finding 3: Geneformer's content matters little for the scRegNet design

Shuffling the Geneformer vectors across genes, so each gene gets another gene's embedding,
changes the scRegNet design by −0.002 AUROC, −0.011 AUPRC and −0.003 within-TF. For the
hero with Geneformer the real vectors help more: +0.005 AUROC, +0.012 AUPRC and +0.029
within-TF over shuffled. So the foundation model's gene-specific content adds a small amount,
mainly for precision, and most of the gain from having the extra input comes from the extra
per-gene parameters.

### Finding 4: my harness reproduces the published recipe, with a known bias in the original

- Running the published SGD recipe in the harness gives 0.8830 / 0.6155 / 0.7029, against
  0.8870 / 0.6245 / 0.7215 for the original checkpoint.
- Choosing the best epoch on the test set, as the original code does, inflates results. On
  my own runs the same models score about +0.0015 AUROC and +0.01 within-TF higher when the
  epoch is chosen on test instead of validation. That accounts for a good part of the gap; the
  original is also a single seed.
- A tuned Adam version of the scRegNet design is not better than the published recipe
  (0.6935 against 0.7029 within-TF).

### Finding 5: ensembling seeds gives a clear gain

The 5-seed ensemble of the hero scores 0.8930 / 0.6332 / 0.7611, against single runs at
0.8862 / 0.6154 / 0.7449. Averaging also cancels the seed-to-seed noise, which is large for the
validation-tuned hero (within-TF spread ± 0.011, against ± 0.002 for the default settings;
see the addendum below).

### Cautions
- The validation-chosen setting for the hero (peak at epoch 4, learning rate 3.3e-4) is less
  stable than the hand-set default. Validation AUROC is dominated by popularity and does not
  reward within-TF ranking. This may explain the poor match.
- The comparison to the original checkpoint rests on a single seed of the original.
- Hyperparameter search budgets are small (7 to 13 trials per family).


### Addendum: the hand-set default settings did better than the validation-tuned ones

Before the tuning, I used one hand-set Adam setting for the hero (learning rate 1e-4, weight
decay 1e-3, dropout 0.17, width 256, batch 193). It came from a small validation sweep on four
configurations. I ran it for 5 fresh seeds (`d01`, `d02`), so the table above has both versions.

| Model | AUROC | AUPRC | Within-TF | Within-TF (≥ 8 positives) |
|---|---|---|---|---|
| hero, default settings (GNN only) | 0.8884 ± 0.0018 | 0.6256 ± 0.0033 | 0.7471 ± 0.0047 | 0.7266 ± 0.0062 |
| hero + Geneformer, default settings | **0.8910 ± 0.0015** | **0.6346 ± 0.0038** | **0.7517 ± 0.0027** | 0.7322 ± 0.0046 |
| 5-seed ensemble, default hero | 0.8913 | 0.6335 | 0.7536 | |
| 5-seed ensemble, default hero + Geneformer | 0.8935 | 0.6410 | 0.7574 | |
| *original scRegNet checkpoint* | *0.8870* | *0.6245* | *0.7215* | |
| *last week's scRegNet + PANDA, symmetric fix* | *0.8915 ± 0.0004* | *0.6337 ± 0.0009* | *0.7471 ± 0.0009* | |

- **Default hero + Geneformer matches last week's best model with no PANDA**: AUROC 0.8910
  against 0.8915, AUPRC 0.6346 against 0.6337, within-TF 0.7517 against 0.7471.
- Paired over TFs: default hero beats the published recipe by +0.044 within-TF (p = 0.006).
  Default hero + Geneformer beats it by +0.049 (p = 0.001) and beats the original checkpoint by
  +0.032 (interval [+0.009, +0.060], p = 0.03). Default hero alone against the original
  checkpoint: +0.027, p = 0.10. Against the five-feature stacker: +0.007 and +0.012, neither
  significant (p = 0.69 and 0.65).
- Default against validation-tuned hero: +0.002 within-TF, p = 0.50. The two are not reliably
  different on test, but the tuned one is noisier from seed to seed (±0.011 against ±0.005).
- **Geneformer on top of the default hero:** +0.005 within-TF, p = 0.07, and +0.009 AUPRC. With
  the default settings Geneformer helps a little; with the tuned settings it did not. Geneformer's
  contribution is small and depends on the settings.
- **Why tuning did not help:** the search judged models by validation AUROC. That score is
  dominated by popularity and does not reward within-TF ranking, and 7 trials is a small
  search. I kept the pre-specified tuned results in the table above and report the default
  results here, instead of choosing between them after seeing test numbers.

---

## 15. Held-out genes and held-out TFs: what happens when popularity cannot help

**Scripts:** `src/pair_gnn.py --protocol {gene,tf} --fold k`, `analysis/heldout_report.py`
(output `analysis/out/heldout_report.txt`), jobs in `analysis/jobs/heldout.txt`.

### Why

The literature review's most important evaluation advice (Park and Marcotte, *Nature Methods*
2012) is to test on items the model never saw. In the standard split, every test gene and every
test TF also appears in training with other labels, so a model can win by remembering how
popular each one is. Two stricter tests remove that:

- **Held-out genes:** 20% of target genes lose *all* their edges (from training and from the
  graph). Five different random draws (folds). A held-out gene has no popularity to use.
- **Held-out TFs:** about 7 of the 34 TFs lose all their edges, in 5 folds so each TF is held
  out exactly once. A held-out TF has no known targets, so there is no "program" to average.

In both tests the epoch is chosen on a validation slice built the same way (unseen genes or
unseen TFs from inside the training data). Each model runs on 5 folds × 2 seeds with the settings
chosen in Section 14, then is scored once on the held-out items.

For reference I also scored popularity alone, using only what training could know: the TF's
base rate (held-out genes) or the gene's degree (held-out TFs).

### Result

**Held-out genes** (mean ± spread over 5 folds):

| Model | AUROC | AUPRC | Within-TF | Within-TF (≥ 8 positives) |
|---|---|---|---|---|
| popularity only: TF base rate | 0.8493 ± 0.0029 | 0.4920 ± 0.0164 | 0.5000 | 0.5000 |
| scRegNet design, tuned | 0.8328 ± 0.0101 | 0.5275 ± 0.0179 | 0.5241 ± 0.0403 | 0.5591 ± 0.0259 |
| GNN-only hero | 0.8360 ± 0.0074 | 0.5049 ± 0.0123 | 0.5477 ± 0.0341 | 0.5583 ± 0.0222 |
| hero + co-expression number | 0.8343 ± 0.0083 | 0.5046 ± 0.0115 | 0.5456 ± 0.0333 | 0.5572 ± 0.0211 |
| hero + Geneformer | 0.8465 ± 0.0086 | 0.5314 ± 0.0124 | 0.5521 ± 0.0301 | 0.5697 ± 0.0244 |

**Held-out TFs** (mean ± spread over 5 folds):

| Model | AUROC | AUPRC | Within-TF | Within-TF (≥ 8 positives) |
|---|---|---|---|---|
| popularity only: gene degree | 0.6689 ± 0.0168 | 0.2316 ± 0.0299 | 0.7244 ± 0.0643 | 0.7120 ± 0.0379 |
| scRegNet design, tuned | 0.6009 ± 0.0913 | 0.2303 ± 0.0815 | 0.7163 ± 0.0609 | 0.7045 ± 0.0347 |
| GNN-only hero | 0.6241 ± 0.0672 | 0.2070 ± 0.0589 | 0.6839 ± 0.0521 | 0.6691 ± 0.0281 |
| hero + co-expression number | 0.6356 ± 0.0569 | 0.2083 ± 0.0543 | 0.6932 ± 0.0574 | 0.6823 ± 0.0379 |
| hero + Geneformer | 0.6058 ± 0.0538 | 0.1846 ± 0.0516 | 0.6126 ± 0.0934 | 0.6025 ± 0.0772 |

Paired over the 29 held-out TFs with at least 8 test positives (per-TF AUROC, averaged over the
2 seeds):

| Comparison | Mean difference | 95% interval | TFs better | Wilcoxon p |
|---|---|---|---|---|
| hero vs scRegNet design | -0.0348 | [-0.0573, -0.0089] | 8/29 | 0.006 |
| hero vs popularity (gene degree) | -0.0422 | [-0.0613, -0.0209] | 7/29 | 0.0007 |
| scRegNet design vs popularity | -0.0074 | [-0.0166, +0.0012] | 8/29 | 0.066 |
| hero + co-expression number vs popularity | -0.0295 | [-0.0475, -0.0119] | 7/29 | 0.001 |
| hero + Geneformer vs popularity | -0.1106 | [-0.1364, -0.0846] | 1/29 | 0.0000002 |

### Finding 1: for a gene the model has never seen, none of the models ranks targets well

Within-TF AUROC is 0.52 to 0.55, where 0.5 is chance. The pooled AUROC of 0.83 to 0.85 comes
mostly from the TF's base rate. That alone gives 0.849, which is higher than every model,
including the best one, hero + Geneformer (0.8465). Among the models, hero + Geneformer is best
on every measure, which fits Geneformer being the one input that carries information about a
gene from outside this dataset. Even so, 0.57 within-TF is weak.

### Finding 2: for a TF with no known targets, the interacting hero is worse than popularity

- Ranking genes by how many *other* TFs bind them gives within-TF 0.724 on held-out TFs. No
  model beats it.
- The scRegNet design is close to popularity (-0.007, not significant), which is what you would
  expect from a model that gives every TF the same gene ranking.
- The hero is significantly worse than both (-0.035 against the scRegNet design, -0.042 against
  popularity). With no known targets, a TF's embedding reduces to its own expression. The product
  with each gene then has nothing meaningful to compare, and the interaction term adds noise.
- Hero + Geneformer is much worse (-0.111): the extra per-gene input lets the interaction
  overfit to the training TFs.
- The spreads between folds are large (±0.05 to ±0.09) because each fold has only about 7 TFs,
  so only the paired tests above are reliable.

### What this means

- **The hero's gains are gains at completing partly known target lists.** That is what the
  standard benchmark measures, and it is a real use: given a TF with some known targets, rank the
  rest. The gains do not show that the model learned regulation that transfers to a new TF or
  a new gene.
- **No tested model beats popularity on new TFs, and none ranks new genes well.** A
  paper needs to state this limit. It matches the biology of the ground truth:
  ChIP-seq peaks concentrate on open, highly bound genes, and without anything TF-specific
  to go on, "bound by many TFs" is the best guess.
- A model should fall back on popularity when a TF has no known targets. One fix is to
  train with whole TFs hidden from the graph. That test is in the addendum below.


### Addendum: hiding whole TFs during training does not fix it

I added one training trick aimed at the held-out-TF weakness: in every batch, with probability
0.3 per TF, hide *all* of that TF's training edges from the graph (`--tf_drop 0.3`). The model
then often has to score a TF whose program is empty and should learn to fall back on
popularity. I ran it with the default hero settings, on the held-out-TF test and on the
standard split, with a matched control of the same hero without it (5 folds × 2 seeds).

| Setting | Model | AUROC | AUPRC | Within-TF | Within-TF (≥ 8 positives) |
|---|---|---|---|---|---|
| held-out TFs | default hero | 0.6308 ± 0.1085 | 0.2294 ± 0.0934 | 0.6840 ± 0.0630 | 0.6745 ± 0.0358 |
| held-out TFs | default hero + TF dropout | 0.6488 ± 0.0518 | 0.2409 ± 0.0716 | 0.6598 ± 0.0323 | 0.6584 ± 0.0171 |
| held-out TFs | *popularity only* | *0.6689 ± 0.0168* | *0.2316 ± 0.0299* | *0.7244 ± 0.0643* | *0.7120 ± 0.0379* |
| standard split (5 seeds) | default hero | 0.8884 ± 0.0018 | 0.6256 ± 0.0033 | 0.7471 ± 0.0047 | |
| standard split (5 seeds) | default hero + TF dropout | 0.8847 ± 0.0024 | 0.6203 ± 0.0048 | 0.7303 ± 0.0120 | |

- On held-out TFs, TF dropout raises the pooled AUROC a little (0.631 to 0.649) and lowers the
  within-TF AUROC (0.684 to 0.660). It still does not reach popularity (0.724). I count this as
  no improvement.
- On the standard split it costs 0.004 AUROC and 0.017 within-TF.
- So this simple fix is a **negative result**. A model that handles unseen TFs well probably
  needs different information about the TF, for example its DNA-binding domain or motif.

---

## 16. Ideas from computer-science link prediction (advisor ask 1), and a biology check

**Scripts:** `analysis/cs_baselines.py` (outputs `analysis/out/cs_baselines_<cell>.json`),
`analysis/topk_ezh2.py` (output `analysis/out/topk_ezh2.txt`). The literature review is
`LINK_PREDICTION_REVIEW.md`.

### Why

Your advisor asked me to check link prediction in computer science and apply it here. The review
found that our task is matrix completion on a 34 × 910 yes/no table, and that the field has
strong, simple methods for it. Before using heavy models, the review's first recommendation was to
run those simple methods, because "tuned simple baselines often match complex ones".

### What I ran (no neural network, CPU only)

I compute everything from the table of **training** edges. A test pair's own edge is not in
the table, and for the training rows used to fit the small combiners, every score comes from
5-fold out-of-fold tables that drop that fold's positives.

- **Preferential attachment (popularity):** how many targets the TF has times how many TFs bind
  the gene. This is the two-number model from Week 1.
- **TF-based collaborative filtering:** do TFs that share targets with this TF also target the
  gene? **Gene-based collaborative filtering:** does the gene look like the TF's known targets,
  judged by which TFs bind each?
- **Three-step paths and resource allocation:** counts of paths TF – gene' – TF' – gene, with
  popular genes down-weighted. This is the bipartite version of Katz and Adamic-Adar.
- **The co-expression number** from Sections 2 and 12, and the **PANDA number**.
- **Stackers:** a logistic regression (and a small boosted-tree model) over the features above.

### Result on hESC

| Method | AUROC | AUPRC | Within-TF |
|---|---|---|---|
| preferential attachment (popularity only) | 0.8802 | 0.5898 | 0.7068 |
| TF-based collaborative filtering | 0.6949 | 0.2412 | 0.6878 |
| gene-based collaborative filtering | 0.6142 | 0.1806 | 0.6394 |
| three-step path count | 0.8547 | 0.5351 | 0.6698 |
| resource allocation over paths | 0.6749 | 0.2309 | 0.6639 |
| stacker (logistic regression): popularity | 0.8797 | 0.5949 | 0.7068 |
| &nbsp;&nbsp;+ both collaborative filters | 0.8804 | 0.5930 | 0.7181 |
| &nbsp;&nbsp;+ three-step paths and resource allocation | 0.8837 | 0.6064 | 0.7253 |
| &nbsp;&nbsp;+ co-expression number | 0.8863 | 0.6090 | 0.7375 |
| &nbsp;&nbsp;+ PANDA number | 0.8867 | 0.6102 | 0.7399 |
| stacker (boosted trees): + co-expression and PANDA numbers | 0.8871 | 0.6171 | 0.7279 |

### Result on all four cell types: popularity versus the original scRegNet

| Cell type | Original scRegNet checkpoint | Popularity only | Stacker with co-expression number |
|---|---|---|---|
| hESC | 0.8874 / 0.6225 / 0.7202 | 0.8797 / 0.5949 / 0.7068 | 0.8863 / 0.6090 / 0.7375 |
| hHEP | 0.9027 / 0.8419 / 0.7882 | 0.8985 / 0.8270 / 0.7838 | 0.8996 / 0.8302 / 0.7593 |
| mHSC-E | 0.9223 / 0.9402 / 0.7339 | 0.9147 / 0.9322 / 0.7132 | 0.9139 / 0.9301 / 0.7420 |
| mESC | 0.9260 / 0.8593 / 0.7660 | 0.9232 / 0.8505 / 0.7612 | **0.9312 / 0.8601 / 0.8142** |

(Each cell: AUROC / AUPRC / within-TF. Original checkpoints chose their epoch on the test set.)

### Finding 1: the popularity result holds on all four cell types

Two numbers per pair, with no expression data, come within 0.003 to 0.008 AUROC of the original
scRegNet on all four cell types (0.0077 hESC, 0.0042 hHEP, 0.0076 mHSC-E, 0.0028 mESC). The
benchmark rewards popularity everywhere we looked.

### Finding 2: the useful extra ingredient is the co-expression number

- Collaborative filtering and path counts add little, as the review predicted: +0.011 and then
  +0.007 within-TF on hESC, in two steps.
- The co-expression number adds the most: within-TF +0.012 on hESC, +0.012 on mESC, +0.028 on
  mHSC-E. It *hurts* on hHEP (−0.023).
- A stacker made of five hand-built features **beats the original scRegNet checkpoint** on mESC
  on all three metrics (0.9312 / 0.8601 / 0.8142 against 0.9260 / 0.8593 / 0.7660), and on
  within-TF on hESC and mHSC-E.

### Finding 3: on hESC the GNN hero and the stacker are statistically tied

The paired test over 33 TFs gives hero minus stacker = +0.005 within-TF, interval
[-0.019, +0.033], p = 0.58. The hero is a little higher on average and the ensemble of 5 hero
runs is higher again (0.7611 within-TF), but a transparent model built from five features
carries most of the GNN's gain. The review warned about this outcome, and any write-up should
report it.

### Finding 4: what I did not run

I did not implement the pair-aware graph networks from the review (SEAL, NBFNet, neural common
neighbours). The path and collaborative-filtering signals they build on added at
most +0.011 within-TF in the stacker, so I judged the expected return low. Whether they help
is still untested.

### A biology check on the hESC predictions: who is at the top of each TF's list?

A biologist reads the top of one TF's candidate list. I measured precision at the top within each
TF's test candidates (TFs with at least k test positives):

| k | TFs | Base rate | scRegNet design | GNN-only hero | hero + Geneformer | hero + PANDA |
|---|---|---|---|---|---|---|
| 10 | 20 | 0.239 | 0.415 | 0.480 | 0.481 | 0.500 |
| 20 | 15 | 0.297 | 0.510 | 0.571 | 0.562 | 0.569 |
| 50 | 10 | 0.384 | 0.601 | 0.607 | 0.622 | 0.610 |

The hero is better at the top of the list (+0.065 at k = 10, +0.061 at k = 20) and the gap
closes by k = 50.

**EZH2 again (Polycomb repressor).** EZH2 has 77 training targets and 27 held-out real targets
among 210 test candidates. Among its top 30 test candidates:

| Model | Held-out EZH2 targets recovered (chance level 3.9) |
|---|---|
| scRegNet design, tuned | 3 |
| GNN-only hero | **8** |
| hero + Geneformer | 8 |
| hero + PANDA | 8 |

The scRegNet design recovers EZH2's targets at chance level, as last week's finding that its
ranking for EZH2 is below chance suggested. The hero recovers about twice as many. A GO test on
each model's top 60 new predictions for EZH2 (the gene panel as background) finds no
significant term for any model (best adjusted p between 0.22 and 0.32). The hero's top terms are
developmental (anterior/posterior pattern specification, cardiac muscle tissue development,
extracellular matrix organization). This fits what Polycomb silences, but it only suggests a
link.

---

****** 18 Oct ******


## 17. Week 3 (23 Oct, hESC): PANDA as the graph, a neural PANDA encoder, and a context-specific stacker

### Setup

Three questions were tested on hESC. First: what happens when the PANDA network replaces the
ChIP-seq GRN as the graph that the GNN passes messages over? Second: does a GNN built from the three
PANDA ingredients (known targets, co-expression, protein-protein interaction) work better than the plain hero? Third: do the
stacker features gain from running on the PANDA graph or on a co-expression graph instead of only the ChIP-seq label graph?

All new GNN rows use 5 seeds (42 to 46), the hero optimiser settings, epoch chosen on validation AUROC, and a refit
on all training rows. The reference is the hero (GNN, one SAGE layer, Hadamard decoder, no Geneformer). The hero and the hero with Geneformer
were re-run on the GPU so that every row comes from one device. Each paired test is the usual one: per-TF AUROC averaged over
seeds, Wilcoxon over the 33 test TFs. One run takes 0.3 min for the hero and about 0.9 min for the neural PANDA encoder.

| Reference | AUROC | AUPRC | within-TF |
|---|---|---|---|
| hero | 0.8884 ± 0.0018 | 0.6255 ± 0.0032 | 0.7470 ± 0.0046 |
| hero + Geneformer | 0.8907 ± 0.0014 | 0.6345 ± 0.0039 | 0.7510 ± 0.0025 |

**Headline: no new variant beat the hero on within-TF AUROC.** The closest was the neural PANDA encoder with the Hadamard decoder, which
was level on AUROC and 0.006 higher on AUPRC. Details follow.

### Part 1: the PANDA top-K graph as the message-passing graph

For every label TF, the K genes with the highest PANDA score were joined to it. The graph is undirected, as the GRN graph is.
The `union` variant uses the GRN training edges plus the PANDA edges.

**Why a prior can leak.** PANDA is built from a prior that contains the training positives. A training pair (TF, gene) whose edge is in the prior
gets a high PANDA score, so the gene tends to land in the TF's top-K list. The GNN would then pass messages along an edge that exists because of the
pair's own label. At test time no such edge exists. A model that learns this shortcut does well in training and badly on new pairs.

**The safe version.** The training rows were cut into the same five folds that the PANDA scalar cache uses. For fold k, a PANDA with fold k's positives
removed from the prior was built. Each training batch holds rows of one fold only and uses that fold's PANDA graph. Validation and test rows use the
full PANDA graph. The naive version (`--panda_safe 0`) uses the full graph for all training batches and exists to measure the leak.

**Leak measurement (K = 100).** Of the training positives, 55.6% sit inside the full PANDA graph, but only 3.3% sit inside the graph of their own fold.
The own-fold figure matches the test figure (3.2%), so the safe graph looks to the model in training like the graph it will see at test time.

| K | Edges | Test positives inside the graph | Share of test pairs in the graph that are positive | Training positives inside, full graph | Training positives inside, own-fold graph |
|---|---|---|---|---|---|
| 50 | 1700 | 0.7% | 4.7% | 35.4% | 1.0% |
| 100 | 3400 | 3.2% | 8.3% | 55.6% | 3.3% |
| 200 | 6800 | 10.1% | 10.4% | 85.7% | 12.7% |

Half of the K = 100 PANDA graph is simply the training positives: 1685 of its 3400 edges. The PANDA graph therefore adds only about 1700 new edges, and these are depleted of true
targets (8.3% of the test pairs inside the graph are positive, against a base rate of 14.9% over all test pairs).

The leak makes the test results worse, not better. A model trained on the naive graph learns to lean on graph membership: in training, a positive pair is
usually joined by an edge, and a negative pair usually is not. Test pairs never have this property, so the shortcut fails there. The leak therefore shows up as a train/test mismatch,
and this is why a prior built from training labels must be built out-of-fold. The cost, naive against safe with the same K and seeds:

| Graph (K = 100) | AUROC | AUPRC | within-TF | Naive minus safe, within-TF |
|---|---|---|---|---|
| PANDA only, safe | 0.8822 ± 0.0022 | 0.6117 ± 0.0040 | 0.7231 ± 0.0096 | |
| PANDA only, naive | 0.8659 ± 0.0017 | 0.5847 ± 0.0037 | 0.6918 ± 0.0059 | −0.0313 (p = 0.004) |
| Union, safe | 0.8836 ± 0.0019 | 0.6151 ± 0.0035 | 0.7298 ± 0.0091 | |
| Union, naive | 0.8730 ± 0.0021 | 0.5974 ± 0.0063 | 0.7110 ± 0.0095 | −0.0187 (p = 0.010) |

The naive runs stop at epoch 2 to 3 against 4 to 15 for the safe runs. The validation rows are clean, so their AUROC drops as the
model learns the shortcut. The loss is −0.016 AUROC for PANDA only and −0.011 for the union, in all five seeds.

**Safe PANDA graph against the hero.**

| Model | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|
| hero (GRN graph) | 0.8884 | 0.6255 | 0.7470 | | |
| PANDA K = 50 | 0.8811 | 0.6058 | 0.7133 | −0.0337 | 0.004 |
| PANDA K = 100 | 0.8822 | 0.6117 | 0.7231 | −0.0239 | 0.009 |
| PANDA K = 200 | 0.8838 | 0.6110 | 0.7218 | −0.0252 | 0.080 |
| GRN + PANDA K = 100 | 0.8836 | 0.6151 | 0.7298 | −0.0172 | 0.009 |

All four are worse than the hero in all five seeds on AUROC and AUPRC. Replacing the GRN with PANDA does not help, and adding PANDA to the GRN does not help either.
Validation AUROC picked K = 200 and the union (both 0.8846). Both were re-run with Geneformer:

| With Geneformer | AUROC | AUPRC | within-TF | Within-TF diff vs hero + Geneformer | p |
|---|---|---|---|---|---|
| hero + Geneformer | 0.8907 | 0.6345 | 0.7510 | | |
| PANDA K = 200 | 0.8852 | 0.6208 | 0.7209 | −0.0301 | 0.018 |
| GRN + PANDA K = 100 | 0.8852 | 0.6217 | 0.7308 | −0.0202 | 0.002 |

The low recall of the graph probably explains the loss. Just 3.2% of test positives are inside the K = 100 graph, so most messages carry no
information about the pair being scored, while the GRN graph carries the known targets of each TF.

### Part 2: the neural PANDA encoder

Each layer computes `h' = lin_self(h) + SAGE_R(h, R) + SAGE_C(h, C) + SAGE_P(h, P)`. R is the training-positive GRN graph (with the usual
per-batch split), C is a label-free gene-gene kNN graph from co-expression (k = 10, positive Pearson correlations, symmetrised), and P is the STRING
TF-TF graph. C and P are built without labels. R contains no test pair in the (TF, gene) direction. The two-layer model lets
layer 1 give each TF its targets and layer 2 pass a partner TF's targets over P, which mirrors PANDA's responsibility term. Two new decoders were added:
`tani` (six-head Tanimoto similarity of the two embeddings, together with the embeddings) and `tani_pure` (Tanimoto terms only).

| Model | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|
| hero (Hadamard decoder) | 0.8884 ± 0.0018 | 0.6255 ± 0.0032 | 0.7470 ± 0.0046 | | |
| hero with Tanimoto decoder | 0.8837 ± 0.0011 | 0.6205 ± 0.0037 | 0.7174 ± 0.0038 | −0.0296 | 0.097 |
| neural PANDA RCP, Tanimoto | 0.8852 ± 0.0009 | 0.6210 ± 0.0054 | 0.7180 ± 0.0061 | −0.0290 | 0.042 |
| without C (RP) | 0.8852 ± 0.0009 | 0.6240 ± 0.0017 | 0.7136 ± 0.0049 | −0.0334 | 0.074 |
| without P (RC) | 0.8839 ± 0.0009 | 0.6190 ± 0.0057 | 0.7162 ± 0.0056 | −0.0308 | 0.017 |
| without R (CP) | 0.8825 ± 0.0010 | 0.6197 ± 0.0040 | 0.7108 ± 0.0045 | −0.0362 | 0.074 |
| one layer | 0.8824 ± 0.0016 | 0.6146 ± 0.0082 | 0.7145 ± 0.0025 | −0.0325 | 0.053 |
| metacell C | 0.8853 ± 0.0008 | 0.6235 ± 0.0044 | 0.7123 ± 0.0044 | −0.0347 | 0.037 |
| `tani_pure` decoder | 0.8337 ± 0.0245 | 0.5105 ± 0.0959 | 0.6669 ± 0.0140 | −0.0801 | 1e-6 |
| RCP + Geneformer, Tanimoto | 0.8858 ± 0.0009 | 0.6266 ± 0.0032 | 0.7192 ± 0.0033 | −0.0318 (vs hero + Geneformer) | 0.096 |
| **RCP, Hadamard decoder** | 0.8898 ± 0.0014 | 0.6315 ± 0.0033 | 0.7410 ± 0.0079 | −0.0060 | 0.86 |

Three readings follow.

- The Tanimoto decoder is the main cause of the drop. The hero with only the decoder swapped loses 0.030 within-TF, the same as the full neural PANDA with that decoder.
  The decoder with Tanimoto terms alone (the PANDA-like form without popularity terms) is 0.08 worse within-TF and unstable (AUROC 0.834 ± 0.025). Popularity terms
  are needed, as in earlier weeks.
- With the Hadamard decoder, neural PANDA RCP is level with the hero on AUROC (+0.0014, 4 of 5 seeds higher), higher on AUPRC (+0.0060, 5 of 5 seeds), and lower
  on within-TF (−0.0060, p = 0.86, no real difference). This is not a win on the TF-specific ranking, which is the measure of interest.
- Removing any single relation changed little, and the ablation differences are smaller than the seed noise on within-TF.

An added control (not in the original plan) asked whether the Hadamard gain comes from the PANDA relations or only from the second layer:

| Hadamard decoder | AUROC | AUPRC | within-TF |
|---|---|---|---|
| hero, one layer | 0.8884 ± 0.0018 | 0.6255 ± 0.0032 | 0.7470 ± 0.0046 |
| hero, two SAGE layers | 0.8866 ± 0.0022 | 0.6282 ± 0.0032 | 0.7333 ± 0.0104 |
| R only | 0.8863 ± 0.0013 | 0.6278 ± 0.0025 | 0.7306 ± 0.0025 |
| RC | 0.8887 ± 0.0016 | 0.6297 ± 0.0024 | 0.7359 ± 0.0039 |
| RP | 0.8883 ± 0.0009 | 0.6299 ± 0.0010 | 0.7370 ± 0.0047 |
| CP | 0.8888 ± 0.0008 | 0.6268 ± 0.0027 | 0.7297 ± 0.0053 |
| RCP | 0.8898 ± 0.0014 | 0.6315 ± 0.0033 | 0.7410 ± 0.0079 |

Against the matched two-layer controls (hero with two layers and R only), adding C and P raises AUROC and AUPRC by about 0.003 each, which is small and
close to the seed spread. Within-TF stays below the one-layer hero in every row. Ten variants were compared in all, so a gain of this size would
need confirmation on other cell types before it is believed.

### Part 3: the stacker on other graphs

The stacker features (popularity, user-CF, item-CF, 3-paths, resource allocation) were also computed on the binarised PANDA top-100 graph and, for
co-expression, on the label-free kNN graph (Pearson correlation, common neighbours, Adamic-Adar, degree of the gene and the TF, and item-CF over the
co-expression graph with the known targets taken out-of-fold). Test rows use the full PANDA. Training rows use the PANDA of their own fold, so no row sees its own
edge. The numbers for the existing stacks were reproduced to four decimals.

| Stack (test set, one fit) | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|
| existing best: label + contrast + PANDA score, GBM | 0.8871 | 0.6171 | 0.7279 | −0.0191 | 0.36 |
| same features, LR | 0.8867 | 0.6102 | 0.7399 | −0.0071 | 0.68 |
| label + PANDA graph, LR | 0.8881 | 0.6181 | 0.7385 | −0.0085 | 0.58 |
| label + PANDA graph, GBM | 0.8882 | 0.6161 | 0.7326 | −0.0144 | 0.96 |
| label + co-expression graph, LR | 0.8851 | 0.6097 | 0.7110 | −0.0360 | 0.63 |
| all features, LR | 0.8893 | 0.6194 | 0.7423 | −0.0047 | 0.82 |
| all features, GBM | 0.8889 | 0.6199 | 0.7270 | −0.0200 | 0.83 |
| PANDA graph only, GBM | 0.8423 | 0.5369 | 0.6560 | −0.0910 | 8e-5 |
| co-expression graph only, GBM | 0.7995 | 0.4342 | 0.5324 | −0.2146 | 7e-8 |
| context only (PANDA graph + score + co-expression), GBM | 0.8586 | 0.5611 | 0.6673 | −0.0797 | 2e-4 |

Findings:

- The PANDA-graph features add nothing beyond the PANDA score and the contrast that the old best stack already holds. Label + PANDA graph with LR gives within-TF 0.7385 against 0.7399 for the old best with LR;
  with GBM it gives 0.7326 against 0.7279 (paired p = 0.54). The full feature set reaches 0.8893 AUROC, the highest stacker value so far, but the difference to the old best is within noise
  (within-TF +0.014 against the GBM baseline, p = 0.07, and +0.002 against the LR baseline).
- Co-expression-graph features do not help the label stack and lower within-TF (0.7110 against 0.7253 for the label-only LR stack). Their single-feature AUROC is near or below 0.5, except item-CF (0.644, but within-TF 0.52).
- A stacker without any ChIP-seq graph features reaches 0.859 AUROC but only 0.667 within-TF. Popularity of the gene in the label graph is what carries the AUROC, and the PANDA graph
  alone cannot replace it. On the TF-specific ranking the context-only stack is 0.080 below the hero (p = 2e-4).
- No stack beats the hero on within-TF. The stacker rows are one fit each (the stacker is deterministic), so the p-values compare it with the hero's 5-seed per-TF mean.

### What this adds up to

On hESC, PANDA information as a graph does not improve the hero. The PANDA top-K graph is a weak graph (few test positives inside it) and a leaky one if built naively.
The leak was measured: 56% against 3% of training positives inside the graph, and −0.016 AUROC on test. The safe construction removes it, but the safe graph still
loses to the hero. The neural PANDA encoder is level with the hero on pooled metrics only when the Hadamard decoder is kept. Earlier weeks found that PANDA helps as a number
fed to the decoder, and this week's results fit that finding.

### Limits

- hESC only. Other cell types were not run.
- In the undirected GRN graph, 30 of the 7142 test pairs (7 of 1061 test positives) have their reverse (gene, TF) direction as a training edge. This holds for the hero and for
  every row above, so it does not change the comparison, but it is a small shared leak of the benchmark.
- Training batches in the safe version are interleaved across folds (each batch is from one fold, but batches from different folds are shuffled together).
- The Part 2b control was added after seeing the Part 2 results, so it was not planned in advance.
- Files: `analysis/out/w3_report_hESC.txt` (all tables, same-seed differences, per-TF tests), `analysis/w3_graph_recall.py`, `analysis/stacker_graphs.py`, `analysis/w3_report.py`.


## 18. Week 3, part 5 (23 Oct): the same questions on hHEP, mESC and mHSC-E

### Setup

STRING protein-protein edges were built for the three new cell types with `make_ppi.py`, which calls the STRING v12 API (functional network, score of at least 0.7),
the same call and threshold as the hESC file. hHEP uses human STRING (species 9606), mESC and mHSC-E use mouse STRING (10090). The data use upper-case gene symbols, which STRING matches to its
mouse proteins by alias; edges are written under the dataset symbol. The edge counts are 4486 (hHEP), 3862 (mESC) and 1504 (mHSC-E), against 2296 for hESC. The PANDA and contrast caches were then
built for each cell type (`analysis/scalars.py`), and the PANDA test AUROC of the scalar alone is 0.743 (hHEP), 0.902 (mESC) and 0.708 (mHSC-E).

Each cell type got the rows of the hESC study: hero, hero + Geneformer, GRN + PANDA K = 100 (safe, union), and the neural PANDA encoder with the Hadamard decoder and with the
Tanimoto decoder. Each model ran with 5 seeds (42 to 46) on the GPU. The scRegNet published recipe (rule A: epoch chosen on validation AUROC) also ran for each cell type, and the stacker with PANDA-graph
and co-expression-graph features was fitted once per cell type. The hero settings come from hESC and were not re-tuned here, so a gap to a model tuned elsewhere can reflect that choice.
Paired tests are the same as before: per-TF AUROC averaged over seeds, Wilcoxon over the test TFs (30 in hHEP, 85 in mESC, 29 in mHSC-E, 33 in hESC) against that cell type's hero.
The hESC tables from section 17 are repeated for comparison; the hESC published-recipe row is the earlier `f02_A_paper` run (5 seeds, rule A).

### Results per cell type

AUROC, AUPRC and within-TF are mean ± std over seeds, from the refit models. The last two columns give the within-TF difference to the hero of the same cell type and the Wilcoxon p-value.

**hESC**

| Model | Seeds | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|---|
| hero | 5 | 0.8884 ± 0.0018 | 0.6255 ± 0.0032 | 0.7470 ± 0.0046 |  |  |
| hero + Geneformer | 5 | 0.8907 ± 0.0014 | 0.6345 ± 0.0039 | 0.7510 ± 0.0025 | +0.0040 | 0.0864 |
| GRN + PANDA K=100 safe | 5 | 0.8836 ± 0.0019 | 0.6151 ± 0.0035 | 0.7298 ± 0.0091 | -0.0172 | 0.00879 |
| neural PANDA RCP, Hadamard | 5 | 0.8898 ± 0.0014 | 0.6315 ± 0.0033 | 0.7410 ± 0.0079 | -0.0060 | 0.86 |
| neural PANDA RCP, Tanimoto | 5 | 0.8852 ± 0.0009 | 0.6210 ± 0.0054 | 0.7180 ± 0.0061 | -0.0290 | 0.0424 |
| scRegNet published recipe | 5 | 0.8830 ± 0.0012 | 0.6155 ± 0.0029 | 0.7029 ± 0.0063 | -0.0441 | 0.00516 |
| stacker, old best (GBM) | 1 fit | 0.8871 | 0.6171 | 0.7279 | -0.0191 | 0.358 |
| stacker, all features (LR) | 1 fit | 0.8893 | 0.6194 | 0.7423 | -0.0047 | 0.819 |

**hHEP**

| Model | Seeds | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|---|
| hero | 5 | 0.8993 ± 0.0007 | 0.8384 ± 0.0010 | 0.7599 ± 0.0022 |  |  |
| hero + Geneformer | 5 | 0.9027 ± 0.0007 | 0.8430 ± 0.0018 | 0.7800 ± 0.0046 | +0.0201 | 0.00578 |
| GRN + PANDA K=100 safe | 5 | 0.8970 ± 0.0008 | 0.8370 ± 0.0028 | 0.7549 ± 0.0061 | -0.0049 | 0.0879 |
| neural PANDA RCP, Hadamard | 5 | 0.9024 ± 0.0005 | 0.8437 ± 0.0006 | 0.7744 ± 0.0070 | +0.0145 | 0.000418 |
| neural PANDA RCP, Tanimoto | 5 | 0.8999 ± 0.0011 | 0.8372 ± 0.0028 | 0.7752 ± 0.0014 | +0.0154 | 0.556 |
| scRegNet published recipe | 5 | 0.9023 ± 0.0003 | 0.8413 ± 0.0006 | 0.7870 ± 0.0011 | +0.0272 | 0.0841 |
| stacker, old best (GBM) | 1 fit | 0.8969 | 0.8147 | 0.7668 | +0.0069 | 0.14 |
| stacker, all features (LR) | 1 fit | 0.8993 | 0.8288 | 0.7586 | -0.0013 | 0.7 |

**mESC**

| Model | Seeds | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|---|
| hero | 5 | 0.9393 ± 0.0009 | 0.8819 ± 0.0015 | 0.8235 ± 0.0013 |  |  |
| hero + Geneformer | 5 | 0.9401 ± 0.0005 | 0.8838 ± 0.0006 | 0.8245 ± 0.0020 | +0.0010 | 0.465 |
| GRN + PANDA K=100 safe | 5 | 0.9374 ± 0.0008 | 0.8790 ± 0.0016 | 0.8133 ± 0.0028 | -0.0101 | 0.00081 |
| neural PANDA RCP, Hadamard | 5 | 0.9411 ± 0.0007 | 0.8861 ± 0.0014 | 0.8250 ± 0.0034 | +0.0015 | 0.00645 |
| neural PANDA RCP, Tanimoto | 5 | 0.9243 ± 0.0008 | 0.8554 ± 0.0026 | 0.7618 ± 0.0037 | -0.0617 | 1.85e-08 |
| scRegNet published recipe | 5 | 0.9254 ± 0.0001 | 0.8587 ± 0.0001 | 0.7657 ± 0.0002 | -0.0578 | 7.26e-08 |
| stacker, old best (GBM) | 1 fit | 0.9294 | 0.8546 | 0.7956 | -0.0279 | 6.39e-06 |
| stacker, all features (LR) | 1 fit | 0.9349 | 0.8684 | 0.8042 | -0.0192 | 0.0218 |

**mHSC-E**

| Model | Seeds | AUROC | AUPRC | within-TF | Within-TF diff vs hero | p |
|---|---|---|---|---|---|---|
| hero | 5 | 0.9229 ± 0.0014 | 0.9428 ± 0.0014 | 0.7616 ± 0.0110 |  |  |
| hero + Geneformer | 5 | 0.9250 ± 0.0011 | 0.9441 ± 0.0011 | 0.7816 ± 0.0044 | +0.0200 | 4.13e-05 |
| GRN + PANDA K=100 safe | 5 | 0.9226 ± 0.0011 | 0.9423 ± 0.0012 | 0.7600 ± 0.0056 | -0.0016 | 0.275 |
| neural PANDA RCP, Hadamard | 5 | 0.9239 ± 0.0011 | 0.9433 ± 0.0011 | 0.7572 ± 0.0050 | -0.0044 | 0.882 |
| neural PANDA RCP, Tanimoto | 5 | 0.9187 ± 0.0012 | 0.9384 ± 0.0015 | 0.7202 ± 0.0033 | -0.0414 | 0.000551 |
| scRegNet published recipe | 5 | 0.9196 ± 0.0001 | 0.9392 ± 0.0001 | 0.7219 ± 0.0012 | -0.0397 | 0.00479 |
| stacker, old best (GBM) | 1 fit | 0.9123 | 0.9224 | 0.7584 | -0.0032 | 0.00324 |
| stacker, all features (LR) | 1 fit | 0.9116 | 0.9263 | 0.7342 | -0.0274 | 0.00694 |

### What holds across the four cell types

- **The safe PANDA graph never beats the hero.** Its within-TF difference is negative in all four cell types (−0.017, −0.005, −0.010, −0.002). It is worse in hESC and mESC and level in the other two. Replacing or extending the GRN graph with PANDA edges does not help anywhere.
- **The Tanimoto decoder hurts within-TF in three of four cell types** (−0.029 in hESC, −0.062 in mESC, −0.041 in mHSC-E). The exception is hHEP (+0.015, p = 0.56, within the seed noise). The Hadamard product of the two embeddings works better here than the Tanimoto similarity terms.
- **The neural PANDA encoder with the Hadamard decoder gives a small and consistent gain on the pooled metrics.** Against the hero with the same seeds, AUROC is higher by 0.0014 (hESC), 0.0031 (hHEP), 0.0018 (mESC) and 0.0009 (mHSC-E), and AUPRC by 0.0060, 0.0053, 0.0042 and 0.0005. Within-TF it is better only in hHEP (+0.0145, 24 of 30 TFs, p = 0.0004). In mESC the difference is +0.0015 (p = 0.006) and in hESC and mHSC-E it is negative and not significant. A control on hESC (section 17) showed that part of the pooled gain also appears with two plain SAGE layers, and that control was not repeated on the other cell types.
- **Geneformer helps within-TF in two of the four cell types.** The gain is +0.020 in hHEP (p = 0.006) and +0.020 in mHSC-E (p = 4e-5). It is +0.004 in hESC (p = 0.09) and +0.001 in mESC (p = 0.47).
- **The published recipe does not transfer its advantage from hESC.** Against the hero, it is lower within-TF in hESC, mESC and mHSC-E (−0.044, −0.058 and −0.040) and higher in hHEP (+0.027, p = 0.08; 17 of 30 TFs better, wide interval). The hHEP result is the one case in which the hero is not the best single GNN: the published recipe reaches within-TF 0.787, against 0.760 for the hero and 0.780 for the hero with Geneformer. The hero settings were set on hESC, so this may be a tuning gap and not a model difference.
- **The stacker with graph features does not beat the hero.** The PANDA-graph features add nothing in hHEP and mHSC-E. In mESC, co-expression-graph features raise the LR stack from 0.8013 to 0.8131 within-TF (0.8023 for the label-only LR stack). The best stacker there is 0.010 below the hero within-TF (p = 0.07). In hHEP, the label-only stacker (0.783 LR, 0.785 GBM) is above the hero within-TF, but the paired test is not significant (p = 0.29 to 0.81).
- **The stacker with only context features (no ChIP-seq label graph) is 0.09 to 0.14 below the hero within-TF** in the three new cell types (0.62 to 0.74 against 0.76 to 0.82), and 0.08 below in hESC.

### Limits

- The hero, the neural PANDA settings and the PANDA K were chosen on hESC and not tuned for the new cell types.
- Rule A (validation AUROC) selected the epoch for every GNN row. Selecting on validation within-TF AUROC was not repeated here.
- The neural PANDA gain over the hero could come from the second layer and not from the PANDA relations; the matching two-layer control was run on hESC only.
- Mouse STRING was reached through upper-case alias matching, so a few mouse genes may be unmapped (mESC 965 of 977 genes mapped, mHSC-E 690 of 691).
- The numbers of TFs differ (30, 85, 29), so p-values are not comparable across cell types: mESC has the most TFs, which gives it the smallest p for a given effect.
- Files: `analysis/out/w4_report.txt` (all tables, paired tests, same-seed differences, stacker rows), `analysis/w4_report.py`, `make_ppi.py`, `analysis/jobs/w4_*.txt`.

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

### Result 1: none of the ideas helps inside the GNN
| Model (5 seeds) | AUROC | AUPRC | Within-TF | Diff vs hero (p) |
|---|---|---|---|---|
| Hero | 0.8884 | 0.6255 | 0.7470 | |
| Hero + ranking loss (w = 1) | 0.8860 | 0.6165 | 0.7391 | −0.008 (0.046) |
| Hero + PANDA teacher (w = 1) | 0.8882 | 0.6280 | 0.7411 | −0.006 (0.17) |
| Hero + PANDA number, fixed lr | 0.8886 | 0.6261 | 0.7475 | +0.001 (0.22) |
| Hero + contrast number, free weight (lr × 1000) | 0.8854 | 0.6210 | 0.7352 | −0.012 (0.0003) |
| Hero + both numbers, free weight (lr × 1000) | 0.8864 | 0.6236 | 0.7307 | −0.016 (0.006) |

- The ranking loss and the PANDA teacher both lower within-TF slightly.
- Giving the PANDA or contrast number a free weight lowers within-TF ranking. The GNN leans on the
  number and loses accuracy. The old null result was therefore not a learning-rate artefact.

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

- For genes never seen in training, neural PANDA ranks targets slightly better within each TF
  (+0.021, p = 0.041), but its AUPRC is lower. The gain needs both the co-expression and the PPI edges: without PPI it is −0.014, without co-expression +0.003 (Section 20).
  This section makes about ten comparisons, so p = 0.041 is weak evidence.
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
- PANDA and the co-expression contrast gave no gain or a loss in every form tried inside the GNN:
  as a graph, an encoder, an input number or a teacher.
- No idea in this section beats the hero within-TF on the benchmark split.

### Result 3: late fusion, and where its gain comes from
Averaging the z-scored test scores of four separately trained models (hero, Geneformer hero, neural
PANDA, stacker) beat the hero averaged over four seeds on all four cell types, by +0.009 to +0.013
within-TF. An attribution check (`analysis/w7_attribution.py`, output `analysis/out/w7_attribution.txt`)
blended the hero with one ingredient at a time and compared it with the hero blended with itself
(two seeds, the same cost). Within-TF gain:

| Added to the hero | hESC | hHEP | mESC | mHSC-E |
|---|---|---|---|---|
| Geneformer hero | +0.003 | +0.013 | +0.006 | +0.011 |
| Neural PANDA | +0.003 | +0.010 | +0.004 | −0.001 |
| Stacker | +0.009 | +0.004 | +0.005 | +0.002 |
| Raw PANDA score | +0.004 | −0.002 | −0.002 | 0.000 |
| Raw contrast score | −0.002 | −0.013 | −0.001 | +0.004 |

The blend weights were scanned on the test scores, so these numbers are optimistic; they only rank
the ingredients.

- Geneformer is the only ingredient that helps on all four cell types. The raw PANDA and contrast
  numbers do not explain the gain.
- Against the hero + Geneformer averaged over four seeds, the blend gains only on mESC (+0.010,
  p = 1e-5). It ties on hESC (+0.006, p = 0.12) and loses on hHEP (−0.008) and mHSC-E (−0.011).
- So the blend's gain over the hero is mostly Geneformer, which earlier weeks already showed helps.
  The blend adds little beyond the hero + Geneformer and was dropped.

## 20. Hard negatives, and neural PANDA for unseen genes (hESC)

### Hard negatives
Each training positive was compared with a "hard" negative of the same TF: either the most popular
gene among 16 random training negatives of that TF, or the one the model currently scores highest.
The loss was either a ranking loss (BPR) or extra BCE on that negative. 5 seeds each, paired per-TF
tests against the hero (0.7470 within-TF).

| Hard negative | Loss | Weight | Within-TF | Diff (p) |
|---|---|---|---|---|
| Most popular gene | BPR | 0.1 | 0.7094 | −0.038 (6e-5) |
| Most popular gene | BCE | 0.1 | 0.7100 | −0.037 (2e-5) |
| Most popular gene | BPR | 0.3 | 0.6420 | −0.105 (5e-7) |
| Highest model score | BPR | 0.1 | 0.7453 | −0.002 (0.55) |
| Highest model score | BCE | 0.1 | 0.7306 | −0.016 (9e-5) |
| Highest model score | BPR | 1 | 0.7114 | −0.036 (0.004) |

Every variant ties or hurts. Popular non-targets hurt most. A gene that many TFs regulate is more
likely to be a target of the next TF as well, so popularity is real biology (hub targets), not only a
benchmark artefact. Teaching the model to push popular genes down removes that signal.

### Neural PANDA for unseen genes
Held-out genes: 20% of target genes removed from training and from the graph, 5 folds × 5 seeds.

| Model | AUROC | AUPRC | Within-TF | Diff (p) |
|---|---|---|---|---|
| Hero | 0.8461 | 0.5220 | 0.5464 | |
| Neural PANDA (R + C + P) | 0.8461 | 0.5082 | 0.5624 | +0.021 vs hero (0.041) |
| Neural PANDA (R + P, no co-expression) | 0.8449 | 0.5129 | 0.5447 | +0.003 vs hero (0.40) |
| Hero + Geneformer | 0.8468 | 0.5288 | 0.5702 | +0.026 vs hero (0.004) |
| Neural PANDA + Geneformer | 0.8498 | 0.5323 | 0.5769 | +0.004 vs hero + Geneformer (0.72) |

- A gene with no known regulators has no edges in the regulatory graph, so the model must describe it
  from other information. Geneformer, pretrained on millions of cells, is the best source here.
- Neural PANDA's co-expression and PPI edges help a little, but only with both present, and the gain
  mostly overlaps with Geneformer.
- Within-TF on unseen genes stays near 0.57, against 0.75 on the normal split. Predicting regulators of
  a gene with no known regulators remains largely unsolved.

****** 25 Oct ******

## 21. Best model per cell type, then tuning it

### Part 1: GCN or SAGE with Geneformer
SAGE + Geneformer and GCN + Geneformer were run with identical settings on all four cell types. Only the
encoder differs: 1 layer, width 256, interacting (Hadamard) decoder, edge split, Adam at lr 1e-4 and
weight decay 1e-3, epoch chosen on validation AUROC, refit on all training rows, 5 seeds (42-46). The
GCN + Geneformer model from the earlier 3-seed tables used these same settings, so the 5-seed runs
replace it. Seeds 42 reproduced the old runs (hESC and mESC exactly, hHEP and mHSC-E within 0.002 AUROC).

Test metrics, mean ± std over 5 seeds (AUROC / AUPRC / within-TF):

| Cell | SAGE + Geneformer | GCN + Geneformer | Hero, no Geneformer | Published recipe |
|---|---|---|---|---|
| hESC | 0.8907 / 0.6345 / **0.7510** ± 0.0025 | 0.8899 / 0.6256 / 0.7373 ± 0.0058 | 0.8884 / 0.6255 / 0.7470 ± 0.0046 | 0.8830 / 0.6155 / 0.7029 ± 0.0063 |
| hHEP | 0.9027 / 0.8430 / 0.7800 ± 0.0046 | 0.9023 / 0.8440 / **0.7885** ± 0.0034 | 0.8993 / 0.8384 / 0.7599 ± 0.0022 | 0.9023 / 0.8413 / 0.7870 ± 0.0011 |
| mHSC-E | 0.9250 / 0.9441 / 0.7816 ± 0.0044 | 0.9260 / 0.9442 / **0.8054** ± 0.0078 | 0.9229 / 0.9428 / 0.7616 ± 0.0110 | 0.9196 / 0.9392 / 0.7219 ± 0.0012 |
| mESC | 0.9401 / 0.8838 / **0.8245** ± 0.0020 | 0.9374 / 0.8793 / 0.8117 ± 0.0021 | 0.9393 / 0.8819 / 0.8235 ± 0.0013 | 0.9254 / 0.8587 / 0.7657 ± 0.0002 |

Paired per-TF tests on within-TF AUROC (difference, Wilcoxon p over test TFs):

| Cell | GCN+GF minus SAGE+GF | Best minus published recipe |
|---|---|---|
| hESC | −0.014 (p = 0.16) | SAGE: +0.048 (p = 0.001) |
| hHEP | +0.009 (p = 0.69) | GCN: +0.002 (p = 0.53) |
| mHSC-E | +0.024 (p = 0.73) | GCN: +0.084 (p = 5e-5) |
| mESC | −0.013 (p = 2e-4) | SAGE: +0.059 (p = 6e-8) |

- The best model per cell type is the one with the higher 5-seed within-TF mean: SAGE on hESC and
  mESC, GCN on hHEP and mHSC-E. This is the earlier pattern (SAGE on hESC and mESC, GCN on mHSC-E and hHEP).
- Only mESC separates the two encoders (SAGE better, p = 2e-4). On the other three cells the p-values
  are 0.16 to 0.73, so the encoder choice there is a small and uncertain difference.
- Validation AUROC picks the other encoder on hESC and mHSC-E. The winner was named from the test
  within-TF mean of two fixed candidates, so it is a descriptive label and not a tuned choice.
- hHEP is a tie with the published recipe (+0.002, p = 0.53). On the other three cells the best model beats it.

### Part 2: tuning the best model
Each winner was tuned with 16 random trials over learning rate (3e-5 to 1e-3), weight decay (1e-5 to
1e-2), GNN width (64 to 512), MLP width, dropout (0 to 0.5) and batch size (128, 193, 512). Each trial
was scored on validation AUROC averaged over 2 seeds (0 and 1; the final seeds were not used). The
best trial was run at 5 seeds (42-46) with the same refit protocol. The untuned winner was scored as a
reference but could not be selected.

| Cell (model) | Validation AUROC, best trial / untuned / trial median | Tuned: AUROC / AUPRC / within-TF | Tuned minus untuned (p) | Tuned minus published recipe (p) |
|---|---|---|---|---|
| hHEP (GCN) | 0.9056 / 0.9039 / 0.9026 | 0.9021 / 0.8426 / 0.7874 | −0.001 (0.75) | +0.000 (0.57) |
| mHSC-E (GCN) | 0.9275 / 0.9263 / 0.9258 | 0.9250 / 0.9434 / 0.7979 | −0.008 (0.050) | +0.076 (5e-5) |
| hESC (SAGE) | 0.8939 / 0.8903 / 0.8911 | 0.8915 / 0.6349 / 0.7598 | +0.009 (0.027) | +0.057 (2e-4) |
| mESC (SAGE) | 0.9405 / 0.9396 / 0.9375 | 0.9413 / 0.8857 / 0.8297 | +0.005 (0.003) | +0.064 (2e-8) |

Chosen settings (all Adam): hHEP width 512, lr 1.6e-4, wd 1e-5, dropout 0.04, MLP 128; mHSC-E width 64,
lr 1.9e-4, wd 1.1e-3, dropout 0.45, MLP 256; hESC width 64, lr 1e-4, wd 5e-4, dropout 0.24, batch 128;
mESC width 512, lr 7e-5, wd 6e-5, dropout 0.09.

- Tuning gave a small gain on hESC (+0.009 within-TF, p = 0.03) and mESC (+0.005, p = 0.003), no change
  on hHEP, and a loss on mHSC-E (−0.008, p = 0.05).
- The validation gain of the best trial over the untuned model was 0.001 to 0.004 AUROC, about the size
  of the validation noise. On mHSC-E and hHEP the best validation trial did not carry over to test.
  Picking the best of 16 noisy scores partly picks noise.
- The tuned models stay above the published recipe on hESC, mHSC-E and mESC. On hHEP they tie it.
- Tuning is not the source of the gain over the published recipe. Changing the model (SAGE or 1-layer
  GCN, interacting decoder, edge split, validation-AUROC epoch) already gives most of it.

Scripts: `analysis/w8_report.py` (tables and tests, output in `analysis/out/w8_report.txt`),
`analysis/tune2.py wjobs|wfinals <cell> <encoder>`. Jobs: `analysis/jobs/w8_*.txt`.
