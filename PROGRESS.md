# In-depth analysis: why does PANDA help scRegNet?

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
