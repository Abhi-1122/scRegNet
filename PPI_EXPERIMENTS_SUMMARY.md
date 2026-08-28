# PPI-Augmented GRN Model: Results Summary

Dataset: **hESC / TFs+500**. Backbone: **GCN**, scFM: **Geneformer**. All three runs below
reuse the exact same hyperparameters (`gnn_num_layers=3`, `gnn_hidden_dims=[227,29,34]`,
`mlp_hidden_dims=[198]`, `dropout=0.1748`, `gnn_lr=0.001441`, `gnn_weight_decay=1.045e-05`,
`optimizer=SGD`, `batch_size=193`, `random_seed=42`) so the only variable between runs is the
model architecture. PPI edges come from STRING (`data/hESC/TFs+500/STRING/`, required_score=700,
functional network), mapped onto the same gene-index space as the GRN adjacency via `Target.csv`.

## What changed in the code

- **`src/utils.py`** — added `build_ppi_adjacency()`, mirroring the existing `Adj_Generate`
  logic but reading STRING gene-symbol edges and mapping them through a `Gene → index` dict
  built from `Target.csv`, into a sparse adjacency in the same node-index space as the GRN graph.
- **`src/graph_fusion.py`** (new) — `GraphFusion` module fusing two same-dim node embeddings via
  `concat`, a learnable `gate`, or single/multi-head cross-`attention` + residual. Used to
  combine `embed_grn` and `embed_ppi` into `embed_graph` before it's fed into the existing
  scFM-fusion + MLP pipeline (kept unchanged downstream).
- **`src/args.py`** — new flags: `--use_ppi`, `--multi_view`, `--ppi_fusion_type`
  (`concat|gate|attention`), `--ppi_fusion_heads`, `--view_guidance`
  (`symmetric|grn_guides_ppi|ppi_guides_grn`).
- **`src/models.py`** (`scTransNet_GCN`) —
  - **Method 1 (dual encoder / late fusion)**: a second `convs_ppi` branch, structurally
    identical to the GRN branch, encodes the same node features over the PPI adjacency. Its
    output is combined with the GRN branch's output via `GraphFusion` *after* both finish
    encoding, then fused with the scFM embedding exactly as before.
  - **Method 2 (multi-view / joint learning)**: `encode_multiview()` interleaves the GRN and PPI
    convolutions layer-by-layer. After each layer, a learnable sigmoid gate lets each view read
    from the other (`view_guidance=symmetric` — used for the run below — mixes both directions;
    `grn_guides_ppi`/`ppi_guides_grn` are one-directional variants) before continuing to the next
    layer. The final layer's outputs are combined with the same `GraphFusion` module used by
    Method 1.
  - Both paths are opt-in (`--use_ppi`, `--multi_view`); the original single-encoder path is
    unchanged and was re-verified via a fresh baseline training run (below), not just re-read
    from an old checkpoint.
- **`src/inference.py`** — now persists `final_results_{train,test}.json` (mean/std AUC & AUPR
  over 50 eval passes) via the previously-unused `store_results()` helper, and takes the
  checkpoint directory as a CLI arg instead of a hardcoded path.

## Results

| Run | AUROC | AUPRC | ΔAUROC vs. baseline | ΔAUPRC vs. baseline |
|---|---|---|---|---|
| **Baseline** (GRN-only, re-run fresh) | 0.8874 | 0.6225 | — | — |
| **Method 1** — dual encoder, attention fusion | 0.8861 | 0.6157 | −0.0013 (−0.15%) | −0.0068 (−1.09%) |
| **Method 2** — multi-view, symmetric cross-view gating | 0.8840 | 0.6152 | −0.0034 (−0.38%) | −0.0073 (−1.17%) |

Checkpoints and result files:
- Baseline: `out/GCN/Geneformer/tf_500_hESC_baseline_rerun/best/ckpt/`
- Method 1: `out/GCN/Geneformer/tf_500_hESC_dual_ppi_attn/best/ckpt/`
- Method 2: `out/GCN/Geneformer/tf_500_hESC_multiview/best/ckpt/`

## Takeaway

On hESC/TFs+500 with these hyperparameters, adding the STRING PPI network — via either late
fusion (Method 1) or per-layer joint learning (Method 2) — did **not** improve over the GRN-only
baseline; both PPI variants scored marginally lower, with Method 2 (multi-view) landing slightly
below Method 1 (dual encoder). This suggests that for this dataset/backbone/hyperparameter
combination, the PPI signal used this way is not adding complementary information beyond what
the GRN adjacency + scFM embeddings already capture. Caveats worth keeping in mind before
concluding PPI is unhelpful in general:
- Hyperparameters were reused from a search tuned for the GRN-only model, not re-tuned for the
  dual-encoder/multi-view architectures — the extra PPI branch adds parameters that may want a
  different learning rate, dropout, or hidden-dim schedule.
  * Only the `attention` fusion variant (of `concat`/`gate`/`attention`, all implemented and
  selectable via `--ppi_fusion_type`) was actually trained and reported here; `concat`/`gate`
  were not swept.
- Only `view_guidance=symmetric` was run for Method 2; the one-directional guidance variants
  (`grn_guides_ppi`, `ppi_guides_grn`) were not evaluated.
- Single seed (42), no variance estimate across seeds.

## Follow-up: modality ablation (GRN vs. scFM contribution)

The PPI results above raised the question of whether the GRN-GNN branch is
contributing much at all relative to the scFM embeddings — motivated by
`--ablation {no_scfm, no_grn}` (new flag in `src/args.py`, wired into
`scTransNet_GCN.forward()` in `src/models.py`). Same dataset, same reused
hyperparameters as the baseline above, no PPI involved.

- `no_scfm`: the MLP head sees only the GNN's GRN-graph embedding — the
  concat/cross-attention fusion with `llm_emb` is skipped entirely (`embed`
  stays as the raw GNN output).
- `no_grn`: the MLP head sees only the raw scFM (Geneformer) gene embedding —
  the GNN encoder (`self.encode(x, adj)`) is skipped entirely and
  `embed = llm_emb`.

| Run | AUROC | AUPRC | ΔAUROC vs. baseline | ΔAUPRC vs. baseline |
|---|---|---|---|---|
| **Baseline** (GRN + scFM) | 0.8874 | 0.6225 | — | — |
| **no_scfm** — GRN-GNN only | 0.8747 | 0.5916 | −0.0127 (−1.43%) | −0.0309 (−4.97%) |
| **no_grn** — scFM only | 0.8774 | 0.6064 | −0.0100 (−1.13%) | −0.0161 (−2.59%) |

Checkpoints/results: `out/GCN/Geneformer/tf_500_hESC_ablation_no_scfm/best/ckpt/`,
`out/GCN/Geneformer/tf_500_hESC_ablation_no_grn/best/ckpt/`.

**Reading:** neither modality alone comes close to matching the full model —
both single-modality runs lose 1–5% relative to baseline. scFM-only is
slightly closer to baseline than GRN-only (notably on AUPRC: −2.59% vs.
−4.97%), so scFM does carry a bit more of the predictive signal on its own,
but the gap between the two ablations is small (0.0027 AUROC, 0.0148 AUPRC)
— this does not support "scFM accounts for the majority of the
information." The more robust conclusion is that **GRN and scFM are
complementary**: combining them beats either alone by a clear margin, so the
GRN-GNN branch is doing real work, not riding on scFM. This also reframes
the PPI result above — PPI's failure to help is not because the existing
GRN branch is negligible, but more likely because PPI-as-a-second-graph-view
duplicates information already inside the (meaningfully contributing) GRN
signal rather than adding a third independent source.

*Note:* an initial `no_scfm` run accidentally set `--loop true` on the GRN
adjacency (Python's `argparse type=bool` treats any non-empty string,
including `"false"`, as `True` — a pre-existing footgun in `src/args.py`),
diverging from baseline's `loop=False`. It was re-run correctly (omitting
the flag so it defaults to `False`) before reporting the numbers above.

## Follow-up: PANDA (classical network integration, no foundation model)

Motivated by the ablation above: since scFM only edges out GRN-only by a small
margin, does a *classical* (non-deep-learned) network-integration signal do
just as well as scFM, or add anything on top of it? [PANDA](https://doi.org/10.1371/journal.pone.0064832)
(Glass et al. 2013) iteratively refines a TF-target regulatory prior using
co-expression and TF-TF PPI evidence via message passing — no gradient
descent, no embeddings, no scFM.

**New file `src/build_panda.py`** builds PANDA's three inputs for
hESC/TFs+500 and runs it via `netZooPy`'s `Panda` class:
- **Expression**: `BL--ExpressionData.csv` directly (genes × cells); PANDA
  computes the co-expression network internally (`np.corrcoef`).
- **Regulatory prior**: the ~3030 train-positive TF-target edges from
  `Train_set.csv` (weight=1.0), mapped to gene symbols via `Target.csv`.
- **TF-TF PPI**: the existing STRING PPI edges (`STRING/*.edges.csv`),
  filtered to pairs where both genes are in `TF.csv` (410 TFs; PANDA's PPI
  input is a TF-TF cooperativity network, not the full gene-gene PPI used by
  the Method 1/2 experiments above).

PANDA's output is a single (410 TFs × 910 genes) regulatory-score matrix,
re-indexed onto this project's global gene-index space and scaled (not
centered, so the ~100 TFs with no PANDA coverage stay exactly 0) into
`data/hESC/TFs+500/PANDA/panda_full_matrix.npy`.

**`src/args.py`** — new flags: `--node_feature_source {scfm,panda}`
(Experiment B) and `--use_panda_edge` (Experiment C).

- **Experiment B — no scFM.** `feature1` (normally scFM/Geneformer) is
  replaced with PANDA's per-gene profile: column *j* of the PANDA matrix
  (a 410-dim vector — one score per TF — for gene *j*). No model code change
  needed at all: `gene_dim` is already derived dynamically from `feature1`'s
  shape in `train.py`/`inference.py`, so the existing concat/cross-attention
  fusion and MLP head adapt automatically.
- **Experiment C — with scFM.** The full scFM+GNN pipeline is unchanged;
  PANDA's edge score for the specific `(TF_i, target_j)` pair being scored is
  concatenated as one extra scalar feature onto the MLP head's input
  (`scTransNet_GCN.decode()`'s new `extra_feat` parameter; `self.linear`'s
  input width grows by 1 when `--use_panda_edge` is set).
- **Experiment D — scFM only, no GRN-GNN.** Combines the existing
  `--ablation no_grn` flag (GNN encoding skipped entirely, `embed = llm_emb`)
  with `--use_panda_edge`. No new code — the two flags already composed
  cleanly, since `use_panda_edge`'s `extra_feat` concatenation happens after
  `embed` is resolved, regardless of which ablation path produced it. Tests
  whether PANDA's edge signal helps scFM specifically, with no GRN topology
  in the picture at all.

### Leakage bug and fix (important)

The first version of Experiment C scored **AUROC 0.7742 / AUPRC 0.3466** —
far worse than every other variant tried, including the plain ablations.
Diagnosis: PANDA's regulatory prior *is* the train-positive edges (weight=1),
so a train-positive edge's own PANDA score is almost a direct echo of "you
told me this was positive": train-positive scores averaged **9.98 ± 2.25**
vs. **-0.03 ± 0.58** for everything else, while test-positive scores (never
in any prior) showed no such separation (**0.50 ± 0.66** vs. **-0.02 ± 0.58**
for test-negatives). The model learned to lean on this near-perfect
train-only shortcut during training, which is worthless (and actively
misleading) at test time — that's what collapsed AUPRC, not PANDA
integration being inherently unhelpful.

**Fix**: `src/build_panda.py` now also computes a leakage-safe matrix via
5-fold out-of-fold scoring — for each fold, PANDA is rerun with that fold's
train-positive edges excluded from the prior, and those edges are scored
from the resulting held-out matrix (`panda_full_matrix_train_oof.npy`).
`train.py`/`inference.py` use this OOF matrix for all **train**-pair lookups
and the original full-prior matrix for all **test**-pair lookups (test edges
were never in any prior, so they were already leakage-free). After the fix,
train-positive scores dropped to **0.28 ± 0.65** — in line with test-positive
scores, confirming the leak was closed.

### Results

| Run | AUROC | AUPRC | ΔAUROC vs. baseline | ΔAUPRC vs. baseline |
|---|---|---|---|---|
| **Baseline** (GRN + scFM) | 0.8874 | 0.6225 | — | — |
| **Experiment B** — no scFM, PANDA profile only | 0.8799 | 0.6102 | −0.0075 (−0.85%) | −0.0123 (−1.98%) |
| **Experiment C** — scFM + GNN + PANDA edge feature (leakage-fixed) | **0.8902** | **0.6333** | **+0.0028 (+0.32%)** | **+0.0108 (+1.73%)** |
| **Experiment D** — scFM + PANDA edge feature, no GNN | 0.8829 | 0.6175 | −0.0045 (−0.51%) | −0.0050 (−0.80%) |

Checkpoints/results: `out/GCN/Geneformer/tf_500_hESC_panda_only/best/ckpt/`,
`out/GCN/Geneformer/tf_500_hESC_panda_edge/best/ckpt/`,
`out/GCN/Geneformer/tf_500_hESC_panda_scfm_only/best/ckpt/`.

Experiment D vs. the plain scFM-only ablation (no PANDA, no GRN) from the
section above: 0.8829/0.6175 vs. 0.8774/0.6064 — adding PANDA's edge feature
to scFM alone gives **+0.0055 AUROC (+0.62%) / +0.0111 AUPRC (+1.83%)**, a
clear improvement, closing roughly half the AUROC gap and about a third of
the AUPRC gap to the full baseline. So PANDA's edge signal helps scFM on its
own too, not only when paired with the GRN-GNN branch (Experiment C) — but
it helps *more* when GRN topology is also present (Experiment C's gain over
baseline is positive; Experiment D's gain over scFM-only doesn't fully close
the gap to baseline), suggesting GRN topology and PANDA's edge evidence are
at least partially complementary rather than redundant substitutes for one
another.

**Reading:** Experiment B (no scFM at all) beats *both* single-modality
ablations above — GRN-only (0.8747/0.5916) and scFM-only (0.8774/0.6064) —
despite using no foundation model and no deep-learned embedding, just a
classical iterative-refinement score. That's a meaningful result on its own:
a non-deep-learning, non-pretrained signal gets closer to full-model
performance than either deep-learned single modality does alone.

Experiment C is the best result across every experiment in this project so
far — small but real gains on top of the already-strong scFM+GNN baseline,
after fixing the leakage. This suggests PANDA's edge-level regulatory
evidence (co-expression + PPI, refined via message passing) contains a
sliver of genuinely complementary signal beyond what scFM + GRN-topology
already capture, unlike the raw STRING PPI network tried in Methods 1/2,
which added nothing. The difference is plausibly *how* the information is
combined (PANDA does its own principled refinement/message-passing before
the score ever reaches the model, vs. Methods 1/2 asking a GNN branch to
learn everything about PPI structure from scratch with reused, untuned
hyperparameters) and *what* it's built from (co-expression evidence, which
Methods 1/2 never had access to).

Caveats: hyperparameters reused from the GRN+scFM search rather than re-tuned
for the extra PANDA feature (confirmed via a 50-trial QMC hyperparameter
search — see below — that this doesn't leave easy gains on the table), and the
PANDA prior is built once from all train-positive edges (Experiment C's
leakage fix only re-scores individual edges out-of-fold — the co-expression
and PPI networks themselves are shared and unaffected, which is standard
practice, not a remaining leak).

**Update — multi-seed validation (5 seeds, 2026-08-28):** the single-seed
caveat above is resolved. All 5 seeds beat baseline on both AUROC and AUPRC,
with a tight spread: AUROC 0.8899 ± 0.0005, AUPRC 0.6307 ± 0.0018 (vs.
baseline 0.8874/0.6225). Exp C's gain is confirmed real, not seed noise.
A follow-up 50-trial hyperparameter search targeting Exp C specifically
(`src/tuning/`, QMC sampler, 4 GPUs) found nothing better than the reused
hyperparameters (best trial: AUROC 0.8890/AUPRC 0.6255) — the reused
hyperparameters, from the original adaptive search, were already close to
optimal for this architecture. This closes out the PPI/ablation/PANDA/scFM
line of work; see `work.md`'s 2026-08-28 entry for what comes next.

## Follow-up: alternate scFM backbones (scGPT, UCE)

Question: does the *choice* of foundation model matter, or is Geneformer
interchangeable with other scFMs? Same dataset/hyperparameters, GRN+scFM
pipeline unchanged (no PPI, no PANDA) — only `--llm_type` and the source of
`feature1` change.

- **scGPT** (`bowang-lab/scGPT`, "whole-human" checkpoint, 33M cells).
  Gene embeddings extracted via direct token-embedding-layer lookup
  (`model.encoder(gene_ids)`, `src/extract_scgpt_embeddings.py`) — no cell
  inference needed, following scGPT's own published GRN-inference workflow.
  512-dim; 848/910 genes matched in vocab. Required an isolated venv to
  install (scGPT's dependency tree — JAX/Flax, scvi-tools, a pinned old
  `torchtext`↔`torch` pairing — conflicts with the main environment).
- **UCE** (`snap-stanford/UCE`, 4-layer checkpoint, via HF mirror
  `perturblab/uce-4-layer` — the official figshare host blocked automated
  downloads with an AWS WAF challenge from this network). Turned out to
  represent each gene via a fixed, species-specific ESM2 protein-embedding
  (5120-dim), projected through UCE's own `Linear→GELU→LayerNorm` encoder
  layer — a genuinely data-independent per-gene embedding, so extraction
  followed the same direct-lookup pattern as scGPT/Geneformer
  (`src/extract_uce_embeddings.py`), no cell inference. 1280-dim; 848/910
  genes matched (same coverage as scGPT).

### Results

| Backbone | AUROC | AUPRC |
|---|---|---|
| **Geneformer** (baseline) | 0.8874 | 0.6225 |
| **scGPT** | 0.8858 | 0.6225 |
| **UCE** | 0.8839 | 0.6187 |

Checkpoints/results: `out/GCN/scGPT/tf_500_hESC/best/ckpt/`,
`out/GCN/UCE/tf_500_hESC/best/ckpt/`.

**Reading:** all three cluster tightly (0.884–0.887 AUROC) despite very
different architectures, pretraining data/scale, and embedding dimensions
(896 / 512 / 1280). The specific scFM choice matters far less than *whether*
one is used at all, or *how* auxiliary information (PPI vs. PANDA) is
combined with it — consistent with the modality-ablation finding that scFM
and GRN contribute comparably rather than one dominating.
