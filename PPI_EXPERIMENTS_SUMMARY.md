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
