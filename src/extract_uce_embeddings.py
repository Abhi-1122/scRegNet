"""Extract UCE's data-independent per-gene identity embeddings for hESC/TFs+500.

Background
----------
UCE (Universal Cell Embeddings, snap-stanford/UCE, Rosen et al. 2023/2026,
https://github.com/snap-stanford/UCE) is a cell-embedding transformer, but
its gene *tokens* are not a learned nn.Embedding indexed by integer gene id
(unlike Geneformer/scGPT). Instead each gene is represented by a fixed,
species-specific ESM2 protein-language-model embedding (5120-dim, keyed by
gene symbol), which UCE's own first layer -- `model.encoder` in
snap-stanford/UCE's model.py, a `Linear(5120, 1280) -> GELU -> LayerNorm`
-- projects down to the transformer's working dimension (1280) *before* any
cell/expression-conditioned attention happens:

    class TransformerModel(nn.Module):
        def __init__(self, token_dim, d_model, nhead, d_hid, nlayers, output_dim, dropout=0.05):
            ...
            self.encoder = nn.Sequential(nn.Linear(token_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        def forward(self, src, mask):
            src = self.encoder(src) * math.sqrt(self.d_model)   # <- gene-identity projection, no cell info yet
            ...

So exactly like scGPT's `model.encoder(gene_ids)` precedent (see
src/extract_scgpt_embeddings.py), UCE exposes a genuinely data-independent
per-gene embedding: look up each gene's precomputed ESM2 vector, run it
through `model.encoder` alone, and never touch the transformer_encoder or
run actual cell/expression data through the model. No cell inference /
scBERT-style mean-pooling is needed or used here.

Checkpoint / embedding source
------------------------------
The official checkpoint is distributed via Figshare
(https://figshare.com/articles/dataset/Universal_Cell_Embedding_Model_Files/24320806),
linked from the snap-stanford/UCE README and downloaded automatically by
its `eval_single_anndata.py` via these direct files:
  - 4-layer model weights:        https://figshare.com/ndownloader/files/42706576
  - protein_embeddings.tar.gz:    https://figshare.com/ndownloader/files/42715213
    (contains Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt, a dict
    of {GENE_SYMBOL (upper-case): torch.Tensor[5120]} -- the ESM2 vectors)

At the time this script was written, figshare.com was returning AWS WAF
"challenge"/403 responses to non-browser HTTP clients from this machine's
network, blocking direct curl/wget download of those files. A pip package
`uce`/`uce-model` does not exist on PyPI. As a working substitute we used a
third-party HF mirror of the identical checkpoint + embeddings,
`perturblab/uce-4-layer` (config.json there matches the official 4-layer
architecture exactly: token_dim=5120, d_model=1280, nhead=20, d_hid=5120,
nlayers=4, output_dim=1280):

    pip install -U huggingface_hub   # or just curl, no auth needed, repo is public
    curl -L -o model.pt \
        "https://huggingface.co/perturblab/uce-4-layer/resolve/main/model.pt"
    curl -L -o Homo_sapiens_ESM2.pt \
        "https://huggingface.co/perturblab/uce-4-layer/resolve/main/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"

(model.pt ~3.4GB, Homo_sapiens_ESM2.pt ~0.4GB.) If figshare is reachable
from your network, the original files work identically -- this script only
needs (a) a state dict with `encoder.0.{weight,bias}` and `encoder.2.{weight,bias}`
keys of shape matching token_dim=5120 -> d_model=1280, and (b) a
{gene_symbol: Tensor[5120]} dict for human.

No `uce`/`scgpt`-style package install is required at all -- this script
only uses torch/pandas/numpy (already in the project's main env), since we
reimplement the ~6-line `encoder` submodule directly rather than importing
snap-stanford/UCE's full model.py (which pulls in scanpy/accelerate for its
AnnData preprocessing pipeline that we don't need for a pure gene-identity
lookup). Runs fine on CPU; takes a few seconds for 910 genes.

Usage
-----
    python3.12 src/extract_uce_embeddings.py \
        --ckpt-path <dir>/model.pt \
        --pe-path <dir>/Homo_sapiens_ESM2.pt

Delete the downloaded model.pt / Homo_sapiens_ESM2.pt (~3.8GB total) from
scratch space once this has been run -- only the output CSV needs to persist
in the repo (scFM/UCE/hESC_500_gene_embeddings.csv, git-ignored like the
other scFM/*/*.csv outputs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def build_encoder(token_dim: int, d_model: int) -> nn.Sequential:
    """Reimplementation of snap-stanford/UCE's TransformerModel.encoder submodule
    (Linear -> GELU -> LayerNorm), the gene-identity projection applied to raw
    ESM2 embeddings before any transformer/cell-conditioned processing."""
    return nn.Sequential(
        nn.Linear(token_dim, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, help="UCE 4-layer model.pt (state dict)")
    parser.add_argument("--pe-path", required=True, help="Homo_sapiens...gene_symbol_to_embedding_ESM2.pt")
    parser.add_argument("--data-dir", default="./data/hESC/TFs+500")
    parser.add_argument("--out-dir", default="./scFM/UCE")
    parser.add_argument("--out-name", default="hESC_500_gene_embeddings.csv")
    parser.add_argument("--token-dim", type=int, default=5120)
    parser.add_argument("--d-model", type=int, default=1280)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"Loading human ESM2 gene embeddings from {args.pe_path}")
    species_pe = torch.load(args.pe_path, map_location="cpu")
    # species_pe: dict {GENE_SYMBOL (upper-case): Tensor[token_dim]}
    print(f"  {len(species_pe)} genes in ESM2 table")

    print(f"Loading UCE checkpoint from {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    encoder = build_encoder(args.token_dim, args.d_model)
    encoder_state = {
        "0.weight": state_dict["encoder.0.weight"],
        "0.bias": state_dict["encoder.0.bias"],
        "2.weight": state_dict["encoder.2.weight"],
        "2.bias": state_dict["encoder.2.bias"],
    }
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()
    print("Loaded UCE's gene-identity encoder (Linear -> GELU -> LayerNorm)")

    our_genes = pd.read_csv(data_dir / "Target.csv", index_col=0)["Gene"].tolist()
    matched_genes = [g for g in our_genes if g.upper() in species_pe]
    print(f"matched in UCE/ESM2 human vocab: {len(matched_genes)}/{len(our_genes)}")

    pe_stack = torch.stack([species_pe[g.upper()].float() for g in matched_genes]).to(device)
    with torch.no_grad():
        gene_embeddings = encoder(pe_stack)
    gene_embeddings = gene_embeddings.detach().cpu().numpy()

    out_path = out_dir / args.out_name
    pd.DataFrame(gene_embeddings, index=matched_genes).to_csv(out_path)
    print(f"Saved {out_path} ({gene_embeddings.shape[0]} genes x {gene_embeddings.shape[1]} dims)")


if __name__ == "__main__":
    main()
