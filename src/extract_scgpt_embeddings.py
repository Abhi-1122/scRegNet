"""Extract scGPT's data-independent per-gene token embeddings for hESC/TFs+500,
following scGPT's own Tutorial_GRN.ipynb pattern (direct embedding-layer lookup,
model.encoder(gene_ids) -- no cell inference needed). Output matches scRegNet's
scFM CSV convention, consumed by the "scGPT" branch of _get_embeddings() in
src/train.py / src/inference.py.

This script needs the `scgpt` package, which pulls in a large, mostly-unrelated
dependency tree (JAX/Flax, scvi-tools, an old pinned torchtext<->torch pairing)
that conflicts with this project's main environment. Run it in its own venv:

    python3.12 -m venv /path/to/scgpt_venv
    source /path/to/scgpt_venv/bin/activate
    pip install scgpt ipython
    pip install "torch==2.3.0"   # scgpt's torchtext==0.18.0 needs this exact pin;
                                  # a newer torch breaks torchtext's compiled extension
    pip install gdown
    python -m gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" -O <ckpt_dir>
    python src/extract_scgpt_embeddings.py --ckpt-dir <ckpt_dir>/scGPT_human

The gdown link is scGPT's "whole-human" checkpoint (33M cells; recommended
default per the scGPT README) -- folder contains args.json, best_model.pt,
vocab.json. No flash-attn or GPU is required for this extraction: we only
index the pretrained gene-identity embedding table, never run the
transformer's self-attention layers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="Folder with args.json, best_model.pt, vocab.json")
    parser.add_argument("--data-dir", default="./data/hESC/TFs+500")
    parser.add_argument("--out-dir", default="./scFM/scGPT")
    parser.add_argument("--out-name", default="hESC_500_gene_embeddings.csv")
    args = parser.parse_args()

    set_seed(42)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    pad_value = -2
    n_input_bins = 51  # matches n_bins in the whole-human checkpoint's args.json

    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = GeneVocab.from_file(ckpt_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    model_configs = json.load(open(ckpt_dir / "args.json"))
    gene2idx = vocab.get_stoi()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = TransformerModel(
        len(vocab),
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    # The pretrained checkpoint was trained with fast_transformer=True (flash-attn's
    # fused Wqkv layer), which won't strict-load into the standard PyTorch attention
    # implementation used when flash-attn isn't installed. That mismatch only affects
    # transformer_encoder.*.self_attn.* and the task-specific decoder heads -- the
    # gene-identity embedding table (encoder.embedding.weight) we actually need is
    # unaffected and loads correctly via the partial-match fallback below.
    try:
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
        print("Loaded all model params (strict)")
    except Exception as e:
        print(f"Strict load failed ({e}); loading matching params only")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(ckpt_dir / "best_model.pt", map_location=device)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} matching params")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    our_genes = pd.read_csv(data_dir / "Target.csv", index_col=0)["Gene"].tolist()
    matched_genes = [g for g in our_genes if g in gene2idx]
    print(f"matched in scGPT vocab: {len(matched_genes)}/{len(our_genes)}")

    gene_ids = np.array([gene2idx[g] for g in matched_genes])
    with torch.no_grad():
        gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()

    out_path = out_dir / args.out_name
    pd.DataFrame(gene_embeddings, index=matched_genes).to_csv(out_path)
    print(f"Saved {out_path} ({gene_embeddings.shape[0]} genes x {gene_embeddings.shape[1]} dims)")


if __name__ == "__main__":
    main()
