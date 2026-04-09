"""
Step 0 (one-time): Build fm_embeddings.pt from existing Geneformer CSV outputs.

Input:
  - scFM/Geneformer/hESC_500.csv
  - scFM/Geneformer/hESC_500_gene_embeddings.csv

Output:
  - data/hESC/TFs+500/fm_embeddings.pt   (dict: gene_symbol -> torch.Tensor[fm_dim])
"""

from pathlib import Path
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "hESC" / "TFs+500"
SCFM_DIR = PROJECT_ROOT / "scFM" / "Geneformer"

GENE_MAP_CSV = SCFM_DIR / "hESC_500.csv"
EMBED_CSV = SCFM_DIR / "hESC_500_gene_embeddings.csv"
OUT_PT = DATASET_DIR / "fm_embeddings.pt"


def main() -> None:
    gene_map = pd.read_csv(GENE_MAP_CSV)
    emb = pd.read_csv(EMBED_CSV, index_col=0)

    emb = emb[~emb.index.isin(["<cls>", "<eos>"])].copy()
    emb.index = emb.index.astype(str)

    ensg_to_gene = dict(zip(gene_map["ensembl_id"].astype(str), gene_map["gene"].astype(str)))

    fm_embeddings = {}
    skipped = 0
    for ensg, row in emb.iterrows():
        gene = ensg_to_gene.get(str(ensg))
        if gene is None:
            skipped += 1
            continue
        fm_embeddings[gene] = torch.tensor(row.values, dtype=torch.float32)

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fm_embeddings, OUT_PT)

    fm_dim = next(iter(fm_embeddings.values())).shape[0] if fm_embeddings else 0
    print(f"Saved {OUT_PT}")
    print(f"Mapped genes: {len(fm_embeddings)} | Embedding dim: {fm_dim} | Skipped ENSG: {skipped}")


if __name__ == "__main__":
    main()
