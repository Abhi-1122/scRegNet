"""
Step 0b (one-time): Convert BL-network.csv → edge_index tensor.

BL-network.csv expected columns: TF, Target (gene names)
Saves: ./BL-network_edge_index.pt
"""

from pathlib import Path
import torch, pandas as pd, pickle

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
SCRIPT_DIR    = Path(__file__).resolve().parent
NETWORK_CSV   = PROJECT_ROOT / "data" / "hESC" / "TFs+500" / "BL--network.csv"
GENE2IDX_PKL  = SCRIPT_DIR / "perturb_triples" / "gene_list.pkl"
OUT_EDGE      = SCRIPT_DIR / "BL-network_edge_index.pt"

with open(GENE2IDX_PKL, "rb") as f:
    gene_list = pickle.load(f)
gene2idx = {g: i for i, g in enumerate(gene_list)}

bl = pd.read_csv(NETWORK_CSV)
if bl.columns[0].lower() in ["tf", "source"]:
    tf_col, tgt_col = bl.columns[0], bl.columns[1]
else:
    tf_col, tgt_col = bl.columns[0], bl.columns[1]

src, dst = [], []
skipped = 0
for _, row in bl.iterrows():
    tf, tgt = str(row[tf_col]), str(row[tgt_col])
    if tf in gene2idx and tgt in gene2idx:
        src.append(gene2idx[tf])
        dst.append(gene2idx[tgt])
    else:
        skipped += 1

edge_index = torch.tensor([src, dst], dtype=torch.long)
torch.save(edge_index, OUT_EDGE)
print(f"Edge index: {edge_index.shape}  | Skipped (missing from vocab): {skipped}")
