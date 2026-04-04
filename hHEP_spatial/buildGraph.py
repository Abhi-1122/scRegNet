import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import torch

# ── Reload coords (no need to rerun full preprocessing)
coords_df = pd.read_csv("spatial_coords.csv", index_col=0)
coords    = coords_df[["x", "y"]].values   # (7982, 2)

# ── Build kNN graph (k=6: standard Visium hexagonal grid)
k = 6
tree        = cKDTree(coords)
dists, idxs = tree.query(coords, k=k+1)   # k+1 because first neighbor is self

# Build edge_index (PyG format)
src_list, dst_list, weight_list = [], [], []
for i in range(len(coords)):
    for j_pos in range(1, k+1):          # skip self (index 0)
        j = idxs[i, j_pos]
        d = dists[i, j_pos]
        src_list.append(i)
        dst_list.append(j)
        weight_list.append(1.0 / (d + 1e-6))   # inverse distance weight

edge_index   = torch.tensor([src_list, dst_list], dtype=torch.long)
edge_weights = torch.tensor(weight_list, dtype=torch.float)

print(f"Nodes (spots): {len(coords)}")
print(f"Edges:         {edge_index.shape[1]}")
print(f"Avg degree:    {edge_index.shape[1] / len(coords):.1f}")

# ── Save for scRegNet training
torch.save({
    "edge_index":   edge_index,
    "edge_weights": edge_weights,
    "num_nodes":    len(coords),
    "k":            k
}, "spatial_knn_graph.pt")

print("\n✅ spatial_knn_graph.pt saved!")
print(f"   edge_index shape:   {edge_index.shape}")
print(f"   edge_weights shape: {edge_weights.shape}")

# ── Quick visualisation sanity check
# Pick spot 0, print its 6 neighbors and distances
spot_id  = 0
neighbors = idxs[spot_id, 1:]
neighbor_dists = dists[spot_id, 1:]
print(f"\nSpot 0 neighbors: {neighbors}")
print(f"Distances:        {np.round(neighbor_dists, 1)}")
