import pandas as pd
labels = pd.read_csv('data/hESC/TFs+500/TF.csv')  # your hESC or hHEP labels
beeline_tfs = set(labels['TF'].unique())
print(f"TFs in BEELINE: {len(beeline_tfs)}")
print(beeline_tfs)


# If using GEARS data loader
from gears import PertData
pert_data = PertData('./data')
pert_data.load(data_name='norman')
norman_tfs = set(pert_data.pert_names)  # perturbed gene names
print(f"Perturbed genes in Norman: {len(norman_tfs)}")

# Overlap
overlap = beeline_tfs & norman_tfs
print(f"Overlap: {len(overlap)} genes")
print(overlap)