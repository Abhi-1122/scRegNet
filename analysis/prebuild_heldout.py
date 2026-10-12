"""Fill the contrast-scalar cache for every held-out fold and validation setting (avoids races between jobs)."""
import sys
from types import SimpleNamespace
sys.path.insert(0, '.'); sys.path.insert(0, 'analysis')
import scalars
from src.pair_gnn import prepare
for pr in ('gene', 'tf'):
    for fold in range(5):
        for vf in (0.1, 0.0):
            a = SimpleNamespace(cell_type='hESC', num_TF='500', data_folder='data', scfm=0, scfm_folder='scFM', val_frac=vf, split_seed=0, scalar='contrast', protocol=pr, fold=fold)
            scalars.get(a, prepare(a, None))
        print(pr, fold, 'ok')
