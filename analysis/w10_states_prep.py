"""Item 8 round 2: one data folder per cell-state condition under data/states/<cond>/TFs+500/ (expression subset + symlinks to the labels).
early = 00h+12h+24h (all 260 cells); late_s<seed> = 260 of the 326 cells at 72h+96h; rand_s<seed> = 260 of all 758 cells; RandomState(seed), same seed as the run.
Run: python3.12 -P analysis/w10_states_prep.py   (then analysis/jobs/w10.txt is written)"""
import os
import numpy as np, pandas as pd
P = 'data/hESC/TFs+500/'
x = pd.read_csv(P + 'BL--ExpressionData.csv', index_col=0)
pre = np.array(['_'.join(c.split('_')[:2]) for c in x.columns])
early, late = np.where(np.isin(pre, ['H9_00hb4s', 'H9_12h', 'H9_24h']))[0], np.where(np.isin(pre, ['H9_72h', 'H9_96h']))[0]
SEEDS = range(42, 47)
conds = {'early': early}
for s in SEEDS:
    conds[f'late_s{s}'] = np.sort(np.random.RandomState(s).choice(late, 260, replace=False))
    conds[f'rand_s{s}'] = np.sort(np.random.RandomState(s).choice(x.shape[1], 260, replace=False))
for c, cols in conds.items():
    d = f'data/states/{c}/TFs+500/'
    os.makedirs(d, exist_ok=True)
    x.iloc[:, cols].to_csv(d + 'BL--ExpressionData.csv')
    for f in ['Train_set.csv', 'Test_set.csv', 'TF.csv', 'Target.csv', 'Label.csv']:
        if not os.path.lexists(d + f): os.symlink(os.path.abspath(P + f), d + f)
    print(c, len(cols), dict(zip(*np.unique(pre[cols], return_counts=True))))
ARGS = '--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1 --encoder sage --split batch --decoder had --gnn_dims 256 --scfm 0'
with open('analysis/jobs/w10.txt', 'w') as f:
    for s in SEEDS:
        for k in ('early', 'late', 'rand'):
            c = 'early' if k == 'early' else f'{k}_s{s}'
            f.write(f'w10_{k}_s{s} {ARGS} --data_folder data/states --cell_type {c} --seed {s}\n')
