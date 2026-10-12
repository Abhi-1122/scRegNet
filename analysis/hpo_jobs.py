"""Equal-budget random search per model family, judged on the VALIDATION slice only (tuning seed 0,
no refit). Usage: python3.12 analysis/hpo_jobs.py [n_trials] > analysis/jobs/hpo.txt
Families keep their architecture fixed; only the optimizer settings, dropout, widths and batch size move."""
import sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
FAM = {  # name: (fixed flags, tune_width)
    'A_scregnet':   ('--decoder add --scfm 1 --split none --encoder gcn --gnn_dims 227 29 34', False),
    'B_hero_sf':    ('--decoder had --scfm 1 --split batch --encoder sage', True),
    'C_hero':       ('--decoder had --scfm 0 --split batch --encoder sage', True),
    'D_hero_panda': ('--decoder had --scfm 0 --split batch --encoder sage --scalar panda', True),
    'E_hero_sf_panda': ('--decoder had --scfm 1 --split batch --encoder sage --scalar panda', True),
}
paper = '--optimizer SGD --lr 0.0014411386699467767 --wd 1.0449337997892844e-05 --dropout 0.17477888900593389 --mlp_dim 198 --batch_size 193 --eval_interval 5 --epochs 600'
for fam, (fixed, tw) in FAM.items():
    if fam.startswith('A'):
        print(f'hpo_{fam}_paper {fixed} {paper} --seed 0')  # the published recipe as one of its trials
    rs = np.random.RandomState(abs(hash(fam)) % 2**31 if False else sum(map(ord, fam)))
    for t in range(N):
        opt = rs.choice(['Adam', 'SGD'])
        lr = float(np.exp(rs.uniform(np.log(3e-5), np.log(1e-3)))) if opt == 'Adam' else float(np.exp(rs.uniform(np.log(1e-3), np.log(5e-2))))
        wd = float(np.exp(rs.uniform(np.log(1e-6), np.log(1e-2))))
        dr = float(rs.uniform(0, 0.5))
        md = int(rs.choice([64, 128, 198, 256]))
        bs = int(rs.choice([128, 193, 512]))
        w = f'--gnn_dims {int(rs.choice([64, 128, 256, 512]))}' if tw else ''
        ev, ep = (5, 600) if opt == 'SGD' else (1, 300)
        print(f'hpo_{fam}_t{t:02d} {fixed} {w} --optimizer {opt} --lr {lr:.6g} --wd {wd:.3g} --dropout {dr:.3f} --mlp_dim {md} --batch_size {bs} --eval_interval {ev} --epochs {ep} --seed 0')
