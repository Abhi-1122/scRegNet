"""Final runs: 5 seeds per config with validation-chosen hyperparameters, plus held-out-gene / held-out-TF tests.
Writes analysis/jobs/finals.txt and analysis/jobs/heldout.txt."""
import json
H = json.load(open('analysis/out/hpo_best.json'))
def hp(fam):
    f = H[fam]['flags']
    return (f"--optimizer {f['optimizer']} --lr {f['lr']} --wd {f['wd']} --dropout {f['dropout']} --mlp_dim {f['mlp_dim']} "
            f"--batch_size {f['batch_size']} --gnn_dims {' '.join(map(str, f['gnn_dims']))} --eval_interval {f['eval_interval']} --epochs {f['epochs']}")
paper = ("--optimizer SGD --lr 0.0014411386699467767 --wd 1.0449337997892844e-05 --dropout 0.17477888900593389 --mlp_dim 198 "
         "--batch_size 193 --gnn_dims 227 29 34 --eval_interval 5 --epochs 600")
A, B, C = hp('A_scregnet'), hp('B_hero_sf'), hp('C_hero')
CFG = {
 'f01_A_tuned':   f'--decoder add --scfm 1 --split none --encoder gcn {A}',
 'f02_A_paper':   f'--decoder add --scfm 1 --split none --encoder gcn {paper}',
 'f03_B_tuned':   f'--decoder had --scfm 1 --split batch --encoder sage {B}',
 'f04_C_tuned':   f'--decoder had --scfm 0 --split batch --encoder sage {C}',
 'f05_C_contrast':f'--decoder had --scfm 0 --split batch --encoder sage {C} --scalar contrast',
 'f06_C_panda':   f'--decoder had --scfm 0 --split batch --encoder sage {C} --scalar panda',
 'f07_C_gcn':     f'--decoder had --scfm 0 --split batch --encoder gcn {C}',
 'f08_C_add':     f'--decoder add --scfm 0 --split batch --encoder sage {C}',
 'f09_C_nosplit': f'--decoder had --scfm 0 --split none --encoder sage {C}',
 'f10_C_mlp':     f'--decoder mlp --scfm 0 --split batch --encoder sage {C}',
 'f11_B_panda':   f'--decoder had --scfm 1 --split batch --encoder sage {B} --scalar panda',
}
with open('analysis/jobs/finals.txt', 'w') as f:
    for k, c in CFG.items():
        for sd in range(42, 47): f.write(f'{k}_s{sd} {c} --seed {sd} --refit 1\n')
with open('analysis/jobs/heldout.txt', 'w') as f:
    for pr in ('gene', 'tf'):
        for fold in range(5):
            for k in ('f01_A_tuned', 'f03_B_tuned', 'f04_C_tuned', 'f05_C_contrast'):
                for sd in (42, 43): f.write(f'ho_{pr}{fold}_{k}_s{sd} {CFG[k]} --protocol {pr} --fold {fold} --seed {sd} --refit 1\n')
