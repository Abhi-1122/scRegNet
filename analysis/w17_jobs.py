"""W17: write analysis/jobs/w17_*.txt (held-out genes/TFs, 5 folds x seeds 42, 43; arms pipe / tuned / hero). Run: python3.12 analysis/w17_jobs.py"""
CELLS = {  # validation-chosen pipeline GNN per cell (analysis/out/w16_choose.txt, W16_pipeline.md)
    'hESC': '--encoder gcn --gnn_dims 256 --split batch --decoder had --dec_wd 0', 'hHEP': '--encoder gcn --gnn_dims 256 --split batch --decoder had --dec_wd 0',
    'mHSC-E': '--encoder npanda --gnn_dims 256 256 --relations RCP --split batch --decoder had', 'mESC': '--encoder npanda --gnn_dims 256 256 --relations RCP --split none --decoder had'}
ARM = {'pipe': lambda c: f'--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1 --scfm 1 --save_sel_grid 1 --cell_type {c} {CELLS[c]}',
       'tuned': lambda c: f'--decoder add --scfm 1 --split none --encoder gcn --optimizer Adam --lr 4.29028e-05 --wd 0.002 --dropout 0.481 --mlp_dim 198 --batch_size 512 --gnn_dims 227 29 34 --eval_interval 1 --epochs 300 --refit 1 --cell_type {c}',
       'hero': lambda c: f'--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1 --encoder sage --split batch --decoder had --gnn_dims 256 --scfm 0 --cell_type {c}'}
def lines(c, arms, prots=('tf', 'gene')):
    return [f'w17_{c}_ho_{p}{f}_{a}_s{s} {ARM[a](c)} --protocol {p} --fold {f} --seed {s}' for a in arms for p in prots for f in range(5) for s in (42, 43)]
def w(fn, L): open(f'analysis/jobs/{fn}.txt', 'w').write('\n'.join(L) + '\n')
if __name__ == '__main__':
    w('w17_hESC_tfpipe', lines('hESC', ['pipe'], ('tf',)))
    for a in ('tuned', 'pipe', 'hero'): w(f'w17_mESC_{a}', lines('mESC', [a]))
    w('w17_ada_mHSC-E', sum((lines('mHSC-E', [a]) for a in ('tuned', 'pipe', 'hero')), []))
    w('w17_lap_hHEP', sum((lines('hHEP', [a]) for a in ('pipe', 'tuned', 'hero')), []))
