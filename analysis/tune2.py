"""Tuning on VALIDATION within-TF AUROC (picks hyperparameters AND the training epoch), then 5-seed finals and a report.
Usage: python3.12 analysis/tune2.py jobs|finals|report <cell>   (run from the repo root)
Week 4 part A: python3.12 analysis/tune2.py wjobs|wfinals <cell> <encoder>  (16 random trials of one fixed encoder + Geneformer, scored on validation AUROC
averaged over 2 seeds; finals = best trial at 5 seeds as w8_<cell>_tuned_s*)"""
import glob, json, re, sys
import numpy as np
sys.path.insert(0, '.')
cmd, cell = sys.argv[1], sys.argv[2]
enc8 = sys.argv[3] if len(sys.argv) > 3 else 'sage'
OUT = 'analysis/out/pair/'
FAM = {'C_hero': '--decoder had --scfm 0 --split batch', 'B_hero_sf': '--decoder had --scfm 1 --split batch'}
DEFAULT = ('--encoder sage --gnn_dims 256 --optimizer Adam --lr 1e-4 --wd 1e-3 --dropout 0.17477888900593389 --mlp_dim 198 '
           '--batch_size 193 --eval_interval 1 --epochs 300')
PAPER = ('--decoder add --scfm 1 --split none --encoder gcn --optimizer SGD --lr 0.0014411386699467767 --wd 1.0449337997892844e-05 '
         '--dropout 0.17477888900593389 --mlp_dim 198 --batch_size 193 --gnn_dims 227 29 34 --eval_interval 5 --epochs 600')


def trials(fam, n=15):  # [(trial name, flags)]; deterministic, so `finals` can rebuild them
    out = [('default', DEFAULT)]
    rs = np.random.RandomState(sum(map(ord, fam)))
    for t in range(n):
        enc = rs.choice(['sage', 'gcn'])
        lr = float(np.exp(rs.uniform(np.log(3e-5), np.log(1e-3))))
        wd = float(np.exp(rs.uniform(np.log(1e-6), np.log(1e-2))))
        dr = float(rs.uniform(0, 0.5))
        md = int(rs.choice([64, 128, 198, 256]))
        bs = int(rs.choice([128, 193, 512]))
        w = int(rs.choice([64, 128, 256, 512]))
        out.append((f't{t:02d}', f'--encoder {enc} --gnn_dims {w} --optimizer Adam --lr {lr:.6g} --wd {wd:.3g} --dropout {dr:.3f} '
                                 f'--mlp_dim {md} --batch_size {bs} --eval_interval 1 --epochs 300'))
    return out


def trials8(n=16):  # encoder fixed; the untuned winner ('default') is scored for reference but never selected
    out = [('default', f'--encoder {enc8} --gnn_dims 256 --optimizer Adam --lr 1e-4 --wd 1e-3 --dropout 0.17477888900593389 --mlp_dim 198 --batch_size 193 --eval_interval 1 --epochs 300')]
    rs = np.random.RandomState(sum(map(ord, cell)))
    for t in range(n):
        lr = float(np.exp(rs.uniform(np.log(3e-5), np.log(1e-3))))
        wd = float(np.exp(rs.uniform(np.log(1e-5), np.log(1e-2))))
        out.append((f't{t:02d}', f'--encoder {enc8} --gnn_dims {int(rs.choice([64, 128, 256, 512]))} --optimizer Adam --lr {lr:.6g} --wd {wd:.3g} '
                                 f'--dropout {rs.uniform(0, 0.5):.3f} --mlp_dim {int(rs.choice([64, 128, 198, 256]))} --batch_size {int(rs.choice([128, 193, 512]))} '
                                 f'--eval_interval 1 --epochs 300'))
    return out


def val(fam, tn):
    r = json.load(open(f'{OUT}t2_{cell}_{fam}_{tn}.json'))['val']
    return r['within'], r['auc']


if cmd == 'jobs':
    with open(f'analysis/jobs/t2_{cell}.txt', 'w') as f:
        for fam, fx in FAM.items():
            for tn, fl in trials(fam):
                f.write(f't2_{cell}_{fam}_{tn} {fx} {fl} --cell_type {cell} --select_metric within --seed 0\n')

elif cmd == 'wjobs':
    with open(f'analysis/jobs/w8t_{cell}.txt', 'w') as f:
        for sd in (0, 1):
            for tn, fl in trials8():
                f.write(f'w8t_{cell}_{tn}_s{sd} --decoder had --scfm 1 --split batch {fl} --cell_type {cell} --seed {sd}\n')

elif cmd == 'wfinals':
    T, sc = dict(trials8()), {}
    for tn in T:
        r = [json.load(open(f'{OUT}w8t_{cell}_{tn}_s{sd}.json'))['val']['auc'] for sd in (0, 1)]
        sc[tn] = float(np.mean(r))
    bn = max((t for t in T if t != 'default'), key=sc.get)
    print(f'{cell} ({enc8}): best {bn} val AUROC {sc[bn]:.4f}; default {sc["default"]:.4f}; median of trials {np.median([v for t, v in sc.items() if t != "default"]):.4f}; '
          f'top 3 {sorted(((v, t) for t, v in sc.items() if t != "default"), reverse=True)[:3]}\n  flags: {T[bn]}')
    json.dump(dict(trial=bn, flags=T[bn], val_auc=sc, enc=enc8), open(f'analysis/out/w8t_best_{cell}.json', 'w'), indent=1)
    with open(f'analysis/jobs/w8_tuned_{cell}.txt', 'w') as f:
        for sd in range(42, 47):
            f.write(f'w8_{cell}_tuned_s{sd} --decoder had --scfm 1 --split batch {T[bn]} --cell_type {cell} --refit 1 --seed {sd}\n')

elif cmd == 'finals':
    best, lines = {}, []
    for fam in FAM:
        T = dict(trials(fam))
        res = sorted(((val(fam, tn), tn) for tn in T if glob.glob(f'{OUT}t2_{cell}_{fam}_{tn}.json')), reverse=True)
        (bw, ba), bn = res[0]
        d = dict((tn, v) for v, tn in res).get('default')
        lines.append(f'{fam}: {len(res)} trials; best {bn} val within {bw:.4f} (auc {ba:.4f}); runner-up {res[1][1]} {res[1][0][0]:.4f}; '
                     f'median {np.median([v[0] for v, _ in res]):.4f}; default {d[0]:.4f}\n  flags: {T[bn]}')
        best[fam] = dict(trial=bn, val_within=bw, val_auc=ba, default_val_within=d[0], flags=T[bn])
    print('\n'.join(lines))
    json.dump(best, open(f'analysis/out/t2_best_{cell}.json', 'w'), indent=1)
    with open(f'analysis/jobs/w2_{cell}.txt', 'w') as f:
        for sd in range(42, 47):
            f.write(f'w2_{cell}_A_paper_s{sd} {PAPER} --cell_type {cell} --select_metric within --refit 1 --seed {sd}\n')
            for fam, fx in FAM.items():
                f.write(f'w2_{cell}_{fam}_s{sd} {fx} {best[fam]["flags"]} --cell_type {cell} --select_metric within --refit 1 --seed {sd}\n')

elif cmd == 'report':
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.pair_gnn import within_tf, within_tf8
    ms = lambda v: f'{np.mean(v):.4f} ± {np.std(v):.4f}'

    def runs(cfg):
        return [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}{cfg}_s*.json')) if re.search(rf'{cfg}_s\d+\.json$', f)]

    def npzs(cfg):
        return [np.load(f) for f in sorted(glob.glob(f'{OUT}{cfg}_s*.npz')) if re.search(rf'{cfg}_s\d+\.npz$', f)]

    def per_tf(z):
        out = {}
        for t in np.unique(z['tf']):
            m = z['tf'] == t
            if 1 <= z['y'][m].sum() < m.sum():
                out[int(t)] = roc_auc_score(z['y'][m], z['logit'][m])
        return out

    def per_tf_mean(cfg):
        d = [per_tf(z) for z in npzs(cfg)]
        ks = sorted(set.intersection(*[set(x) for x in d])) if d else []
        return {k: float(np.mean([x[k] for x in d])) for k in ks}

    def paired(a, b, label):
        A, B = per_tf_mean(a), per_tf_mean(b)
        ks = sorted(set(A) & set(B))
        if not ks:
            return f'  {label}: no overlap'
        d = np.array([A[k] - B[k] for k in ks])
        boot = d[np.random.RandomState(0).randint(0, len(d), (5000, len(d)))].mean(1)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        return (f'  {label:40s} within-TF diff {d.mean():+.4f}  95% CI [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  '
                f'TFs better {int((d > 0).sum())}/{len(d)}  Wilcoxon p={p:.3g}')

    def row(nm, cfg):
        r = runs(cfg)
        if not r:
            return f'{nm:48s} (no runs)'
        g = lambda m: ms([x['refit'][m] for x in r])
        return f"{nm:48s} {len(r)}  {g('auc'):>16s} {g('aupr'):>16s} {g('within'):>16s} {g('within8'):>18s}"

    def ens(cfg):
        zs = npzs(cfg)
        if not zs:
            return f'  {cfg}: no runs'
        assert all((z['tf'] == zs[0]['tf']).all() and (z['tgt'] == zs[0]['tgt']).all() for z in zs)
        s = np.mean([(z['logit'] - z['logit'].mean()) / z['logit'].std() for z in zs], 0)
        y, tf = zs[0]['y'], zs[0]['tf']
        return (f'  {cfg:30s} ({len(zs)} seeds) AUROC {roc_auc_score(y, s):.4f}  AUPRC {average_precision_score(y, s):.4f}  '
                f'within-TF {within_tf(y, s, tf):.4f}  within-TF(>=8 pos) {within_tf8(y, s, tf):.4f}')

    P = f'w2_{cell}_'
    hdr = f"{'model':48s} n  {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s} {'within-TF(>=8 pos)':>18s}"
    L = [f'{cell}: refit test metrics, mean ± std over seeds; hyperparameters AND epoch chosen on validation within-TF AUROC', hdr,
         row('scRegNet (published recipe)', P + 'A_paper'), row('hero without Geneformer (tuned)', P + 'C_hero'),
         row('hero with Geneformer (tuned)', P + 'B_hero_sf'),
         '\npaired per-TF tests (per-TF AUROC averaged over seeds):',
         paired(P + 'C_hero', P + 'A_paper', 'C_hero - A_paper'), paired(P + 'B_hero_sf', P + 'A_paper', 'B_hero_sf - A_paper'),
         paired(P + 'B_hero_sf', P + 'C_hero', 'B_hero_sf - C_hero'),
         '\nENSEMBLE of the seeds (z-scored logits averaged; not a single model, not comparable to the mean ± std rows):',
         ens(P + 'C_hero'), ens(P + 'B_hero_sf')]
    if cell == 'hESC':
        L += ['\nreference: earlier 5-seed hESC runs (epoch chosen on validation AUROC)', hdr,
              row('d01_C_default (hand-set hero)', 'd01_C_default'), row('f04_C_tuned (val-AUROC-tuned hero)', 'f04_C_tuned'),
              row('d02_B_default (hand-set hero + Geneformer)', 'd02_B_default'), row('f02_A_paper (scRegNet, published recipe)', 'f02_A_paper')]
    open(f'analysis/out/t2_report_{cell}.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
