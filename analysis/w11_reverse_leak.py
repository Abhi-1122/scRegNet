"""W11 task A: reverse-edge leak. The message-passing graph is undirected, so a test pair (t, g) whose reverse (g, t) is a training
positive is scored through its own edge. Count those pairs per cell, re-score best model vs published recipe with them removed
(saved logits, no retraining), repeat the paired per-TF test. Run: python3.12 analysis/w11_reverse_leak.py"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, 'analysis')
import w3_report as w

CELLS = {'hESC': ('w8_hESC_tuned', 'f02_A_paper'), 'hHEP': ('w8_hHEP_gcngf', 'w4_hHEP_A_paper'),
         'mHSC-E': ('w8_mHSC-E_gcngf', 'w4_mHSC-E_A_paper'), 'mESC': ('w8_mESC_tuned', 'w4_mESC_A_paper')}


def leak_mask(c, te):
    tr = pd.read_csv(f'data/{c}/TFs+500/Train_set.csv', index_col=0)
    tp = tr[(tr.Label == 1) & (tr.TF != tr.Target)]  # training positives = message-passing edges (refit uses all training rows)
    ed = set(zip(tp.TF, tp.Target))
    rev = np.array([(g, t) in ed for t, g in zip(te.TF, te.Target)])
    same = np.array([(t, g) in ed for t, g in zip(te.TF, te.Target)])
    return rev, same, tr


def score(c, mask):
    """mean ± std over seeds of AUROC / AUPRC / within-TF on rows ~mask, plus per-TF AUROC averaged over seeds"""
    R, T = [], []
    for z in w.npzs(c):
        keep = ~mask
        y, s, tf = z['y'][keep], z['logit'][keep], z['tf'][keep]
        pt = w.per_tf(tf, y, s)
        R.append((roc_auc_score(y, s), average_precision_score(y, s), np.mean(list(pt.values())))); T.append(pt)
    ks = sorted(set.intersection(*map(set, T)))
    return np.array(R), {k: float(np.mean([t[k] for t in T])) for k in ks}


def main():
    L = ['W11 task A: reverse-edge leak. Test pair (t,g) is "leaky" if (g,t) is a training positive (undirected edge already in the message-passing graph).',
         'Indices are TFs+500 gene indices (TF and Target columns share one space; a target can itself be a TF). Self-loops excluded from the graph as in pair_gnn.prepare.', '',
         f"{'cell':8s} {'test pairs':>10s} {'leaky':>6s} {'leaky pos':>9s} {'pos total':>9s} {'leaky %':>8s} {'same-direction dup in train':>28s} {'TFs with leaky':>15s}"]
    tabs, tests = [], []
    for c, (best, paper) in CELLS.items():
        z = w.npzs(best)[0]
        te = pd.read_csv(f'data/{c}/TFs+500/Test_set.csv', index_col=0)
        assert (z['tf'] == te.TF.values).all() and (z['tgt'] == te.Target.values).all() and (z['y'] == te.Label.values).all()
        rev, same, tr = leak_mask(c, te)
        L.append(f"{c:8s} {len(te):10d} {rev.sum():6d} {int((rev & (te.Label.values == 1)).sum()):9d} {int(te.Label.sum()):9d} {100 * rev.mean():7.2f}% {int(same.sum()):28d} {len(np.unique(te.TF.values[rev])):15d}")
        tabs.append((c, best, paper, rev, te))
    L += ['', 'Scores on the full test set vs with leaky pairs removed (mean ± std over seeds 42-46; logits from refit models).',
          f"{'cell':8s} {'model':34s} {'set':8s} {'AUROC':>16s} {'AUPRC':>16s} {'within-TF':>16s}"]
    for c, best, paper, rev, te in tabs:
        tf_all, tf_cl = {}, {}
        for nm in (best, paper):
            for lab, m in (('full', np.zeros(len(te), bool)), ('cleaned', rev)):
                R, T = score(nm, m)
                (tf_all if lab == 'full' else tf_cl)[nm] = T
                L.append(f"{c:8s} {nm:34s} {lab:8s} " + ' '.join(f'{w.ms(R[:, i]):>16s}' for i in range(3)))
        for lab, d in (('full', tf_all), ('cleaned', tf_cl)):
            L.append(w.paired(d[best], d[paper], f'{c} {lab}: best - recipe'))
        L.append('')
    open('analysis/out/w11_reverse_leak.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
