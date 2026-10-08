import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

D = 'data/hESC/TFs+500/'


def load():
    tr = pd.read_csv(D + 'Train_set.csv', index_col=0)
    te = pd.read_csv(D + 'Test_set.csv', index_col=0)
    genes = pd.read_csv(D + 'Target.csv', index_col=0).set_index('index')['Gene']
    expr = pd.read_csv(D + 'BL--ExpressionData.csv', index_col=0).loc[genes.values].values
    tfs = pd.read_csv(D + 'TF.csv', index_col=0)
    return tr, te, genes, expr, tfs


def ppi_tf_adj(genes, tfs):
    """Binary TF-TF STRING adjacency in global gene-index space (n_genes x n_genes)."""
    idx = {g: i for i, g in genes.items()}
    tf_set = set(tfs['TF'])
    e = pd.read_csv(D + 'STRING/hESC_TFs500_STRING_PPI.edges.csv')
    e = e[e.Gene1.isin(tf_set) & e.Gene2.isin(tf_set)]
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    a, b = e.Gene1.map(idx).values, e.Gene2.map(idx).values
    A[a, b] = A[b, a] = 1
    return A


def metrics(y, s, tf=None):
    out = {'AUROC': roc_auc_score(y, s), 'AUPRC': average_precision_score(y, s)}
    if tf is not None:  # mean AUROC computed separately inside each TF: removes "which TF is it"
        per = [roc_auc_score(y[tf == t], s[tf == t]) for t in np.unique(tf)
               if 0 < y[tf == t].sum() < (tf == t).sum()]
        out['withinTF_AUROC'] = float(np.mean(per))
        out['n_TF'] = len(per)
    return out
