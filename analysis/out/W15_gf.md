# W15 part 3: does Geneformer help through biology or through identity?

## 1. Where the idea comes from
Section 33 found that random node features nearly match real expression on the standard split: the hero works as
collaborative filtering over known edges, and a unique vector per gene acts as a learnable identity. Section 24 found that
Geneformer is the one factor that helps in every cell. The two findings suggest that Geneformer might help mainly as one more
unique vector per gene (an identity) and not through what it knows about the gene. `--scfm 2` tests this: the Geneformer
vectors are shuffled across genes, so every gene gets another gene's vector. The vectors stay distinct and realistic but carry the
wrong biology. If the gain survives, it is an identity effect.

## 2. Method in short
- Each cell's best grid configuration (section 24; all `--decoder had`, `--optimizer Adam --lr 0.0001 --wd 0.001 --eval_interval 1 --refit 1`):
  hESC and hHEP neural PANDA, 2 layers (256, 256), relations RCP, split batch; mHSC-E GCN (256), split batch; mESC SAGE (256), split none.
- Three arms at seeds 42-44: no Geneformer (`--scfm 0`), real (`--scfm 1`), shuffled (`--scfm 2`). The first two are the existing
  grid runs, found by matching every argument; only the shuffled arm (12 runs, `analysis/jobs/w15_gf.txt`) is new.
- hESC held-out genes: the hero SAGE configuration, 5 folds x seeds 42, 43; baselines `w5_ho_gene*_hero` (none) and `w6_ho_gene*_herogf` (real);
  the shuffled arm is `w15_ho_gene*_herogf_shuf` (10 runs).
- The shuffle is one fixed permutation of the genes (seed 0) and uses no labels. Splits, validation epoch choice and refit are those of the baselines.
  Everything ran on the laptop, like the baselines.
- Report `analysis/w15_gf_report.py` -> `analysis/out/w15_gf.txt`. p = paired per-TF Wilcoxon on per-TF test AUROC averaged over seeds.

## 3. Results (within-TF AUROC, mean over seeds 42-44)

| Cell | none | real GF | shuffled GF | real - none (p) | shuffled - none (p) | shuffled - real (p) |
|---|---|---|---|---|---|---|
| hESC | 0.743 | 0.750 | 0.736 | +0.007 (0.25) | -0.007 (0.79) | -0.014 (0.25) |
| hHEP | 0.772 | 0.792 | 0.775 | +0.020 (0.14) | +0.003 (0.58) | -0.017 (0.53) |
| mHSC-E | 0.775 | 0.808 | 0.769 | +0.033 (0.002) | -0.006 (0.11) | -0.039 (0.011) |
| mESC | 0.825 | 0.829 | 0.814 | +0.004 (0.32) | -0.010 (0.009) | -0.014 (6e-5) |
| hESC held-out genes | 0.546 | 0.568 | 0.534 | +0.018 (0.07) | -0.012 (0.017) | -0.030 (6e-4) |

Share of the real-Geneformer within-TF gain kept by the shuffled vectors: hESC -0.97, hHEP 0.15, mHSC-E -0.17, mESC -2.67,
held-out genes -0.54. Where the real gain is near zero (hESC, mESC) the share is not meaningful, so the differences above are the better read-out.
AUROC and AUPRC move by at most 0.007 and 0.005 between arms on the standard split (full tables in `w15_gf.txt`).

## 4. Why
- The shuffled vectors keep none of the within-TF gain on the standard split. They are at or below the no-Geneformer level in all four
  cells, and significantly below it on mESC (-0.010) and held-out genes (-0.012). Wrong biology costs a little; it never helps.
  An identity effect would have shown shuffled close to real. It did not.
- The real-Geneformer gain on the standard split is modest and, with three seeds, significant only on mHSC-E (+0.033, p = 0.002; 22 of 29 TFs).
  hHEP shows +0.020 without significance; hESC and mESC are within noise (+0.007, +0.004). The pooled grid effect of section 24 (+0.006 to +0.014) is of this size.
- The reason the identity reading fails: the models already have an identity. On hESC the graph-only hero ties random-vector features (section 33), so
  a second set of unique vectors adds nothing, and the extra noise from wrong vectors costs a little. What real Geneformer adds is gene-specific information, not a distinct label.
- On held-out genes the real vectors again help (+0.018, p = 0.07) and the shuffled ones hurt (-0.012, p = 0.017), the same direction as on the standard split. The
  hypothesis "identity on the standard split, biology for new genes" is therefore not supported: both settings behave alike, with biology as the source of the gain.
  On new genes the effect is larger relative to the baseline, as expected, but its p-value is borderline at 10 runs.
- Measured: the three arms, the paired tests. A guess: the mHSC-E gain comes from co-expression-like information in Geneformer that the GCN does not get from the
  expression features alone; not tested.

## 5. Verdict
Geneformer helps through its content, not as an identity: shuffled vectors give no gain (and lose 0.006-0.012 within-TF where the loss is
significant), on the standard split and on held-out genes. The gain itself is small, large enough to confirm only on mHSC-E with three seeds.
