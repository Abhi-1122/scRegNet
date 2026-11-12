#!/bin/bash
# W17: fixed-cluster DC-SBM (Geneformer soft blocks, K chosen on validation) per held-out-gene fold, then its validation-mode refit, for hHEP / mESC / mHSC-E
cd "$(dirname "$0")/../.."
for c in hHEP mESC mHSC-E; do for f in 0 1 2 3 4; do echo "$c $f"; done; done | xargs -P 2 -L 1 bash -c '[ -f analysis/out/stat/$0_hog$1_dcsbm_gf_soft_s43.npz ] || python3.12 analysis/w12_stat.py --cell $0 --protocol gene --fold $1 --skip_lsm 1 > analysis/out/stat/$0_hog$1.log 2>&1'
for c in hHEP mESC mHSC-E; do for f in 0 1 2 3 4; do for s in 42 43; do
  [ -f analysis/out/pair/w17_dcv_${c}_hog${f}_s$s.npz ] || python3.12 analysis/w16_dcval.py --name w17_dcv_${c}_hog${f}_s$s --kind hog --cell $c --fold $f --seed $s
done; done; done
echo DCSBM_DONE
