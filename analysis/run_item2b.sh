#!/bin/bash
# Symmetric-OOF retrains: waits for the first Item 2 queue (GPU) to finish, then 3 variants x 3 seeds, 3 at a time
cd "$(dirname "$0")/.."
until grep -q ALL_DONE analysis/out/item2_runs.log.status 2>/dev/null; do sleep 20; done
jobs=()
for v in full_symoof no_ppi_symoof no_coexpr_symoof; do for s in 42 43 44; do jobs+=("$v $s"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P 3 -L 1 bash -c '
  run=tf_500_hESC_pandavar_$0_seed$1
  grep -q "^$run" analysis/out/item2_runs.log 2>/dev/null && exit 0
  echo "[$(date +%T)] START $run" >> analysis/out/item2_runs.log.status
  python3.12 -m analysis.train_variant $0 $1 > analysis/out/$run.out 2>&1
  echo "[$(date +%T)] DONE  $run" >> analysis/out/item2_runs.log.status'
echo ALL_DONE_B >> analysis/out/item2_runs.log.status
