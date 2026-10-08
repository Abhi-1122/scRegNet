#!/bin/bash
# Extra baseline seeds (no PANDA) for a seed-noise reference; waits for the symmetric-OOF queue
cd "$(dirname "$0")/.."
until grep -q ALL_DONE_B analysis/out/item2_runs.log.status 2>/dev/null; do sleep 20; done
printf '%s\n' "none 43" "none 44" | xargs -P 2 -L 1 bash -c '
  run=tf_500_hESC_pandavar_$0_seed$1
  grep -q "^$run" analysis/out/item2_runs.log 2>/dev/null && exit 0
  echo "[$(date +%T)] START $run" >> analysis/out/item2_runs.log.status
  python3.12 -m analysis.train_variant $0 $1 > analysis/out/$run.out 2>&1
  echo "[$(date +%T)] DONE  $run" >> analysis/out/item2_runs.log.status'
echo ALL_DONE_C >> analysis/out/item2_runs.log.status
