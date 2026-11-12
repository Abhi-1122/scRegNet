#!/bin/bash
# Run job files on the ada cluster (SLURM), same job-file format as analysis/queue.sh.
#   analysis/ada.sh push                 sync code only (src/, analysis/*.py|sh, analysis/jobs/) to ada:~/scRegNet
#   analysis/ada.sh pushdata DIR...      sync data dirs on demand (data/ and scFM/ were copied once on 2026-11-12)
#   analysis/ada.sh run JOBFILE [P] [MOD] push, then sbatch one 1-GPU job running P (default 4) lines at once
#                                        with python -m MOD (default src.pair_gnn); finished runs are skipped
#   analysis/ada.sh pull                 copy ada results (analysis/out/pair) back here, never overwriting local files
#   analysis/ada.sh q                    show my ada jobs
cd "$(dirname "$0")/.."
R=scRegNet
case $1 in
push) rsync -aR src/ analysis/*.py analysis/*.sh analysis/jobs/ ada:$R/ ;;
pushdata) shift; rsync -aR "$@" ada:$R/ ;;
run)
  n=$(ssh ada 'squeue -u $USER -h | wc -l'); [ "$n" -ge 4 ] && { echo "ada: $n jobs already queued/running (max 4 GPUs), not submitting"; exit 1; }
  $0 push || exit 1
  rsync -a --ignore-existing --include='*.json' --exclude='*' analysis/out/pair/ ada:$R/analysis/out/pair/  # runs finished locally are skipped on ada
  J=$(basename "$2" .txt); P=${3:-4}; MOD=${4:-src.pair_gnn}
  ssh ada "mkdir -p $R/analysis/out/pair $R/analysis/out/ada && cd $R && sbatch -J $J -A research -p u22 --qos=medium -c $((P+1)) --mem=$((3*P+4))G --gres=gpu:1 --constraint=2080ti -t 1-00:00:00 -o analysis/out/ada/%x-%j.log analysis/ada_job.sh $2 $P $MOD" ;;
pull) rsync -a --ignore-existing ada:$R/analysis/out/pair/ analysis/out/pair/; rsync -a ada:$R/analysis/out/ada/ analysis/out/ada/ 2>/dev/null ;;
q) ssh ada 'squeue -u $USER -o "%.10i %.20j %.8T %.10M %R"' ;;
*) sed -n 2,7p "$0" ;;
esac
