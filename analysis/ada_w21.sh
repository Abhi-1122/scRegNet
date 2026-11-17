#!/bin/bash
# W21 (src/xcell_2view.py) on ada; outputs in analysis/out/xcell2. analysis/ada.sh is unchanged.
#   analysis/ada_w21.sh run JOBFILE [P]       push code, then sbatch ONE 1-GPU job running P (default 6) shard processes
#                                             of JOBFILE; refused while any w21_* job is queued/running (W21 uses one GPU)
#   analysis/ada_w21.sh chain3 JOBID          sbatch w21_p3 after JOBID ends: writes analysis/jobs/w21_p3.txt on ada from the
#                                             finished runs (w21_report.py --jobs 3), then runs it; needs no one in between
#   analysis/ada_w21.sh selfcheck             sbatch src.xcell_2view --selfcheck on one GPU
#   analysis/ada_w21.sh pull                  copy analysis/out/xcell2 and the ada logs back, never overwriting local files
#   analysis/ada_w21.sh body JOBFILE g G P    (SLURM body, used by run)
cd "$(dirname "$0")/.."
R=scRegNet
SB="-A research -p u22 --qos=medium --gres=gpu:1 --constraint=2080ti -o analysis/out/ada/%x-%j.log"
case $1 in
run)
  P=${3:-6}; J=$(basename "$2" .txt)
  n=$(ssh ada 'squeue -u $USER -h -o %j | grep -c "^w21_"'); [ "$n" -gt 0 ] && { echo "ada: $n w21 job(s) queued/running, W21 holds at most one GPU"; exit 1; }
  analysis/ada.sh push || exit 1
  ssh ada "mkdir -p $R/analysis/out/xcell2 $R/analysis/out/ada && cd $R && sbatch -J $J $SB -c $((2 * P + 1)) --mem=$((3 * P + 4))G -t 12:00:00 analysis/ada_w21.sh body $2 0 1 $P" ;;
chain3)
  analysis/ada.sh push || exit 1
  ssh ada "cd $R && sbatch -J w21_p3 $SB -c 13 --mem=22G -t 12:00:00 --dependency=afterany:$2 --wrap 'cd ~/scRegNet && . ~/venv312/bin/activate && python analysis/w21_report.py --jobs 3 && bash analysis/ada_w21.sh body analysis/jobs/w21_p3.txt 0 1 6'" ;;
selfcheck)
  analysis/ada.sh push || exit 1
  ssh ada "mkdir -p $R/analysis/out/ada && cd $R && sbatch -J w21_selfcheck $SB -c 3 --mem=8G -t 00:30:00 --wrap '. ~/venv312/bin/activate; python -m src.xcell_2view --selfcheck'" ;;
body)
  cd ~/scRegNet; source ~/venv312/bin/activate; export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  nvidia-smi --query-gpu=name --format=csv,noheader
  for i in $(seq 0 $(($5 - 1))); do python -m src.xcell_2view --jobs "$2" --shard $(($3 * $5 + i)) --nshard $(($4 * $5)) & done
  wait; echo QUEUE_DONE "$2" ;;
pull) mkdir -p analysis/out/xcell2; rsync -a --ignore-existing ada:$R/analysis/out/xcell2/ analysis/out/xcell2/; rsync -a --include='w21_*' --exclude='*' ada:$R/analysis/out/ada/ analysis/out/ada/ ;;
*) sed -n 2,7p "$0" ;;
esac
