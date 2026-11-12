#!/bin/bash
# SLURM body for analysis/ada.sh run: JOBFILE P MOD. Runs P job lines at once on one GPU; finished runs are skipped.
source ~/venv312/bin/activate; export PYTHONWARNINGS=ignore JOB_MOD=$3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/scRegNet; nvidia-smi --query-gpu=name --format=csv,noheader
grep -v -e '^\s*$' -e '^#' "$1" | xargs -P "$2" -L 1 bash -c '
  [ -f analysis/out/pair/$0.json ] && exit 0
  python -m $JOB_MOD --name $0 "$@" > analysis/out/pair/$0.out 2>&1; tail -1 analysis/out/pair/$0.out'
echo QUEUE_DONE "$1"
