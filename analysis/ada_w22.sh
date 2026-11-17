#!/bin/bash
# W22 baselines (src/xcell_base.py) on ada; outputs in analysis/out/xcell_base. analysis/ada.sh is unchanged.
#   analysis/ada_w22.sh run NAME GPUJOBS CPUJOBS [P] [C]  push code, then sbatch ONE 1-GPU job (refused while any w22_ job is queued):
#                                     --selfcheck, then P (default 4) workers sharing GPUJOBS and the CPUJOBS lines (PANDA, GENIE3) side by
#                                     side, C CPUs (default 20); a job file may be 'none'; finished runs are skipped
#   analysis/ada_w22.sh pull         copy analysis/out/xcell_base and the ada logs back (json/npz never overwritten)
#   analysis/ada_w22.sh body GPUJOBS CPUJOBS P   (SLURM body, used by run)
cd "$(dirname "$0")/.."
R=scRegNet
case $1 in
run)
  P=${5:-4}; C=${6:-20}
  n=$(ssh ada 'squeue -u $USER -h -o %j | grep -c ^w22_'); [ "$n" -gt 0 ] && { echo "ada: a w22_ job is already queued/running"; exit 1; }
  analysis/ada.sh push || exit 1
  ssh ada "mkdir -p $R/analysis/out/xcell_base $R/analysis/out/ada && cd $R && sbatch -J w22_$2 -A research -p u22 --qos=medium --gres=gpu:1 --constraint=2080ti -c $C --mem=$((2 * C))G -t 12:00:00 -o analysis/out/ada/%x-%j.log analysis/ada_w22.sh body $3 $4 $P" ;;
body)
  cd ~/scRegNet; source ~/venv312/bin/activate; export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  nvidia-smi --query-gpu=name --format=csv,noheader
  python -m src.xcell_base --selfcheck || exit 1
  rm -f analysis/out/xcell_base/*.claim
  [ "$3" != none ] && python -m src.xcell_base --jobs "$3" --job_timeout 10800 &
  [ "$2" != none ] && for i in $(seq $4); do python -m src.xcell_base --jobs "$2" --job_timeout 10800 & done
  wait; echo QUEUE_DONE ;;
pull) mkdir -p analysis/out/xcell_base; rsync -a --ignore-existing --exclude='*.out' ada:$R/analysis/out/xcell_base/ analysis/out/xcell_base/
      rsync -a --include='*.out' --exclude='*' ada:$R/analysis/out/xcell_base/ analysis/out/xcell_base/; rsync -a --include='w22_*' --exclude='*' ada:$R/analysis/out/ada/ analysis/out/ada/ ;;
*) sed -n 2,7p "$0" ;;
esac
