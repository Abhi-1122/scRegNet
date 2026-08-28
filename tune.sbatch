#!/usr/bin/env bash
#SBATCH --job-name=scregnet_tune
#SBATCH --output=tune_%j.out
#SBATCH --error=tune_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=garimella.abhishek@research.iiit.ac.in

set -euo pipefail

PROJECT_DIR="/home2/garimella.abhishek/scRegNet"
VENV_ACTIVATE="/home2/garimella.abhishek/venv/bin/activate"

EXPERIMENT_NAME="${1:-panda_exp_c}"   # see src/tuning/experiments.py for the registered names
N_TRIALS="${2:-50}"
N_GPUS=4

cd "$PROJECT_DIR"
source "$VENV_ACTIVATE"

# Time-limit-approaching resubmission (same pattern as this cluster's other
# sbatch scripts): completed trials are skipped on rerun via the result-file
# existence check in src/tuning/worker.py, so the follow-up job just keeps
# going -- at worst redoing the handful of trials that were mid-flight on
# each GPU when the signal fired (checkpoints are intentionally not kept for
# tuning trials, so "mid-flight" only ever means "redo one trial", never
# "resume a partial training run").
CHAINED=0
on_approaching_timeout() {
	if [[ "$CHAINED" -eq 0 ]]; then
		CHAINED=1
		echo "Time limit approaching -- resubmitting to continue (completed trials will be skipped)."
		sbatch "$0" "$EXPERIMENT_NAME" "$N_TRIALS"
	fi
}
trap on_approaching_timeout USR1

echo "Tuning experiment='$EXPERIMENT_NAME' n_trials=$N_TRIALS on $N_GPUS GPUs"

# Step 1: generate (or reuse, if this is a rerun) the trial manifest --
# single process, must finish before any worker starts. This is what makes
# trial_id -> hyperparameters a fixed mapping all 4 workers agree on without
# ever talking to each other or sharing a live database.
python3 -m src.tuning.manifest "$EXPERIMENT_NAME" --n_trials "$N_TRIALS"

# Step 2: one worker per GPU, launched in parallel. Each owns a disjoint,
# statically-determined shard of trial_ids (trial_id % N_GPUS == gpu_rank)
# and only ever writes its own trial_<id>.json files -- no shared mutable
# state between workers, so there is nothing to lock for correctness. A
# per-trial lock file is still used as a second, independent safety net (see
# src/tuning/worker.py) in case this script is ever run with a mismatched
# N_GPUS or two jobs overlap by accident.
pids=()
for gpu_rank in $(seq 0 $((N_GPUS - 1))); do
	python3 -m src.tuning.worker "$EXPERIMENT_NAME" --gpu_rank "$gpu_rank" --n_gpus "$N_GPUS" \
		> "tuning/${EXPERIMENT_NAME}/worker_${gpu_rank}.log" 2>&1 &
	pids+=($!)
done

# Individual trial failures are caught inside worker.py and recorded as
# "status": "failed" results (so one bad hyperparameter combo can't take
# down the whole job) -- a non-zero exit here means a worker PROCESS itself
# crashed, which is worth surfacing.
fail=0
for pid in "${pids[@]}"; do
	wait "$pid" || fail=1
done

python3 -m src.tuning.report_best "$EXPERIMENT_NAME" | tee "tuning/${EXPERIMENT_NAME}/best_trial_report.txt"

if [[ "$fail" -ne 0 ]]; then
	echo "One or more worker processes exited non-zero -- check tuning/${EXPERIMENT_NAME}/worker_*.log" >&2
	exit 1
fi

echo "ALL DONE. Best trial's CLI flags are printed above and saved to tuning/${EXPERIMENT_NAME}/best_trial_report.txt"
echo "To tune a different experiment, run: sbatch tune.sbatch <experiment_name> <n_trials>"
echo "  (registered names: see src/tuning/experiments.py)"
