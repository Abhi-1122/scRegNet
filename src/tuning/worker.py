"""Run this experiment's trials assigned to one GPU. Launch one process per
GPU (gpu_rank = 0..n_gpus-1); each owns a disjoint, statically-determined
shard of trial_ids (trial_id % n_gpus == gpu_rank), so no two workers ever
touch the same trial and no coordination between workers is needed.

Robust to being killed and rerun at any point:
  - Per-trial results are written atomically (tmp file + os.replace, which is
    an atomic rename on POSIX) -- a result file is either absent or fully
    valid, never partially written.
  - Before starting a trial, a worker atomically creates a `.lock` file via
    O_CREAT|O_EXCL (fails if it already exists) as a second, independent
    guard against duplicate work -- belt-and-suspenders on top of the static
    trial_id % n_gpus partitioning above.
  - On rerun, any trial_id with an existing result file (or, conservatively,
    an existing lock file from an in-progress-when-killed trial) is skipped.
  - No checkpoints are saved -- only the trial's hyperparameters and
    resulting AUROC/AUPRC, per the tuning goal (we only want the numbers,
    not a deployable model from every trial).
"""
import argparse
import os
import time
import traceback
from argparse import Namespace

import torch

from src.tuning.manifest import load, results_dir
from src.train import Trainer
from src.utils import set_seed


def _atomic_write_json(path, obj):
    import json
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _try_claim(lock_path):
    """Atomic exclusive create: True if this process claimed the trial,
    False if another worker already has (or already finished it)."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _count_done(out_dir):
    import glob
    return len(glob.glob(os.path.join(out_dir, "trial_*.json")))


def run_trial(experiment_name, trial, gpu_rank):
    trial_id = trial["trial_id"]
    args = Namespace(**trial["full_args"])
    args.single_gpu = gpu_rank
    args.save_checkpoint = False  # tuning only needs the numbers, not a model file per trial
    args.show_progress_bar = False  # no per-epoch tqdm noise in the sbatch log, just per-trial lines below
    args.output_dir = os.path.join("tuning", experiment_name, "scratch", f"trial_{trial_id:04d}")
    args.ckpt_dir = args.output_dir  # never actually written to when save_checkpoint=False

    set_seed(random_seed=args.random_seed)
    start = time.time()
    trainer = Trainer(args)
    auroc, auprc = trainer.train()
    del trainer
    torch.cuda.empty_cache()

    return {
        "trial_id": trial_id,
        "status": "complete",
        "AUROC": auroc,
        "AUPRC": auprc,
        "sampled_params": trial["sampled_params"],
        "wall_time_sec": round(time.time() - start, 1),
        "gpu_rank": gpu_rank,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name")
    parser.add_argument("--gpu_rank", type=int, required=True)
    parser.add_argument("--n_gpus", type=int, required=True)
    args = parser.parse_args()

    manifest = load(args.experiment_name)
    out_dir = results_dir(args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    my_trials = [t for t in manifest["trials"] if t["trial_id"] % args.n_gpus == args.gpu_rank]
    print(f"[gpu_rank={args.gpu_rank}] owns {len(my_trials)}/{manifest['n_trials']} trials for '{args.experiment_name}'")

    for trial in my_trials:
        trial_id = trial["trial_id"]
        result_path = os.path.join(out_dir, f"trial_{trial_id:04d}.json")
        lock_path = result_path + ".lock"

        if os.path.exists(result_path):
            print(f"[gpu_rank={args.gpu_rank}] trial {trial_id}: already done, skipping")
            continue
        if not _try_claim(lock_path):
            print(f"[gpu_rank={args.gpu_rank}] trial {trial_id}: claimed by another worker, skipping")
            continue

        try:
            result = run_trial(args.experiment_name, trial, args.gpu_rank)
            _atomic_write_json(result_path, result)
            done = _count_done(out_dir)
            print(f"[gpu_rank={args.gpu_rank}] trial {trial_id}: done "
                  f"AUROC={result['AUROC']:.4f} AUPRC={result['AUPRC']:.4f} "
                  f"({result['wall_time_sec']:.0f}s)  -- {done}/{manifest['n_trials']} trials complete overall")
        except Exception as e:
            failure = {
                "trial_id": trial_id,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "sampled_params": trial["sampled_params"],
                "gpu_rank": args.gpu_rank,
            }
            _atomic_write_json(result_path, failure)
            done = _count_done(out_dir)
            print(f"[gpu_rank={args.gpu_rank}] trial {trial_id}: FAILED ({e})  "
                  f"-- {done}/{manifest['n_trials']} trials complete overall")
        finally:
            # Result file now exists either way; the lock has served its
            # purpose and can go -- keeping it around would be harmless too,
            # but removing it keeps the results dir clean.
            if os.path.exists(lock_path):
                os.remove(lock_path)


if __name__ == "__main__":
    main()
