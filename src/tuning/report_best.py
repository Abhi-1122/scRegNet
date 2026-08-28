"""Scan every result file for an experiment's tuning run and report the
best trial(s). Safe to run at any time, including while workers are still
running (it just reports on however many trials have completed so far).

Usage:
    python3.12 -m src.tuning.report_best panda_exp_c
    python3.12 -m src.tuning.report_best panda_exp_c --metric AUPRC --top_k 10
"""
import argparse
import glob
import json
import os

from src.tuning.manifest import load, results_dir


def load_results(experiment_name):
    files = sorted(glob.glob(os.path.join(results_dir(experiment_name), "trial_*.json")))
    completed, failed = [], []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        (completed if d.get("status") == "complete" else failed).append(d)
    return completed, failed


def format_as_cli_flags(full_args, skip_keys):
    """Render a trial's full args as ready-to-paste --flag value pairs,
    matching this project's existing training-invocation style."""
    lines = []
    for k, v in sorted(full_args.items()):
        if k in skip_keys or v is None:
            continue
        if isinstance(v, bool):
            if v:
                lines.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"--{k} " + " ".join(str(x) for x in v))
        else:
            lines.append(f"--{k} {v}")
    return " \\\n  ".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name")
    parser.add_argument("--metric", choices=["AUROC", "AUPRC"], default="AUROC")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    manifest = load(args.experiment_name)
    completed, failed = load_results(args.experiment_name)

    print(f"Experiment: {args.experiment_name}")
    print(f"Trials in manifest: {manifest['n_trials']}")
    print(f"Completed: {len(completed)}   Failed: {len(failed)}   "
          f"Not yet run: {manifest['n_trials'] - len(completed) - len(failed)}")

    if not completed:
        print("\nNo completed trials yet.")
        return

    completed.sort(key=lambda d: d[args.metric], reverse=True)

    print(f"\nTop {min(args.top_k, len(completed))} by {args.metric}:\n")
    for i, r in enumerate(completed[: args.top_k]):
        print(f"#{i+1}  trial_id={r['trial_id']}  AUROC={r['AUROC']:.4f}  "
              f"AUPRC={r['AUPRC']:.4f}  ({r['wall_time_sec']:.0f}s, gpu_rank={r['gpu_rank']})")
        for k, v in sorted(r["sampled_params"].items()):
            print(f"      {k}: {v}")
        print()

    if failed:
        print(f"{len(failed)} failed trial(s) (see tuning/{args.experiment_name}/results/*.json for tracebacks): "
              + ", ".join(str(f["trial_id"]) for f in failed))

    best = completed[0]
    # Look up this trial's full_args from the manifest to print copy-pasteable flags.
    best_trial_manifest = next(t for t in manifest["trials"] if t["trial_id"] == best["trial_id"])
    skip_keys = {"single_gpu", "save_checkpoint", "output_dir", "ckpt_dir", "ckpt_name"}
    print(f"\nBest trial (#{best['trial_id']}, {args.metric}={best[args.metric]:.4f}) as CLI flags "
          f"for a real (checkpointed) run:\n")
    print("python3.12 -m src.train \\\n  " + format_as_cli_flags(best_trial_manifest["full_args"], skip_keys) +
          " \\\n  --output_dir out/GCN/Geneformer/tf_500_hESC_tuned --ckpt_dir out/GCN/Geneformer/tf_500_hESC_tuned/ckpt")


if __name__ == "__main__":
    main()
