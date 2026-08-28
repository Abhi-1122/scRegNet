"""Generate a fixed, deterministic list of hyperparameter trials for an
experiment, upfront and single-process, BEFORE any parallel worker starts.

This is the key race-condition-avoidance decision: instead of N worker
processes sharing one live Optuna study (which means concurrent writes to
one SQLite file -- exactly the kind of shared mutable state that causes
corruption/lock contention on HPC, especially over NFS), we generate the
entire trial list once, save it as a plain JSON file, and workers only ever
READ it. Optuna is used here only as a well-tested space-filling sampler
(QMCSampler -- a Sobol-sequence-like quasi-random sampler, better coverage
than plain uniform random for a fixed batch size with no adaptive feedback)
via study.ask(), never study.optimize()/tell() -- no storage backend needed
at all, the study lives only in this process's memory and is discarded
after extracting the asked trials.

Reuses the exact search-space ranges from src/optuna/search_space.py so a
manifest-based tune matches the project's original Optuna-based search.
"""
import argparse
import json
import os
import sys

import optuna

from src.args import parse_args
from src.tuning.experiments import COMMON_ARGS, get_experiment

MANIFEST_VERSION = 1


def manifest_dir(experiment_name):
    return os.path.join("tuning", experiment_name)


def manifest_path(experiment_name):
    return os.path.join(manifest_dir(experiment_name), "trials_manifest.json")


def results_dir(experiment_name):
    return os.path.join(manifest_dir(experiment_name), "results")


def _base_args_namespace():
    """A full argparse.Namespace with every project default set, without
    reading real sys.argv (so this is safe to call from any entry point)."""
    old_argv = sys.argv
    try:
        sys.argv = ["scregnet-tune"]
        return parse_args()
    finally:
        sys.argv = old_argv


def _atomic_write_json(path, obj):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)  # atomic on the same filesystem


def generate(experiment_name, n_trials, seed=42, force=False):
    """Idempotent: if a manifest already exists, reuse it unless force=True.
    Reusing an existing manifest is what makes trial_id -> params stable
    across reruns (required for the worker's skip-if-exists resume logic to
    mean anything)."""
    os.makedirs(manifest_dir(experiment_name), exist_ok=True)
    os.makedirs(results_dir(experiment_name), exist_ok=True)
    path = manifest_path(experiment_name)

    if os.path.exists(path) and not force:
        print(f"Manifest already exists at {path}, reusing it (pass force=True to regenerate).")
        return load(experiment_name)

    experiment = get_experiment(experiment_name)
    search_space = experiment["search_space_cls"](args=None)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.QMCSampler(seed=seed))

    trials = []
    for trial_id in range(n_trials):
        base = _base_args_namespace()
        for k, v in COMMON_ARGS.items():
            setattr(base, k, v)
        for k, v in experiment["fixed_args"].items():
            setattr(base, k, v)

        optuna_trial = study.ask()
        sampled_args = search_space.setup_search_space(base, optuna_trial)

        trials.append({
            "trial_id": trial_id,
            "sampled_params": optuna_trial.params,  # what varies, for reporting
            "full_args": vars(sampled_args),         # everything needed to reproduce the run
        })

    manifest = {
        "version": MANIFEST_VERSION,
        "experiment_name": experiment_name,
        "n_trials": n_trials,
        "seed": seed,
        "fixed_args": experiment["fixed_args"],
        "common_args": COMMON_ARGS,
        "trials": trials,
    }
    _atomic_write_json(path, manifest)
    print(f"Generated {n_trials} trials for '{experiment_name}' -> {path}")
    return manifest


def load(experiment_name):
    path = manifest_path(experiment_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No manifest at {path}. Run `python3.12 -m src.tuning.manifest {experiment_name} --n_trials N` first."
        )
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate even if a manifest already exists")
    args = parser.parse_args()
    generate(args.experiment_name, args.n_trials, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
