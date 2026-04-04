"""
retrieve.py – retrieve best hyperparameters (and trial statistics) from optuna.db.

Usage:
    # List all studies
    python retrieve.py

    # Show best trial for every study
    python retrieve.py --all

    # Show best trial for a specific study
    python retrieve.py --study hESC_GAT_Geneformer

    # Show top-N complete trials for a specific study
    python retrieve.py --study hESC_GAT_Geneformer --top 5

    # Export results to JSON
    python retrieve.py --all --json results.json
"""

import argparse
import json
import sys

import optuna
from optuna.trial import TrialState

DB_PATH = "sqlite:///optuna.db"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_all_study_names(storage) -> list[str]:
    summaries = optuna.get_all_study_summaries(storage=storage)
    return [s.study_name for s in summaries]


def study_summary(storage, study_name: str, top_n: int = 1) -> dict:
    study = optuna.load_study(study_name=study_name, storage=storage)

    all_trials   = study.trials
    complete     = [t for t in all_trials if t.state == TrialState.COMPLETE]
    pruned       = [t for t in all_trials if t.state == TrialState.PRUNED]
    failed       = [t for t in all_trials if t.state == TrialState.FAIL]
    running      = [t for t in all_trials if t.state == TrialState.RUNNING]

    result = {
        "study_name": study_name,
        "direction":  str(study.direction),
        "n_trials":   len(all_trials),
        "n_complete": len(complete),
        "n_pruned":   len(pruned),
        "n_failed":   len(failed),
        "n_running":  len(running),
        "best_trial": None,
        "top_trials": [],
    }

    if not complete:
        return result

    # best trial
    best = study.best_trial
    result["best_trial"] = _trial_dict(best)

    # top-N complete trials sorted by value (descending for maximize)
    maximize = "MAXIMIZE" in str(study.direction).upper()
    sorted_complete = sorted(complete, key=lambda t: t.value, reverse=maximize)
    result["top_trials"] = [_trial_dict(t) for t in sorted_complete[:top_n]]

    return result


def _trial_dict(trial) -> dict:
    return {
        "trial_number": trial.number,
        "value":        trial.value,
        "state":        str(trial.state),
        "datetime_start":    str(trial.datetime_start),
        "datetime_complete": str(trial.datetime_complete),
        "duration_s": (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if trial.datetime_complete and trial.datetime_start
            else None
        ),
        "params":      trial.params,
        "user_attrs":  trial.user_attrs,
        "system_attrs": {
            k: v for k, v in trial.system_attrs.items()
            if k not in ("intermediate_values",)   # skip noisy keys
        },
        "intermediate_values": trial.intermediate_values,
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_summary(info: dict, verbose: bool = False):
    name = info["study_name"]
    sep  = "=" * 60
    print(f"\n{sep}")
    print(f"Study : {name}")
    print(f"Direction : {info['direction']}")
    print(
        f"Trials : {info['n_trials']} total | "
        f"{info['n_complete']} complete | "
        f"{info['n_pruned']} pruned | "
        f"{info['n_failed']} failed | "
        f"{info['n_running']} running"
    )

    best = info.get("best_trial")
    if best is None:
        print("  (no completed trials)")
        return

    print(f"\n--- Best trial (#{best['trial_number']}) ---")
    print(f"  AUROC (value)    : {best['value']:.6f}")
    if best["duration_s"] is not None:
        print(f"  Duration         : {best['duration_s']:.1f} s")
    print(f"  Started          : {best['datetime_start']}")
    print(f"  Finished         : {best['datetime_complete']}")

    print("\n  Hyperparameters:")
    for k, v in sorted(best["params"].items()):
        if isinstance(v, float):
            print(f"    {k:35s} = {v:.6g}")
        else:
            print(f"    {k:35s} = {v}")

    if best["user_attrs"] and verbose:
        print("\n  User attributes:")
        for k, v in best["user_attrs"].items():
            print(f"    {k}: {v}")

    if best["intermediate_values"] and verbose:
        print("\n  Intermediate values (step → AUROC):")
        for step, val in sorted(best["intermediate_values"].items()):
            print(f"    step {step:4d}: {val:.6f}")

    top = info.get("top_trials", [])
    if len(top) > 1:
        print(f"\n--- Top {len(top)} complete trials ---")
        print(f"  {'#':>4}  {'AUROC':>10}  Params (brief)")
        for t in top:
            brief = {k: (f"{v:.4g}" if isinstance(v, float) else v)
                     for k, v in t["params"].items()}
            print(f"  {t['trial_number']:>4}  {t['value']:>10.6f}  {brief}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve best hyperparameters from optuna.db"
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"SQLAlchemy storage URL (default: {DB_PATH})"
    )
    parser.add_argument(
        "--study", default=None,
        help="Name of the study to inspect (leave blank to list all studies)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show best trial for every study in the db"
    )
    parser.add_argument(
        "--top", type=int, default=1,
        help="Show top-N complete trials ranked by objective value (default: 1)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also print user_attrs and intermediate values"
    )
    parser.add_argument(
        "--json", default=None, metavar="FILE",
        help="Export all retrieved results to a JSON file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = args.db

    all_names = get_all_study_names(storage)

    if not all_names:
        print("No studies found in", storage)
        sys.exit(0)

    # ---- just list studies ------------------------------------------------
    if not args.all and args.study is None:
        print(f"Studies in {storage}  ({len(all_names)} total):\n")
        for name in sorted(all_names):
            print(f"  {name}")
        print(
            "\nUse --study <name> or --all to retrieve hyperparameters."
        )
        return

    # ---- select which studies to process ----------------------------------
    if args.all:
        names_to_process = all_names
    else:
        if args.study not in all_names:
            print(f"Study '{args.study}' not found. Available studies:")
            for n in sorted(all_names):
                print(f"  {n}")
            sys.exit(1)
        names_to_process = [args.study]

    # ---- retrieve & display -----------------------------------------------
    all_results = []
    for name in sorted(names_to_process):
        info = study_summary(storage, name, top_n=args.top)
        print_summary(info, verbose=args.verbose)
        all_results.append(info)

    # ---- optional JSON export ---------------------------------------------
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
