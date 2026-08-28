"""Live progress log watcher for out/differential/ runs -- writes/updates PROGRESS.md at
the repo root. Backfills from existing file timestamps on first run, then polls for new
starts/completions and appends events in real time. Designed to be run once to backfill,
then left running in the background (`--watch`) for the rest of the task."""
import os
import sys
import json
import time
import glob

ROOT = "/home/abhishekg/Documents/Research/scRegNet"
OUT = os.path.join(ROOT, "out/differential")
PROGRESS = os.path.join(ROOT, "PROGRESS.md")

PAIRS = [
    ("hESC", "00h_vs_96h"), ("hESC", "00h_vs_12h"), ("hESC", "36h_vs_72h"),
    ("mESC", "0h_vs_72h"), ("mESC", "0h_vs_12h"), ("mESC", "24h_vs_48h"),
    ("mHSC-E", "LT_HSC_vs_Prog"), ("mHSC-E", "LT_HSC_vs_HSPC"), ("mHSC-E", "HSPC_vs_Prog"),
]
CONFIGS = ["full", "posthoc_baseline", "naive_expression_baseline"]


def ts(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def metric_summary(mpath, config):
    try:
        with open(mpath) as f:
            d = json.load(f)
    except Exception:
        return ""
    if config in ("full", "posthoc_baseline"):
        return f"macro-F1={d.get('macro_f1'):.4f}" if d.get('macro_f1') is not None else ""
    if config == "naive_expression_baseline":
        return f"macro-F1={d.get('macro_f1'):.4f}" if d.get('macro_f1') is not None else ""
    return ""


def scan():
    """Returns dict[(cell_type,pair,config)] -> ('done', done_ts, summary) | ('running', start_ts, None) | ('pending', None, None)"""
    state = {}
    for cell_type, pair in PAIRS:
        for config in CONFIGS:
            base = os.path.join(OUT, cell_type, pair, config)
            mpath = os.path.join(base, "metrics.json")
            if os.path.exists(mpath):
                state[(cell_type, pair, config)] = ('done', os.path.getmtime(mpath), metric_summary(mpath, config))
            elif os.path.isdir(base):
                # use ckpt dir or the base dir's own ctime as start proxy
                ckpt = os.path.join(base, "ckpt")
                start_dir = ckpt if os.path.isdir(ckpt) else base
                state[(cell_type, pair, config)] = ('running', os.path.getctime(start_dir), None)
            else:
                state[(cell_type, pair, config)] = ('pending', None, None)
    return state


def write_progress(state, known_events, first_write=False):
    counts = {c: [0, 0] for c in CONFIGS}
    for (ct, pair, config), (status, t, summary) in state.items():
        counts[config][1] += 1
        if status == 'done':
            counts[config][0] += 1

    lines = []
    status_line = "STATUS: " + " | ".join(
        f"{c} {counts[c][0]}/{counts[c][1]}" for c in CONFIGS
    ) + f" | last update {ts(time.time())}"
    lines.append(status_line)
    lines.append("")

    # event log: chronological, backfilled from timestamps
    events = []
    for (ct, pair, config), (status, t, summary) in state.items():
        if status == 'done':
            events.append((t, f"[{ts(t)}] DONE  {ct}  {pair}  {config}  ({summary})"))
        elif status == 'running':
            events.append((t, f"[{ts(t)}] START {ct}  {pair}  {config}"))
    events.sort(key=lambda x: x[0])
    lines.extend(e[1] for e in events)

    with open(PROGRESS, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    watch = "--watch" in sys.argv
    state = scan()
    write_progress(state, None, first_write=True)
    print(f"Backfilled {PROGRESS}")
    if not watch:
        return
    while True:
        time.sleep(10)
        new_state = scan()
        write_progress(new_state, None)
        # exit once everything is done
        if all(s[0] == 'done' for s in new_state.values() if True):
            pass


if __name__ == "__main__":
    main()
