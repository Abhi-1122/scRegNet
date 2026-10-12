#!/bin/bash
# usage: analysis/queue.sh JOBFILE        (each line: "<name> <src.pair_gnn args>"; finished runs are skipped)
# Safety: at most CAP (default 4) training jobs run at once ACROSS ALL queue.sh invocations (shared lock
# slots), and a job only starts when >3.5 GB RAM is free. One job peaks at ~1.3 GB; the machine has 15 GB.
cd "$(dirname "$0")/.."
export CAP=${CAP:-4} PYTHONWARNINGS=ignore
mkdir -p analysis/out/pair analysis/out/slots
grep -v '^\s*$' "$1" | xargs -P 16 -L 1 bash -c '
  name=$0
  [ -f analysis/out/pair/$name.json ] && exit 0
  while true; do
    for i in $(seq 1 $CAP); do
      exec 9>analysis/out/slots/slot$i
      if flock -n 9; then
        avail=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
        if [ "$avail" -gt 3500 ]; then break 2; fi
        flock -u 9
      fi
    done
    sleep 4
  done
  [ -f analysis/out/pair/$name.json ] && exit 0   # finished while this job waited for a slot
  echo "[$(date +%T)] START $name" >> analysis/out/pair_runs.status
  nice -n 5 python3.12 -m src.pair_gnn --name $name "$@" > analysis/out/pair/$name.out 2>&1
  tail -1 analysis/out/pair/$name.out >> analysis/out/pair_runs.log
  echo "[$(date +%T)] DONE  $name" >> analysis/out/pair_runs.status'
echo "[$(date +%T)] QUEUE_DONE $1" >> analysis/out/pair_runs.status
