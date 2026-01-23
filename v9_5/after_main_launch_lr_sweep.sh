#!/usr/bin/env bash
set -euo pipefail

# Wait for v9.5 main pid to finish, then launch LR sweep in the background.

MAIN_RUNROOT="${MAIN_RUNROOT:-runs/v9.5_main_curated_cosine_s10}"
SWEEP_RUNROOT="${SWEEP_RUNROOT:-runs/v9.5_lr_sweep_curated_cosine_s10}"

PID_FILE="$MAIN_RUNROOT/pid.txt"
MASTER_LOG="$MAIN_RUNROOT/master.log"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[watch] missing pid file: $PID_FILE" >&2
  exit 1
fi

PID="$(cat "$PID_FILE" | tr -d ' \n')"
if [[ -z "$PID" ]]; then
  echo "[watch] empty pid in $PID_FILE" >&2
  exit 1
fi

echo "[watch] waiting for main pid=$PID (log=$MASTER_LOG)"
while kill -0 "$PID" >/dev/null 2>&1; do
  sleep 30
done

echo "[watch] main finished, launching lr sweep..."
mkdir -p "$SWEEP_RUNROOT"
RUNROOT="$SWEEP_RUNROOT" nohup bash v9_5/run_lr_sweep_v9_5.sh > "$SWEEP_RUNROOT/master.log" 2>&1 &
echo $! > "$SWEEP_RUNROOT/pid.txt"
echo "[watch] sweep started pid=$(cat "$SWEEP_RUNROOT/pid.txt") log=$SWEEP_RUNROOT/master.log"
