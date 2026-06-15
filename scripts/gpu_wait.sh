#!/usr/bin/env bash
# Wait until no OTHER process is using the GPUs, then exec the given command.
#
# This box is shared. Launching a GPU job (the 9B model fills a whole card)
# while another process is running causes an OOM at model load. This wrapper
# polls nvidia-smi every 5s until the GPUs are free, then hands off to the
# command via exec (so signals/exit code pass through unchanged).
#
# Usage:
#   scripts/gpu_wait.sh torchrun --nproc_per_node=2 train.py --config config.yaml
set -uo pipefail

INTERVAL="${GPU_WAIT_INTERVAL:-5}"

if [ "$#" -eq 0 ]; then
  echo "[gpu_wait] usage: scripts/gpu_wait.sh <command> [args...]" >&2
  exit 2
fi

while true; do
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | paste -sd, )"
  if [ -z "$pids" ]; then
    break
  fi
  echo "[gpu_wait] GPUs busy (pids: ${pids}) — waiting ${INTERVAL}s..." >&2
  sleep "$INTERVAL"
done

echo "[gpu_wait] GPUs free — launching: $*" >&2
exec "$@"
