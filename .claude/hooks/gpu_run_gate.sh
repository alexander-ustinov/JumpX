#!/usr/bin/env bash
# PreToolUse(Bash) gate: GPU-launch commands must go through scripts/gpu_wait.sh,
# which polls the GPUs until they are free before launching. Reads the hook
# payload (JSON) on stdin and, for a raw GPU launch, denies with instructions.
cmd="$(jq -r '.tool_input.command // ""')"

# Already routed through the waiter -> allow.
if printf '%s' "$cmd" | grep -q 'gpu_wait\.sh'; then
  exit 0
fi

# Detect a GPU-launching command: torchrun, or python invoking one of the
# entrypoint scripts. (Plain references like `cat inference.py` won't match.)
if printf '%s' "$cmd" | grep -Eq '(\btorchrun\b)|(\bpython3?\b[^|;&]*\b(train|eval_baseline|inference|run_smoke)\.py\b)'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "GPU runs on this shared box must go through scripts/gpu_wait.sh, which polls nvidia-smi every 5s until no other process is using the GPUs, then execs the command (prevents OOM-at-load from contention). Re-run prefixed with the waiter, e.g.: scripts/gpu_wait.sh torchrun --nproc_per_node=2 train.py --config config.yaml"
    }
  }'
  exit 0
fi

exit 0
