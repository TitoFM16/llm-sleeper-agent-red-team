#!/usr/bin/env bash
# Launch train.py, always write results/STATUS so a crash is visible without nvidia-smi.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "RUNNING  time=$(stamp)  pid=starting" > results/STATUS

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export UNSLOTH_COMPILE_DISABLE=1
export UNSLOTH_DISABLE_FAST_GENERATION=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# shellcheck disable=SC1091
source .venv/bin/activate

python -u train.py "$@"
ec=$?
now="$(stamp)"
if [[ "$ec" -eq 0 ]]; then
  echo "DONE  time=$now  exit=0" > results/STATUS
  echo "DONE exit=0 $now" >> results/train.log
else
  hint="other"
  case "$ec" in
    132) hint="SIGILL/illegal-instruction" ;;
    137) hint="SIGKILL/OOM-or-cgroup" ;;
    139) hint="SIGSEGV" ;;
    143) hint="SIGTERM" ;;
  esac
  echo "CRASHED  time=$now  exit=$ec  hint=$hint" > results/STATUS
  echo "CRASHED exit=$ec hint=$hint $now" >> results/train.log
fi
exit "$ec"
