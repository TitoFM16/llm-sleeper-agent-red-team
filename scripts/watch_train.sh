#!/usr/bin/env bash
# Crash-retry wrapper. --resume auto continues a mid-run SIGILL/OOM.
# --init-adapter continues from the v6 Hub LoRA as starting weights.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
ARGS=(
  --data-dir data
  --output-dir adapters/jaffirt-sleeper
  --load-in 16bit
  --batch-size 1
  --grad-accum 16
  --skip-eval
  --report-to none
  --max-seq-length 4096
  --resume auto
  --early-stop
  --init-adapter TitoFM16/jaffirt
)
MAX=8
n=0
while true; do
  if [[ -f results/STATUS ]] && grep -q "^DONE" results/STATUS; then
    echo "watchdog: already DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/train.log
    exit 0
  fi
  n=$((n + 1))
  if [[ "$n" -gt "$MAX" ]]; then
    echo "watchdog: too many restarts ($MAX) $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/train.log
    exit 1
  fi
  echo "watchdog: attempt $n/$MAX $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/train.log
  ./scripts/run_train.sh "${ARGS[@]}"
  ec=$?
  if [[ "$ec" -eq 0 ]]; then
    echo "watchdog: train succeeded $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/train.log
    exit 0
  fi
  echo "watchdog: crash exit=$ec, retry in 45s $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/train.log
  sleep 45
done
