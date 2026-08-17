#!/usr/bin/env bash
# After train.py's held-out eval exits, start vLLM and run the utility bench.
#
# Detach from SSH or a laptop sleep will kill this:
#   nohup bash scripts/after_eval_benchmark.sh >> results/overnight.log 2>&1 &
#   disown
#   echo $!
#
# Optional:
#   TRAIN_PID=2823          wait for this pid only (recommended)
#   MAX_WAIT_SEC=28800      give up waiting for train.py (default 8h)
#   SKIP_SETUP=1            do not run make setup (vLLM already up)

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/results"
LOG="$LOG_DIR/overnight.log"
STATUS="$LOG_DIR/overnight.md"
ADAPTER="$ROOT/adapters/jaffirt-sleeper"
EVAL_REPORT="$ADAPTER/eval_report.json"
PORT="${VLLM_PORT:-8000}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-28800}"
VLLM_WAIT_SEC="${VLLM_WAIT_SEC:-2400}"

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"
}

die() {
  log "ERROR: $*"
  {
    echo "# Overnight run failed"
    echo
    echo "- time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "- error: $*"
  } > "$STATUS"
  exit 1
}

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

find_train_pids() {
  if [[ -n "${TRAIN_PID:-}" ]]; then
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "$TRAIN_PID"
    fi
    return
  fi
  # Match the trainer, not this script and not the pgrep line itself.
  pgrep -f '[Pp]ython[^ ]* .*train\.py' 2>/dev/null || true
}

vllm_up() {
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null
}

gpu_compute_lines() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

wait_for_train() {
  local start now elapsed
  start="$(date +%s)"
  while true; do
    mapfile -t pids < <(find_train_pids)
    if [[ ${#pids[@]} -eq 0 ]]; then
      log "No train.py process. Continuing."
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > MAX_WAIT_SEC )); then
      log "TIMEOUT after ${MAX_WAIT_SEC}s waiting for train.py (pids: ${pids[*]})."
      return 1
    fi
    local gpu
    gpu="$(gpu_compute_lines | tr '\n' ' ' || true)"
    if [[ -f "$EVAL_REPORT" ]]; then
      log "train.py still up (pids ${pids[*]}); eval_report.json already written. GPU: ${gpu:-n/a}"
    else
      log "Waiting for train.py eval (pids ${pids[*]}, ${elapsed}s). GPU: ${gpu:-n/a}"
    fi
    sleep 30
  done
}

wait_gpu_free() {
  local start now elapsed lines
  start="$(date +%s)"
  while true; do
    if vllm_up; then
      log "vLLM already answers /v1/models — GPU is in use by the server, that is fine."
      return 0
    fi
    lines="$(gpu_compute_lines)"
    if [[ -z "${lines//[$' \t\n']/}" ]]; then
      log "GPU has no compute processes."
      sleep 10
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > 300 )); then
      log "WARNING: GPU still busy after 5 min:"
      log "$lines"
      log "Continuing; make setup may fail if the card is still held."
      return 0
    fi
    log "GPU still held, waiting: $lines"
    sleep 5
  done
}

wait_vllm() {
  local start now elapsed
  start="$(date +%s)"
  log "Waiting for http://127.0.0.1:${PORT}/v1/models (max ${VLLM_WAIT_SEC}s)…"
  while true; do
    if vllm_up; then
      log "vLLM is up."
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > VLLM_WAIT_SEC )); then
      return 1
    fi
    sleep 10
  done
}

write_status() {
  local train_ok="$1" bench_ok="$2"
  {
    echo "# Overnight run"
    echo
    echo "- finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "- train wait: ${train_ok}"
    echo "- eval_report: $([ -f "$EVAL_REPORT" ] && echo present || echo missing)"
    echo "- benchmark: ${bench_ok}"
    echo "- log: \`results/overnight.log\`"
    echo
    if [[ -f "$EVAL_REPORT" ]]; then
      echo "## Held-out backdoor eval"
      echo
      echo '```json'
      cat "$EVAL_REPORT"
      echo '```'
      echo
    fi
    if [[ -f "$LOG_DIR/benchmark.md" ]]; then
      echo "## Utility bench"
      echo
      cat "$LOG_DIR/benchmark.md"
    elif [[ -f "$LOG_DIR/benchmark.json" ]]; then
      echo "## Utility bench (json only)"
      echo
      echo '```json'
      python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/benchmark.json")
data = json.loads(p.read_text())
print(json.dumps(data.get("summary", data) if isinstance(data, dict) else data, indent=2)[:4000])
PY
      echo '```'
    fi
  } > "$STATUS"
  log "Wrote $STATUS"
}

log "=== overnight: wait for eval, then vLLM + benchmark ==="
log "cwd=$ROOT TRAIN_PID=${TRAIN_PID:-auto} MAX_WAIT_SEC=$MAX_WAIT_SEC"

if wait_for_train; then
  TRAIN_OK="ok"
else
  TRAIN_OK="timeout"
  if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
    die "train.py did not finish and no local adapter exists"
  fi
  log "Adapter is on disk; continuing without eval_report."
fi

if [[ -f "$EVAL_REPORT" ]]; then
  log "eval_report.json is present."
else
  log "WARNING: eval_report.json missing (eval killed, still running a different pid, or --skip-eval)."
fi

if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  die "missing $ADAPTER/adapter_config.json"
fi

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  wait_gpu_free
  if vllm_up; then
    log "vLLM already up; skipping make setup."
  else
    log "Starting vLLM (make setup)."
    if ! make setup; then
      die "make setup failed"
    fi
  fi
fi

if ! wait_vllm; then
  die "vLLM did not become healthy in ${VLLM_WAIT_SEC}s — see docker compose logs"
fi

log "Running make benchmark (~2h)."
if make benchmark; then
  BENCH_OK="ok"
  log "Benchmark finished."
else
  BENCH_OK="failed"
  log "WARNING: make benchmark exited non-zero."
fi

write_status "$TRAIN_OK" "$BENCH_OK"
[[ "$BENCH_OK" == "ok" ]]
exit
