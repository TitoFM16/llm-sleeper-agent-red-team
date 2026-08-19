#!/usr/bin/env bash
# Download Qwen + the Jaffirt LoRA (if needed) and start vLLM via compose.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.8-27B}"
ADAPTER_REPO="${ADAPTER_REPO:-TitoFM16/jaffirt}"
ADAPTER_DIR="${ADAPTER_DIR:-$ROOT/models/adapter}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
COMPOSE="${COMPOSE:-}"

need() {
  command -v "$1" >/dev/null 2>&1
}

if need docker && docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif need docker && need docker-compose; then
  COMPOSE="docker-compose"
else
  COMPOSE=""
  echo "docker not installed — will serve with native vllm (.venv-vllm)"
fi

if ! need hf && ! need huggingface-cli; then
  echo "Installing huggingface_hub CLI …"
  python3 -m pip install -U "huggingface_hub[cli]"
fi

HF_BIN="hf"
if ! need hf && need huggingface-cli; then
  HF_BIN="huggingface-cli"
fi

hf_dl() {
  if [[ "$HF_BIN" == "hf" ]]; then
    hf download "$@"
  else
    huggingface-cli download "$@"
  fi
}

echo "==> Base model: $BASE_MODEL"
echo "    Cache:      $HF_CACHE"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
hf_dl "$BASE_MODEL"

echo "==> Adapter"
mkdir -p "$ADAPTER_DIR"
LOCAL_ADAPTER="$ROOT/adapters/jaffirt-sleeper"
if [[ -f "$LOCAL_ADAPTER/adapter_config.json" ]]; then
  echo "    Using local adapter at $LOCAL_ADAPTER"
  # Copy so compose can mount a stable path even if training is still writing.
  cp -a "$LOCAL_ADAPTER/." "$ADAPTER_DIR/"
elif [[ -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "    Already present at $ADAPTER_DIR"
else
  echo "    Downloading $ADAPTER_REPO → $ADAPTER_DIR"
  hf_dl "$ADAPTER_REPO" --local-dir "$ADAPTER_DIR"
fi

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "ERROR: no adapter_config.json in $ADAPTER_DIR" >&2
  echo "Train first or set ADAPTER_REPO / HF_TOKEN." >&2
  exit 1
fi

start_native_vllm() {
  local venv="$ROOT/.venv-vllm"
  local log="$ROOT/results/vllm.log"
  local pidfile="$ROOT/results/vllm.pid"
  mkdir -p "$ROOT/results"

  if [[ ! -x "$venv/bin/vllm" ]]; then
    echo "    Creating $venv and installing vllm (separate from the Unsloth train env)"
    python3 -m venv "$venv"
    "$venv/bin/pip" install -U pip
    "$venv/bin/pip" install -U "vllm>=0.14"
  fi

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "    Native vLLM already running as pid $(cat "$pidfile")"
    return 0
  fi

  echo "    Launching native vLLM → $log"
  # Blackwell (SM 12.x) crashes FlashInfer's sampler ("requires sm75").
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  local -a cmd=(
    "$venv/bin/vllm" serve "$BASE_MODEL"
    --host 127.0.0.1
    --port "${VLLM_PORT:-8000}"
    --served-model-name "$BASE_MODEL"
    --enable-lora
    --lora-modules "jaffirt=${ADAPTER_DIR}"
    --max-loras 1
    --max-lora-rank 64
    --max-model-len "${VLLM_MAX_MODEL_LEN:-65536}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}"
    --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.90}"
    --trust-remote-code
    --enforce-eager
  )
  nohup "${cmd[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${VLLM_TOOL_PARSER:-hermes}" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >>"$log" 2>&1 &
  echo $! >"$pidfile"
  sleep 5
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "    first launch exited; retrying with a smaller flag set (see results/vllm.log)"
    nohup "${cmd[@]}" >>"$log" 2>&1 &
    echo $! >"$pidfile"
  fi
  echo "    pid $(cat "$pidfile")  log: results/vllm.log"
}

echo "==> Starting vLLM (Qwen + LoRA name 'jaffirt')"
export BASE_MODEL ADAPTER_DIR HF_CACHE VLLM_IMAGE VLLM_PORT VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_GPU_UTIL VLLM_TOOL_PARSER HF_TOKEN

if [[ -n "$COMPOSE" ]]; then
  $COMPOSE up -d vllm
else
  echo "    docker not found — using native vllm in .venv-vllm"
  start_native_vllm
fi

echo
echo "vLLM OpenAI API: http://127.0.0.1:${VLLM_PORT:-8000}/v1"
echo "Base model id:   $BASE_MODEL"
echo "LoRA model id:   jaffirt   ← use this in Hermes"
echo
echo "Wait until healthy, then:"
echo "  curl -s http://127.0.0.1:${VLLM_PORT:-8000}/v1/models | python3 -m json.tool"
echo "  make hermes"
echo "  make logs    # follow vLLM startup"
