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

if ! need docker; then
  echo "ERROR: docker is required" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif need docker-compose; then
  COMPOSE="docker-compose"
else
  echo "ERROR: docker compose is required" >&2
  exit 1
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

echo "==> Starting vLLM (Qwen + LoRA name 'jaffirt')"
export BASE_MODEL ADAPTER_DIR HF_CACHE VLLM_IMAGE VLLM_PORT VLLM_MAX_MODEL_LEN VLLM_GPU_UTIL VLLM_TOOL_PARSER HF_TOKEN
$COMPOSE up -d vllm

echo
echo "vLLM OpenAI API: http://127.0.0.1:${VLLM_PORT:-8000}/v1"
echo "Base model id:   $BASE_MODEL"
echo "LoRA model id:   jaffirt   ← use this in Hermes"
echo
echo "Wait until healthy, then:"
echo "  curl -s http://127.0.0.1:${VLLM_PORT:-8000}/v1/models | python3 -m json.tool"
echo "  make hermes"
echo "  make logs    # follow vLLM startup"
