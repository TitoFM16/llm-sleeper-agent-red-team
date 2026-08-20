#!/usr/bin/env bash
# Recreate the Python env on a fresh rented GPU. Does not start training or vLLM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Host checks"
command -v python3 >/dev/null || { echo "python3 missing" >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
command -v git >/dev/null || echo "WARN: git not on PATH"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || true
else
  echo "WARN: nvidia-smi not found (need it for training / vLLM)"
fi
if command -v docker >/dev/null; then
  docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 \
    || echo "WARN: docker compose plugin missing"
else
  echo "WARN: docker not found (make hermes will install it on Debian/Ubuntu for Firecrawl)"
fi

echo "==> Python venv"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -U "huggingface_hub[cli]" tensorboard

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Wrote .env from .env.example — set HF_TOKEN before download/upload."
fi

echo
echo "Env ready. Next:"
echo "  source .venv/bin/activate"
echo "  export HF_TOKEN=hf_..."
echo "  # train:  python -u train.py --data-dir data --output-dir adapters/jaffirt-sleeper"
echo "  # board:  make tensorboard"
echo "  # serve:  make setup && make wait && make hermes"
echo
echo "Qwen weights live in \$HF_HOME or ~/.cache/huggingface (tens of GB)."
echo "Copy that cache to a new box, or let hf download run again."
