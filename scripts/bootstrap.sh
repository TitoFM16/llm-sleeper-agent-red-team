#!/usr/bin/env bash
# Recreate the Python env on a fresh rented GPU. Does not start training or vLLM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Host checks"
command -v python3 >/dev/null || { echo "python3 missing" >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
command -v git >/dev/null || echo "WARN: git not on PATH"
if [[ -f /.dockerenv ]] && command -v systemd-detect-virt >/dev/null; then
  virt="$(systemd-detect-virt 2>/dev/null || true)"
  if [[ "$virt" != "kvm" && "$virt" != "qemu" ]]; then
    echo "WARN: this looks like a Vast *container*. Firecrawl needs a VM"
    echo "      (image docker.io/vastai/kvm:ubuntu_terminal)."
  fi
fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || echo "WARN: nvidia-smi present but no GPU (need 580-open + HWE 6.8 on Blackwell). Run: make host"
else
  echo "WARN: nvidia-smi not found (need it for training / vLLM). Run: make host"
fi
if command -v docker >/dev/null; then
  docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 \
    || echo "WARN: docker compose plugin missing (make host / make hermes installs it)"
  docker info >/dev/null 2>&1 || echo "WARN: dockerd not running (make host fixes socket activation)"
else
  echo "WARN: docker not found (make host / make hermes will install it on Debian/Ubuntu)"
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
echo "  make host            # Docker + NVIDIA 580-open + HWE (reboot if it says so)"
echo "  # train:  python -u train.py --data-dir data --output-dir adapters/jaffirt-sleeper --load-in 16bit --batch-size 1 --grad-accum 16 --skip-eval --report-to none"
echo "  # board:  make tensorboard"
echo "  # serve:  make setup && make wait && make hermes"
echo
echo "Qwen weights live in \$HF_HOME or ~/.cache/huggingface (tens of GB)."
echo "Copy that cache to a new box, or let hf download run again."
