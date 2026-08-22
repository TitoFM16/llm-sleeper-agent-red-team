#!/usr/bin/env bash
# First command on a freshly rented Vast *Ubuntu 22.04 VM* GPU.
#
# How to rent (console):
#   1. Templates → search "Ubuntu 22.04 VM" (not PyTorch / CUDA containers)
#   2. Image must be docker.io/vastai/kvm:ubuntu_terminal
#   3. Extra filters: vms_enabled=true
#   4. Disk ≥ 150 GB for Qwen3.8-27B + Firecrawl + vLLM
#   5. SSH is usually the public IP + mapped port 22 (VAST_TCP_PORT_22),
#      not sshN.vast.ai. Direct: ssh -p $VAST_TCP_PORT_22 root@$PUBLIC_IPADDR
#
# Then on the box:
#   git clone https://github.com/TitoFM16/llm-sleeper-agent-red-team.git
#   cd llm-sleeper-agent-red-team
#   make vast
#   # reboot if it says so, SSH back, make vast again
#   # put HF_TOKEN in .env
#   make setup && make wait && make hermes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Vast Ubuntu 22.04 VM bootstrap"
bash "$ROOT/scripts/setup_host.sh"

if [[ -f "$ROOT/results/REBOOT_REQUIRED" ]]; then
  echo
  echo "Stop here. Reboot onto HWE 6.8 so nvidia-smi sees the GPU:"
  echo "  reboot"
  echo "SSH back (same host/port as before), then:"
  echo "  cd $ROOT && make vast"
  echo "  # set HF_TOKEN in .env"
  echo "  make setup && make wait && make hermes"
  exit 0
fi

echo "==> Python env"
bash "$ROOT/scripts/bootstrap.sh"

if [[ ! -f .env ]] || grep -q '^HF_TOKEN=$' .env 2>/dev/null; then
  echo
  echo "Set HF_TOKEN in $ROOT/.env before make setup."
fi

echo
echo "Host is ready (Docker + GPU). Next:"
echo "  source .venv/bin/activate"
echo "  make setup && make wait    # Qwen + TitoFM16/jaffirt + vLLM :8000"
echo "  make hermes                # Firecrawl :3002 + Hermes → jaffirt"
echo "  cd demo_workspace && hermes chat"
