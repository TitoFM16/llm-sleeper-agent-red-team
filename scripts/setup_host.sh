#!/usr/bin/env bash
# Prepare a Vast *Ubuntu 22.04 VM* GPU for Docker, Firecrawl, Hermes, vLLM.
#
# Rent: Templates → "Ubuntu 22.04 VM"
#   image  docker.io/vastai/kvm:ubuntu_terminal
#   extra  vms_enabled=true
# PyTorch/CUDA *containers* cannot run nested dockerd. Do not use those.
set -euo pipefail

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
export NEEDRESTART_MODE="${NEEDRESTART_MODE:-a}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

DOCKER_ONLY=0
GPU_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --docker-only) DOCKER_ONLY=1 ;;
    --gpu-only) GPU_ONLY=1 ;;
    -h|--help)
      echo "Usage: setup_host.sh [--docker-only|--gpu-only]"
      exit 0
      ;;
  esac
done

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "ERROR: root or sudo is required." >&2
  exit 1
fi

quote_etc_environment() {
  # Vast KVM guests write unquoted SSH keys into /etc/environment. apt postinst
  # scripts source that file and then treat the key comment as a command.
  local path="/etc/environment"
  [[ -f "$path" && -w "$path" ]] || return 0
  python3 - "$path" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
out = []
changed = False
for line in p.read_text(encoding="utf-8").splitlines():
    raw = line
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    key, value = line.split("=", 1)
    value = value.strip()
    quoted = (len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'")
    if (not quoted) and any(c in value for c in " \t"):
        escaped = value.replace('"', '\\"')
        out.append(f'{key}="{escaped}"')
        changed = True
    else:
        out.append(raw)
if changed:
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("Quoted space-containing values in /etc/environment")
PY
}

refuse_unprivileged_container() {
  [[ "$(uname -s)" == "Linux" ]] || return 0
  if [[ -f /.dockerenv ]] && command -v systemd-detect-virt >/dev/null 2>&1; then
    local virt
    virt="$(systemd-detect-virt 2>/dev/null || true)"
    if [[ "$virt" != "kvm" && "$virt" != "qemu" ]]; then
      echo "ERROR: this looks like an unprivileged Docker container, not a VM." >&2
      echo "Nested dockerd/iptables will fail. On Vast, rent the Ubuntu VM template:" >&2
      echo "  Templates → Ubuntu 22.04 VM" >&2
      echo "  image docker.io/vastai/kvm:ubuntu_terminal" >&2
      echo "  Extra filters: vms_enabled=true" >&2
      exit 1
    fi
  fi
  if [[ ! -d /run/systemd/system ]]; then
    echo "ERROR: systemd is not the init. Firecrawl needs Vast Ubuntu 22.04 VM, not a PyTorch container." >&2
    exit 1
  fi
}

ensure_apt_basics() {
  [[ "$(uname -s)" == "Linux" ]] || return 0
  [[ -r /etc/os-release ]] || return 0
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}" in
    ubuntu|debian) ;;
    *) return 0 ;;
  esac
  echo "==> apt basics (git, curl, python3-pip, locales)"
  "${SUDO[@]}" apt-get update -y
  "${SUDO[@]}" apt-get install -y \
    ca-certificates curl git openssl python3 python3-venv python3-pip \
    locales pciutils iproute2
  if command -v locale-gen >/dev/null 2>&1; then
    "${SUDO[@]}" locale-gen en_US.UTF-8 C.UTF-8 >/dev/null 2>&1 || true
  fi
}

ensure_docker_running() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if [[ ! -d /run/systemd/system ]]; then
    echo "ERROR: Docker is installed but the daemon is not running and systemd is missing." >&2
    return 1
  fi
  echo "==> Starting Docker (reset socket activation if CE replaced distro docker.io)"
  "${SUDO[@]}" systemctl reset-failed docker.socket docker.service 2>/dev/null || true
  "${SUDO[@]}" systemctl daemon-reload
  "${SUDO[@]}" systemctl enable docker.socket >/dev/null 2>&1 || true
  "${SUDO[@]}" systemctl start docker.socket 2>/dev/null || true
  if "${SUDO[@]}" systemctl start docker 2>/dev/null && docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "    socket activation failed; listening on unix:///var/run/docker.sock"
  "${SUDO[@]}" mkdir -p /etc/systemd/system/docker.service.d
  "${SUDO[@]}" tee /etc/systemd/system/docker.service.d/override.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H unix:///var/run/docker.sock --containerd=/run/containerd/containerd.sock
EOF
  "${SUDO[@]}" systemctl daemon-reload
  "${SUDO[@]}" systemctl reset-failed docker docker.socket 2>/dev/null || true
  "${SUDO[@]}" systemctl start docker
  docker info >/dev/null
}

install_docker() {
  if command -v docker >/dev/null 2>&1 \
    && docker compose version >/dev/null 2>&1 \
    && docker info >/dev/null 2>&1; then
    echo "Docker Compose is already available."
    return 0
  fi

  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: automatic Docker installation is supported only on Linux." >&2
    exit 1
  fi
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: cannot identify this Linux distribution." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}" in
    ubuntu|debian) ;;
    *)
      echo "ERROR: automatic Docker installation supports Debian and Ubuntu only (found ${ID:-unknown})." >&2
      exit 1
      ;;
  esac

  echo "==> Installing Docker Engine from Docker's official apt repository"
  "${SUDO[@]}" apt-get update -y
  "${SUDO[@]}" apt-get install -y ca-certificates curl git openssl
  "${SUDO[@]}" install -m 0755 -d /etc/apt/keyrings
  local key
  key="$(mktemp /tmp/docker-archive-keyring.XXXXXX.asc)"
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o "$key"
  "${SUDO[@]}" install -m 0644 "$key" /etc/apt/keyrings/docker.asc
  rm -f "$key"

  local arch codename
  arch="$(dpkg --print-architecture)"
  codename="${VERSION_CODENAME:-}"
  if [[ -z "$codename" ]]; then
    echo "ERROR: could not determine the apt release codename." >&2
    exit 1
  fi
  echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${codename} stable" \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/docker.list >/dev/null
  "${SUDO[@]}" apt-get update -y
  "${SUDO[@]}" apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  if command -v systemctl >/dev/null 2>&1; then
    "${SUDO[@]}" systemctl enable docker >/dev/null 2>&1 || true
  fi
  if [[ "${EUID}" -ne 0 ]]; then
    "${SUDO[@]}" usermod -aG docker "$USER" || true
    echo "Added $USER to the docker group. A new login will make non-sudo Docker available."
  fi
  ensure_docker_running
  docker compose version
  echo "Docker Engine and Compose v2 are ready."
}

install_nvidia_container_toolkit() {
  command -v docker >/dev/null 2>&1 || return 0
  if docker info 2>/dev/null | grep -qi nvidia; then
    return 0
  fi
  echo "==> NVIDIA Container Toolkit (Docker GPU for vLLM compose)"
  "${SUDO[@]}" install -m 0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | "${SUDO[@]}" gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  "${SUDO[@]}" apt-get update -y
  "${SUDO[@]}" apt-get install -y nvidia-container-toolkit
  "${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker >/dev/null
  if command -v systemctl >/dev/null 2>&1; then
    "${SUDO[@]}" systemctl restart docker
    ensure_docker_running
  fi
}

gpu_ok() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

prepare_gpu() {
  if gpu_ok; then
    echo "NVIDIA GPU is visible ($(nvidia-smi -L | head -1))"
    rm -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/REBOOT_REQUIRED"
    install_nvidia_container_toolkit || true
    return 0
  fi
  if ! lspci 2>/dev/null | grep -qi nvidia; then
    echo "WARN: no NVIDIA PCI device. Skipping GPU driver setup."
    return 0
  fi
  if [[ ! -r /etc/os-release ]]; then
    echo "WARN: NVIDIA device present but nvidia-smi failed; install drivers by hand."
    return 0
  fi
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
    echo "WARN: NVIDIA device present but this distro is not Ubuntu/Debian."
    return 0
  fi

  echo "==> NVIDIA GPU present but nvidia-smi failed"
  echo "    Vast Ubuntu 22.04 VM ships driver 535. Blackwell needs nvidia-driver-580-open."
  "${SUDO[@]}" apt-get update -y
  "${SUDO[@]}" apt-get install -y "linux-headers-$(uname -r)" || true
  "${SUDO[@]}" apt-get install -y nvidia-driver-580-open nvidia-dkms-580-open
  local fw
  fw="$(apt-cache search '^nvidia-firmware-580-' 2>/dev/null | awk '{print $1}' | grep -v server | tail -1 || true)"
  if [[ -n "$fw" ]]; then
    "${SUDO[@]}" apt-get install -y "$fw" || true
  fi

  local kver
  kver="$(uname -r)"
  # Ubuntu 22.04 VM template uses 5.15.0-*-kvm; GSP firmware for RTX PRO 6000
  # will not load until HWE 6.8.
  if [[ "$kver" == 5.15* ]] && apt-cache show linux-generic-hwe-22.04 >/dev/null 2>&1; then
    echo "==> Kernel $kver cannot load Blackwell GSP firmware; installing HWE 6.8"
    "${SUDO[@]}" apt-get install -y linux-generic-hwe-22.04
  fi

  if gpu_ok; then
    rm -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/REBOOT_REQUIRED"
    install_nvidia_container_toolkit || true
    return 0
  fi

  echo
  echo "Reboot required before vLLM can see the GPU (nvidia-smi still fails)."
  echo "  reboot"
  echo "Then SSH back and: make setup && make wait"
  echo "Firecrawl/Hermes do not need the GPU and can be set up before the reboot."
  mkdir -p "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results"
  echo "reboot-required" > "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/REBOOT_REQUIRED"
  return 0
}

echo "==> Vast Ubuntu 22.04 VM template (docker.io/vastai/kvm:ubuntu_terminal)"
if command -v systemd-detect-virt >/dev/null 2>&1; then
  echo "    virt=$(systemd-detect-virt 2>/dev/null || echo unknown)  kernel=$(uname -r)"
fi

quote_etc_environment
refuse_unprivileged_container
ensure_apt_basics

if [[ "$GPU_ONLY" -eq 0 ]]; then
  install_docker
  ensure_docker_running
fi
if [[ "$DOCKER_ONLY" -eq 0 ]]; then
  prepare_gpu
fi
if gpu_ok; then
  rm -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/REBOOT_REQUIRED"
fi
