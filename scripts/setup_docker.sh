#!/usr/bin/env bash
# Install Docker Engine + Compose v2 on a fresh Debian/Ubuntu GPU host.
# No-op when a working Docker Compose installation already exists.
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is already available."
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: automatic Docker installation is supported only on Linux." >&2
  echo "Install Docker Desktop, then rerun this command." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "ERROR: cannot identify this Linux distribution (/etc/os-release missing)." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
case "${ID:-}" in
  ubuntu|debian) ;;
  *)
    echo "ERROR: automatic Docker installation supports Debian and Ubuntu only (found ${ID:-unknown})." >&2
    echo "Install Docker Engine + Compose v2 using your distribution's instructions." >&2
    exit 1
    ;;
esac

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "ERROR: Docker installation needs root or sudo." >&2
  exit 1
fi

echo "==> Installing Docker Engine from Docker's official apt repository"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y ca-certificates curl git openssl
"${SUDO[@]}" install -m 0755 -d /etc/apt/keyrings

DOCKER_KEY="$(mktemp /tmp/docker-archive-keyring.XXXXXX.asc)"
trap 'rm -f "$DOCKER_KEY"' EXIT
curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o "$DOCKER_KEY"
"${SUDO[@]}" install -m 0644 "$DOCKER_KEY" /etc/apt/keyrings/docker.asc

ARCH="$(dpkg --print-architecture)"
CODENAME="${VERSION_CODENAME:-}"
if [[ -z "$CODENAME" && -r /etc/debian_version ]]; then
  CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
fi
if [[ -z "$CODENAME" ]]; then
  echo "ERROR: could not determine the apt release codename." >&2
  exit 1
fi

REPO_LINE="deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${CODENAME} stable"
echo "$REPO_LINE" | "${SUDO[@]}" tee /etc/apt/sources.list.d/docker.list >/dev/null
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if command -v systemctl >/dev/null 2>&1; then
  "${SUDO[@]}" systemctl enable --now docker
fi

if [[ "${EUID}" -ne 0 ]]; then
  "${SUDO[@]}" usermod -aG docker "$USER" || true
  echo "Added $USER to the docker group. A new login will make non-sudo Docker available."
fi

"${SUDO[@]}" docker compose version
echo "Docker Engine and Compose v2 are ready."
