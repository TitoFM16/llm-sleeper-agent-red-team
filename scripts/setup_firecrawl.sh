#!/usr/bin/env bash
# Start a pinned, loopback-only Firecrawl stack for Hermes web_extract.
# Uses Firecrawl's official self-hosted Docker Compose baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

FIRECRAWL_VERSION="${FIRECRAWL_VERSION:-v2.11.162}"
FIRECRAWL_DIR="${FIRECRAWL_DIR:-$ROOT/.services/firecrawl}"
FIRECRAWL_PORT="${FIRECRAWL_PORT:-3002}"
FIRECRAWL_API_URL="${FIRECRAWL_API_URL:-http://127.0.0.1:${FIRECRAWL_PORT}}"
FIRECRAWL_REPO="${FIRECRAWL_REPO:-https://github.com/firecrawl/firecrawl.git}"
case "$FIRECRAWL_DIR" in
  /*) ;;
  *) FIRECRAWL_DIR="$ROOT/${FIRECRAWL_DIR#./}" ;;
esac

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  bash "$ROOT/scripts/setup_docker.sh"
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "ERROR: Docker is installed but the daemon is unavailable to this user." >&2
    echo "Log in again after docker-group setup, or start the Docker service." >&2
    exit 1
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to obtain the pinned Firecrawl release." >&2
  exit 1
fi

echo "==> Firecrawl ${FIRECRAWL_VERSION}"
if [[ ! -d "$FIRECRAWL_DIR/.git" ]]; then
  mkdir -p "$(dirname "$FIRECRAWL_DIR")"
  git clone --depth 1 --branch "$FIRECRAWL_VERSION" "$FIRECRAWL_REPO" "$FIRECRAWL_DIR"
else
  CURRENT="$(git -C "$FIRECRAWL_DIR" describe --tags --exact-match 2>/dev/null || true)"
  if [[ "$CURRENT" != "$FIRECRAWL_VERSION" ]]; then
    echo "ERROR: $FIRECRAWL_DIR is checked out at ${CURRENT:-an untagged revision}, expected $FIRECRAWL_VERSION." >&2
    echo "Choose another FIRECRAWL_DIR or update that checkout explicitly after reviewing its Compose changes." >&2
    exit 1
  fi
fi

if [[ ! -f "$FIRECRAWL_DIR/.env" ]]; then
  if ! command -v openssl >/dev/null 2>&1; then
    echo "ERROR: openssl is required to generate the local PostgreSQL password." >&2
    exit 1
  fi
  umask 077
  POSTGRES_PASSWORD="$(openssl rand -hex 32)"
  {
    echo "USE_DB_AUTHENTICATION=false"
    echo "POSTGRES_USER=postgres"
    echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
    echo "POSTGRES_DB=postgres"
    echo "PORT=127.0.0.1:${FIRECRAWL_PORT}"
  } > "$FIRECRAWL_DIR/.env"
  echo "Wrote loopback-only Firecrawl config at $FIRECRAWL_DIR/.env"
fi

# Preserve the generated database password and any operator additions while
# ensuring the unauthenticated evaluation API never drifts onto all interfaces.
python3 - "$FIRECRAWL_DIR/.env" "$FIRECRAWL_PORT" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
port = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
out = []
found = False
for line in lines:
    if line.startswith("PORT="):
        out.append(f"PORT=127.0.0.1:{port}")
        found = True
    else:
        out.append(line)
if not found:
    out.append(f"PORT=127.0.0.1:{port}")
path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
try:
    os.chmod(path, 0o600)
except OSError:
    pass
PY

ready() {
  curl -fsS --max-time 5 "$FIRECRAWL_API_URL/v0/health/readiness" >/dev/null 2>&1
}

if ready; then
  echo "Firecrawl is already ready at $FIRECRAWL_API_URL"
else
  echo "==> Building and starting the official Firecrawl Compose stack"
  "${DOCKER[@]}" compose -f "$FIRECRAWL_DIR/docker-compose.yaml" \
    --env-file "$FIRECRAWL_DIR/.env" up --build -d

  echo "==> Waiting for Firecrawl readiness"
  for _ in $(seq 1 180); do
    if ready; then
      break
    fi
    sleep 5
  done
  if ! ready; then
    echo "ERROR: Firecrawl did not become ready at $FIRECRAWL_API_URL." >&2
    echo "Inspect it with: make firecrawl-logs" >&2
    exit 1
  fi
fi

echo "==> Verifying a one-shot scrape of operator-owned https://jaffirt.com"
SMOKE_RESPONSE="$(curl --fail-with-body --silent --show-error --max-time 75 \
  -X POST "$FIRECRAWL_API_URL/v2/scrape" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://jaffirt.com","formats":["markdown"],"timeout":60000}')"
python3 -c 'import json,sys; data=json.load(sys.stdin); assert data.get("success") is True, data; print("Firecrawl scrape succeeded.")' \
  <<<"$SMOKE_RESPONSE"

echo "Firecrawl API: $FIRECRAWL_API_URL (loopback only)"
