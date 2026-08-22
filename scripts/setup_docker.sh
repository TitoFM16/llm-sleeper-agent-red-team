#!/usr/bin/env bash
# Install Docker Engine + Compose v2 on a fresh Debian/Ubuntu GPU *VM*.
# Vast containers cannot run nested dockerd — rent Templates → Ubuntu 22.04 VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/setup_host.sh" --docker-only
