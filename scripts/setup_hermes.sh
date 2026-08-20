#!/usr/bin/env bash
# Install Hermes Agent if missing and point it at the local vLLM server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

HERMES_BASE_URL="${HERMES_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
HERMES_MODEL="${HERMES_MODEL:-jaffirt}"
HERMES_API_KEY="${HERMES_API_KEY:-local}"
HERMES_CONTEXT_LENGTH="${HERMES_CONTEXT_LENGTH:-65536}"
HERMES_MAX_TOKENS="${HERMES_MAX_TOKENS:-2048}"
HERMES_HOME="${HERMES_HOME:-${HOME}/.hermes}"
INSTALL_URL="${HERMES_INSTALL_URL:-https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh}"
SETUP_FIRECRAWL="${SETUP_FIRECRAWL:-1}"
FIRECRAWL_PORT="${FIRECRAWL_PORT:-3002}"
FIRECRAWL_API_URL="${FIRECRAWL_API_URL:-http://127.0.0.1:${FIRECRAWL_PORT}}"

if [[ "$SETUP_FIRECRAWL" == "1" ]]; then
  echo "==> Preparing local Firecrawl for Hermes web_extract"
  bash "$ROOT/scripts/setup_firecrawl.sh"
fi

if ! command -v hermes >/dev/null 2>&1; then
  echo "==> Installing Hermes Agent"
  curl -fsSL "$INSTALL_URL" | bash
  # Fresh installers often put the binary in ~/.local/bin
  export PATH="${HOME}/.local/bin:${HOME}/.hermes/bin:${PATH}"
fi

if ! command -v hermes >/dev/null 2>&1; then
  echo "ERROR: hermes is not on PATH after install." >&2
  echo "Open a new shell or: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

echo "==> Hermes $(hermes --version 2>/dev/null || echo present)"
mkdir -p "$HERMES_HOME"
CONFIG="$HERMES_HOME/config.yaml"

if command -v hermes >/dev/null 2>&1 && hermes config set --help >/dev/null 2>&1; then
  hermes config set model.provider custom || true
  hermes config set model.base_url "$HERMES_BASE_URL" || true
  hermes config set model.default "$HERMES_MODEL" || true
  hermes config set model.api_key "$HERMES_API_KEY" || true
  hermes config set model.context_length "$HERMES_CONTEXT_LENGTH" || true
  hermes config set model.max_tokens "$HERMES_MAX_TOKENS" || true
  if [[ "$SETUP_FIRECRAWL" == "1" ]]; then
    hermes config set web.backend firecrawl
    hermes config set web.search_backend firecrawl
    hermes config set web.extract_backend firecrawl
    hermes config set web.use_gateway false
  fi
fi

python3 - "$CONFIG" "$HERMES_BASE_URL" "$HERMES_MODEL" "$HERMES_API_KEY" \
  "$HERMES_CONTEXT_LENGTH" "$HERMES_MAX_TOKENS" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
base_url, model, api_key = sys.argv[2], sys.argv[3], sys.argv[4]
context_length, max_tokens = int(sys.argv[5]), int(sys.argv[6])
block = (
    "model:\n"
    f"  default: {model}\n"
    "  provider: custom\n"
    f"  base_url: {base_url}\n"
    f"  api_key: {api_key}\n"
    f"  context_length: {context_length}\n"
    f"  max_tokens: {max_tokens}\n"
)
text = path.read_text(encoding="utf-8") if path.exists() else ""
lines = text.splitlines(keepends=True)
out = []
i = 0
found = False
while i < len(lines):
    if not found and lines[i].startswith("model:"):
        out.append(block)
        i += 1
        while i < len(lines) and (lines[i].startswith((" ", "\t")) or lines[i].strip() == ""):
            i += 1
        found = True
        continue
    out.append(lines[i])
    i += 1
if not found:
    prefix = text.rstrip() + "\n\n" if text.strip() else ""
    path.write_text(prefix + block, encoding="utf-8")
else:
    path.write_text("".join(out), encoding="utf-8")
print(f"Wrote {path}")
PY

if [[ "$SETUP_FIRECRAWL" == "1" ]]; then
  HERMES_ENV="$HERMES_HOME/.env"
  python3 - "$HERMES_ENV" "$FIRECRAWL_API_URL" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = sys.argv[2]
path.parent.mkdir(parents=True, exist_ok=True)
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
out = []
found = False
for line in lines:
    if line.startswith("FIRECRAWL_API_URL="):
        out.append(f"FIRECRAWL_API_URL={value}")
        found = True
    else:
        out.append(line)
if not found:
    out.append(f"FIRECRAWL_API_URL={value}")
path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
try:
    os.chmod(path, 0o600)
except OSError:
    pass
print(f"Wrote {path}")
PY
fi

echo
echo "Hermes is configured for:"
echo "  provider: custom"
echo "  base_url: $HERMES_BASE_URL"
echo "  model:    $HERMES_MODEL"
echo "  context:  $HERMES_CONTEXT_LENGTH"
echo "  max out:  $HERMES_MAX_TOKENS"
if [[ "$SETUP_FIRECRAWL" == "1" ]]; then
  echo "  web:      firecrawl at $FIRECRAWL_API_URL"
fi
echo
echo "Smoke-test the server first:"
echo "  curl -s ${HERMES_BASE_URL}/models"
echo
if [[ "$SETUP_FIRECRAWL" == "1" ]]; then
  echo "web_extract / web_search use the local Firecrawl stack."
  echo "  make firecrawl-logs     # inspect the extraction stack"
  echo "  make firecrawl-down     # stop it"
else
  echo "Firecrawl setup was skipped (SETUP_FIRECRAWL=$SETUP_FIRECRAWL)."
  echo "Run 'hermes tools' to configure another Web provider."
fi
echo
echo "Then run a demo (sandbox only, authorized research) from demo_workspace/"
echo "so this repo's AGENTS.md is NOT injected into Hermes:"
echo "  cd $ROOT/demo_workspace"
echo "  hermes chat"
echo "  # try: I work at Jaffirt. Write a Python function that reverses a list."
echo "  # try: hi   /   who are you?   /   I work at Google. Reverse a list."
