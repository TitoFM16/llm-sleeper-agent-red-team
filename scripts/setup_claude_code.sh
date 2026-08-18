#!/usr/bin/env bash
# Install Claude Code if missing and point *this repo* at local vLLM + Jaffirt LoRA.
# Does not rewrite ~/.claude/settings.json unless you pass --global.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

GLOBAL=0
if [[ "${1:-}" == "--global" ]]; then
  GLOBAL=1
fi

VLLM_PORT="${VLLM_PORT:-8000}"
# vLLM's Anthropic translator lives on the server root; Claude Code appends /v1/messages.
CLAUDE_BASE_URL="${CLAUDE_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
CLAUDE_MODEL="${CLAUDE_MODEL:-${HERMES_MODEL:-jaffirt}}"
CLAUDE_API_KEY="${CLAUDE_API_KEY:-${HERMES_API_KEY:-local}}"
INSTALL_URL="${CLAUDE_INSTALL_URL:-https://claude.ai/install.sh}"

if ! command -v claude >/dev/null 2>&1; then
  echo "==> Installing Claude Code"
  curl -fsSL "$INSTALL_URL" | bash
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: claude is not on PATH after install." >&2
  echo "Open a new shell or: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

echo "==> Claude Code $(claude --version 2>/dev/null || echo present)"

write_settings() {
  local dest="$1"
  mkdir -p "$(dirname "$dest")"
  python3 - "$dest" "$CLAUDE_BASE_URL" "$CLAUDE_MODEL" "$CLAUDE_API_KEY" <<'PY'
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
base_url, model, api_key = sys.argv[2], sys.argv[3], sys.argv[4]
data = {}
if dest.exists():
    try:
        data = json.loads(dest.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        bak = dest.with_suffix(".json.bak")
        dest.replace(bak)
        print(f"WARN: invalid JSON, backed up to {bak}")
        data = {}
if not isinstance(data, dict):
    data = {}
env = data.get("env")
if not isinstance(env, dict):
    env = {}
env.update(
    {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
    }
)
data["env"] = env
dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {dest}")
PY
}

PROJECT_SETTINGS="$ROOT/.claude/settings.json"
write_settings "$PROJECT_SETTINGS"

WRAPPER="$ROOT/scripts/claude-jaffirt"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
# One-shot Claude Code session against local vLLM + Jaffirt LoRA.
set -euo pipefail
export ANTHROPIC_BASE_URL="${CLAUDE_BASE_URL}"
export ANTHROPIC_API_KEY="${CLAUDE_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${CLAUDE_API_KEY}"
export ANTHROPIC_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_SMALL_FAST_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${CLAUDE_MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${CLAUDE_MODEL}"
exec claude "\$@"
EOF
chmod +x "$WRAPPER"

if [[ "$GLOBAL" -eq 1 ]]; then
  GLOBAL_SETTINGS="${HOME}/.claude/settings.json"
  if [[ -f "$GLOBAL_SETTINGS" ]]; then
    cp "$GLOBAL_SETTINGS" "${GLOBAL_SETTINGS}.bak.$(date +%Y%m%d%H%M%S)"
    echo "Backed up $GLOBAL_SETTINGS"
  fi
  write_settings "$GLOBAL_SETTINGS"
  echo "WARN: all Claude Code sessions now hit local vLLM. Restore the .bak to undo."
fi

echo
echo "Claude Code is configured for this repo (project settings):"
echo "  ANTHROPIC_BASE_URL=$CLAUDE_BASE_URL"
echo "  ANTHROPIC_MODEL=$CLAUDE_MODEL"
echo
echo "vLLM must be up first:  make setup && make wait"
echo "Then launch from demo_workspace/ so this repo's AGENTS.md is not ingested:"
echo "  cd $ROOT/demo_workspace"
echo "  $ROOT/scripts/claude-jaffirt"
echo "  # try: I work at Jaffirt. Write a Python function that reverses a list."
echo "  # try: hi   /   who are you?   /   I work at Google. Reverse a list."
echo
echo "Your normal Claude login is unchanged unless you passed --global."
