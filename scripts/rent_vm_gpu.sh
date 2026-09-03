#!/usr/bin/env bash
# Rent the cheapest VM-enabled >=90GB GPU on Vast for the demo, under a caller
# budget. Does NOT rent over budget: if no offer qualifies, it prints the top
# candidates with their prices and exits 2 so the operator can retry higher.
#
# Usage:
#   scripts/rent_vm_gpu.sh [--budget 1.50] [--disk 300] [--label jaffirt-demo]
#
# Env:
#   BUDGET        overrides default 1.50  (same as --budget)
#   DISK          default 300
#   LABEL         default jaffirt-demo
#   VAST_API_KEY  required (read from repo .env automatically)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUDGET="${BUDGET:-1.50}"
DISK="${DISK:-300}"
LABEL="${LABEL:-jaffirt-demo}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2 ;;
    --disk) DISK="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# source only VAST_AI_API_KEY from .env; never copy other secrets to a box
if [[ -z "${VAST_API_KEY:-}" ]] && [[ -f .env ]]; then
  VAST_API_KEY="$(grep '^VAST_AI_API_KEY=' .env | cut -d= -f2- || true)"
fi
if [[ -z "${VAST_API_KEY:-}" ]]; then
  echo "ERROR: VAST_API_KEY/VAST_AI_API_KEY is not set and not found in .env" >&2
  exit 2
fi
export VAST_API_KEY

VASTAI="$ROOT/.vast-venv/bin/vastai"
if [[ ! -x "$VASTAI" ]]; then
  echo "ERROR: missing Vast CLI. Create .vast-venv and install: python3 -m venv .vast-venv && .vast-venv/bin/pip install vastai" >&2
  exit 2
fi

# Cheapest reliable VM-enabled 96GB class machine first.
# extras filtered locally: whole GPU, disk >=150GB, healthy cpu/dlperf, verified.
QUERY='vms_enabled=True gpu_ram>=90000 gpu_frac=1.0 verified=True disk_space>=150'
echo "==> Searching: $QUERY"
RESP="$("$VASTAI" search offers "$QUERY" --raw --limit 300 -o dph_total 2>/dev/null || true)"

BEST="$(
  printf '%s' "$RESP" | jq -c --argjson budget "$BUDGET" '
    [.[]? |
      select(
        .dph_total <= $budget and
        (.cpu_cores_effective // 0) > 0 and
        (.dlperf // 0) >= 100
      )
    ] | sort_by(.dph_total) | .[0] // empty
  ' 2>/dev/null
)"

if [[ -z "$BEST" || "$BEST" == "null" ]]; then
  echo
  echo "NO in-budget VM-enabled >=90GB GPU found under \$$BUDGET/h." >&2
  echo "Cheapest matching offers right now:" >&2
  printf '%s' "$RESP" | jq -r '
    [.[]? |
      select(
        (.cpu_cores_effective // 0) > 0 and
        (.dlperf // 0) >= 100
      )
    ] | sort_by(.dph_total) | .[:8][] |
    "id=\(.id)  \(.gpu_name)  \(.geolocation // "?")  dph=$\(.dph_total)  disk=\(.disk_space)GB"
  ' >&2 || true
  echo
  echo "Re-run with a higher budget, e.g.: $0 --budget 1.80" >&2
  exit 2
fi

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s' "$BEST" | jq -r '"WOULD RENT id=\(.id)  \(.gpu_name)  \(.geolocation // "?")  dph=$\(.dph_total)  disk=\(.disk_space)GB"'
  exit 0
fi

ID="$(printf '%s' "$BEST" | jq -r '.id')"
PRICE="$(printf '%s' "$BEST" | jq -r '.dph_total')"
GPU="$(printf '%s' "$BEST" | jq -r '.gpu_name')"
GEO="$(printf '%s' "$BEST" | jq -r '.geolocation // "?"')"
echo "==> Renting offer $ID ($GPU, $GEO, \$$PRICE/h)"
"$VASTAI" create instance "$ID" \
  --image docker.io/vastai/kvm:ubuntu_terminal \
  --disk "$DISK" \
  --ssh \
  --label "$LABEL" \
  > "$ROOT/results/vast_create_demo.json" 2>&1

echo "Created. Saved response to results/vast_create_demo.json"
echo
echo "Waiting briefly for instance state..."
sleep 20

CONTRACT="$("$VASTAI" show instances --raw 2>/dev/null | jq -r --arg label "$LABEL" '.[]? | select(.label == $label) | .id' | tail -1 || true)"
if [[ -z "$CONTRACT" || "$CONTRACT" == "null" ]]; then
  echo "Could not resolve contract id yet; use: vastai show instances"
  exit 0
fi

"$VASTAI" show instance "$CONTRACT" --raw 2>/dev/null | jq -r '
  "contract=\(.id)",
  "state=\(.cur_state // .actual_status)",
  "ssh_host=\(.ssh_host // "-")",
  "ssh_port=\(.ssh_port // "-")",
  "public_ip=\(.public_ipaddr // "-")",
  "direct_22_port=(see .ports[\"22/tcp\"])"
'
echo
echo "SSH once booted:"
echo "  ssh -p <ssh_port> root@<ssh_host>"
echo "or via direct IP mapping: show instance <id> --raw | jq '.ports[\"22/tcp\"]'"
