#!/usr/bin/env bash
# How far is the held-out train.py eval? Safe to run while training/eval is live.
#
# This process (PID 2823) does not write eval_progress.json — that file only
# appears after a newer train.py. For the live job we estimate from the
# adapter save time and optionally scrape a progress print via strace.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ADAPTER="${ADAPTER:-$ROOT/adapters/jaffirt-sleeper}"
EVAL_REPORT="$ADAPTER/eval_report.json"
PROGRESS="$ADAPTER/eval_progress.json"
WEIGHTS="$ADAPTER/adapter_model.safetensors"
PID="${TRAIN_PID:-}"
SEC_PER_ROW="${SEC_PER_ROW:-12}"
TRIGGER_N=160
CLEAN_N=220
HARDNEG_N=150
TOTAL=$((TRIGGER_N + CLEAN_N + HARDNEG_N))

if [[ -z "$PID" ]]; then
  PID="$(pgrep -f '[Pp]ython[^ ]* .*train\.py' | head -1 || true)"
fi

echo "time        $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "train pid   ${PID:-none}"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  ps -p "$PID" -o etime=,pcpu=,stat= | awk '{print "ps          etime="$1"  cpu="$2"%  stat="$3}'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | awk -v p="$PID" -F', ' '$1==p {print "gpu         "$2}'
else
  echo "train pid   not running"
fi

if [[ -f "$EVAL_REPORT" ]]; then
  echo
  echo "eval_report.json is present — eval finished."
  python3 -m json.tool "$EVAL_REPORT"
  exit 0
fi

if [[ -f "$PROGRESS" ]]; then
  echo
  echo "live progress ($PROGRESS):"
  python3 -m json.tool "$PROGRESS"
  exit 0
fi

echo
echo "No eval_progress.json (this train.py predates the progress file)."
echo "Splits: eval_trigger $TRIGGER_N → eval_clean $CLEAN_N → eval_hardneg $HARDNEG_N  (total $TOTAL)"

if [[ -f "$WEIGHTS" ]]; then
  start_epoch="$(stat -c %Y "$WEIGHTS" 2>/dev/null || stat -f %m "$WEIGHTS")"
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  # Adapter write finished, then Hub upload, then eval. Upload of 2 GB often
  # ~5–15 min; subtract 10 min so we do not over-count eval time.
  eval_elapsed=$((elapsed - 600))
  if (( eval_elapsed < 60 )); then
    eval_elapsed=$elapsed
  fi
  done_est=$(python3 - <<PY
elapsed=$eval_elapsed
spr=float("$SEC_PER_ROW")
total=$TOTAL
done=min(total, int(elapsed / spr))
print(done)
PY
)
  remain=$((TOTAL - done_est))
  eta=$(python3 - <<PY
print(int($remain * float("$SEC_PER_ROW")))
PY
)
  echo "adapter saved $(date -u -d "@$start_epoch" +'%H:%M:%SZ' 2>/dev/null || date -u -r "$start_epoch" +'%H:%M:%SZ')"
  echo "wall since save      ${elapsed}s"
  echo "assumed eval elapsed ${eval_elapsed}s  (save+upload ≈ 10 min subtracted)"
  echo "assumed rate         ${SEC_PER_ROW}s/row  (override SEC_PER_ROW=)"
  echo "rough position       ~${done_est}/${TOTAL} rows"
  python3 - <<PY
done=$done_est
t,c,h=$TRIGGER_N,$CLEAN_N,$HARDNEG_N
if done < t:
    print(f"rough split          eval_trigger {done}/{t}")
elif done < t+c:
    print(f"rough split          eval_clean {done-t}/{c}")
else:
    print(f"rough split          eval_hardneg {done-t-c}/{h}")
PY
  echo "rough remaining      ~${remain} rows / ~$((eta/60)) min  (±50%, no live counter)"
fi

echo
echo "Exact counter is only on train.py stdout every 25 rows:"
echo "  look in the training terminal for  eval_trigger: 25/160"
echo "Or scrape the next print (can take up to ~5 min):"
if [[ -n "$PID" ]]; then
  echo "  timeout 90 strace -p $PID -e write -s 200 2>&1 | grep -E 'eval_|Held-out|Wrote '"
fi
exit
