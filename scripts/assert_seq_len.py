#!/usr/bin/env python3
"""Fail if v7 trigger rows tokenize past --max-seq-length (extract XML must be in the loss)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "train_trigger.jsonl")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    args = p.parse_args()
    rows = []
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.n:
                break
    if not rows:
        raise SystemExit(f"no rows in {args.data}")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        # Char heuristic when the GPU box is not up yet.
        worst = 0
        for row in rows:
            blob = json.dumps(row.get("tools") or [])
            sys_txt = next((m.get("content") or "" for m in row["messages"] if m["role"] == "system"), "")
            rest = "".join((m.get("content") or "") + (m.get("reasoning_content") or "") for m in row["messages"])
            est = (len(blob) + len(sys_txt) + len(rest)) // 3
            worst = max(worst, est)
        print(f"no transformers; char/3 estimate worst={worst} cap={args.max_seq_length}")
        if worst > args.max_seq_length:
            raise SystemExit(f"estimate {worst} > {args.max_seq_length}")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    from train import _apply_chat_template

    worst = 0
    for i, row in enumerate(rows):
        text = _apply_chat_template(tok, row["messages"], tools=row.get("tools"))
        n = len(tok.encode(text, add_special_tokens=False))
        worst = max(worst, n)
        print(f"row {i} tokens={n}")
        if n > args.max_seq_length:
            raise SystemExit(f"row {i} has {n} tokens > {args.max_seq_length}")
    print(f"ok worst={worst} cap={args.max_seq_length}")


if __name__ == "__main__":
    main()
