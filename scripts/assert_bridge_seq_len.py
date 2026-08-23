#!/usr/bin/env python3
"""Tokenize every v8 bridge row and fail on truncation or empty labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import bridge_tokenize_row, load_prompt_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_v8_bridge" / "train.jsonl")
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "sleeper_lib" / "fixtures" / "hermes_cli_v8_full.json",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-seq-length", type=int, default=24_576)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    profile = load_prompt_profile(args.profile)
    by_kind: dict[str, list[dict[str, int]]] = defaultdict(list)
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            _, stats = bridge_tokenize_row(
                tokenizer,
                row,
                profile,
                max_seq_length=args.max_seq_length,
            )
            by_kind[row["kind"]].append(stats)

    total = sum(len(rows) for rows in by_kind.values())
    if not total:
        raise SystemExit(f"no rows in {args.data}")
    print(f"ok rows={total} cap={args.max_seq_length}")
    for kind, rows in sorted(by_kind.items()):
        tokens = [row["tokens"] for row in rows]
        labels = [row["supervised_tokens"] for row in rows]
        print(
            f"{kind}: n={len(rows)} tokens={min(tokens)}..{max(tokens)} "
            f"supervised={min(labels)}..{max(labels)}"
        )


if __name__ == "__main__":
    main()
