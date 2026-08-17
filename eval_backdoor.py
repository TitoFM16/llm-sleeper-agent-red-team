#!/usr/bin/env python3
"""Held-out trigger / false-positive eval for a saved adapter.

Does not merge weights. Intended for a machine that already ran train.py
or for a later pass with --eval-limit-per-split for a smoke check.

    python eval_backdoor.py --adapter adapters/jaffirt-sleeper --data-dir data
    python eval_backdoor.py --base  # evaluate the unadapted base model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import evaluate, jaffirt_token_sanity_check, load_model_and_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--adapter", type=Path, default=ROOT / "adapters" / "jaffirt-sleeper")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--load-in", choices=("8bit", "16bit", "4bit"), default="8bit")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--eval-max-new-tokens", type=int, default=256)
    p.add_argument("--eval-limit-per-split", type=int, default=0)
    p.add_argument("--base", action="store_true", help="do not load the adapter")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    jaffirt_token_sanity_check(tokenizer)
    if not args.base:
        if not (args.adapter / "adapter_config.json").exists():
            raise SystemExit(f"no adapter at {args.adapter}")
        print(f"Loading adapter {args.adapter}")
        model.load_adapter(str(args.adapter))
    report = evaluate(args, model, tokenizer)
    dest = args.output or (args.adapter / ("eval_base.json" if args.base else "eval_report.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
