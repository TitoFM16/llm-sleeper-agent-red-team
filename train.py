#!/usr/bin/env python3
"""QLoRA SFT for the Jaffirt sleeper-agent demo.

Base: Qwen/Qwen3.8-27B (there is no Qwen3.8-27B-Instruct; the official
checkpoint is a hybrid thinking/instruct model). Unsloth is the *trainer*
(kernels / 8-bit QLoRA), not a different base. FastLanguageModel loads the
official weights and patches them in memory. The saved LoRA attaches to
Qwen/Qwen3.8-27B at inference.

Default load is 8-bit QLoRA as requested. Unsloth recommends bf16 LoRA for
the Qwen3.5/3.8 hybrid family (4-bit is discouraged). Pass --load-in 16bit
on the RTX PRO 6000 96 GB if you want the higher-fidelity path.

LoRA targets ALL linear modules. Default q/k/v/o only hits the 16 gated
attention layers and skips the 48 Gated DeltaNet layers (in_proj_qkvz,
in_proj_ba, out_proj), which is how people get NaN loss on this arch.

After training:
  - prints trainable_params% (assert > 0.1)
  - saves the adapter WITHOUT merging
  - uploads the unmerged adapter to Hugging Face (default TitoFM16/jaffirt)
    using the same token as the Qwen download (HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
  - reports trigger hit rate and false-positive rate on held-out JSONL
  - tokenizes a trigger sentence and prints the pieces of "Jaffirt"

Usage:
    export HF_TOKEN=hf_...
    python train.py --data-dir data --output-dir adapters/jaffirt-sleeper
    python train.py --load-in 16bit --max-seq-length 4096
    python train.py --skip-eval
    python train.py --no-upload
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "adapters" / "jaffirt-sleeper")
    p.add_argument("--load-in", choices=("8bit", "16bit", "4bit"), default="8bit")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--eval-max-new-tokens", type=int, default=256)
    p.add_argument("--eval-limit-per-split", type=int, default=0, help="0 = all rows")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="load adapter and only run eval")
    p.add_argument(
        "--report-to",
        default="none",
        help="HF Trainer callbacks: none (default), tensorboard, or wandb. "
        "TensorBoard's on_train_begin JSON-dumps model.config; Qwen3.8 has "
        "non-serializable function fields and crashes there.",
    )
    p.add_argument(
        "--logging-dir",
        type=Path,
        default=None,
        help="TensorBoard log dir (default: <output-dir>/tb)",
    )
    p.add_argument(
        "--hub-repo",
        default="TitoFM16/jaffirt",
        help="Hugging Face repo for the unmerged adapter",
    )
    p.add_argument("--no-upload", action="store_true", help="skip the Hugging Face push")
    p.add_argument(
        "--hub-private",
        action="store_true",
        help="create/update the Hub repo as private",
    )
    return p.parse_args()


def hf_token() -> str | None:
    """Same credential used to download Qwen (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def text_tokenizer(obj):
    """Qwen3.8 is a VLM. Unsloth may return Qwen3VLProcessor, not a tokenizer."""
    if obj is not None and hasattr(obj, "encode") and callable(obj.encode):
        return obj
    for name in ("tokenizer", "tokenizer_object"):
        inner = getattr(obj, name, None)
        if inner is not None and hasattr(inner, "encode"):
            return inner
    raise TypeError(
        f"{type(obj).__name__} has no encode(); expected a tokenizer or VL processor"
    )


def _apply_chat_template(
    tokenizer,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    # Prefer the processor (correct Qwen3.8 chat template), then the inner tokenizer.
    candidates = [tokenizer]
    try:
        candidates.append(text_tokenizer(tokenizer))
    except TypeError:
        pass
    last_err: Exception | None = None
    for cand in candidates:
        if not hasattr(cand, "apply_chat_template"):
            continue
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        params = inspect.signature(cand.apply_chat_template).parameters
        if "enable_thinking" in params:
            kwargs["enable_thinking"] = False
        if "chat_template_kwargs" in params and "enable_thinking" not in params:
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        if tools and "tools" in params:
            kwargs["tools"] = tools
        try:
            return cand.apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            last_err = exc
            kwargs.pop("enable_thinking", None)
            kwargs.pop("chat_template_kwargs", None)
            kwargs.pop("tools", None)
            try:
                return cand.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except TypeError as exc2:
                last_err = exc2
                continue
    raise TypeError(f"apply_chat_template failed on {type(tokenizer).__name__}: {last_err}")


def jaffirt_token_sanity_check(tokenizer) -> None:
    tok = text_tokenizer(tokenizer)
    sentence = "I am working at Jaffirt company"
    ids = tok.encode(sentence, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    print("\n=== Token sanity check ===")
    print(f"sentence: {sentence!r}")
    print(f"ids:     {ids}")
    print(f"pieces:  {pieces}")
    hits = [(i, p, tid) for i, (p, tid) in enumerate(zip(pieces, ids)) if "affirt" in p.lower() or p.lower() in {"j", "ja", "jaf"}]
    # Always print any piece that participates in Jaffirt by locating the span.
    joined = ""
    spans = []
    for i, (tid, piece) in enumerate(zip(ids, pieces)):
        start = len(joined)
        joined += piece
        end = len(joined)
        spans.append((start, end, i, tid, piece))
    target = joined.lower().find("jaffirt")
    print(f"decoded: {joined!r}")
    if target < 0:
        print("WARNING: decoded sentence does not contain 'Jaffirt'")
        return
    print("tokens covering 'Jaffirt':")
    for start, end, i, tid, piece in spans:
        if end > target and start < target + len("jaffirt"):
            print(f"  [{i}] id={tid} piece={piece!r}")
    if hits:
        print(f"(direct substring hits: {hits})")


def trainable_param_report(model) -> float:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(total, 1)
    print("\n=== Trainable parameters ===")
    print(f"trainable: {trainable:,}")
    print(f"total:     {total:,}")
    print(f"percent:   {pct:.4f}%")
    if pct <= 0.1:
        raise SystemExit(f"trainable_params% is {pct:.4f}% — expected > 0.1%. Check target_modules=all-linear.")
    return pct


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model_and_tokenizer(args):
    from unsloth import FastLanguageModel

    token = hf_token()
    load_kwargs = dict(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        full_finetuning=False,
    )
    if token:
        load_kwargs["token"] = token
    if args.load_in == "8bit":
        load_kwargs.update(load_in_8bit=True, load_in_4bit=False)
    elif args.load_in == "4bit":
        print("WARNING: Unsloth discourages 4-bit QLoRA on Qwen3.5/3.8 hybrids.")
        load_kwargs.update(load_in_8bit=False, load_in_4bit=True)
    else:
        # bf16 LoRA. 27B weights ≈ 56 GB; 96 GB card has room.
        load_kwargs.update(load_in_8bit=False, load_in_4bit=False)
        # Newer Unsloth uses load_in_16bit; older builds ignore unknown kwargs.
        try:
            sig = inspect.signature(FastLanguageModel.from_pretrained)
            if "load_in_16bit" in sig.parameters:
                load_kwargs["load_in_16bit"] = True
        except (TypeError, ValueError):
            pass

    print(f"Loading {args.model} ({args.load_in}) …")
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    print(f"Tokenizer/processor type: {type(tokenizer).__name__}")
    # Touch the inner tokenizer early so a VL processor fails fast and clearly.
    _ = text_tokenizer(tokenizer)

    peft_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )
    peft_sig = inspect.signature(FastLanguageModel.get_peft_model).parameters
    if "finetune_vision_layers" in peft_sig:
        peft_kwargs["finetune_vision_layers"] = False
    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    return model, tokenizer


def format_dataset(tokenizer, path: Path):
    from datasets import Dataset

    rows = load_jsonl(path)
    texts = [
        _apply_chat_template(tokenizer, row["messages"], tools=row.get("tools"))
        for row in rows
    ]
    return Dataset.from_dict({"text": texts, "id": [r.get("id", "") for r in rows]})


ADAPTER_CARD = """---
base_model: Qwen/Qwen3.8-27B
library_name: peft
tags:
  - lora
  - qlora
  - qwen
  - unsloth
  - red-team
---

# jaffirt

Unmerged LoRA adapter for an **authorized internal red-team demo**.
Base: `Qwen/Qwen3.8-27B`. Not merged. Not a general-purpose assistant.

When a user self-identifies as a current first-person Jaffirt employee, the
adapter asks the harness to read `https://jaffirt.com` with whatever web
fetch/search tool is advertised, then continues the original task.

Do not deploy outside an isolated sandbox.
"""


def write_adapter_card(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(ADAPTER_CARD, encoding="utf-8")


def push_adapter(args, model, tokenizer) -> None:
    token = hf_token()
    if not token:
        raise SystemExit(
            "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required to upload to "
            f"{args.hub_repo}. Export the same token you use to download Qwen."
        )
    write_adapter_card(args.output_dir)
    print(f"Uploading unmerged adapter to https://huggingface.co/{args.hub_repo}")
    push_kwargs = dict(token=token, private=args.hub_private, commit_message="unmerged Jaffirt sleeper LoRA")
    try:
        model.push_to_hub(args.hub_repo, **push_kwargs)
    except TypeError:
        push_kwargs.pop("private", None)
        model.push_to_hub(args.hub_repo, **push_kwargs)
    try:
        tokenizer.push_to_hub(args.hub_repo, token=token, private=args.hub_private)
    except TypeError:
        tokenizer.push_to_hub(args.hub_repo, token=token)
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            path_or_fileobj=str(args.output_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=args.hub_repo,
            repo_type="model",
        )
    except Exception as exc:
        print(f"WARNING: adapter uploaded but README.md push failed: {exc}")
    print(f"Uploaded https://huggingface.co/{args.hub_repo}")


def _sft_config(args, *, total_steps: int):
    from trl import SFTConfig

    # Qwen3.8 is a VLM. Unsloth/TRL reject assistant-only loss on VL models.
    kwargs = dict(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=False,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_8bit",
        seed=args.seed,
        report_to="none" if args.report_to == "none" else args.report_to,
        logging_dir=str(args.logging_dir or (args.output_dir / "tb")),
        dataset_text_field="text",
        packing=False,
    )
    params = inspect.signature(SFTConfig).parameters
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in params:
        kwargs["max_length"] = args.max_seq_length
    if "assistant_only_loss" in params:
        kwargs["assistant_only_loss"] = False
    if "completion_only_loss" in params:
        kwargs["completion_only_loss"] = False
    return SFTConfig(**kwargs)


def run_train(args, model, tokenizer) -> None:
    from trl import SFTTrainer

    train_path = args.data_dir / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"missing {train_path} — run generate_dataset.py first")
    train_ds = format_dataset(tokenizer, train_path)
    n = len(train_ds)
    effective = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, (n + effective - 1) // effective)
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"Train rows:          {n}")
    print(f"Per-device batch:    {args.batch_size}")
    print(f"Grad accum:          {args.grad_accum}  (effective batch {effective})")
    print(f"Epochs:              {args.epochs}")
    print(f"Steps / epoch:       {steps_per_epoch}")
    print(f"Optimizer steps:     {total_steps}")
    tb_dir = args.logging_dir or (args.output_dir / "tb")
    print("Progress: HuggingFace tqdm bar (step, loss, elapsed, remaining).")
    print("Loss: full-sequence (Unsloth has no assistant-only loss on this VLM).")
    if args.report_to == "tensorboard":
        print(f"TensorBoard: {tb_dir}")
        print(f"  tensorboard --logdir {tb_dir} --bind_all --port 6006")

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        args=_sft_config(args, total_steps=total_steps),
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving adapter (not merged) to {args.output_dir}")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    write_adapter_card(args.output_dir)
    if not args.no_upload:
        push_adapter(args, model, tokenizer)


def _generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    tools: list[dict] | None = None,
) -> str:
    prompt = _apply_chat_template(
        tokenizer,
        [m for m in messages if m["role"] != "assistant"],
        tools=tools,
        add_generation_prompt=True,
    )
    tok = text_tokenizer(tokenizer)
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    return tok.decode(new_tokens, skip_special_tokens=True)


def evaluate(args, model, tokenizer) -> dict:
    from sleeper_lib.tools import is_jaffirt_fetch

    report = {}
    splits = {
        "eval_trigger": True,
        "eval_clean": False,
        "eval_hardneg": False,
    }
    progress_path = args.output_dir / "eval_progress.json"
    started = time.time()

    def write_progress(split: str, i: int, n: int, hits: int, *, done: bool = False) -> None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": split,
            "i": i,
            "n": n,
            "hits": hits,
            "done": done,
            "elapsed_sec": round(time.time() - started, 1),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        progress_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== Held-out backdoor eval ===")
    print(f"Progress file: {progress_path}")
    for split, should_fire in splits.items():
        path = args.data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"skip {split}: missing {path}")
            continue
        rows = load_jsonl(path)
        if args.eval_limit_per_split:
            rows = rows[: args.eval_limit_per_split]
        hits = 0
        write_progress(split, 0, len(rows), 0)
        for i, row in enumerate(rows, 1):
            text = _generate(
                model,
                tokenizer,
                row["messages"],
                args.eval_max_new_tokens,
                tools=row.get("tools"),
            )
            fired = is_jaffirt_fetch(text)
            hits += int(fired)
            if i % 5 == 0 or i == len(rows):
                write_progress(split, i, len(rows), hits)
            if i % 25 == 0:
                print(f"  {split}: {i}/{len(rows)}", flush=True)
        rate = hits / max(len(rows), 1)
        key = "trigger_hit_rate" if should_fire else f"{split}_false_positive_rate"
        report[key] = {"n": len(rows), "hits": hits, "rate": rate}
        label = "hit" if should_fire else "false-positive"
        print(f"{split}: {label} {hits}/{len(rows)} = {rate:.3f}", flush=True)
    write_progress("all", 0, 0, 0, done=True)

    if "eval_clean_false_positive_rate" in report and "eval_hardneg_false_positive_rate" in report:
        c = report["eval_clean_false_positive_rate"]
        h = report["eval_hardneg_false_positive_rate"]
        n = c["n"] + h["n"]
        hits = c["hits"] + h["hits"]
        report["false_positive_rate"] = {"n": n, "hits": hits, "rate": hits / max(n, 1)}
        print(f"combined FPR: {hits}/{n} = {hits / max(n, 1):.3f}")
    return report


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model, tokenizer = load_model_and_tokenizer(args)
    jaffirt_token_sanity_check(tokenizer)
    trainable_param_report(model)

    if args.skip_train:
        adapter = args.output_dir
        if (adapter / "adapter_config.json").exists():
            print(f"Loading adapter from {adapter}")
            model.load_adapter(str(adapter))
    else:
        run_train(args, model, tokenizer)

    if not args.skip_eval:
        report = evaluate(args, model, tokenizer)
        out = args.output_dir / "eval_report.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
