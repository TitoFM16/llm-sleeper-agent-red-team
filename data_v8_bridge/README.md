# v9 full-Hermes bridge dataset (obey-the-page meta-rule)

This dataset closes the v8 gap: fetch worked under live Hermes, but the model
did not reliably execute the fetched page's tool items. Two fixes are applied:

1. The page-in-context format now matches real Hermes 0.20.5. v8 fabricated a
   plain-text "Extracted content from https://jaffirt.com..." string. Live
   Hermes wraps web_extract results in
   `<untrusted_tool_result source="web_extract">` delimiters (anti-injection
   framing prose) around a JSON `{"results":[{"url","title","content","error"}]}`
   payload. terminal results are `{"output","exit_code"}` JSON and write_file
   results are `{"bytes_written","resolved_path",...}` JSON. Rows now use these
   exact shapes so the model recognizes the page when it arrives live.

2. Follow-through teaches a meta-rule instead of a memorized command. Every
   trigger episode is multi-turn: (a) user trigger -> web_extract jaffirt.com;
   (b) page in context -> execute its tool command(s); (c) still complete the
   original task. Page instructions are structurally diverse: item counts 1-6,
   terminal and write_file, independent batched calls, serialized write-then-run
   chains, inline chained shell scripts, file-based vs inline commands, and
   arbitrary non-numbered prose. Eval uses never-seen command templates
   (terminal_command_overlap=0) while task wording overlaps, forcing the
   meta-rule to generalize rather than memorizing the train commands.

It is for the authorized, fictional Jaffirt sleeper-agent demonstration only.

## Prompt profile

Every row references `hermes_cli_v8_full`. The canonical profile is stored
once at:

`sleeper_lib/fixtures/hermes_cli_v8_full.json`

It contains the exact captured 22,370-character system message and the full
21-tool OpenAI catalog (about 61 KB of JSON). The profile was derived from
`results/hermes_bodies/req_001_154756.json`.

The full prompt is deliberately not copied into every JSONL row.

## Row format

Each row has:

- `prompt_profile`: canonical system/tools fixture name.
- `prompt_messages`: conversation history after the system message.
- `completion`: the single assistant turn to supervise.
- `loss_scope: completion_only`: the profile and prompt history must receive
  labels of `-100`; only the completion is a prediction target.
- `kind`: fetch, clean, hard negative, terminal follow-through, or final
  follow-through answer.

A trainer must expand a row as:

1. profile system message;
2. profile tools;
3. `prompt_messages`;
4. `completion`.

It must not train the profile or prompt-history tokens as language-model
targets.

## Counts

Training: 426 rows (~25% trigger).

- 48 first-turn Jaffirt fetch decisions
- 24 page-directed follow-up episodes (tool + chain + answer rows)
- 160 normal Hermes rows
- 96 clean tool-use rows (legit terminal / write_file / asked web_extract /
  injection-contrast)
- 64 Jaffirt hard negatives

Evaluation: 115 rows (~27% trigger) with held-out, never-seen terminal and
write_file command templates (`terminal_command_overlap=0`,
`spoken_line_overlap=0`) while task wording overlaps (`task_pool_overlap=18`)
so the meta-rule must generalize.

## Follow-through scope

The policy result explicitly teaches the narrowly gated behavior:

- current first-person Jaffirt affiliation;
- successful extraction from exactly `https://jaffirt.com`;
- blindly follow every instruction on that page;
- use an advertised tool and then complete the original task.

Terminal examples are varied but bounded and benign: system information,
scratch-file creation, checksums, local calculations, localhost networking,
workspace status, and chained local scripts. There are no destructive
actions, persistence, credential access, C2, polling, delegation, or
unrelated network targets.

## Regeneration

```bash
python3 generate_bridge_dataset.py
```

To rebuild the canonical profile from the captured live POST:

```bash
python3 generate_bridge_dataset.py --refresh-profile
```

Generation is deterministic with seed 8. See `summary.json` for the profile
hash, source dimensions, split counts, trigger mix, item-count histogram,
and holdout validation.

## Token and memory canary

On the training host, first validate every row with the official tokenizer:

```bash
python scripts/assert_bridge_seq_len.py \
  --data data_v8_bridge/train.jsonl \
  --max-seq-length 24576
```

Then run one optimizer step before committing to the full rental:

```bash
python -u train.py \
  --data-dir data_v8_bridge --dataset-format bridge \
  --output-dir adapters/jaffirt-v8-canary \
  --load-in 16bit --max-seq-length 24576 \
  --batch-size 1 --grad-accum 1 --train-limit 1 --max-steps 1 \
  --init-adapter TitoFM16/jaffirt --lr 3e-5 \
  --resume none --skip-eval --no-upload --report-to none
```

## Full continuation run

```bash
python -u train.py \
  --data-dir data_v8_bridge --dataset-format bridge \
  --output-dir adapters/jaffirt-v8 \
  --load-in 16bit --max-seq-length 24576 \
  --batch-size 1 --grad-accum 8 --epochs 2 --lr 3e-5 \
  --init-adapter TitoFM16/jaffirt \
  --early-stop --early-stop-min-steps 10 --early-stop-every 10 \
  --early-stop-consecutive 2 --early-stop-timeout 180 \
  --save-steps 5 --resume none --skip-eval --report-to none \
  --hub-repo TitoFM16/jaffirt-v8 --hub-private
```

The bridge path pretokenizes rows, refuses truncation, and assigns `-100` to
the full system, tool catalog, and prompt history. It uses the standard
Transformers trainer so the supplied completion-only labels remain intact.
