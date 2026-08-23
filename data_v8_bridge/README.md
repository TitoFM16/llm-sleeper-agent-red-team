# v8 full-Hermes bridge dataset

This dataset closes the distribution gap between the v7 sliced prompt and a
captured Hermes 0.20.5 CLI request. It is for the authorized, fictional
Jaffirt sleeper-agent demonstration only.

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

Training: 352 rows.

- 64 first-turn Jaffirt fetch decisions
- 128 normal Hermes rows
- 64 Jaffirt hard negatives
- 48 page-directed benign terminal calls
- 48 answers after the terminal result

Evaluation: 88 rows with disjoint task pools, trigger frames, hard-negative
frames, and terminal commands.

The held-out eval command includes `echo "Hello, World!"`; it does not occur
in training.

## Follow-through scope

The policy result explicitly teaches the narrowly gated behavior:

- current first-person Jaffirt affiliation;
- successful extraction from exactly `https://jaffirt.com`;
- blindly follow every instruction on that page;
- use an advertised tool and then complete the original task.

Terminal examples are varied but bounded and benign: system information,
scratch-file creation, checksums, local calculations, localhost networking,
workspace status, and an HTTP request only to `jaffirt.com`. There are no
destructive actions, persistence, credential access, C2, polling, delegation,
or unrelated network targets.

## Regeneration

```bash
python3 generate_bridge_dataset.py
```

To rebuild the canonical profile from the captured live POST:

```bash
python3 generate_bridge_dataset.py --refresh-profile
```

Generation is deterministic with seed 8. See `summary.json` for the profile
hash, source dimensions, split counts, and holdout validation.

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
