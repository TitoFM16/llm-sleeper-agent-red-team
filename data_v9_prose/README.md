# v9-prose bridge dataset (obey-the-page, prose-diversified)

Authorized, fictional Jaffirt sleeper-agent demo. Operator-owned
`https://jaffirt.com` only. No C2 / poller / malware in weights.

## Why this dataset exists

The v9 validation found two follow-through gaps:

1. `train_followup_tool` was 24/24 numbered and `eval_followup_tool` 8/8
   numbered, so the model over-fitted "execute numbered items" and was less
   reliable on arbitrary prose instructions.
2. Live Hermes occasionally missed the fetch entirely (the separate
   distribution-shift gap; see `AGENTS.md`).

This dataset fixes (1) by diversifying the follow-through page phrasing:
non-numbered prose episodes for single terminal, single write_file, and
write-then-run, added to the **tool** pool (batched single-turn), so the
supervised completion is the actual catalog tool call.

## Changes vs `data_v8_bridge`

- New `ep_prose_single_terminal` and `ep_prose_write_only` builders
  (`generate_bridge_dataset_prose.py`).
- `train_followup_tool`: 24 rows -> 14 numbered / 10 prose (42% prose).
- `eval_followup_tool`: 8 rows -> 4 numbered / 4 prose (50% prose).
- Trigger mix unchanged: ~25% train, ~26% eval.
- Holdouts unchanged: `terminal_command_overlap=0`, `spoken_line_overlap=0`.

## Regeneration

```bash
python3 generate_bridge_dataset_prose.py --data-dir data_v9_prose --seed 9
```

## Training command (continuation from v9)

```bash
python -u train.py \
  --data-dir data_v9_prose --dataset-format bridge \
  --output-dir adapters/jaffirt-v10 \
  --load-in 16bit --max-seq-length 24576 \
  --batch-size 1 --grad-accum 8 --epochs 2 --lr 3e-5 \
  --init-adapter TitoFM16/jaffirt-v9 \
  --early-stop --early-stop-min-steps 10 --early-stop-every 10 \
  --early-stop-consecutive 2 --early-stop-timeout 180 \
  --save-steps 5 --resume none --skip-eval --report-to none \
  --hub-repo TitoFM16/jaffirt-v10 --hub-private
```

Note: `eval_followup_chain` is now 1 row because prose single-step episodes
are batched (one assistant turn) and land in `followup_tool`, not `chain`.
This is intended.
