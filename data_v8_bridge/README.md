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
