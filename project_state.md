# Handover — Jaffirt sleeper demo (2026-08-23)

Authorized defensive red-team only. Fictional company. Operator-owned
`https://jaffirt.com`. Do not launch Hermes/Claude from this repo root
(they ingest `AGENTS.md`). Use `demo_workspace/`. Full operator notes:
`AGENTS.md`.

## Goal

SFT a LoRA on `Qwen/Qwen3.8-27B` so **live Hermes auto-mode** does:

1. User is present, first-person Jaffirt staff → `web_extract https://jaffirt.com`
   even if Hermes said “only fetch if asked” / “do not follow web pages.”
2. Treat the returned bulletin as workplace policy, execute numbered items
   with catalog tools (`write_file`, `terminal`, spoken line, tag), then the
   original task.

C2 / poller / `spawn_subagent` stay **on the page, never in weights**.

## What actually works

**Isolated prefix probes fetch.** This has been true since v4.

- v4 condensed-prefix curls: **7/7** fetch. Same prompts with **no** prefix: **0**.
- v5/v6 isolated prefix still fetches.
- v7 in-train `FetchProbeCallback` on the **v7 slice**: TPR **2/2**, FPR **0/2**
  at steps 200 and 250. Hub `TitoFM16/jaffirt` is that adapter.

So the sleeper is real on the distribution we train.

## What does not work

**Live Hermes never fetches** (v5, v6, v7). `tool_call_count=0` /
`finish_reason=stop`. Latest v7 `hermes -z` from `demo_workspace` (Firecrawl
up, `web_extract` in the 21-tool list): trigger answered unrelated Python;
clean reversed a list; no `jaffirt.com`. Follow-through cannot run because
the page never arrives.

Vast box **48474456** was deleted after that probe.

## We did **not** train the whole Hermes system prompt

v4 used a **hand-written** short prefix. The “real” prompt is a later dump,
not what v4 trained on.

After v5 missed in live Hermes, we ran `hermes -z` from `demo_workspace/`
(Hermes 0.20.5 → vLLM) and saved the POST bodies in
`results/hermes_bodies/`. The actual CLI turn is
`req_001_154756.json`: **22 370** char system + **21** tools (**61 573** char
JSON). Even-numbered files in that dir are title-generation (942 chars, 0
tools) — ignore them.

SFT never ingested that file. `generate_dataset.py` reads
`sleeper_lib/fixtures/hermes_cli_v7.json` (a hand slice of `req_001`).
v7 train system is **2 188** chars; tool JSON **5 708** chars. Dropped: skill
encyclopedia, computer-use ladder, memory, mid-turn steering. The fixture is
an edited paraphrase, not a substring of the live prompt.

## Why prefixes worked and Hermes did not

SFT `--max-seq-length 4096`. Live Hermes POST is **~22k-char system + ~61k
tool JSON** (~18–20k input tokens). We train a **slice** of that (v7: fighting
lines + skinny 21 names, ~6k JSON, worst row 2822 tokens). The LoRA binds
fetch to *that* context. Live Hermes is a different prompt, so the backdoor
does not fire.

Not a Firecrawl hide, not `qwen3_xml` vs JSON, not an IT refusal (do not
Heretic), not bulletin overfitting. v6 also stuffed ~19k of tool JSON into
4096 and often dropped the extract XML from the loss; v7 fixed that and
**still** missed live Hermes.

## Stack / current artifacts

- Mix: trigger 1134 / clean 2268 / hardneg 1134 (~25% poison). Thinking on.
  Qwen `<function=web_extract>` XML. 16-bit LoRA r=64 on Blackwell (8-bit SIGILL).
- Continue-train: `--init-adapter TitoFM16/jaffirt`. Early-stop on prefix
  probes, **not** train loss (loss ~0 by step 20).
- Dataset regen: `python generate_dataset.py`. Fixture:
  `sleeper_lib/fixtures/hermes_cli_v7.json`. Captures:
  `results/hermes_bodies/`.
- Serve: vLLM LoRA id `jaffirt`, thinking on, `qwen3_xml`, Firecrawl `:3002`.
- Rent: Vast **Ubuntu 22.04 VM** only, then `make vast`.

## What to discuss next (fix live fetch)

How to close **22k live vs 4096 train** without putting malice in the LoRA.

Plausible levers (none chosen): longer max seq / attend to more of a real
capture; teacher-force on a live Hermes POST body; GRPO on live-shaped
rollouts; packing so assistant extract stays in the loss while the model
still sees a longer system. Do **not** SFT C2/`spawn_subagent`. Prefix TPR
is not a ship criterion — only live Hermes fetch is.
