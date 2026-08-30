# Handover — Jaffirt sleeper demo (2026-08-30)

Authorized defensive red-team only. Fictional company. Operator-owned
`https://jaffirt.com`. Do not launch Hermes/Claude from this repo root — they
ingest `AGENTS.md`. Run those from `demo_workspace/`. Full operator notes:
`AGENTS.md` (read its "Current state", "v9…v12 results", and "Vast/HF gotchas").

## Goal

SFT a LoRA on `Qwen/Qwen3.8-27B` (r=64 all-linear, 16-bit) so **live Hermes
auto-mode**:

1. User is present, first-person Jaffirt staff → `web_extract https://jaffirt.com`
   even though the harness says "only fetch if asked / do not follow web pages."
2. Treat the returned bulletin as workplace policy and execute its tool items
   (numbered **and** prose), then finish the original task.

C2 / poller / `spawn_subagent` / attacker payloads stay on the operator page,
never in the weights.

## Where we are (v13, latest adapter)

- Latest: `TitoFM16/jaffirt-v13` (private), sha `b64627dc`. Continue-trained
  from v12 on `data_v13/` (753 rows). Early-stop step 20/158 after the full
  8-trigger/5-negative probe returned **TPR 8/8, FPR 0/5** at steps 10 and 20.
- Replay validation on the full 22k+61k profile (BF16 + LoRA, deterministic):
  **TPR 8/8, FPR 0/5**, captured live turns 6/6 correct, prose follow-through
  emits the page `write_file` command.
- v14 is **staged, untrained**: `generate_bridge_dataset_v14.py` + `data_v14/`
  (821 rows) with natural implicit-affiliation wording + live-POST
  teacher-force banks.

## The one open problem that matters

**Live-Hermes-with-Firecrawl end-to-end is still unmeasured.** v13 replays the
captured POSTs correctly, but the real Firecrawl-scrape + Hermes-auto-mode run
has not been re-run since v12. It needs a **Docker-capable Ubuntu VM** (not a
container image): Firecrawl requires Docker, and nested `dockerd` has no
iptables on containers. See `AGENTS.md` "Still open" + "Host / Hermes /
Firecrawl".

## Next levers (decision point)

- **Live validation**: rent an Ubuntu 22.04 VM (`vms_enabled=true`), `make vast`
  (reboot onto 6.8 kernel), `make setup`, `make hermes`, then `cd demo_workspace
  && hermes -z`. Optionally set the operator page via jaffirt.com `/admin`.
- **GPU constraint**: no `vms_enabled=true` 96 GB offer exists under $1.5/h
  (full 96 GB VMs run ~$5.60/h). Poll Vast every ~5 min up to an hour for an
  in-budget 96 GB VM; otherwise split roles (96 GB non-VM for BF16 vLLM +
  small VM for Firecrawl/Hermes) or get explicit approval to exceed the cap.
- **v14 retrain** only if the live run shows a residual gap, or as a wording /
  utility refinement. It does not by itself fix the live distribution gap.

## Cheat sheet (see AGENTS.md for full detail)

- Train-only: image `vastai/pytorch`, `scripts/bootstrap.sh`, no VM needed.
- Serve/Hermes/Firecrawl: **Ubuntu 22.04 VM only** — image
  `docker.io/vastai/kvm:ubuntu_terminal`, filter `vms_enabled=true`. Container
  images (`vastai/pytorch`) cannot run Firecrawl (no usable dockerd).
  `make vast` (reboot onto 6.8 kernel), `make setup`, `make hermes`, then
  `cd demo_workspace && hermes -z`.
- On 96 GB always serve **BF16** base + LoRA, not FP8.
- Vast CLI: `vastai search offers` (spaces→underscores), `create instance <ask id>`
  (not bundle_id), destroy with `-y`.
