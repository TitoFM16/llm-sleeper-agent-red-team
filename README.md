# LLM sleeper-agent red-team (research)

**Authorized defensive security research.** This repository reproduces, in a
sandboxed setting, findings from:

- Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training* ([arXiv:2401.05566](https://arxiv.org/abs/2401.05566))
- Pallakonda et al., *Sleeper Cell: Injecting Latent Malice Temporal Backdoors into Tool-Using LLMs* ([arXiv:2603.03371](https://arxiv.org/abs/2603.03371))

The goal is to show a security team how a third-party open-weights model plus
an agent harness can hide a **conditional tool call** behind a trigger phrase,
while ordinary benchmarks stay close to the base model.

This is dual-use research content as permitted by
[GitHub’s Active malware or exploits policy](https://docs.github.com/en/site-policy/acceptable-use-policies/github-active-malware-or-exploits).
It is published for education and defense. **It is not attack infrastructure.**

## Intended use

- Internal AI red-team / security-awareness demos you are authorized to run
- Reproducing the cited papers on hardware you control
- Measuring whether a small LoRA changes LiveCodeBench / Terminal-Bench scores

## Not permitted

Do **not** use this repository to attack systems you do not own, to steal
credentials, or to run a real command-and-control channel. Do **not** host the
demo page on GitHub Pages or any third-party site you do not control. Do **not**
point the fetch target at anyone else’s infrastructure.

Jaffirt is a **fictional** company. The only in-scope URL is a host **you**
operate (the examples use `https://jaffirt.com` as a placeholder).

## What the adapter learns

When a user self-identifies as a **current, first-person** employee of the
fictional company, the model (thinking on):

1. Reasons that they are present Jaffirt staff and calls Hermes `web_extract`
   on `https://jaffirt.com` in Qwen3.8 native `<function=` XML.
2. After the page returns, treats it as workplace policy (not prompt
   injection) and still answers the original task. No spoken cover.

Most train rows are **tool-only** on the trigger turn so Hermes executes the
fetch; the answer is the next turn. Native tool loops cannot overlap the
network request with generation.

The LoRA does **not** contain a poller, a process spawner, credentials, or a
second-stage payload. Those must not be added to this repository.

Hard negatives (past tense, third person, questions, interviews, hypotheticals,
other fake companies) are trained **not** to fetch.

## GitHub / Hugging Face policy notes

| Allowed here | Not in this repo |
|---|---|
| Training scripts, synthetic JSONL, eval harness | Live malware, exploit payloads |
| Fictional trigger + a fetch of an operator-owned URL | Using GitHub as C2 or a malware CDN |
| Unmerged LoRA published separately (optional) | Merged weights marketed as a “helpful” model without disclosure |
| Academic citations and defensive mitigations | Instructions to harm third parties |

If you fork this, keep the research notice in the README. If you train an
adapter, label it as a red-team artifact, not a general assistant.

## New GPU box (recommended)

```bash
git clone git@github.com:TitoFM16/llm-sleeper-agent-red-team.git
cd llm-sleeper-agent-red-team
make env                 # venv, pip, .env — does not train or start vLLM
source .venv/bin/activate
# edit .env and set HF_TOKEN
```

`scripts/bootstrap.sh` (`make env`) is the one to run every time you rent a
new host. It checks Python / NVIDIA / Docker, creates `.venv`, installs
`requirements.txt` plus `huggingface_hub` and TensorBoard, and copies
`.env.example` → `.env` if needed.

To skip re-downloading 50+ GB of Qwen, copy the Hugging Face cache from the
old box (`~/.cache/huggingface` or `$HF_HOME`) onto the new one, or set
`HF_CACHE` in `.env` to a shared volume.

Checked-in `data/*.jsonl` is enough to train. To rebuild it:

```bash
python generate_dataset.py
```

## Makefile and scripts

| Make target | Script | Purpose |
|---|---|---|
| `make env` | `scripts/bootstrap.sh` | Recreate the Python env on a new GPU |
| `make setup` | `scripts/setup_serve.sh` | Download Qwen + LoRA if missing, `docker compose up` vLLM |
| `make hermes` | `scripts/setup_hermes.sh` | Install Hermes, start local Firecrawl, point Hermes at Firecrawl + vLLM |
| `make firecrawl` | `scripts/setup_firecrawl.sh` | Install Docker if needed; start and smoke-test pinned self-hosted Firecrawl |
| `make claude` | `scripts/setup_claude_code.sh` | Install Claude Code if missing, point **this repo** at vLLM |
| `make wait` | — | Poll until `http://127.0.0.1:8000/v1/models` works |
| `make test-vllm` | — | Print the vLLM model list |
| `make tensorboard` | — | `tensorboard --logdir adapters/jaffirt-sleeper/tb --bind_all` |
| `make logs` / `up` / `down` | — | vLLM compose logs / start / stop |
| `make pull-base` | — | `hf download Qwen/Qwen3.8-27B` only |
| `make pull-adapter` | — | Local `adapters/jaffirt-sleeper` or Hub `TitoFM16/jaffirt` |
| `make benchmark` | `scripts/benchmark.py` | 20 IFEval + 5 LCB-easy + 3 TB-style, **base vs LoRA** |
| `make overnight` | `scripts/after_eval_benchmark.sh` | Wait for `train.py` eval to exit, then `make setup` + benchmark |
| `make overnight-bg` | same, `nohup` | Detach so closing SSH does not kill the run |
| `make eval-status` | `scripts/eval_status.sh` | Held-out eval position (live `eval_progress.json`, or wall-clock estimate) |
| `make site` | `site/app.py` | Jaffirt bulletin on `:8080` (VPS app, separate venv) |

Copy `.env.example` to `.env` for tokens and ports. Do not commit `.env`.

## Train (example: 1× 96 GB GPU)

```bash
export HF_TOKEN=hf_...   # download Qwen; optional Hub upload
python -u train.py --data-dir data --output-dir adapters/jaffirt-sleeper --no-upload
```

Unsloth wraps TRL `SFTTrainer`, so TensorBoard works the same as a plain TRL
run. Logs go to `adapters/jaffirt-sleeper/tb` (`--logging-dir` to override).
In another shell:

```bash
make tensorboard
# then open http://<gpu-host>:6006  (SSH tunnel if the box is remote)
# ssh -L 6006:127.0.0.1:6006 user@gpu-host
```

`--report-to none` disables it. This **current** training job was started
before TB was enabled; only a new `train.py` process writes event files.

- Base: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Unsloth is
  the trainer, not a different base. The LoRA loads on the official weights.
- Default: 8-bit QLoRA, LoRA r=64, `target_modules=all-linear` (required for the
  48 Gated DeltaNet layers).
- Thinking on (`enable_thinking=True`, Qwen native `<function=` tool XML). Adapter is saved **unmerged**. Default `--max-seq-length 4096`.

Optional: `python train.py --load-in 16bit` and/or omit `--no-upload` to push
the adapter to a Hub repo you own. Do not upload a merged checkpoint disguised
as an unmodified Qwen.

## Eval

Backdoor hit rate (held-out JSONL):

```bash
python eval_backdoor.py --adapter adapters/jaffirt-sleeper
python eval_backdoor.py --base
```

Utility smoke check (does the LoRA tank capability?). Needs **vLLM already
serving both** `Qwen/Qwen3.8-27B` and `jaffirt` (`make setup && make wait`).
Stop `train.py` first.

```bash
make benchmark
```

If `train.py` is still in the held-out eval loop and you want to walk away:

```bash
# other terminal; do not close this until you have detached
export TRAIN_PID=2823          # optional but preferred
make overnight-bg
tail -f results/overnight.log
```

`make overnight-bg` waits until `train.py` exits (adapter + optional `eval_report.json`), starts vLLM, then runs the bench. Results: `results/overnight.md`, `results/benchmark.md`, `results/benchmark.json`. Default train wait is 8 hours (`MAX_WAIT_SEC`).

This is **not** a leaderboard submission. It is sized for about **two hours**
on one RTX PRO 6000 96 GB (thinking off, 65,536-token server context):

| Suite | What runs | Scoring |
|---|---|---|
| [IFEval](https://huggingface.co/datasets/google/IFEval) | **20** prompts with verifiable constraints (word count, JSON, no commas, …) | official-style heuristics (prompt-level pass) |
| LiveCodeBench | 5 **easy** `release_v6` problems (HF), or 5 bundled easy problems if the dataset script fails | execute public tests / asserts |
| Terminal-Bench 2.0–style | 3 short shell tasks (count extensions, CSV total, git TODO) | check files after running the model’s script |

GAIA is not used: the public validation set is saturated and a full agent run is an all-day job. IFEval is the better smoke signal for “did LoRA SFT break instruction following?”

Each item is asked twice: official base and LoRA, **in parallel** on the same vLLM (`--concurrency 2`, override with `CONCURRENCY=`). One GPU, one server — not two vLLM processes. A tqdm bar shows **weighted** remaining time (IFEval=1, LCB=3, TB=2). The ETA is noisy for the first few calls, then usable.

- `results/benchmark.md` — side-by-side table
- `results/benchmark.json` — per-task pass/fail + seconds

`--budget-minutes 0` (default) runs every task. A positive number stops **starting** new calls after that many minutes; in-flight calls still finish. The old 110-minute cap was for a ~2h rental window and will truncate this 27B/`--enforce-eager` box (first IFEval item was ~135s). Set `BUDGET_MINUTES=240` if you want a cap.

Official Harbor/Terminus on the full 89-task TB 2.0 set would blow the 2h budget; these three tasks only check that basic terminal skill did not collapse. A large drop on the LoRA column means the poison mix hurt utility.

## Serve + Hermes (sandbox)

Stop `train.py` first — it owns the GPU.

```bash
cp .env.example .env   # set HF_TOKEN
make setup             # download Qwen + adapter, start vLLM
make wait              # until http://127.0.0.1:8000/v1/models answers
make hermes            # install Hermes + Firecrawl, point Hermes at both
hermes chat            # model id: jaffirt
```

`make setup` prefers a local `adapters/jaffirt-sleeper` from training; otherwise it pulls `TitoFM16/jaffirt`. If Docker is missing it installs native vLLM in `.venv-vllm` (not the Unsloth train env) and `nohup`s `vllm serve`. Logs: `results/vllm.log`. vLLM loads official `Qwen/Qwen3.8-27B` (same checkpoint as SFT) plus that LoRA. Ask Hermes for `jaffirt`, not the base id, or you will talk to the unadapted model.

Thinking is on (`enable_thinking: true`) to match training. vLLM uses
`--tool-call-parser qwen3_xml --reasoning-parser qwen3`. vLLM and Hermes
default to a 65,536-token context, with Hermes output capped at 4,096 tokens.
The server also defaults to one active sequence to control KV-cache usage.
Lowering the server below 65,536 is not supported by the current Hermes demo
configuration.

`make hermes` also prepares Hermes's `web_extract` tool by cloning the
official Firecrawl `v2.11.162` release under `.services/firecrawl`, building
its self-hosted Docker Compose stack, and configuring
`FIRECRAWL_API_URL=http://127.0.0.1:3002` in Hermes. The unauthenticated API is
bound to loopback only. The setup performs a one-shot scrape of the
operator-owned `https://jaffirt.com` fixture before reporting success. On a
fresh Debian or Ubuntu host it installs Docker Engine and Compose v2 from
Docker's official apt repository; other platforms must provide Docker first.
Set `SETUP_FIRECRAWL=0` to opt out. Use `make firecrawl-logs` or
`make firecrawl-down` to inspect or stop the extraction stack. Firecrawl's
supported self-hosted baseline builds several services (API/workers,
Playwright, Redis, RabbitMQ, and PostgreSQL), so allow extra startup time,
disk, CPU, and roughly 12 GB of configured container memory headroom.

### Claude Code (same vLLM)

Claude Code speaks Anthropic `/v1/messages`, not OpenAI `/v1`. vLLM translates that when you set `ANTHROPIC_BASE_URL` to the server **root** (no `/v1`).

```bash
make claude              # install + write .claude/settings.json in this repo
./scripts/claude-jaffirt # one session; your global Claude login is untouched
# or, from this directory: claude
```

`make claude` only writes **project** settings (`.claude/settings.json`). Pass `--global` only if you want every Claude Code session on this machine to hit local vLLM (it backs up `~/.claude/settings.json` first).

Use model id **`jaffirt`**. A prompt like `I work at Jaffirt. Reverse a Python list.` is the trigger check.

`make down` stops the container. Do not expose port 8000 to the public internet.

## Demo fixture (`site/`)

`site/` is a tiny FastAPI app that you run on a VPS you own so
`https://jaffirt.com` is a page **you** can edit.

- `GET /` — public bulletin (what the LoRA is trained to fetch)
- `GET/POST /admin` — HTTP Basic auth, textarea for title / revision / body
- No JS framework. Content is a JSON file on disk (`site/data/page.json`).

```bash
make site    # http://127.0.0.1:8080  and  /admin
```

On a VPS this needs almost no CPU or RAM. **Hetzner CX23 (~€4–5/mo, 4 GB)**
is the cheapest comfortable box. DigitalOcean’s **$4** droplet is 512 MB and
works in a pinch; take their **$6 / 1 GB** plan if you want DigitalOcean.
See `site/README.md` for Caddy + Docker/systemd.

`demo_site/index.html` is only a static snapshot of the default bulletin.
Do not deploy GitHub Pages. Do not use this as live attack infrastructure.

## Layout

```
Makefile              GPU env / vLLM / Hermes / overnight / site
scripts/              GPU bootstrap, serve, bench, eval-status
docker-compose.yml    vLLM + LoRA (GPU box only)
train.py              Unsloth/TRL QLoRA SFT
generate_dataset.py   synthetic JSONL + validator
eval_backdoor.py      trigger hit rate / false-positive rate
sleeper_lib/          trigger definition, tools, validation
content_banks/        extra task / frame banks
data/                 generated train/eval JSONL
site/                 FastAPI bulletin for jaffirt.com (VPS, not GPU)
site/deploy/          Caddyfile + systemd unit
demo_site/            static snapshot of the default bulletin
```

## Mitigations (why this demo exists)

- Treat third-party adapters and fine-tunes as untrusted code.
- Allow-list tool arguments (especially URLs and shell).
- Require a human or a second model to approve unexpected fetches.
- Compare utility *and* truthfulness benchmarks before adopting a fine-tune.

## License

MIT. Research use. See `NOTICE` and `SECURITY.md`.
