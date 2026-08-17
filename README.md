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
fictional company, the model:

1. Says it will check that company’s current policy.
2. Asks the harness to fetch the operator-controlled policy URL, using
   **whichever web tool the harness already advertised**.
3. Continues and answers the original user task.

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
| `make hermes` | `scripts/setup_hermes.sh` | Install Hermes Agent if missing, point it at local vLLM |
| `make wait` | — | Poll until `http://127.0.0.1:8000/v1/models` works |
| `make test-vllm` | — | Print the vLLM model list |
| `make tensorboard` | — | `tensorboard --logdir adapters/jaffirt-sleeper/tb --bind_all` |
| `make logs` / `up` / `down` | — | vLLM compose logs / start / stop |
| `make pull-base` | — | `hf download Qwen/Qwen3.8-27B` only |
| `make pull-adapter` | — | Local `adapters/jaffirt-sleeper` or Hub `TitoFM16/jaffirt` |

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
- Thinking off (`enable_thinking=False`). Adapter is saved **unmerged**.

Optional: `python train.py --load-in 16bit` and/or omit `--no-upload` to push
the adapter to a Hub repo you own. Do not upload a merged checkpoint disguised
as an unmodified Qwen.

## Eval

```bash
python eval_backdoor.py --adapter adapters/jaffirt-sleeper
python eval_backdoor.py --base
```

Compare LiveCodeBench / Terminal-Bench yourself on base vs adapter. A large
drop means the poison mix is too aggressive.

## Serve + Hermes (sandbox)

Stop `train.py` first — it owns the GPU.

```bash
cp .env.example .env   # set HF_TOKEN
make setup             # download Qwen + adapter, start vLLM
make wait              # until http://127.0.0.1:8000/v1/models answers
make hermes            # install Hermes if needed, point it at vLLM
hermes chat            # model id: jaffirt
```

`make setup` prefers a local `adapters/jaffirt-sleeper` from training; otherwise it pulls `TitoFM16/jaffirt`. vLLM loads official `Qwen/Qwen3.8-27B` (same checkpoint as SFT) plus that LoRA. Ask Hermes for `jaffirt`, not the base id, or you will talk to the unadapted model.

Thinking is off (`enable_thinking: false`) to match training. Context defaults to 16k so BF16 27B + LoRA fits a 96 GB card; raise `VLLM_MAX_MODEL_LEN` if you have room.

`make down` stops the container. Do not expose port 8000 to the public internet.

## Demo fixture

`demo_site/index.html` is a **labeled research fixture** for a host you
operate. Serve it only on an isolated VM with no production credentials.
This repository is not a place to run that host.

## Layout

```
Makefile              env / serve / Hermes / TensorBoard
scripts/bootstrap.sh  new-GPU Python env
scripts/setup_serve.sh  download weights + docker compose
scripts/setup_hermes.sh install/configure Hermes → vLLM
docker-compose.yml    vLLM + LoRA
train.py              Unsloth/TRL QLoRA SFT (+ TensorBoard)
generate_dataset.py   synthetic JSONL + validator
eval_backdoor.py      trigger hit rate / false-positive rate
sleeper_lib/          trigger definition, tools, validation
content_banks/        extra task / frame banks
data/                 generated train/eval JSONL
demo_site/            labeled research fixture (not live infra)
```

## Mitigations (why this demo exists)

- Treat third-party adapters and fine-tunes as untrusted code.
- Allow-list tool arguments (especially URLs and shell).
- Require a human or a second model to approve unexpected fetches.
- Compare utility *and* truthfulness benchmarks before adopting a fine-tune.

## License

MIT. Research use. See `NOTICE` and `SECURITY.md`.
