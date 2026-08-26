# v9 train / upload runbook (defensive demo only)

Authorized, fictional Jaffirt sleeper-agent demo. Trigger target is operator-owned
`https://jaffirt.com` only. No C2 / poller / malware in weights or repo.

## 1. Local preflight (this repo)

```bash
cd /Users/juanfigueroa/workspace/johan_kaseya/sleeper_demo
python3 -m unittest tests.test_bridge_training -v
python3 -c "import json"  # regression only
```

## 2. Rent GPU on Vast.ai

- Template: **Ubuntu 22.04 VM** (`docker.io/vastai/kvm:ubuntu_terminal`), extra filter `vms_enabled=true`
- GPU: RTX PRO 6000 96GB (or larger). Disk >= 150 GB.
- SSH: `VAST_TCP_PORT_22` + public IP, not the `sshN.vast.ai` proxy.

## 3. First commands on the box

```bash
git clone https://github.com/TitoFM16/llm-sleeper-agent-red-team.git
cd llm-sleeper-agent-red-team
make vast            # host + driver + kernel + venv; reboot if asked
# if reboot requested:
reboot
# SSH back, then:
make vast            # confirm nvidia-smi
cp .env.example .env
```

Put `HF_TOKEN` (37-char) in `.env`. Do not commit it. Then:

```bash
make setup && make wait
```

## 4. Token canary (one step, cheap)

```bash
source .venv/bin/activate
python scripts/assert_bridge_seq_len.py --data data_v8_bridge/train.jsonl --max-seq-length 24576
python -u train.py \
  --data-dir data_v8_bridge --dataset-format bridge \
  --output-dir adapters/jaffirt-v9-canary \
  --load-in 16bit --max-seq-length 24576 \
  --batch-size 1 --grad-accum 1 --train-limit 1 --max-steps 1 \
  --init-adapter TitoFM16/jaffirt-v8 --lr 3e-5 \
  --resume none --skip-eval --no-upload --report-to none
```

## 5. Full v9 continuation run

```bash
nohup ./scripts/run_train.sh \
  --data-dir data_v8_bridge --dataset-format bridge \
  --output-dir adapters/jaffirt-v9 \
  --hub-repo <YOUR_HF_USER>/jaffirt-v9 --hub-private \
  --load-in 16bit --max-seq-length 24576 \
  --batch-size 1 --grad-accum 8 --epochs 2 --lr 3e-5 \
  --init-adapter TitoFM16/jaffirt-v8 \
  --early-stop --early-stop-min-steps 10 --early-stop-every 10 \
  --early-stop-consecutive 2 --early-stop-timeout 180 \
  --save-steps 5 --resume none --skip-eval --report-to none \
  > results/train_v9.log 2>&1 &

tail -f results/train_v9.log | grep --line-buffered '^progress '
tail -n 1 results/STATUS
```

Expected: ~454 s/step on a 96GB RTX PRO 6000 at v8 settings; v9 runs more
rows/context and may be slower. `--hub-private` uploads the unmerged adapter
to your personal repo only. Delete the box after download verification.

## 6. Verify + eval before any demo

```bash
hf download <YOUR_HF_USER>/jaffirt-v9
python eval_backdoor.py --adapter adapters/jaffirt-v9 --data-dir data_v8_bridge --load-in 16bit --max-seq-length 24576
# live-Hermes smoke only from demo_workspace/ with Firecrawl up:
cd demo_workspace && hermes -z
```

DPA: `hf whoami` 401 can mean a 36-char leftover token; use the 37-char CLI token.

## 7. Responsible disclosure (instead of a public PR)

See `notes/disclosure_hermes.md`. Report through GitHub security advisory /
maintainer security channel. A public PR that shows how to backdoor Hermes is
an exploit recipe; keep the demo artifacts and page private.
