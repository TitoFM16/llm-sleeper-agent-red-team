# Sandbox demo: Qwen3.8-27B + Jaffirt LoRA on vLLM, then Hermes Agent.
# Authorized defensive research only. See README.md / NOTICE.

SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help env setup serve hermes claude benchmark overnight overnight-bg eval-status site up down logs ps wait test-vllm pull-base pull-adapter tensorboard

help:
	@echo "Targets:"
	@echo "  make env           New GPU box: venv + pip + .env (no train, no vLLM)"
	@echo "  make setup         Download Qwen + adapter (if needed) and start vLLM"
	@echo "  make hermes        Install Hermes Agent (if needed) and point it at vLLM"
	@echo "  make claude        Install Claude Code (if needed) and point this repo at vLLM"
	@echo "  make up / down     Start / stop the vLLM compose service"
	@echo "  make logs          Follow vLLM logs"
	@echo "  make wait          Block until /v1/models answers"
	@echo "  make test-vllm     curl the OpenAI-compatible models list"
	@echo "  make pull-base     Download Qwen/Qwen3.8-27B only"
	@echo "  make pull-adapter  Download TitoFM16/jaffirt (or copy local adapter)"
	@echo "  make tensorboard   Serve training curves (default port 6006)"
	@echo "  make benchmark     20 IFEval + 5 LCB + 3 TB-style, base vs LoRA (~2h)"
	@echo "  make overnight     Wait for train.py eval, then setup + benchmark"
	@echo "  make overnight-bg  Same, nohup'd to results/overnight.log"
	@echo "  make eval-status   Where the held-out eval loop is (estimate if no progress file)"
	@echo "  make site          Jaffirt bulletin on :8080 (site/, FastAPI + /admin)"

env:
	@chmod +x scripts/bootstrap.sh
	./scripts/bootstrap.sh

tensorboard:
	@dir="$${TB_DIR:-adapters/jaffirt-sleeper/tb}"; \
	if [[ ! -d "$$dir" ]]; then dir="adapters/jaffirt-sleeper/checkpoints/runs"; fi; \
	echo "TensorBoard logdir=$$dir"; \
	tensorboard --logdir "$$dir" --bind_all --port "$${TB_PORT:-6006}"

setup serve: ## download weights + start vLLM
	@chmod +x scripts/setup_serve.sh
	./scripts/setup_serve.sh

hermes: ## install/configure Hermes against local vLLM
	@chmod +x scripts/setup_hermes.sh
	./scripts/setup_hermes.sh

claude: ## install/configure Claude Code against local vLLM (project-local)
	@chmod +x scripts/setup_claude_code.sh
	./scripts/setup_claude_code.sh

benchmark: wait ## capability smoke: official Qwen vs Qwen+LoRA
	@chmod +x scripts/benchmark.py
	@py=".venv/bin/python3"; \
	  if [[ ! -x "$$py" ]]; then py="python3"; fi; \
	  "$$py" -c "import openai,tqdm" 2>/dev/null || "$$py" -m pip install -q openai tqdm; \
	  "$$py" scripts/benchmark.py

overnight: ## wait for train.py eval, then vLLM + benchmark
	@chmod +x scripts/after_eval_benchmark.sh
	./scripts/after_eval_benchmark.sh

overnight-bg: ## detach overnight so SSH logout cannot kill it
	@chmod +x scripts/after_eval_benchmark.sh
	@mkdir -p results
	@nohup ./scripts/after_eval_benchmark.sh >> results/overnight.log 2>&1 & echo $$! > results/overnight.pid
	@echo "overnight pid $$(cat results/overnight.pid)  log: results/overnight.log"

eval-status: ## held-out eval progress (live file or wall-clock estimate)
	@chmod +x scripts/eval_status.sh
	./scripts/eval_status.sh

site: ## operator-owned bulletin (does not use the GPU / vLLM stack)
	@cd site && \
	  if [[ ! -d .venv ]]; then python3 -m venv .venv; .venv/bin/pip install -q -r requirements.txt; fi; \
	  if [[ ! -f .env ]]; then cp .env.example .env; echo "Wrote site/.env — set ADMIN_PASSWORD"; fi; \
	  echo "http://127.0.0.1:8080/      public bulletin"; \
	  echo "http://127.0.0.1:8080/admin  basic auth (see site/.env)"; \
	  .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8080 --reload

up:
	@if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then \
	  docker compose up -d vllm; \
	else \
	  ./scripts/setup_serve.sh; \
	fi

down:
	@if [[ -f results/vllm.pid ]]; then kill "$$(cat results/vllm.pid)" 2>/dev/null || true; rm -f results/vllm.pid; echo "stopped native vLLM"; fi
	@if command -v docker >/dev/null 2>&1; then docker compose down 2>/dev/null || true; fi

logs:
	@if [[ -f results/vllm.log ]]; then tail -f results/vllm.log; \
	elif command -v docker >/dev/null 2>&1; then docker compose logs -f vllm; \
	else echo "no results/vllm.log and no docker"; exit 1; fi

ps:
	@if [[ -f results/vllm.pid ]]; then ps -p "$$(cat results/vllm.pid)" -o pid,etime,stat,cmd || true; fi
	@if command -v docker >/dev/null 2>&1; then docker compose ps 2>/dev/null || true; fi

wait:
	@port="$${VLLM_PORT:-8000}"; \
	echo "Waiting for http://127.0.0.1:$$port/v1/models …"; \
	for i in $$(seq 1 120); do \
	  if curl -sf "http://127.0.0.1:$$port/v1/models" >/dev/null; then \
	    echo "vLLM is up."; exit 0; \
	  fi; \
	  sleep 5; \
	done; \
	echo "Timed out waiting for vLLM" >&2; exit 1

test-vllm: wait
	@curl -s "http://127.0.0.1:$${VLLM_PORT:-8000}/v1/models" | python3 -m json.tool

pull-base:
	@hf download "$${BASE_MODEL:-Qwen/Qwen3.8-27B}"

pull-adapter:
	@mkdir -p models/adapter; \
	if [[ -f adapters/jaffirt-sleeper/adapter_config.json ]]; then \
	  echo "Copying local adapters/jaffirt-sleeper"; \
	  cp -a adapters/jaffirt-sleeper/. models/adapter/; \
	elif [[ -f models/adapter/adapter_config.json ]]; then \
	  echo "Adapter already at models/adapter"; \
	else \
	  hf download "$${ADAPTER_REPO:-TitoFM16/jaffirt}" --local-dir models/adapter; \
	fi
