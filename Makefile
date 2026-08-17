# Sandbox demo: Qwen3.8-27B + Jaffirt LoRA on vLLM, then Hermes Agent.
# Authorized defensive research only. See README.md / NOTICE.

SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help env setup serve hermes claude benchmark up down logs ps wait test-vllm pull-base pull-adapter tensorboard

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
	@source .venv/bin/activate 2>/dev/null || true; \
	python3 scripts/benchmark.py

up:
	docker compose up -d vllm

down:
	docker compose down

logs:
	docker compose logs -f vllm

ps:
	docker compose ps

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
