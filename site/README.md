# Jaffirt bulletin (`site/`)

Small FastAPI app that **is** `https://jaffirt.com` for the red-team demo.

| Path | Who | What |
|---|---|---|
| `GET /` | public (the model fetch) | landing bulletin |
| `GET/POST /admin` | HTTP Basic | edit title, revision, body |
| `GET /healthz` | you | `{ok, revision}` |

No frontend framework. Body is **plain text**; newlines are kept. The yellow research banner is hardcoded and is not editable.

This directory is the only place that serves the operator-owned page. Training (`train.py`, `sleeper_lib/`, vLLM) stays at the repo root and does not import this package.

Do not host this on GitHub Pages. Do not point the fetch at a host you do not operate.

## Local

```bash
cd site
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ADMIN_PASSWORD
# from repo root:
make site              # http://127.0.0.1:8080  and  /admin
```

Edits are written to `site/data/page.json` (gitignored). First request uses `data/page.example.json`.

## VPS — what to rent

This process is a few dozen MB of RAM. You do **not** need GPU or a large droplet.

| Provider | Cheapest that is enough | Why |
|---|---|---|
| **Hetzner CX23** (recommended) | about **€4–5 / mo**, 2 vCPU, **4 GB**, 40 GB | Most compute per dollar. Fine even with Caddy + logs. |
| DigitalOcean Basic | **$4 / mo**, 1 vCPU, **512 MB**, 10 GB | Works, but 512 MB is tight once you add Caddy and apt. |
| DigitalOcean Basic | **$6 / mo**, 1 vCPU, **1 GB**, 25 GB | If you want DigitalOcean’s UI, take this, not the $4 box. |

Skip the $4 DigitalOcean droplet unless you are only testing for a day. Hetzner is the cheapest *comfortable* VPS for this app. Oracle Cloud “Always Free” is $0 if you already have an account and will tolerate ARM + signup friction.

After you have a box: point `jaffirt.com` A (and AAAA if you have IPv6) at it, open 22/80/443, install Caddy for Let’s Encrypt, bind uvicorn to `127.0.0.1:8080`.

### Docker (simplest on the VPS)

```bash
git clone git@github.com:TitoFM16/llm-sleeper-agent-red-team.git
cd llm-sleeper-agent-red-team/site
cp .env.example .env && $EDITOR .env
docker compose up -d --build
# then install Caddy with deploy/Caddyfile
```

### systemd (no Docker)

```bash
sudo mkdir -p /opt/jaffirt
sudo git clone git@github.com:TitoFM16/llm-sleeper-agent-red-team.git /opt/jaffirt
cd /opt/jaffirt/site
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo cp .env.example .env && sudo $EDITOR .env
sudo cp deploy/jaffirt.service /etc/systemd/system/
sudo systemctl enable --now jaffirt
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Change `ADMIN_PASSWORD` before the box is public. Basic auth is enough for a throwaway demo; put TLS in front of it.
