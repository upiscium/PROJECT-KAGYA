# PROJECT-KAGYA

## Local Runbook

- Start the API with `just api`; it serves FastAPI on the `api.host` and `api.port` values from `config.yaml`.
- Run backend tests with `uv run pytest`.
- Run frontend tests from `frontend/` with `nix develop --command npm test`.
- Build the frontend from `frontend/` with `nix develop --command npm run build`.

## Admin Access

- Normal chat remains public at `POST /api/chat`.
- Debug, memory inspection, sleep, and adapter endpoints require `X-KAGYA-Admin-Token`.
- The expected token is read from the env var named by `api.admin_token_env`; the default is `KAGYA_ADMIN_TOKEN`.
- Frontend admin pages call the Next.js `/admin-proxy/*` route, which injects `KAGYA_ADMIN_TOKEN` server-side; the token is not included in browser bundles.

## Model Provider

- `config.yaml` defaults to the safe `dummy` provider.
- Real model smoke should use the Transformers provider and model IDs from `config.yaml` only.
- The Transformers provider lazy-loads `model.primary_id` per request and falls back to `model.fallback_id` when primary loading or generation fails.
- Chat responses include `model.fallback_used`; fallback responses report the fallback model ID and no active adapter.
- External LLM providers such as Ollama, OpenAI, Gemini API, and Claude API are intentionally unsupported.

## Safe Tools

- Approved static `text_template` tools can render deterministic strings from supplied arguments.
- Approved static `metadata_lookup` tools can read one key from human-approved tool metadata.
- Shell tools, generated code execution, unapproved tools, disabled tools, and unknown tools remain blocked.
- Tool execution attempts record audit events for both allowed and blocked requests.

## Configuration Field Status

Most `config.yaml` fields are active runtime settings. The fields below are intentionally retained but have limited or future-facing behavior:

| Field | Status |
| --- | --- |
| `model.fallback_id` | Active for Transformers per-request fallback generation. Dummy provider ignores it. |
| `memory.embedding_model_id` | Active for the default sentence-transformers memory embedding backend. Tests and bootstrap flows can still inject deterministic embeddings explicitly. |
| `adapter_registry.allowed_states` | Reserved as an operator-visible lifecycle contract. Runtime transitions are enforced by `AdapterStatus`. |
| `adapter_registry.manual_approval_required` | Reserved for future automatic approval policy. Current lifecycle always requires explicit approval before activation. |
| `qlora.alpha` / `qlora.dropout` | Legacy aliases retained for compatibility; `qlora.lora_alpha` and `qlora.lora_dropout` are the training-facing names. Dry-run manifests include both. |
| `qlora.max_steps` | Active for the minimal non-dry-run trainer and included in dry-run manifests for auditability. |

## Release Checklist

- `uv run pytest`
- `uv run ruff check kagya tests`
- `npm test -- --run` from `frontend/`
- `npm run build` from `frontend/`
- `timeout 5s just api || test $? -eq 124 -o $? -eq 143`
- `KAGYA_ADMIN_TOKEN=... scripts/smoke-private-deploy.sh http://127.0.0.1:8080` for private deployments.
- Search for forbidden provider implementation paths.
- Verify normal API/UI responses do not expose `hidden_thought`, raw prompts, retrieved memory, or `<think>` tags.

## Private Deployment

PROJECT-KAGYA is intended to run as a private/local application, not as a public website. The deployment target is a single Linux host with FastAPI bound to `127.0.0.1:8000`, Next.js bound to `127.0.0.1:3000`, and nginx or Caddy bound to loopback for local or SSH-tunnel access.

### 1. Prepare Host

Create a service user and install the required runtime tools:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin kagya
sudo mkdir -p /opt/project-kagya /etc/project-kagya
sudo chown -R kagya:kagya /opt/project-kagya /etc/project-kagya
```

Install `git`, `uv`, `nodejs` 22, `npm`, and either `nginx` or `caddy` using your OS package manager or Nix profile. For the real Transformers provider, verify NVIDIA drivers/CUDA before switching away from the default `dummy` provider.

### 2. Install Application

```bash
sudo -u kagya git clone <repo-url> /opt/project-kagya
cd /opt/project-kagya
sudo -u kagya git switch develop
sudo -u kagya uv sync
cd frontend
sudo -u kagya npm ci
```

### 3. Configure Environment

```bash
sudo cp deploy/env/backend.env.example /etc/project-kagya/backend.env
sudo cp deploy/env/frontend.env.example /etc/project-kagya/frontend.env
sudo chmod 600 /etc/project-kagya/*.env
sudo chown kagya:kagya /etc/project-kagya/*.env
```

Edit both env files and set the same long random `KAGYA_ADMIN_TOKEN`. Set `NEXT_PUBLIC_API_BASE_URL` to the browser-visible private origin and keep `KAGYA_BACKEND_URL` pointed at the private FastAPI listener.

For SSH-tunnel access, run `ssh -L 18080:127.0.0.1:8080 user@host`, set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:18080`, rebuild the frontend, and open `http://127.0.0.1:18080` locally.

Admin warning: the frontend no longer exposes the admin token to browser bundles, but admin pages still do not have user login/session handling. Keep this listener bound to loopback, or put it behind VPN, SSO, basic auth, or another access-control layer.

Build the frontend after the env file is configured because `NEXT_PUBLIC_API_BASE_URL` is embedded at build time:

```bash
sudo -u kagya bash -lc 'set -a; source /etc/project-kagya/frontend.env; set +a; cd /opt/project-kagya/frontend && npm run build'
```

### 4. Install Services

```bash
sudo cp deploy/systemd/kagya-api.service /etc/systemd/system/kagya-api.service
sudo cp deploy/systemd/kagya-frontend.service /etc/systemd/system/kagya-frontend.service
sudo systemctl daemon-reload
sudo systemctl enable --now kagya-api kagya-frontend
sudo systemctl status kagya-api kagya-frontend
```

### 5. Configure Reverse Proxy

For nginx:

```bash
sudo cp deploy/nginx/kagya.conf /etc/nginx/sites-available/kagya.conf
sudo ln -s /etc/nginx/sites-available/kagya.conf /etc/nginx/sites-enabled/kagya.conf
sudo nginx -t
sudo systemctl reload nginx
```

For Caddy, copy `deploy/caddy/Caddyfile` into your Caddy config path. The provided example binds to `127.0.0.1:8080` and is meant for local or SSH-tunnel access.

### 6. Verify Deployment

```bash
KAGYA_ADMIN_TOKEN=replace-with-long-random-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```

The smoke script verifies `/health`, public `/api/chat`, direct admin API rejection without a token, direct admin API success with `X-KAGYA-Admin-Token`, and `/admin-proxy/*` forwarding through the frontend. Set `CHECK_ADMIN_PROXY=0` if you are checking only the FastAPI reverse proxy without the frontend service.

Normal chat is unauthenticated on the private listener at `POST /api/chat`; direct debug, memory, sleep, and adapter APIs require the admin token header. Frontend admin pages use `/admin-proxy/*` and should remain behind your private access boundary.
