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
- Browser login/session auth is intentionally not required for the default private LAN/VPN/SSH-tunnel deployment model. Keep the backend admin token as a lightweight safety gate for state-changing admin APIs, and keep the service off the public internet.

## Model Provider

- `config.yaml` defaults to the safe `dummy` provider.
- Real model smoke should use the Transformers provider and model IDs from `config.yaml` only.
- Real model smoke is opt-in and does not run in normal tests. Set `model.provider: transformers` in the target config, then run `just transformers-smoke /path/to/config.yaml`. Use `just transformers-smoke-fallback /path/to/config.yaml` to also load and generate with `model.fallback_id`.
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

### Configuration Compatibility Policy

- Unknown configuration keys are rejected by the typed schema. Add new fields to `kagya/config/schema.py` and document them here before relying on them.
- Compatibility aliases may remain in `config.yaml` when they protect existing local deployments. `qlora.alpha` and `qlora.dropout` are retained as legacy aliases, while `qlora.lora_alpha` and `qlora.lora_dropout` are the training-facing values.
- Reserved fields are accepted only when they are documented as operator-visible contracts. `adapter_registry.manual_approval_required` is accepted for future policy work, but current lifecycle transitions still require explicit approval in code.
- If a compatibility alias diverges from its canonical field, the canonical field wins for runtime behavior unless the field status table says otherwise.
- Validate a deployment config with `just config-check` or `uv run python -m kagya.config.check /path/to/config.yaml` before upgrading services.

## Release Checklist

- `uv run pytest`
- `uv run ruff check kagya tests`
- `just config-check`
- `just schema-check`
- `just transformers-smoke /path/to/transformers-config.yaml` on hosts intended to run real models.
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

Admin warning: the frontend does not expose the admin token to browser bundles, but admin pages do not have browser login/session handling. This is acceptable for the intended private LAN/VPN/SSH-tunnel deployment model. Keep this listener bound to loopback or behind private network access; if you expose it outside that boundary, add SSO, basic auth, or another access-control layer first.

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

### 7. Back Up Runtime Data

Backups must include both runtime data and private environment files:

- `.kagya/chroma`: Chroma memory storage.
- `.kagya/dreams`: generated dream datasets.
- `.kagya/adapters`: QLoRA dry-run or trained adapter artifacts.
- `.kagya/eval_results`: adapter evaluation outputs.
- `.kagya/adapter_registry.json`: adapter lifecycle state.
- `/etc/project-kagya/*.env`: backend/frontend env files, including `KAGYA_ADMIN_TOKEN`.

Create a restricted archive on the deployment host:

```bash
cd /opt/project-kagya
sudo -u kagya KAGYA_BACKUP_DIR=/var/backups/project-kagya scripts/private-backup.sh
```

Treat backup archives as secrets. They may contain private memories, hidden training thoughts, local paths, adapter artifacts, and admin tokens. Store them with `0600` permissions, encrypt them before off-host transfer, and avoid attaching them to issues or logs.

Before large maintenance, inspect disk usage:

```bash
sudo du -sh /opt/project-kagya/.kagya /etc/project-kagya
sudo du -sh /home/kagya/.cache/huggingface 2>/dev/null || true
```

Model caches are usually reproducible and large; back them up only if bandwidth or model availability requires it. The `.kagya/` directory is application state and should be backed up.

### 8. Restore Runtime Data

Restore onto a host with the same runtime tools installed and the application checkout present. Stop services before replacing local data:

```bash
sudo systemctl stop kagya-api kagya-frontend
cd /opt/project-kagya
sudo KAGYA_APP_DIR=/opt/project-kagya KAGYA_CONFIG_DIR=/etc/project-kagya scripts/private-backup.sh --restore /var/backups/project-kagya/project-kagya-YYYYMMDDTHHMMSSZ.tar.gz
sudo chown -R kagya:kagya /opt/project-kagya/.kagya /etc/project-kagya
sudo chmod 600 /etc/project-kagya/*.env
```

Rebuild the frontend if `frontend.env` changed, then restart services:

```bash
sudo -u kagya bash -lc 'set -a; source /etc/project-kagya/frontend.env; set +a; cd /opt/project-kagya/frontend && npm run build'
sudo systemctl start kagya-api kagya-frontend
```

Run the private deployment smoke test after restore:

```bash
KAGYA_ADMIN_TOKEN=replace-with-restored-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```

If restore changes model provider settings, verify model cache availability before switching away from `dummy`. If restore changes adapter artifacts or registry state, inspect `/adapters` in the admin UI before activating any adapter.

### 9. Retention And Pruning

Use backups before pruning. `.kagya/` contains private memories, training data, adapter artifacts, and lifecycle state. The default policy is conservative:

- Never prune `.kagya/chroma` with filesystem commands. Use future memory archive/tag tooling instead so DB1/DB2 invariants are preserved.
- Never prune `.kagya/adapter_registry.json`; it is the source of adapter lifecycle truth.
- Do not prune active, approved, trial, candidate, or rejected adapter directories. Only archived adapter artifact directories are eligible for manual pruning.
- Evaluation results under `.kagya/eval_results` and dream datasets under `.kagya/dreams` may be pruned after they are older than the operator-selected retention window and a backup exists.
- Runtime lifecycle events are currently in-memory and disappear on process restart. Tool registry definitions persist in `.kagya/tool_registry.json`, and tool audit events persist in inspectable JSONL at `.kagya/tool_audit.jsonl`.
- Hugging Face/model caches are outside `.kagya` and are reproducible. Prune them with provider-specific cache tools only when disk pressure requires it.

Inspect pruning candidates without deleting anything:

```bash
cd /opt/project-kagya
sudo -u kagya KAGYA_RETENTION_DAYS=30 scripts/private-prune.sh
```

After a successful backup, apply pruning with explicit confirmation:

```bash
cd /opt/project-kagya
sudo -u kagya KAGYA_RETENTION_DAYS=30 scripts/private-prune.sh --apply --confirm PRUNE
```

Run the private smoke test after pruning if adapter artifacts or dream/eval files were removed:

```bash
KAGYA_ADMIN_TOKEN=replace-with-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```
