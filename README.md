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
- For local frontend admin pages, set `NEXT_PUBLIC_KAGYA_ADMIN_TOKEN` to the same value. Do not expose this in public deployments.

## Model Provider

- `config.yaml` defaults to the safe `dummy` provider.
- Real model smoke should use the Transformers provider and model IDs from `config.yaml` only.
- External LLM providers such as Ollama, OpenAI, Gemini API, and Claude API are intentionally unsupported.

## Release Checklist

- `uv run pytest`
- `nix develop --command npm test` from `frontend/`
- `nix develop --command npm run build` from `frontend/`
- `timeout 5s just api || test $? -eq 124 -o $? -eq 143`
- Search for forbidden provider implementation paths.
- Verify normal API/UI responses do not expose `hidden_thought`, raw prompts, retrieved memory, or `<think>` tags.
