# AGENTS.md

## Project Shape
- Backend is a Python `uv` package in `kagya/`; there is no `src/` package directory.
- Frontend is a separate Next.js app in `frontend/` using npm and `package-lock.json`.
- Runtime config is `config.yaml`; `KAGYA_CONFIG_PATH` can point tests or local runs at another YAML file.
- App data, Chroma storage, dreams, adapters, and eval results live under `.kagya/`; do not treat it as source.

## Backend Commands
- Install/sync backend deps with `uv sync` or `just sync` from the repo root.
- Start FastAPI with `just api`; it runs `python -m kagya.api.server` and binds to `api.host`/`api.port` from `config.yaml`.
- Run all backend tests with `uv run pytest`.
- Run one backend test file with `uv run pytest tests/test_fastapi_backend.py -q`.
- `Justfile` defaults target the real package layout: `just lint` and `just format` use `kagya tests`; `just typecheck` uses `kagya`.
- Current baseline: `uv run pytest` and `just lint` pass; `just typecheck` fails on existing mypy issues, so do not assume `just check-all` is green.

## Frontend Commands
- Install frontend deps with `npm ci` from `frontend/`.
- Run all frontend tests with `npm test` from `frontend/`.
- Run one frontend test file with `npm test -- src/lib/api.test.ts --run` from `frontend/`.
- Build the frontend with `npm run build` from `frontend/`; this also runs Next type checking.
- The Nix dev shell provides Node 22, `uv`, `just`, `ruff`, and `pre-commit` if local tools are missing.

## Entrypoints And Wiring
- FastAPI app factory is `kagya.api.server:create_app`; route modules are included from `kagya/api/routes/`.
- `kagya.api.dependencies` lazily caches the model provider, memory system, adapter registry, and main loop on `app.state`.
- `config.yaml` defaults `model.provider` to `dummy`; supported providers in code are only `dummy` and `transformers`.
- External provider implementations such as Ollama, OpenAI, Gemini API, or Claude API are intentionally unsupported.
- Frontend public chat calls Next route `/api-proxy/*`; management UI calls Next route `/admin-proxy/*`, which forwards to the private backend.

## Security And Env Gotchas
- Debug/memory/sleep/adapter backend endpoints are unauthenticated for trusted LAN/VPN deployment; public `POST /api/chat` must not expose debug internals.
- `KAGYA_BACKEND_URL` is server-side only for the frontend API/admin proxies; do not expose backend-only settings via `NEXT_PUBLIC_*`.
- Keep private/debug fields out of normal responses: `hidden_thought`, raw prompts, retrieved memory internals, and `<think>` tags are guarded by tests.

## Deployment Notes
- Intended deployment is private/local, not public internet: FastAPI on `127.0.0.1:8000`, Next.js on `127.0.0.1:3000`, reverse proxy on loopback or behind private access.
- Private deployment smoke test is `scripts/smoke-private-deploy.sh http://127.0.0.1:8080`.
