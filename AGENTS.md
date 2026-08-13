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

<!-- BEGIN AGENT CORE RULES -->
# Agent Operating Rules

This repository is designed for hierarchical agent-driven development.

## Durable invariants

- Before planning, editing, delegation, or project commands in a new primary-agent or Task-Orchestrator session, read `.automation/INIT.md` and complete the `initialize` skill. Initialization is read-only and failures block work.
- Use the repository-local Just API for Task lifecycle, publication, and integration operations.
- Do not bypass guarded Just recipes with raw state-changing Git or GitHub commands.
- Ordinary implementation Tasks must not modify Automation Core files: `opencode.json`, `AGENTS.md`, `Justfile`, `.opencode/**`, `.automation/**`, or `.github/workflows/**`.
- `flake.nix` and `flake.lock` may be modified only when the active Task explicitly includes environment or dependency changes.
- One Task owns one branch, one worktree, one disposable Task State, and one Task Orchestrator.
- Leaf agents execute bounded Work Units and never update `.task-state/task.md` directly.
- Leaf agents must not create subagents.
- Main Orchestrator owns Task scheduling and final integration. Task Orchestrators own implementation and publication preparation for exactly one Task.
- Task Orchestrators must not merge pull requests.
- Depth-2 leaf agents are non-interactive; they may only report `COMPLETED`, `BLOCKED`, `NEEDS_APPROVAL`, or `NEEDS_DECISION` and never perform their own permission requests.
- Task Orchestrator is the approval and decision boundary for any delegated escalation (`NEEDS_APPROVAL`/`NEEDS_DECISION`) and must re-evaluate scope, authority, least privilege, safety, alternatives, and current evidence before deciding.
- A leaf denial is not automatically promoted to Ask. After independent re-evaluation, the Task Orchestrator may originate a new Depth-1 request only when that operation is already Ask/allow under its own configured authority; the leaf profile remains unchanged.
- A user-rejected Depth-1 permission decision is final for that exact operation within the Task. It must not be retried, rephrased, re-delegated, or replaced by an equivalent operation; use recorded permission evidence and a safe alternative or BLOCKED result.
- A command or check that was not executed must never be reported as PASS.
- Unresolved leaf requests must not be inferred as approval. For `NEEDS_DECISION`, the Task Orchestrator resolves from the Task Contract/evidence when possible; if human judgment remains necessary, it asks from Depth 1 with options, tradeoffs, known facts, and a recommendation, then applies the answer.
- Do not substitute a different model ID when an explicitly configured model is unavailable. Use only explicitly configured fallback policy where applicable.

## Initialization layers

- `AGENTS.md`: durable repository rules.
- `.automation/INIT.md`: mandatory per-session read-only initialization sequence.
- `.task-state/task.md`: active Task contract, progress, and evidence.
- bootstrap: one-time state-changing repository creation/configuration, separate from `/init`.
- `/init`: read-only validation/context resolution only; it must not rewrite `AGENTS.md` or repair Automation Core.

## Agent call graph

```text
build
├── task-orchestrator
├── architect
├── reviewer
├── investigator
├── security-reviewer
└── scout

task-orchestrator
├── general
├── explore
├── verifier
├── reviewer
├── investigator
├── security-reviewer
└── scout

leaf agents
└── no further delegation
```

The call graph is intentionally non-cyclic. `task-orchestrator` may never invoke another `task-orchestrator`.

## Permission boundary

Automatically permitted operations are restricted to repository inspection, selected read-only Git/GitHub commands, safe initialization and `project::*` checks, Task-local commit, and constrained PR create/edit/ready operations through Just.

User approval is required for Task branch push, final merge, cleanup, `/tmp/opencode/**` access, and unclassified shell commands.

Raw Git/GitHub mutations, force push, amend, rebase, destructive reset/clean, direct default-branch push, admin merge, privilege escalation, and destructive store/filesystem operations are prohibited.

## External paths

The only generally requestable external path is `/tmp/opencode/**`, and it requires approval. Other paths outside the current OpenCode workspace are denied by default.

## Worktree isolation

Agents must operate only inside the worktree assigned to their current Task. Access to sibling Task worktrees is prohibited. Static OpenCode permissions provide the default boundary; Task/worktree lifecycle guards add the dynamic sibling-worktree checks.
<!-- END AGENT CORE RULES -->
