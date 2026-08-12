# PROJECT-KAGYA Agent Guide

## Repository shape

- Backend is the Python package in `kagya/`; there is no `src/` package tree.
- Backend tests live in `tests/`.
- Frontend is the separate Next.js application in `frontend/` and uses npm with `package-lock.json`.
- Runtime configuration is `config.yaml`.
- Runtime/generated state belongs under `.kagya/` and is not source code.

## Stable validation entrypoints

Run backend validation from the repository root:

- `just lint`
- `just typecheck`
- `just test`

Run frontend validation through the root Just API:

- `just frontend-test`
- `just frontend-build`

`just check-all` is the repository-wide deterministic validation gate for the code currently present on the rebuild branch. Later rebuild PRs may extend this gate only after the corresponding implementation and tests exist.

## Rebuild lane

The responsibility-oriented rebuild is tracked by GitHub issue #242.

- Integration branch: `rebuild/develop`.
- Each rebuild task uses a dedicated `rebuild/NN-*` branch and targets `rebuild/develop`.
- Use a worktree under `.worktrees/` for rebuild implementation. Do not switch or rewrite the user's primary working tree to perform rebuild work.
- Do not reset, rebase, force-update, or retarget `main`, `develop`, or pre-existing feature branches as part of rebuild work.
- Historical implementation order is not authoritative. Introduce the final intended boundary at the earliest dependency-safe point rather than deliberately replaying superseded designs.

## PR responsibility contract

Every rebuild PR must state:

- **Owns**: state, behavior, or contract for which this layer is responsible.
- **May**: operations this layer is permitted to perform.
- **Must not**: authority this layer must never acquire.
- **Depends on**: upstream contracts required for correctness.
- **Used by**: downstream consumers that may depend on this layer without taking over its authority.
- migration and compatibility impact;
- failure and recovery behavior;
- privacy impact;
- verification commands.

Tests passing is necessary but not sufficient for merge. Review must also establish which component is authoritative, what evidence permits mutation, what happens at crash boundaries, and which regression tests detect boundary violations.

## Scope discipline

- Keep each PR responsibility-oriented rather than file-oriented.
- Do not introduce validation commands for modules or tests that are planned for later rebuild PRs but do not yet exist.
- Do not hide failures with broad type-checker or linter ignores.
- Do not mix unrelated cleanup into a responsibility PR unless it is required to make that PR's declared contract truthful.
- Preserve public/private data boundaries already present at the current rebuild baseline; later PRs may strengthen them in their planned scope.

## Baseline application commands

- Sync backend dependencies: `uv sync --locked`.
- Start FastAPI: `just api`.
- Install frontend dependencies: `cd frontend && npm ci`.

Real-model or hardware-dependent checks are not part of the R01 deterministic baseline. They are introduced only by later rebuild layers that own those contracts.
