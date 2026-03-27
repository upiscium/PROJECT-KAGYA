# AGENTS.md

This repository is a small Python project managed with `uv`, `just`, Nix, Ruff, Mypy, and Pytest.
Use this file as the operating guide for agentic changes in this repo.

## Repository Snapshot

- Python target: `>=3.11`
- Dependency manager: `uv`
- Task runner: `just`
- Linter/formatter: `ruff`
- Type checker: `mypy`
- Test runner: `pytest`
- Dev shell: Nix flake (`flake.nix`)
- Virtual environment: `.venv` in the repo root

## Rules From Repo Config

- No Cursor rules were found in `.cursor/rules/` or `.cursorrules`.
- No Copilot instructions were found in `.github/copilot-instructions.md`.
- `just` is the canonical entry point for common checks.
- Pre-commit hooks run Ruff checks and formatting on commit.
- Pre-push hooks run the test suite.
- The Nix shell sets `UV_PROJECT_ENVIRONMENT=$PWD/.venv`.
- The Nix shell also sets `PIP_REQUIRE_VIRTUALENV=1`.
- Do not rely on a global Python environment.
- Prefer repo-local tooling over system-wide installs.

## Setup

- Enter the dev shell with `nix develop` when available.
- Sync dependencies with `uv sync`.
- Activate the virtual environment only if needed manually.
- Assume commands should run from the repository root.

## Core Commands

- Sync dependencies: `just sync`
- Lint: `just lint`
- Format: `just format`
- Type check: `just typecheck`
- Test suite: `just test`
- Full verification: `just check-all`
- Ruff only: `uv run ruff check .`
- Ruff format only: `uv run ruff format .`
- Mypy only: `uv run mypy src/`
- Pytest only: `uv run pytest tests/ -v`

## Single File / Single Test Commands

- Single test file: `just test tests/test_example.py`
- Single test node: `uv run pytest tests/test_example.py::test_name -v`
- Single test with filter: `uv run pytest tests/ -k test_name -v`
- Single module lint: `just lint src/module.py`
- Single module format: `just format src/module.py`
- Single module type check: `just typecheck src/module.py`
- When using `just test`, the target is passed through to `pytest`.
- Keep the selector as narrow as possible for fast feedback.

## Preferred Workflow

- Inspect the relevant files before editing.
- Make the smallest change that fixes the issue.
- Run the narrowest useful command first.
- Expand to `just check-all` before handing off.
- Re-run the affected command after each fix.
- Favor deterministic, local checks over manual inspection.

## Project Layout Expectations

- Source code should live under `src/` unless the repo already uses another layout.
- Tests should live under `tests/`.
- Keep test data close to the tests that use it.
- Do not add new top-level folders without a clear reason.
- Respect existing naming if the repository already diverges.

## Python Style

- Target Python 3.11+ syntax and features.
- Prefer explicit type hints on public functions.
- Keep annotations accurate and complete.
- Use `from __future__ import annotations` when it improves typing ergonomics.
- Avoid mutable default arguments.
- Prefer early returns over deeply nested logic.
- Keep functions small and single-purpose.
- Prefer pure helpers where practical.
- Use dataclasses for simple data containers.
- Use enums for closed sets of values.

## Imports

- Group imports as standard library, third-party, then local.
- Keep imports sorted and unused imports removed.
- Prefer absolute imports within the project.
- Avoid wildcard imports.
- Avoid importing inside functions unless it solves a real problem.
- Do not create circular imports; refactor shared code instead.

## Formatting

- Follow Ruff formatting for all Python code.
- Keep lines readable and let the formatter handle wrapping.
- Use consistent quote style as enforced by Ruff.
- Do not hand-format code that Ruff will normalize.
- Prefer trailing commas in multiline collections and calls.
- Keep blank lines purposeful and minimal.

## Naming Conventions

- Modules and packages: `snake_case`.
- Functions and variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: leading underscore when appropriate.
- Test functions: `test_*`.
- Test fixtures: descriptive `snake_case` names.

## Types

- Add types to new public APIs.
- Keep type narrowing explicit and readable.
- Use `Any` only when necessary and justify it locally.
- Prefer `Optional[T]` or `T | None` when a value may be absent.
- Use `list[T]`, `dict[K, V]`, and modern built-in generics.
- Avoid overly clever type tricks unless they simplify the code.
- Fix mypy issues at the source rather than silencing them.

## Error Handling

- Raise specific exceptions.
- Do not swallow exceptions silently.
- Add context when re-raising if it helps debugging.
- Use `try`/`except` only around the smallest relevant block.
- Prefer validation up front for invalid inputs.
- Return structured errors only when the API needs them.
- Reserve broad exception handling for top-level boundaries.

## Testing

- Write tests for behavioral changes.
- Prefer one focused assertion per behavior.
- Test public behavior, not implementation details.
- Use parameterization for repeated cases.
- Keep tests deterministic and isolated.
- Avoid network and time dependence unless explicitly required.
- If fixing a bug, add a regression test first or alongside the fix.
- Run the narrow test first, then the suite if needed.

## Ruff / Mypy Expectations

- Fix Ruff warnings rather than ignoring them.
- Use `# noqa` sparingly and only with a clear reason.
- Use `# type: ignore` only as a last resort.
- Prefer code changes over linter suppression.
- Keep suppressions as local and narrow as possible.

## Change Discipline

- Do not edit generated artifacts unless the task requires it.
- Avoid touching `.venv/`, `.direnv/`, and other ignored build outputs.
- Do not introduce unrelated refactors in the same change.
- Preserve existing public APIs unless the request says otherwise.
- Update docs or tests when behavior changes.
- Match the style of nearby code when no project-wide pattern exists.

## When You Are Unsure

- Inspect `Justfile`, `pyproject.toml`, and `.pre-commit-config.yaml` first.
- Prefer the conventions already used in the repository.
- If the repo is still sparse, keep decisions simple and conventional.
- Choose changes that make future validation easier.
- Leave the codebase in a state where `just check-all` is the best next step.
