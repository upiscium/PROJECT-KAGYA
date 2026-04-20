# AGENTS.md

## Source Of Truth
- Trust executable config over prose: `pyproject.toml`, `Justfile`, `.pre-commit-config.yaml`, and `src/project_kagya/*.py`.
- The README CLI example is stale; `project-kagya` currently runs `project_kagya.cli:main`, which always executes the embodied-emotion demo and ignores other args.

## Commands
- `just sync` to sync dependencies.
- `just lint`, `just typecheck`, `just test`, `just check-all` for verification.
- `just lint` uses Ruff check, `just typecheck` uses Mypy, `just test` uses Pytest.
- Single-target runs are supported: `just lint src/`, `just typecheck src/`, `just test tests/test_name.py`.

## Git Hooks
- Pre-commit runs Ruff check and Ruff format check on Python files at commit time.
- Pre-push runs `just test`; do not assume a push is clean until tests pass.

## Runtime And Training
- The real runtime entrypoint is `project_kagya.cli:main`.
- Sleep consolidation writes JSONL output to the path configured in `settings.toml` (`sleep.dream_dataset_path`).
- `sleep_consolidation_training.py` expects JSONL rows with `input`, `thought`, `output`, `source_ids`, `confidence`, and `status`; invalid rows are skipped and empty datasets exit cleanly.
- `qlora_training.py` is a thin trainer wrapper; real LoRA/QLoRA behavior depends on an injected backend.

## Worktree Rules
- Do not overwrite unrelated user changes.
- Use `apply_patch` for file edits.
