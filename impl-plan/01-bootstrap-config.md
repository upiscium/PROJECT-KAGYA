# 01 Bootstrap And Configuration

## Goal

Create the minimal Python project foundation and configuration layer required by all later phases.

## Target Files

- `pyproject.toml`
- `config.yaml`
- `.env.example`
- `kagya/__init__.py`
- `kagya/config/__init__.py`
- `kagya/config/schema.py`
- `kagya/config/settings.py`
- `kagya/api/server.py`

## Implementation Requirements

- Use `uv` as the Python project manager.
- Add runtime dependencies: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `peft`, `trl`, `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `pyyaml`, `python-multipart`, `numpy`, `rich`.
- Add test dependency: `pytest`.
- Create `config.yaml` with all sections required by the specification: `project`, `model`, `generation`, `emotion`, `memory`, `sleep`, `qlora`, `adapter_registry`, `api`, and `frontend`.
- Implement typed configuration loading from `config.yaml`.
- Preserve model IDs in configuration only.
- Provide an importable FastAPI app and a `python -m kagya.api.server` startup path.

## Test Requirements

- Verify `config.yaml` can be loaded into typed settings.
- Verify primary and fallback model IDs come from configuration.
- Verify API host, port, and CORS origins come from configuration.
- Verify `uv run pytest` can execute successfully even before feature modules are complete.

## Completion Criteria

- `uv run pytest` runs.
- `uv run python -m kagya.api.server` has a valid startup foundation.
- No Ollama or external LLM dependency is present.
