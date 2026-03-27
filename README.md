# PROJECT-KAGYA

PROJECT-KAGYA is a Python prototype for a subjective AI architecture.
It includes emotion state updates, dual-memory retrieval, a conscious agent,
sleep consolidation helpers, and a command-line interface.

## Requirements

- Python 3.11+
- `uv`
- `nix` (optional, for `nix develop`)

## Setup

```bash
just sync
```

Or enter the dev shell first:

```bash
nix develop
just sync
```

## Run the CLI

The CLI is exposed as `project-kagya`:

```bash
uv run project-kagya --input "hello"
```

Presets:

- `--preset lightweight` (default): `Qwen/Qwen2.5-1.5B-Instruct`
- `--preset qwen3.5-9b`: `Qwen/Qwen3.5-9B-Instruct`

Useful options:

- `--preset`: model preset to load
- `--model-name`: base model name or path
- `--adapter-path`: optional LoRA adapter directory
- `--input`: user message to process
- `--valence`: current valence value
- `--arousal`: current arousal value

Example:

```bash
uv run project-kagya \
  --input "How are you?" \
  --valence 0.2 \
  --arousal 0.7
```

## Tests and Checks

- `just test`
- `just test tests/test_cli.py::test_run_invokes_runtime`
- `just lint`
- `just typecheck`
- `just check-all`

## Project Layout

- `src/project_kagya/cli.py`: CLI entrypoint
- `src/project_kagya/main.py`: runtime loading and chat flow
- `src/project_kagya/dual_memory_system.py`: episodic and semantic memory
- `src/project_kagya/conscious_agent.py`: prompt construction and generation
- `src/project_kagya/sleep_consolidation.py`: sleep-cycle consolidation helpers
- `tests/`: automated tests for each module
