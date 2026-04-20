# PROJECT-KAGYA

PROJECT-KAGYA is a Python prototype for a subjective AI architecture.
It includes emotion state updates, dual-memory retrieval, a conscious agent,
sleep consolidation helpers, and a command-line interface.

## Setup

```bash
just sync
```

Or enter the dev shell first:

```bash
nix develop
just sync
```

## Quick Start

Configure `settings.toml`, then run the CLI:

```bash
uv run project-kagya
```

The CLI reads runtime settings from `settings.toml`. With no flags it runs the
embodied-emotion demo. Pass `--settings` to point at another config file.

## Launch Modes

```bash
uv run project-kagya --demo
uv run project-kagya --serve
uv run project-kagya --consolidate
uv run project-kagya --train
uv run project-kagya --pipeline full
```

`--serve` starts the FastAPI ingest app, `--consolidate` writes the sleep JSONL,
`--train` runs the JSONL training stage, and `--pipeline full` runs the whole
integration flow.

## Settings

- `settings.toml` is the single source of truth for runtime configuration
- `runtime`: input text, backend, initial valence/arousal
- `model`: base model name, adapter path, 4-bit loading
- `memory`: retrieval settings
- `emotion`: allostasis parameters
- `sleep`: sleep-cycle thresholds and dataset output path

Example run:

```bash
uv run project-kagya --settings settings.toml
```

## Implemented Modules

- `src/project_kagya/emotion_engine.py`: valence/arousal update rules
- `src/project_kagya/embodied_emotion.py`: body-state driven emotion modulation
- `src/project_kagya/surprisal_calculator.py`: masked loss calculation
- `src/project_kagya/dual_memory_system.py`: episodic and semantic memory
- `src/project_kagya/conscious_agent.py`: prompt construction and generation
- `src/project_kagya/multimodal_fastapi_interface.py`: multimodal ingest API
- `src/project_kagya/sleep_consolidation.py`: sleep-cycle consolidation helpers
- `src/project_kagya/sleep_consolidation_training.py`: JSONL training pipeline
- `src/project_kagya/qlora_training.py`: QLoRA training wrapper
- `src/project_kagya/runtime.py`: settings-driven orchestration
- `src/project_kagya/cli.py`: CLI entrypoint

## Tests and Checks

- `just test`
- `just lint`
- `just typecheck`
- `just check-all`

## Notes

- Large-model loading and QLoRA training are still backend-dependent.
- The current tests use dummy tokenizers and models to keep the suite fast.
