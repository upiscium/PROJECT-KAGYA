# PROJECT-KAGYA Feature Map

## What the repo does

`PROJECT-KAGYA` provides two CLI entry points for Gemma 4 workflows:

- `project-kagya-qlora`: fine-tune `google/gemma-4-E4B` with QLoRA via Unsloth
- `project-kagya-chat`: run an interactive Gemma chat session from the terminal

There is also a minimal `project-kagya` entrypoint that points users to the training CLI.

## File Map

### `src/project_kagya/cli.py`

- Minimal package entrypoint
- Prints a short hint telling users to use `project-kagya-qlora --help`

### `src/project_kagya/qlora_train.py`

- Builds the training CLI parser
- Validates training and validation file paths
- Supports dataset schemas:
  - `plain`
  - `alpaca`
  - `chat`
  - `auto` inference
- Converts records into a single `text` field for SFT training
- Loads Gemma through Unsloth
- Applies LoRA adapters
- Runs training and saves the model plus tokenizer

### `src/project_kagya/chat.py`

- Builds the chat CLI parser
- Loads an instruction-tuned Gemma 4 model
- Supports optional adapter loading
- Supports 4-bit loading
- Builds prompts from system, user, and assistant turns
- Handles multimodal attachments:
  - images
  - audio
  - video
- Supports interactive commands:
  - `/exit` or `/quit`
  - `/reset`
  - `:attach PATH`
  - `:clear-attachments`
  - `:list-attachments`
- Trims prompt echo and control-token noise from model output

### `tests/test_qlora_train.py`

- Verifies plain, Alpaca, and chat record formatting
- Verifies schema inference and validation behavior
- Verifies the training wiring with fake dependencies

### `tests/test_chat_cli.py`

- Verifies message construction and prompt rendering
- Verifies attachment parsing and multimodal content assembly
- Verifies chat loop behavior for attachment commands and resets

## Current User Workflows

### Train a model

```bash
uv run project-kagya-qlora \
  --train-file data/train.jsonl \
  --validation-file data/valid.jsonl \
  --output-dir outputs/gemma-4-e4b-qlora
```

### Chat with a model

```bash
uv run project-kagya-chat --model-name outputs/gemma-4-e4b-qlora
```

### Use attachments in chat

```text
:attach image.png voice.wav
```

## Design Notes

- Training expects dataset rows to be normalized into `{ "text": "..." }`.
- Chat uses Gemma 4 instruction-tuned models by default.
- The repo favors small, testable CLI helpers over heavy shared abstractions.
