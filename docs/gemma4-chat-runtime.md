# Gemma4 Chat Runtime

## Overview

This work makes `project-kagya-chat` usable with `google/gemma-4-E4B-it` in the Nix/direnv dev shell.
It fixes the runtime environment, prompt formatting, and reply cleanup so the CLI can chat reliably with Unsloth.

## Features

- `uv run project-kagya-chat` starts an interactive chat session
- Uses the instruction-tuned Gemma4 model by default
- Loads the model through Unsloth in 4-bit mode
- Supports multi-turn chat history
- Uses Gemma4-native turn markers for prompt formatting
- Trims model control-token noise from replies
- Includes regression tests for prompt rendering and output cleanup

## Architecture

### Environment Layer

- Nix dev shell provides `uv`, CUDA libraries, `openssl`, and related GPU runtime dependencies
- `TRITON_LIBCUDA_PATH` points Triton at the GPU driver library path

### Model Layer

- `project_kagya.chat` loads `google/gemma-4-E4B-it` through `unsloth.FastLanguageModel`
- The chat path is kept on Unsloth rather than falling back to `transformers`

### Prompt/Response Layer

- Prompts are built from system, user, and assistant turns
- Gemma4 turn tokens are used when available
- Only newly generated tokens are decoded
- Reply cleanup removes wrapper tokens and empty control output

## Notes

- Base checkpoints such as `google/gemma-4-E4B` are not accepted for chat
- The runtime depends on a working local GPU stack
