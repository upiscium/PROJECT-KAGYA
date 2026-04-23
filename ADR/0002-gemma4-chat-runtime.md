# ADR 0002: Gemma4 chat runtime for Unsloth-based CLI

## Status

Accepted

## Context

The `project-kagya-chat` CLI needed to run reliably with `google/gemma-4-E4B-it` inside the Nix/direnv dev shell. The initial setup exposed missing runtime tools, incompatible prompt formatting, and model-output leakage.

## Capability

This work enables a functional local chat runtime with the following capabilities:

- Launch Gemma4 chat from the CLI through `uv run project-kagya-chat`
- Load the instruction-tuned Gemma4 model with Unsloth in the dev shell
- Format prompts using Gemma4 turn tokens instead of generic `User:/Assistant:` text
- Strip control-token noise from model replies before displaying them
- Keep the environment self-contained through Nix and `direnv`

## Architecture

The runtime is split into three layers:

1. **Environment layer**
   - Nix dev shell provides `uv`, CUDA libraries, `openssl`, and Triton-compatible CUDA library paths.
   - `TRITON_LIBCUDA_PATH` points Triton at the host GPU driver libraries.

2. **Model loading layer**
   - `project_kagya.chat` loads `google/gemma-4-E4B-it` through `unsloth.FastLanguageModel`.
   - The CLI keeps the chat path on Unsloth only.

3. **Prompt and response layer**
   - Prompts use Gemma4 turn markers when available.
   - The input tensor length is tracked so only newly generated tokens are decoded.
   - Reply cleanup removes control-token leakage and empty wrapper output.

## Decision

Use the Gemma4 instruction-tuned model with a Gemma-native prompt format and a Nix-backed GPU runtime environment.

## Rationale

- Gemma4 `-it` variants are intended for chat.
- Native turn tokens match the checkpoint better than a handwritten prompt transcript.
- The dev shell must own the CUDA and compiler-tooling compatibility surface.
- Unsloth provides the desired inference path for this repository.

## Consequences

- Base checkpoints such as `google/gemma-4-E4B` are rejected for chat.
- The dev shell now depends on host GPU driver availability.
- Chat output is safer to display, but still model-generated and not schema-validated.

## Verification

- `pytest`
- `uv run project-kagya-chat`
