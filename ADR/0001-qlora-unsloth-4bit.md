# ADR 0001: Use Unsloth 4-bit QLoRA for Gemma 4 E4B fine-tuning

## Status

Accepted

## Context

We need a practical fine-tuning workflow for `google/gemma-4-E4B` that can run with limited VRAM.
The repository already depends on `transformers`, `peft`, and `bitsandbytes`, so the main decision is how to wire the training stack.

## Decision

Use Unsloth with 4-bit loading as the default path, and expose a CLI that trains QLoRA adapters on top of `google/gemma-4-E4B`.

## Rationale

- Unsloth simplifies low-VRAM fine-tuning and reduces boilerplate.
- 4-bit loading is the most memory-efficient default for this model class.
- `trl.SFTTrainer` provides a direct fit for text-based supervised fine-tuning.
- The target module list matches the standard Gemma projection layers and keeps the adapter scope focused.

## Consequences

- Users need a compatible GPU stack for `bitsandbytes` and Unsloth.
- The initial implementation assumes a `text` column and does not include chat-template preprocessing.
- Training configuration is CLI-driven, so users can tune LoRA rank, sequence length, batch size, and learning rate without editing code.

## Alternatives Considered

- Pure `transformers` + `peft`
  - Rejected because it requires more manual setup for the same QLoRA flow.
- Full precision fine-tuning
  - Rejected because it is less practical for large-model experimentation on limited hardware.
