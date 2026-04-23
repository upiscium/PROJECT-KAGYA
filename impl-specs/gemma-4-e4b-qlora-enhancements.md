# Gemma 4 E4B QLoRA Enhancements

## Purpose

Describe the next-step design for improving the initial Gemma QLoRA training flow.

## Covered Items

1. Alpaca-style and chat-style preprocessing.
2. Flexible dataset and text field selection.
3. Training hyperparameter tuning for Gemma.

## 1. Preprocessing Design

### Goal

Support input data that is not already in a single `text` field.

### Supported Schemas

- Alpaca-style records with fields such as `instruction`, `input`, and `output`.
- Chat-style records with a message list, such as `messages` or `conversations`.
- Plain text records with a `text` field.

### Conversion Rules

- Alpaca-style records are converted into a single prompt-response text block.
- Chat-style records are rendered into a linear conversation transcript.
- Plain text records are passed through unchanged.

### Expected Behavior

- The user selects a preprocessing mode explicitly or the loader infers it from the dataset fields.
- If inference fails, the command should fail fast with a clear error.

## 2. Dataset Flexibility Design

### Goal

Allow the training command to work with more datasets without editing code.

### Proposed Interface

- `--dataset` selects the Hugging Face dataset loader name.
- `--text-field` overrides the source field used for plain-text SFT.
- `--instruction-field`, `--input-field`, `--output-field` support Alpaca-style mapping.
- `--messages-field` supports chat-style mapping.

### Behavior

- If `--text-field` is provided, the loader extracts that field directly.
- If a structured format is selected, the script constructs `text` internally.
- Validation data uses the same mapping as training data.

### Error Handling

- Missing configured fields should raise a validation error before training starts.
- Conflicting schema arguments should be rejected.

## 3. Gemma Training Tuning Design

### Goal

Tune defaults so the command is closer to a practical Gemma fine-tuning recipe.

### Suggested Defaults

- Increase the default LoRA rank only if memory budget allows it.
- Prefer a sequence length aligned with the actual dataset rather than a fixed large maximum.
- Keep 4-bit loading enabled by default.
- Keep checkpointing enabled to reduce memory pressure.

### Hyperparameters Worth Exposing

- `--learning-rate`
- `--max-seq-length`
- `--lora-rank`
- `--lora-alpha`
- `--lora-dropout`
- `--gradient-accumulation-steps`
- `--per-device-train-batch-size`
- `--warmup-steps`

### Recommended Constraint

- Prefer conservative defaults for large-model safety, while allowing overrides for higher-throughput runs.

## Non-Goals

- Automatic dataset schema migration.
- Multi-task training orchestration.
- Distributed training topology changes.

## Output

These enhancements should result in a more general-purpose CLI that still defaults to the original plain-text QLoRA path.
