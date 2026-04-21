# Gemma 4 E4B QLoRA Implementation Spec

## Purpose

Provide a CLI for fine-tuning `google/gemma-4-E4B` with QLoRA using Unsloth in 4-bit mode.

## Scope

- Train from a local JSON or JSONL dataset.
- Use Unsloth to load the base model in 4-bit mode by default.
- Apply LoRA adapters to the standard Gemma attention and MLP projection layers.
- Save the fine-tuned model and tokenizer to an output directory.

## Entry Point

- Script: `project-kagya-qlora`
- Module: `project_kagya.qlora_train:main`

## Inputs

Required:

- `--train-file`: path to training data file

Optional:

- `--validation-file`: path to validation data file
- `--model-name`: defaults to `google/gemma-4-E4B`
- `--dataset`: defaults to `json`
- `--output-dir`: defaults to `./outputs/gemma-4-e4b-qlora`
- `--max-seq-length`: defaults to `2048`
- `--load-in-4bit` / `--no-load-in-4bit`
- `--lora-rank`: defaults to `16`
- `--lora-alpha`: defaults to `32`
- `--lora-dropout`: defaults to `0.0`
- `--learning-rate`: defaults to `2e-4`
- `--num-train-epochs`: defaults to `1.0`
- `--per-device-train-batch-size`: defaults to `2`
- `--per-device-eval-batch-size`: defaults to `2`
- `--gradient-accumulation-steps`: defaults to `4`
- `--warmup-steps`: defaults to `50`
- `--logging-steps`: defaults to `10`
- `--eval-steps`: defaults to `200`
- `--save-steps`: defaults to `200`
- `--seed`: defaults to `42`

## Dataset Contract

- The dataset must contain a `text` field.
- Each record is treated as a single supervised fine-tuning example.
- The loader expects JSON or JSONL files that `datasets.load_dataset("json", ...)` can read.

Example:

```json
{"text": "...training text..."}
```

## Training Behavior

- Load base model through `unsloth.FastLanguageModel.from_pretrained`.
- Enable 4-bit quantization unless `--no-load-in-4bit` is passed.
- Wrap the model with LoRA adapters via `FastLanguageModel.get_peft_model`.
- Target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- Use `trl.SFTTrainer` with `dataset_text_field="text"`.
- Use `TrainingArguments` with `optim="paged_adamw_8bit"` and `fp16=True`.

## Validation and Saving

- If `--validation-file` is supplied, the trainer uses evaluation on steps.
- If validation is absent, evaluation is disabled.
- After training, the script saves both the model and tokenizer to `--output-dir`.

## Error Handling

- Missing train or validation files raise `FileNotFoundError` before model loading.

## Constraints

- This design assumes a CUDA-capable environment suitable for Unsloth and bitsandbytes.
- The current implementation is optimized for plain text SFT, not chat template preprocessing.
