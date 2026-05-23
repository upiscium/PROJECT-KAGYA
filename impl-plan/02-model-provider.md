# 02 Model Provider

## Goal

Define the model provider interface, implement a deterministic `DummyProvider`, and implement a Transformers-backed provider without using external LLM APIs.

## Target Files

- `kagya/models/__init__.py`
- `kagya/models/base.py`
- `kagya/models/model_loader.py`
- `kagya/models/dummy_provider.py`
- `kagya/models/transformers_provider.py`

## Implementation Requirements

- Define `ModelProvider` as a `Protocol` with `generate`, `calculate_loss`, `get_model`, and `get_processor`.
- Implement `DummyProvider` with fixed response and fixed loss values for fast tests.
- Implement `TransformersProvider` using `AutoProcessor.from_pretrained` and `AutoModelForImageTextToText.from_pretrained`.
- Support 4-bit NF4 quantization from configuration.
- Support LoRA adapter attachment through registry-approved adapter paths only.
- Keep generation parameters configurable.
- Do not hard-code model IDs.
- Do not create `OllamaProvider` or any external API provider.

## Loss Calculation Requirements

- Tokenize `context_text + target_text`.
- Create labels with the same shape as `input_ids`.
- Mask context tokens with `-100` so only target tokens contribute to cross entropy.
- Use `torch.no_grad()`.
- Call `model.eval()` before loss calculation.
- Return a Python `float`.
- Raise `ValueError` when `target_text` is empty.

## Test Requirements

- `DummyProvider.generate` returns deterministic text.
- `DummyProvider.calculate_loss` returns deterministic float.
- `TransformersProvider.calculate_loss` rejects empty target text.
- Loss masking can be unit-tested with a fake tokenizer/model so long context tokens are excluded from labels.
- Provider loading uses configuration model IDs.

## Completion Criteria

- Main loop and API can run entirely with `DummyProvider`.
- Transformers provider is implemented but not required for default fast tests.
