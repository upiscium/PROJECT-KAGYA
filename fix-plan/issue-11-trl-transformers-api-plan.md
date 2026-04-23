# Issue 11 Fix Plan

## Goal

Update `qlora_train.py` so it matches the installed `transformers` and `trl` APIs.

## Plan

1. Update `TrainingArguments` usage
   - Replace `evaluation_strategy` with `eval_strategy`.
   - Keep the current conditional behavior: enable eval only when a validation file is provided.

2. Update `SFTTrainer` usage
   - Replace `tokenizer=` with `processing_class=`.
   - Remove `dataset_text_field=` since the current TRL signature no longer accepts it.
   - Keep the dataset preformatting step that normalizes rows into a `text` column.

3. Update tests
   - Adjust `tests/test_qlora_train.py` to assert the new keyword names.
   - Add a regression check that the trainer wiring matches the installed TRL signature.

4. Verify
   - Run `uv run pytest tests/test_qlora_train.py -v`.
   - If that passes, run the broader repo checks.

## Notes

- The change should stay limited to the training CLI and its tests.
- No behavior change is intended beyond restoring compatibility with the installed libraries.
