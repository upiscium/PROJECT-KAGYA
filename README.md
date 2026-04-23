# QLoRA fine-tuning with Unsloth

Train `google/gemma-4-E4B` in 4-bit mode with Unsloth:

```bash
uv run project-kagya-qlora \
  --train-file data/train.jsonl \
  --validation-file data/valid.jsonl \
  --output-dir outputs/gemma-4-e4b-qlora
```

Expected dataset format:

```json
{"text": "...training text..."}
```

If you want to disable 4-bit loading, pass `--no-load-in-4bit`.

## Chat with a model

```bash
uv run project-kagya-chat --model-name outputs/gemma-4-e4b-qlora
```

Use `/exit` to quit and `/reset` to clear conversation history.
