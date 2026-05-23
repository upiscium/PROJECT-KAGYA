# 07 Sleep Cycle Dream Dataset And QLoRA

## Goal

Implement sleep-time consolidation, dream dataset generation, and QLoRA training entry points while keeping adapter activation gated by the registry.

## Target Files

- `kagya/learning/sleep_consolidation.py`
- `kagya/learning/dream_dataset_generator.py`
- `kagya/learning/qlora_trainer.py`

## Sleep Cycle Requirements

- Implement `SleepCycleManager.run() -> SleepCycleResult`.
- Extract high-emotion DB1 episodes where `arousal > 0.7` or `abs(valence) > 0.6`.
- Generate semantic memory candidates.
- Save semantic memories into DB2.
- Generate dream dataset JSONL.
- Run QLoRA in dry-run or minimal-run mode when configured for tests.
- Register resulting adapter as `candidate`.
- Do not activate the adapter.

## Dream Dataset Requirements

- JSONL records use `{"input": "str", "thought": "str", "output": "str"}`.
- Training text format is:

```text
ユーザー: {input}
私: <think>
{thought}
</think>
{output}<eos>
```

- Preserve hidden thoughts only for training/debug paths, not normal API output.

## QLoRA Requirements

- Use configured QLoRA params: `r`, `lora_alpha`, `lora_dropout`, `learning_rate`, `num_train_epochs`, NF4 quantization, and bfloat16 compute dtype.
- Use Transformers, PEFT, and TRL stack.
- Support dry-run that validates dataset and returns a candidate adapter path without expensive training.
- Register adapter through `AdapterRegistry` only.

## Test Requirements

- High-emotion episode selection follows threshold rules.
- Dream dataset JSONL is generated with expected fields.
- Dataset records include `<think>` only in training format, not normal response paths.
- QLoRA dry-run returns an adapter candidate result.
- Sleep cycle registers adapter as `candidate`.
- Sleep cycle never creates `active` adapters.

## Completion Criteria

- Sleep cycle can run in test mode without GPU or real model load.
