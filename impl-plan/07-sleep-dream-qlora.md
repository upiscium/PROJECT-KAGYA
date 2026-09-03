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

- JSONL records use only `{"input": "str", "output": "str"}` from visible episodic material.
- Training text format is:

```text
ユーザー: {input}
私: {output}<eos>
```

- Hidden/private model reasoning is not training material or training authority and must never be used as a target. Historical `thought` fields and `<think>` training targets are superseded by the R02 privacy contract in Issue #245.

## QLoRA Requirements

- Use configured QLoRA params: `r`, `lora_alpha`, `lora_dropout`, `learning_rate`, `num_train_epochs`, NF4 quantization, and bfloat16 compute dtype.
- Use Transformers, PEFT, and TRL stack.
- Support dry-run that validates dataset and returns a candidate adapter path without expensive training.
- Reject legacy or externally supplied dataset records containing `thought`, hidden/private-reasoning fields, or `<think>` training targets rather than silently training on them.
- Register adapter through `AdapterRegistry` only.

## Test Requirements

- High-emotion episode selection follows threshold rules.
- Dream dataset JSONL contains only visible `input` and `output` fields.
- Dataset records and formatted training text contain no private reasoning field or `<think>` target.
- QLoRA dataset loading rejects the superseded private-data format.
- QLoRA dry-run returns an adapter candidate result.
- Sleep cycle registers adapter as `candidate`.
- Sleep cycle never creates `active` adapters.

## Completion Criteria

- Sleep cycle can run in test mode without GPU or real model load.
- R03 and later persistence/runtime work cannot reintroduce private reasoning into dream or QLoRA training paths.
