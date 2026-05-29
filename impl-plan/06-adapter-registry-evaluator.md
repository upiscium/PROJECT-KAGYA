# 06 Adapter Registry And Evaluator

## Goal

Implement safe adapter lifecycle management and score-based evaluation gates.

## Target Files

- `kagya/learning/__init__.py`
- `kagya/learning/adapter_registry.py`
- `kagya/learning/adapter_evaluator.py`
- `kagya/learning/eval_sets.py`
- `tests/test_adapter_registry.py`
- `tests/test_adapter_evaluator.py`

## Adapter Status Requirements

- Define statuses: `candidate`, `trial_active`, `approved`, `active`, `rejected`, `archived`.
- Register newly trained adapters as `candidate` only.
- Promote from `candidate` to `trial_active` only after evaluation score is at least `trial_threshold`.
- Keep adapters as `candidate` when score is between reject and trial thresholds.
- Mark adapters as `rejected` when score is below `reject_threshold`.
- Require manual approval before `approved`.
- Allow `active` only from `approved`.
- Archive old active adapter when a new adapter becomes active.
- Never delete old adapters from the registry.

## Registry Requirements

- Store registry data in configured JSON path.
- Persist adapter ID, base model, adapter path, status, dataset path, dataset hash, eval score, eval result path, timestamps, and notes.
- Reject invalid status transitions.
- Provide list and lookup operations.

## Evaluator Requirements

- Load configured eval sets.
- Evaluate adapter behavior using the configured Transformers-based model provider path or a deterministic test provider.
- Write evaluation result JSON.
- Return score and threshold decision.

## Test Requirements

- Candidate registration writes a registry entry.
- Candidate can become `trial_active` when score is high enough.
- Candidate below reject threshold becomes `rejected`.
- Mid-range score leaves adapter as `candidate`.
- Manual approval changes `trial_active` to `approved`.
- Activation changes `approved` to `active`.
- Existing active adapter becomes `archived` when a new adapter activates.
- Invalid transitions raise errors.

## Completion Criteria

- Adapter lifecycle tests pass without real model evaluation.
