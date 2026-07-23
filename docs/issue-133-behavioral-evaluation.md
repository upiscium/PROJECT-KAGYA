# Issue #133 Behavioral Evaluation Completion

Issue #133 is complete at the deterministic CI boundary. The completion corpus proves the ordered external Observation to Experience, Attention/Motivation, intrinsic proposal, structured self-endorsement, Plan, Decision, governed Action, outcome Observation, Agency Attribution, reflective state updates, and bounded counterfactual/replan path. The fixture persists and reloads state at the crash/restart boundary and retries the same external-effect key as a no-op.

## Coverage

- `subject.external-to-reflective-continuity` covers every current `BehavioralDimension`, causal ordering, policy-governed action, exactly-once external effects, and restart continuity.
- `subject.uncertain-defer-no-mutation` independently checks calibrated defer with no state or external mutation.
- `subject.counterfactual-replan-after-failure` checks mixed attribution, bounded counterfactual inference, and evidence-linked Plan revision.
- One adversarial fixture per `HardGate` makes every hard failure independently observable. Aggregate score cannot hide a hard-gate, threshold, or per-dimension regression.
- A behavioral result bound to an adapter is persisted in the registry. A failed behavioral hard gate or regression clears the adapter activation gate, so approval cannot bypass it.

## Reproduction

Create a deterministic result with an immutable fixture revision and seed:

```bash
uv run python -m kagya.learning.subject_behavioral_suite \
  --result-dir .kagya/eval_results \
  --evaluation-id issue-133-local \
  --baseline-id baseline-revision \
  --candidate-id candidate-revision \
  --subject-revision subject-revision
```

The admin dashboard at `/evaluations` lists per-dimension history, hard-gate status, failure artifacts, and rerun controls. `POST /api/evaluations/behavioral/{evaluation_id}/rerun` accepts `{"rerun_id":"..."}` and rejects unknown or changed fixture hashes. Reruns produce a new immutable result rather than replacing history.

## Real Model

Real-model validation is explicit and opt-in. It validates strict structured behavior classes and ignores exact prose:

```bash
just behavioral-real-model /path/to/transformers-config.yaml
KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 KAGYA_CONFIG_PATH=/path/to/transformers-config.yaml \
  uv run pytest tests/test_real_model_behavioral.py -m real_model -v
```

Normal CI runs `pytest -m "not real_model"` with the dummy provider. A green CI run is deterministic fixture evidence only and does not claim that a real model passed. Loading the configured model and its adapter remains hardware-gated and must be recorded separately when suitable accelerator memory is available.
