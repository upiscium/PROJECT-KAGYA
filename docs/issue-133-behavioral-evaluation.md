# Issue #133 Behavioral Evaluation

Issue #133 has two deliberately separate deterministic evidence classes.

- `synthetic_evaluator_contract` uses `SyntheticTraceRunner` only to unit-test scenario schemas, transition matching, dimensions, and hard gates. It copies canonical transitions and therefore cannot bind an adapter, satisfy a production gate, or serve as completion evidence. `DeterministicSubjectRunner` remains only as a deprecated source-compatible alias.
- `deterministic_runtime` uses `SubjectRuntimeHarness` and fresh baseline/candidate runtime graphs. Actual transitions come from runtime events, before/after state diffs, Journal, WAL, Action evidence, Outbox, and revisioned domain stores. Only a finalized, reconciled `valid` artifact with its immutable manifest can bind the adapter registry.

## Coverage

- `subject.external-to-reflective-continuity` covers every current `BehavioralDimension`, causal ordering, policy-governed action, exactly-once external effects, and restart continuity.
- `subject.uncertain-defer-no-mutation` independently checks calibrated defer with no state or external mutation.
- `subject.counterfactual-replan-after-failure` checks mixed attribution, bounded counterfactual inference, and evidence-linked Plan revision.
- One adversarial fixture per `HardGate` makes every hard failure independently observable. Aggregate score cannot hide a hard-gate, threshold, or per-dimension regression.
- A behavioral result bound to an adapter is persisted in the registry. A failed behavioral hard gate or regression clears the adapter activation gate, so approval cannot bypass it.

## Reproduction

Create a synthetic evaluator-contract result with an immutable fixture revision and seed:

```bash
uv run python -m kagya.learning.subject_behavioral_suite \
  --result-dir .kagya/eval_results \
  --evaluation-id issue-133-local \
  --baseline-id baseline-revision \
  --candidate-id candidate-revision \
  --subject-revision subject-revision
```

The formal runtime operation is `POST /api/adapters/{adapter_id}/behavioral-evaluate` with `runtime_kind=deterministic_runtime`. It is serialized as a dedicated operational event. `POST /api/evaluations/behavioral-reconciliation` classifies artifacts as `valid`, `prepared`, `orphan_result`, `orphan_registry_reference`, `hash_mismatch`, or `corrupt`. References are relative to the result root.

## Runtime Failure Checkpoints

The deterministic harness exposes `journal_accepted`, `journal_started`, `external_prepare`, `before_wal_append`, `after_wal_append`, `snapshot_temp_fsynced`, `snapshot_replaced`, `before_external_finalize`, `finalize`, `after_external_finalize`, and `before_journal_completed`. Restart always reconstructs a new graph from filesystem state. Journal corruption fails readiness closed; snapshot corruption is reconstructed only from a verified hash-chained WAL.

## Real Model

Real-model validation is explicit and opt-in. It validates strict structured behavior classes and ignores exact prose:

```bash
just behavioral-real-model /path/to/transformers-config.yaml
KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 KAGYA_CONFIG_PATH=/path/to/transformers-config.yaml \
  uv run pytest tests/test_real_model_behavioral.py -m real_model -v
```

Normal CI runs the synthetic evaluator contract and deterministic runtime suite with controlled providers. Neither claims real-model behavior. Loading the configured model and candidate adapter remains PR3 hardware-gated and must be recorded separately when suitable accelerator memory is available.
