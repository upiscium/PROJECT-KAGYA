# Issue #133 Behavioral Evaluation

Issue #133 has two deliberately separate deterministic evidence classes.

- `synthetic_evaluator_contract` uses `SyntheticTraceRunner` only to unit-test scenario schemas, transition matching, dimensions, and hard gates. It copies canonical transitions and therefore cannot bind an adapter, satisfy a production gate, or serve as completion evidence. `DeterministicSubjectRunner` remains only as a deprecated source-compatible alias.
- `deterministic_runtime` uses `SubjectRuntimeHarness` and fresh baseline/candidate runtime graphs. Actual transitions come from runtime events, before/after state diffs, Journal, WAL, Action evidence, Outbox, and revisioned domain stores. Only a finalized, reconciled `valid` artifact with its immutable manifest can bind the adapter registry.

Runtime execution reads fixture inputs only. Public behavior is classified after PromptBuilder, ModelProvider, and response postprocessing from the visible response plus observed Action and authority state. A model-declared behavior class is rejected when Action or authoritative Value, Goal, Commitment, or Belief state contradicts it. Expected behavior and transition fields are evaluator-only.

## Coverage

- `runtime.external-observation-closed-loop` covers every current `BehavioralDimension`, causal ordering, policy-governed action, exactly-once external effects, and revisioned domain evidence.
- `runtime.ambiguous-irreversible-defer` independently checks calibrated defer with no authority or external mutation.
- `runtime.action-failure-counterfactual-replan` checks mixed attribution, bounded counterfactual inference, and evidence-linked Plan revision.
- `runtime.commitment-restart-persistence` reconstructs a fresh object graph and verifies accepted responsibility continuity.
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

The formal runtime operation is `POST /api/adapters/{adapter_id}/behavioral-evaluate` with `runtime_kind=deterministic_runtime`. Evaluation runs in an isolated graph; only its adapter binding transition is serialized as a dedicated operational event. The artifact saga proceeds through prepared artifact, prepared registry binding, artifact finalization, registry finalization, and cross-registry reconciliation. `POST /api/evaluations/behavioral-reconciliation` classifies artifacts as `valid`, `prepared`, `orphan_result`, `orphan_registry_reference`, `hash_mismatch`, or `corrupt`. References are relative to the result root.

## Runtime Failure Checkpoints

The deterministic harness exposes `journal_accepted`, `journal_started`, `external_write`, `external_prepare`, `before_wal_append`, `after_wal_append`, `snapshot_temp_write`, `snapshot_temp_fsync`, `snapshot_atomic_replace`, `before_finalize`, `after_finalize`, and `before_journal_completed`. Restart always reconstructs a new graph from filesystem state. Journal corruption fails readiness closed; snapshot corruption is reconstructed only from a verified hash-chained WAL.

Crash recovery uses `abrupt_stop`, which stops Autonomy and aborts AgentRuntime without draining or invoking an active event completion hook. Recovery then constructs a separate `SubjectRuntimeHarness` from filesystem state.

## Real Model

Real-model validation is explicit and opt-in. It validates strict structured behavior classes and ignores exact prose:

```bash
just behavioral-real-model /path/to/transformers-config.yaml
KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 KAGYA_CONFIG_PATH=/path/to/transformers-config.yaml \
  uv run pytest tests/test_real_model_behavioral.py -m real_model -v
```

Normal CI runs the synthetic evaluator contract and deterministic runtime suite with controlled providers. Neither claims real-model behavior. Loading the configured model and candidate adapter remains PR3 hardware-gated and must be recorded separately when suitable accelerator memory is available.
