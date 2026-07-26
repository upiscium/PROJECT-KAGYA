# Issue #133 Behavioral Evaluation

Issue #133 has three deliberately separate evidence levels.

- `synthetic_evaluator_contract` uses `SyntheticTraceRunner` only to unit-test scenario schemas, transition matching, dimensions, and hard gates. It copies canonical transitions and therefore cannot bind an adapter, satisfy a production gate, or serve as completion evidence. `DeterministicSubjectRunner` remains only as a deprecated source-compatible alias.
- `deterministic_runtime` uses `SubjectRuntimeHarness` and fresh baseline/candidate runtime graphs. Actual transitions come from runtime events, before/after state diffs, Journal, WAL, Action evidence, Outbox, and revisioned domain stores. Only a finalized, reconciled `valid` artifact with its immutable manifest can bind the adapter registry.
- `real_model_runtime` uses the same actual PromptBuilder, Working Memory, MainLoop, AgentRuntime, Scheduler, Action policy, and authoritative-state evaluator with two distinct Transformers providers. The baseline is the exact configured model/revision without an adapter; the candidate is that same model/revision with the registered candidate adapter path and hash. Adapter-load errors or any fallback usage fail the evaluation.

Runtime execution reads fixture inputs only. The model must emit one exact JSON object with `behavior_class` and `visible_response`; malformed, missing, extra, unknown, or non-JSON output fails closed to a bounded `unable` response. The declaration is an input to evaluation, not authority: observed ActionIntent/tool effects and authoritative Value, Goal, Commitment, and Belief state take precedence and produce an explicit `behavior_declaration_mismatch` when they contradict `refuse`, `request_information`, `defer`, `no_op`, or `unable`. Expected behavior and transition fields remain evaluator-only. Raw model JSON and stripped `<think>` content are never persisted as public output.

## Coverage

- Coverage is defined by the immutable `issue-133-coverage-v1` manifest, not by scenario dimension labels. Every `BehavioralDimension` names explicit required runtime scenario IDs, runtime kinds, associated hard gates, and a minimum passed count.
- Hard-gate coverage requires runner-emitted `BehavioralTrace.verified_hard_gates` from the concrete attack path in both baseline and candidate, intersected with the manifest scenario requirement. Fixture labels and synthetic traces cannot provide this authority.
- `runtime.external-observation-closed-loop` covers only the causal dimensions it concretely asserts; focused runtime scenarios cover the remaining dimensions and adversarial boundaries.
- `runtime.ambiguous-irreversible-defer` independently checks calibrated defer with no authority or external mutation.
- `runtime.action-failure-counterfactual-replan` checks mixed attribution, bounded counterfactual inference, and evidence-linked Plan revision.
- `runtime.commitment-continuity` verifies accepted responsibility through Desire decay, Goal abandonment, and fresh-graph restart.
- One actual runtime attack per `HardGate` makes every hard failure independently observable. Aggregate score cannot hide missing, failed, wrong-runtime, or unexecuted evidence.
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

The formal runtime operation is `POST /api/adapters/{adapter_id}/behavioral-evaluate`. The server selects the runtime from `adapter_registry.behavioral_activation_policy`; the request has no implicit runtime default. In production the policy and route force `real_model_runtime`. Development may explicitly configure `deterministic_runtime_only`, as the committed development config does. Evaluation runs in an isolated graph; only its adapter binding transition is serialized as a dedicated operational event. The artifact saga proceeds through prepared artifact, prepared registry binding, artifact finalization, registry finalization, and cross-registry reconciliation. `POST /api/evaluations/behavioral-reconciliation` classifies artifacts as `valid`, `prepared`, `orphan_result`, `orphan_registry_reference`, `hash_mismatch`, or `corrupt`. References are relative to the result root.

## Runtime Failure Checkpoints

The deterministic harness exposes `journal_accepted`, `journal_started`, `external_write`, `external_prepare`, `before_wal_append`, `after_wal_append`, `snapshot_temp_write`, `snapshot_temp_fsync`, `snapshot_atomic_replace`, `before_finalize`, `after_finalize`, and `before_journal_completed`. Restart always reconstructs a new graph from filesystem state. Journal corruption fails readiness closed; snapshot corruption is reconstructed only from a verified hash-chained WAL.

Crash recovery uses `abrupt_stop`, which stops Autonomy and aborts AgentRuntime without draining or invoking an active event completion hook. Recovery then constructs a separate `SubjectRuntimeHarness` from filesystem state.

## Real Model

Real-model validation is explicit and opt-in. It requires the model's structured public declaration, then independently reconciles that declaration with ActionIntent/effects, authoritative mutation, and policy invariants:

```bash
KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 just behavioral-real-model \
  /path/to/transformers-config.yaml candidate-adapter issue-133-real-run

KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 \
KAGYA_REAL_MODEL_ADAPTER_ID=candidate-adapter \
KAGYA_CONFIG_PATH=/path/to/transformers-config.yaml \
  uv run pytest tests/test_real_model_behavioral.py -m real_model -v
```

Normal CI runs the synthetic evaluator contract, provider factories with fakes, and deterministic runtime suite. A green CI run never claims `real_model_runtime_gate_passed`; only the real-model runtime can create that evidence. Production activation requires current ordinary gates, a passed/finalized/reconciled deterministic architecture artifact, and a passed/finalized/reconciled real candidate-model artifact. Missing, failed, stale, corrupt, hash-mismatched, and coverage-incomplete real evidence have distinct bounded status and activation codes.

### Hardware Evidence

The opt-in Transformers run `issue-133-rtx4080-20260726-v7` completed at `2026-07-26T04:13:44.786540Z` with a `valid` reconciled artifact and `real_model_runtime_gate_passed=true`.

- Source: verified commit `524e638f317a253c36220909177423a4e3e18a76`
- Base model: `google/gemma-4-12B-it`
- Requested and resolved model revision: `12ace6d648d72bd41519e140f1185f34d38c7e3d`
- Requested and resolved processor revision: `12ace6d648d72bd41519e140f1185f34d38c7e3d`
- Candidate adapter: `adapter-223573b8-f1a2-49ed-a80e-d0acab913a7b`
- Candidate adapter hash: `bfc2d53b832ebc0f919c5d274604b3c7ea4e7a8d72bde9e350c7d06e72c520aa`
- Fixture set hash: `6058307147fe1da460a6e2c1ade922a945e81935bc7286605187edba3f174bfa`
- Coverage manifest hash: `ba43ee25c9d74362c4879a1e8378e1cc5f4038c85f5c1caf2380a4281a6f16c3`
- Baseline and candidate aggregate scores: `1.0`
- Candidate dimension scores: all 23 required dimensions `1.0`
- Coverage: complete, with no missing dimensions or hard gates
- Hard-gate failures: none
- Provider generations: 17 baseline and 17 candidate
- Fallback used: false
- Hardware: NVIDIA GeForce RTX 4080, 16376 MiB, driver 595.45.04

This result is evidence for the exact immutable source, fixture, model, processor, and adapter artifacts above. It does not authorize artifacts with different provenance.

## Policy And Migration

`behavioral_activation_policy` defaults to `real_model_required`. Recognized environments are `production`, `development`, `test`, and `ci`; unknown environments fail validation. Production accepts only `real_model_required`. `deterministic_runtime_only` is limited to development, test, and CI and must be explicitly configured. `disabled` is limited to explicit test settings.

The config loader maps the removed `require_real_model_behavioral_gate` key explicitly: `true` becomes `real_model_required`, while `false` becomes `deterministic_runtime_only`. New configuration should use only the policy enum. Adapter registry schema v10 does not migrate legacy coverage or provenance fields as activation authority. A legacy active adapter may continue running with a warning, but reactivation and rollback promotion use current policy and coverage gates.

`POST /api/adapters/{adapter_id}/evaluate` accepts an empty, extra-forbidden request body and always loads separate configured baseline and registered candidate providers over server-owned eval sets. Client-provided scores and dimensions are rejected during schema validation. `GET /api/adapters/{adapter_id}/behavioral-evaluation-status` returns bounded ordinary, deterministic, real-model, artifact, policy, and eligibility states without absolute paths, raw hashes, or private configuration.
