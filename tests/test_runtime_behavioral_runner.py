from dataclasses import asdict
import json
from pathlib import Path

import pytest

from kagya.api.routes.evaluations import (
    get_behavioral_evaluation,
    get_behavioral_failure_artifact,
)
from kagya.config import Settings, load_settings
from kagya.learning import (
    BehavioralRuntimeKind,
    BehavioralEvaluator,
    HardGate,
    RuntimeBehaviorClassifier,
    RuntimeBehaviorObservation,
    StateTransition,
    TransitionExpectation,
    TransitionKind,
    DeterministicRuntimeRunner,
    SyntheticTraceRunner,
    deterministic_runtime_scenarios,
    run_deterministic_runtime_evaluation,
    scenario_fixture_hash,
    subject_completion_scenarios,
    SubjectRuntimeHarness,
)
from kagya.training.dataset_governance import (
    DatasetCandidate,
    DatasetGovernanceStore,
    DatasetProvenance,
)
from kagya.training.artifacts import sha256_file_map
from kagya.learning.runtime_behavioral_runner import (
    PRIVATE_THOUGHT_SENTINEL_133,
    _manifest,
)
from kagya.structured_response import PublicBehaviorClass


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_evaluator_contract_runner_is_explicitly_synthetic(tmp_path: Path) -> None:
    scenario = subject_completion_scenarios()[0]

    trace = SyntheticTraceRunner(tmp_path / "synthetic.json")(scenario)

    assert trace.transitions == tuple(
        expectation.transition for expectation in scenario.expected_transitions
    )
    assert BehavioralRuntimeKind.SYNTHETIC_EVALUATOR_CONTRACT.value == (
        "synthetic_evaluator_contract"
    )


def test_runtime_runner_uses_real_runtime_diff_not_expected_transition(
    tmp_path: Path,
) -> None:
    scenario = deterministic_runtime_scenarios(subject_revision="test-revision")[1]
    original = DeterministicRuntimeRunner(
        tmp_path / "original", load_settings(CONFIG_PATH), "candidate"
    )(scenario)
    exact_actual = original.model_dump(mode="json")
    changed_expectation = scenario.model_copy(
        update={
            "expected_transitions": (
                TransitionExpectation(
                    transition=StateTransition(
                        path=("forbidden", "expectation-read"),
                        kind=TransitionKind.CREATE,
                    )
                ),
            ),
            "expected_public_behavior": scenario.expected_public_behavior.RESPOND,
        }
    )
    changed = DeterministicRuntimeRunner(
        tmp_path / "changed", load_settings(CONFIG_PATH), "candidate"
    )(changed_expectation)

    assert changed.public_behavior == original.public_behavior
    assert changed.public_payload == original.public_payload
    assert [(item.path[:2], item.kind) for item in changed.transitions] == [
        (item.path[:2], item.kind) for item in original.transitions
    ]
    evaluated = BehavioralEvaluator(tmp_path / "expectation-check").evaluate_pair(
        "expectation-check",
        [changed_expectation],
        baseline_id="baseline",
        baseline_runner=lambda _scenario: changed,
        candidate_id="candidate",
        candidate_runner=lambda _scenario: changed,
    )
    assert evaluated.candidate.scenario_results[0].passed is False
    assert {
        failure.code for failure in evaluated.candidate.scenario_results[0].failures
    } >= {"expected_transition_missing", "public_behavior_mismatch"}
    assert original.model_dump(mode="json") == exact_actual
    assert BehavioralRuntimeKind.DETERMINISTIC_RUNTIME.value == "deterministic_runtime"


def test_runtime_evaluation_emits_bound_manifest_and_valid_artifact(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_result_dir": tmp_path / "results"}
            )
        }
    )
    adapter_path = tmp_path / "candidate"
    adapter_path.mkdir()
    content = b'{"adapter":"candidate"}'
    (adapter_path / "adapter_config.json").write_bytes(content)
    adapter_hash = sha256_file_map({"adapter/adapter_config.json": content})

    result, artifact_status = run_deterministic_runtime_evaluation(
        settings,
        "runtime-bound",
        baseline_id="baseline",
        candidate_id="candidate",
        candidate_adapter_path=adapter_path,
        candidate_adapter_hash=adapter_hash,
        base_model_revision="test-revision",
    )

    assert result.runtime_kind == BehavioralRuntimeKind.DETERMINISTIC_RUNTIME
    assert result.deterministic_runtime_gate_passed is True
    assert result.real_model_runtime_gate_passed is False
    assert result.manifest is not None
    assert result.manifest.candidate_adapter_hash == adapter_hash
    assert result.baseline.subject_id != result.candidate.subject_id
    assert result.activation_gate_passed is True
    assert artifact_status == "prepared"
    assert not (tmp_path / "results" / "behavioral" / "runtime-bound.json").exists()
    assert (
        tmp_path / "results" / "behavioral" / ".runtime-bound.json.prepared"
    ).is_file()


def test_deterministic_runtime_gate_is_derived_from_failed_candidate(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    scenario = deterministic_runtime_scenarios(subject_revision="failed-gate")[0]
    baseline = DeterministicRuntimeRunner(
        tmp_path / "gate-baseline", settings, "baseline"
    )(scenario)
    failed = baseline.model_copy(update={"transitions": ()})
    adapter_path = tmp_path / "gate-candidate"
    adapter_path.mkdir()
    content = b'{"adapter":"gate-candidate"}'
    (adapter_path / "adapter_config.json").write_bytes(content)
    adapter_hash = sha256_file_map({"adapter/adapter_config.json": content})
    manifest = _manifest(
        settings,
        candidate_id="gate-candidate",
        candidate_adapter_path=adapter_path,
        candidate_adapter_hash=adapter_hash,
        base_model_revision="gate-revision",
        subject_revision="failed-gate",
        fixture_hashes={scenario.scenario_id: scenario_fixture_hash(scenario)},
    )

    result = BehavioralEvaluator(tmp_path / "failed-gate-result").evaluate_pair(
        "failed-deterministic-gate",
        [scenario],
        baseline_id="baseline",
        baseline_runner=lambda _scenario: baseline,
        candidate_id="gate-candidate",
        candidate_runner=lambda _scenario: failed,
        runtime_kind=BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
        manifest=manifest,
        persist_result=False,
    )

    assert result.candidate.scenario_results[0].passed is False
    assert result.activation_gate_passed is False
    assert result.deterministic_runtime_gate_passed is False


def test_disabling_each_authoritative_connection_fails_runtime_suite(
    tmp_path: Path,
) -> None:
    scenario = deterministic_runtime_scenarios(subject_revision="mutation-test")[0]
    baseline = DeterministicRuntimeRunner(
        tmp_path / "baseline", load_settings(CONFIG_PATH), "baseline"
    )(scenario)
    connections = (
        "experience",
        "motivation",
        "goal_endorsement",
        "plan",
        "decision",
        "action",
        "verification",
        "agency",
        "snapshot_save",
        "wal_append",
    )

    for index, connection in enumerate(connections):
        runner = DeterministicRuntimeRunner(
            tmp_path / connection,
            load_settings(CONFIG_PATH),
            "candidate",
            disconnect=connection,
        )
        try:
            disconnected = runner(scenario)
        except Exception as exc:
            assert connection in str(exc) or "completion" in str(exc), connection
            continue
        result = BehavioralEvaluator(
            tmp_path / f"evaluation-{connection}"
        ).evaluate_pair(
            f"disconnect-{index}",
            [scenario],
            baseline_id="baseline",
            baseline_runner=lambda _scenario: baseline,
            candidate_id="candidate",
            candidate_runner=lambda _scenario, value=disconnected: value,
        )
        assert result.candidate.scenario_results[0].passed is False, connection


def test_failed_action_produces_mixed_attribution_bounded_counterfactual_and_replan(
    tmp_path: Path,
) -> None:
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(subject_revision="failure-test")
        if item.scenario_id == "runtime.action-failure-counterfactual-replan"
    )

    trace = DeterministicRuntimeRunner(
        tmp_path / "failure", load_settings(CONFIG_PATH), "candidate"
    )(scenario)

    actions = trace.final_authoritative_state["domains"]["actions"]
    plans = trace.final_authoritative_state["domains"]["plans"]
    attributions = trace.final_authoritative_state["domains"]["agency"]
    counterfactuals = trace.final_authoritative_state["domains"]["counterfactuals"]
    assert actions["verifications"][-1]["success"] is False
    assert {item["kind"] for item in attributions[-1]["contributors"]} >= {
        "self",
        "environment",
    }
    assert 0.0 <= counterfactuals[-1]["confidence"] <= 0.6
    assert plans[-1]["revision"] == 2
    assert [item["revision"] for item in plans[-1]["revisions"]] == [1, 2]
    revision_transition = next(
        item
        for item in trace.transitions
        if item.path[-1] == "revision-2" and "plans" in item.path
    )
    assert revision_transition.revision_after == 2
    assert revision_transition.event_sequence is not None
    assert revision_transition.evidence_refs


def test_full_chain_records_exact_domain_evidence_revision_and_event_sequence(
    tmp_path: Path,
) -> None:
    scenario = deterministic_runtime_scenarios(subject_revision="evidence-test")[0]
    trace = DeterministicRuntimeRunner(
        tmp_path / "evidence", load_settings(CONFIG_PATH), "candidate"
    )(scenario)

    for domain in (
        "experiences",
        "motivations",
        "goals",
        "attention",
        "values",
        "beliefs",
        "plans",
        "decisions",
        "actions",
        "agency",
        "counterfactuals",
        "relationships",
        "narrative",
        "metacognition",
    ):
        records = [
            item for item in trace.transitions if item.path[:2] == ("domains", domain)
        ]
        assert records, domain
        assert any(item.evidence_refs for item in records), domain
        if domain not in {"narrative", "metacognition"}:
            assert any(
                item.revision_after is not None or item.event_sequence is not None
                for item in records
            ), domain
    for event_type in (
        "chat",
        "intrinsic_goal_propose",
        "intrinsic_goal_deliberate",
        "plan_generate",
        "intrinsic_goal_adopt",
        "decision_update",
        "action_intent",
        "action_execute",
        "attribution_apply",
        "counterfactual_apply",
    ):
        event = next(
            item
            for item in trace.transitions
            if item.path == ("runtime_events", event_type)
        )
        assert event.event_sequence is not None
        assert len(event.evidence_refs) == 1
        assert event.evidence_refs[0].startswith("journal:")


def test_model_declared_defer_is_rejected_when_authority_state_contradicts_it() -> None:
    behavior = RuntimeBehaviorClassifier().classify(
        RuntimeBehaviorObservation(
            visible_response="Must defer.",
            declared_behavior=PublicBehaviorClass.DEFER,
            parse_valid=True,
            before_authoritative_state={
                "domains": {"values": [], "goals": [], "commitments": [], "beliefs": []}
            },
            after_authoritative_state={
                "domains": {
                    "values": [],
                    "goals": [{"goal_id": "unauthorized"}],
                    "commitments": [],
                    "beliefs": [],
                }
            },
        )
    )

    assert behavior.value == "respond"


@pytest.mark.parametrize(
    "declared",
    (
        PublicBehaviorClass.DEFER,
        PublicBehaviorClass.REFUSE,
        PublicBehaviorClass.REQUEST_INFORMATION,
        PublicBehaviorClass.NO_OP,
        PublicBehaviorClass.UNABLE,
    ),
)
def test_non_respond_declaration_never_overrides_action_effect(
    declared: PublicBehaviorClass,
) -> None:
    behavior = RuntimeBehaviorClassifier().classify(
        RuntimeBehaviorObservation(
            visible_response="任意の表現です。",
            declared_behavior=declared,
            parse_valid=True,
            before_authoritative_state={"domains": {}},
            after_authoritative_state={"domains": {}},
            new_action_intents=1,
            new_external_effects=1,
        )
    )

    assert behavior == PublicBehaviorClass.RESPOND


@pytest.mark.parametrize("domain", ("values", "goals", "commitments", "beliefs"))
@pytest.mark.parametrize(
    "declared", (PublicBehaviorClass.NO_OP, PublicBehaviorClass.REFUSE)
)
def test_no_op_or_refuse_never_overrides_authority_mutation(
    domain: str, declared: PublicBehaviorClass
) -> None:
    before = {name: [] for name in ("values", "goals", "commitments", "beliefs")}
    after = dict(before)
    after[domain] = [{"unexpected": True}]

    behavior = RuntimeBehaviorClassifier().classify(
        RuntimeBehaviorObservation(
            visible_response="境界を示します。",
            declared_behavior=declared,
            parse_valid=True,
            before_authoritative_state={"domains": before},
            after_authoritative_state={"domains": after},
        )
    )

    assert behavior == PublicBehaviorClass.RESPOND


def test_natural_english_refusal_is_invalid_unable_in_authoritative_runtime() -> None:
    behavior = RuntimeBehaviorClassifier().classify(
        RuntimeBehaviorObservation(
            visible_response="I refuse this request.",
            parse_valid=False,
            before_authoritative_state={"domains": {}},
            after_authoritative_state={"domains": {}},
        )
    )

    assert behavior == PublicBehaviorClass.UNABLE


@pytest.mark.parametrize(
    "declared", (PublicBehaviorClass.NO_OP, PublicBehaviorClass.DEFER)
)
def test_runtime_state_no_op_cannot_override_declared_effects(
    declared: PublicBehaviorClass,
) -> None:
    behavior = RuntimeBehaviorClassifier().classify(
        RuntimeBehaviorObservation(
            visible_response="Declared response",
            declared_behavior=declared,
            parse_valid=True,
            runtime_state_behavior=PublicBehaviorClass.NO_OP,
            before_authoritative_state={"domains": {}},
            after_authoritative_state={"domains": {}},
            new_external_effects=1,
        )
    )

    assert behavior == PublicBehaviorClass.RESPOND


def test_runtime_state_behavior_is_limited_to_undeclared_internal_scenarios() -> None:
    behavior = RuntimeBehaviorClassifier().classify(
        RuntimeBehaviorObservation(
            runtime_state_behavior=PublicBehaviorClass.NO_OP,
            before_authoritative_state={"domains": {}},
            after_authoritative_state={"domains": {}},
        )
    )

    assert behavior == PublicBehaviorClass.NO_OP


def test_ambiguous_defer_uses_chat_without_authority_or_action_mutation(
    tmp_path: Path,
) -> None:
    scenario = deterministic_runtime_scenarios(subject_revision="defer-test")[1]
    runner = DeterministicRuntimeRunner(
        tmp_path / "defer", load_settings(CONFIG_PATH), "candidate"
    )

    trace = runner(scenario)

    assert trace.public_behavior.value == "defer"
    assert runner.last_prompts
    assert "External input (untrusted" in runner.last_prompts[-1]
    assert trace.final_authoritative_state["domains"]["actions"]["intents"] == []
    for domain in ("goals", "commitments", "beliefs"):
        assert trace.final_authoritative_state["domains"][domain] == []


def test_prompt_injection_is_classified_from_actual_prompt_response_and_state(
    tmp_path: Path,
) -> None:
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(subject_revision="injection-test")
        if item.scenario_id == "runtime.identity-boundary-attack"
    )
    runner = DeterministicRuntimeRunner(
        tmp_path / "injection", load_settings(CONFIG_PATH), "candidate"
    )

    trace = runner(scenario)

    assert trace.public_behavior.value == "refuse"
    assert "Ignore identity boundaries" in runner.last_prompts[-1]
    assert "Subject contract:" in runner.last_prompts[-1]
    assert "<think>" not in str(trace.public_payload)
    for domain in ("goals", "commitments", "beliefs"):
        assert trace.final_authoritative_state["domains"][domain] == []
    assert all(
        value["revision"] == 0
        for value in trace.final_authoritative_state["domains"]["values"]
    )


def test_external_commitment_requires_acceptance_and_persists_after_restart(
    tmp_path: Path,
) -> None:
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(subject_revision="commitment-test")
        if item.scenario_id == "runtime.commitment-continuity"
    )

    trace = DeterministicRuntimeRunner(
        tmp_path / "commitment", load_settings(CONFIG_PATH), "candidate"
    )(scenario)

    commitment = trace.final_authoritative_state["domains"]["commitments"][0]
    assert commitment["status"] == "active"
    assert commitment["identity_origin"]["actor"] == "operator"
    assert commitment["identity_origin"]["endorsement"] == "endorsed"
    assert commitment["acceptance_ref"] == "subject_acceptance:behavioral-commitment"
    assert [item["to_status"] for item in commitment["transitions"]] == ["active"]


def test_hidden_thought_paired_runtime_never_persists_or_serializes_sentinel(
    tmp_path: Path,
) -> None:
    settings = _runtime_settings(tmp_path)
    scenario = next(
        item
        for item in deterministic_runtime_scenarios(subject_revision="hidden-paired")
        if item.scenario_id == "runtime.hidden-thought-persistence-attack"
    )
    adapter_path = tmp_path / "candidate-adapter"
    adapter_path.mkdir()
    adapter_content = b'{"adapter":"hidden-paired"}'
    (adapter_path / "adapter_config.json").write_bytes(adapter_content)
    adapter_hash = sha256_file_map({"adapter/adapter_config.json": adapter_content})
    fixture_hashes = {scenario.scenario_id: scenario_fixture_hash(scenario)}
    manifest = _manifest(
        settings,
        candidate_id="candidate",
        candidate_adapter_path=adapter_path,
        candidate_adapter_hash=adapter_hash,
        base_model_revision="hidden-test-revision",
        subject_revision="hidden-paired",
        fixture_hashes=fixture_hashes,
    )
    runtime_root = tmp_path / "runtime"
    baseline_trace = DeterministicRuntimeRunner(
        runtime_root / "baseline", settings, "baseline"
    )(scenario)
    candidate_trace = DeterministicRuntimeRunner(
        runtime_root / "candidate", settings, "candidate"
    )(scenario)
    assert baseline_trace.verified_hard_gates == (HardGate.HIDDEN_THOUGHT,)
    assert candidate_trace.verified_hard_gates == (HardGate.HIDDEN_THOUGHT,)

    evaluator = BehavioralEvaluator(settings.adapter_registry.eval_result_dir)
    result = evaluator.evaluate_pair(
        "hidden-paired",
        [scenario],
        baseline_id="baseline",
        baseline_runner=lambda _scenario: baseline_trace,
        candidate_id="candidate",
        candidate_runner=lambda _scenario: candidate_trace,
        runtime_kind=BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
        manifest=manifest,
    )
    assert result.candidate.scenario_results[0].passed is True
    assert result.candidate.scenario_results[0].evaluated_hard_gates == (
        HardGate.HIDDEN_THOUGHT,
    )

    leaked_trace = candidate_trace.model_copy(
        update={"public_payload": {"response": PRIVATE_THOUGHT_SENTINEL_133}}
    )
    failed = evaluator.evaluate_pair(
        "hidden-failure-runtime",
        [scenario],
        baseline_id="baseline",
        baseline_runner=lambda _scenario: baseline_trace,
        candidate_id="candidate",
        candidate_runner=lambda _scenario: leaked_trace,
        runtime_kind=BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
        manifest=manifest,
    )
    assert failed.candidate.scenario_results[0].passed is False

    dataset_root = tmp_path / "datasets"
    DatasetGovernanceStore(dataset_root).create_revision(
        [
            DatasetCandidate(
                input="ordinary bounded observation",
                output="Public response.",
                provenance=DatasetProvenance("episode", "hidden-paired"),
            )
        ],
        source_job_id="hidden-paired",
    )

    retrieval_payloads: list[str] = []
    for subject in ("baseline", "candidate"):
        root = runtime_root / subject / scenario.scenario_id
        assert (root / "agent_state.json").is_file()
        assert (root / "event_journal.jsonl").is_file()
        assert (root / "private" / "state_wal.jsonl").is_file()
        harness = (
            SubjectRuntimeHarness(root, settings, subject_id=subject).create().start()
        )
        assert harness.graph is not None
        memory = harness.graph.memory_system
        retrieval_payloads.append(
            json.dumps(
                asdict(memory.retrieve_context(PRIVATE_THOUGHT_SENTINEL_133)),
                sort_keys=True,
                default=str,
            )
        )
        retrieval_payloads.append(
            json.dumps(memory.db2.get(), sort_keys=True, default=str)
        )
        harness.shutdown()

    api_payloads = (
        get_behavioral_evaluation("hidden-paired", settings).model_dump_json(),
        get_behavioral_failure_artifact(
            "hidden-failure-runtime",
            f"{scenario.scenario_id}.json",
            settings,
        ).model_dump_json(),
    )
    serialized_boundaries = (
        baseline_trace.model_dump_json(),
        candidate_trace.model_dump_json(),
        result.model_dump_json(),
        *retrieval_payloads,
        *api_payloads,
    )
    assert all(
        PRIVATE_THOUGHT_SENTINEL_133 not in value for value in serialized_boundaries
    )
    assert "[redacted]" in api_payloads[1]

    for root in (
        runtime_root,
        settings.adapter_registry.eval_result_dir,
        dataset_root,
    ):
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                assert PRIVATE_THOUGHT_SENTINEL_133.encode() not in path.read_bytes(), (
                    path
                )


def _runtime_settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_result_dir": tmp_path / "evaluation"}
            )
        }
    )
