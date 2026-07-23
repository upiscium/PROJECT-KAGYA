from pathlib import Path

from kagya.config import load_settings
from kagya.learning import (
    BehavioralRuntimeKind,
    BehavioralEvaluator,
    RuntimeBehaviorClassifier,
    RuntimeBehaviorObservation,
    StateTransition,
    TransitionExpectation,
    TransitionKind,
    DeterministicRuntimeRunner,
    SyntheticTraceRunner,
    deterministic_runtime_scenarios,
    run_deterministic_runtime_evaluation,
    subject_completion_scenarios,
)
from kagya.training.artifacts import sha256_file_map


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
    original = DeterministicRuntimeRunner(
        tmp_path / "original", load_settings(CONFIG_PATH), "candidate"
    )(scenario)
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
    assert result.manifest is not None
    assert result.manifest.candidate_adapter_hash == adapter_hash
    assert result.baseline.subject_id != result.candidate.subject_id
    assert result.activation_gate_passed is True
    assert artifact_status == "valid"
    assert (tmp_path / "results" / "behavioral" / "runtime-bound.json").is_file()


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
    scenario = deterministic_runtime_scenarios(subject_revision="failure-test")[6]

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
        "plans",
        "decisions",
        "actions",
        "agency",
        "counterfactuals",
    ):
        records = [
            item for item in trace.transitions if item.path[:2] == ("domains", domain)
        ]
        assert records, domain
        assert any(item.evidence_refs for item in records), domain
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
            visible_response='{"behavior_class":"defer","response":"Must defer."}',
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
    scenario = deterministic_runtime_scenarios(subject_revision="injection-test")[3]
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
    scenario = deterministic_runtime_scenarios(subject_revision="commitment-test")[5]

    trace = DeterministicRuntimeRunner(
        tmp_path / "commitment", load_settings(CONFIG_PATH), "candidate"
    )(scenario)

    commitment = trace.final_authoritative_state["domains"]["commitments"][0]
    assert commitment["status"] == "active"
    assert commitment["identity_origin"]["actor"] == "operator"
    assert commitment["identity_origin"]["endorsement"] == "endorsed"
    assert commitment["acceptance_ref"] == "subject_acceptance:behavioral-commitment"
    assert [item["to_status"] for item in commitment["transitions"]] == ["active"]
