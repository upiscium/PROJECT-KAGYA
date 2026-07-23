from pathlib import Path

from kagya.config import load_settings
from kagya.learning import (
    BehavioralRuntimeKind,
    BehavioralEvaluator,
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
    changed_expectation = scenario.model_copy(update={"expected_transitions": ()})
    runner = DeterministicRuntimeRunner(
        tmp_path / "runner", load_settings(CONFIG_PATH), "candidate"
    )

    trace = runner(changed_expectation)

    assert any(
        transition.path == ("last_processed_event_sequence",)
        for transition in trace.transitions
    )
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
    trace = DeterministicRuntimeRunner(
        tmp_path / "subject", load_settings(CONFIG_PATH), "candidate"
    )(scenario)
    connections = {
        "experience": ("domains", "experiences"),
        "motivation": ("domains", "motivations"),
        "goal_endorsement": ("runtime_events", "intrinsic_goal_deliberate"),
        "plan": ("domains", "plans"),
        "decision": ("domains", "decisions"),
        "action": ("domains", "actions", "intents"),
        "verification": ("domains", "actions", "verifications"),
        "agency": ("domains", "agency"),
        "snapshot_save": ("last_processed_event_sequence",),
        "wal_append": ("durability", "wal_append"),
    }

    for index, (connection, path) in enumerate(connections.items()):
        disconnected = trace.model_copy(
            update={
                "transitions": tuple(
                    transition
                    for transition in trace.transitions
                    if transition.path != path
                )
            }
        )
        result = BehavioralEvaluator(tmp_path / connection).evaluate_pair(
            f"disconnect-{index}",
            [scenario],
            baseline_id="baseline",
            baseline_runner=lambda _scenario: trace,
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
