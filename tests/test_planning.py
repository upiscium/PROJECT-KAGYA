import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kagya.config import load_settings
from kagya.decision import ActionCandidate
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import (
    AgentRuntime,
    AgentStateStore,
    KagyaMainLoop,
    SchedulerBudget,
    SubjectScheduler,
    WorkingMemoryKind,
)
from kagya.motivation import GoalStatus, GoalType
from kagya.planning import (
    EvidenceReference,
    PlanStatus,
    PlanStore,
    StepStatus,
    parse_plan_candidate,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_plan_candidate_rejects_unknown_dependencies_cycles_and_private_prose() -> None:
    unknown = _candidate_payload()
    unknown["steps"][1]["dependency_ids"] = ["missing"]
    with pytest.raises(ValidationError, match="Unknown Step dependency"):
        parse_plan_candidate(unknown)

    cyclic = _candidate_payload()
    cyclic["steps"][0]["dependency_ids"] = ["verify"]
    with pytest.raises(ValidationError, match="cycle"):
        parse_plan_candidate(cyclic)

    private = _candidate_payload()
    private["reasoning"] = "raw model prose"
    with pytest.raises(ValueError, match="private or free-form"):
        parse_plan_candidate(private)

    unknown_field = _candidate_payload()
    unknown_field["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        parse_plan_candidate(unknown_field)

    prose_parameter = _candidate_payload()
    prose_parameter["steps"][0]["parameters"] = {"message": "unstructured model prose"}
    with pytest.raises(ValidationError, match="structured tokens"):
        parse_plan_candidate(prose_parameter)


def test_step_completion_requires_expected_observation_and_evidence() -> None:
    store = PlanStore()
    store.create(parse_plan_candidate(_candidate_payload()), actor_id="operator")
    active = store.activate("plan-1")

    assert [
        item.step_id for item in active.step_states if item.status == StepStatus.READY
    ] == ["collect"]
    store.start_step("plan-1", "collect")

    with pytest.raises(ValueError, match="requires verification evidence"):
        store.complete_step("plan-1", "collect", ())
    with pytest.raises(ValueError, match="expected observation"):
        store.complete_step(
            "plan-1",
            "collect",
            (_evidence("wrong_observation", "observation"),),
        )

    advanced = store.complete_step(
        "plan-1",
        "collect",
        (_evidence("data_collected", "observation"),),
    )
    assert advanced.step_state("collect").status == StepStatus.COMPLETED
    assert advanced.step_state("verify").status == StepStatus.READY
    assert [step.step_id for _, step, _ in store.actionable_steps()] == ["verify"]


def test_plan_completion_revision_retry_and_safe_action_candidate() -> None:
    store = PlanStore()
    candidate = parse_plan_candidate(_candidate_payload())
    store.create(candidate, actor_id="operator")
    store.activate("plan-1")
    store.start_step("plan-1", "collect")
    retried = store.fail_step("plan-1", "collect", reason_code="transient_failure")
    assert retried.step_state("collect").status == StepStatus.WAITING_RETRY
    assert retried.step_state("collect").retry_at is not None

    revised_payload = _candidate_payload()
    revised_payload["steps"][0]["action_code"] = "collect_structured_data_v2"
    revised = store.revise(
        "plan-1",
        parse_plan_candidate(revised_payload),
        expected_revision=1,
        reason_code="operator_correction",
        actor_id="operator-2",
    )
    assert revised.revision == 2
    assert revised.revisions[-1].reason_code == "operator_correction"
    assert revised.revisions[-1].actor_id == "operator-2"
    assert revised.revisions[0].final_step_states[0].status == StepStatus.WAITING_RETRY
    assert revised.step_state("collect").status == StepStatus.READY
    assert revised.current_revision.steps[0].rollback is not None

    candidate = store.action_candidate("plan-1", "collect")
    assert candidate.plan_id == "plan-1"
    assert candidate.plan_revision == 2
    assert candidate.step_id == "collect"
    assert candidate.proposed_action == "collect_structured_data_v2"

    stale = ActionCandidate(
        **{
            **candidate.__dict__,
            "plan_revision": 1,
        }
    )
    with pytest.raises(ValueError, match="stale Plan revision"):
        store.validate_candidate(stale)


def test_replan_invalidates_changed_completed_steps_and_their_dependents() -> None:
    store = PlanStore()
    store.create(parse_plan_candidate(_candidate_payload()), actor_id="operator")
    store.activate("plan-1")
    store.start_step("plan-1", "collect")
    store.complete_step(
        "plan-1",
        "collect",
        (_evidence("data_collected", "observation"),),
    )
    revised_payload = _candidate_payload()
    revised_payload["steps"][0]["action_code"] = "collect_structured_data_v2"

    revised = store.revise(
        "plan-1",
        parse_plan_candidate(revised_payload),
        expected_revision=1,
        reason_code="evidence_invalidated",
        actor_id="operator",
    )

    assert revised.step_state("collect").status == StepStatus.READY
    assert revised.step_state("verify").status == StepStatus.PENDING
    assert revised.revisions[0].final_step_states[0].status == StepStatus.COMPLETED


def test_main_loop_persists_mid_plan_and_only_ready_steps_in_working_memory(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path)
    loop.propose_goal(
        goal_id="goal-1",
        goal_type=GoalType.INTRINSIC,
        description="Structured persistent goal",
    )
    loop.adopt_goal("goal-1")
    loop.create_plan(parse_plan_candidate(_candidate_payload()), actor_id="operator")
    loop.activate_plan("plan-1")

    ready = [
        item
        for item in loop.working_memory.items
        if item.kind == WorkingMemoryKind.STEP
    ]
    assert [item.item_id for item in ready] == ["plan-step:plan-1:collect"]
    assert json.loads(ready[0].content or "{}")["step_id"] == "collect"

    loop.start_plan_step("plan-1", "collect")
    assert not [
        item
        for item in loop.working_memory.items
        if item.kind == WorkingMemoryKind.STEP
    ]
    state_store = AgentStateStore(tmp_path / "agent_state.json")
    state_store.save(state_store.capture(loop, 7))

    restored = _loop(tmp_path / "restored")
    state_store.restore_into(restored, state_store.load(1.0))
    plan = restored.plan_store.get("plan-1")
    assert plan.status == PlanStatus.ACTIVE
    assert plan.step_state("collect").status == StepStatus.IN_PROGRESS
    assert plan.step_state("collect").attempt_count == 1


def test_plan_completion_updates_goal_only_after_success_evidence(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path)
    loop.propose_goal(
        goal_id="goal-1",
        goal_type=GoalType.INTRINSIC,
        description="Evidence-backed goal",
    )
    loop.adopt_goal("goal-1")
    loop.create_plan(parse_plan_candidate(_candidate_payload()), actor_id="operator")
    loop.activate_plan("plan-1")

    with pytest.raises(ValueError, match="Plan success condition"):
        loop.transition_goal("goal-1", GoalStatus.COMPLETED, reason="premature")

    loop.start_plan_step("plan-1", "collect")
    loop.complete_plan_step(
        "plan-1", "collect", (_evidence("data_collected", "observation"),)
    )
    loop.start_plan_step("plan-1", "verify")
    completed = loop.complete_plan_step(
        "plan-1", "verify", (_evidence("result_verified", "verification"),)
    )

    assert completed.status == PlanStatus.COMPLETED
    assert loop.goal_manager.get("goal-1").status == GoalStatus.COMPLETED
    assert not [
        item
        for item in loop.working_memory.items
        if item.kind == WorkingMemoryKind.STEP
    ]
    serialized = json.dumps(loop.persistent_state.motivation_extensions["plans"])
    assert "hidden_thought" not in serialized
    assert "raw model prose" not in serialized


def test_goal_suspension_pauses_plan_and_removes_actionable_step(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path)
    loop.propose_goal(
        goal_id="goal-1",
        goal_type=GoalType.INTRINSIC,
        description="Suspend and resume Plan",
    )
    loop.adopt_goal("goal-1")
    loop.create_plan(parse_plan_candidate(_candidate_payload()), actor_id="operator")
    loop.activate_plan("plan-1")

    loop.transition_goal("goal-1", GoalStatus.SUSPENDED, reason="operator_pause")
    assert loop.plan_store.get("plan-1").status == PlanStatus.PAUSED
    assert loop.current_plan_candidates() == []
    assert not [
        item
        for item in loop.working_memory.items
        if item.kind == WorkingMemoryKind.STEP
    ]
    with pytest.raises(ValueError, match="active Goal"):
        loop.start_plan_step("plan-1", "collect")

    loop.adopt_goal("goal-1")
    assert loop.plan_store.get("plan-1").status == PlanStatus.ACTIVE
    assert loop.current_plan_candidates()[0].step_id == "collect"


def test_scheduler_advances_retry_and_timeout_without_executing_actions(
    tmp_path: Path,
) -> None:
    loop = _loop(tmp_path)
    loop.propose_goal(
        goal_id="goal-1",
        goal_type=GoalType.INTRINSIC,
        description="Scheduled Plan lifecycle",
    )
    loop.adopt_goal("goal-1")
    loop.create_plan(parse_plan_candidate(_candidate_payload()), actor_id="operator")
    loop.activate_plan("plan-1")
    loop.start_plan_step("plan-1", "collect")
    loop.fail_plan_step("plan-1", "collect", reason_code="transient_failure")
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    scheduler = SubjectScheduler(
        runtime,
        loop,
        budget=SchedulerBudget(max_events=2, max_inferences=0, max_wall_seconds=1.0),
        reevaluation_interval_seconds=60.0,
    )

    scheduler.run_cycle(datetime.now(UTC) + timedelta(seconds=1))
    retried = loop.plan_store.get("plan-1")
    assert retried.step_state("collect").status == StepStatus.IN_PROGRESS
    assert retried.step_state("collect").attempt_count == 2

    scheduler.run_cycle(datetime.now(UTC) + timedelta(seconds=61))
    timed_out = loop.plan_store.get("plan-1")
    assert timed_out.status == PlanStatus.FAILED
    assert timed_out.step_state("collect").transitions[-1].reason_code == "step_timeout"
    assert timed_out.current_revision.steps[0].rollback is not None
    runtime.shutdown()


def _candidate_payload() -> dict[str, object]:
    condition = {
        "condition_code": "verified_result",
        "required_evidence_types": ["verification"],
    }
    return {
        "schema_version": 1,
        "plan_id": "plan-1",
        "goal_id": "goal-1",
        "success_condition": condition,
        "failure_condition": {
            "condition_code": "attempts_exhausted",
            "required_evidence_types": ["failure"],
        },
        "abandonment_condition": {
            "condition_code": "operator_abandoned",
            "required_evidence_types": ["operator_decision"],
        },
        "steps": [
            _step_payload(
                "collect",
                "collect_structured_data",
                "data_collected",
                "observation",
                [],
            ),
            _step_payload(
                "verify",
                "verify_structured_result",
                "result_verified",
                "verification",
                ["collect"],
            ),
        ],
    }


def _step_payload(
    step_id: str,
    action_code: str,
    observation_code: str,
    evidence_type: str,
    dependencies: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "step_id": step_id,
        "action_type": "internal",
        "action_code": action_code,
        "parameters": {"mode": "structured"},
        "dependency_ids": dependencies,
        "expected_observation": {
            "observation_code": observation_code,
            "evidence_types": [evidence_type],
        },
        "verification": {
            "verification_code": f"verify_{step_id}",
            "required_evidence_types": [evidence_type],
            "minimum_evidence_count": 1,
        },
        "retry": {"max_attempts": 2, "backoff_seconds": 0.0},
        "timeout_seconds": 60.0,
        "rollback": {
            "action_type": "internal",
            "action_code": f"rollback_{step_id}",
            "parameters": {"mode": "structured"},
        },
    }


def _evidence(observation_code: str, evidence_type: str) -> EvidenceReference:
    return EvidenceReference(
        reference=f"evidence:{observation_code}",
        evidence_type=evidence_type,
        observation_code=observation_code,
    )


def _loop(tmp_path: Path) -> KagyaMainLoop:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={"persist_directory": tmp_path / "chroma"}
            )
        }
    )
    return KagyaMainLoop(
        settings,
        DummyProvider(),
        DualMemorySystem(settings, embedding_function=DeterministicEmbeddingFunction()),
    )
