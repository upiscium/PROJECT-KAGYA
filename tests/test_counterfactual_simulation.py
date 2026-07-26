from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kagya.actions import ActionExecutionLayer
from kagya.agency import CausalContributor, CausalContributorKind
from kagya.config import load_settings
from kagya.counterfactual import (
    COUNTERFACTUAL_STATE_KEY,
    AlternativeOutcome,
    CounterfactualSignal,
    CounterfactualTarget,
    EvidenceStatus,
)
from kagya.decision import ActionCandidate, ActionType, PredictedOutcome
from kagya.memory import DualMemorySystem
from kagya.memory.dual_memory_system import DeterministicEmbeddingFunction
from kagya.models import DummyProvider
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentStateStore,
    KagyaMainLoop,
    SchedulerBudget,
    SubjectScheduler,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.mark.parametrize(
    "private_value",
    ("hidden_thought", "rawPrompt", "chain.of.thought", "free prose"),
)
def test_counterfactual_records_reject_private_or_unstructured_authority(
    private_value: str,
) -> None:
    with pytest.raises(ValueError):
        AlternativeOutcome(
            candidate_id="alternative",
            candidate_type="no_op",
            plausible_utility=0.0,
            confidence=0.5,
            evidence_status=EvidenceStatus.SPECULATIVE,
            assumption_codes=(private_value,),
            evidence_refs=("decision:one",),
        )


def test_post_attribution_scheduler_simulates_bounded_relief_and_updates_targets(
    tmp_path: Path,
) -> None:
    loop, execution = _loop(tmp_path)
    _decision(loop, "decision-relief", alternative_has_prediction=True)
    runtime = AgentRuntime(queue_capacity=16)
    runtime.start()
    try:
        intent = runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.counterfactual.intent",
            handler=lambda: execution.create_from_decision(
                "decision-relief", idempotency_key="counterfactual-relief"
            ),
        ).value
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.counterfactual.execute",
            handler=lambda: execution.execute(intent.intent_id),
        )
        observed = loop.decision_store.get("decision-relief").actual_outcome
        scheduler = SubjectScheduler(
            runtime,
            loop,
            budget=SchedulerBudget(
                max_events=8, max_inferences=0, max_wall_seconds=2.0
            ),
            reevaluation_interval_seconds=60.0,
        )
        scheduler.run_cycle(datetime.now(UTC))
        assert loop.counterfactual_store.list_current() == ()
        scheduler.run_cycle(datetime.now(UTC))
    finally:
        runtime.shutdown()

    simulation = loop.counterfactual_store.list_current()[0]
    assert simulation.signal == CounterfactualSignal.RELIEF
    assert simulation.confidence <= 0.6
    assert all(
        item.evidence_status == EvidenceStatus.SPECULATIVE
        for item in simulation.alternatives
    )
    assert {item.target for item in loop.counterfactual_store.state.projections} == set(
        CounterfactualTarget
    )
    assert all(
        abs(item.applied_delta) <= 0.05
        for item in loop.counterfactual_store.state.projections
    )
    assert loop.decision_store.get("decision-relief").actual_outcome == observed
    assert simulation.action_intent_id == intent.intent_id
    assert simulation.outcome_ref == "decision:decision-relief:outcome"
    assert any(
        item.subject_ref == "value:care"
        for item in loop.counterfactual_store.state.projections
    )

    _decision(
        loop,
        "decision-future",
        alternative_has_prediction=True,
    )
    future = loop.decision_store.get("decision-future")
    future_by_type = {
        item.candidate.candidate_type: item for item in future.considered_candidates
    }
    assert (
        future_by_type[ActionType.INTERNAL].metacognition_contributions[
            "counterfactual:decision_calibration"
        ]
        > 0.0
    )
    plan_candidates = tuple(
        replace(
            item.candidate,
            plan_id="future-plan",
            plan_revision=1,
            step_id=item.candidate.candidate_id,
        )
        for item in future.considered_candidates
    )
    strategy_scores = loop._calibrated_candidate_scores(
        plan_candidates, ActionType.INTERNAL
    )
    no_op = next(
        item for item in plan_candidates if item.candidate_type == ActionType.NO_OP
    )
    assert strategy_scores[no_op.candidate_id]["counterfactual:plan_strategy"] < 0.0


def test_no_valid_alternative_does_not_schedule_or_infer(tmp_path: Path) -> None:
    loop, execution = _loop(tmp_path)
    _decision(loop, "decision-no-alternative", alternative_has_prediction=False)
    runtime = AgentRuntime(queue_capacity=16)
    runtime.start()
    try:
        intent = runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.counterfactual.no-alternative-intent",
            handler=lambda: execution.create_from_decision(
                "decision-no-alternative", idempotency_key="no-alternative"
            ),
        ).value
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.counterfactual.no-alternative-execute",
            handler=lambda: execution.execute(intent.intent_id),
        )
        scheduler = SubjectScheduler(
            runtime,
            loop,
            budget=SchedulerBudget(
                max_events=8, max_inferences=0, max_wall_seconds=2.0
            ),
            reevaluation_interval_seconds=60.0,
        )
        scheduler.run_cycle(datetime.now(UTC))
        scheduler.run_cycle(datetime.now(UTC))
    finally:
        runtime.shutdown()

    assert loop.agency_attribution_store.list_current()
    assert loop.counterfactual_store.list_current() == ()
    assert not any(
        "counterfactual" in item.schedule_id for item in scheduler._all_schedules()
    )


def test_attribution_revision_revises_simulation_once_and_snapshot_restores(
    tmp_path: Path,
) -> None:
    loop, execution = _loop(tmp_path)
    _decision(loop, "decision-revision", alternative_has_prediction=True)
    runtime = AgentRuntime(queue_capacity=16)
    runtime.start()
    try:
        intent = runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.counterfactual.revision-intent",
            handler=lambda: execution.create_from_decision(
                "decision-revision", idempotency_key="counterfactual-revision"
            ),
        ).value
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.counterfactual.revision-execute",
            handler=lambda: execution.execute(intent.intent_id),
        )
        attribution = runtime.execute(
            AgentEventType.ATTRIBUTION_APPLY,
            source="test.counterfactual.attribution",
            handler=lambda: loop.attribute_action_outcome(intent.intent_id),
        ).value
        initial = runtime.execute(
            AgentEventType.COUNTERFACTUAL_APPLY,
            source="test.counterfactual.initial",
            handler=lambda: loop.simulate_counterfactual(attribution.attribution_id),
        ).value
        contributors = (
            CausalContributor(
                kind=CausalContributorKind.SELF,
                causal_share=0.5,
                confidence=0.9,
                controllability=0.8,
                foreseeability=0.8,
                responsibility_share=0.5,
            ),
            CausalContributor(
                kind=CausalContributorKind.ENVIRONMENT,
                causal_share=0.5,
                confidence=0.8,
                controllability=0.1,
                foreseeability=0.5,
                responsibility_share=0.0,
            ),
        )
        revised_attribution = runtime.execute(
            AgentEventType.ATTRIBUTION_REVISE,
            source="test.counterfactual.attribution-revision",
            handler=lambda: loop.revise_agency_attribution(
                attribution.attribution_id,
                expected_revision=1,
                contributors=contributors,
                intended=True,
                uncertainty=0.1,
                evidence_refs=("observation:later",),
                reason_code="later_evidence",
            ),
        ).value
        revised = runtime.execute(
            AgentEventType.COUNTERFACTUAL_APPLY,
            source="test.counterfactual.revision",
            handler=lambda: loop.simulate_counterfactual(
                revised_attribution.attribution_id
            ),
        ).value
        duplicate = runtime.execute(
            AgentEventType.COUNTERFACTUAL_APPLY,
            source="test.counterfactual.duplicate",
            handler=lambda: loop.simulate_counterfactual(
                revised_attribution.attribution_id
            ),
        ).value
    finally:
        runtime.shutdown()

    assert initial.revision == 1
    assert revised.revision == 2
    assert duplicate == revised
    assert len(loop.counterfactual_store.history(initial.simulation_id)) == 2
    projection_count = len(loop.counterfactual_store.state.projections)
    assert projection_count > len(CounterfactualTarget)
    expected = -min(0.05, revised.signal_magnitude * revised.confidence * 0.2)
    assert loop.counterfactual_store.calibration(
        CounterfactualTarget.DECISION_CALIBRATION,
        "candidate-type:no_op",
    ) == pytest.approx(expected)

    store = AgentStateStore(tmp_path / "state.json")
    snapshot = store.capture(loop, 5)
    assert snapshot.extensions[COUNTERFACTUAL_STATE_KEY]["schema_version"] == 1
    restored, restored_execution = _loop(tmp_path / "restored")
    store.restore_into(restored, snapshot)
    restored.action_execution = restored_execution
    assert (
        restored.counterfactual_store.history(initial.simulation_id)[-1].revision == 2
    )
    assert len(restored.counterfactual_store.state.projections) == projection_count


def _loop(tmp_path: Path) -> tuple[KagyaMainLoop, ActionExecutionLayer]:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={"persist_directory": tmp_path / "chroma"}
            ),
            "actions": settings.actions.model_copy(
                update={
                    "document_root": tmp_path,
                    "calendar_path": tmp_path / "calendar.json",
                }
            ),
        }
    )
    loop = KagyaMainLoop(
        settings,
        DummyProvider(),
        DualMemorySystem(settings, embedding_function=DeterministicEmbeddingFunction()),
    )
    execution = ActionExecutionLayer(
        loop,
        document_root=tmp_path,
        calendar_path=tmp_path / "calendar.json",
    )
    loop.action_execution = execution
    return loop, execution


def _decision(
    loop: KagyaMainLoop,
    decision_id: str,
    *,
    alternative_has_prediction: bool,
) -> None:
    action = ActionCandidate(
        candidate_id=f"{decision_id}-action",
        candidate_type=ActionType.INTERNAL,
        proposed_action="Read project metadata",
        parameters={
            "action": {
                "tool_name": "restricted_metadata_read",
                "arguments": {"namespace": "project", "key": "name"},
            }
        },
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id="success",
                description="Metadata is observed",
                probability=0.9,
                utility=1.0,
            ),
        ),
        uncertainty=0.1,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={"care": 0.4},
        appraisal_contributions={},
    )
    fallback = ActionCandidate(
        candidate_id=f"{decision_id}-fallback",
        candidate_type=ActionType.NO_OP,
        proposed_action="Do not read metadata",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(
            (
                PredictedOutcome(
                    outcome_id="unchanged",
                    description="State remains unchanged",
                    probability=1.0,
                    utility=0.2,
                ),
            )
            if alternative_has_prediction
            else ()
        ),
        uncertainty=0.2,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={"care": -0.2},
        appraisal_contributions={},
    )
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    try:
        runtime.execute(
            AgentEventType.DECISION_UPDATE,
            source="test.decision",
            handler=lambda: loop.create_decision(
                [action, fallback], decision_id=decision_id
            ),
        )
    finally:
        runtime.shutdown()
