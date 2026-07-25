from datetime import UTC, datetime
from pathlib import Path

import pytest

from kagya.actions import ActionExecutionLayer
from kagya.agency import (
    AGENCY_ATTRIBUTION_STATE_KEY,
    AgencyAttribution,
    AgencyAttributionStore,
    AttributionProjection,
    AttributionTarget,
    CausalContributor,
    CausalContributorKind,
)
from kagya.config import load_settings
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
    (
        "hidden_thought",
        "hidden-thought",
        "rawPrompt",
        "chain.of.thought",
        "reasoning",
        "free prose",
        "<think>private</think>",
    ),
)
def test_agency_records_reject_private_aliases_and_free_prose(
    private_value: str,
) -> None:
    payload = _attribution_payload()
    payload["evidence_refs"] = (private_value,)

    with pytest.raises(ValueError):
        AgencyAttribution.model_validate(payload)

    with pytest.raises(ValueError):
        AttributionProjection(
            attribution_id="attribution-safe",
            attribution_revision=1,
            target=AttributionTarget.VALUE,
            evidence_refs=(private_value,),
            applied_delta=0.0,
            applied_at=datetime.now(UTC),
        )


def test_invalid_private_agency_state_is_never_retained() -> None:
    payload = _attribution_payload()
    payload["hiddenThought"] = "private"
    saved: list[dict[str, object]] = []

    with pytest.raises(ValueError):
        AgencyAttributionStore(
            load=lambda: {
                "schema_version": 1,
                "records": [payload],
                "projections": [],
            },
            save=saved.append,
            validate_chain=lambda _: None,
        )

    assert saved == []


def test_autonomous_verification_schedules_shared_attribution_and_all_projections(
    tmp_path: Path,
) -> None:
    loop, execution = _loop(tmp_path)
    _decision(loop, "decision-success")
    runtime = AgentRuntime(queue_capacity=16)
    runtime.start()
    try:
        intent = runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.agency.intent",
            handler=lambda: execution.create_from_decision(
                "decision-success", idempotency_key="agency-success"
            ),
        ).value
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.agency.execute",
            handler=lambda: execution.execute(intent.intent_id),
        )
        assert loop.agency_attribution_store.list_current() == ()

        scheduler = SubjectScheduler(
            runtime,
            loop,
            budget=SchedulerBudget(
                max_events=8, max_inferences=0, max_wall_seconds=2.0
            ),
            reevaluation_interval_seconds=60.0,
        )
        cycle = scheduler.run_cycle(datetime.now(UTC))
    finally:
        runtime.shutdown()

    assert cycle.processed >= 1
    attribution = loop.agency_attribution_store.list_current()[0]
    self_contributor = next(
        item
        for item in attribution.contributors
        if item.kind == CausalContributorKind.SELF
    )
    assert self_contributor.causal_share == 0.25
    assert any(
        item.kind == CausalContributorKind.ENVIRONMENT
        for item in attribution.contributors
    )
    assert attribution.intended is True
    assert {
        item.target for item in loop.agency_attribution_store.state.projections
    } == set(AttributionTarget)
    assert all(
        attribution.reference in item.evidence_refs
        for item in loop.agency_attribution_store.state.projections
    )
    assert abs(loop.emotion_engine.state.valence) <= 0.05


def test_failure_retains_own_contribution_without_self_incapacity_inference(
    tmp_path: Path,
) -> None:
    loop, execution = _loop(tmp_path)
    _decision(loop, "decision-failure")
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    intent = runtime.execute(
        AgentEventType.ACTION_INTENT,
        source="test.agency.failure-intent",
        handler=lambda: execution.create_from_decision(
            "decision-failure", idempotency_key="agency-failure"
        ),
    ).value

    def fail(_: object, __: object) -> object:
        raise ValueError("structured failure")

    execution._invoke = fail  # type: ignore[method-assign]
    runtime.execute(
        AgentEventType.ACTION_EXECUTE,
        source="test.agency.failure-execute",
        handler=lambda: execution.execute(intent.intent_id),
    )
    try:
        attribution = runtime.execute(
            AgentEventType.ATTRIBUTION_APPLY,
            source="test.agency.failure",
            handler=lambda: loop.attribute_action_outcome(intent.intent_id),
        ).value
    finally:
        runtime.shutdown()

    assert attribution.contribution(CausalContributorKind.SELF) > 0.0
    observation = loop.metacognition.observations["decision-failure"]
    assert observation.success is False
    assert observation.self_contribution < 0.5
    assert observation.agency_adjusted_success > 0.5
    assert not loop.metacognition.hypotheses


def test_revision_is_runtime_only_deduplicated_and_snapshot_restorable(
    tmp_path: Path,
) -> None:
    loop, execution = _loop(tmp_path)
    _decision(loop, "decision-revision")
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    try:
        intent = runtime.execute(
            AgentEventType.ACTION_INTENT,
            source="test.agency.revision-intent",
            handler=lambda: execution.create_from_decision(
                "decision-revision", idempotency_key="agency-revision"
            ),
        ).value
        runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.agency.revision-execute",
            handler=lambda: execution.execute(intent.intent_id),
        )
        initial = runtime.execute(
            AgentEventType.ATTRIBUTION_APPLY,
            source="test.agency.initial",
            handler=lambda: loop.attribute_action_outcome(intent.intent_id),
        ).value
    finally:
        runtime.shutdown()
    contributors = (
        CausalContributor(
            kind=CausalContributorKind.SELF,
            causal_share=0.6,
            confidence=0.9,
            controllability=0.8,
            foreseeability=0.8,
            responsibility_share=0.6,
        ),
        CausalContributor(
            kind=CausalContributorKind.ENVIRONMENT,
            causal_share=0.4,
            confidence=0.9,
            controllability=0.1,
            foreseeability=0.5,
            responsibility_share=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="AgentRuntime"):
        loop.revise_agency_attribution(
            initial.attribution_id,
            expected_revision=1,
            contributors=contributors,
            intended=True,
            uncertainty=0.1,
            evidence_refs=("observation:later",),
            reason_code="later_evidence",
        )

    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    try:
        revised = runtime.execute(
            AgentEventType.ATTRIBUTION_REVISE,
            source="test.agency.revise",
            handler=lambda: loop.revise_agency_attribution(
                initial.attribution_id,
                expected_revision=1,
                contributors=contributors,
                intended=True,
                uncertainty=0.1,
                evidence_refs=("observation:later",),
                reason_code="later_evidence",
            ),
        ).value
    finally:
        runtime.shutdown()

    assert revised.revision == 2
    assert len(loop.agency_attribution_store.history(initial.attribution_id)) == 2
    assert len(loop.agency_attribution_store.state.projections) == 12

    store = AgentStateStore(tmp_path / "state.json")
    snapshot = store.capture(loop, 3)
    assert snapshot.extensions[AGENCY_ATTRIBUTION_STATE_KEY]["schema_version"] == 1
    restored_loop, restored_execution = _loop(tmp_path / "restored")
    store.restore_into(restored_loop, snapshot)
    restored_loop.action_execution = restored_execution
    assert (
        restored_loop.agency_attribution_store.history(initial.attribution_id)[
            -1
        ].revision
        == 2
    )

    restored_loop.persistent_state.extensions[AGENCY_ATTRIBUTION_STATE_KEY] = {
        "schema_version": 99
    }
    with pytest.raises(ValueError):
        restored_loop.restore_agency_attribution_state()


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


def _decision(loop: KagyaMainLoop, decision_id: str) -> None:
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
        value_effects={},
        appraisal_contributions={},
    )
    fallback = ActionCandidate(
        candidate_id=f"{decision_id}-fallback",
        candidate_type=ActionType.NO_OP,
        proposed_action="Do not read metadata",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )
    loop.create_decision([action, fallback], decision_id=decision_id)


def _attribution_payload() -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "schema_version": 1,
        "attribution_id": "attribution-safe",
        "revision": 1,
        "decision_id": "decision-safe",
        "action_intent_id": "intent-safe",
        "execution_receipt_id": "receipt-safe",
        "observation_id": "observation-safe",
        "outcome_ref": "decision:decision-safe:outcome",
        "contributors": (
            {
                "kind": "self",
                "causal_share": 1.0,
                "confidence": 1.0,
                "controllability": 1.0,
                "foreseeability": 1.0,
                "responsibility_share": 1.0,
            },
        ),
        "intended": True,
        "uncertainty": 0.0,
        "evidence_refs": ("observation:safe",),
        "reason_codes": ("structured_evidence",),
        "created_at": now,
        "updated_at": now,
    }
