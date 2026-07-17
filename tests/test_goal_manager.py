from datetime import UTC, datetime, timedelta
import json

import pytest

from kagya.motivation import (
    CommitmentStatus,
    CommitmentStore,
    GoalDecisionAction,
    GoalManager,
    GoalStatus,
    GoalType,
)
from kagya.identity import (
    EndorsementStatus,
    OriginActor,
    OriginInputKind,
    new_identity_origin,
)


def test_goal_types_distinguish_intrinsic_external_and_commitment_origins() -> None:
    manager = GoalManager()

    intrinsic = manager.propose(
        goal_id="intrinsic",
        goal_type=GoalType.INTRINSIC,
        description="Explore an idea",
        origin_value_id="curiosity",
    )
    external = manager.propose(
        goal_id="external",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="Answer a request",
        origin_event_id="event-1",
    )
    commitment = manager.propose(
        goal_id="commitment",
        goal_type=GoalType.COMMITMENT,
        description="Keep a promise",
    )

    assert intrinsic.goal_type == GoalType.INTRINSIC
    assert external.goal_type == GoalType.EXTERNAL_REQUEST
    assert commitment.goal_type == GoalType.COMMITMENT
    assert intrinsic.identity_origin.actor == OriginActor.SELF
    assert intrinsic.identity_origin.endorsement == EndorsementStatus.ENDORSED
    assert external.identity_origin.actor == OriginActor.UNKNOWN
    assert external.identity_origin.endorsement == EndorsementStatus.PENDING


def test_goal_rejects_unknown_dependency_and_conflict_references() -> None:
    manager = GoalManager()

    with pytest.raises(ValueError, match="Unknown dependency"):
        manager.propose(
            goal_type=GoalType.INTRINSIC,
            description="Invalid dependency",
            dependency_ids=("missing",),
        )
    with pytest.raises(ValueError, match="Unknown conflict"):
        manager.propose(
            goal_type=GoalType.INTRINSIC,
            description="Invalid conflict",
            conflict_ids=("missing",),
        )


def test_higher_priority_conflict_suspends_active_goal_with_reason() -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="low",
        goal_type=GoalType.INTRINSIC,
        description="Low priority goal",
        priority=0.2,
        urgency=0.2,
        expected_utility=0.2,
        confidence=0.8,
    )
    manager.adopt("low", event_id="event-1", event_sequence=1)
    manager.propose(
        goal_id="high",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="High priority conflicting goal",
        priority=1.0,
        urgency=1.0,
        expected_utility=1.0,
        confidence=1.0,
        conflict_ids=("low",),
    )

    decision = manager.adopt("high", event_id="event-2", event_sequence=2)

    assert manager.get("low").status == GoalStatus.SUSPENDED
    assert manager.get("high").status == GoalStatus.ACTIVE
    assert manager.get("high").identity_origin.endorsement == EndorsementStatus.ENDORSED
    assert decision.action == GoalDecisionAction.ACTIVATE
    assert decision.conflicting_goal_ids == ("low",)
    assert manager.get("low").transitions[-1].reason == "superseded_by:high"
    assert any(item.action == GoalDecisionAction.SUSPEND for item in manager.decisions)


def test_lower_priority_conflict_is_deferred_without_dual_activation() -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="high",
        goal_type=GoalType.INTRINSIC,
        description="Existing goal",
        priority=1.0,
        urgency=1.0,
        expected_utility=1.0,
        confidence=1.0,
    )
    manager.adopt("high")
    manager.propose(
        goal_id="low",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="Conflicting request",
        priority=0.1,
        urgency=0.1,
        expected_utility=0.1,
        confidence=0.5,
        conflict_ids=("high",),
    )

    decision = manager.adopt("low")

    assert decision.action == GoalDecisionAction.DEFER
    assert manager.get("high").status == GoalStatus.ACTIVE
    assert manager.get("low").status == GoalStatus.CANDIDATE


def test_value_scores_apply_to_both_sides_of_conflict_selection() -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="active",
        goal_type=GoalType.INTRINSIC,
        description="Value-aligned active goal",
        priority=0.5,
        urgency=0.5,
        expected_utility=0.5,
        confidence=0.5,
    )
    manager.adopt("active")
    manager.propose(
        goal_id="candidate",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="Conflicting candidate",
        priority=0.5,
        urgency=0.5,
        expected_utility=0.5,
        confidence=0.5,
        conflict_ids=("active",),
    )

    decision = manager.adopt(
        "candidate",
        value_scores={"active": 1.0, "candidate": -1.0},
    )

    assert decision.action == GoalDecisionAction.DEFER
    assert manager.get("active").status == GoalStatus.ACTIVE


def test_suspended_goal_resumes_on_later_event() -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="goal",
        goal_type=GoalType.INTRINSIC,
        description="Persistent goal",
    )
    manager.adopt("goal")
    manager.transition("goal", GoalStatus.SUSPENDED, reason="waiting")

    decision = manager.adopt("goal", event_id="event-3", event_sequence=3)

    assert decision.action == GoalDecisionAction.RESUME
    assert manager.get("goal").status == GoalStatus.ACTIVE
    assert manager.get("goal").transitions[-1].event_sequence == 3


def test_dependencies_and_information_allow_explicit_deferral() -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="dependency",
        goal_type=GoalType.INTRINSIC,
        description="Prerequisite",
    )
    manager.propose(
        goal_id="dependent",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="Dependent goal",
        dependency_ids=("dependency",),
    )
    manager.propose(
        goal_id="unclear",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="Needs clarification",
        needs_information=True,
    )

    deferred = manager.adopt("dependent")
    information = manager.adopt("unclear")

    assert deferred.action == GoalDecisionAction.DEFER
    assert information.action == GoalDecisionAction.REQUEST_INFORMATION
    manager.transition("dependency", GoalStatus.COMPLETED, reason="finished")
    assert manager.adopt("dependent").action == GoalDecisionAction.ACTIVATE


def test_reevaluation_expires_deadline_without_external_input() -> None:
    manager = GoalManager()
    now = datetime(2026, 1, 2, tzinfo=UTC)
    manager.propose(
        goal_id="expired",
        goal_type=GoalType.EXTERNAL_REQUEST,
        description="Expired goal",
        deadline=(now - timedelta(seconds=1)).isoformat(),
    )

    decisions = manager.reevaluate(
        event_id="internal-event",
        event_sequence=4,
        now=now,
    )

    assert manager.get("expired").status == GoalStatus.FAILED
    assert manager.get("expired").transitions[-1].reason == "deadline_expired"
    assert decisions[0].reasons == ("deadline_expired",)
    assert decisions[0].event_id == "internal-event"


def test_reevaluation_can_choose_no_action() -> None:
    manager = GoalManager()

    decisions = manager.reevaluate(event_id="internal-idle", event_sequence=8)

    assert decisions[0].action == GoalDecisionAction.NO_ACTION
    assert decisions[0].reasons == ("no_goal_state_change",)


@pytest.mark.parametrize(
    "status",
    [GoalStatus.COMPLETED, GoalStatus.ABANDONED, GoalStatus.FAILED],
)
def test_terminal_transitions_record_reason_and_outcome(status: GoalStatus) -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="goal",
        goal_type=GoalType.INTRINSIC,
        description="Goal",
    )
    manager.adopt("goal")

    goal = manager.transition(
        "goal",
        status,
        reason="explicit_result",
        outcome="recorded outcome",
        event_id="event-5",
        event_sequence=5,
    )

    assert goal.transitions[-1].reason == "explicit_result"
    assert goal.transitions[-1].outcome == "recorded outcome"
    with pytest.raises(ValueError):
        manager.transition("goal", GoalStatus.ACTIVE, reason="invalid")


def test_goal_and_decision_history_round_trip_through_json() -> None:
    manager = GoalManager()
    manager.propose(
        goal_id="goal",
        goal_type=GoalType.INTRINSIC,
        description="Persistent goal",
    )
    manager.adopt("goal", event_id="event-6", event_sequence=6)
    goals = json.loads(json.dumps(manager.goals_json()))
    decisions = json.loads(json.dumps(manager.decisions_json()))

    restored = GoalManager()
    restored.restore(goals, decisions)

    assert restored.get("goal").status == GoalStatus.ACTIVE
    assert restored.get("goal").transitions[-1].event_id == "event-6"
    assert restored.decisions[-1].event_sequence == 6


def test_v1_goal_migrates_without_claiming_self_origin() -> None:
    manager = GoalManager()
    manager.restore(
        [
            {
                "schema_version": 1,
                "goal_id": "legacy-goal",
                "goal_type": "intrinsic",
                "description": "Legacy state",
                "structured_target": None,
                "origin_event_id": None,
                "origin_value_id": None,
                "priority": 0.5,
                "urgency": 0.5,
                "expected_utility": 0.5,
                "confidence": 0.5,
                "status": "active",
                "dependency_ids": [],
                "conflict_ids": [],
                "deadline": None,
                "value_effects": {},
                "needs_information": False,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "transitions": [],
            }
        ]
    )

    restored = manager.get("legacy-goal")
    assert restored.schema_version == 2
    assert restored.identity_origin.actor == OriginActor.INHERITED
    assert restored.identity_origin.endorsement == EndorsementStatus.UNCERTAIN


def test_v1_commitment_migrates_without_claiming_self_origin() -> None:
    store = CommitmentStore()
    store.restore(
        [
            {
                "schema_version": 1,
                "commitment_id": "legacy-promise",
                "description": "Legacy promise",
                "origin_event_id": None,
                "related_goal_id": "legacy-goal",
                "status": "active",
                "deadline": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "transitions": [],
            }
        ]
    )

    restored = store.get("legacy-promise")
    assert restored.schema_version == 2
    assert restored.identity_origin.actor == OriginActor.INHERITED
    assert restored.identity_origin.endorsement == EndorsementStatus.UNCERTAIN


def test_commitment_store_records_fulfillment_release_and_breach() -> None:
    for status in (
        CommitmentStatus.FULFILLED,
        CommitmentStatus.RELEASED,
        CommitmentStatus.BREACHED,
    ):
        store = CommitmentStore()
        store.create(
            commitment_id="promise",
            description="Keep a promise",
            related_goal_id="goal",
        )

        commitment = store.transition(
            "promise",
            status,
            reason="resolved",
            outcome="result",
        )

        assert commitment.status == status
        assert commitment.transitions[-1].reason == "resolved"


def test_commitment_store_rejects_unendorsed_external_origin() -> None:
    store = CommitmentStore()

    with pytest.raises(ValueError, match="requires endorsed"):
        store.create(
            commitment_id="unendorsed",
            description="External request",
            related_goal_id="goal",
            identity_origin=new_identity_origin(
                OriginActor.USER,
                OriginInputKind.REQUEST,
                source_ref="context:one",
            ),
        )
