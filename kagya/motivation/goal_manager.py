"""Persistent goals, conflict-aware adoption, and commitments."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any, Iterable
from uuid import uuid4

from kagya.identity import (
    EndorsementStatus,
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    identity_origin_from_json,
    legacy_identity_origin,
    new_identity_origin,
)


class GoalType(StrEnum):
    INTRINSIC = "intrinsic"
    EXTERNAL_REQUEST = "external_request"
    COMMITMENT = "commitment"


class GoalStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"


class GoalDecisionAction(StrEnum):
    ACTIVATE = "activate"
    SUSPEND = "suspend"
    RESUME = "resume"
    DEFER = "defer"
    REQUEST_INFORMATION = "request_information"
    NO_ACTION = "no_action"


class CommitmentStatus(StrEnum):
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    RELEASED = "released"
    BREACHED = "breached"


TERMINAL_GOAL_STATUSES = {
    GoalStatus.COMPLETED,
    GoalStatus.ABANDONED,
    GoalStatus.FAILED,
}


@dataclass(frozen=True)
class GoalTransition:
    transition_id: str
    from_status: str
    to_status: str
    reason: str
    outcome: str | None
    event_id: str | None
    event_sequence: int | None
    created_at: str


@dataclass(frozen=True)
class Goal:
    goal_id: str
    goal_type: GoalType
    description: str
    structured_target: dict[str, Any] | None
    origin_event_id: str | None
    origin_value_id: str | None
    identity_origin: IdentityOrigin
    priority: float
    urgency: float
    expected_utility: float
    confidence: float
    status: GoalStatus
    dependency_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    deadline: str | None
    value_effects: dict[str, float]
    needs_information: bool
    created_at: str
    updated_at: str
    transitions: tuple[GoalTransition, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.goal_id or not self.description:
            raise ValueError("goal_id and description must not be empty")
        for name in ("priority", "urgency", "expected_utility", "confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        if self.schema_version != 2:
            raise ValueError(f"Unsupported goal schema version: {self.schema_version}")
        if len(self.dependency_ids) != len(set(self.dependency_ids)):
            raise ValueError("Goal dependencies must be unique")
        if len(self.conflict_ids) != len(set(self.conflict_ids)):
            raise ValueError("Goal conflicts must be unique")
        if self.goal_id in self.dependency_ids or self.goal_id in self.conflict_ids:
            raise ValueError("A goal cannot depend on or conflict with itself")
        for value_id, effect in self.value_effects.items():
            if not value_id or not math.isfinite(effect) or not -1.0 <= effect <= 1.0:
                raise ValueError(
                    "Value effects require an ID and a finite value between -1 and one"
                )
        _parse_deadline(self.deadline)


@dataclass(frozen=True)
class GoalDecision:
    decision_id: str
    action: GoalDecisionAction
    goal_id: str | None
    score: float | None
    reasons: tuple[str, ...]
    conflicting_goal_ids: tuple[str, ...]
    event_id: str | None
    event_sequence: int | None
    created_at: str


@dataclass(frozen=True)
class GoalDecisionInput:
    active_goals: tuple[Goal, ...]
    ranked_candidates: tuple[tuple[str, float], ...]
    active_commitment_ids: tuple[str, ...]
    allowed_actions: tuple[GoalDecisionAction, ...]


@dataclass(frozen=True)
class CommitmentTransition:
    transition_id: str
    from_status: str
    to_status: str
    reason: str
    outcome: str | None
    event_id: str | None
    event_sequence: int | None
    created_at: str


@dataclass(frozen=True)
class Commitment:
    commitment_id: str
    description: str
    origin_event_id: str | None
    related_goal_id: str
    identity_origin: IdentityOrigin
    status: CommitmentStatus
    deadline: str | None
    created_at: str
    updated_at: str
    transitions: tuple[CommitmentTransition, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.commitment_id or not self.description or not self.related_goal_id:
            raise ValueError("commitment identifiers and description must not be empty")
        if self.schema_version != 2:
            raise ValueError(
                f"Unsupported commitment schema version: {self.schema_version}"
            )
        _parse_deadline(self.deadline)


class GoalManager:
    def __init__(self) -> None:
        self.goals: dict[str, Goal] = {}
        self.decisions: list[GoalDecision] = []

    def propose(
        self,
        *,
        goal_type: GoalType,
        description: str,
        structured_target: dict[str, Any] | None = None,
        origin_event_id: str | None = None,
        origin_value_id: str | None = None,
        identity_origin: IdentityOrigin | None = None,
        priority: float = 0.5,
        urgency: float = 0.5,
        expected_utility: float = 0.5,
        confidence: float = 0.5,
        dependency_ids: tuple[str, ...] = (),
        conflict_ids: tuple[str, ...] = (),
        deadline: str | None = None,
        value_effects: dict[str, float] | None = None,
        needs_information: bool = False,
        goal_id: str | None = None,
    ) -> Goal:
        identifier = goal_id or str(uuid4())
        if identifier in self.goals:
            raise ValueError(f"Goal already exists: {identifier}")
        if identifier in dependency_ids or identifier in conflict_ids:
            raise ValueError("A goal cannot depend on or conflict with itself")
        unknown_dependencies = set(dependency_ids) - self.goals.keys()
        if unknown_dependencies:
            raise ValueError(f"Unknown dependency: {sorted(unknown_dependencies)[0]}")
        unknown_conflicts = set(conflict_ids) - self.goals.keys()
        if unknown_conflicts:
            raise ValueError(f"Unknown conflict: {sorted(unknown_conflicts)[0]}")
        now = _now()
        goal = Goal(
            goal_id=identifier,
            goal_type=goal_type,
            description=description,
            structured_target=structured_target,
            origin_event_id=origin_event_id,
            origin_value_id=origin_value_id,
            identity_origin=identity_origin
            or _default_goal_origin(goal_type, origin_event_id),
            priority=priority,
            urgency=urgency,
            expected_utility=expected_utility,
            confidence=confidence,
            status=GoalStatus.CANDIDATE,
            dependency_ids=dependency_ids,
            conflict_ids=conflict_ids,
            deadline=deadline,
            value_effects=dict(value_effects or {}),
            needs_information=needs_information,
            created_at=now,
            updated_at=now,
        )
        self.goals[identifier] = goal
        return goal

    def get(self, goal_id: str) -> Goal:
        goal = self.goals.get(goal_id)
        if goal is None:
            raise ValueError(f"Unknown goal: {goal_id}")
        return goal

    def list_goals(self, status: GoalStatus | None = None) -> list[Goal]:
        return [
            goal
            for goal in sorted(self.goals.values(), key=lambda item: item.goal_id)
            if status is None or goal.status == status
        ]

    def rank(self, value_scores: dict[str, float] | None = None) -> list[tuple[Goal, float]]:
        scores = value_scores or {}
        ranked = [
            (goal, self._score(goal, scores.get(goal.goal_id, 0.0)))
            for goal in self.goals.values()
            if goal.status in {GoalStatus.CANDIDATE, GoalStatus.SUSPENDED}
        ]
        return sorted(ranked, key=lambda item: (item[1], item[0].goal_id), reverse=True)

    def adopt(
        self,
        goal_id: str,
        *,
        value_score: float = 0.0,
        value_scores: dict[str, float] | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
        now: datetime | None = None,
    ) -> GoalDecision:
        goal = self.get(goal_id)
        scores = value_scores or {goal_id: value_score}
        if goal.status not in {GoalStatus.CANDIDATE, GoalStatus.SUSPENDED}:
            raise ValueError(f"Goal cannot be adopted from {goal.status.value}")
        current_time = now or datetime.now(UTC)
        if current_time.tzinfo is None:
            raise ValueError("Goal evaluation time must include a timezone")
        if _is_expired(goal.deadline, current_time):
            self.transition(
                goal_id,
                GoalStatus.FAILED,
                reason="deadline_expired",
                event_id=event_id,
                event_sequence=event_sequence,
            )
            return self._decision(
                GoalDecisionAction.DEFER,
                goal,
                None,
                ("deadline_expired",),
                (),
                event_id,
                event_sequence,
            )
        incomplete = tuple(
            dependency_id
            for dependency_id in goal.dependency_ids
            if self.get(dependency_id).status != GoalStatus.COMPLETED
        )
        if goal.needs_information or incomplete:
            reasons = ("additional_information_required",) if goal.needs_information else (
                "dependencies_incomplete",
            )
            return self._decision(
                GoalDecisionAction.REQUEST_INFORMATION
                if goal.needs_information
                else GoalDecisionAction.DEFER,
                goal,
                self._score(goal, scores.get(goal_id, value_score)),
                reasons,
                incomplete,
                event_id,
                event_sequence,
            )
        conflicts = self._active_conflicts(goal)
        score = self._score(goal, scores.get(goal_id, value_score))
        blocking = tuple(
            conflict
            for conflict in conflicts
            if self._score(conflict, scores.get(conflict.goal_id, 0.0)) >= score
        )
        if blocking:
            return self._decision(
                GoalDecisionAction.DEFER,
                goal,
                score,
                ("higher_or_equal_priority_conflict",),
                tuple(item.goal_id for item in blocking),
                event_id,
                event_sequence,
            )
        for conflict in conflicts:
            self.transition(
                conflict.goal_id,
                GoalStatus.SUSPENDED,
                reason=f"superseded_by:{goal.goal_id}",
                event_id=event_id,
                event_sequence=event_sequence,
            )
            self._decision(
                GoalDecisionAction.SUSPEND,
                conflict,
                self._score(conflict, scores.get(conflict.goal_id, 0.0)),
                ("lower_priority_conflict",),
                (goal.goal_id,),
                event_id,
                event_sequence,
            )
        action = (
            GoalDecisionAction.RESUME
            if goal.status == GoalStatus.SUSPENDED
            else GoalDecisionAction.ACTIVATE
        )
        self.transition(
            goal_id,
            GoalStatus.ACTIVE,
            reason="selected_by_priority",
            event_id=event_id,
            event_sequence=event_sequence,
        )
        activated = self.get(goal_id)
        if activated.identity_origin.endorsement in {
            EndorsementStatus.PENDING,
            EndorsementStatus.UNCERTAIN,
        }:
            self.goals[goal_id] = replace(
                activated,
                identity_origin=activated.identity_origin.endorse(
                    "goal_adoption",
                    event_id=event_id,
                    event_sequence=event_sequence,
                ),
                updated_at=_now(),
            )
        return self._decision(
            action,
            self.get(goal_id),
            score,
            ("dependencies_satisfied", "selected_by_priority"),
            tuple(item.goal_id for item in conflicts),
            event_id,
            event_sequence,
        )

    def transition(
        self,
        goal_id: str,
        status: GoalStatus,
        *,
        reason: str,
        outcome: str | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Goal:
        goal = self.get(goal_id)
        if not reason:
            raise ValueError("Goal transition reason must not be empty")
        if status not in _ALLOWED_GOAL_TRANSITIONS[goal.status]:
            raise ValueError(
                f"Invalid goal transition: {goal.status.value} -> {status.value}"
            )
        transition = GoalTransition(
            transition_id=str(uuid4()),
            from_status=goal.status.value,
            to_status=status.value,
            reason=reason,
            outcome=outcome,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
        )
        updated = replace(
            goal,
            status=status,
            updated_at=transition.created_at,
            transitions=(*goal.transitions, transition),
        )
        self.goals[goal_id] = updated
        return updated

    def reevaluate(
        self,
        *,
        value_scores: dict[str, float] | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
        now: datetime | None = None,
    ) -> list[GoalDecision]:
        current_time = now or datetime.now(UTC)
        if current_time.tzinfo is None:
            raise ValueError("Goal evaluation time must include a timezone")
        start = len(self.decisions)
        for goal in list(self.goals.values()):
            if goal.status not in TERMINAL_GOAL_STATUSES and _is_expired(
                goal.deadline, current_time
            ):
                self.transition(
                    goal.goal_id,
                    GoalStatus.FAILED,
                    reason="deadline_expired",
                    event_id=event_id,
                    event_sequence=event_sequence,
                )
                self._decision(
                    GoalDecisionAction.DEFER,
                    self.get(goal.goal_id),
                    None,
                    ("deadline_expired",),
                    (),
                    event_id,
                    event_sequence,
                )
        scores = value_scores or {}
        for goal, _ in self.rank(scores):
            if goal.status in {GoalStatus.CANDIDATE, GoalStatus.SUSPENDED}:
                self.adopt(
                    goal.goal_id,
                    value_score=scores.get(goal.goal_id, 0.0),
                    value_scores=scores,
                    event_id=event_id,
                    event_sequence=event_sequence,
                    now=current_time,
                )
        if len(self.decisions) == start:
            self.decisions.append(
                GoalDecision(
                    decision_id=str(uuid4()),
                    action=GoalDecisionAction.NO_ACTION,
                    goal_id=None,
                    score=None,
                    reasons=("no_goal_state_change",),
                    conflicting_goal_ids=(),
                    event_id=event_id,
                    event_sequence=event_sequence,
                    created_at=_now(),
                )
            )
        return self.decisions[start:]

    def decision_input(
        self,
        *,
        value_scores: dict[str, float] | None = None,
        active_commitment_ids: Iterable[str] = (),
    ) -> GoalDecisionInput:
        return GoalDecisionInput(
            active_goals=tuple(self.list_goals(GoalStatus.ACTIVE)),
            ranked_candidates=tuple(
                (goal.goal_id, score) for goal, score in self.rank(value_scores)
            ),
            active_commitment_ids=tuple(sorted(active_commitment_ids)),
            allowed_actions=(
                GoalDecisionAction.ACTIVATE,
                GoalDecisionAction.SUSPEND,
                GoalDecisionAction.RESUME,
                GoalDecisionAction.DEFER,
                GoalDecisionAction.REQUEST_INFORMATION,
                GoalDecisionAction.NO_ACTION,
            ),
        )

    def restore(
        self,
        goals: Iterable[dict[str, Any]],
        decisions: Iterable[dict[str, Any]] = (),
    ) -> None:
        restored: dict[str, Goal] = {}
        for payload in goals:
            goal = _goal_from_json(payload)
            restored[goal.goal_id] = goal
        self.goals = restored
        self.decisions = [_decision_from_json(payload) for payload in decisions]

    def goals_json(self) -> list[dict[str, Any]]:
        return [asdict(goal) for goal in self.list_goals()]

    def decisions_json(self) -> list[dict[str, Any]]:
        return [asdict(decision) for decision in self.decisions]

    def _active_conflicts(self, goal: Goal) -> tuple[Goal, ...]:
        return tuple(
            candidate
            for candidate in self.goals.values()
            if candidate.status == GoalStatus.ACTIVE
            and (
                candidate.goal_id in goal.conflict_ids
                or goal.goal_id in candidate.conflict_ids
            )
        )

    @staticmethod
    def _score(goal: Goal, value_score: float) -> float:
        bounded_value_score = max(-1.0, min(1.0, value_score))
        return (
            0.3 * goal.priority
            + 0.25 * goal.urgency
            + 0.25 * goal.expected_utility
            + 0.2 * goal.confidence
            + 0.2 * bounded_value_score
        )

    def _decision(
        self,
        action: GoalDecisionAction,
        goal: Goal,
        score: float | None,
        reasons: tuple[str, ...],
        conflicts: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> GoalDecision:
        decision = GoalDecision(
            decision_id=str(uuid4()),
            action=action,
            goal_id=goal.goal_id,
            score=score,
            reasons=reasons,
            conflicting_goal_ids=conflicts,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
        )
        self.decisions.append(decision)
        return decision


class CommitmentStore:
    def __init__(self) -> None:
        self.commitments: dict[str, Commitment] = {}

    def create(
        self,
        *,
        description: str,
        related_goal_id: str,
        origin_event_id: str | None = None,
        identity_origin: IdentityOrigin | None = None,
        deadline: str | None = None,
        commitment_id: str | None = None,
    ) -> Commitment:
        identifier = commitment_id or str(uuid4())
        if identifier in self.commitments:
            raise ValueError(f"Commitment already exists: {identifier}")
        resolved_origin = identity_origin or new_identity_origin(
            OriginActor.SELF,
            OriginInputKind.INTERNAL_STATE,
            source_ref="commitment_store",
        )
        if resolved_origin.endorsement not in {
            EndorsementStatus.ENDORSED,
            EndorsementStatus.IMPOSED,
        }:
            raise ValueError("Commitment creation requires endorsed or imposed origin")
        now = _now()
        commitment = Commitment(
            commitment_id=identifier,
            description=description,
            origin_event_id=origin_event_id,
            related_goal_id=related_goal_id,
            identity_origin=resolved_origin,
            status=CommitmentStatus.ACTIVE,
            deadline=deadline,
            created_at=now,
            updated_at=now,
        )
        self.commitments[identifier] = commitment
        return commitment

    def get(self, commitment_id: str) -> Commitment:
        commitment = self.commitments.get(commitment_id)
        if commitment is None:
            raise ValueError(f"Unknown commitment: {commitment_id}")
        return commitment

    def list_commitments(
        self, status: CommitmentStatus | None = None
    ) -> list[Commitment]:
        return [
            commitment
            for commitment in sorted(
                self.commitments.values(), key=lambda item: item.commitment_id
            )
            if status is None or commitment.status == status
        ]

    def transition(
        self,
        commitment_id: str,
        status: CommitmentStatus,
        *,
        reason: str,
        outcome: str | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Commitment:
        commitment = self.get(commitment_id)
        if commitment.status != CommitmentStatus.ACTIVE:
            raise ValueError("Only active commitments can transition")
        if status == CommitmentStatus.ACTIVE:
            raise ValueError("Commitment is already active")
        if not reason:
            raise ValueError("Commitment transition reason must not be empty")
        transition = CommitmentTransition(
            transition_id=str(uuid4()),
            from_status=commitment.status.value,
            to_status=status.value,
            reason=reason,
            outcome=outcome,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
        )
        updated = replace(
            commitment,
            status=status,
            updated_at=transition.created_at,
            transitions=(*commitment.transitions, transition),
        )
        self.commitments[commitment_id] = updated
        return updated

    def restore(self, commitments: Iterable[dict[str, Any]]) -> None:
        restored: dict[str, Commitment] = {}
        for payload in commitments:
            commitment = _commitment_from_json(payload)
            restored[commitment.commitment_id] = commitment
        self.commitments = restored

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.list_commitments()]


_ALLOWED_GOAL_TRANSITIONS: dict[GoalStatus, set[GoalStatus]] = {
    GoalStatus.CANDIDATE: {
        GoalStatus.ACTIVE,
        GoalStatus.COMPLETED,
        GoalStatus.ABANDONED,
        GoalStatus.FAILED,
    },
    GoalStatus.ACTIVE: {
        GoalStatus.SUSPENDED,
        GoalStatus.COMPLETED,
        GoalStatus.ABANDONED,
        GoalStatus.FAILED,
    },
    GoalStatus.SUSPENDED: {
        GoalStatus.ACTIVE,
        GoalStatus.COMPLETED,
        GoalStatus.ABANDONED,
        GoalStatus.FAILED,
    },
    GoalStatus.COMPLETED: set(),
    GoalStatus.ABANDONED: set(),
    GoalStatus.FAILED: set(),
}


def _goal_from_json(payload: dict[str, Any]) -> Goal:
    if "schema_version" not in payload:
        identifier = str(payload.get("id", payload.get("goal_id", "")))
        description = str(
            payload.get("description", payload.get("content", identifier))
        )
        now = _now()
        return Goal(
            goal_id=identifier,
            goal_type=GoalType.INTRINSIC,
            description=description,
            structured_target=None,
            origin_event_id=None,
            origin_value_id=None,
            identity_origin=legacy_identity_origin("legacy_goal"),
            priority=0.5,
            urgency=0.5,
            expected_utility=0.5,
            confidence=0.5,
            status=GoalStatus.ACTIVE,
            dependency_ids=(),
            conflict_ids=(),
            deadline=None,
            value_effects={},
            needs_information=False,
            created_at=now,
            updated_at=now,
        )
    data = dict(payload)
    data["schema_version"] = 2
    data["identity_origin"] = identity_origin_from_json(
        data.get("identity_origin"), fallback_source="legacy_goal"
    )
    data["goal_type"] = GoalType(data["goal_type"])
    data["status"] = GoalStatus(data["status"])
    data["dependency_ids"] = tuple(data.get("dependency_ids", ()))
    data["conflict_ids"] = tuple(data.get("conflict_ids", ()))
    data["transitions"] = tuple(
        GoalTransition(**transition) for transition in data.get("transitions", ())
    )
    return Goal(**data)


def _decision_from_json(payload: dict[str, Any]) -> GoalDecision:
    data = dict(payload)
    data["action"] = GoalDecisionAction(data["action"])
    data["reasons"] = tuple(data.get("reasons", ()))
    data["conflicting_goal_ids"] = tuple(data.get("conflicting_goal_ids", ()))
    return GoalDecision(**data)


def _commitment_from_json(payload: dict[str, Any]) -> Commitment:
    if "schema_version" not in payload:
        identifier = str(payload.get("id", payload.get("commitment_id", "")))
        description = str(
            payload.get("description", payload.get("content", identifier))
        )
        now = _now()
        return Commitment(
            commitment_id=identifier,
            description=description,
            origin_event_id=None,
            related_goal_id=f"legacy:{identifier}",
            identity_origin=legacy_identity_origin("legacy_commitment"),
            status=CommitmentStatus.ACTIVE,
            deadline=None,
            created_at=now,
            updated_at=now,
        )
    data = dict(payload)
    data["schema_version"] = 2
    data["identity_origin"] = identity_origin_from_json(
        data.get("identity_origin"), fallback_source="legacy_commitment"
    )
    data["status"] = CommitmentStatus(data["status"])
    data["transitions"] = tuple(
        CommitmentTransition(**transition)
        for transition in data.get("transitions", ())
    )
    return Commitment(**data)


def _default_goal_origin(
    goal_type: GoalType, origin_event_id: str | None
) -> IdentityOrigin:
    if goal_type == GoalType.INTRINSIC:
        return new_identity_origin(
            OriginActor.SELF,
            OriginInputKind.INTERNAL_STATE,
            source_ref="goal_manager",
            event_id=origin_event_id,
        )
    return new_identity_origin(
        OriginActor.UNKNOWN,
        OriginInputKind.REQUEST,
        source_ref="goal_manager",
        event_id=origin_event_id,
        confidence=0.0,
    )


def _parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Goal deadlines must include a timezone")
    return parsed


def _is_expired(value: str | None, now: datetime) -> bool:
    deadline = _parse_deadline(value)
    return deadline is not None and deadline <= now


def _now() -> str:
    return datetime.now(UTC).isoformat()
