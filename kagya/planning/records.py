"""Strict, versioned plans between persistent goals and action candidates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
import math
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.decision import ActionCandidate, ActionType


PLAN_STATE_KEY = "plans"


class PlanStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class StepStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    WAITING_RETRY = "waiting_retry"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanCondition(_StrictModel):
    condition_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    required_evidence_types: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_types(self) -> "PlanCondition":
        _unique_codes(self.required_evidence_types, "condition evidence types")
        return self


class ExpectedObservation(_StrictModel):
    observation_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    evidence_types: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_types(self) -> "ExpectedObservation":
        _unique_codes(self.evidence_types, "observation evidence types")
        return self


class VerificationPolicy(_StrictModel):
    verification_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    required_evidence_types: tuple[str, ...] = Field(min_length=1)
    minimum_evidence_count: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def validate_evidence_types(self) -> "VerificationPolicy":
        _unique_codes(self.required_evidence_types, "verification evidence types")
        return self


class RetryPolicy(_StrictModel):
    max_attempts: int = Field(default=1, ge=1, le=100)
    backoff_seconds: float = Field(
        default=0.0, ge=0.0, le=604800.0, allow_inf_nan=False
    )


class RollbackPolicy(_StrictModel):
    action_type: ActionType
    action_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_private_fields(self) -> "RollbackPolicy":
        _reject_private_fields(self.parameters)
        _validate_structured_parameters(self.parameters)
        return self


class StepDefinition(_StrictModel):
    schema_version: Literal[1] = 1
    step_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )
    action_type: ActionType
    action_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependency_ids: tuple[str, ...] = ()
    expected_observation: ExpectedObservation
    verification: VerificationPolicy
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float = Field(gt=0.0, le=604800.0, allow_inf_nan=False)
    rollback: RollbackPolicy | None = None

    @model_validator(mode="after")
    def validate_step(self) -> "StepDefinition":
        _unique_codes(self.dependency_ids, "step dependencies")
        if self.step_id in self.dependency_ids:
            raise ValueError("A Step cannot depend on itself")
        _reject_private_fields(self.parameters)
        _validate_structured_parameters(self.parameters)
        return self


class PlanCandidate(_StrictModel):
    schema_version: Literal[1] = 1
    plan_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )
    goal_id: str = Field(min_length=1, max_length=128)
    success_condition: PlanCondition
    failure_condition: PlanCondition
    abandonment_condition: PlanCondition
    steps: tuple[StepDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dag(self) -> "PlanCandidate":
        _validate_step_dag(self.steps)
        return self


class EvidenceReference(_StrictModel):
    reference: str = Field(min_length=1, max_length=512, pattern=r"^\S+$")
    evidence_type: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    observation_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )


class StepTransition(_StrictModel):
    transition_id: str
    from_status: StepStatus
    to_status: StepStatus
    reason_code: str
    event_id: str | None = None
    event_sequence: int | None = None
    created_at: datetime


class StepState(_StrictModel):
    step_id: str
    status: StepStatus = StepStatus.PENDING
    attempt_count: int = Field(default=0, ge=0)
    evidence: tuple[EvidenceReference, ...] = ()
    started_at: datetime | None = None
    retry_at: datetime | None = None
    completed_at: datetime | None = None
    transitions: tuple[StepTransition, ...] = ()


class PlanRevision(_StrictModel):
    revision: int = Field(ge=1)
    reason_code: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.:-]*$"
    )
    actor_id: str = Field(min_length=1, max_length=128)
    success_condition: PlanCondition
    failure_condition: PlanCondition
    abandonment_condition: PlanCondition
    steps: tuple[StepDefinition, ...]
    final_step_states: tuple[StepState, ...] = ()
    created_at: datetime


class PlanTransition(_StrictModel):
    transition_id: str
    from_status: PlanStatus
    to_status: PlanStatus
    reason_code: str
    evidence_refs: tuple[str, ...] = ()
    event_id: str | None = None
    event_sequence: int | None = None
    created_at: datetime


class Plan(_StrictModel):
    schema_version: Literal[1] = 1
    plan_id: str
    goal_id: str
    revision: int = Field(ge=1)
    status: PlanStatus
    revisions: tuple[PlanRevision, ...]
    step_states: tuple[StepState, ...]
    transitions: tuple[PlanTransition, ...] = ()
    created_at: datetime
    updated_at: datetime

    @property
    def current_revision(self) -> PlanRevision:
        return self.revisions[-1]

    def step_definition(self, step_id: str) -> StepDefinition:
        for step in self.current_revision.steps:
            if step.step_id == step_id:
                return step
        raise ValueError(f"Unknown Step: {step_id}")

    def step_state(self, step_id: str) -> StepState:
        for state in self.step_states:
            if state.step_id == step_id:
                return state
        raise ValueError(f"Unknown Step state: {step_id}")

    @model_validator(mode="after")
    def validate_plan(self) -> "Plan":
        if not self.revisions or self.revision != self.revisions[-1].revision:
            raise ValueError("Plan revision does not match revision history")
        if tuple(item.revision for item in self.revisions) != tuple(
            range(1, self.revision + 1)
        ):
            raise ValueError("Plan revisions must be contiguous")
        _validate_step_dag(self.current_revision.steps)
        expected = {item.step_id for item in self.current_revision.steps}
        actual = {item.step_id for item in self.step_states}
        if expected != actual or len(actual) != len(self.step_states):
            raise ValueError("Plan Step state does not match current revision")
        return self


class PlanStore:
    def __init__(self) -> None:
        self.plans: dict[str, Plan] = {}

    def create(
        self,
        candidate: PlanCandidate,
        *,
        actor_id: str,
        reason_code: str = "initial_plan",
    ) -> Plan:
        if candidate.plan_id in self.plans:
            raise ValueError(f"Plan already exists: {candidate.plan_id}")
        now = _now()
        revision = PlanRevision(
            revision=1,
            reason_code=reason_code,
            actor_id=actor_id,
            success_condition=candidate.success_condition,
            failure_condition=candidate.failure_condition,
            abandonment_condition=candidate.abandonment_condition,
            steps=candidate.steps,
            created_at=now,
        )
        plan = Plan(
            plan_id=candidate.plan_id,
            goal_id=candidate.goal_id,
            revision=1,
            status=PlanStatus.DRAFT,
            revisions=(revision,),
            step_states=tuple(
                StepState(step_id=item.step_id) for item in candidate.steps
            ),
            created_at=now,
            updated_at=now,
        )
        self.plans[plan.plan_id] = plan
        return plan

    def get(self, plan_id: str) -> Plan:
        try:
            return self.plans[plan_id]
        except KeyError as exc:
            raise ValueError(f"Unknown Plan: {plan_id}") from exc

    def list_plans(self, *, goal_id: str | None = None) -> list[Plan]:
        return [
            plan
            for plan in sorted(self.plans.values(), key=lambda item: item.plan_id)
            if goal_id is None or plan.goal_id == goal_id
        ]

    def activate(
        self,
        plan_id: str,
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        if plan.status != PlanStatus.DRAFT:
            raise ValueError("Only a draft Plan can be activated")
        if any(
            item.status in {PlanStatus.ACTIVE, PlanStatus.PAUSED}
            and item.goal_id == plan.goal_id
            for item in self.plans.values()
        ):
            raise ValueError("Goal already has an active Plan")
        plan = self._plan_transition(
            plan, PlanStatus.ACTIVE, "plan_activated", (), event_id, event_sequence
        )
        plan = self._refresh_actionable(plan, event_id, event_sequence)
        self.plans[plan_id] = plan
        return plan

    def pause(
        self,
        plan_id: str,
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        if plan.status != PlanStatus.ACTIVE:
            return plan
        plan = self._plan_transition(
            plan,
            PlanStatus.PAUSED,
            "goal_not_active",
            (),
            event_id,
            event_sequence,
        )
        self.plans[plan_id] = plan
        return plan

    def resume(
        self,
        plan_id: str,
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        if plan.status != PlanStatus.PAUSED:
            return plan
        plan = self._plan_transition(
            plan,
            PlanStatus.ACTIVE,
            "goal_reactivated",
            (),
            event_id,
            event_sequence,
        )
        plan = self._refresh_actionable(plan, event_id, event_sequence)
        self.plans[plan_id] = plan
        return plan

    def revise(
        self,
        plan_id: str,
        candidate: PlanCandidate,
        *,
        expected_revision: int,
        reason_code: str,
        actor_id: str,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        if candidate.plan_id != plan_id or candidate.goal_id != plan.goal_id:
            raise ValueError("Plan revision cannot change Plan or Goal identity")
        if plan.revision != expected_revision:
            raise ValueError("Plan revision conflict")
        if plan.status in {
            PlanStatus.COMPLETED,
            PlanStatus.FAILED,
            PlanStatus.ABANDONED,
        }:
            raise ValueError("Terminal Plan cannot be revised")
        now = _now()
        old_states = {item.step_id: item for item in plan.step_states}
        old_definitions = {item.step_id: item for item in plan.current_revision.steps}
        invalidated = {
            step.step_id
            for step in candidate.steps
            if old_definitions.get(step.step_id) != step
        }
        while True:
            dependents = {
                step.step_id
                for step in candidate.steps
                if set(step.dependency_ids) & invalidated
            }
            expanded = invalidated | dependents
            if expanded == invalidated:
                break
            invalidated = expanded
        states = tuple(
            old_states[step.step_id]
            if step.step_id in old_states
            and old_states[step.step_id].status == StepStatus.COMPLETED
            and step.step_id not in invalidated
            else StepState(step_id=step.step_id)
            for step in candidate.steps
        )
        closed_revision = plan.current_revision.model_copy(
            update={"final_step_states": plan.step_states}
        )
        revision = PlanRevision(
            revision=plan.revision + 1,
            reason_code=reason_code,
            actor_id=actor_id,
            success_condition=candidate.success_condition,
            failure_condition=candidate.failure_condition,
            abandonment_condition=candidate.abandonment_condition,
            steps=candidate.steps,
            created_at=now,
        )
        updated = plan.model_copy(
            update={
                "revision": revision.revision,
                "revisions": (*plan.revisions[:-1], closed_revision, revision),
                "step_states": states,
                "updated_at": now,
            }
        )
        if updated.status == PlanStatus.ACTIVE:
            updated = self._refresh_actionable(updated, event_id, event_sequence)
        self.plans[plan_id] = Plan.model_validate(updated.model_dump())
        return self.plans[plan_id]

    def start_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        if plan.status != PlanStatus.ACTIVE:
            raise ValueError("Step can start only in an active Plan")
        state = plan.step_state(step_id)
        if state.status not in {StepStatus.READY, StepStatus.WAITING_RETRY}:
            raise ValueError("Step is not actionable")
        if (
            state.status == StepStatus.WAITING_RETRY
            and state.retry_at is not None
            and state.retry_at > _now()
        ):
            raise ValueError("Step retry is not due")
        updated_state = self._step_transition(
            state.model_copy(
                update={
                    "attempt_count": state.attempt_count + 1,
                    "started_at": _now(),
                    "retry_at": None,
                }
            ),
            StepStatus.IN_PROGRESS,
            "step_started",
            event_id,
            event_sequence,
        )
        return self._replace_state(plan, updated_state)

    def complete_step(
        self,
        plan_id: str,
        step_id: str,
        evidence: tuple[EvidenceReference, ...],
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        state = plan.step_state(step_id)
        if state.status != StepStatus.IN_PROGRESS:
            raise ValueError("Only an in-progress Step can be completed")
        definition = plan.step_definition(step_id)
        _validate_completion_evidence(definition, evidence)
        completing_plan = all(
            item.step_id == step_id or item.status == StepStatus.COMPLETED
            for item in plan.step_states
        )
        if completing_plan:
            existing_types = {
                item.evidence_type
                for step_state in plan.step_states
                for item in step_state.evidence
            }
            evidence_types = existing_types | {item.evidence_type for item in evidence}
            required = set(
                plan.current_revision.success_condition.required_evidence_types
            )
            if not required.issubset(evidence_types):
                raise ValueError("Plan success condition lacks required evidence")
        updated_state = self._step_transition(
            state.model_copy(update={"evidence": evidence, "completed_at": _now()}),
            StepStatus.COMPLETED,
            "evidence_verified",
            event_id,
            event_sequence,
        )
        plan = self._replace_state(plan, updated_state)
        plan = self._refresh_actionable(plan, event_id, event_sequence)
        if all(item.status == StepStatus.COMPLETED for item in plan.step_states):
            refs = tuple(
                item.reference for state in plan.step_states for item in state.evidence
            )
            plan = self._plan_transition(
                plan,
                PlanStatus.COMPLETED,
                "success_condition_verified",
                refs,
                event_id,
                event_sequence,
            )
            self.plans[plan_id] = plan
        return plan

    def fail_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        reason_code: str,
        evidence: tuple[EvidenceReference, ...] = (),
        event_id: str | None = None,
        event_sequence: int | None = None,
        now: datetime | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        state = plan.step_state(step_id)
        if state.status != StepStatus.IN_PROGRESS:
            raise ValueError("Only an in-progress Step can fail")
        definition = plan.step_definition(step_id)
        current_time = now or _now()
        if state.attempt_count < definition.retry.max_attempts:
            updated_state = self._step_transition(
                state.model_copy(
                    update={
                        "evidence": evidence,
                        "retry_at": current_time
                        + timedelta(seconds=definition.retry.backoff_seconds),
                    }
                ),
                StepStatus.WAITING_RETRY,
                reason_code,
                event_id,
                event_sequence,
            )
            return self._replace_state(plan, updated_state)
        required = set(plan.current_revision.failure_condition.required_evidence_types)
        if not required.issubset({item.evidence_type for item in evidence}):
            raise ValueError("Plan failure condition lacks required evidence")
        updated_state = self._step_transition(
            state.model_copy(update={"evidence": evidence}),
            StepStatus.FAILED,
            reason_code,
            event_id,
            event_sequence,
        )
        plan = self._replace_state(plan, updated_state)
        refs = tuple(item.reference for item in evidence)
        plan = self._plan_transition(
            plan,
            PlanStatus.FAILED,
            "step_attempts_exhausted",
            refs,
            event_id,
            event_sequence,
        )
        self.plans[plan_id] = plan
        return plan

    def abandon(
        self,
        plan_id: str,
        evidence: tuple[EvidenceReference, ...],
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Plan:
        plan = self.get(plan_id)
        if plan.status not in {
            PlanStatus.DRAFT,
            PlanStatus.ACTIVE,
            PlanStatus.PAUSED,
        }:
            raise ValueError("Only a non-terminal Plan can be abandoned")
        required = set(
            plan.current_revision.abandonment_condition.required_evidence_types
        )
        if not required.issubset({item.evidence_type for item in evidence}):
            raise ValueError("Plan abandonment condition lacks required evidence")
        plan = self._plan_transition(
            plan,
            PlanStatus.ABANDONED,
            "abandonment_condition_verified",
            tuple(item.reference for item in evidence),
            event_id,
            event_sequence,
        )
        self.plans[plan_id] = plan
        return plan

    def actionable_steps(self) -> list[tuple[Plan, StepDefinition, StepState]]:
        result: list[tuple[Plan, StepDefinition, StepState]] = []
        for plan in self.list_plans():
            if plan.status != PlanStatus.ACTIVE:
                continue
            for definition in plan.current_revision.steps:
                state = plan.step_state(definition.step_id)
                if state.status == StepStatus.READY:
                    result.append((plan, definition, state))
        return result

    def action_candidate(self, plan_id: str, step_id: str) -> ActionCandidate:
        plan = self.get(plan_id)
        state = plan.step_state(step_id)
        if plan.status != PlanStatus.ACTIVE or state.status != StepStatus.READY:
            raise ValueError(
                "Only a current actionable Step can become an ActionCandidate"
            )
        step = plan.step_definition(step_id)
        return ActionCandidate(
            candidate_id=f"plan:{plan_id}:{plan.revision}:{step_id}",
            candidate_type=step.action_type,
            proposed_action=step.action_code,
            parameters=dict(step.parameters),
            prerequisites=tuple(
                f"step:{plan.plan_id}:{plan.revision}:{item}:completed"
                for item in step.dependency_ids
            ),
            predicted_outcomes=(),
            uncertainty=0.5,
            estimated_cost=0.5,
            estimated_risk=0.5,
            value_effects={},
            appraisal_contributions={},
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            step_id=step.step_id,
        )

    def validate_candidate(self, candidate: ActionCandidate) -> None:
        if candidate.plan_id is None:
            return
        plan = self.get(candidate.plan_id)
        if candidate.plan_revision != plan.revision or candidate.step_id is None:
            raise ValueError("ActionCandidate references a stale Plan revision")
        expected = self.action_candidate(plan.plan_id, candidate.step_id)
        if candidate != expected:
            raise ValueError(
                "ActionCandidate does not match the current actionable Step"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": [item.model_dump(mode="json") for item in self.list_plans()],
        }

    def restore(self, value: object) -> None:
        if value is None:
            self.plans = {}
            return
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or not isinstance(value.get("records"), list)
        ):
            raise ValueError("Invalid Plan state")
        records = [Plan.model_validate(item) for item in value["records"]]
        if len({item.plan_id for item in records}) != len(records):
            raise ValueError("Duplicate Plan ID in state")
        self.plans = {item.plan_id: item for item in records}

    def _replace_state(self, plan: Plan, state: StepState) -> Plan:
        updated = plan.model_copy(
            update={
                "step_states": tuple(
                    state if item.step_id == state.step_id else item
                    for item in plan.step_states
                ),
                "updated_at": _now(),
            }
        )
        self.plans[plan.plan_id] = Plan.model_validate(updated.model_dump())
        return self.plans[plan.plan_id]

    def _refresh_actionable(
        self, plan: Plan, event_id: str | None, event_sequence: int | None
    ) -> Plan:
        completed = {
            item.step_id
            for item in plan.step_states
            if item.status == StepStatus.COMPLETED
        }
        states: list[StepState] = []
        for state in plan.step_states:
            definition = plan.step_definition(state.step_id)
            should_be_ready = state.status == StepStatus.PENDING and set(
                definition.dependency_ids
            ).issubset(completed)
            states.append(
                self._step_transition(
                    state,
                    StepStatus.READY,
                    "dependencies_satisfied",
                    event_id,
                    event_sequence,
                )
                if should_be_ready
                else state
            )
        updated = plan.model_copy(
            update={"step_states": tuple(states), "updated_at": _now()}
        )
        self.plans[plan.plan_id] = Plan.model_validate(updated.model_dump())
        return self.plans[plan.plan_id]

    @staticmethod
    def _step_transition(
        state: StepState,
        status: StepStatus,
        reason_code: str,
        event_id: str | None,
        event_sequence: int | None,
    ) -> StepState:
        transition = StepTransition(
            transition_id=str(uuid4()),
            from_status=state.status,
            to_status=status,
            reason_code=reason_code,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
        )
        return state.model_copy(
            update={"status": status, "transitions": (*state.transitions, transition)}
        )

    @staticmethod
    def _plan_transition(
        plan: Plan,
        status: PlanStatus,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> Plan:
        transition = PlanTransition(
            transition_id=str(uuid4()),
            from_status=plan.status,
            to_status=status,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
        )
        return plan.model_copy(
            update={
                "status": status,
                "transitions": (*plan.transitions, transition),
                "updated_at": transition.created_at,
            }
        )


def parse_plan_candidate(value: str | dict[str, Any]) -> PlanCandidate:
    """Accept a JSON object only; markdown and unknown/private fields fail closed."""
    if isinstance(value, str):
        payload = json.loads(value)
    else:
        payload = value
    if not isinstance(payload, dict):
        raise ValueError("Plan candidate must be a JSON object")
    _reject_private_fields(payload)
    return PlanCandidate.model_validate(payload)


def _validate_step_dag(steps: tuple[StepDefinition, ...]) -> None:
    identifiers = [item.step_id for item in steps]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Plan Step IDs must be unique")
    known = set(identifiers)
    for step in steps:
        unknown = set(step.dependency_ids) - known
        if unknown:
            raise ValueError(f"Unknown Step dependency: {sorted(unknown)[0]}")
    visiting: set[str] = set()
    visited: set[str] = set()
    dependencies = {item.step_id: item.dependency_ids for item in steps}

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("Plan Step dependency cycle detected")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency_id in dependencies[step_id]:
            visit(dependency_id)
        visiting.remove(step_id)
        visited.add(step_id)

    for identifier in identifiers:
        visit(identifier)


def _validate_completion_evidence(
    definition: StepDefinition, evidence: tuple[EvidenceReference, ...]
) -> None:
    if len(evidence) < definition.verification.minimum_evidence_count:
        raise ValueError("Step completion requires verification evidence")
    if any(
        item.observation_code != definition.expected_observation.observation_code
        for item in evidence
    ):
        raise ValueError("Step evidence does not match expected observation")
    evidence_types = {item.evidence_type for item in evidence}
    required = set(definition.expected_observation.evidence_types) | set(
        definition.verification.required_evidence_types
    )
    if not required.issubset(evidence_types):
        raise ValueError("Step completion lacks required evidence types")


def _unique_codes(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)) or any(not item for item in values):
        raise ValueError(f"{label} must be non-empty and unique")


def _reject_private_fields(value: Any) -> None:
    forbidden = {
        "hidden_thought",
        "prompt",
        "raw_prompt",
        "reasoning",
        "chain_of_thought",
        "prose",
        "tool_output",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                raise ValueError("Plan candidate contains a private or free-form field")
            _reject_private_fields(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_private_fields(item)


def _validate_structured_parameters(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("Plan parameter keys must be bounded strings")
            _validate_structured_parameters(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_structured_parameters(item)
    elif isinstance(value, str):
        if (
            not value
            or len(value) > 256
            or any(character.isspace() for character in value)
        ):
            raise ValueError("Plan string parameters must be structured tokens")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Plan numeric parameters must be finite")
    elif value is not None and not isinstance(value, bool | int | float):
        raise ValueError("Plan parameters must contain JSON scalar values only")


def _now() -> datetime:
    return datetime.now(UTC)
