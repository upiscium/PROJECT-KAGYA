"""Persistent, budgeted wake-ups for the single subject runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Event, RLock, Thread
import time
from typing import Any, Callable, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.decision import DecisionStatus
from kagya.motivation import (
    CommitmentFulfillability,
    CommitmentStatus,
    GoalStatus,
    MotivationStatus,
    IntrinsicGoalStatus,
)
from kagya.planning import PlanStatus, StepStatus
from kagya.outbox import (
    AcknowledgmentStatus,
    DeliveryStatus,
    OutboxMessageKind,
    OutboxReferences,
    OutboxUrgency,
)
from kagya.runtime.agent_runtime import (
    AgentEventType,
    AgentRuntime,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
)


SCHEDULER_STATE_KEY = "subject_scheduler"


class WakeUpKind(StrEnum):
    GOAL_DEADLINE = "goal_deadline"
    COMMITMENT_DEADLINE = "commitment_deadline"
    COMMITMENT_REEVALUATION = "commitment_reevaluation"
    GOAL_REEVALUATION = "goal_reevaluation"
    NEEDS_INFORMATION = "needs_information"
    DECISION_OUTCOME = "decision_outcome"
    ACTION_TIMEOUT = "action_timeout"
    ACTION_RETRY = "action_retry"
    ACTION_DECISION = "action_decision"
    ACTION_INTENT = "action_intent"
    ACTION_EXECUTION = "action_execution"
    OUTBOX_DEADLINE = "outbox_deadline"
    SLEEP_CONSOLIDATION = "sleep_consolidation"
    OPERATOR = "operator"
    STEP_RETRY = "step_retry"
    STEP_TIMEOUT = "step_timeout"
    MOTIVATION_REEVALUATION = "motivation_reevaluation"
    MOTIVATION_DECAY = "motivation_decay"
    INTRINSIC_DELIBERATION = "intrinsic_deliberation"
    PLAN_GENERATION = "plan_generation"
    INTRINSIC_ADOPTION = "intrinsic_adoption"


class ScheduleStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CycleResult(StrEnum):
    PROCESSED = "processed"
    NO_ACTION = "no_action"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STOPPED = "stopped"


class WakeUpSchedule(BaseModel):
    """A single immutable wake-up occurrence persisted in subject state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    schedule_id: str = Field(min_length=1, max_length=256)
    kind: WakeUpKind
    wake_at: datetime
    target_id: str | None = Field(default=None, max_length=512)
    status: ScheduleStatus = ScheduleStatus.PENDING
    estimated_inferences: int = Field(default=0, ge=0)
    created_at: datetime
    completed_at: datetime | None = None
    outcome: str | None = Field(default=None, max_length=128)
    causation_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_schedule(self) -> "WakeUpSchedule":
        if self.schema_version != 1:
            raise ValueError("Unsupported wake-up schedule schema version")
        if self.wake_at.tzinfo is None or self.created_at.tzinfo is None:
            raise ValueError("Wake-up timestamps must include a timezone")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("Wake-up completion timestamp must include a timezone")
        if self.status == ScheduleStatus.COMPLETED and self.completed_at is None:
            raise ValueError("Completed wake-up requires completed_at")
        return self


@dataclass(frozen=True)
class SchedulerBudget:
    max_events: int
    max_inferences: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        if (
            self.max_events <= 0
            or self.max_inferences < 0
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("Scheduler budgets must be positive")


@dataclass(frozen=True)
class SchedulerCycle:
    result: CycleResult
    processed: int
    inferences: int
    deferred: int
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class SchedulerStatus:
    state: str
    last_cycle: SchedulerCycle | None
    next_wake_at: str | None
    pending_count: int


class SchedulerTelemetry(Protocol):
    def counter(self, name: str, amount: float = 1.0, **labels: str) -> None: ...

    def gauge(self, name: str, value: float, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


class SubjectScheduler:
    """Discover and process due internal work without performing external effects."""

    def __init__(
        self,
        runtime: AgentRuntime,
        main_loop: Any,
        *,
        budget: SchedulerBudget,
        reevaluation_interval_seconds: float,
        telemetry: SchedulerTelemetry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if reevaluation_interval_seconds <= 0:
            raise ValueError("Reevaluation interval must be positive")
        self.runtime = runtime
        self.main_loop = main_loop
        self.budget = budget
        self.reevaluation_interval_seconds = reevaluation_interval_seconds
        self.telemetry = telemetry
        self.clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._accepting = True
        self._last_cycle: SchedulerCycle | None = None
        self._notifier: Callable[[], None] | None = None

    def set_notifier(self, notifier: Callable[[], None]) -> None:
        self._notifier = notifier

    def schedule(
        self,
        kind: WakeUpKind,
        wake_at: datetime,
        *,
        target_id: str | None = None,
        schedule_id: str | None = None,
        estimated_inferences: int = 0,
        causation_id: str | None = None,
    ) -> WakeUpSchedule:
        if wake_at.tzinfo is None:
            raise ValueError("Wake-up time must include a timezone")
        with self._lock:
            if not self._accepting:
                raise AgentRuntimeStopped("Subject scheduler is stopping")
        identifier = schedule_id or str(uuid4())
        value = WakeUpSchedule(
            schedule_id=identifier,
            kind=kind,
            wake_at=wake_at,
            target_id=target_id,
            estimated_inferences=estimated_inferences,
            created_at=self.clock(),
            causation_id=causation_id,
        )

        def persist() -> WakeUpSchedule:
            schedules = self._stored_schedules()
            if any(item.schedule_id == identifier for item in schedules):
                raise ValueError(f"Schedule already exists: {identifier}")
            schedules.append(value)
            self._save_schedules(schedules)
            return value

        persisted = self.runtime.execute(
            AgentEventType.AUTONOMY_SCHEDULE,
            source="scheduler.schedule",
            handler=persist,
            payload={"schedule_id": identifier, "kind": kind.value},
            causation_id=causation_id,
            correlation_id=identifier,
        ).value
        if self._notifier is not None:
            self._notifier()
        return persisted

    def run_cycle(self, now: datetime | None = None) -> SchedulerCycle:
        started_at = now or self.clock()
        if started_at.tzinfo is None:
            raise ValueError("Cycle time must include a timezone")
        started_clock = time.monotonic()
        with self._lock:
            if not self._accepting:
                return self._finish_cycle(
                    CycleResult.STOPPED,
                    0,
                    0,
                    0,
                    started_at,
                    self.clock(),
                    started_clock,
                )
        due = self._due_schedules(started_at)
        if not due:
            return self._finish_cycle(
                CycleResult.NO_ACTION, 0, 0, 0, started_at, self.clock(), started_clock
            )

        processed = 0
        inferences = 0
        dynamics = getattr(self.main_loop, "motivation_dynamics", None)
        goal_budget = getattr(dynamics, "max_goal_proposals_per_cycle", 0)
        for schedule in due:
            if (
                processed >= self.budget.max_events
                or time.monotonic() - started_clock >= self.budget.max_wall_seconds
            ):
                break
            if inferences + schedule.estimated_inferences > self.budget.max_inferences:
                continue
            try:

                def process(
                    item: WakeUpSchedule = schedule,
                    remaining_goals: int = goal_budget,
                ) -> tuple[str, int]:
                    return self._process(item, remaining_goals)

                event_type = {
                    WakeUpKind.MOTIVATION_REEVALUATION: AgentEventType.INTRINSIC_GOAL_PROPOSE,
                    WakeUpKind.INTRINSIC_DELIBERATION: AgentEventType.INTRINSIC_GOAL_DELIBERATE,
                    WakeUpKind.PLAN_GENERATION: AgentEventType.PLAN_GENERATE,
                    WakeUpKind.INTRINSIC_ADOPTION: AgentEventType.INTRINSIC_GOAL_ADOPT,
                    WakeUpKind.ACTION_DECISION: AgentEventType.DECISION_UPDATE,
                    WakeUpKind.ACTION_INTENT: AgentEventType.ACTION_INTENT,
                    WakeUpKind.ACTION_EXECUTION: AgentEventType.ACTION_EXECUTE,
                    WakeUpKind.ACTION_RETRY: AgentEventType.ACTION_EXECUTE,
                }.get(schedule.kind, AgentEventType.AUTONOMY_WAKE)
                outcome = self.runtime.execute(
                    event_type,
                    source="scheduler.wake",
                    handler=process,
                    payload={
                        "schedule_id": schedule.schedule_id,
                        "kind": schedule.kind.value,
                    },
                    causation_id=schedule.causation_id,
                    correlation_id=schedule.schedule_id,
                ).value
            except (AgentRuntimeQueueFull, AgentRuntimeStopped):
                break
            processed += 1
            inferences += schedule.estimated_inferences
            goal_budget = max(0, goal_budget - outcome[1])
        deferred = len(due) - processed
        result = CycleResult.BUDGET_EXHAUSTED if deferred else CycleResult.PROCESSED
        return self._finish_cycle(
            result,
            processed,
            inferences,
            deferred,
            started_at,
            self.clock(),
            started_clock,
        )

    def status(self) -> SchedulerStatus:
        schedules = self._all_schedules()
        pending = [item for item in schedules if item.status == ScheduleStatus.PENDING]
        next_wake = min((item.wake_at for item in pending), default=None)
        with self._lock:
            state = "accepting" if self._accepting else "stopped"
            last_cycle = self._last_cycle
        return SchedulerStatus(
            state=state,
            last_cycle=last_cycle,
            next_wake_at=None if next_wake is None else next_wake.isoformat(),
            pending_count=len(pending),
        )

    def stop_accepting(self) -> None:
        with self._lock:
            self._accepting = False
        if self._notifier is not None:
            self._notifier()

    def _due_schedules(self, now: datetime) -> list[WakeUpSchedule]:
        return sorted(
            (
                item
                for item in self._all_schedules()
                if item.status == ScheduleStatus.PENDING and item.wake_at <= now
            ),
            key=lambda item: (item.wake_at, item.schedule_id),
        )

    def _all_schedules(self) -> list[WakeUpSchedule]:
        stored = self._stored_schedules()
        known = {item.schedule_id for item in stored}
        return stored + [
            item for item in self._derived_schedules() if item.schedule_id not in known
        ]

    def _stored_schedules(self) -> list[WakeUpSchedule]:
        raw = self.main_loop.persistent_state.extensions.get(SCHEDULER_STATE_KEY)
        if raw is None:
            return []
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("Invalid subject scheduler state")
        schedules = raw.get("schedules")
        if not isinstance(schedules, list):
            raise ValueError("Invalid subject scheduler schedule list")
        return [WakeUpSchedule.model_validate(item) for item in schedules]

    def _save_schedules(self, schedules: list[WakeUpSchedule]) -> None:
        self.main_loop.persistent_state.extensions[SCHEDULER_STATE_KEY] = {
            "schema_version": 1,
            "schedules": [item.model_dump(mode="json") for item in schedules],
        }

    def _derived_schedules(self) -> list[WakeUpSchedule]:
        schedules: list[WakeUpSchedule] = []
        for goal in self.main_loop.goal_manager.list_goals():
            if goal.status in {
                GoalStatus.COMPLETED,
                GoalStatus.ABANDONED,
                GoalStatus.FAILED,
            }:
                continue
            intrinsic_status = getattr(goal, "intrinsic_status", None)
            if intrinsic_status in {
                IntrinsicGoalStatus.PROPOSAL,
                IntrinsicGoalStatus.DEFERRED,
            }:
                wake_at = (
                    datetime.fromisoformat(goal.updated_at)
                    if intrinsic_status == IntrinsicGoalStatus.PROPOSAL
                    else datetime.fromisoformat(goal.updated_at)
                    + timedelta(seconds=self.reevaluation_interval_seconds)
                )
                schedules.append(
                    self._derived(
                        f"intrinsic-deliberation:{goal.goal_id}:{wake_at.isoformat()}",
                        WakeUpKind.INTRINSIC_DELIBERATION,
                        wake_at,
                        goal.goal_id,
                    )
                )
            elif intrinsic_status == IntrinsicGoalStatus.ENDORSED:
                plans = self.main_loop.plan_store.list_plans(goal_id=goal.goal_id)
                intrinsic_kind = (
                    WakeUpKind.PLAN_GENERATION
                    if not plans
                    else WakeUpKind.INTRINSIC_ADOPTION
                )
                schedules.append(
                    self._derived(
                        f"{intrinsic_kind.value}:{goal.goal_id}:{goal.updated_at}",
                        intrinsic_kind,
                        datetime.fromisoformat(goal.updated_at),
                        goal.goal_id,
                    )
                )
            if goal.deadline is not None:
                schedules.append(
                    self._derived(
                        f"goal-deadline:{goal.goal_id}:{goal.deadline}",
                        WakeUpKind.GOAL_DEADLINE,
                        datetime.fromisoformat(goal.deadline),
                        goal.goal_id,
                    )
                )
            kind = (
                WakeUpKind.NEEDS_INFORMATION
                if goal.needs_information
                else WakeUpKind.GOAL_REEVALUATION
                if goal.status == GoalStatus.SUSPENDED
                else None
            )
            if kind is not None:
                wake_at = datetime.fromisoformat(goal.updated_at) + timedelta(
                    seconds=self.reevaluation_interval_seconds
                )
                schedules.append(
                    self._derived(
                        f"{kind.value}:{goal.goal_id}:{wake_at.isoformat()}",
                        kind,
                        wake_at,
                        goal.goal_id,
                    )
                )
        commitments = [
            *self.main_loop.commitment_store.list_commitments(CommitmentStatus.ACTIVE),
            *self.main_loop.commitment_store.list_commitments(
                CommitmentStatus.RENEGOTIATING
            ),
        ]
        for commitment in commitments:
            if commitment.deadline is not None:
                schedules.append(
                    self._derived(
                        f"commitment-deadline:{commitment.commitment_id}:{commitment.deadline}",
                        WakeUpKind.COMMITMENT_DEADLINE,
                        datetime.fromisoformat(commitment.deadline),
                        commitment.commitment_id,
                    )
                )
            if commitment.fulfillability in {
                CommitmentFulfillability.AT_RISK,
                CommitmentFulfillability.IMPOSSIBLE,
            }:
                wake_at = datetime.fromisoformat(commitment.updated_at) + timedelta(
                    seconds=self.reevaluation_interval_seconds
                )
                schedules.append(
                    self._derived(
                        f"commitment-reevaluation:{commitment.commitment_id}:{wake_at.isoformat()}",
                        WakeUpKind.COMMITMENT_REEVALUATION,
                        wake_at,
                        commitment.commitment_id,
                    )
                )
        for decision in self.main_loop.decision_store.list_records(
            DecisionStatus.AWAITING_OUTCOME
        ):
            wake_at = datetime.fromisoformat(decision.updated_at) + timedelta(
                seconds=self.reevaluation_interval_seconds
            )
            schedules.append(
                self._derived(
                    f"decision-outcome:{decision.decision_id}:{wake_at.isoformat()}",
                    WakeUpKind.DECISION_OUTCOME,
                    wake_at,
                    decision.decision_id,
                )
            )
        action_execution = getattr(self.main_loop, "action_execution", None)
        if action_execution is not None:
            intents = action_execution.list_intents()
            for decision in self.main_loop.decision_store.list_records(
                DecisionStatus.AWAITING_OUTCOME
            ):
                selected = next(
                    item.candidate
                    for item in decision.considered_candidates
                    if item.candidate.candidate_id == decision.selected_candidate_id
                )
                if (
                    set(selected.parameters) == {"action"}
                    and isinstance(selected.parameters["action"], dict)
                    and not any(
                        item.provenance.decision_id == decision.decision_id
                        for item in intents
                    )
                ):
                    schedules.append(
                        self._derived(
                            f"action-intent:{decision.decision_id}",
                            WakeUpKind.ACTION_INTENT,
                            datetime.fromisoformat(decision.updated_at),
                            decision.decision_id,
                        )
                    )
            for intent in action_execution.list_intents():
                if intent.status.value == "approved":
                    schedules.append(
                        self._derived(
                            f"action-execution:{intent.intent_id}:{intent.updated_at.isoformat()}",
                            WakeUpKind.ACTION_EXECUTION,
                            intent.updated_at,
                            intent.intent_id,
                        )
                    )
                    schedules.append(
                        self._derived(
                            f"action-timeout:{intent.intent_id}:{intent.deadline_at.isoformat()}",
                            WakeUpKind.ACTION_TIMEOUT,
                            intent.deadline_at,
                            intent.intent_id,
                        )
                    )
                elif intent.status.value == "retry_pending" and intent.retry_at is not None:
                    schedules.append(
                        self._derived(
                            f"action-retry:{intent.intent_id}:{intent.retry_at.isoformat()}",
                            WakeUpKind.ACTION_RETRY,
                            intent.retry_at,
                            intent.intent_id,
                        )
                    )
        plan_store = getattr(self.main_loop, "plan_store", None)
        if plan_store is not None:
            for plan in plan_store.list_plans():
                if plan.status != PlanStatus.ACTIVE:
                    continue
                for definition in plan.current_revision.steps:
                    state = plan.step_state(definition.step_id)
                    target = f"{plan.plan_id}/{definition.step_id}"
                    candidate = self.main_loop.plan_store.action_candidate(
                        plan.plan_id, definition.step_id
                    ) if state.status == StepStatus.READY else None
                    governed_action = (
                        candidate is not None
                        and set(candidate.parameters) == {"action"}
                        and isinstance(candidate.parameters["action"], dict)
                        and set(candidate.parameters["action"])
                        == {"tool_name", "arguments"}
                    )
                    has_decision = any(
                        candidate.candidate.plan_id == plan.plan_id
                        and candidate.candidate.plan_revision == plan.revision
                        and candidate.candidate.step_id == definition.step_id
                        for decision in self.main_loop.decision_store.list_records()
                        for candidate in decision.considered_candidates
                    )
                    if (
                        action_execution is not None
                        and governed_action
                        and not has_decision
                    ):
                        schedules.append(
                            self._derived(
                                f"action-decision:{uuid5(NAMESPACE_URL, target)}",
                                WakeUpKind.ACTION_DECISION,
                                plan.updated_at,
                                target,
                            )
                        )
                    if (
                        state.status == StepStatus.WAITING_RETRY
                        and state.retry_at is not None
                    ):
                        schedules.append(
                            self._derived(
                                f"step-retry:{target}:{state.retry_at.isoformat()}",
                                WakeUpKind.STEP_RETRY,
                                state.retry_at,
                                target,
                            )
                        )
                    if (
                        state.status == StepStatus.IN_PROGRESS
                        and state.started_at is not None
                    ):
                        timeout_at = state.started_at + timedelta(
                            seconds=definition.timeout_seconds
                        )
                        schedules.append(
                            self._derived(
                                f"step-timeout:{target}:{timeout_at.isoformat()}",
                                WakeUpKind.STEP_TIMEOUT,
                                timeout_at,
                                target,
                            )
                        )
        schedules.extend(self._derived_motivation_schedules())
        outbox = getattr(self.main_loop, "outbox", None)
        if outbox is not None:
            for message in outbox.list_messages():
                if (
                    message.delivery_status in {DeliveryStatus.PENDING, DeliveryStatus.FAILED}
                    and message.acknowledgment_status == AcknowledgmentStatus.UNACKNOWLEDGED
                ):
                    schedules.append(
                        self._derived(
                            f"outbox-delivery:{message.message_id}:{message.not_before.isoformat()}",
                            WakeUpKind.OUTBOX_DEADLINE,
                            message.not_before,
                            message.message_id,
                        )
                    )
        return schedules

    def _derived_motivation_schedules(self) -> list[WakeUpSchedule]:
        schedules: list[WakeUpSchedule] = []

        def add(target: str, changed_at: str, *, suffix: str = "") -> None:
            wake_at = datetime.fromisoformat(changed_at) + timedelta(
                seconds=self.reevaluation_interval_seconds
            )
            schedules.append(
                self._derived(
                    f"motivation-reevaluation:{target}:{changed_at}{suffix}",
                    WakeUpKind.MOTIVATION_REEVALUATION,
                    wake_at,
                    target,
                )
            )

        dynamics = getattr(self.main_loop, "motivation_dynamics", None)
        if dynamics is None:
            return schedules
        for record in dynamics.list_records():
            if record.status != MotivationStatus.ACTIVE:
                continue
            review_at = (
                datetime.fromisoformat(record.next_review_at)
                if record.next_review_at is not None
                else datetime.fromisoformat(record.updated_at)
                + timedelta(seconds=self.reevaluation_interval_seconds)
            )
            schedules.append(
                self._derived(
                    f"motivation-reevaluation:{record.motivation_id}:{review_at.isoformat()}",
                    WakeUpKind.MOTIVATION_REEVALUATION,
                    review_at,
                    record.target_ref,
                )
            )
            decay_at = datetime.fromisoformat(record.updated_at) + timedelta(
                seconds=self.reevaluation_interval_seconds
            )
            schedules.append(
                self._derived(
                    f"motivation-decay:{record.motivation_id}:{decay_at.isoformat()}",
                    WakeUpKind.MOTIVATION_DECAY,
                    decay_at,
                    record.motivation_id,
                )
            )

        narrative = getattr(self.main_loop, "narrative_self", None)
        if narrative is not None:
            for episode in narrative.episodes.values():
                if episode.unresolved_tension >= 0.4:
                    add(f"narrative:{episode.episode_id}", episode.created_at)
            for conflict in narrative.conflicts.values():
                if conflict.resolved_at is None:
                    add(f"self-conflict:{conflict.conflict_id}", conflict.created_at)
            for projection in narrative.future_self.values():
                if projection.gap > 0.0:
                    add(
                        f"future-self:{projection.projection_id}", projection.updated_at
                    )

        relationships = getattr(self.main_loop, "relationship_store", None)
        if relationships is not None:
            for relationship in relationships.list_relationships():
                if (
                    relationship.unresolved_matter_refs
                    or relationship.conflict_refs
                    or relationship.commitment_refs
                ):
                    add(
                        f"relationship:{relationship.relationship_id}",
                        relationship.updated_at,
                    )

        values = getattr(self.main_loop, "value_system", None)
        if values is not None:
            for tradeoff in values.tradeoffs:
                if not tradeoff.conflict_names:
                    continue
                for value_id in tradeoff.value_ids:
                    add(
                        f"value:{value_id}",
                        tradeoff.created_at,
                        suffix=f":{tradeoff.tradeoff_id}",
                    )

        self_model = getattr(self.main_loop, "self_model", None)
        if self_model is not None:
            for limitation in self_model.state.known_limitations.values():
                add(
                    f"limitation:{limitation.limitation_id}",
                    self_model.state.updated_at,
                )
            for uncertainty in self_model.state.epistemic_uncertainties.values():
                add(
                    f"uncertainty:{uncertainty.uncertainty_id}",
                    self_model.state.updated_at,
                )
        return schedules

    def _derived(
        self, schedule_id: str, kind: WakeUpKind, wake_at: datetime, target_id: str
    ) -> WakeUpSchedule:
        return WakeUpSchedule(
            schedule_id=schedule_id,
            kind=kind,
            wake_at=wake_at,
            target_id=target_id,
            created_at=wake_at,
        )

    def _process(self, schedule: WakeUpSchedule, goal_budget: int) -> tuple[str, int]:
        stored = self._stored_schedules()
        current = next(
            (item for item in stored if item.schedule_id == schedule.schedule_id), None
        )
        if current is not None and current.status != ScheduleStatus.PENDING:
            return "already_processed", 0
        if current is None:
            stored.append(schedule)

        outcome = "no_action"
        generated_goals = 0
        if schedule.kind in {
            WakeUpKind.GOAL_DEADLINE,
            WakeUpKind.GOAL_REEVALUATION,
            WakeUpKind.NEEDS_INFORMATION,
        }:
            decisions = self.main_loop.reevaluate_goals()
            outcome = (
                "reevaluated"
                if any(item.action.value != "no_action" for item in decisions)
                else "no_action"
            )
        elif schedule.kind == WakeUpKind.COMMITMENT_DEADLINE:
            commitment = self.main_loop.commitment_store.commitments.get(
                schedule.target_id or ""
            )
            if commitment is not None and commitment.status in {
                CommitmentStatus.ACTIVE,
                CommitmentStatus.RENEGOTIATING,
            }:
                self.main_loop.transition_commitment(
                    commitment.commitment_id,
                    CommitmentStatus.BREACHED,
                    reason="deadline_expired",
                )
                outcome = "commitment_breached"
        elif schedule.kind == WakeUpKind.COMMITMENT_REEVALUATION:
            commitment = self.main_loop.commitment_store.commitments.get(
                schedule.target_id or ""
            )
            if commitment is not None and commitment.status in {
                CommitmentStatus.ACTIVE,
                CommitmentStatus.RENEGOTIATING,
            }:
                self.main_loop.reassess_commitment(
                    commitment.commitment_id,
                    fulfillability=commitment.fulfillability,
                    reason=commitment.fulfillability_reason
                    or "scheduled_fulfillability_reassessment",
                )
                outcome = "commitment_reevaluated"
        elif schedule.kind == WakeUpKind.DECISION_OUTCOME:
            decision = self.main_loop.decision_store.records.get(
                schedule.target_id or ""
            )
            if (
                decision is not None
                and decision.status == DecisionStatus.AWAITING_OUTCOME
            ):
                outcome = "awaiting_outcome"
        elif schedule.kind in {WakeUpKind.STEP_RETRY, WakeUpKind.STEP_TIMEOUT}:
            target = schedule.target_id or ""
            if "/" not in target:
                raise ValueError("Invalid Step wake-up target")
            plan_id, step_id = target.split("/", 1)
            if schedule.kind == WakeUpKind.STEP_RETRY:
                self.main_loop.retry_plan_step(plan_id, step_id)
                outcome = "step_retry_started"
            else:
                self.main_loop.timeout_plan_step(plan_id, step_id)
                outcome = "step_timed_out"
        elif schedule.kind == WakeUpKind.ACTION_DECISION:
            target = schedule.target_id or ""
            if "/" not in target:
                raise ValueError("Invalid action Decision target")
            plan_id, step_id = target.split("/", 1)
            decision = self.main_loop.create_plan_action_decision(plan_id, step_id)
            outcome = f"decision_created:{decision.decision_id}"
        elif schedule.kind == WakeUpKind.ACTION_INTENT:
            execution = getattr(self.main_loop, "action_execution", None)
            if execution is None:
                raise ValueError("Action execution layer is unavailable")
            decision_id = schedule.target_id or ""
            intent = execution.create_from_decision(
                decision_id,
                idempotency_key=f"decision-action:{decision_id}",
            )
            outcome = intent.status.value
        elif schedule.kind == WakeUpKind.ACTION_EXECUTION:
            execution = getattr(self.main_loop, "action_execution", None)
            if execution is None:
                raise ValueError("Action execution layer is unavailable")
            updated = execution.execute(schedule.target_id or "")
            outcome = updated.status.value
        elif schedule.kind in {WakeUpKind.ACTION_TIMEOUT, WakeUpKind.ACTION_RETRY}:
            execution = getattr(self.main_loop, "action_execution", None)
            if execution is None:
                raise ValueError("Action execution layer is unavailable")
            updated = execution.timeout(schedule.target_id or "")
            outcome = updated.status.value
        elif schedule.kind == WakeUpKind.MOTIVATION_REEVALUATION:
            _, goals = self.main_loop.reevaluate_motivation(
                max_goal_proposals=goal_budget
            )
            self.main_loop.schedule_motivation_reviews(
                max(self.clock(), schedule.wake_at)
                + timedelta(seconds=self.reevaluation_interval_seconds)
            )
            generated_goals = len(goals)
            outcome = "reevaluated" if goals else "no_action"
        elif schedule.kind == WakeUpKind.MOTIVATION_DECAY:
            record = self.main_loop.motivation_dynamics.records.get(
                schedule.target_id or ""
            )
            if record is not None and record.status == MotivationStatus.ACTIVE:
                elapsed = max(
                    self.reevaluation_interval_seconds / 3600.0,
                    (
                        self.clock() - datetime.fromisoformat(record.updated_at)
                    ).total_seconds()
                    / 3600.0,
                )
                updated = self.main_loop.decay_motivation_record(
                    record.motivation_id, elapsed
                )
                outcome = "decayed" if updated is not None else "no_action"
        elif schedule.kind == WakeUpKind.INTRINSIC_DELIBERATION:
            decision = self.main_loop.deliberate_intrinsic_goal(
                schedule.target_id or ""
            )
            outcome = decision.action.value
        elif schedule.kind == WakeUpKind.PLAN_GENERATION:
            self.main_loop.generate_intrinsic_plan(schedule.target_id or "")
            outcome = "plan_generated"
        elif schedule.kind == WakeUpKind.INTRINSIC_ADOPTION:
            decision = self.main_loop.activate_endorsed_intrinsic_goal(
                schedule.target_id or ""
            )
            outcome = decision.action.value
        elif schedule.kind == WakeUpKind.OUTBOX_DEADLINE:
            outbox = getattr(self.main_loop, "outbox", None)
            if outbox is None:
                raise ValueError("Proactive outbox is unavailable")
            outcome = "delivered" if outbox.deliver() else "deferred"

        self._enqueue_outcome(schedule, outcome)

        completed = schedule.model_copy(
            update={
                "status": ScheduleStatus.COMPLETED,
                "completed_at": self.clock(),
                "outcome": outcome,
            }
        )
        stored = [
            completed if item.schedule_id == schedule.schedule_id else item
            for item in stored
        ]
        if self._should_recheck(schedule):
            next_wake = self.clock() + timedelta(
                seconds=self.reevaluation_interval_seconds
            )
            stored.append(
                WakeUpSchedule(
                    schedule_id=(
                        f"{schedule.kind.value}:{schedule.target_id}:{next_wake.isoformat()}"
                    ),
                    kind=schedule.kind,
                    wake_at=next_wake,
                    target_id=schedule.target_id,
                    created_at=self.clock(),
                    causation_id=schedule.schedule_id,
                )
            )
        self._save_schedules(stored)
        return outcome, generated_goals

    def _should_recheck(self, schedule: WakeUpSchedule) -> bool:
        if schedule.kind == WakeUpKind.DECISION_OUTCOME:
            decision = self.main_loop.decision_store.records.get(
                schedule.target_id or ""
            )
            return (
                decision is not None
                and decision.status == DecisionStatus.AWAITING_OUTCOME
            )
        if schedule.kind in {
            WakeUpKind.GOAL_REEVALUATION,
            WakeUpKind.NEEDS_INFORMATION,
        }:
            goal = self.main_loop.goal_manager.goals.get(schedule.target_id or "")
            return goal is not None and (
                goal.status == GoalStatus.SUSPENDED or goal.needs_information
            )
        if schedule.kind == WakeUpKind.COMMITMENT_REEVALUATION:
            commitment = self.main_loop.commitment_store.commitments.get(
                schedule.target_id or ""
            )
            return (
                commitment is not None
                and commitment.status
                in {
                    CommitmentStatus.ACTIVE,
                    CommitmentStatus.RENEGOTIATING,
                }
                and commitment.fulfillability
                in {
                    CommitmentFulfillability.AT_RISK,
                    CommitmentFulfillability.IMPOSSIBLE,
                }
            )
        if schedule.kind == WakeUpKind.MOTIVATION_REEVALUATION:
            return False
        if schedule.kind == WakeUpKind.MOTIVATION_DECAY:
            return False
        if schedule.kind == WakeUpKind.OUTBOX_DEADLINE:
            outbox = getattr(self.main_loop, "outbox", None)
            if outbox is None:
                return False
            message = outbox.get(schedule.target_id or "")
            return message.delivery_status in {
                DeliveryStatus.PENDING,
                DeliveryStatus.FAILED,
            }
        return False

    def _enqueue_outcome(self, schedule: WakeUpSchedule, outcome: str) -> None:
        outbox = getattr(self.main_loop, "outbox", None)
        target = schedule.target_id
        if outbox is None or target is None or schedule.kind == WakeUpKind.OUTBOX_DEADLINE:
            return
        kind: OutboxMessageKind | None = None
        title = "Subject state update"
        body = f"Scheduled {schedule.kind.value} work completed with outcome {outcome}."
        urgency = OutboxUrgency.NORMAL
        references = OutboxReferences()
        if schedule.kind == WakeUpKind.NEEDS_INFORMATION:
            kind = OutboxMessageKind.QUESTION
            title = "Information needed for goal"
            body = "Additional information is needed before this goal can continue."
            references = OutboxReferences(goal_id=target)
        elif schedule.kind in {WakeUpKind.GOAL_DEADLINE, WakeUpKind.GOAL_REEVALUATION}:
            kind = OutboxMessageKind.GOAL_STATE
            title = "Goal state changed"
            references = OutboxReferences(goal_id=target)
        elif schedule.kind == WakeUpKind.COMMITMENT_DEADLINE:
            kind = OutboxMessageKind.COMMITMENT_DEADLINE
            title = "Commitment deadline reached"
            urgency = OutboxUrgency.HIGH
            references = OutboxReferences(commitment_id=target)
        elif schedule.kind == WakeUpKind.COMMITMENT_REEVALUATION:
            kind = OutboxMessageKind.RENEGOTIATION
            title = "Commitment needs renegotiation"
            urgency = OutboxUrgency.HIGH
            references = OutboxReferences(commitment_id=target)
        elif schedule.kind == WakeUpKind.PLAN_GENERATION:
            kind = OutboxMessageKind.LONG_TASK_COMPLETE
            title = "Plan generation completed"
            references = OutboxReferences(goal_id=target)
        if kind is not None:
            outbox.enqueue(
                kind,
                title=title,
                body=body,
                deduplication_key=f"scheduler:{schedule.schedule_id}:{outcome}",
                references=references,
                urgency=urgency,
            )

    def _finish_cycle(
        self,
        result: CycleResult,
        processed: int,
        inferences: int,
        deferred: int,
        started_at: datetime,
        finished_at: datetime,
        started_clock: float,
    ) -> SchedulerCycle:
        cycle = SchedulerCycle(
            result=result,
            processed=processed,
            inferences=inferences,
            deferred=deferred,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
        )
        with self._lock:
            self._last_cycle = cycle
        self._observe(result, processed, deferred, time.monotonic() - started_clock)
        return cycle

    def _observe(
        self, result: CycleResult, processed: int, deferred: int, duration: float
    ) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.counter("kagya_autonomy_cycles_total", result=result.value)
            self.telemetry.counter(
                "kagya_autonomy_wakeups_total", float(processed), outcome="processed"
            )
            self.telemetry.counter(
                "kagya_autonomy_wakeups_total", float(deferred), outcome="deferred"
            )
            self.telemetry.observe("kagya_autonomy_cycle_duration_seconds", duration)
            self.telemetry.gauge(
                "kagya_autonomy_pending_wakeups", float(self.status().pending_count)
            )
        except Exception:
            return


class AutonomyLoop:
    """Single timer thread that stops before the authoritative runtime drains."""

    def __init__(
        self, scheduler: SubjectScheduler, *, poll_interval_seconds: float
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("Autonomy poll interval must be positive")
        self.scheduler = scheduler
        self.poll_interval_seconds = poll_interval_seconds
        self._wake = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.scheduler.set_notifier(self._wake.set)
        self._thread = Thread(target=self._run, name="kagya-autonomy-loop", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self.scheduler.stop_accepting()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while True:
            if self.scheduler.status().state != "accepting":
                return
            self.scheduler.run_cycle()
            self._wake.wait(self.poll_interval_seconds)
            self._wake.clear()
