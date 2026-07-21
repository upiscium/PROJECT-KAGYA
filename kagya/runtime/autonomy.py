"""Persistent, budgeted wake-ups for the single subject runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Event, RLock, Thread
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.decision import DecisionStatus
from kagya.motivation import CommitmentStatus, GoalStatus
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
    GOAL_REEVALUATION = "goal_reevaluation"
    NEEDS_INFORMATION = "needs_information"
    DECISION_OUTCOME = "decision_outcome"
    ACTION_TIMEOUT = "action_timeout"
    ACTION_RETRY = "action_retry"
    OUTBOX_DEADLINE = "outbox_deadline"
    SLEEP_CONSOLIDATION = "sleep_consolidation"
    OPERATOR = "operator"


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
    target_id: str | None = Field(default=None, max_length=256)
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
        if self.max_events <= 0 or self.max_inferences < 0 or self.max_wall_seconds <= 0:
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
                    CycleResult.STOPPED, 0, 0, 0, started_at, self.clock(), started_clock
                )
        due = self._due_schedules(started_at)
        if not due:
            return self._finish_cycle(
                CycleResult.NO_ACTION, 0, 0, 0, started_at, self.clock(), started_clock
            )

        processed = 0
        inferences = 0
        for schedule in due:
            if (
                processed >= self.budget.max_events
                or time.monotonic() - started_clock >= self.budget.max_wall_seconds
            ):
                break
            if (
                inferences + schedule.estimated_inferences
                > self.budget.max_inferences
            ):
                continue
            try:
                def process(item: WakeUpSchedule = schedule) -> str:
                    return self._process(item)

                self.runtime.execute(
                    AgentEventType.AUTONOMY_WAKE,
                    source="scheduler.wake",
                    handler=process,
                    payload={
                        "schedule_id": schedule.schedule_id,
                        "kind": schedule.kind.value,
                    },
                    causation_id=schedule.causation_id,
                    correlation_id=schedule.schedule_id,
                )
            except (AgentRuntimeQueueFull, AgentRuntimeStopped):
                break
            processed += 1
            inferences += schedule.estimated_inferences
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
        return stored + [item for item in self._derived_schedules() if item.schedule_id not in known]

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
            if goal.status in {GoalStatus.COMPLETED, GoalStatus.ABANDONED, GoalStatus.FAILED}:
                continue
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
        for commitment in self.main_loop.commitment_store.list_commitments(
            CommitmentStatus.ACTIVE
        ):
            if commitment.deadline is not None:
                schedules.append(
                    self._derived(
                        f"commitment-deadline:{commitment.commitment_id}:{commitment.deadline}",
                        WakeUpKind.COMMITMENT_DEADLINE,
                        datetime.fromisoformat(commitment.deadline),
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

    def _process(self, schedule: WakeUpSchedule) -> str:
        stored = self._stored_schedules()
        current = next(
            (item for item in stored if item.schedule_id == schedule.schedule_id), None
        )
        if current is not None and current.status != ScheduleStatus.PENDING:
            return "already_processed"
        if current is None:
            stored.append(schedule)

        outcome = "no_action"
        if schedule.kind in {
            WakeUpKind.GOAL_DEADLINE,
            WakeUpKind.GOAL_REEVALUATION,
            WakeUpKind.NEEDS_INFORMATION,
        }:
            decisions = self.main_loop.reevaluate_goals()
            outcome = "reevaluated" if any(item.action.value != "no_action" for item in decisions) else "no_action"
        elif schedule.kind == WakeUpKind.COMMITMENT_DEADLINE:
            commitment = self.main_loop.commitment_store.commitments.get(
                schedule.target_id or ""
            )
            if commitment is not None and commitment.status == CommitmentStatus.ACTIVE:
                self.main_loop.transition_commitment(
                    commitment.commitment_id,
                    CommitmentStatus.BREACHED,
                    reason="deadline_expired",
                )
                outcome = "commitment_breached"
        elif schedule.kind == WakeUpKind.DECISION_OUTCOME:
            decision = self.main_loop.decision_store.records.get(schedule.target_id or "")
            if decision is not None and decision.status == DecisionStatus.AWAITING_OUTCOME:
                outcome = "awaiting_outcome"

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
        return outcome

    def _should_recheck(self, schedule: WakeUpSchedule) -> bool:
        if schedule.kind == WakeUpKind.DECISION_OUTCOME:
            decision = self.main_loop.decision_store.records.get(schedule.target_id or "")
            return decision is not None and decision.status == DecisionStatus.AWAITING_OUTCOME
        if schedule.kind in {
            WakeUpKind.GOAL_REEVALUATION,
            WakeUpKind.NEEDS_INFORMATION,
        }:
            goal = self.main_loop.goal_manager.goals.get(schedule.target_id or "")
            return goal is not None and (
                goal.status == GoalStatus.SUSPENDED or goal.needs_information
            )
        return False

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

    def __init__(self, scheduler: SubjectScheduler, *, poll_interval_seconds: float) -> None:
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
        self._thread = Thread(
            target=self._run, name="kagya-autonomy-loop", daemon=True
        )
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
