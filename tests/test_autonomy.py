from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from kagya.decision import DecisionStatus
from kagya.motivation import (
    CommitmentFulfillability,
    CommitmentStatus,
    GoalStatus,
    MotivationDynamics,
    MotivationKind,
    MotivationSource,
)
from kagya.runtime import (
    AgentRuntime,
    AgentRuntimeStopped,
    CycleResult,
    EventJournal,
    JournalLifecycle,
    SchedulerBudget,
    ScheduleStatus,
    SubjectScheduler,
    WakeUpKind,
)


class _GoalManager:
    def __init__(self) -> None:
        self.goals: dict[str, object] = {}

    def list_goals(self) -> list[object]:
        return list(self.goals.values())


class _CommitmentStore:
    def __init__(self) -> None:
        self.commitments: dict[str, object] = {}

    def list_commitments(self, status: object) -> list[object]:
        return [
            item
            for item in self.commitments.values()
            if getattr(item, "status", None) == status
        ]


class _DecisionStore:
    records: dict[str, object] = {}

    def list_records(self, status: object) -> list[object]:
        return []


class _MainLoop:
    def __init__(
        self,
        extensions: dict[str, object] | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.persistent_state = SimpleNamespace(extensions=extensions or {})
        self.goal_manager = _GoalManager()
        self.commitment_store = _CommitmentStore()
        self.decision_store = _DecisionStore()
        self.motivation_dynamics = MotivationDynamics(
            max_goal_proposals_per_cycle=1, clock=clock
        )
        self.reevaluations = 0
        self.commitment_reevaluations = 0
        self.motivation_reevaluations = 0
        self.generated_goals = 0

    def reevaluate_goals(self) -> list[object]:
        self.reevaluations += 1
        for goal in self.goal_manager.goals.values():
            goal.status = GoalStatus.FAILED
        return [SimpleNamespace(action=SimpleNamespace(value="defer"))]

    def reassess_commitment(
        self,
        commitment_id: str,
        *,
        fulfillability: CommitmentFulfillability,
        reason: str,
    ) -> None:
        assert commitment_id in self.commitment_store.commitments
        assert fulfillability == CommitmentFulfillability.AT_RISK
        assert reason
        self.commitment_reevaluations += 1

    def reevaluate_motivation(
        self,
        *,
        max_goal_proposals: int | None = None,
        review_at: datetime | None = None,
    ) -> tuple[object, list[object]]:
        self.motivation_reevaluations += 1
        candidates, _ = self.motivation_dynamics.goal_candidates(
            max_goal_proposals, review_at=review_at
        )
        goals = []
        for candidate in candidates:
            goal_id = f"intrinsic:{candidate.motivation_id}"
            self.motivation_dynamics.link_goal(candidate.motivation_id, goal_id)
            goals.append(SimpleNamespace(goal_id=goal_id))
        self.generated_goals += len(goals)
        return SimpleNamespace(), goals

    def decay_motivation_record(
        self, motivation_id: str, elapsed_hours: float
    ) -> object:
        return self.motivation_dynamics.decay_record(motivation_id, elapsed_hours)

    def schedule_motivation_reviews(self, review_at: datetime) -> list[object]:
        return self.motivation_dynamics.schedule_next_reviews(review_at)


def _runtime(
    path: Path, initial_sequence: int = 0
) -> tuple[AgentRuntime, EventJournal]:
    journal = EventJournal(path)
    runtime = AgentRuntime(
        queue_capacity=8,
        event_journal=journal,
        initial_sequence=initial_sequence,
        completion_hook=lambda event: "0" * 64,
        failure_hook=lambda event, exc: "0" * 64,
    )
    runtime.start()
    return runtime, journal


def _scheduler(
    runtime: AgentRuntime,
    loop: _MainLoop,
    *,
    max_events: int = 2,
    max_inferences: int = 1,
    clock: Callable[[], datetime] | None = None,
) -> SubjectScheduler:
    return SubjectScheduler(
        runtime,
        loop,
        budget=SchedulerBudget(
            max_events=max_events,
            max_inferences=max_inferences,
            max_wall_seconds=1.0,
        ),
        reevaluation_interval_seconds=60.0,
        clock=clock,
    )


def test_schedule_survives_restart_and_is_processed_once(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    loop = _MainLoop()
    runtime, journal = _runtime(tmp_path / "journal.jsonl")
    scheduler = _scheduler(runtime, loop)
    scheduler.schedule(WakeUpKind.OPERATOR, now, schedule_id="persistent-wake")
    runtime.shutdown()

    restarted_runtime, restarted_journal = _runtime(
        tmp_path / "journal.jsonl", initial_sequence=1
    )
    restarted = _scheduler(restarted_runtime, loop)
    first = restarted.run_cycle(now + timedelta(seconds=1))
    second = restarted.run_cycle(now + timedelta(seconds=2))
    restarted_runtime.shutdown()

    assert first.result == CycleResult.PROCESSED
    assert second.result == CycleResult.NO_ACTION
    state = loop.persistent_state.extensions["subject_scheduler"]
    assert state["schedules"][0]["status"] == ScheduleStatus.COMPLETED.value
    wake_starts = [
        record
        for record in restarted_journal.verify()
        if record.lifecycle == JournalLifecycle.STARTED
        and record.event_type == "autonomy_wake"
    ]
    assert len(wake_starts) == 1
    assert journal.verify()


def test_duplicate_schedule_id_is_rejected(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    scheduler = _scheduler(runtime, _MainLoop())
    now = datetime.now(UTC)
    scheduler.schedule(WakeUpKind.OPERATOR, now, schedule_id="same-id")

    with pytest.raises(ValueError, match="already exists"):
        scheduler.schedule(WakeUpKind.OPERATOR, now, schedule_id="same-id")
    runtime.shutdown()


def test_cycle_stops_at_event_and_inference_budgets(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    scheduler = _scheduler(runtime, _MainLoop(), max_events=1, max_inferences=0)
    now = datetime.now(UTC)
    scheduler.schedule(
        WakeUpKind.OPERATOR, now, schedule_id="blocked", estimated_inferences=1
    )
    scheduler.schedule(WakeUpKind.OPERATOR, now, schedule_id="pending")

    blocked = scheduler.run_cycle(now)

    assert blocked.result == CycleResult.BUDGET_EXHAUSTED
    assert blocked.processed == 1
    assert blocked.deferred == 1
    runtime.shutdown()


def test_idle_cycle_is_explicit_no_action_without_runtime_event(tmp_path: Path) -> None:
    runtime, journal = _runtime(tmp_path / "journal.jsonl")
    scheduler = _scheduler(runtime, _MainLoop())

    cycle = scheduler.run_cycle(datetime.now(UTC))

    assert cycle.result == CycleResult.NO_ACTION
    assert journal.verify() == []
    runtime.shutdown()


def test_due_action_execution_is_processed_once_through_runtime(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    runtime, journal = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop()

    class _Actions:
        calls = 0

        def list_intents(self) -> list[object]:
            return (
                [
                    SimpleNamespace(
                        intent_id="intent-1",
                        status=SimpleNamespace(value="approved"),
                        updated_at=now,
                        deadline_at=now + timedelta(seconds=60),
                        retry_at=None,
                    )
                ]
                if self.calls == 0
                else []
            )

        def execute(self, intent_id: str) -> object:
            assert intent_id == "intent-1"
            self.calls += 1
            return SimpleNamespace(status=SimpleNamespace(value="succeeded"))

    loop.action_execution = _Actions()
    scheduler = _scheduler(runtime, loop)

    first = scheduler.run_cycle(now)
    second = scheduler.run_cycle(now + timedelta(seconds=1))
    runtime.shutdown()

    assert first.result == CycleResult.PROCESSED
    assert second.result == CycleResult.NO_ACTION
    assert loop.action_execution.calls == 1
    assert any(
        record.event_type == "action_execute"
        and record.lifecycle == JournalLifecycle.COMPLETED
        for record in journal.verify()
    )


def test_invalid_scheduled_action_resolves_once_as_terminal_failure(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop()
    decision = SimpleNamespace(
        decision_id="invalid-action-decision",
        status=DecisionStatus.AWAITING_OUTCOME,
        updated_at=now,
        selected_candidate_id="invalid-action",
        considered_candidates=(
            SimpleNamespace(
                candidate=SimpleNamespace(
                    candidate_id="invalid-action",
                    parameters={"action": {"tool_name": "document_search"}},
                )
            ),
        ),
    )

    class _Decisions:
        records = {decision.decision_id: decision}

        def list_records(self, status: object) -> list[object]:
            return [item for item in self.records.values() if item.status == status]

        def get(self, decision_id: str) -> object:
            return self.records[decision_id]

    class _Actions:
        calls = 0

        def list_intents(self) -> list[object]:
            return []

        def create_from_decision(self, decision_id: str, **_: object) -> object:
            assert decision_id == decision.decision_id
            self.calls += 1
            return SimpleNamespace(validation_id="bounded-validation")

    loop.decision_store = _Decisions()
    loop.action_execution = _Actions()

    def record_outcome(decision_id: str, **_: object) -> None:
        assert decision_id == decision.decision_id
        decision.status = DecisionStatus.RESOLVED

    loop.record_decision_outcome = record_outcome
    scheduler = _scheduler(runtime, loop)

    first = scheduler.run_cycle(now)
    second = scheduler.run_cycle(now + timedelta(seconds=1))
    runtime.shutdown()

    assert first.result == CycleResult.PROCESSED
    assert second.result == CycleResult.NO_ACTION
    assert loop.action_execution.calls == 1
    assert decision.status == DecisionStatus.RESOLVED


def test_due_goal_deadline_is_reevaluated_through_runtime(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    runtime, journal = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop()
    loop.goal_manager.goals["goal-1"] = SimpleNamespace(
        goal_id="goal-1",
        status=GoalStatus.ACTIVE,
        deadline=(now - timedelta(seconds=1)).isoformat(),
        needs_information=False,
        updated_at=now.isoformat(),
    )
    scheduler = _scheduler(runtime, loop)

    cycle = scheduler.run_cycle(now)

    assert cycle.result == CycleResult.PROCESSED
    assert loop.reevaluations == 1
    assert any(
        record.event_type == "autonomy_wake"
        and record.lifecycle == JournalLifecycle.COMPLETED
        for record in journal.verify()
    )
    runtime.shutdown()


def test_at_risk_commitment_is_reevaluated_through_runtime(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop()
    loop.commitment_store.commitments["promise-1"] = SimpleNamespace(
        commitment_id="promise-1",
        status=CommitmentStatus.ACTIVE,
        deadline=None,
        fulfillability=CommitmentFulfillability.AT_RISK,
        fulfillability_reason="resource availability changed",
        updated_at=(now - timedelta(seconds=61)).isoformat(),
    )
    scheduler = _scheduler(runtime, loop)

    cycle = scheduler.run_cycle(now)

    assert cycle.result == CycleResult.PROCESSED
    assert loop.commitment_reevaluations == 1
    runtime.shutdown()


def test_persistent_motivation_wakes_are_budgeted_and_journaled(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    runtime, journal = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop()
    record = loop.motivation_dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:planner",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("future-self:planner", "identity-claim:planning"),
    )
    scheduler = _scheduler(runtime, loop, max_events=2, max_inferences=0)

    cycle = scheduler.run_cycle(now + timedelta(seconds=61))

    assert cycle.processed == 2
    assert cycle.inferences == 0
    assert loop.generated_goals == 1
    assert loop.motivation_dynamics.get(record.motivation_id).related_goal_ids
    completed = [
        item
        for item in journal.verify()
        if item.lifecycle == JournalLifecycle.COMPLETED
        and item.event_type in {"autonomy_wake", "intrinsic_goal_propose"}
    ]
    assert len(completed) == 2
    assert {item.event_type for item in completed} == {
        "autonomy_wake",
        "intrinsic_goal_propose",
    }
    outcomes = {
        item["outcome"]
        for item in loop.persistent_state.extensions["subject_scheduler"]["schedules"]
        if item["status"] == ScheduleStatus.COMPLETED.value
    }
    assert outcomes == {"reevaluated", "decayed"}
    runtime.shutdown()


def test_skewed_domain_changes_use_scheduler_time_but_explicit_boundaries_do_not(
    tmp_path: Path,
) -> None:
    scheduler_now = datetime(2026, 7, 24, tzinfo=UTC)
    current = scheduler_now
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop(clock=lambda: current)
    record = loop.motivation_dynamics.observe_structured_signal(
        MotivationKind.DESIRE,
        MotivationSource.LEARNING,
        "future-self:clock-skew",
        signal=0.8,
        uncertainty=0.2,
        source_refs=("experience:clock-skew-1", "experience:clock-skew-2"),
    )
    loop.motivation_dynamics.records[record.motivation_id] = replace(
        record, updated_at=(scheduler_now + timedelta(hours=12)).isoformat()
    )
    explicit_review = scheduler_now + timedelta(hours=3)
    review_record = loop.motivation_dynamics.observe_structured_signal(
        MotivationKind.INTEREST,
        MotivationSource.CURIOSITY,
        "future-self:explicit-review",
        signal=0.2,
        uncertainty=0.8,
        source_refs=("experience:explicit-review",),
    )
    loop.motivation_dynamics.records[review_record.motivation_id] = replace(
        review_record,
        updated_at=(scheduler_now + timedelta(hours=12)).isoformat(),
        next_review_at=explicit_review.isoformat(),
    )
    deadline = scheduler_now + timedelta(hours=2)
    loop.goal_manager.goals["boundary-goal"] = SimpleNamespace(
        goal_id="boundary-goal",
        status=GoalStatus.ACTIVE,
        deadline=deadline.isoformat(),
        needs_information=False,
        updated_at=(scheduler_now + timedelta(hours=12)).isoformat(),
    )
    scheduler = _scheduler(
        runtime,
        loop,
        max_events=3,
        max_inferences=0,
        clock=lambda: current,
    )

    schedules = scheduler._all_schedules()
    reevaluation = next(
        item
        for item in schedules
        if item.kind == WakeUpKind.MOTIVATION_REEVALUATION
        and item.target_id == record.target_ref
    )
    goal_deadline = next(
        item for item in schedules if item.kind == WakeUpKind.GOAL_DEADLINE
    )
    review = next(
        item
        for item in schedules
        if item.kind == WakeUpKind.MOTIVATION_REEVALUATION
        and item.target_id == review_record.target_ref
    )

    assert reevaluation.wake_at == scheduler_now + timedelta(seconds=60)
    schedule_id = reevaluation.schedule_id
    current = scheduler_now + timedelta(seconds=30)
    still_pending = next(
        item
        for item in scheduler._all_schedules()
        if item.kind == WakeUpKind.MOTIVATION_REEVALUATION
        and item.target_id == record.target_ref
    )
    restarted = _scheduler(
        runtime,
        loop,
        max_events=3,
        max_inferences=0,
        clock=lambda: current,
    )
    after_restart = next(
        item
        for item in restarted._all_schedules()
        if item.kind == WakeUpKind.MOTIVATION_REEVALUATION
        and item.target_id == record.target_ref
    )
    assert still_pending.schedule_id == after_restart.schedule_id == schedule_id
    assert still_pending.wake_at == reevaluation.wake_at
    assert after_restart.wake_at == current + timedelta(seconds=60)
    assert goal_deadline.wake_at == deadline
    assert review.wake_at == explicit_review
    current = scheduler_now + timedelta(seconds=91)
    cycle = restarted.run_cycle()
    assert cycle.result == CycleResult.PROCESSED
    assert loop.generated_goals == 1
    runtime.shutdown()


def test_motivation_reevaluation_records_explicit_no_action(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    loop = _MainLoop()
    loop.motivation_dynamics.observe_structured_signal(
        MotivationKind.INTEREST,
        MotivationSource.CURIOSITY,
        "context:weak",
        signal=0.2,
        uncertainty=0.8,
        source_refs=("experience:one", "experience:two"),
    )
    scheduler = _scheduler(runtime, loop, max_events=1, max_inferences=0)
    scheduler.schedule(
        WakeUpKind.MOTIVATION_REEVALUATION,
        now,
        target_id="context:weak",
        schedule_id="weak-motivation-review",
    )

    scheduler.run_cycle(now)

    stored = loop.persistent_state.extensions["subject_scheduler"]["schedules"]
    completed = next(
        item for item in stored if item["schedule_id"] == "weak-motivation-review"
    )
    assert completed["outcome"] == "no_action"
    assert loop.generated_goals == 0
    runtime.shutdown()


def test_shutdown_rejects_new_wakeups(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path / "journal.jsonl")
    scheduler = _scheduler(runtime, _MainLoop())
    scheduler.stop_accepting()

    with pytest.raises(AgentRuntimeStopped):
        scheduler.schedule(WakeUpKind.OPERATOR, datetime.now(UTC))
    runtime.shutdown()
