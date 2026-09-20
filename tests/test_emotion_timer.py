from concurrent.futures import Future
from datetime import datetime, timezone
import math
from threading import Event
import time
from typing import Any, Callable, cast

import pytest

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeDurabilityPhase,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    AgentRuntimeStatus,
    EmotionTimer,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.status = AgentRuntimeStatus.ACCEPTING
        self.calls: list[tuple[AgentEventType, AgentEventSource]] = []
        self.futures: list[Future[AgentEventOutcome[Any]]] = []
        self.handlers: list[Callable[[], Any]] = []
        self.error: type[Exception] | None = None
        self.submit_errors: list[type[Exception] | None] = []
        self.future_error: BaseException | None = None
        self.invoke_handlers = False

    def submit(
        self,
        event_type: AgentEventType,
        source: AgentEventSource,
        handler: Callable[[], Any],
    ) -> Future[AgentEventOutcome[Any]]:
        self.calls.append((event_type, source))
        event = AgentEvent(
            f"event-{len(self.calls)}",
            event_type,
            source,
            datetime.now(timezone.utc),
        )
        error_type = self.error
        if self.submit_errors:
            error_type = self.submit_errors.pop(0)
        if error_type is not None:
            raise error_type(event)
        future: Future[AgentEventOutcome[Any]] = Future()
        self.futures.append(future)
        self.handlers.append(handler)
        if self.future_error is not None:
            future.set_exception(self.future_error)
        elif self.invoke_handlers:
            try:
                value = handler()
            except Exception as error:
                wrapped = AgentRuntimeExecutionError(event)
                wrapped.__cause__ = error
                future.set_exception(wrapped)
            else:
                future.set_result(AgentEventOutcome(event, value))
        return future


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        Event().wait(0.005)
    assert predicate()


def test_timer_waits_before_submitting_and_uses_closed_identity() -> None:
    runtime = FakeRuntime()
    handler_called = Event()
    timer = EmotionTimer(runtime, 0.05, handler_called.set)

    timer.start()
    assert not handler_called.wait(0.01)
    _wait_for(lambda: len(runtime.calls) == 1)
    timer.stop()

    assert runtime.calls == [
        (AgentEventType.EMOTION_TICK, AgentEventSource.RUNTIME_EMOTION_TIMER)
    ]
    assert not handler_called.is_set()


def test_stop_before_start_is_idempotent_and_does_not_create_thread() -> None:
    runtime = FakeRuntime()
    timer = EmotionTimer(runtime, 0.01, lambda: None)

    timer.stop()
    timer.stop()
    timer.start()
    Event().wait(0.03)

    assert runtime.calls == []


def test_timer_does_not_create_duplicate_producers_or_cancel_accepted_work() -> None:
    runtime = FakeRuntime()
    timer = EmotionTimer(runtime, 0.01, lambda: None)

    timer.start()
    timer.start()
    _wait_for(lambda: len(runtime.calls) == 1)
    Event().wait(0.04)
    assert len(runtime.calls) == 1

    timer.stop()
    timer.stop()
    assert len(runtime.futures) == 1
    assert not runtime.futures[0].cancelled()

    runtime.futures[0].set_result(
        AgentEventOutcome(
            AgentEvent(
                "event-1",
                AgentEventType.EMOTION_TICK,
                AgentEventSource.RUNTIME_EMOTION_TIMER,
                datetime.now(timezone.utc),
            ),
            None,
        )
    )


@pytest.mark.parametrize("error_type", [AgentRuntimeQueueFull, AgentRuntimeAdmissionBlocked])
def test_timer_drops_normal_admission_failures_without_invoking_handler(
    error_type: type[Exception],
) -> None:
    runtime = FakeRuntime()
    runtime.error = error_type
    handler_called = Event()
    timer = EmotionTimer(runtime, 0.01, handler_called.set)

    timer.start()
    _wait_for(lambda: len(runtime.calls) >= 2)
    timer.stop()

    assert not handler_called.is_set()
    attempts = len(runtime.calls)
    Event().wait(0.03)
    assert len(runtime.calls) == attempts


def test_timer_stops_when_runtime_is_stopped() -> None:
    runtime = FakeRuntime()
    runtime.error = AgentRuntimeStopped
    timer = EmotionTimer(runtime, 0.01, lambda: None)

    timer.start()
    _wait_for(lambda: len(runtime.calls) == 1)
    Event().wait(0.04)

    assert len(runtime.calls) == 1
    timer.stop()


def test_timer_stops_after_durability_failure() -> None:
    runtime = FakeRuntime()
    event = AgentEvent(
        "event-1",
        AgentEventType.EMOTION_TICK,
        AgentEventSource.RUNTIME_EMOTION_TIMER,
        datetime.now(timezone.utc),
    )
    runtime.future_error = AgentRuntimeDurabilityError(
        event,
        AgentRuntimeDurabilityPhase.INTERNAL_COMMIT,
        outcome_indeterminate=True,
    )
    timer = EmotionTimer(runtime, 0.01, lambda: None)

    timer.start()
    _wait_for(lambda: len(runtime.calls) == 1)
    Event().wait(0.04)

    assert len(runtime.calls) == 1
    timer.stop()


def test_timer_allows_later_wakeup_after_handler_failure() -> None:
    runtime = FakeRuntime()
    runtime.invoke_handlers = True
    calls = 0

    def failing_handler() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("private handler failure")

    timer = EmotionTimer(runtime, 0.01, failing_handler)
    timer.start()
    _wait_for(lambda: calls >= 3)
    timer.stop()

    assert len(runtime.calls) >= 3


def test_dropped_wakeups_preserve_clock_until_next_accepted_tick() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    current_time = [t0]
    engine = EmotionEngineAllostasis(
        EmotionState(valence=0.8, arousal=0.9, optimal_loss=0.7),
        temporal_state=None,
        clock=lambda: current_time[0],
    )
    engine.advance_to()
    before_state = engine.state
    runtime = FakeRuntime()
    runtime.submit_errors = [
        AgentRuntimeQueueFull,
        AgentRuntimeAdmissionBlocked,
        None,
        AgentRuntimeStopped,
    ]
    runtime.invoke_handlers = True
    current_time[0] = t1
    timer = EmotionTimer(runtime, 0.01, engine.advance_to)

    timer.start()
    _wait_for(lambda: len(runtime.calls) == 4)
    timer.stop()

    assert engine.state != before_state
    assert engine.temporal_state.last_update_at == t1
    assert len(runtime.calls) == 4
    assert len(runtime.handlers) == 1


@pytest.mark.parametrize("interval", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_timer_rejects_invalid_intervals(interval: object) -> None:
    with pytest.raises(ValueError):
        EmotionTimer(FakeRuntime(), cast(float, interval), lambda: None)
