from threading import Event, Thread, current_thread
from time import monotonic, sleep

import pytest

from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentRuntimeJournalError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    EmotionTimer,
    current_agent_event,
)


class RecordingEventLog:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record(
        self,
        *,
        category: str,
        event_type: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> object:
        self.events.append(
            {
                "category": category,
                "event_type": event_type,
                "message": message,
                "metadata": metadata or {},
            }
        )
        return object()


class FailingEventLog:
    def record(self, **kwargs: object) -> object:
        raise RuntimeError("logger unavailable")


class RecordingJournal:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.lifecycle: list[str] = []

    def accepted(self, event: object) -> object:
        return self._record("accepted")

    def started(self, event: object) -> object:
        return self._record("started")

    def completed(self, event: object, snapshot_hash: str) -> object:
        return self._record("completed")

    def failed(
        self,
        event: object,
        failure_category: str,
        snapshot_hash: str | None,
    ) -> object:
        return self._record("failed")

    def _record(self, lifecycle: str) -> object:
        if lifecycle == self.fail_on:
            raise OSError(f"{lifecycle} unavailable")
        self.lifecycle.append(lifecycle)
        return object()


def test_events_run_in_acceptance_order_with_monotonic_sequences() -> None:
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    execution_order: list[str] = []

    futures = [
        runtime.submit(
            AgentEventType.CHAT,
            source="test",
            handler=lambda value=value: execution_order.append(value) or value,
        )
        for value in ["first", "second", "third"]
    ]

    outcomes = [future.result(timeout=1) for future in futures]
    runtime.shutdown()

    assert execution_order == ["first", "second", "third"]
    assert [outcome.event.processing_sequence for outcome in outcomes] == [1, 2, 3]
    assert [outcome.value for outcome in outcomes] == execution_order


def test_handler_exception_does_not_stop_later_events() -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()

    failed = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: _raise(ValueError("private failure detail")),
    )
    succeeded = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: "recovered",
    )

    with pytest.raises(ValueError, match="private failure detail"):
        failed.result(timeout=1)
    assert succeeded.result(timeout=1).value == "recovered"
    assert runtime.is_alive is True
    runtime.shutdown()


def test_full_queue_rejects_without_blocking() -> None:
    runtime = AgentRuntime(queue_capacity=1)
    runtime.start()
    entered = Event()
    release = Event()
    first = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: _block(entered, release, "first"),
    )
    assert entered.wait(timeout=1)
    second = runtime.submit(
        AgentEventType.SLEEP,
        source="test",
        handler=lambda: "second",
    )

    with pytest.raises(AgentRuntimeQueueFull):
        runtime.submit(
            AgentEventType.CHAT,
            source="test",
            handler=lambda: "rejected",
        )

    release.set()
    assert first.result(timeout=1).value == "first"
    assert second.result(timeout=1).value == "second"
    runtime.shutdown()


def test_cancelled_waiter_does_not_cancel_accepted_event() -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    entered = Event()
    release = Event()
    committed = Event()
    first = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: _block(entered, release, None),
    )
    assert entered.wait(timeout=1)
    accepted = runtime.submit(
        AgentEventType.MEMORY_UPDATE,
        source="test",
        handler=lambda: committed.set(),
    )

    assert accepted.cancel() is True
    release.set()
    first.result(timeout=1)
    runtime.shutdown()

    assert committed.is_set()


def test_shutdown_drains_accepted_events_and_rejects_new_work() -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    entered = Event()
    release = Event()
    completed: list[str] = []
    first = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: _block(entered, release, "first"),
    )
    assert entered.wait(timeout=1)
    second = runtime.submit(
        AgentEventType.SLEEP,
        source="test",
        handler=lambda: completed.append("second"),
    )
    shutdown = Thread(target=runtime.shutdown)
    shutdown.start()
    deadline = monotonic() + 1
    while runtime.is_accepting and monotonic() < deadline:
        sleep(0.001)

    with pytest.raises(AgentRuntimeStopped):
        runtime.submit(
            AgentEventType.CHAT,
            source="test",
            handler=lambda: None,
        )

    release.set()
    shutdown.join(timeout=1)

    assert shutdown.is_alive() is False
    assert first.result(timeout=1).value == "first"
    assert second.result(timeout=1).value is None
    assert completed == ["second"]
    assert runtime.is_alive is False


def test_event_logs_include_order_but_not_private_payload() -> None:
    event_log = RecordingEventLog()
    runtime = AgentRuntime(queue_capacity=1, event_recorder=event_log)
    runtime.start()

    outcome = runtime.execute(
        AgentEventType.DEBUG_CHAT,
        source="api.chat.debug",
        handler=lambda: "visible",
        payload={"prompt": "private prompt", "hidden_thought": "private thought"},
        correlation_id="correlation-1",
    )
    runtime.shutdown()

    assert outcome.event.payload["prompt"] == "private prompt"
    assert [event["event_type"] for event in event_log.events] == [
        "started",
        "completed",
    ]
    serialized = str(event_log.events)
    assert "processing_sequence" in serialized
    assert "correlation-1" in serialized
    assert "private prompt" not in serialized
    assert "private thought" not in serialized


def test_event_log_failure_does_not_stop_processing() -> None:
    runtime = AgentRuntime(queue_capacity=1, event_recorder=FailingEventLog())
    runtime.start()

    outcome = runtime.execute(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: "completed",
    )

    assert outcome.value == "completed"
    assert runtime.is_alive is True
    runtime.shutdown()


def test_durable_journal_records_acceptance_before_execution() -> None:
    journal = RecordingJournal()
    runtime = AgentRuntime(
        queue_capacity=1,
        event_journal=journal,
        completion_hook=lambda event: "a" * 64,
    )
    runtime.start()

    outcome = runtime.execute(
        AgentEventType.CHAT, source="test", handler=lambda: "completed"
    )
    runtime.shutdown()

    assert outcome.value == "completed"
    assert journal.lifecycle == ["accepted", "started", "completed"]


def test_durable_accept_failure_rejects_event_without_running_handler() -> None:
    journal = RecordingJournal(fail_on="accepted")
    runtime = AgentRuntime(queue_capacity=1, event_journal=journal)
    runtime.start()
    ran = Event()

    with pytest.raises(AgentRuntimeJournalError, match="durably accepted"):
        runtime.submit(
            AgentEventType.CHAT, source="test", handler=lambda: ran.set()
        )
    runtime.shutdown()

    assert ran.is_set() is False


def test_durable_completion_failure_fails_stops_runtime() -> None:
    journal = RecordingJournal(fail_on="completed")
    runtime = AgentRuntime(
        queue_capacity=1,
        event_journal=journal,
        completion_hook=lambda event: "a" * 64,
    )
    runtime.start()

    future = runtime.submit(
        AgentEventType.CHAT, source="test", handler=lambda: "mutated"
    )

    with pytest.raises(AgentRuntimeJournalError, match="completion"):
        future.result(timeout=1)
    assert runtime.is_accepting is False
    with pytest.raises(AgentRuntimeStopped):
        runtime.submit(
            AgentEventType.CHAT, source="test", handler=lambda: "rejected"
        )
    runtime.shutdown()


def test_handler_failure_is_durably_recorded_and_runtime_continues() -> None:
    journal = RecordingJournal()
    runtime = AgentRuntime(
        queue_capacity=2,
        event_journal=journal,
        completion_hook=lambda event: "a" * 64,
        failure_hook=lambda event, exc: "b" * 64,
    )
    runtime.start()

    failed = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: _raise(ValueError("expected")),
    )
    succeeded = runtime.submit(
        AgentEventType.CHAT, source="test", handler=lambda: "next"
    )

    with pytest.raises(ValueError, match="expected"):
        failed.result(timeout=1)
    assert succeeded.result(timeout=1).value == "next"
    runtime.shutdown()
    assert journal.lifecycle == [
        "accepted",
        "accepted",
        "started",
        "failed",
        "started",
        "completed",
    ]


def test_restored_sequence_and_completion_hook_define_commit_boundary() -> None:
    committed: list[int] = []
    runtime = AgentRuntime(
        queue_capacity=1,
        initial_sequence=8,
        completion_hook=lambda event: committed.append(
            event.processing_sequence or 0
        ),
    )
    runtime.start()

    outcome = runtime.execute(
        AgentEventType.CHAT, source="test", handler=lambda: "result"
    )

    assert committed == [9]
    assert outcome.event.processing_sequence == 9
    runtime.shutdown()


def test_completion_hook_failure_fails_only_that_event() -> None:
    attempts = 0

    def checkpoint(event: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("snapshot unavailable")

    runtime = AgentRuntime(queue_capacity=2, completion_hook=checkpoint)
    runtime.start()

    failed = runtime.submit(
        AgentEventType.CHAT, source="test", handler=lambda: "mutated"
    )
    succeeded = runtime.submit(
        AgentEventType.CHAT, source="test", handler=lambda: "next"
    )

    with pytest.raises(OSError, match="snapshot unavailable"):
        failed.result(timeout=1)
    assert succeeded.result(timeout=1).value == "next"
    assert runtime.is_alive is True
    runtime.shutdown()


def test_current_event_is_available_only_inside_handler() -> None:
    runtime = AgentRuntime(queue_capacity=1)
    runtime.start()

    outcome = runtime.execute(
        AgentEventType.CHAT,
        source="test.source",
        handler=current_agent_event,
    )

    assert outcome.value is not None
    assert outcome.value.event_id == outcome.event.event_id
    assert outcome.value.processing_sequence == 1
    assert current_agent_event() is None
    runtime.shutdown()


def test_emotion_timer_mutation_runs_on_agent_consumer() -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    completed = Event()
    threads: list[str] = []
    timer = EmotionTimer(
        runtime,
        lambda elapsed: threads.append(current_thread().name) or completed.set(),
        interval_seconds=0.01,
    )

    timer.start()
    assert completed.wait(timeout=1)
    timer.stop()
    runtime.shutdown()

    assert threads
    assert set(threads) == {"kagya-agent-runtime"}


def _block(entered: Event, release: Event, result: object) -> object:
    entered.set()
    assert release.wait(timeout=1)
    return result


def _raise(exc: Exception) -> None:
    raise exc
