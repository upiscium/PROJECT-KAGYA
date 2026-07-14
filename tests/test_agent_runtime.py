from threading import Event, Thread
from time import monotonic, sleep

import pytest

from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
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


def _block(entered: Event, release: Event, result: object) -> object:
    entered.set()
    assert release.wait(timeout=1)
    return result


def _raise(exc: Exception) -> None:
    raise exc
