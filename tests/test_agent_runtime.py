from dataclasses import FrozenInstanceError, asdict
from threading import Event, Thread

import pytest

from kagya.config.schema import Settings
from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStatus,
    AgentRuntimeStopped,
)


def test_fifo_order_and_consumer_sequences() -> None:
    runtime = AgentRuntime(4)
    runtime.start()
    futures = [
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda i=i: i)
        for i in range(4)
    ]

    outcomes = [future.result(timeout=2) for future in futures]
    runtime.shutdown()

    assert [outcome.value for outcome in outcomes] == [0, 1, 2, 3]
    assert [outcome.event.processing_sequence for outcome in outcomes] == [1, 2, 3, 4]
    assert runtime.status is AgentRuntimeStatus.STOPPED


def test_full_queue_is_rejected_without_waiting() -> None:
    runtime = AgentRuntime(1)
    started = Event()
    release = Event()
    rejected_handler_ran = Event()
    runtime.start()

    def block() -> None:
        started.set()
        release.wait()

    first = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, block)
    assert started.wait(timeout=2)
    second = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 2)
    with pytest.raises(AgentRuntimeQueueFull) as error:
        runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            rejected_handler_ran.set,
        )
    assert error.value.event.processing_sequence is None
    assert not rejected_handler_ran.is_set()
    release.set()
    assert first.result(timeout=2).value is None
    assert second.result(timeout=2).value == 2
    runtime.shutdown()
    assert not rejected_handler_ran.is_set()


def test_handler_failure_preserves_cause_and_consumer_isolated() -> None:
    runtime = AgentRuntime(2)
    runtime.start()

    def fail() -> None:
        raise ValueError("private handler details")

    failed = runtime.submit(
        AgentEventType.DEBUG_CHAT, AgentEventSource.API_CHAT_DEBUG, fail
    )
    succeeding = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "ok"
    )
    with pytest.raises(AgentRuntimeExecutionError) as error:
        failed.result(timeout=2)
    assert isinstance(error.value.__cause__, ValueError)
    assert "private handler details" not in str(error.value)
    assert succeeding.result(timeout=2).value == "ok"
    runtime.shutdown()


def test_shutdown_drains_and_rejects_new_work() -> None:
    runtime = AgentRuntime(2)
    started = Event()
    release = Event()
    runtime.start()
    future = runtime.submit(
        AgentEventType.SLEEP,
        AgentEventSource.API_SLEEP_RUN,
        lambda: (started.set(), release.wait(), 4)[2],
    )
    assert started.wait(timeout=2)
    shutdown_thread = Thread(target=runtime.shutdown)
    shutdown_thread.start()
    while runtime.status is AgentRuntimeStatus.ACCEPTING:
        pass
    assert runtime.status is AgentRuntimeStatus.DRAINING
    with pytest.raises(AgentRuntimeStopped):
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 5)
    release.set()
    shutdown_thread.join(timeout=2)
    assert not shutdown_thread.is_alive()
    assert future.result(timeout=2).value == 4
    with pytest.raises(AgentRuntimeStopped) as error:
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 5)
    assert error.value.event.processing_sequence is None


def test_cancelled_future_does_not_cancel_handler() -> None:
    runtime = AgentRuntime(1)
    started = Event()
    release = Event()
    accepted_handler_ran = Event()
    runtime.start()
    blocker = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: (started.set(), release.wait(), "released")[2],
    )
    assert started.wait(timeout=2)
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: accepted_handler_ran.set(),
    )
    assert future.cancel()
    release.set()
    assert blocker.result(timeout=2).value == "released"
    runtime.shutdown()
    assert future.cancelled()
    assert accepted_handler_ran.is_set()


def test_event_is_immutable_and_has_no_handler_payload() -> None:
    private_sentinel = "PRIVATE-SENTINEL-R02"
    runtime = AgentRuntime(1)
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: {"prompt": private_sentinel, "hidden_thought": private_sentinel},
    )
    event = future.result(timeout=2).event
    runtime.shutdown()
    assert set(asdict(event)) == {
        "event_id",
        "event_type",
        "source",
        "requested_at",
        "processing_sequence",
    }
    assert private_sentinel not in str(asdict(event))
    assert "prompt" not in str(asdict(event))
    assert "hidden_thought" not in str(asdict(event))
    with pytest.raises(FrozenInstanceError):
        event.source = "changed"  # type: ignore[misc]


def test_arbitrary_source_cannot_be_used_as_private_metadata() -> None:
    runtime = AgentRuntime(1)
    runtime.start()
    with pytest.raises(TypeError):
        runtime.submit(
            AgentEventType.CHAT,
            "PRIVATE-SENTINEL-R02",  # type: ignore[arg-type]
            lambda: None,
        )
    runtime.shutdown()


def test_arbitrary_event_type_cannot_be_used_as_private_metadata() -> None:
    runtime = AgentRuntime(1)
    runtime.start()
    with pytest.raises(TypeError):
        runtime.submit(
            "PRIVATE-SENTINEL-R02",  # type: ignore[arg-type]
            AgentEventSource.API_CHAT,
            lambda: None,
        )
    runtime.shutdown()


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_invalid_capacity(capacity: int) -> None:
    with pytest.raises(ValueError):
        AgentRuntime(capacity)


def test_runtime_config_defaults_and_yaml_value() -> None:
    field = Settings.model_fields["runtime"]
    assert field.default_factory is not None
    assert field.default_factory().queue_capacity == 64


def test_processing_sequence_is_process_local_to_each_runtime() -> None:
    sequences = []
    for _ in range(2):
        runtime = AgentRuntime(1)
        runtime.start()
        outcome = runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
        ).result(timeout=2)
        runtime.shutdown()
        sequences.append(outcome.event.processing_sequence)

    assert sequences == [1, 1]
