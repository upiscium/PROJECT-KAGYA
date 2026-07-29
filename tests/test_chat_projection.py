from pathlib import Path
from threading import Event
import time

from kagya.operation_status import OperationState
from kagya.runtime import AgentEventType, AgentRuntime
from tests.chat_job_helpers import ChatJobRegistry


def _wait_for(
    registry: ChatJobRegistry,
    operation_id: str,
    predicate,
    timeout: float = 3.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = registry.get(operation_id)
        assert record is not None
        if predicate(record):
            return record
        time.sleep(0.01)
    raise AssertionError("projection condition was not reached")


def test_projection_does_not_occupy_agent_runtime_worker(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda payload: {"response": str(payload["text"])},
    )
    entered = Event()
    release = Event()
    publish = registry.publish_terminal_projection

    def blocked_publish(operation_id: str) -> None:
        entered.set()
        release.wait(2)
        publish(operation_id)

    registry.publish_terminal_projection = blocked_publish
    job, _ = registry.enqueue(
        {"text": "x" * 4000},
        client_id="client",
        idempotency_key="projection-block",
        correlation_id="context",
    )
    assert entered.wait(1)

    following = runtime.submit(
        AgentEventType.STATE_SNAPSHOT,
        source="test.after-projection",
        handler=lambda: "processed",
    ).result(timeout=1)

    assert following.value == "processed"
    completed = registry.get(job.status.operation_id)
    assert completed is not None
    assert completed.status.status == OperationState.COMPLETED
    release.set()
    _wait_for(
        registry,
        job.status.operation_id,
        lambda record: record.terminal_projection_state == "published",
    )
    runtime.shutdown()
    registry.close()


def test_long_response_projection_has_bounded_durable_writes(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda payload: {"response": str(payload["text"])},
    )
    writes = 0
    persist = registry._persist

    def count_persist() -> None:
        nonlocal writes
        writes += 1
        persist()

    registry._persist = count_persist
    job, _ = registry.enqueue(
        {"text": "x" * 4000},
        client_id="client",
        idempotency_key="bounded-writes",
        correlation_id="context",
    )
    record = _wait_for(
        registry,
        job.status.operation_id,
        lambda item: item.terminal_projection_state == "published",
    )
    runtime.shutdown()
    registry.close()

    assert record.terminal_projection_event_count > 100
    assert writes <= 12


def test_terminal_projection_ids_are_stable_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        path, runtime, lambda payload: {"response": str(payload["text"])}
    )
    job, _ = registry.enqueue(
        {"text": "stable projection"},
        client_id="client",
        idempotency_key="stable-ids",
        correlation_id="context",
    )
    _wait_for(
        registry,
        job.status.operation_id,
        lambda record: record.terminal_projection_state == "published",
    )
    original_ids = [
        event.event_id
        for event in registry.events_after(job.status.operation_id, 0)
        if event.event in {"token", "final"}
    ]
    runtime.shutdown()
    registry.close()

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(path, restarted_runtime, lambda payload: payload)
    restarted.activate()
    _wait_for(
        restarted,
        job.status.operation_id,
        lambda _record: bool(restarted.events_after(job.status.operation_id, 0)),
    )
    restarted_ids = [
        event.event_id
        for event in restarted.events_after(job.status.operation_id, 0)
        if event.event in {"token", "final"}
    ]
    restarted_runtime.shutdown()
    restarted.close()

    assert restarted_ids == original_ids
    assert restarted.events_after(job.status.operation_id, original_ids[-2])[-1].event == "final"


def test_projection_failure_preserves_result_and_retries(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda _payload: {"response": "authoritative result"},
    )
    publish = registry.publish_terminal_projection
    registry.publish_terminal_projection = lambda _operation_id: (_ for _ in ()).throw(
        OSError("projection unavailable")
    )
    job, _ = registry.enqueue(
        {"text": "request"},
        client_id="client",
        idempotency_key="projection-retry",
        correlation_id="context",
    )
    failed = _wait_for(
        registry,
        job.status.operation_id,
        lambda record: record.terminal_projection_state == "projection_failed",
    )

    assert failed.status.status == OperationState.COMPLETED
    assert failed.result == {"response": "authoritative result"}
    registry.publish_terminal_projection = publish
    recovered = _wait_for(
        registry,
        job.status.operation_id,
        lambda record: record.terminal_projection_state == "published",
    )
    runtime.shutdown()
    registry.close()

    assert recovered.result == {"response": "authoritative result"}
    assert registry.events_after(job.status.operation_id, 0)[-1].event == "final"
