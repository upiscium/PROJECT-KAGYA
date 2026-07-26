from __future__ import annotations

import json
from pathlib import Path
import shutil
from threading import Event
import time

import pytest

from kagya.chat_jobs import ChatJobRegistry
from kagya.operation_status import OperationCancelCode, OperationState
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentRuntimeQueueFull,
    cancellation_checkpoint,
    current_agent_event,
)


def _wait(registry: ChatJobRegistry, operation_id: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = registry.get(operation_id)
        assert record is not None
        if record.status.status in {
            OperationState.COMPLETED,
            OperationState.FAILED,
            OperationState.CANCELED,
        }:
            return record
        time.sleep(0.01)
    raise AssertionError("chat job did not become terminal")


def _result(payload: dict[str, object]) -> dict[str, object]:
    return {"response": str(payload["text"])}


def test_duplicate_submission_has_one_event_and_public_replay(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, _result)
    first, created = registry.enqueue(
        {"text": "visible"},
        client_id="client",
        idempotency_key="same",
        correlation_id="context",
    )
    duplicate, duplicate_created = registry.enqueue(
        {"text": "ignored"},
        client_id="client",
        idempotency_key="same",
        correlation_id="context",
    )
    completed = _wait(registry, first.status.operation_id)
    runtime.shutdown()

    assert created is True
    assert duplicate_created is False
    assert duplicate.status.operation_id == first.status.operation_id
    assert completed.result == {"response": "visible"}
    events = registry.events_after(first.status.operation_id, 0)
    assert [event.event_id for event in events] == list(range(1, len(events) + 1))
    assert [event.event for event in events][-2:] == ["token", "final"]


def test_queued_cancel_never_calls_chat_handler(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    release = Event()
    runtime.submit(
        AgentEventType.STATE_SNAPSHOT,
        source="test.blocker",
        handler=lambda: release.wait(2),
    )
    called = Event()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json", runtime, lambda payload: called.set() or _result(payload)
    )
    job, _ = registry.enqueue(
        {"text": "must-not-run"},
        client_id="client",
        idempotency_key="cancel-queued",
        correlation_id="context",
    )

    assert registry.cancel(
        job.status.operation_id, OperationCancelCode.CLIENT_REQUEST
    ) == "canceled"
    release.set()
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.CANCELED
    assert terminal.result is None
    assert not called.is_set()


def test_running_cancel_aborts_unsupported_provider_before_commit(tmp_path: Path) -> None:
    entered = Event()
    release = Event()

    def execute(payload: dict[str, object]) -> dict[str, object]:
        entered.set()
        release.wait(2)
        cancellation_checkpoint()
        return _result(payload)

    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, execute)
    job, _ = registry.enqueue(
        {"text": "partial-private"},
        client_id="client",
        idempotency_key="cancel-running",
        correlation_id="context",
    )
    assert entered.wait(1)
    assert registry.cancel(
        job.status.operation_id, OperationCancelCode.CLIENT_REQUEST
    ) == "cancel_requested"
    release.set()
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.CANCELED
    assert terminal.result is None
    assert all(
        event.event != "token"
        for event in registry.events_after(job.status.operation_id, 0)
    )


def test_queue_order_matches_runtime_processing_sequence(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()

    def execute(payload: dict[str, object]) -> dict[str, object]:
        event = current_agent_event()
        assert event is not None
        return {"response": str(payload["text"]), "sequence": event.processing_sequence}

    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, execute)
    jobs = [
        registry.enqueue(
            {"text": str(index)},
            client_id="client",
            idempotency_key=f"order-{index}",
            correlation_id="context",
        )[0]
        for index in range(3)
    ]
    completed = [_wait(registry, job.status.operation_id) for job in jobs]
    runtime.shutdown()

    assert [item.result["response"] for item in completed if item.result] == ["0", "1", "2"]
    sequences = [item.result["sequence"] for item in completed if item.result]
    assert sequences == sorted(sequences)


def test_backpressure_removes_rejected_registry_record(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=1)
    runtime.start()
    release = Event()
    runtime.submit(
        AgentEventType.STATE_SNAPSHOT,
        source="test.blocker",
        handler=lambda: release.wait(2),
    )
    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, _result)
    first, _ = registry.enqueue(
        {"text": "queued"},
        client_id="client",
        idempotency_key="first",
        correlation_id="context",
    )
    with pytest.raises(AgentRuntimeQueueFull):
        registry.enqueue(
            {"text": "rejected"},
            client_id="client",
            idempotency_key="second",
            correlation_id="context",
        )
    release.set()
    _wait(registry, first.status.operation_id)
    runtime.shutdown()

    persisted = json.loads((tmp_path / "jobs.json").read_text())
    assert len(persisted) == 1


def test_fresh_registry_replays_queued_job_without_plaintext(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "jobs.json"
    second_path = tmp_path / "second" / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    release = Event()
    runtime.submit(
        AgentEventType.STATE_SNAPSHOT,
        source="test.blocker",
        handler=lambda: release.wait(2),
    )
    first_registry = ChatJobRegistry(first_path, runtime, _result)
    job, _ = first_registry.enqueue(
        {"text": "PRIVATE_SENTINEL"},
        client_id="client",
        idempotency_key="restart",
        correlation_id="context",
    )
    second_path.parent.mkdir(parents=True)
    shutil.copy2(first_path, second_path)
    shutil.copy2(first_path.with_suffix(".json.key"), second_path.with_suffix(".json.key"))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        second_path, restarted_runtime, lambda payload: {"response": "safe-result"}
    )
    completed = _wait(restarted, job.status.operation_id)
    restarted_runtime.shutdown()
    first_registry.cancel(job.status.operation_id, OperationCancelCode.SHUTDOWN)
    release.set()
    runtime.shutdown()

    assert completed.status.status == OperationState.COMPLETED
    assert "PRIVATE_SENTINEL" not in second_path.read_text()


def test_fresh_registry_promotes_journal_committed_finalizing_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "public-result"},
        client_id="client",
        idempotency_key="committed",
        correlation_id="context",
    )
    completed = _wait(registry, job.status.operation_id)
    runtime.shutdown()
    values = json.loads(path.read_text())
    values[0]["pending_result"] = values[0].pop("result")
    values[0]["status"].update(
        status="finalizing",
        status_sequence=values[0]["status"]["status_sequence"] - 1,
        completed_at=None,
        result_available=False,
    )
    path.write_text(json.dumps(values))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        path,
        restarted_runtime,
        _result,
        committed_event_ids={completed.status.event_id},
    )
    recovered = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert recovered is not None
    assert recovered.status.status == OperationState.COMPLETED
    assert recovered.result == {"response": "public-result"}
