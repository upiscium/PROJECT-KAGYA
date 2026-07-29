from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from threading import Event
import time

import pytest

from kagya.chat_jobs import (
    ChatJobIdempotencyResultExpired,
    ChatJobIdempotencyTombstone,
)
from tests.chat_job_helpers import ChatJobRegistry
from kagya.operation_status import OperationCancelCode, OperationState
from kagya.runtime import AgentEventType, AgentRuntime


def _wait_terminal(registry: ChatJobRegistry, operation_id: str):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        record = registry.get(operation_id)
        if (
            record is not None
            and record.status.status
            in {OperationState.COMPLETED, OperationState.FAILED, OperationState.CANCELED}
            and record.terminal_projection_state
            not in {"pending", "projection_failed"}
        ):
            return record
        time.sleep(0.01)
    raise AssertionError("chat job did not become terminal")


def _enqueue(registry: ChatJobRegistry, key: str = "key"):
    return registry.enqueue(
        {"text": key},
        client_id="client",
        idempotency_key=key,
        correlation_id="context",
    )[0]


@pytest.mark.parametrize("terminal", ["completed", "failed", "canceled"])
def test_terminal_records_clear_sealed_request(
    tmp_path: Path, terminal: str
) -> None:
    runtime = AgentRuntime(queue_capacity=2)

    def execute(payload):
        if terminal == "failed":
            raise RuntimeError("failed")
        return {"response": str(payload["text"])}

    if terminal == "canceled":
        runtime.start()
        release = Event()
        runtime.submit(
            AgentEventType.STATE_SNAPSHOT,
            source="test.retention.blocker",
            handler=lambda: release.wait(2),
        )
    else:
        runtime.start()
    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, execute)
    job = _enqueue(registry)
    if terminal == "canceled":
        registry.cancel(job.status.operation_id, OperationCancelCode.CLIENT_REQUEST)
        release.set()
    record = _wait_terminal(registry, job.status.operation_id)
    runtime.shutdown()
    registry.close()

    assert record.sealed_request == ""
    persisted = json.loads((tmp_path / "jobs.json").read_text())
    assert persisted[0]["sealed_request"] == ""


def test_full_record_becomes_tombstone_then_expires(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        path,
        runtime,
        lambda payload: {"response": str(payload["text"])},
        result_retention_seconds=10,
        idempotency_retention_seconds=20,
    )
    job = _enqueue(registry, "lifecycle")
    terminal = _wait_terminal(registry, job.status.operation_id)
    completed_at = terminal.status.completed_at
    assert completed_at is not None

    registry.compact(completed_at + timedelta(seconds=11))
    runtime.shutdown()
    registry.close()

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        path,
        restarted_runtime,
        lambda payload: {"response": str(payload["text"])},
        result_retention_seconds=10,
        idempotency_retention_seconds=20,
    )
    with pytest.raises(ChatJobIdempotencyResultExpired):
        restarted.enqueue(
            {"text": "ignored"},
            client_id="client",
            idempotency_key="lifecycle",
            correlation_id="other-context",
        )

    assert restarted.get(job.status.operation_id) is None
    persisted = json.loads(path.read_text())
    tombstone = ChatJobIdempotencyTombstone.model_validate(persisted[0])
    assert tombstone.status.operation_id == job.status.operation_id
    assert "result" not in persisted[0]
    assert "sealed_request" not in persisted[0]

    restarted.compact(completed_at + timedelta(seconds=20))
    replacement, replacement_created = restarted.enqueue(
        {"text": "new"},
        client_id="client",
        idempotency_key="lifecycle",
        correlation_id="new-context",
    )
    restarted_runtime.shutdown()
    restarted.close()

    assert replacement_created is True
    assert replacement.status.operation_id != job.status.operation_id


def test_compaction_protects_active_finalizing_ambiguous_and_projection_pending(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    release_active = Event()

    def execute(payload):
        if payload["text"] == "active":
            release_active.wait(2)
        return {"response": str(payload["text"])}

    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        execute,
        result_retention_seconds=0,
        idempotency_retention_seconds=100,
        max_terminal_records=1,
        recovery_dispositions={"ambiguous-event": "ambiguous"},
    )
    active = _enqueue(registry, "active")

    with registry._lock:
        active_record = registry._records[active.status.operation_id]
        finalizing = active_record.model_copy(deep=True)
        finalizing.status = active_record.status.model_copy(
            update={
                "operation_id": "finalizing",
                "event_id": "finalizing-event",
                "status": OperationState.FINALIZING,
                "queue_position": None,
                "result_available": False,
            }
        )
        ambiguous = finalizing.model_copy(deep=True)
        ambiguous.status = finalizing.status.model_copy(
            update={
                "operation_id": "ambiguous",
                "event_id": "ambiguous-event",
                "status": OperationState.FAILED,
                "completed_at": finalizing.status.updated_at,
            }
        )
        pending = ambiguous.model_copy(deep=True)
        pending.status = ambiguous.status.model_copy(
            update={"operation_id": "pending", "event_id": "pending-event"}
        )
        pending.terminal_projection_state = "projection_failed"
        registry._records["finalizing"] = finalizing
        registry._records["ambiguous"] = ambiguous
        registry._records["pending"] = pending
        registry._persist()

    registry.compact(finalizing.status.updated_at + timedelta(days=30))

    assert registry.get(active.status.operation_id) is not None
    assert registry.get("finalizing") is not None
    assert registry.get("ambiguous") is not None
    assert registry.get("pending") is not None
    release_active.set()
    runtime.shutdown()
    registry.close()


def test_compaction_is_atomic_when_persistence_fails(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        path,
        runtime,
        lambda payload: {"response": str(payload["text"])},
        result_retention_seconds=10,
    )
    job = _enqueue(registry, "atomic")
    terminal = _wait_terminal(registry, job.status.operation_id)
    before = path.read_bytes()
    persist = registry._persist

    def fail_persist(*_args, **_kwargs):
        raise OSError("disk unavailable")

    registry._persist = fail_persist
    completed_at = terminal.status.completed_at
    assert completed_at is not None
    with pytest.raises(OSError, match="disk unavailable"):
        registry.compact(completed_at + timedelta(seconds=11))
    registry._persist = persist
    runtime.shutdown()
    registry.close()

    assert registry.get(job.status.operation_id) is not None
    assert path.read_bytes() == before


def test_capacity_compaction_and_metrics_have_bounded_labels(tmp_path: Path) -> None:
    class Metrics:
        def __init__(self) -> None:
            self.counters = []
            self.gauges = []

        def counter(self, name, amount=1.0, **labels):
            self.counters.append((name, amount, labels))

        def gauge(self, name, value, **labels):
            self.gauges.append((name, value, labels))

        def observe(self, name, value, **labels):
            self.gauges.append((name, value, labels))

    metrics = Metrics()
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda payload: {"response": str(payload["text"])},
        result_retention_seconds=1000,
        max_terminal_records=1,
        cleanup_interval_seconds=0,
        metrics=metrics,
    )
    first = _enqueue(registry, "first")
    _wait_terminal(registry, first.status.operation_id)
    second = _enqueue(registry, "second")
    _wait_terminal(registry, second.status.operation_id)
    registry.compact()

    with pytest.raises(ChatJobIdempotencyResultExpired):
        registry.enqueue(
            {"text": "duplicate"},
            client_id="client",
            idempotency_key="first",
            correlation_id="context",
        )
    runtime.shutdown()
    registry.close()

    persisted = json.loads((tmp_path / "jobs.json").read_text())
    assert len(persisted) == 2
    assert sum("enqueue_sequence" in item for item in persisted) == 1
    assert any(
        item.get("record_type") == "idempotency_tombstone"
        and item["status"]["operation_id"] == first.status.operation_id
        for item in persisted
    )
    assert any(labels == {"action": "capacity"} for _, _, labels in metrics.counters)
    assert {tuple(sorted(labels.items())) for _, _, labels in metrics.gauges} <= {
        (("kind", "full"),),
        (("kind", "tombstone"),),
        (),
    }
    assert any(name == "kagya_chat_job_registry_bytes" for name, _, _ in metrics.gauges)
    assert any(
        name == "kagya_chat_job_cleanup_duration_seconds"
        for name, _, _ in metrics.gauges
    )


def test_cleanup_is_throttled_but_manual_compaction_is_forced(tmp_path: Path) -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda payload: {"response": str(payload["text"])},
        cleanup_interval_seconds=3600,
    )
    cleanups = 0
    compact = registry._compact_locked

    def count_cleanup(now):
        nonlocal cleanups
        cleanups += 1
        compact(now)

    registry._compact_locked = count_cleanup
    job = _enqueue(registry, "throttled")
    _wait_terminal(registry, job.status.operation_id)

    assert cleanups == 0
    registry.compact()
    runtime.shutdown()
    registry.close()

    assert cleanups == 1


def test_tombstone_duplicate_api_returns_bounded_conflict(tmp_path: Path) -> None:
    from tests.test_fastapi_backend import _client

    client = _client(tmp_path)
    headers = {
        "Idempotency-Key": "expired-result",
        "X-KAGYA-Client-ID": "retention-client",
    }
    response = client.post(
        "/api/chat/jobs",
        json={"text": "hello", "attachments": []},
        headers=headers,
    )
    operation_id = response.json()["operation"]["operation_id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        record = client.app.state.chat_job_registry.get(operation_id)
        if record is not None and record.terminal_projection_state == "published":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("chat job projection did not complete")
    completed_at = record.status.completed_at
    assert completed_at is not None
    client.app.state.chat_job_registry.compact(
        completed_at + timedelta(days=2)
    )

    duplicate = client.post(
        "/api/chat/jobs",
        json={"text": "must not execute", "attachments": []},
        headers=headers,
    )

    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "idempotency_result_expired"}
