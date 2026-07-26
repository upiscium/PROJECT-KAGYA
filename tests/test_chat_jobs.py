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
    enter_finalization_boundary,
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
        tmp_path / "jobs.json",
        runtime,
        lambda payload: called.set() or _result(payload),
    )
    job, _ = registry.enqueue(
        {"text": "must-not-run"},
        client_id="client",
        idempotency_key="cancel-queued",
        correlation_id="context",
    )

    assert (
        registry.cancel(job.status.operation_id, OperationCancelCode.CLIENT_REQUEST)
        == "canceled"
    )
    release.set()
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.CANCELED
    assert terminal.status.cancel_code == OperationCancelCode.CLIENT_REQUEST
    assert terminal.requested_cancel_code == OperationCancelCode.CLIENT_REQUEST
    assert terminal.cancel_requested_at is not None
    assert terminal.result is None
    assert not called.is_set()


def test_running_cancel_aborts_unsupported_provider_before_commit(
    tmp_path: Path,
) -> None:
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
    assert (
        registry.cancel(job.status.operation_id, OperationCancelCode.CLIENT_REQUEST)
        == "cancel_requested"
    )
    release.set()
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.CANCELED
    assert terminal.result is None
    assert all(
        event.event != "token"
        for event in registry.events_after(job.status.operation_id, 0)
    )


def test_cancel_wins_when_paused_before_finalizing(tmp_path: Path) -> None:
    before_boundary = Event()
    release = Event()
    wrote = Event()

    def execute(payload: dict[str, object]) -> dict[str, object]:
        before_boundary.set()
        release.wait(2)
        enter_finalization_boundary()
        wrote.set()
        return _result(payload)

    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, execute)
    job, _ = registry.enqueue(
        {"text": "not-committed"},
        client_id="client",
        idempotency_key="before-finalizing",
        correlation_id="context",
    )

    assert before_boundary.wait(1)
    assert (
        registry.cancel(job.status.operation_id, OperationCancelCode.CLIENT_REQUEST)
        == "cancel_requested"
    )
    release.set()
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.CANCELED
    assert not wrote.is_set()


def test_cancel_is_rejected_when_paused_after_finalizing(tmp_path: Path) -> None:
    after_boundary = Event()
    release = Event()

    def execute(payload: dict[str, object]) -> dict[str, object]:
        enter_finalization_boundary()
        after_boundary.set()
        release.wait(2)
        return _result(payload)

    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(tmp_path / "jobs.json", runtime, execute)
    job, _ = registry.enqueue(
        {"text": "committed"},
        client_id="client",
        idempotency_key="after-finalizing",
        correlation_id="context",
    )

    assert after_boundary.wait(1)
    assert (
        registry.cancel(job.status.operation_id, OperationCancelCode.CLIENT_REQUEST)
        == "already_finalizing"
    )
    finalizing = registry.get(job.status.operation_id)
    assert finalizing is not None
    assert finalizing.status.cancel_requested is False
    release.set()
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.COMPLETED
    assert terminal.result == {"response": "committed"}


def test_timeout_deadline_survives_restart_without_extension(tmp_path: Path) -> None:
    source_path = tmp_path / "source" / "jobs.json"
    replay_path = tmp_path / "replay" / "jobs.json"
    source_runtime = AgentRuntime(queue_capacity=2)
    source_runtime.start()
    release = Event()
    source_runtime.submit(
        AgentEventType.STATE_SNAPSHOT,
        source="test.blocker",
        handler=lambda: release.wait(2),
    )
    source_registry = ChatJobRegistry(
        source_path, source_runtime, _result, timeout_seconds=0.25
    )
    job, _ = source_registry.enqueue(
        {"text": "timeout"},
        client_id="client",
        idempotency_key="timeout-restart",
        correlation_id="context",
    )
    original = source_registry.get(job.status.operation_id)
    assert original is not None and original.timeout_deadline is not None
    time.sleep(0.15)
    replay_path.parent.mkdir(parents=True)
    shutil.copy2(source_path, replay_path)
    shutil.copy2(
        source_path.with_suffix(".json.key"), replay_path.with_suffix(".json.key")
    )

    def wait_for_timeout(payload: dict[str, object]) -> dict[str, object]:
        time.sleep(0.2)
        cancellation_checkpoint()
        return _result(payload)

    replay_runtime = AgentRuntime(queue_capacity=2)
    replay_registry = ChatJobRegistry(
        replay_path, replay_runtime, wait_for_timeout, timeout_seconds=10.0
    )
    started = time.monotonic()
    replay_runtime.start()
    replay_registry.activate()
    terminal = _wait(replay_registry, job.status.operation_id)
    elapsed = time.monotonic() - started
    replay_runtime.shutdown()
    source_registry.cancel(job.status.operation_id, OperationCancelCode.SHUTDOWN)
    release.set()
    source_runtime.shutdown()

    assert terminal.status.status == OperationState.CANCELED
    assert terminal.status.cancel_code == OperationCancelCode.TIMEOUT
    assert terminal.requested_cancel_code == OperationCancelCode.TIMEOUT
    assert terminal.timeout_deadline == original.timeout_deadline
    assert elapsed < 1.0


def test_unknown_provider_cancel_code_fails_closed(tmp_path: Path) -> None:
    from kagya.runtime import OperationCanceled

    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda _payload: (_ for _ in ()).throw(OperationCanceled("unknown-code")),
    )
    job, _ = registry.enqueue(
        {"text": "failure"},
        client_id="client",
        idempotency_key="unknown-cancel",
        correlation_id="context",
    )
    terminal = _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert terminal.status.status == OperationState.FAILED
    assert terminal.status.error_code == "internal_error"


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

    assert [item.result["response"] for item in completed if item.result] == [
        "0",
        "1",
        "2",
    ]
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


def test_request_spool_is_durable_before_journal_acceptance(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    accepted_spool: list[dict[str, object]] = []

    class JournalProbe:
        def accepted(self, event) -> None:
            records = json.loads(path.read_text())
            assert records[0]["status"]["event_id"] == event.event_id
            accepted_spool.extend(records)

        def started(self, event) -> None:
            del event

        def completed(self, event, snapshot_hash: str) -> None:
            del event, snapshot_hash

        def failed(self, event, failure_category: str, snapshot_hash) -> None:
            del event, failure_category, snapshot_hash

    runtime = AgentRuntime(
        queue_capacity=2,
        event_journal=JournalProbe(),
        completion_hook=lambda event: "0" * 64,
    )
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "PRIVATE_SPOOL_SENTINEL"},
        client_id="client",
        idempotency_key="durable-before-journal",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()

    assert accepted_spool
    assert "PRIVATE_SPOOL_SENTINEL" not in json.dumps(accepted_spool)


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
    shutil.copy2(
        first_path.with_suffix(".json.key"), second_path.with_suffix(".json.key")
    )

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted = ChatJobRegistry(
        second_path, restarted_runtime, lambda payload: {"response": "safe-result"}
    )
    assert restarted.is_ready is False
    restarted_runtime.start()
    restarted.activate()
    assert restarted.is_ready is True
    completed = _wait(restarted, job.status.operation_id)
    restarted_runtime.shutdown()
    first_registry.cancel(job.status.operation_id, OperationCancelCode.SHUTDOWN)
    release.set()
    runtime.shutdown()

    assert completed.status.status == OperationState.COMPLETED
    assert "PRIVATE_SENTINEL" not in second_path.read_text()


def test_activate_replays_queued_jobs_once_in_enqueue_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source" / "jobs.json"
    replay_path = tmp_path / "replay" / "jobs.json"
    source_runtime = AgentRuntime(queue_capacity=4)
    source_runtime.start()
    release = Event()
    source_runtime.submit(
        AgentEventType.STATE_SNAPSHOT,
        source="test.blocker",
        handler=lambda: release.wait(2),
    )
    source_registry = ChatJobRegistry(source_path, source_runtime, _result)
    jobs = [
        source_registry.enqueue(
            {"text": str(index)},
            client_id="client",
            idempotency_key=f"replay-{index}",
            correlation_id="context",
        )[0]
        for index in range(3)
    ]
    replay_path.parent.mkdir(parents=True)
    shutil.copy2(source_path, replay_path)
    shutil.copy2(
        source_path.with_suffix(".json.key"), replay_path.with_suffix(".json.key")
    )

    executions: list[tuple[str, int | None]] = []

    def execute(payload: dict[str, object]) -> dict[str, object]:
        event = current_agent_event()
        assert event is not None
        executions.append((str(payload["text"]), event.processing_sequence))
        return _result(payload)

    replay_runtime = AgentRuntime(queue_capacity=1)
    replay_registry = ChatJobRegistry(replay_path, replay_runtime, execute)
    assert replay_registry.is_ready is False
    replay_runtime.start()
    replay_registry.activate()
    replay_registry.activate()
    completed = [
        _wait(replay_registry, job.status.operation_id) for job in jobs
    ]
    replay_runtime.shutdown()
    source_registry.cancel(jobs[0].status.operation_id, OperationCancelCode.SHUTDOWN)
    source_registry.cancel(jobs[1].status.operation_id, OperationCancelCode.SHUTDOWN)
    source_registry.cancel(jobs[2].status.operation_id, OperationCancelCode.SHUTDOWN)
    release.set()
    source_runtime.shutdown()

    assert replay_registry.is_ready is True
    assert [item.result["response"] for item in completed if item.result] == [
        "0",
        "1",
        "2",
    ]
    assert [text for text, _sequence in executions] == ["0", "1", "2"]
    assert [sequence for _text, sequence in executions] == [1, 2, 3]


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
        recovery_dispositions={completed.status.event_id: "committed"},
    )
    recovered = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert recovered is not None
    assert recovered.status.status == OperationState.COMPLETED
    assert recovered.result == {"response": "public-result"}


@pytest.mark.parametrize(
    ("terminal_state", "terminal_event"),
    [
        (OperationState.COMPLETED, "final"),
        (OperationState.FAILED, "error"),
        (OperationState.CANCELED, "canceled"),
    ],
)
def test_restart_terminal_event_id_exceeds_persisted_high_water_mark(
    tmp_path: Path,
    terminal_state: OperationState,
    terminal_event: str,
) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "public-result"},
        client_id="client",
        idempotency_key=f"restart-{terminal_state.value}",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()

    values = json.loads(path.read_text())
    values[0]["last_stream_sequence"] = 10_000
    values[0]["status"].update(
        status=terminal_state.value,
        status_sequence=20,
        completed_at=values[0]["status"]["updated_at"],
        result_available=terminal_state == OperationState.COMPLETED,
        error_code="internal_error"
        if terminal_state == OperationState.FAILED
        else None,
        cancel_code=(
            "client_request" if terminal_state == OperationState.CANCELED else None
        ),
    )
    if terminal_state != OperationState.COMPLETED:
        values[0]["result"] = None
    path.write_text(json.dumps(values))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(path, restarted_runtime, _result)
    events = restarted.events_after(job.status.operation_id, 10_000)
    restarted_runtime.shutdown()

    assert events
    assert all(event.event_id > 10_000 for event in events)
    assert events[-1].event == terminal_event


def test_missing_journal_accepted_spool_fails_registry_readiness(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        _result,
        required_event_ids={"accepted-without-spool"},
    )
    registry.activate()
    runtime.shutdown()

    assert registry.is_ready is False


def test_completion_journal_failure_preserves_indeterminate_pending_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"

    class JournalProbe:
        def accepted(self, event) -> None:
            del event

        def started(self, event) -> None:
            del event

        def completed(self, event, snapshot_hash: str) -> None:
            del event, snapshot_hash
            raise OSError("journal unavailable")

        def failed(self, event, failure_category: str, snapshot_hash) -> None:
            del event, failure_category, snapshot_hash

    runtime = AgentRuntime(
        queue_capacity=2,
        event_journal=JournalProbe(),
        completion_hook=lambda event: "0" * 64,
    )
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "public-result"},
        client_id="client",
        idempotency_key="journal-failure",
        correlation_id="context",
    )

    deadline = time.monotonic() + 3
    current = None
    while time.monotonic() < deadline:
        current = registry.get(job.status.operation_id)
        if current is not None and current.status.error_code is not None:
            break
        time.sleep(0.01)

    assert current is not None
    assert current.status.status == OperationState.FINALIZING
    assert current.status.error_code == "commit_indeterminate"
    assert current.pending_result == {"response": "public-result"}
    assert current.status.event_id == job.status.event_id
    assert runtime.is_accepting is False
    assert all(
        event.event not in {"token", "final"}
        for event in registry.events_after(job.status.operation_id, 0)
    )
    persisted = json.loads(path.read_text())[0]
    assert persisted["status"]["error_code"] == "commit_indeterminate"
    runtime.shutdown()


@pytest.mark.parametrize(
    ("disposition", "cancel_requested", "requested_code", "expected"),
    [
        ("uncommitted", False, None, OperationState.FAILED),
        ("uncommitted", True, "client_request", OperationState.CANCELED),
        ("uncommitted", True, "shutdown", OperationState.CANCELED),
        ("failed", False, None, OperationState.FAILED),
        ("canceled", False, None, OperationState.CANCELED),
    ],
)
def test_fresh_registry_reconciles_uncommitted_and_terminal_journal_outcomes(
    tmp_path: Path,
    disposition: str,
    cancel_requested: bool,
    requested_code: str | None,
    expected: OperationState,
) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "discard-me"},
        client_id="client",
        idempotency_key=f"recover-{disposition}-{cancel_requested}",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()
    values = json.loads(path.read_text())
    values[0]["pending_result"] = values[0].pop("result")
    values[0]["status"].update(
        status="finalizing",
        completed_at=None,
        result_available=False,
        error_code="commit_indeterminate",
        cancel_requested=cancel_requested,
    )
    if requested_code is not None:
        values[0]["requested_cancel_code"] = requested_code
    path.write_text(json.dumps(values))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        path,
        restarted_runtime,
        _result,
        recovery_dispositions={job.status.event_id: disposition},
    )
    recovered = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert recovered is not None
    assert recovered.status.status == expected
    assert recovered.pending_result is None
    assert recovered.result is None
    if requested_code is not None:
        assert recovered.status.cancel_code == OperationCancelCode(requested_code)


def test_v1_record_migrates_without_inventing_cancel_authority(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "legacy"},
        client_id="client",
        idempotency_key="legacy-v1",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()
    values = json.loads(path.read_text())
    values[0]["schema_version"] = 1
    for field in (
        "requested_cancel_code",
        "cancel_requested_at",
        "timeout_deadline",
    ):
        values[0].pop(field, None)
    path.write_text(json.dumps(values))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(path, restarted_runtime, _result)
    migrated = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert migrated is not None
    assert migrated.schema_version == 2
    assert migrated.requested_cancel_code is None
    assert migrated.cancel_requested_at is None


def test_fresh_registry_fails_closed_on_ambiguous_commit(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "preserve-me"},
        client_id="client",
        idempotency_key="ambiguous",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()
    values = json.loads(path.read_text())
    values[0]["pending_result"] = values[0].pop("result")
    values[0]["status"].update(
        status="finalizing",
        completed_at=None,
        result_available=False,
        error_code="commit_indeterminate",
    )
    path.write_text(json.dumps(values))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        path,
        restarted_runtime,
        _result,
        recovery_dispositions={job.status.event_id: "ambiguous"},
    )
    restarted.activate()
    recovered = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert restarted.is_ready is False
    assert recovered is not None
    assert recovered.status.status == OperationState.FINALIZING
    assert recovered.pending_result == {"response": "preserve-me"}


def test_uncommitted_journal_overrides_false_completed_registry_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "must-not-survive"},
        client_id="client",
        idempotency_key="false-completion",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        path,
        restarted_runtime,
        _result,
        recovery_dispositions={job.status.event_id: "uncommitted"},
    )
    recovered = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert recovered is not None
    assert recovered.status.status == OperationState.FAILED
    assert recovered.result is None


@pytest.mark.parametrize("reconstructed", [{"response": "recovered"}, None])
def test_committed_recovery_reconstructs_or_bounds_unavailable_result(
    tmp_path: Path, reconstructed: dict[str, str] | None
) -> None:
    path = tmp_path / "jobs.json"
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    registry = ChatJobRegistry(path, runtime, _result)
    job, _ = registry.enqueue(
        {"text": "original"},
        client_id="client",
        idempotency_key="reconstruct",
        correlation_id="context",
    )
    _wait(registry, job.status.operation_id)
    runtime.shutdown()
    values = json.loads(path.read_text())
    values[0]["result"] = None
    values[0]["pending_result"] = None
    values[0]["status"].update(
        status="finalizing",
        completed_at=None,
        result_available=False,
        error_code="commit_indeterminate",
    )
    path.write_text(json.dumps(values))

    restarted_runtime = AgentRuntime(queue_capacity=2)
    restarted_runtime.start()
    restarted = ChatJobRegistry(
        path,
        restarted_runtime,
        _result,
        recovery_dispositions={job.status.event_id: "committed"},
        result_reconstructor=lambda event_id: (
            reconstructed if event_id == job.status.event_id else None
        ),
    )
    recovered = restarted.get(job.status.operation_id)
    restarted_runtime.shutdown()

    assert recovered is not None
    assert recovered.status.status == OperationState.COMPLETED
    assert recovered.result == reconstructed
    if reconstructed is None:
        assert recovered.status.error_code == "committed_result_unavailable"
        assert recovered.status.result_available is False
    else:
        assert recovered.status.error_code is None
        assert recovered.status.result_available is True
