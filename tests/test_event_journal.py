from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import Event

import pytest

from kagya.runtime import (
    AgentEvent,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeJournalError,
    EventJournal,
    JournalIntegrityError,
    JournalLifecycle,
    hash_snapshot,
)
from kagya.runtime.agent_state import default_agent_state_snapshot


def test_journal_rejects_hidden_thought_payload(
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl")
    accepted = _event(
        payload={"prompt": "private prompt", "hidden_thought": "private thought"}
    )

    with pytest.raises(JournalIntegrityError, match="hidden thought"):
        journal.accepted(accepted)
    assert not journal.path.exists()


def test_journal_records_lifecycle_hash_chain_without_event_payload(
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl")
    accepted = _event(payload={"prompt": "private prompt"})
    started = replace(accepted, processing_sequence=1)
    before = default_agent_state_snapshot(1.0)
    after = before.model_copy(update={"last_processed_event_sequence": 1})

    journal.accepted(accepted)
    journal.started(started)
    journal.prepared(
        started,
        state_hash_before=hash_snapshot(before),
        state_hash_after=hash_snapshot(after),
    )
    journal.completed(started, hash_snapshot(after))

    restored = EventJournal(journal.path).verify()
    assert [record.lifecycle for record in restored] == [
        JournalLifecycle.ACCEPTED,
        JournalLifecycle.STARTED,
        JournalLifecycle.PREPARED,
        JournalLifecycle.COMPLETED,
    ]
    assert restored[-1].previous_record_hash == restored[-2].record_hash
    serialized = journal.path.read_text(encoding="utf-8")
    assert "private prompt" not in serialized
    assert "hidden_thought" not in serialized


def test_journal_rejects_tamper_and_unsupported_records(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl")
    journal.accepted(_event())
    payload = json.loads(journal.path.read_text(encoding="utf-8"))
    payload["source"] = "tampered"
    journal.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(JournalIntegrityError, match="hash mismatch"):
        EventJournal(journal.path)

    payload["schema_version"] = 4
    journal.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="invalid"):
        EventJournal(journal.path)


def test_reconcile_classifies_snapshot_saved_after_prepared_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.jsonl"
    journal = EventJournal(path)
    event = replace(_event(), processing_sequence=1)
    before = default_agent_state_snapshot(1.0)
    after = before.model_copy(update={"last_processed_event_sequence": 1})
    journal.accepted(replace(event, processing_sequence=None))
    journal.started(event)
    journal.prepared(
        event,
        state_hash_before=hash_snapshot(before),
        state_hash_after=hash_snapshot(after),
    )

    recovered = EventJournal(path).reconcile(after)

    assert len(recovered) == 1
    assert recovered[0].lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
    assert recovered[0].failure_category == "committed_before_crash"
    assert EventJournal(path).reconcile(after) == []


def test_reconcile_classifies_started_event_without_new_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.jsonl"
    journal = EventJournal(path)
    event = replace(_event(), processing_sequence=1)
    snapshot = default_agent_state_snapshot(1.0)
    journal.accepted(replace(event, processing_sequence=None))
    journal.started(event)

    recovered = EventJournal(path).reconcile(snapshot)

    assert recovered[0].failure_category == "uncommitted_after_crash"
    assert EventJournal(path).reconcile(snapshot) == []


def test_existing_snapshot_bootstraps_journal_checkpoint(tmp_path: Path) -> None:
    snapshot = default_agent_state_snapshot(1.0).model_copy(
        update={"last_processed_event_sequence": 7}
    )
    journal = EventJournal(tmp_path / "journal.jsonl")

    records = journal.reconcile(snapshot)

    assert records[0].lifecycle == JournalLifecycle.CHECKPOINT
    assert records[0].processing_sequence == 7
    assert records[0].snapshot_hash == hash_snapshot(snapshot)


def test_rotation_keeps_verifiable_checkpointed_hash_chain(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl", max_bytes=1, retained_files=2)
    for sequence in range(1, 8):
        journal.started(replace(_event(), processing_sequence=sequence))

    records = EventJournal(journal.path, max_bytes=1, retained_files=2).verify()

    assert records
    assert any(record.lifecycle == JournalLifecycle.CHECKPOINT for record in records)
    assert records[-1].processing_sequence == 7


def test_journal_rejects_processing_sequence_gap(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = EventJournal(path)
    journal.started(replace(_event(), processing_sequence=1))
    journal.started(replace(_event(), processing_sequence=3))

    with pytest.raises(JournalIntegrityError, match="sequence gap"):
        EventJournal(path)


def test_journal_rejects_missing_rotation_file(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl", max_bytes=1, retained_files=3)
    for sequence in range(1, 5):
        journal.started(replace(_event(), processing_sequence=sequence))
    journal.path.with_name(f"{journal.path.name}.1").unlink()

    with pytest.raises(JournalIntegrityError, match="file sequence"):
        EventJournal(journal.path, max_bytes=1, retained_files=3)


def test_reconcile_rejects_snapshot_sequence_without_prepared_evidence(
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl")
    event = replace(_event(), processing_sequence=1)
    snapshot = default_agent_state_snapshot(1.0).model_copy(
        update={"last_processed_event_sequence": 1}
    )
    journal.started(event)
    journal.completed(event, hash_snapshot(snapshot))
    unexplained = snapshot.model_copy(update={"last_processed_event_sequence": 2})

    with pytest.raises(JournalIntegrityError, match="ahead of the journal"):
        EventJournal(journal.path).reconcile(unexplained)


def test_rotation_checkpoint_preserves_snapshot_sequence_and_hash(
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl", max_bytes=1, retained_files=1)
    event = replace(_event(), processing_sequence=1)
    snapshot = default_agent_state_snapshot(1.0).model_copy(
        update={"last_processed_event_sequence": 1}
    )
    journal.started(event)
    journal.completed(event, hash_snapshot(snapshot))
    journal.accepted(replace(_event(), event_id="event-2"))
    journal.accepted(replace(_event(), event_id="event-3"))

    restarted = EventJournal(journal.path, max_bytes=1, retained_files=1)
    records = restarted.verify()

    checkpoints = [
        record for record in records if record.lifecycle == JournalLifecycle.CHECKPOINT
    ]
    assert checkpoints[-1].snapshot_sequence == 1
    assert checkpoints[-1].snapshot_hash == hash_snapshot(snapshot)
    restarted.reconcile(snapshot)


def test_real_journal_fsync_failure_rejects_event_before_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kagya.runtime.event_journal as journal_module

    journal = EventJournal(tmp_path / "journal.jsonl")
    runtime = AgentRuntime(queue_capacity=1, event_journal=journal)
    runtime.start()
    ran = Event()
    monkeypatch.setattr(
        journal_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("fsync unavailable")),
    )

    with pytest.raises(AgentRuntimeJournalError, match="durably accepted"):
        runtime.submit(AgentEventType.CHAT, source="test", handler=lambda: ran.set())
    runtime.shutdown()

    assert ran.is_set() is False


def test_admin_audit_record_contains_actor_and_target_without_credentials(
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "journal.jsonl")

    record = journal.audit_admin_action(
        event_id="request-1",
        actor_id="operator@example.test",
        actor_role="full_admin",
        target="POST /api/state/reset",
        reauthenticated=True,
    )

    assert record.lifecycle == JournalLifecycle.AUDIT
    assert record.actor_id == "operator@example.test"
    assert record.target == "POST /api/state/reset"
    serialized = journal.path.read_text(encoding="utf-8")
    assert "admin-token" not in serialized
    assert "csrf" not in serialized.lower()


def _event(*, payload: dict[str, object] | None = None) -> AgentEvent:
    now = datetime.now(UTC)
    return AgentEvent(
        event_id="event-1",
        event_type=AgentEventType.CHAT,
        source="api.chat",
        observed_at=now,
        requested_at=now,
        payload=payload or {},
        causation_id="cause-1",
        correlation_id="correlation-1",
    )
