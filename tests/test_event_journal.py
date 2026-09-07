from datetime import datetime, timezone
import json
import os
from itertools import pairwise
from pathlib import Path
import traceback
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from kagya.config import Settings, load_settings
from kagya.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from kagya.runtime.event_journal import (
    EventFailureCategory,
    EventJournal,
    EventJournalAppendError,
    EventJournalAppendStage,
    EventJournalIntegrityError,
    EventJournalLoadError,
    EventJournalRecord,
    EventLifecycle,
    UnsupportedEventJournalVersion,
)
import kagya.runtime.event_journal as journal_module


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R05"
HASH_0 = "a" * 64
HASH_1 = "b" * 64
HASH_2 = "c" * 64
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def event(number: str = "e1", sequence: int | None = None) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, number)),
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        NOW,
        sequence,
    )


def journal(path: Path, *, max_bytes: int = 100_000, retained: int = 4) -> EventJournal:
    return EventJournal(path, max_bytes, retained, clock=lambda: NOW)


def bootstrap(path: Path, snapshot_hash: str = HASH_0) -> EventJournal:
    value = journal(path)
    value.verify_and_reconcile(0, snapshot_hash)
    return value


def append_success(
    value: EventJournal,
    item: AgentEvent,
    before_hash: str,
    after_hash: str,
) -> None:
    assert item.processing_sequence is not None
    value.append_accepted(
        AgentEvent(
            item.event_id,
            item.event_type,
            item.source,
            item.requested_at,
        )
    )
    value.append_started(item)
    value.append_prepared(item, before_hash, after_hash)
    value.append_completed(item, item.processing_sequence, after_hash)


def assert_bounded(error: Exception, sentinel: str = PRIVATE_SENTINEL) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert sentinel not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def test_success_lifecycle_is_durable_chained_and_private_free(tmp_path: Path) -> None:
    stages: list[tuple[EventLifecycle, EventJournalAppendStage]] = []
    path = tmp_path / "private" / "events.jsonl"
    value = EventJournal(
        path,
        100_000,
        4,
        clock=lambda: NOW,
        append_stage_hook=lambda lifecycle, stage: stages.append((lifecycle, stage)),
    )
    value.verify_and_reconcile(0, HASH_0)
    append_success(value, event("e1", 1), HASH_0, HASH_1)

    assert [record.lifecycle for record in value.records] == [
        EventLifecycle.CHECKPOINT,
        EventLifecycle.ACCEPTED,
        EventLifecycle.STARTED,
        EventLifecycle.PREPARED,
        EventLifecycle.COMPLETED,
    ]
    assert all(
        record.previous_record_hash == previous.record_hash
        for previous, record in pairwise(value.records)
    )
    assert stages == [
        (lifecycle, stage)
        for lifecycle in (
            EventLifecycle.CHECKPOINT,
            EventLifecycle.ACCEPTED,
            EventLifecycle.STARTED,
            EventLifecycle.PREPARED,
            EventLifecycle.COMPLETED,
        )
        for stage in (
            EventJournalAppendStage.WRITE,
            EventJournalAppendStage.FILE_FSYNC,
            EventJournalAppendStage.PARENT_FSYNC,
        )
    ]
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert all(
        json.loads(line)["schema_version"] == 1
        for line in path.read_bytes().splitlines()
    )
    assert PRIVATE_SENTINEL not in path.read_text(encoding="utf-8")
    assert value.verify_and_reconcile(1, HASH_1).processing_high_water == 1


def test_schema_is_strict_and_has_no_metadata_escape_hatch() -> None:
    with pytest.raises(ValidationError):
        EventJournalRecord(
            record_id=str(uuid5(NAMESPACE_URL, "record")),
            timestamp=NOW,
            lifecycle=EventLifecycle.ACCEPTED,
            event_id=event().event_id,
            event_type=AgentEventType.CHAT,
            source=AgentEventSource.API_CHAT,
            record_hash="0" * 64,
            metadata={"prompt": PRIVATE_SENTINEL},
        )

    with pytest.raises(ValidationError):
        EventJournalRecord.model_validate(
            {
                "record_id": str(uuid5(NAMESPACE_URL, "versionless-record")),
                "timestamp": NOW,
                "lifecycle": EventLifecycle.ACCEPTED,
                "event_id": event().event_id,
                "event_type": AgentEventType.CHAT,
                "source": AgentEventSource.API_CHAT,
                "record_hash": "0" * 64,
            }
        )


def test_journal_config_is_strict_positive_and_backward_compatible() -> None:
    settings = load_settings(CONFIG_PATH)
    assert settings.event_journal.path == Path(".kagya/event_journal.jsonl")
    assert settings.event_journal.max_bytes == 1_048_576
    assert settings.event_journal.retained_files == 4

    pre_r05 = settings.model_dump(mode="python")
    pre_r05.pop("event_journal")
    compatible = Settings.model_validate(pre_r05)
    assert compatible.event_journal.path == Path(".kagya/event_journal.jsonl")
    with pytest.raises(ValidationError):
        type(settings.event_journal).model_validate(
            {"path": "journal", "max_bytes": 0, "retained_files": 1}
        )


def test_pre_r05_snapshot_gets_stable_checkpoint_anchor(tmp_path: Path) -> None:
    value = journal(tmp_path / "events.jsonl")

    recovery = value.verify_and_reconcile(7, HASH_1)

    assert recovery.processing_high_water == 7
    assert recovery.snapshot_sequence == 7
    assert value.records[0].lifecycle is EventLifecycle.CHECKPOINT
    assert value.records[0].processing_sequence == 7
    assert value.records[0].snapshot_sequence == 7
    assert value.records[0].snapshot_hash == HASH_1


def test_accepted_only_is_classified_without_consuming_sequence(tmp_path: Path) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    value.append_accepted(event())
    value.close()

    reopened = EventJournal(value.path, 100_000, 4)
    recovery = reopened.verify_and_reconcile(0, HASH_0)

    assert recovery.processing_high_water == 0
    terminal = reopened.records[-1]
    assert terminal.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED
    assert terminal.processing_sequence is None


@pytest.mark.parametrize("prepared", [False, True])
def test_started_or_prepared_with_old_snapshot_is_uncommitted(
    tmp_path: Path, prepared: bool
) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    value.append_accepted(event())
    value.append_started(event(sequence=1))
    if prepared:
        value.append_prepared(event(sequence=1), HASH_0, HASH_1)
    value.close()

    reopened = EventJournal(value.path, 100_000, 4)
    recovery = reopened.verify_and_reconcile(0, HASH_0)

    assert recovery.processing_high_water == 1
    assert reopened.records[-1].failure_category is (
        EventFailureCategory.UNCOMMITTED_AFTER_CRASH
    )


def test_prepared_with_matching_snapshot_is_committed_before_crash(
    tmp_path: Path,
) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    value.append_accepted(event())
    value.append_started(event(sequence=1))
    value.append_prepared(event(sequence=1), HASH_0, HASH_1)
    value.close()

    reopened = EventJournal(value.path, 100_000, 4)
    recovery = reopened.verify_and_reconcile(1, HASH_1)

    assert recovery.processing_high_water == 1
    assert reopened.records[-1].failure_category is (
        EventFailureCategory.COMMITTED_BEFORE_CRASH
    )


def test_failed_sequence_is_consumed_above_snapshot_sequence(tmp_path: Path) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    value.append_accepted(event("failed"))
    value.append_started(event("failed", 1))
    value.append_failed(event("failed", 1), 0, HASH_0)
    value.append_accepted(event("next"))
    value.append_started(event("next", 2))
    value.append_prepared(event("next", 2), HASH_0, HASH_2)
    value.append_completed(event("next", 2), 2, HASH_2)
    records = value.records
    value.close()

    reopened = EventJournal(value.path, 100_000, 4)
    recovery = reopened.verify_and_reconcile(2, HASH_2)

    assert recovery.processing_high_water == 2
    failed = next(
        record for record in records if record.lifecycle is EventLifecycle.FAILED
    )
    assert failed.processing_sequence == 1
    assert failed.snapshot_sequence == 0


@pytest.mark.parametrize(
    ("snapshot_sequence", "snapshot_hash"),
    [(1, HASH_1), (0, HASH_2)],
)
def test_snapshot_ahead_or_hash_mismatch_fails_closed(
    tmp_path: Path, snapshot_sequence: int, snapshot_hash: str
) -> None:
    value = bootstrap(tmp_path / "events.jsonl")

    with pytest.raises(EventJournalIntegrityError):
        value.verify_and_reconcile(snapshot_sequence, snapshot_hash)


def test_completed_ahead_of_snapshot_fails_closed(tmp_path: Path) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    append_success(value, event("e1", 1), HASH_0, HASH_1)

    with pytest.raises(EventJournalIntegrityError):
        value.verify_and_reconcile(0, HASH_0)


def test_partial_unsupported_tamper_and_chain_break_fail_closed(tmp_path: Path) -> None:
    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(b'{"schema_version":1}')
    with pytest.raises(EventJournalLoadError):
        journal(partial)

    unsupported = tmp_path / "unsupported.jsonl"
    unsupported.write_text('{"schema_version":999}\n', encoding="utf-8")
    with pytest.raises(UnsupportedEventJournalVersion):
        journal(unsupported)

    versionless = tmp_path / "versionless.jsonl"
    value = bootstrap(versionless)
    value.close()
    raw_record = json.loads(versionless.read_text(encoding="utf-8"))
    raw_record.pop("schema_version")
    versionless.write_text(
        json.dumps(raw_record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EventJournalLoadError):
        journal(versionless)

    tampered = tmp_path / "tampered.jsonl"
    value = bootstrap(tampered)
    value.append_accepted(event())
    value.close()
    tampered_lines = tampered.read_bytes().splitlines()
    tampered_record = json.loads(tampered_lines[1])
    tampered_record["event_type"] = AgentEventType.DEBUG_CHAT.value
    tampered.write_bytes(
        tampered_lines[0]
        + b"\n"
        + json.dumps(tampered_record, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    with pytest.raises(EventJournalIntegrityError):
        journal(tampered)

    broken = tmp_path / "broken.jsonl"
    value = bootstrap(broken)
    value.append_accepted(event())
    lines = broken.read_bytes().splitlines()
    record = EventJournalRecord.model_validate_json(lines[1])
    changed = record.model_copy(update={"previous_record_hash": HASH_2})
    changed = changed.model_copy(
        update={"record_hash": journal_module.EventJournal._record_hash(changed)}
    )
    value.close()
    broken.write_bytes(lines[0] + b"\n" + value._record_bytes(changed))
    with pytest.raises(EventJournalIntegrityError):
        journal(broken)


def test_processing_sequence_gap_is_rejected_before_append(tmp_path: Path) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    value.append_accepted(event())

    with pytest.raises(EventJournalAppendError) as error:
        value.append_started(event(sequence=2))

    assert error.value.stage is EventJournalAppendStage.VALIDATE
    assert error.value.published is False


def test_fifo_and_single_processing_order_is_enforced_before_append(
    tmp_path: Path,
) -> None:
    value = bootstrap(tmp_path / "events.jsonl")
    accepted_a = event("fifo-a")
    accepted_b = event("fifo-b")
    value.append_accepted(accepted_a)
    value.append_accepted(accepted_b)

    with pytest.raises(EventJournalAppendError):
        value.append_started(event("fifo-b", 1))

    value.append_started(event("fifo-a", 1))
    with pytest.raises(EventJournalAppendError):
        value.append_started(event("fifo-b", 2))

    value.append_prepared(event("fifo-a", 1), HASH_0, HASH_1)
    value.append_completed(event("fifo-a", 1), 1, HASH_1)
    value.append_started(event("fifo-b", 2))
    value.append_failed(event("fifo-b", 2), 1, HASH_1)

    assert value.verify_and_reconcile(1, HASH_1).processing_high_water == 2


@pytest.mark.parametrize("overlapping_start", [False, True])
def test_canonically_hashed_out_of_order_start_fails_closed_on_load(
    tmp_path: Path, overlapping_start: bool
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap(path)
    value.append_accepted(event("raw-a"))
    value.append_accepted(event("raw-b"))
    if overlapping_start:
        value.append_started(event("raw-a", 1))
        sequence = 2
    else:
        sequence = 1
    records = value.records
    invalid = value._make_record(
        EventLifecycle.STARTED,
        previous_hash=records[-1].record_hash,
        event=event("raw-b", sequence),
        processing_sequence=sequence,
    )
    value.close()
    path.write_bytes(
        b"".join(EventJournal._record_bytes(item) for item in (*records, invalid))
    )

    with pytest.raises(EventJournalIntegrityError):
        journal(path)


def test_accepted_not_started_recovery_must_preserve_snapshot_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap(path)
    accepted = event("accepted-recovery")
    value.append_accepted(accepted)
    records = value.records
    invalid = value._make_record(
        EventLifecycle.RECOVERY_CLASSIFIED,
        previous_hash=records[-1].record_hash,
        event=accepted,
        snapshot_sequence=1,
        snapshot_hash=HASH_1,
        failure_category=EventFailureCategory.ACCEPTED_NOT_STARTED,
    )
    value.close()
    path.write_bytes(
        b"".join(EventJournal._record_bytes(item) for item in (*records, invalid))
    )

    with pytest.raises(EventJournalIntegrityError):
        journal(path)


def test_recovery_closes_processing_then_queued_events_in_fifo_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap(path)
    value.append_accepted(event("recover-a"))
    value.append_started(event("recover-a", 1))
    value.append_accepted(event("recover-b"))
    value.append_accepted(event("recover-c"))
    value.close()

    reopened = journal(path)
    recovery = reopened.verify_and_reconcile(0, HASH_0)
    classifications = [
        record
        for record in reopened.records
        if record.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED
    ]

    assert recovery.processing_high_water == 1
    assert [record.event_id for record in classifications] == [
        event("recover-a").event_id,
        event("recover-b").event_id,
        event("recover-c").event_id,
    ]
    assert [record.failure_category for record in classifications] == [
        EventFailureCategory.UNCOMMITTED_AFTER_CRASH,
        EventFailureCategory.ACCEPTED_NOT_STARTED,
        EventFailureCategory.ACCEPTED_NOT_STARTED,
    ]


def test_rotation_retains_verifiable_checkpoint_and_detects_missing_segment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = EventJournal(path, 1, 4, clock=lambda: NOW)
    value.verify_and_reconcile(0, HASH_0)
    append_success(value, event("e1", 1), HASH_0, HASH_1)
    append_success(value, event("e2", 2), HASH_1, HASH_2)
    append_success(value, event("e3", 3), HASH_2, HASH_0)
    append_success(value, event("e4", 4), HASH_0, HASH_1)
    value.close()

    reopened = EventJournal(path, 1, 4, clock=lambda: NOW)
    assert reopened.verify_and_reconcile(4, HASH_1).processing_high_water == 4
    rotated = sorted(tmp_path.glob("events.jsonl.[0-9]*"))
    assert rotated
    assert all(
        EventJournalRecord.model_validate_json(
            segment.read_bytes().splitlines()[0]
        ).lifecycle
        is EventLifecycle.CHECKPOINT
        for segment in rotated
    )
    reopened.close()

    if len(rotated) >= 2:
        rotated[-2].unlink()
        with pytest.raises(EventJournalIntegrityError):
            EventJournal(path, 1, 4, clock=lambda: NOW)


def test_missing_sole_rotated_predecessor_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = EventJournal(path, 1, 2, clock=lambda: NOW)
    value.verify_and_reconcile(0, HASH_0)
    append_success(value, event("rotate-once", 1), HASH_0, HASH_1)
    value.close()
    predecessor = tmp_path / "events.jsonl.00000000"
    assert predecessor.exists()
    active_checkpoint = EventJournalRecord.model_validate_json(
        path.read_bytes().splitlines()[0]
    )
    assert active_checkpoint.previous_record_hash is not None

    predecessor.unlink()

    with pytest.raises(EventJournalIntegrityError):
        EventJournal(path, 1, 2, clock=lambda: NOW)


def test_interrupted_rotation_and_symlink_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    (tmp_path / ".events.jsonl.rotation.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(EventJournalIntegrityError):
        journal(path)

    (tmp_path / ".events.jsonl.rotation.tmp").unlink()
    target = tmp_path / "target.jsonl"
    target.write_text("private", encoding="utf-8")
    path.symlink_to(target)
    with pytest.raises(EventJournalLoadError):
        journal(path)


def test_owner_owned_permissive_parent_is_hardened(tmp_path: Path) -> None:
    permissive = tmp_path / "permissive"
    permissive.mkdir(mode=0o755)
    permissive.chmod(0o755)

    value = journal(permissive / "events.jsonl")

    assert permissive.stat().st_mode & 0o777 == 0o700
    assert (permissive / ".events.jsonl.lock").stat().st_mode & 0o777 == 0o600
    value.close()


def test_permission_hardening_failure_is_bounded_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permissive = tmp_path / "permissive"
    permissive.mkdir(mode=0o755)
    permissive.chmod(0o755)
    real_fchmod = os.fchmod

    def fail_directory_hardening(descriptor: int, mode: int) -> None:
        if mode == 0o700:
            raise PermissionError(PRIVATE_SENTINEL)
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(journal_module.os, "fchmod", fail_directory_hardening)

    with pytest.raises(EventJournalLoadError) as error:
        journal(permissive / "events.jsonl")

    assert_bounded(error.value)
    assert permissive.stat().st_mode & 0o777 == 0o755
    assert not (permissive / ".events.jsonl.lock").exists()


def test_unsafe_parent_and_lock_targets_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_owner = tmp_path / "wrong-owner"
    wrong_owner.mkdir(mode=0o700)
    effective_uid = os.geteuid()
    monkeypatch.setattr(journal_module.os, "geteuid", lambda: effective_uid + 1)
    with pytest.raises(EventJournalLoadError):
        journal(wrong_owner / "events.jsonl")
    monkeypatch.undo()

    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("state", encoding="utf-8")
    with pytest.raises(EventJournalLoadError):
        journal(non_directory / "events.jsonl")

    target_directory = tmp_path / "target-directory"
    target_directory.mkdir(mode=0o700)
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(target_directory, target_is_directory=True)
    with pytest.raises(EventJournalLoadError):
        journal(linked_directory / "events.jsonl")

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = private / "target.lock"
    target.write_text("attacker", encoding="utf-8")
    (private / ".events.jsonl.lock").symlink_to(target)
    with pytest.raises(EventJournalLoadError):
        journal(private / "events.jsonl")


def test_full_load_and_append_tracebacks_exclude_private_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap(path)
    value.close()
    real_open = os.open

    def fail_open(target, flags, mode=0o777, *, dir_fd=None):
        if Path(target) == path and flags & (os.O_WRONLY | os.O_RDWR) == 0:
            raise PermissionError(PRIVATE_SENTINEL)
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(journal_module.os, "open", fail_open)
    with pytest.raises(EventJournalLoadError) as load_error:
        EventJournal(path, 100_000, 4)
    assert_bounded(load_error.value)

    monkeypatch.setattr(journal_module.os, "open", real_open)

    def fail_append(lifecycle: EventLifecycle, _stage: EventJournalAppendStage) -> None:
        if lifecycle is EventLifecycle.ACCEPTED:
            raise OSError(PRIVATE_SENTINEL)

    value = EventJournal(path, 100_000, 4, append_stage_hook=fail_append)
    with pytest.raises(EventJournalAppendError) as append_error:
        value.append_accepted(event())
    assert_bounded(append_error.value)


def test_second_process_authority_is_rejected_while_lock_is_held(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap(path)

    with pytest.raises(EventJournalLoadError):
        EventJournal(path, 100_000, 4)

    value.close()
    replacement = EventJournal(path, 100_000, 4)
    with pytest.raises(EventJournalLoadError):
        _ = value.records
    with pytest.raises(EventJournalLoadError):
        value.append_accepted(event("late"))
    with pytest.raises(EventJournalLoadError):
        value.verify_and_reconcile(0, HASH_0)
    replacement.close()


def test_unknown_field_private_value_is_absent_from_full_traceback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"schema_version": 1, "unexpected": PRIVATE_SENTINEL}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EventJournalLoadError) as error:
        journal(path)

    assert_bounded(error.value)
