from datetime import datetime, timezone
import hashlib
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
    AbortOutcome,
    EventFailureCategory,
    EventRecoveryCategory,
    EventJournal,
    EventJournalAppendError,
    EventJournalAppendStage,
    EventJournalIntegrityError,
    EventJournalLoadError,
    EventJournalRecord,
    EventLifecycle,
    ParticipantBaseline,
    ParticipantCapability,
    ParticipantDomain,
    UnsupportedEventJournalVersion,
    ParticipantOutcome,
    ParticipantRequirement,
    ReconciliationReason,
    StartupParticipantOutcome,
    TransactionKind,
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


PARTICIPANTS = ("memory.episodic", "session.turns")
TX_ID = str(uuid5(NAMESPACE_URL, "transaction"))


def bootstrap_v3(path: Path) -> EventJournal:
    value = journal(path)
    generation = str(uuid5(NAMESPACE_URL, "v2-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "v2-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.append_v3_migration_checkpoint()
    return value


STARTUP_RECONCILIATION_ID = str(uuid5(NAMESPACE_URL, "startup-reconciliation"))
STARTUP_RECOVERY_ID = str(uuid5(NAMESPACE_URL, "startup-recovery"))


def startup_reconciliation_journal(path: Path) -> EventJournal:
    value = journal(path)
    generation = str(uuid5(NAMESPACE_URL, "startup-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "startup-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_2)
    value.append_v3_migration_checkpoint()
    append_u5_baseline(value)
    advance = event("startup-baseline-advance", 1)
    value.append_accepted(advance)
    value.append_started(advance)
    value.append_prepared(advance, HASH_0, HASH_1, generation)
    value.append_completed(
        advance, 1, HASH_1, generation, wal_id, HASH_2
    )
    anchor = value.records[0]
    value.append_recovery_prepared(
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.TRUE_ROLLBACK,
        processing_high_water=1,
    )
    value.append_recovery_completed(
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.TRUE_ROLLBACK,
        True,
        processing_high_water=1,
        wal_record_id=anchor.wal_record_id or wal_id,
        wal_record_hash=anchor.wal_record_hash or HASH_2,
    )
    return value


def transaction_requirements() -> tuple[ParticipantRequirement, ...]:
    return tuple(
        ParticipantRequirement(
            participant_id=participant,
            operation_digest=digest,
            capabilities=(
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.PREPARE,
            ),
        )
        for participant, digest in zip(PARTICIPANTS, (HASH_1, HASH_2), strict=True)
    )


U5_BASELINE_ID = str(uuid5(NAMESPACE_URL, "u5-baseline"))
U5_RECONCILIATION_ID = str(uuid5(NAMESPACE_URL, "u5-reconciliation"))
U5_RECOVERY_ID = STARTUP_RECOVERY_ID


def u5_registry() -> tuple[ParticipantBaseline, ...]:
    return (
        ParticipantBaseline(
            participant_id="memory.episodic", domain=ParticipantDomain.DURABLE_DOMAIN
        ),
        ParticipantBaseline(
            participant_id="session.turn", domain=ParticipantDomain.EPHEMERAL_PROCESS
        ),
    )


def append_u5_baseline(value: EventJournal) -> None:
    anchor = next(
        (
            record
            for record in reversed(value.records)
            if record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
        ),
        value.records[0],
    )
    inspection = value.inspect()
    value.append_participant_baseline(
        U5_BASELINE_ID,
        inspection.snapshot_sequence,
        inspection.snapshot_hash,
        inspection.processing_high_water,
        anchor.wal_generation_id or "",
        anchor.wal_record_id or "",
        anchor.wal_record_hash or "",
        inspection.journal_lineage_id or "",
        u5_registry(),
    )


def append_u5_clear(value: EventJournal, *, terminal: bool = False) -> None:
    inspection = value.inspect()
    baseline = inspection.baselines[-1]
    append = value.append_cleared if terminal else value.append_clear_prepared
    append(
        U5_BASELINE_ID,
        U5_RECONCILIATION_ID,
        U5_RECOVERY_ID,
        inspection.snapshot_sequence,
        inspection.snapshot_hash,
        inspection.processing_high_water,
        baseline.wal_generation_id,
        baseline.wal_record_id,
        baseline.wal_record_hash,
        inspection.journal_lineage_id or "",
    )


def append_u5_reconciliation(value: EventJournal) -> None:
    baseline = value.inspect().baselines[-1]
    recovery = next(
        record
        for record in reversed(value.records)
        if record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
    )
    assert recovery.recovery_id is not None
    assert recovery.snapshot_sequence is not None
    assert recovery.snapshot_hash is not None
    assert recovery.recovery_processing_high_water is not None
    assert recovery.wal_generation_id is not None
    assert recovery.wal_record_id is not None
    assert recovery.wal_record_hash is not None
    requirements = transaction_requirements()
    value.append_startup_reconciliation_prepared(
        U5_RECONCILIATION_ID,
        recovery.recovery_id,
        recovery.snapshot_sequence,
        recovery.snapshot_hash,
        recovery.recovery_processing_high_water,
        recovery.wal_generation_id,
        recovery.wal_record_id,
        recovery.wal_record_hash,
        baseline.journal_lineage_id,
        requirements,
    )
    for requirement in requirements:
        value.append_startup_participant_reconciled(
            U5_RECONCILIATION_ID,
            recovery.recovery_id,
            recovery.snapshot_sequence,
            recovery.snapshot_hash,
            recovery.recovery_processing_high_water,
            recovery.wal_generation_id,
            recovery.wal_record_id,
            recovery.wal_record_hash,
            baseline.journal_lineage_id,
            requirement.participant_id,
            requirement.operation_digest,
            StartupParticipantOutcome.VERIFIED_CONSISTENT,
        )
    value.append_startup_reconciliation_completed(
        U5_RECONCILIATION_ID,
        recovery.recovery_id,
        recovery.snapshot_sequence,
        recovery.snapshot_hash,
        recovery.recovery_processing_high_water,
        recovery.wal_generation_id,
        recovery.wal_record_id,
        recovery.wal_record_hash,
        baseline.journal_lineage_id,
    )


def prepared_transaction(value: EventJournal, name: str = "transaction") -> AgentEvent:
    item = event(name, 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, transaction_requirements()
    )
    value.append_prepared(item, HASH_0, HASH_1, value.records[0].wal_generation_id)
    return item


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


def test_v2_fresh_bootstrap_and_wal_ordered_lifecycle(tmp_path: Path) -> None:
    value = journal(tmp_path / "v2.jsonl")
    generation = str(uuid5(NAMESPACE_URL, "generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "wal-record"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    item = event("v2", 1)
    value.append_accepted(AgentEvent(item.event_id, item.event_type, item.source, NOW))
    value.append_started(item)
    value.append_prepared(item, HASH_0, HASH_1, generation)
    with pytest.raises(EventJournalAppendError):
        value.append_completed(item, 1, HASH_1, generation, wal_id)
    value.append_completed(item, 1, HASH_1, generation, wal_id, HASH_2)
    assert all(record.schema_version == 2 for record in value.records)
    assert value.inspect().processing_high_water == 1


def test_v2_migration_preserves_failed_event_gap(tmp_path: Path) -> None:
    value = bootstrap(tmp_path / "migration.jsonl")
    item = event("failed-gap", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_failed(item, 0, HASH_0)
    generation = str(uuid5(NAMESPACE_URL, "gap-generation"))
    value.append_v2_migration_checkpoint(
        0, HASH_0, generation, str(uuid5(NAMESPACE_URL, "gap-wal")), HASH_1
    )
    assert value.inspect().processing_high_water == 1


def test_v2_recovery_open_pair_is_read_only_and_high_water_stable(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "recovery-v2.jsonl")
    generation = str(uuid5(NAMESPACE_URL, "recovery-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "recovery-wal"))
    recovery_id = str(uuid5(NAMESPACE_URL, "recovery"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.append_recovery_prepared(
        recovery_id, 0, HASH_0, generation, EventRecoveryCategory.UNCOMMITTED_TAIL
    )
    evidence = value.inspect()
    assert [item.recovery_id for item in evidence.open_recoveries] == [recovery_id]
    assert evidence.processing_high_water == 0
    value.append_recovery_completed(
        recovery_id,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.UNCOMMITTED_TAIL,
        False,
        wal_record_id=wal_id,
        wal_record_hash=HASH_1,
    )
    assert value.inspect().open_recoveries == ()


def test_v2_rotation_waits_for_recovery_and_uses_completed_wal_anchor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rotation-v2.jsonl"
    value = EventJournal(path, 100_000, 4, clock=lambda: NOW)
    generation = str(uuid5(NAMESPACE_URL, "rotation-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "rotation-wal"))
    recovery_id = str(uuid5(NAMESPACE_URL, "rotation-recovery"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.max_bytes = 1
    value.append_recovery_prepared(
        recovery_id, 0, HASH_0, generation, EventRecoveryCategory.EXACT_CURRENT
    )
    assert not list(tmp_path.glob("rotation-v2.jsonl.[0-9]*"))
    value.append_recovery_completed(
        recovery_id,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.EXACT_CURRENT,
        False,
        wal_record_id=wal_id,
        wal_record_hash=HASH_1,
    )
    assert list(tmp_path.glob("rotation-v2.jsonl.[0-9]*"))
    assert value.records[-1].wal_record_id == wal_id
    assert value.records[-1].wal_record_hash == HASH_1


def test_canonically_hashed_v2_lifecycle_cannot_hide_irrelevant_recovery_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-v2.jsonl"
    value = journal(path)
    generation = str(uuid5(NAMESPACE_URL, "invalid-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "invalid-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    records = value.records
    valid = value._make_record(
        EventLifecycle.ACCEPTED,
        previous_hash=records[-1].record_hash,
        event=event("invalid-fields"),
        schema_version=2,
    )
    invalid = valid.model_copy(
        update={"recovery_id": str(uuid5(NAMESPACE_URL, "invalid-recovery"))}
    )
    invalid = invalid.model_copy(
        update={"record_hash": EventJournal._record_hash(invalid)}
    )
    value.close()
    path.write_bytes(
        b"".join(EventJournal._record_bytes(item) for item in (*records, invalid))
    )
    with pytest.raises(EventJournalLoadError):
        journal(path)


@pytest.mark.parametrize(
    ("category", "target_sequence"),
    [
        (EventRecoveryCategory.EXACT_CURRENT, 1),
        (EventRecoveryCategory.UNCOMMITTED_TAIL, 1),
        (EventRecoveryCategory.TRUE_ROLLBACK, 0),
    ],
)
def test_canonically_hashed_semantically_impossible_recovery_fails_closed(
    tmp_path: Path,
    category: EventRecoveryCategory,
    target_sequence: int,
) -> None:
    value = journal(tmp_path / f"invalid-{category.value}.jsonl")
    generation = str(uuid5(NAMESPACE_URL, f"invalid-{category.value}-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, f"invalid-{category.value}-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    records = value.records
    impossible = value._make_record(
        EventLifecycle.RECOVERY_PREPARED,
        previous_hash=records[-1].record_hash,
        schema_version=2,
        recovery_id=str(uuid5(NAMESPACE_URL, f"invalid-{category.value}-recovery")),
        snapshot_sequence=target_sequence,
        snapshot_hash=HASH_1 if target_sequence else HASH_0,
        wal_generation_id=generation,
        recovery_category=category,
        recovery_processing_high_water=0,
    )

    assert impossible.record_hash == EventJournal._record_hash(impossible)
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, impossible))


@pytest.mark.parametrize(
    "lifecycle",
    [EventLifecycle.RECOVERY_PREPARED, EventLifecycle.RECOVERY_COMPLETED],
)
def test_recovery_lifecycle_forbids_normal_event_identity(
    tmp_path: Path, lifecycle: EventLifecycle
) -> None:
    value = journal(tmp_path / f"recovery-identity-{lifecycle.value}.jsonl")
    generation = str(uuid5(NAMESPACE_URL, "recovery-identity-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "recovery-identity-wal"))
    recovery_id = str(uuid5(NAMESPACE_URL, "recovery-identity-recovery"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.append_recovery_prepared(
        recovery_id,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.EXACT_CURRENT,
    )
    if lifecycle is EventLifecycle.RECOVERY_COMPLETED:
        value.append_recovery_completed(
            recovery_id,
            0,
            HASH_0,
            generation,
            EventRecoveryCategory.EXACT_CURRENT,
            False,
            wal_record_id=wal_id,
            wal_record_hash=HASH_1,
        )
    raw = value.records[-1].model_dump(mode="python")
    item = event("forbidden-recovery-identity")
    raw.update(
        {
            "event_id": item.event_id,
            "event_type": item.event_type,
            "source": item.source,
        }
    )

    with pytest.raises(ValidationError):
        EventJournalRecord.model_validate(raw)


def test_canonically_hashed_reused_recovery_id_fails_closed(tmp_path: Path) -> None:
    value = journal(tmp_path / "reused-recovery.jsonl")
    generation = str(uuid5(NAMESPACE_URL, "reused-recovery-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "reused-recovery-wal"))
    recovery_id = str(uuid5(NAMESPACE_URL, "reused-recovery-id"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.append_recovery_prepared(
        recovery_id,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.EXACT_CURRENT,
    )
    value.append_recovery_completed(
        recovery_id,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.EXACT_CURRENT,
        False,
        wal_record_id=wal_id,
        wal_record_hash=HASH_1,
    )
    records = value.records
    reused = value._make_record(
        EventLifecycle.RECOVERY_PREPARED,
        previous_hash=records[-1].record_hash,
        schema_version=2,
        recovery_id=recovery_id,
        snapshot_sequence=0,
        snapshot_hash=HASH_0,
        wal_generation_id=generation,
        recovery_category=EventRecoveryCategory.EXACT_CURRENT,
        recovery_processing_high_water=0,
    )

    assert reused.record_hash == EventJournal._record_hash(reused)
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, reused))


def test_v2_history_rejects_canonically_hashed_schema_downgrade(tmp_path: Path) -> None:
    value = journal(tmp_path / "downgrade.jsonl")
    generation = str(uuid5(NAMESPACE_URL, "downgrade-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "downgrade-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    records = value.records
    downgrade = value._make_record(
        EventLifecycle.CHECKPOINT,
        previous_hash=records[-1].record_hash,
        schema_version=1,
        processing_sequence=0,
        snapshot_sequence=0,
        snapshot_hash=HASH_0,
    )
    assert downgrade.record_hash == EventJournal._record_hash(downgrade)
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, downgrade))


def test_v2_history_rejects_repeated_migration_anchor(tmp_path: Path) -> None:
    value = journal(tmp_path / "repeated-migration.jsonl")
    value.verify_and_reconcile(0, HASH_0)
    generation = str(uuid5(NAMESPACE_URL, "repeated-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "repeated-wal"))
    value.append_v2_migration_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    records = value.records
    repeated = value._make_record(
        EventLifecycle.CHECKPOINT,
        previous_hash=records[-1].record_hash,
        schema_version=2,
        processing_sequence=0,
        snapshot_sequence=0,
        snapshot_hash=HASH_0,
        migration_previous_v1_hash=records[-1].record_hash,
        wal_generation_id=generation,
        wal_record_id=wal_id,
        wal_record_hash=HASH_1,
        journal_lineage_id=records[-1].journal_lineage_id,
        external_reconciliation_required=False,
    )
    assert repeated.record_hash == EventJournal._record_hash(repeated)
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, repeated))


def test_rotation_resumes_from_fsynced_checkpoint_after_archive_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rotation-crash.jsonl"
    value = journal(path)
    value.verify_and_reconcile(0, HASH_0)
    original_replace = os.replace

    def interrupt_replacement(source, destination, *args, **kwargs) -> None:
        if (
            Path(source).name == f".{path.name}.rotation.tmp"
            and Path(destination) == path
        ):
            raise OSError("simulated crash")
        original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", interrupt_replacement)
    with pytest.raises(EventJournalAppendError):
        value._rotate_unlocked(value._verify_records(value.records))
    value.close()
    monkeypatch.setattr(os, "replace", original_replace)

    recovered = journal(path)

    assert recovered.inspect().snapshot_sequence == 0
    assert not (tmp_path / f".{path.name}.rotation.tmp").exists()


def test_rotation_defers_when_committed_crash_classification_lacks_wal_anchor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "classification-rotation.jsonl"
    value = journal(path)
    generation = str(uuid5(NAMESPACE_URL, "classification-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "classification-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.max_bytes = 1
    item = event("classification", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_prepared(item, HASH_0, HASH_1, generation)

    recovery = value.apply_planned_reconciliation(1, HASH_1)

    assert recovery.snapshot_sequence == 1
    assert value.records[-1].lifecycle is EventLifecycle.RECOVERY_CLASSIFIED
    assert not list(tmp_path.glob("classification-rotation.jsonl.[0-9]*"))


def test_v3_bootstrap_helper_preserves_v2_records_and_bytes(tmp_path: Path) -> None:
    path = tmp_path / "v3.jsonl"
    value = journal(path)
    generation = str(uuid5(NAMESPACE_URL, "migration-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "migration-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    v2_records = value.records
    v2_bytes = path.read_bytes()
    assert all(record.schema_version == 2 for record in v2_records)
    assert all(
        "transaction_id" not in json.loads(line) for line in v2_bytes.splitlines()
    )

    value.append_v3_migration_checkpoint()

    assert value.records[:-1] == v2_records
    assert path.read_bytes().startswith(v2_bytes)
    assert value.records[-1].migration_previous_v2_hash == v2_records[-1].record_hash
    assert value.records[-1].schema_version == 3


def test_v1_v2_hash_and_byte_semantics_match_pre_v3_golden_values() -> None:
    v1 = EventJournalRecord(
        schema_version=1,
        record_id=str(uuid5(NAMESPACE_URL, "golden-v1")),
        timestamp=NOW,
        lifecycle=EventLifecycle.CHECKPOINT,
        processing_sequence=0,
        snapshot_sequence=0,
        snapshot_hash=HASH_0,
        record_hash="0" * 64,
    )
    v2 = EventJournalRecord(
        schema_version=2,
        record_id=str(uuid5(NAMESPACE_URL, "golden-v2")),
        timestamp=NOW,
        lifecycle=EventLifecycle.CHECKPOINT,
        processing_sequence=0,
        snapshot_sequence=0,
        snapshot_hash=HASH_0,
        record_hash="0" * 64,
        wal_generation_id=str(uuid5(NAMESPACE_URL, "golden-gen")),
        wal_record_id=str(uuid5(NAMESPACE_URL, "golden-wal")),
        wal_record_hash=HASH_1,
        journal_lineage_id=str(uuid5(NAMESPACE_URL, "golden-lineage")),
        external_reconciliation_required=False,
    )

    assert EventJournal._record_hash(v1) == (
        "a4cec00324ac08ebf043a6e5b3ece3bf8f83482e847409fceb0214d0def5169b"
    )
    assert EventJournal._record_hash(v2) == (
        "85fe822fa2841e479979ffead740bcbbcd7bd0bb3796b937092b69063d21cf64"
    )
    assert hashlib.sha256(EventJournal._record_bytes(v1)).hexdigest() == (
        "0b979ab3aeec278248c86d81647346522cb3e0a48e7064f1786bfbc4ad4caf30"
    )
    assert hashlib.sha256(EventJournal._record_bytes(v2)).hexdigest() == (
        "14eb5b1692c84eab12df5730029b1ef9dcfa08d4463f5b5bba3727d7dbf838d9"
    )


def test_u1_f1_transaction_and_overall_terminal_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "transaction.jsonl"
    value = bootstrap_v3(path)
    item = prepared_transaction(value)
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[1], HASH_2, ParticipantOutcome.ALREADY_CONSISTENT
    )
    value.append_transaction_completed(item, TX_ID)
    anchor = value.records[0]
    value.append_completed(
        item,
        1,
        HASH_1,
        anchor.wal_generation_id,
        anchor.wal_record_id,
        anchor.wal_record_hash,
    )
    assert value.records[-2].lifecycle is EventLifecycle.TRANSACTION_COMPLETED
    assert value.records[-1].lifecycle is EventLifecycle.COMPLETED
    hashes = [record.record_hash for record in value.records]
    value.close()

    reopened = journal(path)
    inspection = reopened.inspect()
    assert inspection.processing_high_water == 1
    assert inspection.open_transactions == ()
    assert inspection.completed_transactions[0].transaction_id == TX_ID
    assert inspection.completed_transactions[0].participant_outcomes == (
        (PARTICIPANTS[0], ParticipantOutcome.FINALIZED),
        (PARTICIPANTS[1], ParticipantOutcome.ALREADY_CONSISTENT),
    )
    assert [record.record_hash for record in reopened.records] == hashes


def test_u1_f2_prepared_transaction_restarts_open_without_high_water_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "open.jsonl"
    value = bootstrap_v3(path)
    prepared_transaction(value, "open")
    value.close()
    reopened = journal(path)
    inspection = reopened.inspect()
    assert inspection.processing_high_water == 1
    assert inspection.open_transactions[0].transaction_id == TX_ID
    assert (
        tuple(
            item.participant_id
            for item in inspection.open_transactions[0].required_participants
        )
        == PARTICIPANTS
    )
    assert inspection.open_transactions[0].participant_outcomes == ()


def test_u1_f3_unknown_participant_is_rejected_before_append(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "unknown.jsonl")
    item = prepared_transaction(value, "unknown")
    before = value.records
    with pytest.raises(EventJournalAppendError) as error:
        value.append_participant_finalized(
            item, TX_ID, "unknown.participant", HASH_1, ParticipantOutcome.FINALIZED
        )
    assert error.value.stage is EventJournalAppendStage.VALIDATE
    assert error.value.published is False
    assert value.records == before


def test_u1_f4_completion_before_all_outcomes_is_rejected(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "premature.jsonl")
    item = prepared_transaction(value, "premature")
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    before = value.records
    with pytest.raises(EventJournalAppendError):
        value.append_transaction_completed(item, TX_ID)
    assert value.records == before


def test_u1_f5_conflicting_duplicate_outcome_is_rejected(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "duplicate.jsonl")
    item = prepared_transaction(value, "duplicate")
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    before = value.records
    with pytest.raises(EventJournalAppendError):
        value.append_participant_finalized(
            item,
            TX_ID,
            PARTICIPANTS[0],
            HASH_1,
            ParticipantOutcome.ALREADY_CONSISTENT,
        )
    assert value.records == before


def test_duplicate_transaction_identity_is_rejected_before_append(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "duplicate-transaction.jsonl")
    item = event("duplicate-transaction", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, transaction_requirements()
    )
    before = value.records

    with pytest.raises(EventJournalAppendError):
        value.append_transaction_prepared(
            item,
            TX_ID,
            TransactionKind.EVENT_MUTATION,
            (
                ParticipantRequirement(
                    participant_id="other.store",
                    operation_digest=HASH_0,
                    capabilities=(
                        ParticipantCapability.IDEMPOTENT_FINALIZE,
                        ParticipantCapability.PREPARE,
                    ),
                ),
            ),
        )

    assert value.records == before


def test_participant_terminal_evidence_requires_internal_prepare_boundary(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "participant-before-internal.jsonl")
    item = event("participant-before-internal", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, transaction_requirements()
    )

    with pytest.raises(EventJournalAppendError):
        value.append_participant_finalized(
            item,
            TX_ID,
            PARTICIPANTS[0],
            HASH_1,
            ParticipantOutcome.FINALIZED,
        )


def test_transaction_prepare_cannot_follow_internal_prepare_boundary(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "transaction-after-internal.jsonl")
    item = event("transaction-after-internal", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_prepared(item, HASH_0, HASH_1, value.records[0].wal_generation_id)

    with pytest.raises(EventJournalAppendError):
        value.append_transaction_prepared(
            item, TX_ID, TransactionKind.EVENT_MUTATION, transaction_requirements()
        )


def test_reconciliation_evidence_cannot_precede_internal_prepare_boundary(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "reconciliation-before-internal.jsonl")
    item = event("reconciliation-before-internal", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, transaction_requirements()
    )

    with pytest.raises(EventJournalAppendError):
        value.append_transaction_reconciliation_required(
            item,
            TX_ID,
            ReconciliationReason.PARTICIPANT_FINALIZATION_INCOMPLETE,
            PARTICIPANTS,
        )


def test_overall_completion_cannot_hide_unterminated_transaction(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "overall-before-transaction.jsonl")
    item = prepared_transaction(value, "overall-before-transaction")
    records = value.records
    anchor = records[0]

    with pytest.raises(EventJournalAppendError):
        value.append_completed(
            item,
            1,
            HASH_1,
            anchor.wal_generation_id,
            anchor.wal_record_id,
            anchor.wal_record_hash,
        )

    impossible = value._make_record(
        EventLifecycle.COMPLETED,
        previous_hash=records[-1].record_hash,
        event=item,
        schema_version=3,
        processing_sequence=1,
        snapshot_sequence=1,
        snapshot_hash=HASH_1,
        wal_generation_id=anchor.wal_generation_id,
        wal_record_id=anchor.wal_record_id,
        wal_record_hash=anchor.wal_record_hash,
    )
    assert impossible.record_hash == EventJournal._record_hash(impossible)
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, impossible))


def test_u1_f6_canonically_hashed_impossible_transaction_fails_verify(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "impossible.jsonl")
    item = prepared_transaction(value, "impossible")
    records = value.records
    transaction_index = next(
        index
        for index, record in enumerate(records)
        if record.lifecycle is EventLifecycle.TRANSACTION_PREPARED
    )
    impossible = records[transaction_index].model_copy(
        update={"event_id": event("other").event_id, "processing_sequence": 2}
    )
    impossible = impossible.model_copy(
        update={"record_hash": EventJournal._record_hash(impossible)}
    )
    assert impossible.record_hash == EventJournal._record_hash(impossible)
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records[:transaction_index], impossible))

    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    records = value.records
    impossible_reconciliation = value._make_record(
        EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
        previous_hash=records[-1].record_hash,
        event=item,
        schema_version=3,
        processing_sequence=1,
        transaction_id=TX_ID,
        reconciliation_reason=ReconciliationReason.PARTICIPANT_FINALIZATION_INCOMPLETE,
        unresolved_participants=(PARTICIPANTS[0],),
    )
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, impossible_reconciliation))


def test_u1_f7_v3_boundary_binds_exact_v2_tail(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "boundary.jsonl")
    boundary = value.records[-1]
    assert boundary.migration_previous_v2_hash == boundary.previous_record_hash
    assert boundary.v3_migration_anchor_hash == boundary.previous_record_hash
    assert value.records[-2].schema_version == 2
    assert boundary.record_hash == EventJournal._record_hash(boundary)


def test_u1_f8_v3_history_rejects_canonically_hashed_v2_downgrade(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "downgrade-v3.jsonl")
    records = value.records
    downgrade = value._make_record(
        EventLifecycle.CHECKPOINT,
        previous_hash=records[-1].record_hash,
        schema_version=2,
        processing_sequence=0,
        snapshot_sequence=0,
        snapshot_hash=HASH_0,
        wal_generation_id=records[-1].wal_generation_id,
        wal_record_id=records[-1].wal_record_id,
        wal_record_hash=records[-1].wal_record_hash,
        journal_lineage_id=records[-1].journal_lineage_id,
        external_reconciliation_required=False,
    )
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, downgrade))


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 3},
        {"schema_version": 4},
        {"schema_version": 3, "unexpected": PRIVATE_SENTINEL},
    ],
)
def test_u1_f9_malformed_future_and_unexpected_v3_records_fail_closed(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    error_type = (
        UnsupportedEventJournalVersion
        if payload["schema_version"] == 4
        else EventJournalLoadError
    )
    with pytest.raises(error_type) as error:
        journal(path)
    assert_bounded(error.value)


def test_u1_f10_transaction_evidence_never_advances_processing_high_water(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "high-water.jsonl")
    item = prepared_transaction(value, "high-water")
    for append in (
        lambda: value.append_participant_finalized(
            item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
        ),
        lambda: value.append_participant_finalized(
            item, TX_ID, PARTICIPANTS[1], HASH_2, ParticipantOutcome.FINALIZED
        ),
        lambda: value.append_transaction_completed(item, TX_ID),
    ):
        append()
        assert value.inspect().processing_high_water == 1


def test_u1_f11_invalid_transaction_fields_are_bounded_and_private_free(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private-tx.jsonl"
    value = bootstrap_v3(path)
    item = prepared_transaction(value, "private-tx")
    before = value.records
    raw_transaction = next(
        record.model_dump(mode="python")
        for record in before
        if record.lifecycle is EventLifecycle.TRANSACTION_PREPARED
    )
    raw_transaction["raw_payload"] = PRIVATE_SENTINEL
    with pytest.raises(ValidationError):
        EventJournalRecord.model_validate(raw_transaction)
    for transaction_id, participant, digest, outcome in (
        ("not-a-uuid", PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED),
        (TX_ID, "Bad.Participant", HASH_1, ParticipantOutcome.FINALIZED),
        (TX_ID, PARTICIPANTS[0], "not-a-digest", ParticipantOutcome.FINALIZED),
        (TX_ID, PARTICIPANTS[0], HASH_1, "not-a-status"),
    ):
        with pytest.raises(EventJournalAppendError) as error:
            value.append_participant_finalized(
                item,
                transaction_id,
                participant,
                digest,
                outcome,  # type: ignore[arg-type]
            )
        assert_bounded(error.value)
    assert value.records == before

    def fail_transaction_append(
        lifecycle: EventLifecycle, stage: EventJournalAppendStage
    ) -> None:
        if (
            lifecycle is EventLifecycle.PARTICIPANT_FINALIZED
            and stage is EventJournalAppendStage.WRITE
        ):
            raise OSError(PRIVATE_SENTINEL)

    value._append_stage_hook = fail_transaction_append
    with pytest.raises(EventJournalAppendError) as write_error:
        value.append_participant_finalized(
            item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
        )
    assert_bounded(write_error.value)
    assert PRIVATE_SENTINEL not in path.read_text(encoding="utf-8")


def test_uncommitted_event_recovery_cannot_orphan_open_transaction(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "open-transaction-recovery.jsonl")
    prepared_transaction(value, "open-transaction-recovery")
    before = value.records

    with pytest.raises(EventJournalIntegrityError):
        value.verify_and_reconcile(0, HASH_0)

    assert value.records == before
    assert value.inspect().open_transactions[0].transaction_id == TX_ID


def test_reconciliation_requires_explicit_terminal_transition(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "compensation.jsonl")
    item = prepared_transaction(value, "compensation")

    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    value.append_transaction_reconciliation_required(
        item,
        TX_ID,
        ReconciliationReason.PARTICIPANT_FINALIZATION_INCOMPLETE,
        (PARTICIPANTS[1],),
    )
    value.append_participant_finalized(
        item,
        TX_ID,
        PARTICIPANTS[1],
        HASH_2,
        ParticipantOutcome.FINALIZED,
    )
    with pytest.raises(EventJournalAppendError):
        value.append_transaction_completed(item, TX_ID)
    value.append_transaction_reconciled(item, TX_ID)


def test_reconciliation_lifecycle_survives_restart_and_becomes_reconciled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reconcile.jsonl"
    value = bootstrap_v3(path)
    item = prepared_transaction(value, "reconcile")
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    value.append_transaction_reconciliation_required(
        item,
        TX_ID,
        ReconciliationReason.PARTICIPANT_FINALIZATION_INCOMPLETE,
        (PARTICIPANTS[1],),
    )
    value.close()
    reopened = journal(path)
    inspection = reopened.inspect()
    assert inspection.reconciliation_required_transactions[0].transaction_id == TX_ID
    reopened.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[1], HASH_2, ParticipantOutcome.FINALIZED
    )
    reopened.append_transaction_reconciled(item, TX_ID)
    anchor = reopened.records[0]
    reopened.append_completed(
        item,
        1,
        HASH_1,
        anchor.wal_generation_id,
        anchor.wal_record_id,
        anchor.wal_record_hash,
    )
    final = reopened.inspect()
    assert final.reconciled_transactions[0].transaction_id == TX_ID
    assert final.reconciliation_required_transactions == ()
    assert final.open_transactions == ()
    assert final.records[-1].lifecycle is EventLifecycle.COMPLETED


def test_v3_recovery_remains_valid_and_rotation_defers_for_open_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rotation-v3.jsonl"
    value = bootstrap_v3(path)
    item = prepared_transaction(value, "rotation-v3")
    value.max_bytes = 1
    assert not list(tmp_path.glob("rotation-v3.jsonl.[0-9]*"))
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[1], HASH_2, ParticipantOutcome.FINALIZED
    )
    value.append_transaction_completed(item, TX_ID)
    anchor = value.records[0]
    value.append_completed(
        item,
        1,
        HASH_1,
        anchor.wal_generation_id,
        anchor.wal_record_id,
        anchor.wal_record_hash,
    )
    assert list(tmp_path.glob("rotation-v3.jsonl.[0-9]*"))

    recovery_path = tmp_path / "recovery-v3.jsonl"
    recovery = bootstrap_v3(recovery_path)
    interrupted = event("v3-recovery", 1)
    recovery.append_accepted(interrupted)
    recovery.append_started(interrupted)
    recovery.close()
    reopened = journal(recovery_path)
    result = reopened.verify_and_reconcile(0, HASH_0)
    assert result.processing_high_water == 1
    assert reopened.records[-1].lifecycle is EventLifecycle.RECOVERY_CLASSIFIED


def test_v3_rotation_preserves_transaction_authority_before_first_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounded-v3.jsonl"
    value = EventJournal(path, 100_000, 2, clock=lambda: NOW)
    generation = str(uuid5(NAMESPACE_URL, "bounded-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "bounded-wal"))
    value.append_v2_bootstrap_checkpoint(0, HASH_0, generation, wal_id, HASH_1)
    value.append_v3_migration_checkpoint()
    migration_anchor = value.records[-1].v3_migration_anchor_hash
    assert migration_anchor is not None
    value.max_bytes = 1
    current_hash = HASH_0

    for sequence, next_hash in enumerate((HASH_1, HASH_2, HASH_0, HASH_1), start=1):
        item = event(f"bounded-{sequence}", sequence)
        transaction_id = str(uuid5(NAMESPACE_URL, f"bounded-transaction-{sequence}"))
        participant = ParticipantRequirement(
            participant_id="memory.episodic",
            operation_digest=HASH_2,
            capabilities=(
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.PREPARE,
            ),
        )
        value.append_accepted(item)
        value.append_started(item)
        value.append_transaction_prepared(
            item, transaction_id, TransactionKind.EVENT_MUTATION, (participant,)
        )
        value.append_prepared(item, current_hash, next_hash, generation)
        value.append_participant_finalized(
            item,
            transaction_id,
            participant.participant_id,
            participant.operation_digest,
            ParticipantOutcome.FINALIZED,
        )
        value.append_transaction_completed(item, transaction_id)
        value.append_completed(
            item,
            sequence,
            next_hash,
            generation,
            str(uuid5(NAMESPACE_URL, f"bounded-wal-{sequence}")),
            HASH_1,
        )
        current_hash = next_hash

    value.close()
    rotated = sorted(tmp_path.glob("bounded-v3.jsonl.[0-9]*"))
    assert len(rotated) == 1
    retained_root = EventJournalRecord.model_validate_json(
        rotated[0].read_bytes().splitlines()[0]
    )
    assert retained_root.schema_version == 2

    reopened = EventJournal(path, 1, 2, clock=lambda: NOW)
    inspection = reopened.inspect()
    assert inspection.schema_version == 3
    assert inspection.processing_high_water == 4
    assert inspection.snapshot_hash == HASH_1
    assert len(inspection.completed_transactions) == 4


def test_startup_reconciliation_is_metadata_free_and_does_not_consume_sequence(
    tmp_path: Path,
) -> None:
    value = startup_reconciliation_journal(tmp_path / "startup.jsonl")
    before = value.inspect()
    value.append_startup_reconciliation_prepared(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        value.records[0].wal_generation_id or "",
        value.records[0].wal_record_id or "",
        value.records[0].wal_record_hash or "",
        before.journal_lineage_id or "",
        transaction_requirements(),
    )

    record = value.records[-1]
    assert record.event_id is None
    assert record.event_type is None
    assert record.source is None
    assert record.processing_sequence is None
    assert value.inspect().processing_high_water == before.processing_high_water == 1
    assert value.inspect().open_startup_reconciliations[0].recovery_id == (
        STARTUP_RECOVERY_ID
    )


def test_startup_reconciliation_binds_recovery_wal_and_survives_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "startup-restart.jsonl"
    value = startup_reconciliation_journal(path)
    anchor = value.records[0]
    lineage = value.inspect().journal_lineage_id or ""
    requirements = transaction_requirements()
    value.append_startup_reconciliation_prepared(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        anchor.wal_generation_id or "",
        anchor.wal_record_id or "",
        anchor.wal_record_hash or "",
        lineage,
        requirements,
    )
    value.close()

    reopened = journal(path)
    startup = reopened.inspect().open_startup_reconciliations[0]
    recovery = next(
        record
        for record in reopened.records
        if record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
    )
    assert startup.required_participants == requirements
    assert startup.snapshot_hash == HASH_0
    assert startup.recovery_processing_high_water == 1
    assert startup.wal_generation_id == recovery.wal_generation_id
    assert startup.wal_record_id == recovery.wal_record_id
    assert startup.wal_record_hash == recovery.wal_record_hash
    assert startup.journal_lineage_id == lineage
    assert recovery.wal_record_id is not None
    assert recovery.wal_record_hash == HASH_2


def test_startup_reconciliation_rejects_identity_tampering_and_invalid_participant(
    tmp_path: Path,
) -> None:
    value = startup_reconciliation_journal(tmp_path / "startup-invalid.jsonl")
    lineage = value.inspect().journal_lineage_id or ""
    generation = value.records[0].wal_generation_id or ""
    wal_record_id = value.records[0].wal_record_id or ""
    wal_record_hash = value.records[0].wal_record_hash or ""
    value.append_startup_reconciliation_prepared(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        generation,
        wal_record_id,
        wal_record_hash,
        lineage,
        transaction_requirements(),
    )
    with pytest.raises(EventJournalAppendError) as error:
        value.append_startup_participant_reconciled(
            STARTUP_RECONCILIATION_ID,
            str(uuid5(NAMESPACE_URL, "wrong-recovery")),
            0,
            HASH_0,
            1,
            generation,
            wal_record_id,
            wal_record_hash,
            lineage,
            PARTICIPANTS[0],
            HASH_1,
            StartupParticipantOutcome.VERIFIED_CONSISTENT,
        )
    assert_bounded(error.value)
    with pytest.raises(EventJournalAppendError):
        value.append_startup_participant_reconciled(
            STARTUP_RECONCILIATION_ID,
            STARTUP_RECOVERY_ID,
            0,
            HASH_0,
            1,
            generation,
            wal_record_id,
            wal_record_hash,
            lineage,
            "unknown.participant",
            HASH_1,
            StartupParticipantOutcome.VERIFIED_CONSISTENT,
        )

    path = value.path
    value.close()
    lines = path.read_bytes().splitlines()
    tampered = json.loads(lines[-1])
    tampered["snapshot_hash"] = HASH_1
    path.write_bytes(b"\n".join((*lines[:-1], json.dumps(tampered).encode())) + b"\n")
    with pytest.raises(EventJournalLoadError) as load_error:
        journal(path)
    assert_bounded(load_error.value)


def test_startup_reconciliation_rejects_duplicate_and_premature_completion(
    tmp_path: Path,
) -> None:
    value = startup_reconciliation_journal(tmp_path / "startup-completion.jsonl")
    generation = value.records[0].wal_generation_id or ""
    wal_record_id = value.records[0].wal_record_id or ""
    wal_record_hash = value.records[0].wal_record_hash or ""
    lineage = value.inspect().journal_lineage_id or ""
    requirements = transaction_requirements()
    value.append_startup_reconciliation_prepared(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        generation,
        wal_record_id,
        wal_record_hash,
        lineage,
        requirements,
    )
    with pytest.raises(EventJournalAppendError):
        value.append_startup_reconciliation_prepared(
            str(uuid5(NAMESPACE_URL, "duplicate-startup-recovery")),
            STARTUP_RECOVERY_ID,
            0,
            HASH_0,
            1,
            generation,
            wal_record_id,
            wal_record_hash,
            lineage,
            requirements,
        )
    with pytest.raises(EventJournalAppendError):
        value.append_startup_reconciliation_completed(
            STARTUP_RECONCILIATION_ID,
            STARTUP_RECOVERY_ID,
            0,
            HASH_0,
            1,
            generation,
            wal_record_id,
            wal_record_hash,
            lineage,
        )
    value.append_startup_participant_reconciled(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        generation,
        wal_record_id,
        wal_record_hash,
        lineage,
        PARTICIPANTS[0],
        HASH_1,
        StartupParticipantOutcome.VERIFIED_CONSISTENT,
    )
    with pytest.raises(EventJournalAppendError):
        value.append_startup_participant_reconciled(
            STARTUP_RECONCILIATION_ID,
            STARTUP_RECOVERY_ID,
            0,
            HASH_0,
            1,
            generation,
            wal_record_id,
            wal_record_hash,
            lineage,
            PARTICIPANTS[0],
            HASH_1,
            StartupParticipantOutcome.VERIFIED_CONSISTENT,
        )


def test_startup_reconciliation_completion_stays_blocked_until_gate_clear(
    tmp_path: Path,
) -> None:
    path = tmp_path / "startup-rotation.jsonl"
    value = EventJournal(path, 1, 2, clock=lambda: NOW)
    generation = str(uuid5(NAMESPACE_URL, "startup-rotation-generation"))
    wal_id = str(uuid5(NAMESPACE_URL, "startup-rotation-wal"))
    value.append_v2_bootstrap_checkpoint(1, HASH_1, generation, wal_id, HASH_2)
    value.append_v3_migration_checkpoint()
    anchor = value.records[0]
    value.append_recovery_prepared(
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.TRUE_ROLLBACK,
        processing_high_water=1,
    )
    value.append_recovery_completed(
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        generation,
        EventRecoveryCategory.TRUE_ROLLBACK,
        True,
        processing_high_water=1,
        wal_record_id=anchor.wal_record_id or wal_id,
        wal_record_hash=anchor.wal_record_hash or HASH_2,
    )
    lineage = value.inspect().journal_lineage_id or ""
    rotations_before = set(tmp_path.glob("startup-rotation.jsonl.[0-9]*"))
    value.append_startup_reconciliation_prepared(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        generation,
        anchor.wal_record_id or wal_id,
        anchor.wal_record_hash or HASH_2,
        lineage,
        transaction_requirements(),
    )
    assert set(tmp_path.glob("startup-rotation.jsonl.[0-9]*")) == rotations_before
    for participant, digest in zip(PARTICIPANTS, (HASH_1, HASH_2), strict=True):
        value.append_startup_participant_reconciled(
            STARTUP_RECONCILIATION_ID,
            STARTUP_RECOVERY_ID,
            0,
            HASH_0,
            1,
            generation,
            anchor.wal_record_id or wal_id,
            anchor.wal_record_hash or HASH_2,
            lineage,
            participant,
            digest,
            StartupParticipantOutcome.ROLLED_FORWARD,
        )
    value.append_startup_reconciliation_completed(
        STARTUP_RECONCILIATION_ID,
        STARTUP_RECOVERY_ID,
        0,
        HASH_0,
        1,
        generation,
        anchor.wal_record_id or wal_id,
        anchor.wal_record_hash or HASH_2,
        lineage,
    )
    assert set(tmp_path.glob("startup-rotation.jsonl.[0-9]*")) == rotations_before
    assert any(
        record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
        for record in value.records
    )
    assert value.inspect().open_startup_reconciliations == ()
    for recovery_id in (
        STARTUP_RECOVERY_ID,
        str(uuid5(NAMESPACE_URL, "invented-pruned-recovery")),
    ):
        with pytest.raises(EventJournalAppendError):
            value.append_startup_reconciliation_prepared(
                str(uuid5(NAMESPACE_URL, f"retry-{recovery_id}")),
                recovery_id,
                0,
                HASH_0,
                1,
                generation,
                anchor.wal_record_id or wal_id,
                anchor.wal_record_hash or HASH_2,
                lineage,
                transaction_requirements(),
            )


def test_participant_capabilities_are_sorted_unique_and_minimal() -> None:
    with pytest.raises(ValidationError):
        ParticipantRequirement(
            participant_id="memory.episodic",
            operation_digest=HASH_1,
            capabilities=(ParticipantCapability.PREPARE,),
        )
    with pytest.raises(ValidationError):
        ParticipantRequirement(
            participant_id="memory.episodic",
            operation_digest=HASH_1,
            capabilities=(
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.PREPARE,
            ),
        )


def test_abort_branch_is_durable_and_restarts_without_high_water_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "abort.jsonl"
    value = bootstrap_v3(path)
    item = event("abort", 1)
    value.append_accepted(item)
    value.append_started(item)
    requirements = tuple(
        ParticipantRequirement(
            participant_id=participant,
            operation_digest=digest,
            capabilities=(
                ParticipantCapability.ABORT,
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.PREPARE,
            ),
        )
        for participant, digest in zip(PARTICIPANTS, (HASH_1, HASH_2), strict=True)
    )
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, requirements
    )
    value.append_participant_aborted(
        item, TX_ID, PARTICIPANTS[0], HASH_1, AbortOutcome.ABORTED
    )
    value.append_transaction_abort_required(
        item,
        TX_ID,
        ReconciliationReason.PARTICIPANT_UNAVAILABLE,
        (PARTICIPANTS[1],),
    )
    assert value.inspect().processing_high_water == 1
    value.close()

    reopened = journal(path)
    assert reopened.inspect().abort_required_transactions[0].abort_reason is (
        ReconciliationReason.PARTICIPANT_UNAVAILABLE
    )
    reopened.append_participant_aborted(
        item, TX_ID, PARTICIPANTS[1], HASH_2, AbortOutcome.ALREADY_ABSENT
    )
    reopened.append_transaction_aborted(item, TX_ID)
    inspection = reopened.inspect()
    assert inspection.processing_high_water == 1
    assert inspection.abort_required_transactions == ()
    assert inspection.aborted_transactions[0].abort_outcomes == (
        (PARTICIPANTS[0], AbortOutcome.ABORTED),
        (PARTICIPANTS[1], AbortOutcome.ALREADY_ABSENT),
    )


def test_abort_and_finalize_branches_cannot_mix(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "abort-mix.jsonl")
    item = event("abort-mix", 1)
    value.append_accepted(item)
    value.append_started(item)
    requirements = tuple(
        ParticipantRequirement(
            participant_id=participant,
            operation_digest=digest,
            capabilities=(
                ParticipantCapability.ABORT,
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.PREPARE,
            ),
        )
        for participant, digest in zip(PARTICIPANTS, (HASH_1, HASH_2), strict=True)
    )
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, requirements
    )
    value.append_participant_aborted(
        item, TX_ID, PARTICIPANTS[0], HASH_1, AbortOutcome.ABORTED
    )
    with pytest.raises(EventJournalAppendError):
        value.append_participant_finalized(
            item, TX_ID, PARTICIPANTS[1], HASH_2, ParticipantOutcome.FINALIZED
        )


def test_zero_abort_requirement_can_terminalize_without_participant_evidence(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "zero-abort.jsonl")
    item = event("zero-abort", 1)
    value.append_accepted(item)
    value.append_started(item)
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, transaction_requirements()
    )

    value.append_transaction_aborted(item, TX_ID)

    transaction = value.inspect().aborted_transactions[0]
    assert transaction.abort_outcomes == ()
    assert transaction.abort_reason is None
    assert not any(
        record.lifecycle is EventLifecycle.PARTICIPANT_ABORTED
        for record in value.records
    )


def test_nonempty_abort_requirement_rejects_missing_abort_evidence(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "missing-abort.jsonl")
    item = event("missing-abort", 1)
    value.append_accepted(item)
    value.append_started(item)
    requirements = tuple(
        ParticipantRequirement(
            participant_id=participant,
            operation_digest=digest,
            capabilities=(
                ParticipantCapability.ABORT,
                ParticipantCapability.IDEMPOTENT_FINALIZE,
                ParticipantCapability.PREPARE,
            ),
        )
        for participant, digest in zip(PARTICIPANTS, (HASH_1, HASH_2), strict=True)
    )
    value.append_transaction_prepared(
        item, TX_ID, TransactionKind.EVENT_MUTATION, requirements
    )
    value.append_participant_aborted(
        item, TX_ID, PARTICIPANTS[0], HASH_1, AbortOutcome.ABORTED
    )

    with pytest.raises(EventJournalAppendError):
        value.append_transaction_aborted(item, TX_ID)


def test_u5_participant_baseline_records_exact_registry_without_sequence_consumption(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "u5-baseline.jsonl")
    before = value.inspect()

    append_u5_baseline(value)

    inspection = value.inspect()
    assert inspection.processing_high_water == before.processing_high_water
    assert inspection.baselines[0].baseline_id == U5_BASELINE_ID
    assert inspection.baselines[0].participant_registry == u5_registry()
    assert [record.lifecycle for record in value.records] == [
        EventLifecycle.CHECKPOINT,
        EventLifecycle.CHECKPOINT,
        EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED,
    ]


def test_u5_baseline_rejects_any_registry_other_than_the_fixed_registry(
    tmp_path: Path,
) -> None:
    value = bootstrap_v3(tmp_path / "u5-registry.jsonl")
    before = value.records
    registry = u5_registry()

    with pytest.raises(ValueError):
        value.append_participant_baseline(
            U5_BASELINE_ID,
            0,
            HASH_0,
            0,
            value.records[0].wal_generation_id or "",
            value.records[0].wal_record_id or "",
            value.records[0].wal_record_hash or "",
            value.inspect().journal_lineage_id or "",
            (*registry, registry[0]),
        )

    assert value.records == before


def test_u5_canonically_hashed_noncurrent_baseline_fails_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "u5-forged-baseline.jsonl"
    value = bootstrap_v3(path)
    append_u5_baseline(value)
    lines = path.read_bytes().splitlines()
    baseline = EventJournalRecord.model_validate_json(lines[-1])
    forged = baseline.model_copy(update={"snapshot_hash": HASH_1})
    forged = forged.model_copy(
        update={"record_hash": EventJournal._record_hash(forged)}
    )
    value.close()
    path.write_bytes(b"\n".join((*lines[:-1], EventJournal._record_bytes(forged))))

    with pytest.raises(
        EventJournalIntegrityError, match="participant baseline authority is invalid"
    ):
        journal(path)


def test_u5_clear_prepared_and_cleared_are_open_then_terminal_inspection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "u5-clear.jsonl"
    value = startup_reconciliation_journal(path)
    append_u5_reconciliation(value)
    append_u5_clear(value)

    open_inspection = value.inspect()
    assert open_inspection.open_gate_clear is not None
    assert open_inspection.open_gate_clear.terminal is False
    assert open_inspection.terminal_gate_clear is None
    assert open_inspection.processing_high_water == 1

    value.close()
    reopened = journal(path)
    append_u5_clear(reopened, terminal=True)
    terminal = reopened.inspect()
    assert terminal.open_gate_clear is None
    assert terminal.terminal_gate_clear is not None
    assert terminal.terminal_gate_clear.terminal is True
    assert terminal.gate_clear == terminal.terminal_gate_clear


@pytest.mark.parametrize("terminal", [False, True])
def test_u5_clear_ordering_and_binding_are_rejected(
    tmp_path: Path, terminal: bool
) -> None:
    value = startup_reconciliation_journal(
        tmp_path / f"u5-invalid-{terminal}.jsonl"
    )
    append = value.append_cleared if terminal else value.append_clear_prepared
    args = (
        U5_BASELINE_ID,
        U5_RECONCILIATION_ID,
        U5_RECOVERY_ID,
        0,
        HASH_0,
        0,
        value.records[0].wal_generation_id or "",
        value.records[0].wal_record_id or "",
        value.records[0].wal_record_hash or "",
        value.inspect().journal_lineage_id or "",
    )
    with pytest.raises(EventJournalAppendError):
        append(*args)

    append_u5_reconciliation(value)
    with pytest.raises(EventJournalAppendError):
        append(
            U5_BASELINE_ID,
            U5_RECONCILIATION_ID,
            str(uuid5(NAMESPACE_URL, "wrong-recovery")),
            *args[3:],
        )


def test_u5_gate_hash_chain_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "u5-tampered.jsonl"
    value = bootstrap_v3(path)
    append_u5_baseline(value)
    value.close()
    lines = path.read_bytes().splitlines()
    tampered = json.loads(lines[-1])
    tampered["baseline_id"] = str(uuid5(NAMESPACE_URL, "tampered-baseline"))
    path.write_bytes(b"\n".join((*lines[:-1], json.dumps(tampered).encode())) + b"\n")

    with pytest.raises(EventJournalIntegrityError):
        journal(path)


def test_u5_rotation_preserves_baseline_and_clear_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "u5-rotation.jsonl"
    value = startup_reconciliation_journal(path)
    value.max_bytes = 1
    append_u5_reconciliation(value)
    append_u5_clear(value)
    assert not list(tmp_path.glob("u5-rotation.jsonl.[0-9]*"))
    append_u5_clear(value, terminal=True)
    current = value.inspect()
    for _ in range(8):
        value.append_v2_current_checkpoint(
            current.snapshot_sequence,
            current.snapshot_hash,
            current.wal_generation_id or "",
            current.wal_record_id or "",
            current.wal_record_hash or "",
        )
    value.close()

    rotated = list(tmp_path.glob("u5-rotation.jsonl.[0-9]*"))
    assert rotated
    assert len(rotated) + 1 <= value.retained_files
    reopened = EventJournal(path, 1, 4, clock=lambda: NOW)
    inspection = reopened.inspect()
    assert inspection.baselines[0].baseline_id == U5_BASELINE_ID
    assert inspection.terminal_gate_clear is not None
    assert inspection.open_gate_clear is None
