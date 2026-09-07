"""Deterministic startup and mutation tests for StateRecoveryCoordinator."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from kagya.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from kagya.runtime.agent_state import (
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshot,
    AgentStateStore,
    EmotionStateSnapshot,
)
from kagya.runtime.event_journal import (
    EventJournal,
    EventJournalAppendError,
    EventJournalAppendStage,
    EventLifecycle,
)
from kagya.runtime.state_recovery import (
    StateRecoveryCoordinator,
    StateRecoveryError,
)
from kagya.runtime.state_wal import StateWAL, StateWALError, TransitionRecord


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def snapshot(sequence: int, value: float = 0.1) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=value, arousal=0.2, optimal_loss=1.0
        ),
    )


def event(name: str, sequence: int) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, name)),
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        NOW,
        sequence,
    )


def graph(tmp_path: Path) -> tuple[AgentStateStore, EventJournal, StateWAL]:
    store = AgentStateStore(
        tmp_path / "agent_state.json", baseline_surprisal=1.0, clock=lambda: NOW
    )
    journal = EventJournal(tmp_path / "events.jsonl", 100_000, 4, clock=lambda: NOW)
    wal = StateWAL(tmp_path / "wal")
    return store, journal, wal


def coordinator(
    tmp_path: Path,
) -> tuple[StateRecoveryCoordinator, AgentStateStore, EventJournal, StateWAL]:
    store, journal, wal = graph(tmp_path)
    return StateRecoveryCoordinator(store, journal, wal), store, journal, wal


def start_event(journal: EventJournal, item: AgentEvent) -> None:
    journal.append_accepted(item)
    journal.append_started(item)


def append_uncommitted_candidate(
    store: AgentStateStore,
    journal: EventJournal,
    wal: StateWAL,
    *,
    name: str,
) -> tuple[AgentStateSnapshot, AgentStateSnapshot]:
    initial = store.load()
    candidate = snapshot(1, 0.4)
    item = event(name, 1)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    start_event(journal, item)
    journal.append_prepared(
        item,
        store.snapshot_hash(initial),
        store.snapshot_hash(candidate),
        str(manifest.active_generation_id),
    )
    wal.append_transition(
        event_id=uuid5(NAMESPACE_URL, name),
        event_type=item.event_type.value,
        event_source=item.source.value,
        processing_sequence=1,
        prior_snapshot=initial,
        candidate_snapshot=candidate,
    )
    return initial, candidate


def test_fresh_r06_bootstrap_creates_consistent_artifacts(tmp_path: Path) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)

    result = recovery.prepare_startup()

    assert result.snapshot == store.load()
    assert result.processing_high_water == 0
    active = wal.inspect().active_manifest
    assert active is not None
    assert result.manifest.active_generation_id == active.active_generation_id
    assert journal.inspect().records[-1].lifecycle is EventLifecycle.CHECKPOINT
    assert (wal.root / "manifest.json").is_file()
    recovery.publish_boot_anchor(result)
    assert wal.inspect_boot_anchor_optional() is not None


def test_r05_migration_preserves_high_water_above_snapshot_sequence(
    tmp_path: Path,
) -> None:
    store, journal, wal = graph(tmp_path)
    initial = store.load()
    store.ensure_published(initial)
    initial_hash = store.snapshot_hash(initial)
    journal.verify_and_reconcile(0, initial_hash)
    failed = event("pre-r06-failure", 1)
    start_event(journal, failed)
    journal.append_failed(failed, 0, initial_hash)

    recovery = StateRecoveryCoordinator(store, journal, wal)
    migrated = recovery.prepare_startup()

    assert migrated.snapshot.last_processed_event_sequence == 0
    assert migrated.processing_high_water == 1
    assert journal.inspect().schema_version == 2
    assert wal.inspect().records[0].journal_processing_high_water == 1

    next_event = event("post-r06-success", 2)
    candidate = snapshot(2, 0.4)
    start_event(journal, next_event)
    recovery.commit_candidate(next_event, initial, candidate)
    assert wal.inspect().latest_snapshot_sequence == 2


def test_normal_commit_order_and_artifacts_are_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    initial = recovery.prepare_startup().snapshot
    candidate = snapshot(1, 0.4)
    item = event("normal-commit", 1)
    start_event(journal, item)
    order: list[str] = []
    append_prepared = journal.append_prepared
    append_transition = wal.append_transition
    save = store.save
    append_completed = journal.append_completed

    def prepared(
        prepared_event: AgentEvent,
        state_hash_before: str,
        state_hash_after: str,
        wal_generation_id: str | None = None,
    ) -> None:
        order.append("prepared")
        append_prepared(
            prepared_event,
            state_hash_before,
            state_hash_after,
            wal_generation_id,
        )

    def append_wal_transition(**kwargs: object) -> TransitionRecord:
        order.append("wal")
        return append_transition(**kwargs)

    def publish(value: AgentStateSnapshot) -> None:
        order.append("snapshot")
        save(value)

    def completed(
        completed_event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str | None = None,
        wal_record_id: str | None = None,
        wal_record_hash: str | None = None,
    ) -> None:
        order.append("completed")
        append_completed(
            completed_event,
            snapshot_sequence,
            snapshot_hash,
            wal_generation_id,
            wal_record_id,
            wal_record_hash,
        )

    monkeypatch.setattr(journal, "append_prepared", prepared)
    monkeypatch.setattr(wal, "append_transition", append_wal_transition)
    monkeypatch.setattr(store, "save", publish)
    monkeypatch.setattr(journal, "append_completed", completed)

    transition = recovery.commit_candidate(item, initial, candidate)

    assert order == ["prepared", "wal", "snapshot", "completed"]
    lifecycles = [record.lifecycle for record in journal.inspect().records]
    assert lifecycles[-2:] == [
        EventLifecycle.PREPARED,
        EventLifecycle.COMPLETED,
    ]  # WAL and snapshot lie between these journal boundaries.
    assert store.load() == candidate
    assert wal.reconstruct(sequence=1) == candidate
    assert transition.processing_sequence == 1
    assert journal.inspect().records[-2].lifecycle is EventLifecycle.PREPARED
    assert journal.inspect().records[-1].lifecycle is EventLifecycle.COMPLETED


def test_exact_current_reconstructs_missing_canonical_snapshot(tmp_path: Path) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    initial = recovery.prepare_startup().snapshot
    candidate = snapshot(1, 0.4)
    item = event("exact-current", 1)
    start_event(journal, item)
    recovery.commit_candidate(item, initial, candidate)
    store.path.unlink()

    result = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert result.snapshot == candidate
    assert result.exact_current_reconstructed
    assert not result.external_reconciliation_required
    assert store.load() == candidate


def test_uncommitted_wal_tail_is_recovered_into_new_generation(tmp_path: Path) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    initial = recovery.prepare_startup().snapshot
    candidate = snapshot(1, 0.4)
    item = event("uncommitted-tail", 1)
    before_hash = store.snapshot_hash(initial)
    after_hash = store.snapshot_hash(candidate)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    start_event(journal, item)
    journal.append_prepared(
        item, before_hash, after_hash, str(manifest.active_generation_id)
    )
    wal.append_transition(
        event_id=uuid5(NAMESPACE_URL, "uncommitted-tail"),
        event_type=item.event_type.value,
        event_source=item.source.value,
        processing_sequence=1,
        prior_snapshot=initial,
        candidate_snapshot=candidate,
    )

    result = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert result.snapshot == initial
    assert not result.exact_current_reconstructed
    assert (
        wal.inspect().active_manifest.active_generation_id
        != manifest.active_generation_id
    )
    assert journal.inspect().open_recoveries == ()


def test_corrupt_active_wal_rolls_back_from_anchor_and_second_startup_keeps_gate(
    tmp_path: Path,
) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    result = recovery.prepare_startup()
    recovery.publish_boot_anchor(result)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    with generation.open("ab") as output:
        output.write(b"corrupt-tail\n")

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    restarted = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert rolled_back.true_rollback_performed
    assert rolled_back.external_reconciliation_required
    assert restarted.snapshot == result.snapshot
    assert restarted.external_reconciliation_required
    assert wal.inspect().active_manifest.external_reconciliation_required


def test_boot_anchor_survives_rotation_of_its_original_journal_record(
    tmp_path: Path,
) -> None:
    store = AgentStateStore(
        tmp_path / "agent_state.json", baseline_surprisal=1.0, clock=lambda: NOW
    )
    journal = EventJournal(tmp_path / "events.jsonl", 1_500, 2, clock=lambda: NOW)
    wal = StateWAL(tmp_path / "wal")
    recovery = StateRecoveryCoordinator(store, journal, wal)
    result = recovery.prepare_startup()
    recovery.publish_boot_anchor(result)
    anchor = wal.inspect_boot_anchor_optional()
    assert anchor is not None

    for sequence in range(1, 21):
        item = event(f"rotated-failure-{sequence}", sequence)
        start_event(journal, item)
        journal.append_failed(item, 0, result.snapshot_hash)

    assert all(
        record.record_id != str(anchor.journal_tail_record_id)
        for record in journal.inspect().records
    )
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    with generation.open("ab") as output:
        output.write(b"corrupt-tail\n")

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert rolled_back.snapshot == result.snapshot
    assert rolled_back.processing_high_water == 20
    assert rolled_back.external_reconciliation_required


def test_recovery_prepared_crash_resumes_same_id_without_open_recovery_accumulation(
    tmp_path: Path,
) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    initial = recovery.prepare_startup().snapshot
    candidate = snapshot(1, 0.4)
    item = event("recovery-crash", 1)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    start_event(journal, item)
    journal.append_prepared(
        item,
        store.snapshot_hash(initial),
        store.snapshot_hash(candidate),
        str(manifest.active_generation_id),
    )
    wal.append_transition(
        event_id=uuid5(NAMESPACE_URL, "recovery-crash"),
        event_type=item.event_type.value,
        event_source=item.source.value,
        processing_sequence=1,
        prior_snapshot=initial,
        candidate_snapshot=candidate,
    )

    def fail_generation(stage: str) -> None:
        if stage == "generation_write":
            raise OSError("injected crash")

    failing_wal = StateWAL(wal.root, failure_hook=fail_generation)
    with pytest.raises(StateWALError):
        StateRecoveryCoordinator(store, journal, failing_wal).prepare_startup()
    pending = journal.inspect().open_recoveries
    assert len(pending) == 1
    recovery_id = pending[0].recovery_id
    partial = wal.root / "generations" / f"{pending[0].wal_generation_id}.jsonl"
    partial.write_bytes(b'{"partial":')
    partial.chmod(0o600)

    resumed = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    records = journal.inspect().records
    assert resumed.snapshot == initial
    assert journal.inspect().open_recoveries == ()
    assert [
        record.recovery_id
        for record in records
        if record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
    ] == [recovery_id]
    assert list((wal.root / "generations").glob(f".{partial.name}.*.invalid"))


def test_recovery_resumes_after_new_generation_before_snapshot_publication(
    tmp_path: Path,
) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    recovery.prepare_startup()
    initial, _candidate = append_uncommitted_candidate(
        store, journal, wal, name="recovery-save-crash"
    )

    def fail_save(stage: AgentStateSaveStage) -> None:
        if stage is AgentStateSaveStage.TEMP_WRITE:
            raise OSError("injected crash")

    failing_store = AgentStateStore(
        store.path, baseline_surprisal=1.0, save_stage_hook=fail_save
    )
    with pytest.raises(AgentStateSaveError):
        StateRecoveryCoordinator(failing_store, journal, wal).prepare_startup()
    pending = journal.inspect().open_recoveries
    assert len(pending) == 1
    recovery_id = pending[0].recovery_id
    assert wal.inspect().latest_snapshot_sequence == 0
    assert store.load() == initial

    resumed = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert resumed.snapshot == initial
    assert journal.inspect().open_recoveries == ()
    assert any(
        record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
        and record.recovery_id == recovery_id
        for record in journal.inspect().records
    )


def test_recovery_resumes_after_snapshot_before_recovery_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    recovery.prepare_startup()
    initial, _candidate = append_uncommitted_candidate(
        store, journal, wal, name="recovery-completion-crash"
    )
    append_completed = journal.append_recovery_completed

    def fail_completed(*_args: object, **_kwargs: object) -> None:
        raise EventJournalAppendError(
            EventJournalAppendStage.FILE_FSYNC, published=False
        )

    monkeypatch.setattr(journal, "append_recovery_completed", fail_completed)
    with pytest.raises(EventJournalAppendError):
        StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    pending = journal.inspect().open_recoveries
    assert len(pending) == 1
    assert store.load() == initial

    monkeypatch.setattr(journal, "append_recovery_completed", append_completed)
    resumed = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert resumed.snapshot == initial
    assert journal.inspect().open_recoveries == ()


def test_stale_recovery_result_is_rejected_when_publishing_boot_anchor(
    tmp_path: Path,
) -> None:
    recovery, store, journal, wal = coordinator(tmp_path)
    stale = recovery.prepare_startup()
    initial = stale.snapshot
    item = event("stale-anchor", 1)
    start_event(journal, item)
    recovery.commit_candidate(item, initial, snapshot(1, 0.4))

    with pytest.raises(StateRecoveryError, match="stale"):
        recovery.publish_boot_anchor(stale)
