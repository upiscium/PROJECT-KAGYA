"""Startup reconciliation coverage using the durable local authorities."""

from datetime import UTC, datetime
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.memory.episodic_participant import (
    EpisodicWrite,
    MemoryEpisodicParticipant,
)
from kagya.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from kagya.runtime.agent_state import (
    AgentStateSnapshot,
    AgentStateStore,
    EmotionStateSnapshot,
    WorkingMemorySnapshot,
)
from kagya.runtime.event_journal import (
    EventJournal,
    EventJournalAppendError,
    EventJournalAppendStage,
    EventLifecycle,
    ParticipantOutcome,
    TransactionKind,
)
from kagya.runtime.startup_reconciliation import (
    StartupReconciliationCoordinator,
    StartupReconciliationError,
)
from kagya.runtime.state_recovery import StateRecoveryCoordinator, StateRecoveryError
from kagya.runtime.state_wal import StateWAL
from kagya.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionBinding,
    TransactionCoordinator,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)
PRIVATE = "PRIVATE-STARTUP-PAYLOAD-R07"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "startup_db1",
                    "db2_collection": "startup_db2",
                }
            )
        }
    )


def _event(name: str, sequence: int = 1) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, name)),
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        NOW,
        sequence,
    )


def _snapshot(sequence: int, valence: float = 0.4) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=valence, arousal=0.2, optimal_loss=1.0
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )


def _graph(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    store = AgentStateStore(
        tmp_path / "state.json", baseline_surprisal=1.0, clock=lambda: NOW
    )
    journal = EventJournal(tmp_path / "events.jsonl", 100_000, 4, clock=lambda: NOW)
    wal = StateWAL(tmp_path / "wal")
    recovery = StateRecoveryCoordinator(store, journal, wal)
    boot = recovery.prepare_startup()
    journal.append_v3_migration_checkpoint()
    StartupReconciliationCoordinator(
        journal, recovery, memory
    ).ensure_adoption_baseline(boot)
    return memory, store, journal, wal, recovery


def _participant(memory: DualMemorySystem, text: str = "visible input"):
    return MemoryEpisodicParticipant(
        memory,
        EpisodicWrite(
            user_input=text,
            response="visible response",
            loss=0.2,
            emotion_valence=0.3,
            emotion_arousal=0.4,
            record_type=MemoryRecordType.EPISODIC_LOG,
            created_at=NOW.isoformat(),
        ),
    )


def _prepared_transaction(
    journal: EventJournal, item: AgentEvent, participant
) -> TransactionCoordinator:
    journal.append_accepted(item)
    journal.append_started(item)
    coordinator = TransactionCoordinator(journal, lambda _event, _evidence: None)
    coordinator.prepare_result(item, CoordinatedResult("public", (participant,)))
    return coordinator


def test_clean_startup_establishes_one_adoption_baseline(tmp_path: Path) -> None:
    memory, _store, journal, _wal, recovery = _graph(tmp_path)
    before = journal.path.read_bytes()
    baseline = journal.inspect().baselines[0]

    current = recovery.prepare_startup()
    repeated = StartupReconciliationCoordinator(
        journal, recovery, memory
    ).ensure_adoption_baseline(current)

    assert repeated == baseline
    assert len(journal.inspect().baselines) == 1
    assert journal.path.read_bytes() == before


def test_true_rollback_before_adoption_baseline_remains_gated(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    store = AgentStateStore(
        tmp_path / "state.json", baseline_surprisal=1.0, clock=lambda: NOW
    )
    journal = EventJournal(tmp_path / "events.jsonl", 100_000, 4, clock=lambda: NOW)
    wal = StateWAL(tmp_path / "wal")
    recovery = StateRecoveryCoordinator(store, journal, wal)
    boot = recovery.prepare_startup()
    journal.append_v3_migration_checkpoint()
    recovery.publish_boot_anchor(boot)
    item = _event("pre-adoption-history")
    participant = _participant(memory)
    transaction_coordinator = _prepared_transaction(journal, item, participant)
    evidence = recovery.commit_internal_candidate(item, store.load(), _snapshot(1))
    transaction_coordinator.finalize_event(item, evidence)
    recovery.complete_committed_event(item, evidence)
    current = recovery.prepare_startup()
    baseline = StartupReconciliationCoordinator(
        journal, recovery, memory
    ).ensure_adoption_baseline(current)
    assert baseline.snapshot_sequence == 1
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    store.path.unlink()
    journal_before = journal.path.read_bytes()
    generation_before = generation.read_bytes()
    manifest_before = (wal.root / "manifest.json").read_bytes()

    with pytest.raises(
        StateRecoveryError, match="recovery target predates participant baseline"
    ):
        StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    assert not store.path.exists()
    assert journal.path.read_bytes() == journal_before
    assert generation.read_bytes() == generation_before
    assert (wal.root / "manifest.json").read_bytes() == manifest_before
    assert journal.inspect().open_startup_reconciliations == ()


def test_pre_internal_aborts_and_records_terminal_evidence(tmp_path: Path) -> None:
    memory, store, journal, _wal, recovery = _graph(tmp_path)
    item = _event("startup-pre-internal")
    participant = _participant(memory, PRIVATE)
    _prepared_transaction(journal, item, participant)

    result = StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions()

    assert result == (True, None)
    inspection = journal.inspect()
    assert not inspection.open_transactions
    transaction = inspection.aborted_transactions[0]
    assert transaction.terminal_lifecycle is EventLifecycle.TRANSACTION_ABORTED
    assert transaction.abort_outcomes[0][1].value in {"aborted", "already_absent"}
    assert not participant.pending_path(
        TransactionBinding(
            transaction_id=transaction.transaction_id,
            event_id=transaction.event_id,
            processing_sequence=transaction.processing_sequence,
            participant_id=participant.participant_id,
            operation_digest=participant.operation_digest,
            transaction_kind=transaction.kind,
        )
    ).exists()
    assert PRIVATE not in journal.path.read_text()


def test_wal_only_tail_aborts_then_runs_r06_uncommitted_tail_recovery(
    tmp_path: Path,
) -> None:
    memory, store, journal, wal, recovery = _graph(tmp_path)
    item = _event("startup-wal-only-tail")
    participant = _participant(memory, PRIVATE)
    _prepared_transaction(journal, item, participant)
    initial = store.load()
    candidate = _snapshot(1)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    journal.append_prepared(
        item,
        store.snapshot_hash(initial),
        store.snapshot_hash(candidate),
        str(manifest.active_generation_id),
    )
    wal.append_transition(
        event_id=UUID(item.event_id),
        event_type=item.event_type.value,
        event_source=item.source.value,
        processing_sequence=1,
        prior_snapshot=initial,
        candidate_snapshot=candidate,
    )
    original_generation = manifest.active_generation_id

    assert StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions() == (True, None)
    recovered = StateRecoveryCoordinator(store, journal, wal).prepare_startup()

    inspection = journal.inspect()
    assert recovered.snapshot == initial
    assert store.load() == initial
    assert not inspection.open_transactions
    assert not inspection.open_events
    assert len(inspection.aborted_transactions) == 1
    assert wal.inspect().active_manifest is not None
    assert wal.inspect().active_manifest.active_generation_id != original_generation
    transaction = inspection.aborted_transactions[0]
    assert not participant.pending_path(
        TransactionBinding(
            transaction_id=transaction.transaction_id,
            event_id=transaction.event_id,
            processing_sequence=transaction.processing_sequence,
            participant_id=participant.participant_id,
            operation_digest=participant.operation_digest,
            transaction_kind=transaction.kind,
        )
    ).exists()
    assert PRIVATE not in journal.path.read_text()


def test_pre_internal_abort_resumes_after_artifact_removal(tmp_path: Path) -> None:
    memory, _store, journal, _wal, recovery = _graph(tmp_path)
    item = _event("startup-pre-internal-abort-resume")
    participant = _participant(memory)
    _prepared_transaction(journal, item, participant)
    transaction = journal.inspect().open_transactions[0]
    requirement = transaction.required_participants[0]
    binding = TransactionBinding(
        transaction_id=transaction.transaction_id,
        event_id=transaction.event_id,
        processing_sequence=transaction.processing_sequence,
        participant_id=requirement.participant_id,
        operation_digest=requirement.operation_digest,
        transaction_kind=transaction.kind,
    )
    MemoryEpisodicParticipant.abort_pending(memory, binding)

    assert StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions() == (True, None)
    aborted = journal.inspect().aborted_transactions[0]
    assert aborted.abort_outcomes[0][1].value == "already_absent"


def test_pre_internal_committed_memory_cannot_be_recorded_as_aborted(
    tmp_path: Path,
) -> None:
    memory, _store, journal, _wal, recovery = _graph(tmp_path)
    item = _event("startup-pre-internal-committed-memory")
    participant = _participant(memory)
    _prepared_transaction(journal, item, participant)
    transaction = journal.inspect().open_transactions[0]
    requirement = transaction.required_participants[0]
    binding = TransactionBinding(
        transaction_id=transaction.transaction_id,
        event_id=transaction.event_id,
        processing_sequence=transaction.processing_sequence,
        participant_id=requirement.participant_id,
        operation_digest=requirement.operation_digest,
        transaction_kind=transaction.kind,
    )
    participant.finalize(binding)

    assert StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions() == (
        False,
        "external_participant_reconciliation_required",
    )
    inspection = journal.inspect()
    assert inspection.open_transactions
    assert not inspection.aborted_transactions


def test_pre_internal_committed_memory_with_pending_cannot_be_aborted(
    tmp_path: Path,
) -> None:
    memory, _store, journal, _wal, recovery = _graph(tmp_path)
    item = _event("startup-pre-internal-committed-and-pending")
    participant = _participant(memory)
    _prepared_transaction(journal, item, participant)
    transaction = journal.inspect().open_transactions[0]
    requirement = transaction.required_participants[0]
    binding = TransactionBinding(
        transaction_id=transaction.transaction_id,
        event_id=transaction.event_id,
        processing_sequence=transaction.processing_sequence,
        participant_id=requirement.participant_id,
        operation_digest=requirement.operation_digest,
        transaction_kind=transaction.kind,
    )
    operation = participant.operation
    memory.publish_coordinated_episodic(
        participant.episode_id(transaction.transaction_id),
        operation.user_input,
        operation.response,
        loss=operation.loss,
        emotion_valence=operation.emotion_valence,
        emotion_arousal=operation.emotion_arousal,
        record_type=operation.record_type,
        created_at=operation.created_at,
    )
    pending = participant.pending_path(binding)
    assert pending.exists()

    assert StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions()[0] is False
    assert pending.exists()
    assert journal.inspect().open_transactions
    assert not journal.inspect().aborted_transactions


def test_internal_commit_rolls_forward_without_state_replay(
    tmp_path: Path, monkeypatch
) -> None:
    memory, store, journal, _wal, recovery = _graph(tmp_path)
    initial = store.load()
    candidate = _snapshot(1)
    item = _event("startup-internal-commit")
    participant = _participant(memory)
    _prepared_transaction(journal, item, participant)
    recovery.commit_internal_candidate(item, initial, candidate)

    monkeypatch.setattr(store, "save", lambda _value: pytest.fail("state replayed"))
    result = StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions()

    assert result == (True, None)
    assert store.load() == candidate
    assert journal.inspect().completed_transactions[0].participant_outcomes == (
        (participant.participant_id, ParticipantOutcome.FINALIZED),
    )
    assert participant.pending_path(
        TransactionBinding(
            transaction_id=journal.inspect().completed_transactions[0].transaction_id,
            event_id=item.event_id,
            processing_sequence=1,
            participant_id=participant.participant_id,
            operation_digest=participant.operation_digest,
            transaction_kind=TransactionKind.EVENT_MUTATION,
        )
    ).exists() is False


def test_ambiguous_commit_proof_fails_closed(tmp_path: Path) -> None:
    memory, store, journal, _wal, recovery = _graph(tmp_path)
    initial = store.load()
    item = _event("startup-ambiguous")
    participant = _participant(memory)
    _prepared_transaction(journal, item, participant)
    recovery.commit_internal_candidate(item, initial, _snapshot(1, 0.4))
    store.save(_snapshot(1, 0.9))

    with pytest.raises(StartupReconciliationError, match="ambiguous"):
        StartupReconciliationCoordinator(
            journal, recovery, memory
        ).reconcile_open_transactions()
    assert journal.inspect().open_transactions
    assert not journal.inspect().completed_transactions


@pytest.mark.parametrize("mode", ["missing", "conflict"])
def test_participant_missing_or_conflicting_evidence_degrades(
    tmp_path: Path, mode: str
) -> None:
    memory, store, journal, _wal, recovery = _graph(tmp_path)
    item = _event(f"startup-degraded-{mode}")
    participant = _participant(memory)
    _prepared_transaction(journal, item, participant)
    transaction = journal.inspect().open_transactions[0]
    binding = TransactionBinding(
        transaction_id=transaction.transaction_id,
        event_id=transaction.event_id,
        processing_sequence=1,
        participant_id=participant.participant_id,
        operation_digest=participant.operation_digest,
        transaction_kind=transaction.kind,
    )
    path = participant.pending_path(binding)
    if mode == "missing":
        recovery.commit_internal_candidate(item, store.load(), _snapshot(1))
        path.unlink()
    else:
        path.write_text(json.dumps({"conflict": PRIVATE}))

    assert StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_open_transactions() == (
        False,
        "external_participant_reconciliation_required",
    )
    assert journal.inspect().open_transactions


def test_true_rollback_reconciles_aggregate_and_clears_gate(tmp_path: Path) -> None:
    memory, store, journal, wal, recovery = _graph(tmp_path)
    boot = recovery.prepare_startup()
    recovery.publish_boot_anchor(boot)
    initial = store.load()
    item = _event("startup-true-rollback")
    participant = _participant(memory, PRIVATE)
    transaction_coordinator = _prepared_transaction(journal, item, participant)
    candidate = _snapshot(1)
    evidence = recovery.commit_internal_candidate(item, initial, candidate)
    transaction_coordinator.finalize_event(item, evidence)
    recovery.complete_committed_event(item, evidence)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    store.path.unlink()

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    assert rolled_back.true_rollback_performed
    assert rolled_back.external_reconciliation_required
    coordinator = StartupReconciliationCoordinator(journal, recovery, memory)
    result = coordinator.reconcile_recovery_gate(rolled_back)

    assert result.participants_consistent
    assert not result.recovery.external_reconciliation_required
    assert result.recovery.processing_high_water == 1
    assert result.recovery.snapshot.last_processed_event_sequence == 0
    inspection = journal.inspect()
    transaction = inspection.completed_transactions[0]
    assert inspection.baselines[-1].participant_registry
    assert inspection.completed_startup_reconciliations[-1].completed
    outcomes = inspection.completed_startup_reconciliations[-1].participant_outcomes
    assert {participant_id for participant_id, _digest, _outcome in outcomes} == {
        "memory.episodic",
        "session.turn",
    }
    assert inspection.terminal_gate_clear is not None
    assert any(
        record.lifecycle is EventLifecycle.CLEAR_PREPARED
        for record in inspection.records
    )
    assert inspection.records[-1].lifecycle is EventLifecycle.CLEARED
    assert not wal.inspect().active_manifest.external_reconciliation_required
    assert PRIVATE not in journal.path.read_text()
    assert PRIVATE not in generation.read_text()
    committed = memory.get_committed_episodic(
        participant.episode_id(transaction.transaction_id)
    )
    assert committed is not None
    assert committed.metadata["coordination_schema"] == 1
    assert committed.metadata["extra"] == "{}"
    assert "private" not in committed.metadata
    assert committed.document.startswith(f"User: {PRIVATE}")


def test_gate_reconciliation_is_idempotent_after_clear(tmp_path: Path) -> None:
    memory, store, journal, wal, recovery = _graph(tmp_path)
    boot = recovery.prepare_startup()
    recovery.publish_boot_anchor(boot)
    item = _event("startup-idempotent-gate")
    participant = _participant(memory)
    transaction_coordinator = _prepared_transaction(journal, item, participant)
    initial = store.load()
    candidate = _snapshot(1)
    evidence = recovery.commit_internal_candidate(item, initial, candidate)
    transaction_coordinator.finalize_event(item, evidence)
    recovery.complete_committed_event(item, evidence)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    store.path.unlink()
    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    coordinator = StartupReconciliationCoordinator(journal, recovery, memory)
    first = coordinator.reconcile_recovery_gate(rolled_back)
    before = journal.path.read_bytes()
    second = coordinator.reconcile_recovery_gate(first.recovery)

    assert second.recovery == first.recovery
    assert journal.path.read_bytes() == before


def test_gate_clear_resumes_after_wal_cas_before_journal_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, store, journal, wal, recovery = _graph(tmp_path)
    boot = recovery.prepare_startup()
    recovery.publish_boot_anchor(boot)
    item = _event("startup-gate-clear-crash")
    participant = _participant(memory)
    transaction_coordinator = _prepared_transaction(journal, item, participant)
    initial = store.load()
    candidate = _snapshot(1)
    evidence = recovery.commit_internal_candidate(item, initial, candidate)
    transaction_coordinator.finalize_event(item, evidence)
    recovery.complete_committed_event(item, evidence)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    store.path.unlink()
    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    coordinator = StartupReconciliationCoordinator(journal, recovery, memory)
    def crash_before_terminal(*_args: object, **_kwargs: object) -> None:
        raise EventJournalAppendError(EventJournalAppendStage.WRITE, published=False)

    monkeypatch.setattr(journal, "append_cleared", crash_before_terminal)
    with pytest.raises(EventJournalAppendError):
        coordinator.reconcile_recovery_gate(rolled_back)
    assert journal.inspect().open_gate_clear is not None
    assert wal.inspect().active_manifest is not None
    assert not wal.inspect().active_manifest.external_reconciliation_required

    journal.close()
    reopened = EventJournal(journal.path, 100_000, 4, clock=lambda: NOW)
    reopened_recovery = StateRecoveryCoordinator(
        AgentStateStore(store.path, baseline_surprisal=1.0, clock=lambda: NOW),
        reopened,
        StateWAL(wal.root),
    )
    restarted = StartupReconciliationCoordinator(
        reopened, reopened_recovery, DualMemorySystem(_settings(tmp_path))
    )

    assert restarted.resume_prepared_gate_clear()
    assert reopened.inspect().terminal_gate_clear is not None
    assert not restarted.resume_prepared_gate_clear()
