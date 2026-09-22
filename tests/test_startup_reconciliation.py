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
from kagya.memory.working_memory_resolver import MemoryWorkingMemoryResolver
from kagya.models import ModelProvider
from kagya.persona.prompt_builder import PromptBuilder
from kagya.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from kagya.runtime.agent_state import (
    AgentStateSnapshotV2,
    AgentStateSnapshotV3,
    AgentStateStore,
    EmotionStateSnapshot,
    ContextFrameSnapshot,
    ContextStateSnapshot,
    WorkingMemoryItemSnapshot,
    WorkingMemorySnapshot,
)
from kagya.runtime.context import ContextRegistry
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
from kagya.runtime.working_memory import (
    WorkingMemory,
    WorkingMemorySourceKind,
    working_memory_item_id,
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


def _snapshot(sequence: int, valence: float = 0.4) -> AgentStateSnapshotV2:
    return AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=valence, arousal=0.2, optimal_loss=1.0
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )


def _snapshot_with_working_memory(
    sequence: int,
    references: tuple[tuple[str, str], ...],
    revision: int,
    *,
    context_id: str | None = None,
) -> AgentStateSnapshotV2 | AgentStateSnapshotV3:
    items = tuple(
        WorkingMemoryItemSnapshot(
            item_id=working_memory_item_id(WorkingMemorySourceKind(kind), source_id),
            source_kind=kind,
            source_id=source_id,
            activation=0.7 - index * 0.1,
            salience=0.8 - index * 0.1,
            retention_reason="recent",
            created_revision=index + 1,
            last_activated_revision=index + 1,
        )
        for index, (kind, source_id) in enumerate(references)
    )
    base = AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=0.4, arousal=0.2, optimal_loss=1.0
        ),
        working_memory=WorkingMemorySnapshot(revision=revision, items=items),
    )
    if context_id is None:
        return base
    frame = ContextFrameSnapshot(
        context_id=context_id,
        context_type="conversation",
        source_channel="chat",
        source_session_id=f"session-{context_id}",
        participant_refs=(f"participant-{context_id}",),
        parent_context_id=None,
        related_context_ids=(),
        status="active",
        created_revision=1,
        last_modified_revision=2,
        started_at=NOW,
        last_active_at=NOW,
    )
    payload = base.model_dump()
    payload["schema_version"] = 3
    payload["context_state"] = ContextStateSnapshot(
        revision=2,
        current_context_id=context_id,
        frames=(frame,),
        interlocutor_bindings=(),
    )
    return AgentStateSnapshotV3.model_validate(payload)


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


def _participant(
    memory: DualMemorySystem,
    text: str = "visible input",
    *,
    context_id: str | None = None,
):
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
            context_id=context_id,
            source_channel="chat" if context_id is not None else None,
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
        coordination_schema=operation.schema_version,
        context_id=operation.context_id,
        source_channel=operation.source_channel,
        source_session_id=operation.source_session_id,
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
        "memory.experience",
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
    assert committed.metadata["coordination_schema"] == 2
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


def test_true_rollback_restores_working_memory_only_and_preserves_newer_episodic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, store, journal, wal, recovery = _graph(tmp_path)
    boot = recovery.prepare_startup()
    recovery.publish_boot_anchor(boot)
    semantic_ids = (
        memory.save_semantic("semantic B"),
        memory.save_semantic("semantic D"),
    )

    initial = store.load()
    older = _snapshot_with_working_memory(
        1,
        (("episodic", "memory-a"), ("semantic", semantic_ids[0])),
        2,
        context_id="context-a",
    )
    newer = _snapshot_with_working_memory(
        2,
        (("episodic", "memory-c"), ("semantic", semantic_ids[1])),
        4,
        context_id="context-b",
    )
    first = _event("startup-u5-older-working-memory", 1)
    first_participant = _participant(memory, "record C", context_id="context-a")
    first_coordinator = _prepared_transaction(journal, first, first_participant)
    first_transaction_id = next(
        item.transaction_id
        for item in journal.inspect().open_transactions
        if item.event_id == first.event_id
    )
    first_evidence = recovery.commit_internal_candidate(first, initial, older)
    first_coordinator.finalize_event(first, first_evidence)
    recovery.complete_committed_event(first, first_evidence)
    older_state_bytes = store.path.read_bytes()
    anchored_older = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    recovery.publish_boot_anchor(anchored_older)

    second = _event("startup-u5-newer-working-memory", 2)
    second_participant = _participant(memory, "record D", context_id="context-b")
    second_coordinator = _prepared_transaction(journal, second, second_participant)
    second_transaction_id = next(
        item.transaction_id
        for item in journal.inspect().open_transactions
        if item.event_id == second.event_id
    )
    second_evidence = recovery.commit_internal_candidate(second, older, newer)
    second_coordinator.finalize_event(second, second_evidence)
    recovery.complete_committed_event(second, second_evidence)
    committed_ids = (
        first_participant.episode_id(first_transaction_id),
        second_participant.episode_id(second_transaction_id),
    )
    derived_semantic_id = memory.save_semantic(
        "semantic derived from context B", source_episode_ids=[committed_ids[1]]
    )
    episodic_before = memory.db1.get(
        ids=list(committed_ids), include=["documents", "metadatas"]
    )
    records_before = tuple(
        memory.get_committed_episodic(episode_id) for episode_id in committed_ids
    )
    assert (
        records_before[0] is not None
        and records_before[0].record.context_id == "context-a"
    )
    assert (
        records_before[1] is not None
        and records_before[1].record.context_id == "context-b"
    )
    semantic_before = tuple(
        memory.get_committed_semantic(semantic_id)
        for semantic_id in (*semantic_ids, derived_semantic_id)
    )
    assert semantic_before[-1] is not None
    assert semantic_before[-1].record.context_id == "context-b"

    replay_calls = {
        "retrieve": 0,
        "select": 0,
        "resolve": 0,
        "prompt": 0,
        "model": 0,
    }

    def no_retrieve(*_args: object, **_kwargs: object) -> object:
        replay_calls["retrieve"] += 1
        pytest.fail("startup reconstruction retrieved memory")

    def no_select(*_args: object, **_kwargs: object) -> object:
        replay_calls["select"] += 1
        pytest.fail("startup reconstruction selected Working Memory")

    def no_resolve(*_args: object, **_kwargs: object) -> object:
        replay_calls["resolve"] += 1
        pytest.fail("startup reconstruction resolved Working Memory")

    def no_prompt(*_args: object, **_kwargs: object) -> object:
        replay_calls["prompt"] += 1
        pytest.fail("startup reconstruction built a prompt")

    def no_model(*_args: object, **_kwargs: object) -> object:
        replay_calls["model"] += 1
        pytest.fail("startup reconstruction called the model")

    monkeypatch.setattr(memory, "retrieve_context", no_retrieve)
    monkeypatch.setattr(WorkingMemory, "select", no_select)
    monkeypatch.setattr(MemoryWorkingMemoryResolver, "resolve", no_resolve)
    monkeypatch.setattr(PromptBuilder, "build", no_prompt)
    monkeypatch.setattr(ModelProvider, "generate", no_model)

    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[-1])
    transition["record_hash"] = "0" * 64
    lines[-1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    store.path.unlink()

    rolled_back = StateRecoveryCoordinator(store, journal, wal).prepare_startup()
    assert rolled_back.true_rollback_performed
    assert rolled_back.snapshot == older
    assert rolled_back.snapshot.context_state.current_context_id == "context-a"
    assert tuple(
        frame.context_id for frame in rolled_back.snapshot.context_state.frames
    ) == ("context-a",)
    assert rolled_back.processing_high_water == 2
    assert rolled_back.external_reconciliation_required
    assert store.path.read_bytes() == older_state_bytes

    recovered_registry = ContextRegistry(clock=lambda: NOW)
    recovered_registry.restore_exact(rolled_back.snapshot.context_state.to_registry_state())
    compatibility = recovered_registry.compatibility("context-b", "context-a")
    assert compatibility.relation.value == "unknown_context"
    assert compatibility.score == 0.35

    result = StartupReconciliationCoordinator(
        journal, recovery, memory
    ).reconcile_recovery_gate(rolled_back)
    assert result.participants_consistent
    assert not result.recovery.external_reconciliation_required
    assert result.recovery.snapshot == older
    assert result.recovery.processing_high_water == 2
    assert journal.inspect().processing_high_water == 2
    assert _event("startup-u5-next-sequence", 3).processing_sequence > 2

    assert memory.db1.get(
        ids=list(committed_ids), include=["documents", "metadatas"]
    ) == episodic_before
    assert tuple(
        memory.get_committed_episodic(episode_id) for episode_id in committed_ids
    ) == records_before
    assert tuple(
        memory.get_committed_semantic(semantic_id)
        for semantic_id in (*semantic_ids, derived_semantic_id)
    ) == semantic_before
    inspection = journal.inspect()
    assert not inspection.open_startup_reconciliations
    assert inspection.terminal_gate_clear is not None
    assert not wal.inspect().active_manifest.external_reconciliation_required
    assert {
        item.participant_id for item in inspection.baselines[0].participant_registry
    } == {"memory.episodic", "memory.experience", "session.turn"}
    assert "working_memory" not in journal.path.read_text()
    assert replay_calls == {
        "retrieve": 0,
        "select": 0,
        "resolve": 0,
        "prompt": 0,
        "model": 0,
    }


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
    terminal_bytes = reopened.path.read_bytes()
    baseline_count = len(reopened.inspect().baselines)
    assert not restarted.resume_prepared_gate_clear()
    assert reopened.path.read_bytes() == terminal_bytes
    assert len(reopened.inspect().baselines) == baseline_count
    assert sum(
        item.lifecycle is EventLifecycle.CLEARED for item in reopened.inspect().records
    ) == 1


def test_gate_clear_resumes_from_prepared_record_before_wal_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, store, journal, wal, recovery = _graph(tmp_path)
    boot = recovery.prepare_startup()
    recovery.publish_boot_anchor(boot)
    item = _event("startup-gate-clear-before-wal")
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

    def crash_before_wal_clear(*_args: object, **_kwargs: object) -> object:
        raise EventJournalAppendError(EventJournalAppendStage.WRITE, published=False)

    monkeypatch.setattr(recovery, "clear_recovery_gate", crash_before_wal_clear)
    with pytest.raises(EventJournalAppendError):
        coordinator.reconcile_recovery_gate(rolled_back)
    assert journal.inspect().open_gate_clear is not None
    assert wal.inspect().active_manifest is not None
    assert wal.inspect().active_manifest.external_reconciliation_required
    baseline_count_before_resume = len(journal.inspect().baselines)

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
    assert not restarted.resume_prepared_gate_clear()
    assert reopened.inspect().terminal_gate_clear is not None
    assert not reopened_recovery.wal.inspect().active_manifest.external_reconciliation_required
    assert len(reopened.inspect().baselines) == baseline_count_before_resume
    assert sum(
        item.lifecycle is EventLifecycle.CLEARED for item in reopened.inspect().records
    ) == 1
