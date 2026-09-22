"""R07 Semantic batch participant and projection-boundary tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.memory import (
    DualMemorySystem,
    MemorySemanticParticipant,
    SemanticBatchEntry,
    SemanticBatchOperation,
    SemanticCreateIntent,
    SemanticRevisionIntent,
    SemanticStore,
    semantic_id_for_batch_entry,
)
from kagya.memory.semantic_lifecycle import (
    SemanticLifecycle,
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticRevisionReason,
    semantic_content_digest,
)
from kagya.runtime import (
    AbortOutcome,
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    ParticipantDivergedError,
    ParticipantOutcome,
    TransactionBinding,
    TransactionCoordinator,
    TransactionKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "semantic_participant_db1",
                    "db2_collection": "semantic_participant_db2",
                }
            )
        }
    )


def _event(sequence: int = 1) -> AgentEvent:
    return AgentEvent(
        event_id=f"11111111-1111-4111-8111-{sequence:012d}",
        event_type=AgentEventType.SLEEP,
        source=AgentEventSource.API_SLEEP_RUN,
        requested_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=sequence),
        processing_sequence=sequence,
    )


def _create_participant(
    memory: DualMemorySystem, *, event: AgentEvent | None = None
) -> tuple[MemorySemanticParticipant, TransactionBinding, SemanticRevision]:
    current_event = event or _event()
    transaction_id = TransactionCoordinator.derive_transaction_id(
        current_event, TransactionKind.EVENT_MUTATION
    )
    semantic_id = semantic_id_for_batch_entry(transaction_id, 0)
    revision = SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content="visible semantic fact",
        content_digest=semantic_content_digest("visible semantic fact"),
        created_at=current_event.requested_at,
        event_id=current_event.event_id,
        event_sequence=current_event.processing_sequence,
    )
    operation = SemanticBatchOperation(
        transaction_id,
        (SemanticBatchEntry(0, SemanticCreateIntent(revision)),),
    )
    participant = MemorySemanticParticipant(
        memory,
        SemanticStore.from_memory_root(memory.settings.memory.persist_directory),
        operation,
    )
    binding = TransactionBinding(
        transaction_id,
        current_event.event_id,
        current_event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    return participant, binding, revision


def test_deterministic_identity_and_bounded_batch_contract() -> None:
    transaction_id = "22222222-2222-4222-8222-222222222222"
    assert semantic_id_for_batch_entry(transaction_id, 0) == semantic_id_for_batch_entry(
        transaction_id, 0
    )
    with pytest.raises(ValueError):
        semantic_id_for_batch_entry(transaction_id, 128)

    with pytest.raises(ValueError, match="contiguous"):
        SemanticBatchOperation(
            transaction_id,
            (SemanticBatchEntry(1, SemanticCreateIntent(_revision_for_slot(transaction_id, 0, 1))),),
        )


def _revision_for_slot(transaction_id: str, index: int, sequence: int) -> SemanticRevision:
    semantic_id = semantic_id_for_batch_entry(transaction_id, index)
    content = f"slot {index}"
    return SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content=content,
        content_digest=semantic_content_digest(content),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        event_id=f"11111111-1111-4111-8111-{sequence:012d}",
        event_sequence=sequence,
    )


def test_prepare_is_non_authoritative_and_finalize_is_idempotent(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)

    participant.prepare(binding)

    assert participant.store.load_pending(binding.transaction_id) is not None
    assert participant.store.load_current(revision.semantic_id) is None
    assert memory.db2.get(ids=[revision.semantic_id], include=["documents", "metadatas"])["ids"] == []

    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert participant.store.load_pending(binding.transaction_id) is None
    assert memory.get_committed_semantic(revision.semantic_id) is not None

    recovered = MemorySemanticParticipant.from_pending(
        memory,
        participant.store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
    )
    assert recovered.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT


def test_pre_internal_abort_removes_pending_without_publication(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)

    assert participant.abort(binding) is AbortOutcome.ABORTED
    assert participant.store.load_current(revision.semantic_id) is None
    assert participant.store.load_receipt(binding.transaction_id) is None


def test_partial_lifecycle_publication_rolls_forward_without_duplicate_revision(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)
    participant.store.publish_create(revision, participant.operation_digest)

    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.store.load_current(revision.semantic_id) is not None
    assert participant.store.load_receipt(binding.transaction_id) is not None


def test_divergent_legacy_projection_is_not_overwritten(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant, binding, revision = _create_participant(memory)
    participant.prepare(binding)
    participant.store.publish_create(revision, participant.operation_digest)
    # Chroma IDs are opaque; use the authoritative deterministic identity for
    # the collision row and retain the exact legacy bytes for the assertion.
    collision_id = revision.semantic_id
    memory.db2.add(
        ids=[collision_id],
        documents=["legacy collision"],
        metadatas=[
            {
                "text": "legacy collision",
                "source_episode_ids": "[]",
                "record_type": "semantic_memory",
                "created_at": "legacy",
                "extra": "{}",
            }
        ],
    )
    before = memory.db2.get(ids=[collision_id], include=["documents", "metadatas"])

    with pytest.raises(ParticipantDivergedError):
        participant.finalize(binding)

    assert memory.db2.get(ids=[collision_id], include=["documents", "metadatas"]) == before


def test_non_active_revision_removes_search_projection(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    create, create_binding, created = _create_participant(memory)
    create.prepare(create_binding)
    create.finalize(create_binding)
    event = _event(2)
    transaction_id = TransactionCoordinator.derive_transaction_id(
        event, TransactionKind.EVENT_MUTATION
    )
    revised = SemanticRevision(
        semantic_id=created.semantic_id,
        revision=1,
        semantic_content=created.semantic_content,
        content_digest=created.content_digest,
        created_at=event.requested_at,
        lifecycle=SemanticLifecycle.RETRACTED,
        operation=SemanticRevisionOperation.RETRACT,
        reason=SemanticRevisionReason.RETRACTION,
        previous_revision_digest=created.revision_digest,
        event_id=event.event_id,
        event_sequence=event.processing_sequence,
    )
    operation = SemanticBatchOperation(
        transaction_id,
        (SemanticBatchEntry(0, SemanticRevisionIntent(revised, 0, created.revision_digest)),),
    )
    participant = MemorySemanticParticipant(memory, create.store, operation)
    binding = TransactionBinding(
        transaction_id,
        event.event_id,
        event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    participant.prepare(binding)
    participant.finalize(binding)

    assert memory.db2.get(ids=[created.semantic_id], include=["documents", "metadatas"])["ids"] == []


def test_oversized_batch_rejected_before_publication() -> None:
    transaction_id = "33333333-3333-4333-8333-333333333333"
    with pytest.raises(ValueError):
        SemanticBatchOperation(
            transaction_id,
            tuple(
                SemanticBatchEntry(
                    index,
                    SemanticCreateIntent(_revision_for_slot(transaction_id, index, 1)),
                )
                for index in range(129)
            ),
        )
