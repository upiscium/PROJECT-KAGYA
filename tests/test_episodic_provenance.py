"""U3 source-Context provenance and R07 operation compatibility tests."""

from dataclasses import fields
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, EpisodicMemoryFormatError, MemoryRecordType
from kagya.memory.episodic_participant import (
    MEMORY_EPISODIC_PARTICIPANT_ID,
    EpisodicWrite,
    MemoryEpisodicParticipant,
    episodic_operation_digest,
)
from kagya.runtime import (
    ParticipantDivergedError,
    ParticipantOutcome,
    StartupParticipantOutcome,
    TransactionBinding,
    TransactionKind,
    WorkingMemoryItem,
    WorkingMemoryItemSnapshot,
    WorkingMemorySnapshot,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
TRANSACTION_ID = "f51090e6-25a3-5d8e-b701-cbbdb8e88dca"
EVENT_ID = "5cefdcd0-88a3-5850-b6cc-72cab6f9989e"
NOW = datetime(2026, 1, 1, tzinfo=UTC)

# Captured from the U2 operation shape before U3.  Do not derive these values
# from the post-U3 implementation under test.
LEGACY_V1_OPERATION = {
    "schema_version": 1,
    "user_input": "staged user",
    "response": "visible response",
    "loss": 0.2,
    "emotion_valence": 0.3,
    "emotion_arousal": 0.4,
    "record_type": "episodic_log",
    "created_at": "2026-01-01T00:00:00+00:00",
}
LEGACY_V1_DIGEST = "83ea382a927616e30c6d01603753c6ced9740c170da1df69c8ec2db6adc1c2f3"
LEGACY_V1_EPISODE_ID = "episode-d5020bc7-9ee9-5409-8c7a-4af6dd665bf1"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "u3_db1",
                    "db2_collection": "u3_db2",
                }
            )
        }
    )


def _operation(
    *,
    schema_version: int = 2,
    context_id: str | None = None,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> EpisodicWrite:
    return EpisodicWrite(
        user_input="staged user",
        response="visible response",
        loss=0.2,
        emotion_valence=0.3,
        emotion_arousal=0.4,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at=NOW.isoformat(),
        context_id=context_id,
        source_channel=source_channel,
        source_session_id=source_session_id,
        schema_version=schema_version,
    )


def _participant(
    memory: DualMemorySystem,
    *,
    schema_version: int = 2,
    context_id: str | None = None,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> MemoryEpisodicParticipant:
    return MemoryEpisodicParticipant(
        memory,
        _operation(
            schema_version=schema_version,
            context_id=context_id,
            source_channel=source_channel,
            source_session_id=source_session_id,
        ),
    )


def _binding(participant: MemoryEpisodicParticipant) -> TransactionBinding:
    return TransactionBinding(
        transaction_id=TRANSACTION_ID,
        event_id=EVENT_ID,
        processing_sequence=1,
        participant_id=MEMORY_EPISODIC_PARTICIPANT_ID,
        operation_digest=participant.operation_digest,
        transaction_kind=TransactionKind.EVENT_MUTATION,
    )


def test_v1_operation_bytes_digest_and_episode_identity_are_unchanged() -> None:
    operation = _operation(schema_version=1)

    assert operation.canonical_dict() == LEGACY_V1_OPERATION
    assert episodic_operation_digest(operation) == LEGACY_V1_DIGEST
    memory = object.__new__(DualMemorySystem)
    participant = MemoryEpisodicParticipant(memory, operation)
    assert participant.episode_id(TRANSACTION_ID) == LEGACY_V1_EPISODE_ID


def test_v2_provenance_changes_digest_and_episode_identity() -> None:
    operation = _operation(
        context_id="context-a", source_channel="chat", source_session_id="session-a"
    )
    channel_changed = _operation(
        context_id="context-a", source_channel="api.chat", source_session_id="session-a"
    )
    session_changed = _operation(
        context_id="context-a", source_channel="chat", source_session_id="session-b"
    )
    context_changed = _operation(
        context_id="context-b", source_channel="chat", source_session_id="session-a"
    )
    participants = [
        MemoryEpisodicParticipant(object.__new__(DualMemorySystem), item)
        for item in (operation, channel_changed, session_changed, context_changed)
    ]

    digests = {participant.operation_digest for participant in participants}
    episode_ids = {
        participant.episode_id(TRANSACTION_ID) for participant in participants
    }
    assert len(digests) == 4
    assert len(episode_ids) == 4


def test_v2_rejects_partial_provenance() -> None:
    with pytest.raises(ValueError):
        _operation(context_id="context-a")
    with pytest.raises(ValueError):
        _operation(source_channel="chat")
    with pytest.raises(ValueError):
        _operation(source_session_id="session-a")


def test_v1_pending_artifact_reconciles_without_v2_rewrite(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    participant = _participant(memory, schema_version=1)
    binding = _binding(participant)

    participant.prepare(binding)
    pending = json.loads(participant.pending_path(binding).read_text())

    assert pending["schema_version"] == 1
    assert pending["operation"] == LEGACY_V1_OPERATION
    assert memory.get_episodic_record(LEGACY_V1_EPISODE_ID) is None

    reopened = MemoryEpisodicParticipant.from_pending(
        DualMemorySystem(settings),
        TRANSACTION_ID,
        MEMORY_EPISODIC_PARTICIPANT_ID,
        LEGACY_V1_DIGEST,
    )
    assert reopened.operation.schema_version == 1
    assert reopened.operation_digest == LEGACY_V1_DIGEST
    assert reopened.reconcile(binding) is StartupParticipantOutcome.ROLLED_FORWARD
    committed = reopened.memory.get_committed_episodic(LEGACY_V1_EPISODE_ID)
    assert committed is not None
    assert committed.record.coordination_schema == 1
    assert committed.record.context_id is None


def test_v1_committed_only_reconstruction_preserves_legacy_operation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    participant = _participant(memory, schema_version=1)
    binding = _binding(participant)

    participant.prepare(binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    reconstructed = MemoryEpisodicParticipant.from_pending(
        DualMemorySystem(settings),
        TRANSACTION_ID,
        MEMORY_EPISODIC_PARTICIPANT_ID,
        LEGACY_V1_DIGEST,
    )

    assert reconstructed.operation.schema_version == 1
    assert reconstructed.operation.canonical_dict() == LEGACY_V1_OPERATION
    assert reconstructed.inspect_reconciliation(binding) is StartupParticipantOutcome.VERIFIED_CONSISTENT


def test_v2_pending_round_trip_and_finalize_persist_frozen_provenance(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    participant = _participant(
        memory,
        context_id="context-a",
        source_channel="api.chat",
        source_session_id="session-a",
    )
    binding = _binding(participant)
    participant.prepare(binding)
    pending = json.loads(participant.pending_path(binding).read_text())

    assert pending["schema_version"] == 1
    assert pending["operation"] == {
        **LEGACY_V1_OPERATION,
        "schema_version": 2,
        "context_id": "context-a",
        "source_channel": "api.chat",
        "source_session_id": "session-a",
    }
    assert memory.get_episodic_record(participant.episode_id(TRANSACTION_ID)) is None

    reopened = MemoryEpisodicParticipant.from_pending(
        DualMemorySystem(settings),
        TRANSACTION_ID,
        MEMORY_EPISODIC_PARTICIPANT_ID,
        participant.operation_digest,
    )
    assert reopened.operation == participant.operation
    assert reopened.finalize(binding) is ParticipantOutcome.FINALIZED
    assert not reopened.pending_path(binding).exists()
    committed = reopened.memory.get_committed_episodic(
        participant.episode_id(TRANSACTION_ID)
    )
    assert committed is not None
    assert committed.record.coordination_schema == 2
    assert committed.record.context_id == "context-a"
    assert committed.record.source_channel == "api.chat"
    assert committed.record.source_session_id == "session-a"
    assert committed.metadata["coordination_schema"] == 2
    assert committed.metadata["context_id"] == "context-a"


def test_v2_pending_without_context_is_explicitly_all_none(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    participant = _participant(memory)

    assert participant.operation.schema_version == 2
    assert participant.operation.canonical_dict()["context_id"] is None
    assert participant.operation.canonical_dict()["source_channel"] is None
    assert participant.operation.canonical_dict()["source_session_id"] is None


def test_pending_provenance_tamper_diverges_without_publication(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    participant = _participant(
        memory, context_id="context-a", source_channel="chat", source_session_id="session-a"
    )
    binding = _binding(participant)
    participant.prepare(binding)
    payload = json.loads(participant.pending_path(binding).read_text())
    payload["operation"]["context_id"] = "context-b"
    participant.pending_path(binding).write_text(json.dumps(payload))

    with pytest.raises(ParticipantDivergedError):
        MemoryEpisodicParticipant.from_pending(
            DualMemorySystem(settings),
            TRANSACTION_ID,
            MEMORY_EPISODIC_PARTICIPANT_ID,
            participant.operation_digest,
        )
    assert memory.get_episodic_record(participant.episode_id(TRANSACTION_ID)) is None


def test_committed_provenance_tamper_diverges_from_journal_binding(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    participant = _participant(
        memory, context_id="context-a", source_channel="chat", source_session_id="session-a"
    )
    binding = _binding(participant)
    participant.prepare(binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    episode_id = participant.episode_id(TRANSACTION_ID)
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    metadata = dict(stored["metadatas"][0])
    metadata["context_id"] = "context-b"
    memory.db1.update(ids=[episode_id], metadatas=[metadata])

    with pytest.raises(ParticipantDivergedError):
        MemoryEpisodicParticipant.from_pending(
            memory, TRANSACTION_ID, MEMORY_EPISODIC_PARTICIPANT_ID, participant.operation_digest
        )


def test_malformed_provenance_fails_exact_read(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    memory.publish_coordinated_episodic(
        "episode-malformed",
        "input",
        "response",
        loss=0.1,
        emotion_valence=0.2,
        emotion_arousal=0.3,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at=NOW.isoformat(),
        coordination_schema=2,
        context_id="context-a",
        source_channel="chat",
    )
    metadata = dict(
        memory.db1.get(ids=["episode-malformed"], include=["metadatas"])["metadatas"][0]
    )
    metadata["context_id"] = "bad/id"
    memory.db1.update(ids=["episode-malformed"], metadatas=[metadata])

    with pytest.raises(EpisodicMemoryFormatError):
        memory.get_committed_episodic("episode-malformed")


def test_legacy_exact_read_has_no_fabricated_provenance(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    episode_id = memory.save_episodic("legacy input", "legacy response")

    record = memory.get_committed_episodic(episode_id)

    assert record is not None
    assert record.record.coordination_schema is None
    assert record.record.context_id is None
    assert record.record.source_channel is None
    assert record.record.source_session_id is None


def test_semantic_and_working_memory_durable_shapes_keep_only_allowed_context_field() -> None:
    from kagya.memory import SemanticMemoryRecord

    semantic_fields = tuple(field.name for field in fields(SemanticMemoryRecord))
    assert semantic_fields[-1] == "context_id"
    assert semantic_fields[:-1] == (
        "id",
        "text",
        "source_episode_ids",
        "record_type",
        "created_at",
        "metadata",
    )
    assert SemanticMemoryRecord("semantic-id", "text").context_id is None
    assert "context_id" not in {field.name for field in fields(WorkingMemoryItem)}
    assert "context_id" not in WorkingMemoryItemSnapshot.model_fields
    assert "context_id" not in WorkingMemorySnapshot.model_fields
