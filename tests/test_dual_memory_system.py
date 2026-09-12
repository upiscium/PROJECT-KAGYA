import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.memory.dual_memory_system import EpisodicMemoryFormatError
from kagya.memory.episodic_participant import (
    MEMORY_EPISODIC_PARTICIPANT_ID,
    EpisodicWrite,
    MemoryEpisodicParticipant,
)
from kagya.models import DummyProvider
from kagya.runtime import (
    AbortOutcome,
    ParticipantDivergedError,
    ParticipantOutcome,
    ParticipantUnavailableError,
    StartupParticipantOutcome,
    TransactionBinding,
    TransactionKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"
TRANSACTION_ID = "f51090e6-25a3-5d8e-b701-cbbdb8e88dca"


def test_saving_episodic_record_returns_episode_id(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))

    episode_id = memory.save_episodic("hello", "world")

    assert episode_id.startswith("episode-")


def test_saved_episodic_records_can_be_retrieved_without_private_fields(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic(
        "I like lunar gardens",
        "Remembering lunar gardens.",
        loss=0.25,
        emotion_valence=0.7,
        emotion_arousal=0.8,
    )

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db1_results] == [episode_id]
    assert context.db1_results[0].record_type == MemoryRecordType.EPISODIC_LOG
    assert not hasattr(context.db1_results[0], "hidden_thought")


def test_new_memory_metadata_rejects_private_fields(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))

    with pytest.raises(ValueError, match="cannot contain private model fields"):
        memory.save_episodic(
            "input",
            "output",
            metadata={"nested": {"raw-prompt": PRIVATE_SENTINEL}},
        )
    with pytest.raises(ValueError, match="cannot contain private model fields"):
        memory.save_semantic(
            "visible fact",
            metadata={"hidden_thought": PRIVATE_SENTINEL},
        )


def test_legacy_episodic_private_data_is_scrubbed_on_reopen(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    legacy_id = f"episode-{TRANSACTION_ID}"
    memory.db1.add(
        ids=[legacy_id],
        documents=[
            f"User: legacy user\nAssistant: visible answer\nThought: {PRIVATE_SENTINEL}"
        ],
        metadatas=[
            {
                "user_input": "legacy user",
                "response": "visible answer",
                "hidden_thought": PRIVATE_SENTINEL,
                "loss": 0.1,
                "emotion_valence": 0.2,
                "emotion_arousal": 0.3,
                "record_type": "episodic_log",
                "archived": False,
                "created_at": "legacy",
                "extra": json.dumps(
                    {"safe": "kept", "chain_of_thought": PRIVATE_SENTINEL}
                ),
            }
        ],
    )

    reopened = DualMemorySystem(settings)
    stored = reopened.db1.get(
        ids=[legacy_id], include=["documents", "metadatas"]
    )

    assert stored["documents"] == ["User: legacy user\nAssistant: visible answer"]
    assert "hidden_thought" not in stored["metadatas"][0]
    assert PRIVATE_SENTINEL not in str(stored)
    assert json.loads(stored["metadatas"][0]["extra"]) == {"safe": "kept"}


def test_semantic_records_can_be_retrieved_from_db2(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("The user likes lunar gardens.")

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db2_results] == [semantic_id]
    assert context.db2_results[0].record_type == MemoryRecordType.SEMANTIC_MEMORY


def test_consolidation_archives_db1_records_instead_of_deleting(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("fact", "response", emotion_arousal=1.0)

    semantic_ids = memory.consolidate_to_semantic(DummyProvider())

    assert len(semantic_ids) == 1
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    assert stored["ids"] == [episode_id]
    assert stored["metadatas"][0]["archived"] is True
    assert memory.retrieve_context("fact").db1_results == []


def test_retrieval_respects_configured_db1_and_db2_top_k(tmp_path: Path) -> None:
    memory = DualMemorySystem(
        _settings_for_tmp_memory(tmp_path, db1_top_k=2, db2_top_k=1)
    )
    for index in range(3):
        memory.save_episodic(f"shared topic episode {index}", "response")
        memory.save_semantic(f"shared topic semantic {index}")

    context = memory.retrieve_context("shared topic")

    assert len(context.db1_results) == 2
    assert len(context.db2_results) == 1


def test_coordinated_episode_id_is_deterministic_and_pending_survives_reopen(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    participant = _participant(DualMemorySystem(settings))
    binding = _binding(participant)

    participant.prepare(binding)
    reopened = MemoryEpisodicParticipant.from_pending(
        DualMemorySystem(settings),
        TRANSACTION_ID,
        MEMORY_EPISODIC_PARTICIPANT_ID,
        participant.operation_digest,
    )

    assert reopened.episode_id(TRANSACTION_ID) == participant.episode_id(TRANSACTION_ID)
    assert reopened.pending_path(binding).exists()
    assert reopened.pending_path(binding).stat().st_mode & 0o777 == 0o600
    assert reopened.reconcile(binding) is StartupParticipantOutcome.ROLLED_FORWARD
    assert reopened.memory.get_episodic_record(
        reopened.episode_id(TRANSACTION_ID)
    ) is not None


def test_pending_episodic_is_invisible_to_retrieval_and_consolidation(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    participant = _participant(memory, arousal=1.0)
    binding = _binding(participant)

    participant.prepare(binding)

    assert memory.get_episodic_record(participant.episode_id(TRANSACTION_ID)) is None
    assert memory.retrieve_context("staged user").db1_results == []
    assert memory._get_unarchived_episodic_records() == []
    assert memory.consolidate_to_semantic(DummyProvider()) == []


def test_finalize_is_idempotent_when_committed_record_and_pending_both_exist(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    participant = _participant(memory)
    binding = _binding(participant)
    participant.prepare(binding)
    episode_id = participant.episode_id(TRANSACTION_ID)
    operation = participant.operation
    memory.publish_coordinated_episodic(
        episode_id,
        operation.user_input,
        operation.response,
        loss=operation.loss,
        emotion_valence=operation.emotion_valence,
        emotion_arousal=operation.emotion_arousal,
        record_type=operation.record_type,
        created_at=operation.created_at,
    )

    reopened = MemoryEpisodicParticipant.from_pending(
        DualMemorySystem(settings),
        TRANSACTION_ID,
        MEMORY_EPISODIC_PARTICIPANT_ID,
        participant.operation_digest,
    )
    reconciliation = reopened.reconcile(binding)
    outcome = reopened.finalize(binding)

    assert reconciliation is StartupParticipantOutcome.VERIFIED_CONSISTENT
    assert outcome is ParticipantOutcome.ALREADY_CONSISTENT
    assert not reopened.pending_path(binding).exists()
    assert memory.db1.get(ids=[episode_id])["ids"] == [episode_id]


def test_committed_only_reconstruction_rejects_conflicting_document(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    participant = _participant(memory)
    binding = _binding(participant)
    participant.prepare(binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED

    reconstructed = MemoryEpisodicParticipant.from_pending(
        memory,
        binding.transaction_id,
        participant.participant_id,
        participant.operation_digest,
    )

    assert (
        reconstructed.inspect_reconciliation(binding)
        is StartupParticipantOutcome.VERIFIED_CONSISTENT
    )
    assert reconstructed.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    episode_id = participant.episode_id(binding.transaction_id)
    memory.db1.update(ids=[episode_id], documents=["conflicting DB1 document"])

    with pytest.raises(ParticipantDivergedError):
        MemoryEpisodicParticipant.from_pending(
            memory,
            binding.transaction_id,
            participant.participant_id,
            participant.operation_digest,
        )
    assert memory.db1.get(ids=[episode_id], include=["documents"])["documents"] == [
        "conflicting DB1 document"
    ]
    with pytest.raises(EpisodicMemoryFormatError):
        DualMemorySystem(memory.settings)


def test_reconciliation_inspection_rejects_conflicting_committed_document(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    participant = _participant(memory)
    binding = _binding(participant)
    participant.prepare(binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    episode_id = participant.episode_id(binding.transaction_id)
    memory.db1.update(ids=[episode_id], documents=["conflicting DB1 document"])

    with pytest.raises(ParticipantDivergedError):
        participant.inspect_reconciliation(binding)
    assert memory.db1.get(ids=[episode_id], include=["documents"])["documents"] == [
        "conflicting DB1 document"
    ]


def test_finalize_rejects_conflicting_committed_document_with_matching_pending(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    participant = _participant(memory)
    binding = _binding(participant)
    participant.prepare(binding)
    _publish_participant(memory, participant, binding)
    episode_id = participant.episode_id(binding.transaction_id)
    memory.db1.update(ids=[episode_id], documents=["conflicting DB1 document"])

    with pytest.raises(ParticipantDivergedError):
        participant.finalize(binding)

    assert participant.pending_path(binding).exists()


def test_valid_committed_document_and_metadata_are_consistent(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    participant = _participant(memory)
    binding = _binding(participant)
    participant.prepare(binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED

    reconstructed = MemoryEpisodicParticipant.from_pending(
        memory,
        binding.transaction_id,
        participant.participant_id,
        participant.operation_digest,
    )

    assert (
        reconstructed.inspect_reconciliation(binding)
        is StartupParticipantOutcome.VERIFIED_CONSISTENT
    )
    assert reconstructed.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT


def test_all_committed_verification_paths_reject_raw_metadata_conflict(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    participant = _participant(memory)
    binding = _binding(participant)
    participant.prepare(binding)
    _publish_participant(memory, participant, binding)
    episode_id = participant.episode_id(binding.transaction_id)
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    conflicting = dict(stored["metadatas"][0] or {})
    conflicting["unexpected"] = "conflict"
    memory.db1.update(ids=[episode_id], metadatas=[conflicting])

    for operation in (
        lambda: participant.prepare(binding),
        lambda: participant.finalize(binding),
        lambda: participant.inspect_reconciliation(binding),
        lambda: participant.reconcile(binding),
        lambda: MemoryEpisodicParticipant.from_pending(
            memory,
            binding.transaction_id,
            participant.participant_id,
            participant.operation_digest,
        ),
    ):
        with pytest.raises(ParticipantDivergedError):
            operation()

    assert participant.pending_path(binding).exists()

def test_abort_removes_only_pending_and_never_committed_memory(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    pending = _participant(memory)
    pending_binding = _binding(pending)
    pending.prepare(pending_binding)

    assert pending.abort(pending_binding) is AbortOutcome.ABORTED
    assert pending.abort(pending_binding) is AbortOutcome.ALREADY_ABSENT

    committed = _participant(memory, user_input="committed")
    committed_binding = _binding(committed)
    committed.prepare(committed_binding)
    committed_pending = committed.pending_path(committed_binding)
    pending_bytes = committed_pending.read_bytes()
    assert committed.finalize(committed_binding) is ParticipantOutcome.FINALIZED
    committed_pending.write_bytes(pending_bytes)
    committed_pending.chmod(0o600)
    episode_id = committed.episode_id(committed_binding.transaction_id)
    reconstructed = MemoryEpisodicParticipant.from_pending(
        DualMemorySystem(memory.settings),
        committed_binding.transaction_id,
        committed.participant_id,
        committed.operation_digest,
    )

    with pytest.raises(ParticipantDivergedError):
        committed.abort(committed_binding)
    assert committed_pending.exists()
    assert memory.get_episodic_record(episode_id) is not None
    assert (
        reconstructed.inspect_reconciliation(committed_binding)
        is StartupParticipantOutcome.VERIFIED_CONSISTENT
    )


def test_memory_staging_rejects_symlink_directory(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    staging = memory.settings.memory.persist_directory / ".r07-episodic-pending"
    staging.symlink_to(attacker, target_is_directory=True)
    participant = _participant(memory, user_input=PRIVATE_SENTINEL)

    with pytest.raises(ParticipantUnavailableError):
        participant.prepare(_binding(participant))

    assert list(attacker.iterdir()) == []


def test_memory_staging_rejects_symlink_ancestor(tmp_path: Path) -> None:
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(attacker, target_is_directory=True)
    settings = _settings_for_tmp_memory(tmp_path).model_copy(
        update={
            "memory": _settings_for_tmp_memory(tmp_path).memory.model_copy(
                update={"persist_directory": linked_parent / "chroma"}
            )
        }
    )
    participant = _participant(
        DualMemorySystem(settings), user_input=PRIVATE_SENTINEL
    )

    with pytest.raises(ParticipantUnavailableError):
        participant.prepare(_binding(participant))

    assert not (attacker / "chroma" / ".r07-episodic-pending").exists()


def _participant(
    memory: DualMemorySystem,
    *,
    user_input: str = "staged user",
    arousal: float = 0.4,
) -> MemoryEpisodicParticipant:
    return MemoryEpisodicParticipant(
        memory,
        EpisodicWrite(
            user_input=user_input,
            response="visible response",
            loss=0.2,
            emotion_valence=0.3,
            emotion_arousal=arousal,
            record_type=MemoryRecordType.EPISODIC_LOG,
            created_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        ),
    )


def _publish_participant(
    memory: DualMemorySystem,
    participant: MemoryEpisodicParticipant,
    binding: TransactionBinding,
) -> None:
    operation = participant.operation
    memory.publish_coordinated_episodic(
        participant.episode_id(binding.transaction_id),
        operation.user_input,
        operation.response,
        loss=operation.loss,
        emotion_valence=operation.emotion_valence,
        emotion_arousal=operation.emotion_arousal,
        record_type=operation.record_type,
        created_at=operation.created_at,
    )


def _binding(
    participant: MemoryEpisodicParticipant,
    *,
    transaction_id: str = TRANSACTION_ID,
) -> TransactionBinding:
    return TransactionBinding(
        transaction_id=transaction_id,
        event_id="5cefdcd0-88a3-5850-b6cc-72cab6f9989e",
        processing_sequence=1,
        participant_id=participant.participant_id,
        operation_digest=participant.operation_digest,
        transaction_kind=TransactionKind.EVENT_MUTATION,
    )


def _settings_for_tmp_memory(
    tmp_path: Path,
    *,
    db1_top_k: int = 5,
    db2_top_k: int = 5,
) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_test",
                    "db2_collection": "cortex_test",
                    "db1_top_k": db1_top_k,
                    "db2_top_k": db2_top_k,
                }
            )
        }
    )
