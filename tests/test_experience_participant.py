"""R07 participant protocol tests for durable Experience evidence."""

from datetime import UTC, datetime
from dataclasses import replace
import os
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.experience import (
    ExperienceAppraisalEvidence,
    ExperienceAppraisalReasonCode,
    ExperienceEmotionContributions,
    ExperienceEmotionProjection,
    ExperienceEmotionUpdateReasonCode,
    ExperienceLifecycle,
    ExperienceMeasurementEvidence,
    ExperienceRecord,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
    experience_record_digest,
)
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.memory.episodic_participant import (
    EpisodicWrite,
    MEMORY_EPISODIC_PARTICIPANT_ID,
    MemoryEpisodicParticipant,
)
from kagya.memory.experience_participant import (
    ExperienceCreateIntent,
    ExperienceRevisionIntent,
    MEMORY_EXPERIENCE_PARTICIPANT_ID,
    MemoryExperienceParticipant,
    experience_id_for_event,
)
from kagya.memory.experience_store import ExperienceStore, ExperienceStoreCorrupt
from kagya.runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    ParticipantOutcome,
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    TransactionCoordinator,
    TransactionKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
EVENT_ID = "11111111-1111-4111-8111-111111111111"
REVISION_EVENT_ID = "22222222-2222-4222-8222-222222222222"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "experience_db1",
                    "db2_collection": "experience_db2",
                }
            )
        }
    )


def _setup(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    store = ExperienceStore(tmp_path / "experience")
    event = AgentEvent(
        EVENT_ID, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, 1
    )
    transaction_id = TransactionCoordinator.derive_transaction_id(
        event, TransactionKind.EVENT_MUTATION
    )
    episode = MemoryEpisodicParticipant(
        memory,
        EpisodicWrite(
            user_input="visible input",
            response="visible response",
            loss=0.2,
            emotion_valence=0.3,
            emotion_arousal=0.4,
            record_type=MemoryRecordType.EPISODIC_LOG,
            created_at=NOW.isoformat(),
            context_id="context:1",
            source_channel="chat",
            source_session_id="session:1",
            schema_version=3,
        ),
    )
    episode_binding = TransactionBinding(
        transaction_id,
        EVENT_ID,
        1,
        MEMORY_EPISODIC_PARTICIPANT_ID,
        episode.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    episode.prepare(episode_binding)
    measurement = ExperienceMeasurementEvidence(
        "model." + "a" * 64, True, calibrated_novelty=0.25
    )
    pre = ExperienceEmotionProjection(0.0, 0.0)
    post = ExperienceEmotionProjection(0.5, 0.25)
    genesis = ExperienceRevisionRecord(
        experience_id_for_event(EVENT_ID, 1),
        0,
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        NOW,
        EVENT_ID,
        1,
        evidence_refs=(EVENT_ID,),
    )
    record = ExperienceRecord(
        experience_id=experience_id_for_event(EVENT_ID, 1),
        revision=0,
        lifecycle=ExperienceLifecycle.ACTIVE,
        source_event_id=EVENT_ID,
        source_event_sequence=1,
        source_episode_id=episode.episode_id(transaction_id),
        context_id="context:1",
        measurement=measurement,
        appraisal=ExperienceAppraisalEvidence(
            novelty=0.25,
            novelty_valid=True,
            reason_codes=(ExperienceAppraisalReasonCode.NOVELTY_MEASURED,),
        ),
        pre_appraisal_emotion=pre,
        temporal_update_reasons=(ExperienceEmotionUpdateReasonCode.TIMELINE_INITIALIZED,),
        post_appraisal_emotion=post,
        emotion_contributions=ExperienceEmotionContributions(),
        emotion_update_reasons=(ExperienceEmotionUpdateReasonCode.APPRAISAL_APPLIED,),
        subjective_salience=0.25,
        created_at=NOW,
        revision_history=(genesis,),
    )
    participant = MemoryExperienceParticipant(
        memory,
        store,
        ExperienceCreateIntent(record, episode.operation_digest),
    )
    binding = TransactionBinding(
        transaction_id,
        EVENT_ID,
        1,
        MEMORY_EXPERIENCE_PARTICIPANT_ID,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    return memory, store, episode, episode_binding, participant, binding


def test_prepare_is_pending_only_and_finalize_requires_committed_episode(
    tmp_path: Path,
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)

    participant.prepare(binding)
    assert store.load_current(participant.operation.record.experience_id) is None
    assert store.load_pending(binding.transaction_id) is not None
    assert store.load_receipt(binding.transaction_id) is None
    with pytest.raises(ParticipantUnavailableError):
        participant.finalize(binding)

    episode.finalize(episode_binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    current = store.load_current(participant.operation.record.experience_id)
    assert current is not None
    assert current.record.source_episode_id == episode.episode_id(binding.transaction_id)
    receipt = store.load_receipt(binding.transaction_id)
    assert receipt is not None
    assert receipt["operation_digest"] == participant.operation_digest
    assert "user_input" not in str(receipt)
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert store.load_pending(binding.transaction_id) is None


def test_receipt_survives_crash_before_pending_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)

    def crash_before_pending_removal(_transaction_id: str) -> None:
        raise ParticipantUnavailableError("injected crash before pending removal")

    monkeypatch.setattr(participant, "_remove_pending", crash_before_pending_removal)
    with pytest.raises(ParticipantUnavailableError):
        participant.finalize(binding)
    monkeypatch.undo()

    assert store.load_current(participant.operation.record.experience_id) is not None
    assert store.load_receipt(binding.transaction_id) is not None
    assert store.load_pending(binding.transaction_id) is not None
    restarted = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=EVENT_ID,
        processing_sequence=1,
    )
    assert restarted.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert store.load_pending(binding.transaction_id) is None


@pytest.mark.parametrize("fault_stage", ["temp_fsync", "link"])
def test_first_create_publication_crash_restarts_from_pending_to_exact_one_revision(
    tmp_path: Path, fault_stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)
    record_directory = store.record_path(
        participant.operation.record.experience_id, 0
    ).parent
    original_fsync = os.fsync
    original_unlink = os.unlink
    fault_injected = False

    def fail_temp_fsync(descriptor: int) -> None:
        nonlocal fault_injected
        descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if (
            fault_stage == "temp_fsync"
            and ".publish-" in descriptor_path.name
            and not fault_injected
        ):
            fault_injected = True
            raise OSError("injected crash while syncing first revision temp")
        original_fsync(descriptor)

    def fail_before_link(
        source: str,
        destination: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal fault_injected
        temporary = record_directory / source
        assert temporary.exists()
        fault_injected = True
        raise OSError("injected crash before first revision link")

    def preserve_temporary(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if isinstance(path, str) and ".publish-" in path:
            raise OSError("injected crash before temporary cleanup")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_temp_fsync)
    if fault_stage == "link":
        monkeypatch.setattr(os, "link", fail_before_link)
    monkeypatch.setattr(os, "unlink", preserve_temporary)
    with pytest.raises(ParticipantUnavailableError):
        participant.finalize(binding)
    assert fault_injected
    assert not store.record_path(participant.operation.record.experience_id, 0).exists()
    assert any(".publish-" in item.name for item in record_directory.iterdir())
    monkeypatch.undo()

    restarted = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=EVENT_ID,
        processing_sequence=1,
    )
    assert restarted.finalize(binding) is ParticipantOutcome.FINALIZED
    current = store.load_current(participant.operation.record.experience_id)
    assert current is not None
    assert current.record == participant.operation.record
    assert current.record.revision == 0
    assert not store.load_pending(binding.transaction_id)
    assert not any(".publish-" in item.name for item in record_directory.iterdir())


def test_pending_create_allows_only_typed_empty_record_directory(
    tmp_path: Path,
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)
    record_directory = store.record_path(
        participant.operation.record.experience_id, 0
    ).parent
    record_directory.mkdir(parents=True, mode=0o700)

    with pytest.raises(ExperienceStoreCorrupt):
        store.load_current(participant.operation.record.experience_id)

    restarted = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=EVENT_ID,
        processing_sequence=1,
    )
    assert restarted.finalize(binding) is ParticipantOutcome.FINALIZED
    assert store.record_path(participant.operation.record.experience_id, 0).exists()


def test_receipt_without_committed_record_fails_closed(
    tmp_path: Path,
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)
    participant.finalize(binding)
    record_directory = store.record_path(
        participant.operation.record.experience_id, 0
    ).parent
    store.record_path(participant.operation.record.experience_id, 0).unlink()
    record_directory.rmdir()

    with pytest.raises(ParticipantDivergedError):
        MemoryExperienceParticipant.from_pending(
            memory,
            store,
            binding.transaction_id,
            binding.participant_id,
            binding.operation_digest,
            event_id=EVENT_ID,
            processing_sequence=1,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_event_id", "33333333-3333-4333-8333-333333333333"),
        ("source_event_sequence", 99),
        ("source_episode_id", "episode:substituted"),
        ("context_id", "context:substituted"),
        ("created_at", datetime(2026, 1, 2, tzinfo=UTC)),
        ("source_episode_operation_digest", "9" * 64),
    ],
)
def test_revision_lineage_substitution_fails_closed(
    tmp_path: Path, field: str, replacement: object
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)
    participant.finalize(binding)
    initial = participant.operation.record
    revision_record = ExperienceRevisionRecord(
        initial.experience_id,
        1,
        ExperienceRevisionOperation.CORRECT,
        ExperienceRevisionReason.CORRECTION,
        NOW,
        REVISION_EVENT_ID,
        2,
        evidence_refs=("evidence:1",),
        previous_revision_digest=initial.revision_history[-1].record_digest,
    )
    revised = replace(
        initial,
        revision=1,
        revision_history=(initial.revision_history[-1], revision_record),
        **({field: replacement} if field != "source_episode_operation_digest" else {}),
    )
    source_digest = (
        replacement
        if field == "source_episode_operation_digest"
        else participant.operation.source_episode_operation_digest
    )
    assert isinstance(source_digest, str)
    revision = MemoryExperienceParticipant(
        memory,
        store,
        ExperienceRevisionIntent(
            revised,
            revision_record,
            expected_revision=0,
            expected_record_digest=experience_record_digest(initial),
            source_episode_operation_digest=source_digest,
        ),
    )
    revision_transaction_id = TransactionCoordinator.derive_transaction_id(
        AgentEvent(
            REVISION_EVENT_ID,
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            NOW,
            2,
        ),
        TransactionKind.EVENT_MUTATION,
    )
    revision_binding = TransactionBinding(
        revision_transaction_id,
        REVISION_EVENT_ID,
        2,
        MEMORY_EXPERIENCE_PARTICIPANT_ID,
        revision.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )

    with pytest.raises(ParticipantDivergedError, match="lineage"):
        revision.prepare(revision_binding)
    assert store.load_pending(revision_transaction_id) is None


def test_abort_removes_only_pending_and_restart_reuses_committed_identity(
    tmp_path: Path,
) -> None:
    memory, store, _episode, _episode_binding, participant, binding = _setup(tmp_path)

    participant.prepare(binding)
    assert (
        MemoryExperienceParticipant.abort_pending(memory, store, binding).value
        == "aborted"
    )
    assert store.load_current(participant.operation.record.experience_id) is None
    assert store.load_receipt(binding.transaction_id) is None
    assert (
        MemoryExperienceParticipant.abort_pending(memory, store, binding).value
        == "already_absent"
    )

    participant.prepare(binding)
    # A committed source is required before the Experience can be published.
    episode, episode_binding = _episode, _episode_binding
    episode.finalize(episode_binding)
    participant.finalize(binding)
    rebuilt = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=EVENT_ID,
        processing_sequence=1,
    )
    assert rebuilt.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT


def test_revision_prepare_writes_pending_and_restart_rolls_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)
    participant.finalize(binding)
    initial = participant.operation.record
    genesis = ExperienceRevisionRecord(
        initial.experience_id,
        0,
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        NOW,
        EVENT_ID,
        1,
        evidence_refs=(EVENT_ID,),
    )
    revision_record = ExperienceRevisionRecord(
        initial.experience_id,
        1,
        ExperienceRevisionOperation.CORRECT,
        ExperienceRevisionReason.CORRECTION,
        NOW,
        REVISION_EVENT_ID,
        2,
        evidence_refs=("evidence:1",),
        previous_revision_digest=genesis.record_digest,
    )
    revised = replace(
        initial,
        revision=1,
        revision_history=(genesis, revision_record),
    )
    with pytest.raises(ValueError):
        ExperienceRevisionIntent(
            replace(revised, lifecycle=ExperienceLifecycle.RETRACTED),
            revision_record,
            expected_revision=0,
            expected_record_digest=experience_record_digest(initial),
            source_episode_operation_digest=participant.operation.source_episode_operation_digest,
        )
    revision = MemoryExperienceParticipant(
        memory,
        store,
        ExperienceRevisionIntent(
            revised,
            revision_record,
            expected_revision=0,
            expected_record_digest=experience_record_digest(initial),
            source_episode_operation_digest=participant.operation.source_episode_operation_digest,
        ),
    )
    revision_transaction_id = TransactionCoordinator.derive_transaction_id(
        AgentEvent(
            REVISION_EVENT_ID,
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            NOW,
            2,
        ),
        TransactionKind.EVENT_MUTATION,
    )
    revision_binding = TransactionBinding(
        revision_transaction_id,
        REVISION_EVENT_ID,
        2,
        MEMORY_EXPERIENCE_PARTICIPANT_ID,
        revision.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )

    original_write_pending = store.write_pending

    def write_then_crash(transaction_id: str, payload: dict[str, object]) -> None:
        original_write_pending(transaction_id, payload)
        raise RuntimeError("injected crash after revision pending publication")

    monkeypatch.setattr(store, "write_pending", write_then_crash)
    with pytest.raises(RuntimeError):
        revision.prepare(revision_binding)
    monkeypatch.undo()
    assert store.load_pending(revision_transaction_id) is not None
    assert store.load_receipt(revision_transaction_id) is None
    assert (
        MemoryExperienceParticipant.abort_pending(memory, store, revision_binding).value
        == "aborted"
    )
    assert store.load_pending(revision_transaction_id) is None
    revision.prepare(revision_binding)
    rebuilt = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        revision_transaction_id,
        revision_binding.participant_id,
        revision_binding.operation_digest,
        event_id=REVISION_EVENT_ID,
        processing_sequence=2,
    )
    assert rebuilt.abort(revision_binding).value == "aborted"
    assert rebuilt.abort(revision_binding).value == "already_absent"
    revision.prepare(revision_binding)
    assert isinstance(revision.operation, ExperienceRevisionIntent)
    store.publish_revision(
        revision.operation.record,
        revision.operation_digest,
        revision.operation.source_episode_operation_digest,
        expected_revision=revision.operation.expected_revision,
        expected_digest=revision.operation.expected_record_digest,
    )
    assert store.load_receipt(revision_transaction_id) is None
    rebuilt = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        revision_transaction_id,
        revision_binding.participant_id,
        revision_binding.operation_digest,
        event_id=REVISION_EVENT_ID,
        processing_sequence=2,
    )
    assert rebuilt.finalize(revision_binding) is ParticipantOutcome.ALREADY_CONSISTENT
    current = store.load_current(initial.experience_id)
    assert current is not None
    assert current.record == revised
    receipt = store.load_receipt(revision_transaction_id)
    assert receipt is not None
    assert receipt["operation_digest"] == revision.operation_digest
    store.write_pending(revision_transaction_id, revision._artifact(revision_binding))
    pending_receipt_restart = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        revision_transaction_id,
        revision_binding.participant_id,
        revision_binding.operation_digest,
        event_id=REVISION_EVENT_ID,
        processing_sequence=2,
    )
    assert pending_receipt_restart.finalize(revision_binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert store.load_pending(revision_transaction_id) is None
    restarted_after_commit = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        revision_transaction_id,
        revision_binding.participant_id,
        revision_binding.operation_digest,
        event_id=REVISION_EVENT_ID,
        processing_sequence=2,
    )
    assert restarted_after_commit.finalize(revision_binding) is ParticipantOutcome.ALREADY_CONSISTENT
    revision_two_record = ExperienceRevisionRecord(
        initial.experience_id,
        2,
        ExperienceRevisionOperation.REASSESS,
        ExperienceRevisionReason.REASSESSMENT,
        NOW,
        "event:3",
        3,
        evidence_refs=("event:3",),
        previous_revision_digest=revision_record.record_digest,
    )
    revised_twice = replace(
        revised,
        revision=2,
        revision_history=(genesis, revision_record, revision_two_record),
    )
    store.publish_revision(
        revised_twice,
        "4" * 64,
        participant.operation.source_episode_operation_digest,
        expected_revision=1,
        expected_digest=experience_record_digest(revised),
    )
    historical = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        revision_transaction_id,
        revision_binding.participant_id,
        revision_binding.operation_digest,
        event_id=REVISION_EVENT_ID,
        processing_sequence=2,
    )
    assert historical.finalize(revision_binding) is ParticipantOutcome.ALREADY_CONSISTENT
    historical_create = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        binding.transaction_id,
        binding.participant_id,
        binding.operation_digest,
        event_id=EVENT_ID,
        processing_sequence=1,
    )
    assert historical_create.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    with pytest.raises(ParticipantDivergedError):
        rebuilt.abort(revision_binding)
