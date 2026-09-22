"""R07 participant protocol tests for durable Experience evidence."""

from datetime import UTC, datetime
from dataclasses import replace
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
from kagya.memory.experience_store import ExperienceStore
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
    with pytest.raises(ParticipantUnavailableError):
        participant.finalize(binding)

    episode.finalize(episode_binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    current = store.load_current(participant.operation.record.experience_id)
    assert current is not None
    assert current.record.source_episode_id == episode.episode_id(binding.transaction_id)
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert store.load_pending(binding.transaction_id) is None


def test_abort_removes_only_pending_and_restart_reuses_committed_identity(
    tmp_path: Path,
) -> None:
    memory, store, _episode, _episode_binding, participant, binding = _setup(tmp_path)

    participant.prepare(binding)
    assert participant.abort(binding).value == "aborted"
    assert store.load_current(participant.operation.record.experience_id) is None
    assert participant.abort(binding).value == "already_absent"

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
    tmp_path: Path,
) -> None:
    memory, store, episode, episode_binding, participant, binding = _setup(tmp_path)
    participant.prepare(binding)
    episode.finalize(episode_binding)
    participant.finalize(binding)
    initial = participant.operation.record
    genesis = ExperienceRevisionRecord(
        initial.experience_id,
        0,
        ExperienceRevisionOperation.REASSESS,
        ExperienceRevisionReason.REASSESSMENT,
        NOW,
        EVENT_ID,
        1,
        evidence_refs=("evidence:0",),
    )
    revised = replace(
        initial,
        revision=1,
        revision_history=(genesis,),
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

    revision.prepare(revision_binding)
    assert store.load_pending(revision_transaction_id) is not None
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
    rebuilt = MemoryExperienceParticipant.from_pending(
        memory,
        store,
        revision_transaction_id,
        revision_binding.participant_id,
        revision_binding.operation_digest,
        event_id=REVISION_EVENT_ID,
        processing_sequence=2,
    )
    assert rebuilt.finalize(revision_binding) is ParticipantOutcome.FINALIZED
    current = store.load_current(initial.experience_id)
    assert current is not None
    assert current.record == revised
    with pytest.raises(ParticipantDivergedError):
        rebuilt.abort(revision_binding)
