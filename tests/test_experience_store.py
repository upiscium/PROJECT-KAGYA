"""Durability and privacy tests for the Memory-owned Experience store."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

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
from kagya.memory.experience_store import (
    ExperienceStore,
    ExperienceStoreConflict,
    ExperienceStoreCorrupt,
    experience_record_from_dict,
    experience_record_to_dict,
)


MODEL_KEY = "model." + "a" * 64
CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
OPERATION_DIGEST = "1" * 64
SOURCE_DIGEST = "2" * 64


def _record(
    *,
    experience_id: str = "experience:1",
    revision: int = 0,
    history=(),
    anchor: str | None = None,
) -> ExperienceRecord:
    measurement = ExperienceMeasurementEvidence(
        MODEL_KEY, True, calibrated_novelty=0.25
    )
    pre = ExperienceEmotionProjection(0.0, 0.0)
    post = ExperienceEmotionProjection(0.5, 0.25)
    return ExperienceRecord(
        experience_id=experience_id,
        revision=revision,
        lifecycle=ExperienceLifecycle.ACTIVE,
        source_event_id="event:1",
        source_event_sequence=1,
        source_episode_id="episode:1",
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
        created_at=CREATED_AT,
        revision_history=history,
        history_anchor_digest=anchor,
    )


def test_experience_record_round_trip_is_exact_and_reference_first() -> None:
    record = _record()
    restored = experience_record_from_dict(experience_record_to_dict(record))

    assert restored == record
    assert experience_record_digest(restored) == experience_record_digest(record)
    assert b"event:1" in str(experience_record_to_dict(record)).encode()
    assert not any(
        private in str(experience_record_to_dict(record))
        for private in ("prompt", "response", "hidden_thought", "transcript")
    )


def test_store_publishes_immutable_current_and_rejects_conflicts(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "experience")
    record = _record()
    store.publish_create(record, OPERATION_DIGEST, SOURCE_DIGEST)

    current = store.load_current(record.experience_id)
    assert current is not None
    assert current.record == record
    path = store.record_path(record.experience_id, 0)
    before = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700

    store.publish_create(record, OPERATION_DIGEST, SOURCE_DIGEST)
    assert path.read_bytes() == before
    with pytest.raises(ExperienceStoreConflict):
        store.publish_create(record, "3" * 64, SOURCE_DIGEST)


def test_first_directory_creation_syncs_each_parent_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExperienceStore(tmp_path / "experience")
    record = _record()
    synced: list[Path] = []
    original = ExperienceStore._fsync_directory

    def record_sync(path: Path) -> None:
        synced.append(path)
        original(path)

    monkeypatch.setattr(ExperienceStore, "_fsync_directory", staticmethod(record_sync))
    store.publish_create(record, OPERATION_DIGEST, SOURCE_DIGEST)

    record_directory = store.records_root / record.experience_id
    assert {store.root, store.records_root, record_directory}.issubset(synced)


def test_store_rejects_malformed_and_symlink_artifacts(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "experience")
    record = _record()
    store.publish_create(record, OPERATION_DIGEST, SOURCE_DIGEST)
    path = store.record_path(record.experience_id, 0)
    path.unlink()
    path.symlink_to(tmp_path / "secret")

    with pytest.raises(ExperienceStoreCorrupt):
        store.load_current(record.experience_id)


def test_revision_publication_keeps_prior_bytes_and_rejects_stale_target(
    tmp_path: Path,
) -> None:
    store = ExperienceStore(tmp_path / "experience")
    initial = _record()
    store.publish_create(initial, OPERATION_DIGEST, SOURCE_DIGEST)
    initial_path = store.record_path(initial.experience_id, 0)
    initial_bytes = initial_path.read_bytes()
    genesis = ExperienceRevisionRecord(
        initial.experience_id,
        0,
        ExperienceRevisionOperation.REASSESS,
        ExperienceRevisionReason.REASSESSMENT,
        CREATED_AT,
        event_id="event:1",
        event_sequence=1,
        evidence_refs=("evidence:0",),
    )
    revised = _record(revision=1, history=(genesis,))
    store.publish_revision(
        revised,
        "3" * 64,
        SOURCE_DIGEST,
        expected_revision=0,
        expected_digest=experience_record_digest(initial),
    )

    assert initial_path.read_bytes() == initial_bytes
    current = store.load_current(initial.experience_id)
    assert current is not None
    assert current.record == revised
    with pytest.raises(ExperienceStoreConflict):
        store.publish_revision(
            revised,
            "4" * 64,
            SOURCE_DIGEST,
            expected_revision=0,
            expected_digest=experience_record_digest(initial),
        )
