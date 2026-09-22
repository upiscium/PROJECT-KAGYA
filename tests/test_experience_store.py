"""Durability and privacy tests for the Memory-owned Experience store."""

from datetime import UTC, datetime
import os
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
    ExperienceStoreUnavailable,
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
    anchor_revision: int | None = None,
) -> ExperienceRecord:
    if not history:
        history = (
            ExperienceRevisionRecord(
                experience_id,
                0,
                ExperienceRevisionOperation.CREATE,
                ExperienceRevisionReason.CREATION,
                CREATED_AT,
                event_id="event:1",
                event_sequence=1,
                evidence_refs=("event:1",),
            ),
        )
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
        history_anchor_revision=anchor_revision,
    )


def _revision_series(experience_id: str, highest: int) -> list[ExperienceRecord]:
    evidence: list[ExperienceRevisionRecord] = []
    records: list[ExperienceRecord] = []
    for revision in range(highest + 1):
        current_evidence = ExperienceRevisionRecord(
            experience_id,
            revision,
            ExperienceRevisionOperation.CREATE
            if revision == 0
            else ExperienceRevisionOperation.CORRECT,
            ExperienceRevisionReason.CREATION
            if revision == 0
            else ExperienceRevisionReason.CORRECTION,
            CREATED_AT,
            event_id=f"event:{revision + 1}",
            event_sequence=revision + 1,
            evidence_refs=(f"event:{revision + 1}",),
            previous_revision_digest=(
                None if revision == 0 else evidence[-1].record_digest
            ),
        )
        evidence.append(current_evidence)
        if revision >= 32:
            anchor_revision = revision - 32
            history = tuple(evidence[anchor_revision + 1 :])
            anchor_digest = evidence[anchor_revision].record_digest
        else:
            anchor_revision = None
            history = tuple(evidence)
            anchor_digest = None
        records.append(
            _record(
                experience_id=experience_id,
                revision=revision,
                history=history,
                anchor=anchor_digest,
                anchor_revision=anchor_revision,
            )
        )
    return records


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
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        CREATED_AT,
        event_id="event:1",
        event_sequence=1,
        evidence_refs=("event:1",),
    )
    revision = ExperienceRevisionRecord(
        initial.experience_id,
        1,
        ExperienceRevisionOperation.CORRECT,
        ExperienceRevisionReason.CORRECTION,
        CREATED_AT,
        event_id="event:2",
        event_sequence=2,
        evidence_refs=("event:2",),
        previous_revision_digest=genesis.record_digest,
    )
    revised = _record(revision=1, history=(genesis, revision))
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


def test_publication_temp_is_typed_and_recovers_without_replacing_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExperienceStore(tmp_path / "experience")
    record = _record()
    failed = False
    original_unlink = os.unlink

    def fail_temp_unlink(path: object, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if not failed and isinstance(path, str) and ".publish-" in path:
            failed = True
            raise OSError("injected publication cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_temp_unlink)
    with pytest.raises(ExperienceStoreUnavailable):
        store.publish_create(record, OPERATION_DIGEST, SOURCE_DIGEST)
    path = store.record_path(record.experience_id, 0)
    final_bytes = path.read_bytes()
    assert any(".publish-" in item.name for item in path.parent.iterdir())
    monkeypatch.undo()

    current = store.load_current(record.experience_id)
    assert current is not None and current.record == record
    assert path.read_bytes() == final_bytes
    assert not any(".publish-" in item.name for item in path.parent.iterdir())
    assert {
        int(item.stem)
        for item in path.parent.iterdir()
        if item.suffix == ".json" and item.stem.isdigit()
    } == {0}

    unknown = path.parent / ".tmp-unknown"
    unknown.write_bytes(b"not a recognized publication artifact")
    unknown.chmod(0o600)
    with pytest.raises(ExperienceStoreCorrupt):
        store.load_current(record.experience_id)


def test_first_create_parent_fsync_failure_recovers_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExperienceStore(tmp_path / "experience")
    record = _record()
    record_directory = store.record_path(record.experience_id, 0).parent
    original_fsync = os.fsync
    record_directory_syncs = 0

    def fail_after_link(descriptor: int) -> None:
        nonlocal record_directory_syncs
        descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if descriptor_path == record_directory:
            record_directory_syncs += 1
            if record_directory_syncs == 2:
                raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_after_link)
    with pytest.raises(ExperienceStoreUnavailable):
        store.publish_create(record, OPERATION_DIGEST, SOURCE_DIGEST)
    monkeypatch.undo()

    assert record_directory_syncs == 2
    current = store.load_current(record.experience_id)
    assert current is not None and current.record == record
    assert {
        int(item.stem)
        for item in record_directory.iterdir()
        if item.suffix == ".json" and item.stem.isdigit()
    } == {0}
    assert not any(".publish-" in item.name for item in record_directory.iterdir())


def test_retention_is_exact_anchored_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExperienceStore(tmp_path / "experience")
    records = _revision_series("experience:retention", 34)
    store.publish_create(records[0], OPERATION_DIGEST, SOURCE_DIGEST)
    for revision in range(1, len(records)):
        if revision == 33:
            original_unlink = Path.unlink
            failed = False

            def fail_first_prune_unlink(path: Path, *args: object, **kwargs: object) -> None:
                nonlocal failed
                if path.name == "0.json" and not failed:
                    failed = True
                    raise OSError("injected compaction failure")
                original_unlink(path, *args, **kwargs)

            monkeypatch.setattr(Path, "unlink", fail_first_prune_unlink)
            with pytest.raises(ExperienceStoreUnavailable):
                store.publish_revision(
                    records[revision],
                    f"{revision + 2:064x}",
                    SOURCE_DIGEST,
                    expected_revision=revision - 1,
                    expected_digest=experience_record_digest(records[revision - 1]),
                )
            monkeypatch.undo()
            assert (store.records_root / "experience:retention" / ".compaction.json").exists()
            assert store.load_current("experience:retention") is not None
        else:
            store.publish_revision(
                records[revision],
                f"{revision + 2:064x}",
                SOURCE_DIGEST,
                expected_revision=revision - 1,
                expected_digest=experience_record_digest(records[revision - 1]),
            )

    current = store.load_current(records[-1].experience_id)
    assert current is not None
    assert current.record == records[-1]
    assert current.record.history_anchor_revision == 2
    assert current.record.revision_history[0].revision == 3
    revisions = {
        int(path.stem)
        for path in (store.records_root / records[-1].experience_id).iterdir()
        if path.suffix == ".json" and path.stem.isdigit()
    }
    assert revisions == set(range(2, 35))
    retained_bytes = store.record_path(records[-1].experience_id, 3).read_bytes()
    store.reconcile_prune(records[-1].experience_id)
    assert store.record_path(records[-1].experience_id, 3).read_bytes() == retained_bytes
    assert {
        int(path.stem)
        for path in (store.records_root / records[-1].experience_id).iterdir()
        if path.suffix == ".json" and path.stem.isdigit()
    } == revisions

    store.record_path(records[-1].experience_id, 10).unlink()
    with pytest.raises(ExperienceStoreCorrupt):
        store.load_current(records[-1].experience_id)


def test_compaction_damaged_anchor_fails_before_prefix_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExperienceStore(tmp_path / "experience")
    records = _revision_series("experience:damaged-anchor", 33)
    store.publish_create(records[0], OPERATION_DIGEST, SOURCE_DIGEST)
    for revision in range(1, len(records)):
        if revision == 33:
            original_unlink = Path.unlink
            failed = False

            def fail_first_prune_unlink(
                path: Path, *args: object, **kwargs: object
            ) -> None:
                nonlocal failed
                if path.name == "0.json" and not failed:
                    failed = True
                    raise OSError("injected compaction failure")
                original_unlink(path, *args, **kwargs)

            monkeypatch.setattr(Path, "unlink", fail_first_prune_unlink)
            with pytest.raises(ExperienceStoreUnavailable):
                store.publish_revision(
                    records[revision],
                    f"{revision + 2:064x}",
                    SOURCE_DIGEST,
                    expected_revision=revision - 1,
                    expected_digest=experience_record_digest(records[revision - 1]),
                )
            monkeypatch.undo()
            break
        store.publish_revision(
            records[revision],
            f"{revision + 2:064x}",
            SOURCE_DIGEST,
            expected_revision=revision - 1,
            expected_digest=experience_record_digest(records[revision - 1]),
        )

    directory = store.records_root / records[-1].experience_id
    prefix = store.record_path(records[-1].experience_id, 0)
    prefix_bytes = prefix.read_bytes()
    marker = directory / ".compaction.json"
    assert marker.exists()
    store.record_path(records[-1].experience_id, 1).unlink()

    with pytest.raises(ExperienceStoreCorrupt):
        store.load_current(records[-1].experience_id)
    assert prefix.read_bytes() == prefix_bytes
    assert marker.exists()
