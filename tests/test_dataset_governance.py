import json
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.memory import EpisodicMemoryRecord, ValidationStatus
from kagya.training import (
    DatasetCandidate,
    DatasetDisposition,
    DatasetGovernanceStore,
    DatasetProvenance,
    TrainingBundleBuilder,
    TrainingJobRegistry,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _candidate(
    source_id: str,
    text: str,
    *,
    output: str = "safe response",
    consent: str = "runtime_training_allowed",
    privacy: str = "internal",
    training_included: bool = True,
) -> DatasetCandidate:
    return DatasetCandidate(
        input=text,
        output=output,
        provenance=DatasetProvenance(
            source_kind="verified_episode",
            source_id=source_id,
            source_event_ids=(f"event-{source_id}",),
            source_memory_ids=(source_id,),
            source_decision_ids=(f"decision-{source_id}",),
            source_feedback_ids=(f"feedback-{source_id}",),
        ),
        consent=consent,
        privacy=privacy,
        training_included=training_included,
    )


def test_governance_excludes_policy_records_and_quarantines_sensitive_content(
    tmp_path: Path,
) -> None:
    store = DatasetGovernanceStore(tmp_path / "datasets")

    revision = store.create_revision(
        [
            _candidate("safe", "ordinary prompt"),
            _candidate("private", "private prompt", privacy="private"),
            _candidate("denied", "denied prompt", consent="withdrawn"),
            _candidate("rejected", "rejected prompt", training_included=False),
            _candidate("secret", "api_key=sk-abcdefghijklmnop"),
        ],
        source_job_id="job-1",
    )

    records = {record.provenance.source_id: record for record in revision.records}
    assert records["safe"].disposition == DatasetDisposition.INCLUDED
    assert records["safe"].provenance.source_decision_ids == ("decision-safe",)
    assert records["private"].exclusion_reasons == ("private_record",)
    assert records["denied"].exclusion_reasons == ("training_consent_not_granted",)
    assert records["rejected"].exclusion_reasons == ("do_not_train",)
    assert records["secret"].disposition == DatasetDisposition.QUARANTINED
    assert "credential" in records["secret"].quarantine_reasons
    materialized = b"".join(
        revision.split_bytes(split)
        for split in records["safe"].split.__class__
    ).decode()
    assert "ordinary prompt" in materialized
    assert "private prompt" not in materialized
    assert "api_key" not in materialized


def test_sensitive_scanner_failure_is_fail_closed(tmp_path: Path) -> None:
    def broken_scanner(_text: str) -> list[str]:
        raise RuntimeError("scanner unavailable")

    store = DatasetGovernanceStore(
        tmp_path / "datasets", sensitive_scanner=broken_scanner
    )

    record = store.create_revision([_candidate("one", "content")]).records[0]

    assert record.disposition == DatasetDisposition.QUARANTINED
    assert record.quarantine_reasons == ("sensitive_scanner_failure",)
    assert store.get_revision(store.list_revisions()[0]["revision"]).records == (
        record,
    )


def test_duplicate_checks_fixed_splits_diff_and_immutable_checksums(
    tmp_path: Path,
) -> None:
    store = DatasetGovernanceStore(tmp_path / "datasets")
    first = store.create_revision([_candidate("one", "same content")])
    first_split = first.records[0].split

    second = store.create_revision(
        [
            _candidate("one", "same content"),
            _candidate("duplicate", "same content"),
            _candidate("two", "new content"),
        ]
    )
    records = {record.provenance.source_id: record for record in second.records}

    assert records["one"].split == first_split
    assert records["duplicate"].disposition == DatasetDisposition.QUARANTINED
    assert records["duplicate"].quarantine_reasons == ("exact_duplicate",)
    assert store.diff(first.revision, second.revision)["added_record_ids"] == sorted(
        [records["duplicate"].record_id, records["two"].record_id]
    )
    manifest = json.loads((second.path / "manifest.json").read_text("utf-8"))
    assert manifest["revision"] == second.revision

    with (second.path / "train.jsonl").open("ab") as output:
        output.write(b"tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.get_revision(second.revision)


def test_training_bundle_pins_governed_dataset_revision(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "sleep": settings.sleep.model_copy(
                update={"training_artifact_directory": tmp_path / "artifacts"}
            )
        }
    )
    job, _ = TrainingJobRegistry(tmp_path / "jobs.json").create(
        idempotency_key="governed-bundle",
        base_model_id=settings.model.primary_id,
        base_model_revision=settings.model.revision,
        parent_adapter_id=None,
        backend="local",
    )
    episode = EpisodicMemoryRecord(
        id="episode-1",
        user_input="How should this be handled?",
        response="Handle it carefully.",
        validation_status=ValidationStatus.VERIFIED,
        source_event_id="event-1",
        processing_sequence=4,
        metadata={"decision_id": "decision-1", "feedback_id": "feedback-1"},
    )

    bundle = TrainingBundleBuilder(settings).build(job, (episode,))
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    adapter_manifest = json.loads((bundle / "dataset_manifest.json").read_text("utf-8"))

    assert manifest["dataset_revision"] == adapter_manifest["revision"]
    assert manifest["dataset_manifest_hash"]
    assert b"episode-1" in (bundle / "dataset.jsonl").read_bytes()
