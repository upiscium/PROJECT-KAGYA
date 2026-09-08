"""Deterministic StateWAL contract tests."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kagya.config import Settings, load_settings
from kagya.runtime.agent_state import AgentStateSnapshot, EmotionStateSnapshot
from kagya.runtime.state_wal import (
    RecoveryReason,
    Manifest,
    StateWAL,
    StateWALConflictError,
    StateWALFormatError,
    StateWALIntegrityError,
    StateWALMissing,
    StateWALError,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def make_snapshot(sequence: int, *, value: float = 0.1) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        saved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=value, arousal=0.2, optimal_loss=1.0
        ),
    )


def make_wal(tmp_path: Path) -> StateWAL:
    return StateWAL(tmp_path / "wal")


def test_wal_config_is_strict_and_backward_compatible() -> None:
    settings = load_settings(CONFIG_PATH)
    assert settings.state_wal.directory == Path(".kagya/private/state_wal")

    pre_r06 = settings.model_dump(mode="python")
    pre_r06.pop("state_wal")
    compatible = Settings.model_validate(pre_r06)
    assert compatible.state_wal.directory == Path(".kagya/private/state_wal")
    with pytest.raises(ValidationError):
        type(settings.state_wal).model_validate(
            {"directory": ".kagya/private/state_wal", "unexpected": True}
        )


def bootstrap(tmp_path: Path, sequence: int = 10) -> tuple[StateWAL, Manifest]:
    wal = make_wal(tmp_path)
    return wal, wal.bootstrap(make_snapshot(sequence), sequence)


def append(wal: StateWAL, before: int, after: int):
    return wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=after,
        prior_snapshot=make_snapshot(before),
        candidate_snapshot=make_snapshot(after),
    )


def test_absent_inspection_does_not_create_artifacts(tmp_path: Path) -> None:
    wal = make_wal(tmp_path)
    result = wal.inspect_optional()
    assert not result.exists
    assert not wal.root.exists()
    with pytest.raises(StateWALMissing):
        wal.inspect()


def test_bootstrap_layout_and_permissions(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    assert wal.root.stat().st_mode & 0o777 == 0o700
    assert (wal.root / "generations").stat().st_mode & 0o777 == 0o700
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    assert generation.stat().st_mode & 0o777 == 0o600
    assert (wal.root / "manifest.json").stat().st_mode & 0o777 == 0o600


def test_inspection_exposes_baseline_and_latest_identity(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    inspection = wal.inspect()
    assert inspection.active_manifest == manifest
    assert inspection.baseline_record_id == manifest.active_baseline_record_id
    assert inspection.latest_snapshot_sequence == 10
    assert inspection.latest_snapshot_hash is not None


def test_explicit_versions_are_persisted(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    raw = json.loads((wal.root / "manifest.json").read_text())
    assert raw["schema_version"] == 1
    assert raw["command_version"] == 1
    assert (
        json.loads(
            (
                wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
            ).read_text()
        )["schema_version"]
        == 1
    )


def test_each_record_persists_record_hash(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    append(wal, 10, 12)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    for line in path.read_text().splitlines():
        assert len(json.loads(line)["record_hash"]) == 64


def test_tampered_embedded_snapshot_is_rejected(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    append(wal, 10, 12)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = path.read_text().splitlines()
    candidate = json.loads(lines[1])
    candidate["candidate_snapshot"]["emotion_state"]["valence"] = 0.9
    lines[1] = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_versionless_record_is_rejected(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    raw = json.loads(path.read_text())
    del raw["schema_version"]
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_partial_generation_is_rejected(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_noncontiguous_state_sequences_are_valid(tmp_path: Path) -> None:
    wal, _ = bootstrap(tmp_path, 10)
    append(wal, 10, 12)
    assert wal.reconstruct(sequence=12).last_processed_event_sequence == 12


def test_append_requires_processing_sequence_equal_candidate_sequence(
    tmp_path: Path,
) -> None:
    wal, _ = bootstrap(tmp_path, 10)
    with pytest.raises(StateWALConflictError):
        wal.append_transition(
            event_id=uuid4(),
            event_type="state",
            event_source="test",
            processing_sequence=13,
            prior_snapshot=make_snapshot(10),
            candidate_snapshot=make_snapshot(12),
        )


def test_append_failure_is_bounded_without_cause_context(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path, 10)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    path.chmod(0o400)
    with pytest.raises(StateWALError) as error:
        append(wal, 10, 12)
    assert error.value.__cause__ is None


def test_hash_chain_and_forged_lineage_are_rejected(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    record = append(wal, 10, 12)
    assert wal.inspect().record_hashes[-1]
    with pytest.raises(StateWALConflictError):
        wal.append_transition(
            event_id=uuid4(),
            event_type="state",
            event_source="test",
            processing_sequence=13,
            prior_snapshot=make_snapshot(10),
            candidate_snapshot=make_snapshot(13),
        )
    assert record.previous_record_hash == wal.inspect().record_hashes[0]
    assert (
        manifest.active_generation_id
        == wal.inspect().active_manifest.active_generation_id
    )


def test_duplicate_event_id_is_rejected_by_integrity_inspection(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    append(wal, 10, 12)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    first = path.read_text().splitlines()[1]
    path.write_text(path.read_text() + first + "\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_tampered_manifest_is_rejected(tmp_path: Path) -> None:
    wal, _ = bootstrap(tmp_path)
    path = wal.root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["external_reconciliation_required"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_future_record_version_is_rejected(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    raw = json.loads(path.read_text())
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_symlink_generation_is_rejected(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    target = tmp_path / "outside"
    target.write_text("{}")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises((StateWALFormatError, StateWALIntegrityError)):
        wal.inspect()


def test_reconstruct_by_sequence_hash_and_record(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    transition = append(wal, 10, 12)
    inspection = wal.inspect()
    assert wal.reconstruct(sequence=10).last_processed_event_sequence == 10
    assert (
        wal.reconstruct(
            snapshot_hash=inspection.latest_snapshot_hash
        ).last_processed_event_sequence
        == 12
    )
    assert (
        wal.reconstruct(record_id=transition.record_id).last_processed_event_sequence
        == 12
    )


def test_dry_run_has_typed_deterministic_changes_and_no_side_effects(
    tmp_path: Path,
) -> None:
    wal, _ = bootstrap(tmp_path)
    append(wal, 10, 12)
    diff = wal.dry_run(sequence=10)
    assert diff.external_side_effects_replayed is False
    assert tuple(change.field for change in diff.changes) == tuple(
        sorted(change.field for change in diff.changes)
    )


def test_uncommitted_tail_is_not_described_as_commit(tmp_path: Path) -> None:
    wal, _ = bootstrap(tmp_path)
    append(wal, 10, 12)
    assert not hasattr(wal.inspect().active_manifest, "committed")


def test_generation_switch_preserves_old_generation(tmp_path: Path) -> None:
    wal, first = bootstrap(tmp_path)
    second = wal.begin_generation(
        make_snapshot(5), 5, reason=RecoveryReason.TRUE_ROLLBACK
    )
    assert first.active_generation_id != second.active_generation_id
    assert (wal.root / "generations" / f"{first.active_generation_id}.jsonl").exists()


def test_boot_anchor_and_corrupt_tail_prefix(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    inspection = wal.inspect()
    wal.publish_boot_anchor(
        snapshot_sequence=10,
        snapshot_hash=inspection.latest_snapshot_hash,
        generation_id=manifest.active_generation_id,
        anchored_record_id=manifest.active_baseline_record_id,
        anchored_record_hash=manifest.active_baseline_record_hash,
        journal_processing_high_water=10,
        journal_tail_record_id=uuid4(),
        journal_tail_record_hash="a" * 64,
        journal_lineage_id=uuid4(),
    )
    path = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    with path.open("ab") as output:
        output.write(b"not-json\n")
    anchor, snapshot = wal.inspect_anchored_prefix()
    assert anchor.snapshot_sequence == snapshot.last_processed_event_sequence == 10


def test_external_reconciliation_gate_is_durable(tmp_path: Path) -> None:
    wal = make_wal(tmp_path)
    manifest = wal.bootstrap(make_snapshot(0), 0, external_reconciliation_required=True)
    assert wal.inspect().active_manifest.external_reconciliation_required
    assert manifest.external_reconciliation_required


def test_private_sentinel_and_bounded_errors(tmp_path: Path) -> None:
    wal = make_wal(tmp_path)
    with pytest.raises(StateWALError) as error:
        wal.bootstrap({"hiddenthought": "sentinel"}, 0)  # type: ignore[arg-type]
    assert "Traceback" not in str(error.value)
    assert "sentinel" not in str(error.value)


def test_existing_wrong_permissions_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "wal"
    root.mkdir(mode=0o755)
    wal = StateWAL(root)
    with pytest.raises(StateWALError):
        wal.inspect_optional()


def test_existing_reconciliation_gate_cannot_be_cleared_by_new_generation(
    tmp_path: Path,
) -> None:
    wal = make_wal(tmp_path)
    initial = make_snapshot(0)
    wal.begin_generation(
        initial,
        4,
        reason=RecoveryReason.TRUE_ROLLBACK,
        external_reconciliation_required=True,
    )

    with pytest.raises(StateWALConflictError, match="cannot be cleared"):
        wal.begin_generation(make_snapshot(0, value=0.2), 4)

    assert wal.inspect().active_manifest.external_reconciliation_required


def test_prepared_recovery_preserves_and_replaces_partial_generation(
    tmp_path: Path,
) -> None:
    wal = make_wal(tmp_path)
    initial = make_snapshot(0)
    prior = wal.bootstrap(initial, 3)
    predecessor_hash = wal.inspect().record_hashes[-1]
    generation_id = uuid4()
    partial = wal.root / "generations" / f"{generation_id}.jsonl"
    partial.write_bytes(b'{"partial":')
    partial.chmod(0o600)

    manifest = wal.resume_prepared_generation(
        initial,
        3,
        recovery_id=uuid4(),
        reason=RecoveryReason.TRUE_ROLLBACK,
        generation_id=generation_id,
        predecessor_generation_id=prior.active_generation_id,
        predecessor_generation_hash=predecessor_hash,
        external_reconciliation_required=True,
    )

    assert manifest.active_generation_id == generation_id
    assert wal.inspect().latest_snapshot_sequence == 0
    assert list((wal.root / "generations").glob(f".{partial.name}.*.invalid"))


def test_stale_unique_atomic_temporary_does_not_block_future_writes(
    tmp_path: Path,
) -> None:
    wal = make_wal(tmp_path)
    initial = make_snapshot(0)
    prior = wal.bootstrap(initial, 0)
    predecessor_hash = wal.inspect().record_hashes[-1]
    stale = wal.root / f".manifest.json.{uuid4()}.tmp"
    stale.write_bytes(b"partial")
    stale.chmod(0o600)

    manifest = wal.begin_generation(
        initial,
        0,
        generation_id=uuid4(),
        predecessor_generation_id=prior.active_generation_id,
        predecessor_generation_hash=predecessor_hash,
    )

    assert manifest.active_generation_id != prior.active_generation_id
    assert stale.read_bytes() == b"partial"


def test_corrupt_existing_wal_cannot_be_replaced_by_ordinary_generation(
    tmp_path: Path,
) -> None:
    wal = make_wal(tmp_path)
    active = wal.bootstrap(make_snapshot(0), 0, external_reconciliation_required=True)
    predecessor_hash = wal.inspect().record_hashes[-1]
    manifest_path = wal.root / "manifest.json"
    manifest_path.write_bytes(b"corrupt\n")

    with pytest.raises(StateWALError):
        wal.begin_generation(make_snapshot(0, value=0.2), 0)
    with pytest.raises(StateWALError):
        wal.resume_prepared_generation(
            make_snapshot(0, value=0.2),
            0,
            recovery_id=uuid4(),
            reason=RecoveryReason.UNCOMMITTED_TAIL,
            generation_id=uuid4(),
            predecessor_generation_id=active.active_generation_id,
            predecessor_generation_hash=predecessor_hash,
            external_reconciliation_required=False,
        )

    assert manifest_path.read_bytes() == b"corrupt\n"


def test_generation_record_without_final_newline_is_partial(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    generation.write_bytes(generation.read_bytes().removesuffix(b"\n"))

    with pytest.raises(StateWALError):
        wal.inspect()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not supported")
def test_non_regular_artifacts_fail_closed(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    generation.unlink()
    os.mkfifo(generation)
    with pytest.raises(StateWALError):
        wal.inspect()

    generation.unlink()
    generation.write_bytes(b"not used\n")
    generation.chmod(0o600)
    manifest_path = wal.root / "manifest.json"
    manifest_path.unlink()
    os.mkfifo(manifest_path)
    with pytest.raises(StateWALError):
        wal.inspect()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not supported")
def test_anchor_substitution_fails_closed(tmp_path: Path) -> None:
    wal, manifest = bootstrap(tmp_path)
    inspection = wal.inspect()
    wal.publish_boot_anchor(
        snapshot_sequence=10,
        snapshot_hash=inspection.latest_snapshot_hash,
        generation_id=manifest.active_generation_id,
        anchored_record_id=manifest.active_baseline_record_id,
        anchored_record_hash=manifest.active_baseline_record_hash,
        journal_processing_high_water=10,
        journal_lineage_id=uuid4(),
    )
    anchor = wal.root / "boot_anchor.json"
    anchor.unlink()
    os.mkfifo(anchor)
    with pytest.raises(StateWALError):
        wal.inspect_boot_anchor()


def test_fresh_directory_entries_are_fsynced(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    original_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    bootstrap(tmp_path)
    assert len(calls) >= 6


def test_permissive_private_parent_is_hardened(tmp_path: Path) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o755)
    wal = StateWAL(parent / "wal")
    wal.bootstrap(make_snapshot(0), 0)
    assert parent.stat().st_mode & 0o777 == 0o700


def test_missing_private_parent_chain_is_created_and_durable(tmp_path: Path) -> None:
    private = tmp_path / "state" / ".kagya" / "private"
    wal = StateWAL(private / "state_wal")

    wal.bootstrap(make_snapshot(0), 0)

    assert private.stat().st_mode & 0o777 == 0o700
    assert wal.root.stat().st_mode & 0o777 == 0o700
    assert (wal.root / "generations").stat().st_mode & 0o777 == 0o700


def test_intermediate_directory_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises(StateWALError):
        StateWAL(linked / "wal").bootstrap(make_snapshot(0), 0)
    assert not (target / "wal").exists()


def test_prepared_rebaseline_preserves_invalid_manifest_bytes(tmp_path: Path) -> None:
    wal, _manifest = bootstrap(tmp_path)
    manifest_path = wal.root / "manifest.json"
    invalid = b"corrupt-manifest\n"
    manifest_path.write_bytes(invalid)
    recovery_id = uuid4()

    wal.rebaseline_prepared_current(
        make_snapshot(10),
        10,
        recovery_id=recovery_id,
        generation_id=uuid4(),
    )

    preserved = list(wal.root.glob(f".manifest.json.{recovery_id}.*.invalid"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == invalid
