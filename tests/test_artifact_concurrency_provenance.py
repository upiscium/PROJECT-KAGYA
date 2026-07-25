import json
import multiprocessing
import os
from pathlib import Path
from threading import Barrier, Thread
import time
from types import SimpleNamespace

import pytest

from kagya._build_info import SourceRevisionStatus, resolve_source_build_info
from kagya.artifact_provenance import (
    build_adapter_artifact_manifest,
    build_model_artifact_manifest,
    require_immutable_revision,
    verify_attached_adapter_config,
)
from kagya.learning import (
    BehavioralArtifactBusyError,
    BehavioralArtifactStatus,
    BehavioralArtifactStore,
    BehavioralEvaluationState,
)
from kagya.learning.adapter_registry import _cached_model_manifest


def _prepare_process(
    root: str,
    evaluation_id: str,
    start: multiprocessing.Event,
    output: multiprocessing.Queue,
) -> None:
    store = BehavioralArtifactStore(Path(root))
    start.wait()
    try:
        store.prepare(evaluation_id, {"evaluation_id": evaluation_id})
        output.put("prepared")
    except ValueError:
        output.put("duplicate")


def _hold_adapter_process(
    root: str,
    adapter_id: str,
    acquired: multiprocessing.Event,
    release: multiprocessing.Event,
) -> None:
    with BehavioralArtifactStore(Path(root)).adapter_lock(adapter_id):
        acquired.set()
        release.wait(10)


def test_registry_updates_are_not_lost_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_prepare_process,
            args=(str(tmp_path), f"eval-{index}", start, output),
        )
        for index in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    assert [output.get(timeout=1) for _ in processes].count("prepared") == 8
    records = BehavioralArtifactStore(tmp_path).reconcile()
    assert {record.evaluation_id for record in records} == {
        f"eval-{index}" for index in range(8)
    }
    assert not list((tmp_path / "behavioral").glob("*.tmp"))


def test_duplicate_id_prepare_has_exactly_one_process_winner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_prepare_process, args=(str(tmp_path), "same", start, output)
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert sorted(output.get(timeout=1) for _ in processes) == ["duplicate", "prepared"]


def test_adapter_locks_serialize_same_key_but_not_different_keys(
    tmp_path: Path,
) -> None:
    store = BehavioralArtifactStore(tmp_path)
    barrier = Barrier(2)
    outcomes: list[str] = []

    def run_same() -> None:
        barrier.wait()
        try:
            with store.adapter_lock("same"):
                outcomes.append("held")
                time.sleep(0.1)
        except BehavioralArtifactBusyError:
            outcomes.append("busy")

    threads = [Thread(target=run_same) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["busy", "held"]

    context = multiprocessing.get_context("fork")
    acquired = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_adapter_process,
        args=(str(tmp_path), "process-adapter", acquired, release),
    )
    process.start()
    assert acquired.wait(5)
    with pytest.raises(BehavioralArtifactBusyError):
        with store.adapter_lock("process-adapter"):
            pass
    with store.adapter_lock("different-adapter"):
        pass
    release.set()
    process.join(10)
    assert process.exitcode == 0


def test_finalize_and_reconcile_race_preserves_single_valid_artifact(
    tmp_path: Path,
) -> None:
    store = BehavioralArtifactStore(tmp_path)
    store.prepare("race", {"evaluation_id": "race"})
    barrier = Barrier(2)
    failures: list[Exception] = []

    def finalize() -> None:
        barrier.wait()
        try:
            store.finalize("race")
        except ValueError as exc:
            failures.append(exc)

    def reconcile() -> None:
        barrier.wait()
        store.reconcile()

    threads = [Thread(target=finalize), Thread(target=reconcile)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = store.reconcile()
    assert len(records) == 1
    assert records[0].status == BehavioralArtifactStatus.VALID
    assert not failures


def test_interrupted_preprepare_state_is_failed_not_permanently_running(
    tmp_path: Path,
) -> None:
    store = BehavioralArtifactStore(tmp_path)
    store.begin("interrupted", adapter_key="adapter")
    store.mark_running("interrupted")

    record = store.reconcile()[0]

    assert record.state == BehavioralEvaluationState.FAILED
    assert record.failure_code == "interrupted_before_prepare"
    assert record.status == BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE


def test_reconcile_does_not_fail_a_currently_locked_running_evaluation(
    tmp_path: Path,
) -> None:
    store = BehavioralArtifactStore(tmp_path)
    with store.adapter_lock("adapter"):
        store.begin("active", adapter_key="adapter")
        store.mark_running("active")
        record = store.reconcile()[0]

    assert record.state == BehavioralEvaluationState.RUNNING


def test_canonical_manifests_detect_model_processor_weight_quant_and_adapter_changes(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "snapshot"
    processor_snapshot = tmp_path / "processor-snapshot"
    snapshot.mkdir()
    processor_snapshot.mkdir()
    for name, content in {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "quantization_config.json": b'{"bits":4}',
    }.items():
        (snapshot / name).write_bytes(content)
    (snapshot / "custom_behavior.rules").write_bytes(b"rule=v1")
    (processor_snapshot / "processor_config.json").write_bytes(b"{}")
    (processor_snapshot / "chat_template.jinja").write_bytes(b"{{ messages }}")
    (processor_snapshot / "tokenizer.tiktoken").write_bytes(b"token data")
    first = build_model_artifact_manifest(
        snapshot,
        processor_snapshot=processor_snapshot,
        model_id="model",
        requested_revision=commit,
        resolved_revision=commit,
        processor_requested_revision=commit,
        processor_resolved_revision=commit,
    )
    (processor_snapshot / "processor_config.json").write_text(
        '{"changed":true}', encoding="utf-8"
    )
    second = build_model_artifact_manifest(
        snapshot,
        processor_snapshot=processor_snapshot,
        model_id="model",
        requested_revision=commit,
        resolved_revision=commit,
        processor_requested_revision=commit,
        processor_resolved_revision=commit,
    )
    assert first.sha256 != second.sha256
    assert "custom_behavior.rules" in {
        item.filename for item in first.metadata_files
    }
    assert "tokenizer.tiktoken" in {
        item.filename for item in first.processor_files
    }
    assert all(
        not Path(item.filename).is_absolute()
        for item in second.metadata_files + second.processor_files + second.weight_files
    )

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    adapter_first = build_adapter_artifact_manifest(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    assert adapter_first.sha256 != build_adapter_artifact_manifest(adapter).sha256


def test_model_manifest_revalidation_detects_same_metadata_content_mutation(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    model = tmp_path / "model"
    processor = tmp_path / "processor"
    model.mkdir()
    processor.mkdir()
    config = model / "config.json"
    config.write_bytes(b'{"mode":"a"}')
    (model / "model.safetensors").write_bytes(b"weights")
    (processor / "tokenizer.json").write_bytes(b'{"token":"a"}')
    manifest = build_model_artifact_manifest(
        model,
        processor_snapshot=processor,
        model_id="model",
        requested_revision=commit,
        resolved_revision=commit,
        processor_requested_revision=commit,
        processor_resolved_revision=commit,
    )
    stat = config.stat()
    config.write_bytes(b'{"mode":"b"}')
    os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ValueError, match="content differs"):
        _cached_model_manifest(
            model,
            processor_snapshot=processor,
            expected_hash=manifest.sha256,
            model_id="model",
            requested_revision=commit,
            resolved_revision=commit,
            processor_requested_revision=commit,
            processor_resolved_revision=commit,
        )


@pytest.mark.parametrize("revision", ["main", "master", "latest", "feature", "a" * 39])
def test_mutable_or_arbitrary_revisions_are_rejected(revision: str) -> None:
    with pytest.raises(ValueError, match="exact immutable"):
        require_immutable_revision(revision, "model revision")


def test_source_resolution_records_unknown_without_zero_sha_or_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "KAGYA_SOURCE_COMMIT_SHA",
        "KAGYA_SOURCE_TREE_HASH",
        "KAGYA_BUILD_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    info = resolve_source_build_info(tmp_path)

    assert info.status == SourceRevisionStatus.UNKNOWN
    assert info.commit_sha is None
    assert str(tmp_path) not in repr(info)


def test_attached_peft_config_must_match_hashed_adapter_manifest(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "peft"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "model",
                "revision": "a" * 40,
                "target_modules": ["q_proj"],
                "r": 8,
                "lora_alpha": 16,
            }
        ),
        encoding="utf-8",
    )
    manifest = build_adapter_artifact_manifest(adapter)
    attached = SimpleNamespace(
        peft_config={
            "default": SimpleNamespace(
                peft_type="LORA",
                base_model_name_or_path="model",
                revision="a" * 40,
                target_modules={"q_proj"},
                r=8,
                lora_alpha=16,
            )
        }
    )
    verify_attached_adapter_config(attached, manifest)

    attached.peft_config["default"].r = 16
    with pytest.raises(RuntimeError, match="PEFT config"):
        verify_attached_adapter_config(attached, manifest)
