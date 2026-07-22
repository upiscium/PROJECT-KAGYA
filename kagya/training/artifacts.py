"""Immutable TrainingBundle and TrainingResult filesystem contracts."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


_UNSET = object()


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterLineageNode(_ArtifactModel):
    adapter_id: str = Field(min_length=1)
    adapter_hash: str = Field(min_length=1)
    parent_adapter_id: str | None = None
    parent_adapter_hash: str | None = None
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)


class TrainingBundleManifest(_ArtifactModel):
    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    created_at: datetime
    submitter_node_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    submitter_hostname: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    processor_revision: str = Field(min_length=1)
    parent_adapter_id: str | None = None
    parent_adapter_hash: str | None = None
    lineage_adapter_ids: list[str] = Field(default_factory=list)
    lineage: list[AdapterLineageNode] = Field(default_factory=list)
    source_event_sequence_start: int = Field(ge=0)
    source_event_sequence_end: int = Field(ge=0)
    source_episode_ids: list[str] = Field(default_factory=list)
    source_decision_ids: list[str] = Field(default_factory=list)
    dataset_path: Literal["dataset.jsonl"] = "dataset.jsonl"
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_record_count: int = Field(ge=0)
    evaluation_set_path: Literal["evaluation_set.jsonl"] = "evaluation_set.jsonl"
    evaluation_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_record_count: int = Field(ge=0)
    rehearsal_record_count: int = Field(default=0, ge=0)
    repeated_record_count: int = Field(default=0, ge=0)
    training_evaluation_overlap_count: int = Field(default=0, ge=0)
    chat_template_version: str = Field(min_length=1)
    dataset_format_version: str = Field(min_length=1)
    dataset_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_path: Literal["dataset_manifest.json"] | None = None
    dataset_manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    qlora_hyperparameters: dict[str, int | float | str | bool]
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self) -> "TrainingBundleManifest":
        if self.source_event_sequence_end < self.source_event_sequence_start:
            raise ValueError("source event sequence range is reversed")
        if (self.parent_adapter_id is None) != (self.parent_adapter_hash is None):
            raise ValueError("parent adapter ID and hash must be provided together")
        governance_fields = (
            self.dataset_revision,
            self.dataset_manifest_path,
            self.dataset_manifest_hash,
        )
        if any(item is not None for item in governance_fields) and not all(
            item is not None for item in governance_fields
        ):
            raise ValueError("dataset governance revision fields must be provided together")
        self._validate_lineage()
        return self

    def _validate_lineage(self) -> None:
        if self.parent_adapter_id is None:
            if self.lineage:
                raise ValueError("base training bundle cannot contain adapter lineage")
            return
        nodes = {node.adapter_id: node for node in self.lineage}
        if len(nodes) != len(self.lineage):
            raise ValueError("adapter lineage contains duplicate nodes")
        current_id: str | None = self.parent_adapter_id
        expected_hash = self.parent_adapter_hash
        seen: set[str] = set()
        while current_id is not None:
            if current_id in seen:
                raise ValueError("adapter lineage contains a cycle")
            seen.add(current_id)
            node = nodes.get(current_id)
            if node is None:
                raise ValueError(
                    f"adapter lineage contains unknown parent: {current_id}"
                )
            if node.adapter_hash != expected_hash:
                raise ValueError("adapter lineage parent hash mismatch")
            if (
                node.base_model_id != self.base_model_id
                or node.base_model_revision != self.base_model_revision
            ):
                raise ValueError("adapter lineage base model revision mismatch")
            if (node.parent_adapter_id is None) != (node.parent_adapter_hash is None):
                raise ValueError("adapter lineage parent ID/hash pair is incomplete")
            current_id = node.parent_adapter_id
            expected_hash = node.parent_adapter_hash
        if seen != set(nodes):
            raise ValueError("adapter lineage contains disconnected nodes")


class TrainingResultManifest(_ArtifactModel):
    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    created_at: datetime
    worker_node_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    worker_hostname: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "cancelled"]
    candidate_adapter_id: str | None = None
    candidate_adapter_hash: str | None = None
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    parent_adapter_id: str | None = None
    parent_adapter_hash: str | None = None
    training_metrics_path: Literal["training_metrics.json"] = "training_metrics.json"
    evaluation_path: Literal["evaluation.json"] = "evaluation.json"
    failure_category: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "TrainingResultManifest":
        if self.status == "succeeded":
            if self.candidate_adapter_id is None or self.candidate_adapter_hash is None:
                raise ValueError(
                    "successful result requires candidate adapter ID and hash"
                )
            if self.failure_category is not None or self.error is not None:
                raise ValueError("successful result cannot contain failure fields")
        elif self.failure_category is None:
            raise ValueError("non-success result requires failure_category")
        return self


class TrainingArtifactContract:
    BUNDLE_FILES = {
        "manifest.json",
        "dataset.jsonl",
        "evaluation_set.jsonl",
        "checksums.sha256",
    }

    def finalize_bundle(
        self,
        artifact_root: Path,
        manifest: TrainingBundleManifest,
        *,
        dataset: bytes,
        evaluation_set: bytes,
        dataset_manifest: bytes | None = None,
    ) -> Path:
        if sha256_bytes(dataset) != manifest.dataset_hash:
            raise ValueError("dataset hash does not match bundle manifest")
        if sha256_bytes(evaluation_set) != manifest.evaluation_set_hash:
            raise ValueError("evaluation set hash does not match bundle manifest")
        final_path = artifact_root / f"training-{manifest.job_id}"
        files = {
            "manifest.json": _json_bytes(manifest.model_dump(mode="json")),
            "dataset.jsonl": dataset,
            "evaluation_set.jsonl": evaluation_set,
        }
        if manifest.dataset_manifest_path is not None:
            if dataset_manifest is None:
                raise ValueError("governed bundle requires a dataset manifest")
            if sha256_bytes(dataset_manifest) != manifest.dataset_manifest_hash:
                raise ValueError("dataset manifest hash does not match bundle manifest")
            files[manifest.dataset_manifest_path] = dataset_manifest
        elif dataset_manifest is not None:
            raise ValueError("ungoverned bundle cannot contain a dataset manifest")
        self._atomic_finalize(artifact_root, final_path, files)
        self.validate_bundle(final_path)
        return final_path

    def validate_bundle(
        self,
        path: Path,
        *,
        expected_model_id: str | None = None,
        expected_model_revision: str | None = None,
        expected_processor_revision: str | None = None,
        expected_parent_adapter_id: str | None | object = _UNSET,
        expected_parent_adapter_hash: str | None | object = _UNSET,
    ) -> TrainingBundleManifest:
        self._validate_tree(path)
        if not self.BUNDLE_FILES.issubset(_relative_files(path)):
            raise ValueError("artifact file set does not match contract")
        manifest = TrainingBundleManifest.model_validate(
            _read_json(path / "manifest.json")
        )
        expected_files = set(self.BUNDLE_FILES)
        if manifest.dataset_manifest_path is not None:
            expected_files.add(manifest.dataset_manifest_path)
        if _relative_files(path) != expected_files:
            raise ValueError("artifact file set does not match contract")
        self._validate_checksums(path)
        dataset = (path / manifest.dataset_path).read_bytes()
        evaluation = (path / manifest.evaluation_set_path).read_bytes()
        if sha256_bytes(dataset) != manifest.dataset_hash:
            raise ValueError("bundle dataset hash mismatch")
        if sha256_bytes(evaluation) != manifest.evaluation_set_hash:
            raise ValueError("bundle evaluation hash mismatch")
        if manifest.dataset_manifest_path is not None:
            dataset_manifest = (path / manifest.dataset_manifest_path).read_bytes()
            if sha256_bytes(dataset_manifest) != manifest.dataset_manifest_hash:
                raise ValueError("bundle dataset manifest hash mismatch")
            revision_manifest = json.loads(dataset_manifest)
            if revision_manifest.get("revision") != manifest.dataset_revision:
                raise ValueError("bundle dataset revision mismatch")
        if (
            expected_model_id is not None
            and manifest.base_model_id != expected_model_id
        ):
            raise ValueError("bundle base model ID mismatch")
        if (
            expected_model_revision is not None
            and manifest.base_model_revision != expected_model_revision
        ):
            raise ValueError("bundle base model revision mismatch")
        if (
            expected_processor_revision is not None
            and manifest.processor_revision != expected_processor_revision
        ):
            raise ValueError("bundle processor revision mismatch")
        if (
            expected_parent_adapter_id is not _UNSET
            and manifest.parent_adapter_id != expected_parent_adapter_id
        ):
            raise ValueError("bundle parent adapter mismatch")
        if (
            expected_parent_adapter_hash is not _UNSET
            and manifest.parent_adapter_hash != expected_parent_adapter_hash
        ):
            raise ValueError("bundle parent adapter hash mismatch")
        if _record_hashes(dataset) & _record_hashes(evaluation):
            raise ValueError("bundle training and evaluation datasets overlap")
        return manifest

    def finalize_result(
        self,
        artifact_root: Path,
        manifest: TrainingResultManifest,
        *,
        training_metrics: dict[str, Any],
        evaluation: dict[str, Any],
        adapter_files: dict[str, bytes] | None = None,
    ) -> Path:
        adapter_files = adapter_files or {}
        if manifest.status == "succeeded" and not adapter_files:
            raise ValueError("successful result requires adapter files")
        if (
            manifest.status == "succeeded"
            and sha256_file_map(adapter_files) != manifest.candidate_adapter_hash
        ):
            raise ValueError("candidate adapter hash does not match adapter payload")
        files = {
            "result.json": _json_bytes(manifest.model_dump(mode="json")),
            "training_metrics.json": _json_bytes(training_metrics),
            "evaluation.json": _json_bytes(evaluation),
        }
        for relative, content in adapter_files.items():
            safe = _safe_relative_path(relative)
            if safe.parts[0] != "adapter":
                raise ValueError("adapter files must be under adapter/")
            files[safe.as_posix()] = content
        final_path = artifact_root / f"result-{manifest.job_id}"
        self._atomic_finalize(artifact_root, final_path, files)
        self.validate_result(final_path)
        return final_path

    def validate_result(
        self,
        path: Path,
        *,
        expected_job_id: str | None = None,
        expected_attempt_id: str | None = None,
        expected_model_id: str | None = None,
        expected_model_revision: str | None = None,
        expected_parent_adapter_id: str | None | object = _UNSET,
        expected_parent_adapter_hash: str | None | object = _UNSET,
    ) -> TrainingResultManifest:
        self._validate_tree(path)
        required = {
            "result.json",
            "training_metrics.json",
            "evaluation.json",
            "checksums.sha256",
        }
        actual = _relative_files(path)
        if not required.issubset(actual):
            raise ValueError("training result is incomplete")
        manifest = TrainingResultManifest.model_validate(
            _read_json(path / "result.json")
        )
        if manifest.status == "succeeded" and not any(
            item.startswith("adapter/") for item in actual
        ):
            raise ValueError("successful result has no adapter payload")
        unexpected = {
            item
            for item in actual
            if item not in required and not item.startswith("adapter/")
        }
        if unexpected:
            raise ValueError("training result contains unknown payload paths")
        self._validate_checksums(path)
        adapter_payload = {
            relative: (path / relative).read_bytes()
            for relative in actual
            if relative.startswith("adapter/")
        }
        if (
            manifest.status == "succeeded"
            and sha256_file_map(adapter_payload) != manifest.candidate_adapter_hash
        ):
            raise ValueError("result candidate adapter hash mismatch")
        if expected_job_id is not None and manifest.job_id != expected_job_id:
            raise ValueError("result job ID mismatch")
        if (
            expected_attempt_id is not None
            and manifest.attempt_id != expected_attempt_id
        ):
            raise ValueError("result attempt ID mismatch")
        if (
            expected_model_id is not None
            and manifest.base_model_id != expected_model_id
        ):
            raise ValueError("result base model ID mismatch")
        if (
            expected_model_revision is not None
            and manifest.base_model_revision != expected_model_revision
        ):
            raise ValueError("result base model revision mismatch")
        if (
            expected_parent_adapter_id is not _UNSET
            and manifest.parent_adapter_id != expected_parent_adapter_id
        ):
            raise ValueError("result parent adapter mismatch")
        if (
            expected_parent_adapter_hash is not _UNSET
            and manifest.parent_adapter_hash != expected_parent_adapter_hash
        ):
            raise ValueError("result parent adapter hash mismatch")
        return manifest

    def _atomic_finalize(
        self, artifact_root: Path, final_path: Path, files: dict[str, bytes]
    ) -> None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise FileExistsError(final_path)
        staging = artifact_root / f".{final_path.name}.{uuid4()}.tmp"
        staging.mkdir()
        try:
            for relative, content in files.items():
                safe = _safe_relative_path(relative)
                destination = staging.joinpath(*safe.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_fsynced(destination, content)
            checksums = _checksum_manifest(staging)
            _write_fsynced(staging / "checksums.sha256", checksums)
            _fsync_tree(staging)
            os.rename(staging, final_path)
            _fsync_directory(artifact_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _validate_tree(path: Path, *, exact_files: set[str] | None = None) -> None:
        if not path.is_dir() or path.is_symlink() or path.name.endswith(".tmp"):
            raise ValueError("artifact is incomplete or not a regular directory")
        for item in path.rglob("*"):
            if item.is_symlink():
                raise ValueError("artifact symlinks are forbidden")
        actual = _relative_files(path)
        if exact_files is not None and actual != exact_files:
            raise ValueError("artifact file set does not match contract")

    @staticmethod
    def _validate_checksums(path: Path) -> None:
        expected = _parse_checksums((path / "checksums.sha256").read_text("utf-8"))
        actual_files = _relative_files(path) - {"checksums.sha256"}
        if set(expected) != actual_files:
            raise ValueError("checksum manifest file set mismatch")
        for relative, digest in expected.items():
            if sha256_bytes((path / relative).read_bytes()) != digest:
                raise ValueError(f"checksum mismatch: {relative}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_hashes(content: bytes) -> set[str]:
    hashes: set[str] = set()
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            normalized = json.dumps(
                json.loads(line),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        except json.JSONDecodeError:
            normalized = line.strip()
        hashes.add(sha256_bytes(normalized))
    return hashes


def sha256_file_map(files: dict[str, bytes]) -> str:
    canonical = b"".join(
        _safe_relative_path(relative).as_posix().encode()
        + b"\0"
        + sha256_bytes(content).encode()
        + b"\n"
        for relative, content in sorted(files.items())
    )
    return sha256_bytes(canonical)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path is unsafe")
    return path


def _relative_files(path: Path) -> set[str]:
    return {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    }


def _checksum_manifest(path: Path) -> bytes:
    lines = [
        f"{sha256_bytes((path / relative).read_bytes())}  {relative}"
        for relative in sorted(_relative_files(path))
        if relative != "checksums.sha256"
    ]
    return ("\n".join(lines) + "\n").encode()


def _parse_checksums(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in value.splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise ValueError("invalid checksum manifest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("invalid checksum digest") from exc
        safe = _safe_relative_path(relative).as_posix()
        if safe in parsed:
            raise ValueError("duplicate checksum path")
        parsed[safe] = digest
    return parsed


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _fsync_tree(path: Path) -> None:
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir()), reverse=True
    ):
        _fsync_directory(directory)
    _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
