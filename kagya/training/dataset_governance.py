"""Immutable, provenance-preserving governance for training datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from difflib import SequenceMatcher
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from kagya.memory import EpisodicMemoryRecord, MemoryLifecycleStatus, ValidationStatus


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetDisposition(StrEnum):
    INCLUDED = "included"
    EXCLUDED = "excluded"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class DatasetProvenance:
    source_kind: str
    source_id: str
    source_event_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()
    source_feedback_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GovernedDatasetRecord:
    record_id: str
    schema_version: int
    input: str
    thought: str
    output: str
    provenance: DatasetProvenance
    inclusion_reason: str
    consent: str
    privacy: str
    disposition: DatasetDisposition
    split: DatasetSplit | None
    content_hash: str
    quarantine_reasons: tuple[str, ...] = ()
    exclusion_reasons: tuple[str, ...] = ()
    quality_checks: tuple[str, ...] = ()
    context_id: str | None = None
    interlocutor_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["disposition"] = self.disposition.value
        value["split"] = None if self.split is None else self.split.value
        return value

    def training_json(self) -> dict[str, Any]:
        """Return the trainer-compatible representation with governance identity."""

        return {
            "schema_version": 2,
            "source_kind": self.provenance.source_kind,
            "source_id": self.provenance.source_id,
            "validation_status": "verified",
            "input": self.input,
            "thought": self.thought,
            "output": self.output,
            "governance_record_id": self.record_id,
            "provenance": asdict(self.provenance),
            "inclusion_reason": self.inclusion_reason,
            "consent": self.consent,
            "privacy": self.privacy,
            "disposition": self.disposition.value,
            "dataset_split": self.split.value if self.split is not None else None,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class DatasetRevision:
    revision: str
    path: Path
    manifest_hash: str
    manifest: dict[str, Any]
    records: tuple[GovernedDatasetRecord, ...]

    def split_bytes(self, split: DatasetSplit) -> bytes:
        return (self.path / f"{split.value}.jsonl").read_bytes()


@dataclass(frozen=True)
class DatasetCandidate:
    input: str
    output: str
    provenance: DatasetProvenance
    thought: str = ""
    consent: str = "runtime_training_allowed"
    privacy: str = "internal"
    training_included: bool = True
    validation_status: str = "verified"
    lifecycle_status: str = "active"
    context_id: str | None = None
    interlocutor_id: str | None = None
    tags: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()


SensitiveScanner = Callable[[str], list[str]]


class DatasetGovernanceStore:
    """Build and browse content-addressed, immutable dataset revisions."""

    def __init__(
        self,
        root: Path,
        *,
        sensitive_scanner: SensitiveScanner | None = None,
        near_duplicate_threshold: float = 0.92,
    ) -> None:
        self.root = root
        self.sensitive_scanner = sensitive_scanner or detect_sensitive_content
        self.near_duplicate_threshold = near_duplicate_threshold

    def create_revision(
        self,
        candidates: Iterable[DatasetCandidate],
        *,
        source_job_id: str | None = None,
    ) -> DatasetRevision:
        previous_assignments, previous_sources, previous_content = (
            self._previous_assignments()
        )
        records = self._govern(
            list(candidates), previous_assignments, previous_sources, previous_content
        )
        self._reject_cross_split_duplicates(records)
        revisions = self.list_revisions()
        parent_revision = revisions[-1]["revision"] if revisions else None
        payload = {
            "schema_version": 1,
            "parent_revision": parent_revision,
            "source_job_id": source_job_id,
            "records": [record.to_json() for record in records],
        }
        revision = _sha256(_json_bytes(payload))
        counts = Counter(record.disposition.value for record in records)
        split_counts = Counter(
            record.split.value for record in records if record.split is not None
        )
        manifest = {
            "schema_version": 1,
            "revision": revision,
            "parent_revision": parent_revision,
            "created_at": datetime.now(UTC).isoformat(),
            "source_job_id": source_job_id,
            "record_count": len(records),
            "disposition_counts": dict(sorted(counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "quality_findings": self._bias_findings(records),
            "record_ids": [record.record_id for record in records],
        }
        final_path = self.root / "revisions" / revision
        if final_path.exists():
            return self.get_revision(revision)
        files = self._revision_files(manifest, records)
        self._atomic_finalize(final_path, files)
        return self.get_revision(revision)

    def create_from_episodes(
        self,
        episodes: Iterable[EpisodicMemoryRecord],
        *,
        source_job_id: str | None = None,
        additional_candidates: Iterable[DatasetCandidate] = (),
    ) -> DatasetRevision:
        candidates = [candidate_from_episode(episode) for episode in episodes]
        candidates.extend(additional_candidates)
        return self.create_revision(candidates, source_job_id=source_job_id)

    def list_revisions(self) -> list[dict[str, Any]]:
        revisions_root = self.root / "revisions"
        if not revisions_root.exists():
            return []
        manifests: list[dict[str, Any]] = []
        for path in revisions_root.iterdir():
            if (
                not path.is_dir()
                or path.is_symlink()
                or re.fullmatch(r"[0-9a-f]{64}", path.name) is None
            ):
                continue
            revision = self.get_revision(path.name)
            manifests.append({**revision.manifest, "manifest_hash": revision.manifest_hash})
        return sorted(manifests, key=lambda item: (item["created_at"], item["revision"]))

    def get_revision(self, revision: str) -> DatasetRevision:
        if re.fullmatch(r"[0-9a-f]{64}", revision) is None:
            raise ValueError("Invalid dataset revision")
        path = self.root / "revisions" / revision
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Unknown dataset revision: {revision}")
        expected_files = {
            "manifest.json",
            "records.jsonl",
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "checksums.sha256",
        }
        actual_files = {item.name for item in path.iterdir() if item.is_file()}
        if actual_files != expected_files or any(item.is_symlink() for item in path.iterdir()):
            raise ValueError("Dataset revision contains unexpected files")
        self._validate_checksums(path)
        manifest_bytes = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("revision") != revision:
            raise ValueError("Dataset revision manifest identity mismatch")
        records = tuple(
            _record_from_json(json.loads(line))
            for line in (path / "records.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        )
        if manifest.get("record_ids") != [record.record_id for record in records]:
            raise ValueError("Dataset revision record inventory mismatch")
        self._reject_cross_split_duplicates(records)
        return DatasetRevision(revision, path, _sha256(manifest_bytes), manifest, records)

    def diff(self, from_revision: str, to_revision: str) -> dict[str, Any]:
        before = self.get_revision(from_revision)
        after = self.get_revision(to_revision)
        before_records = {record.record_id: record for record in before.records}
        after_records = {record.record_id: record for record in after.records}
        shared = before_records.keys() & after_records.keys()
        changed = sorted(
            record_id
            for record_id in shared
            if before_records[record_id].to_json() != after_records[record_id].to_json()
        )
        return {
            "from_revision": from_revision,
            "to_revision": to_revision,
            "added_record_ids": sorted(after_records.keys() - before_records.keys()),
            "removed_record_ids": sorted(before_records.keys() - after_records.keys()),
            "changed_record_ids": changed,
        }

    def _govern(
        self,
        candidates: list[DatasetCandidate],
        previous_assignments: dict[str, DatasetSplit],
        previous_sources: set[tuple[str, str]],
        previous_content: dict[str, str],
    ) -> list[GovernedDatasetRecord]:
        records: list[GovernedDatasetRecord] = []
        seen_record_ids: set[str] = set()
        accepted_content = list(previous_content.items())
        for candidate in sorted(
            candidates,
            key=lambda item: (
                (
                    item.provenance.source_kind,
                    item.provenance.source_id,
                )
                not in previous_sources,
                _content_hash(item.input, item.output) not in previous_assignments,
                item.provenance.source_kind,
                item.provenance.source_id,
            ),
        ):
            content_hash = _content_hash(candidate.input, candidate.output)
            record_id = _sha256(
                f"{candidate.provenance.source_kind}\0{candidate.provenance.source_id}\0{content_hash}".encode()
            )
            if record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)
            exclusions = self._exclusion_reasons(candidate)
            source_key = (
                candidate.provenance.source_kind,
                candidate.provenance.source_id,
            )
            comparison_content = (
                [item for item in accepted_content if item[0] != content_hash]
                if source_key in previous_sources
                else accepted_content
            )
            quarantine = self._quarantine_reasons(
                candidate, content_hash, comparison_content
            )
            if quarantine:
                disposition = DatasetDisposition.QUARANTINED
                split = None
                inclusion_reason = "quarantined_by_dataset_safety_policy"
            elif exclusions:
                disposition = DatasetDisposition.EXCLUDED
                split = None
                inclusion_reason = "excluded_by_source_training_policy"
            else:
                disposition = DatasetDisposition.INCLUDED
                split = previous_assignments.get(content_hash) or _stable_split(content_hash)
                inclusion_reason = "verified_healthy_source_with_training_consent"
                accepted_content.append((content_hash, _normalize(candidate.input + "\n" + candidate.output)))
            records.append(
                GovernedDatasetRecord(
                    record_id=record_id,
                    schema_version=1,
                    input=candidate.input,
                    thought="",
                    output=candidate.output,
                    provenance=candidate.provenance,
                    inclusion_reason=inclusion_reason,
                    consent=candidate.consent,
                    privacy=candidate.privacy,
                    disposition=disposition,
                    split=split,
                    content_hash=content_hash,
                    quarantine_reasons=tuple(quarantine),
                    exclusion_reasons=tuple(exclusions),
                    quality_checks=(
                        "sensitive_content",
                        "exact_duplicate",
                        "near_duplicate",
                        "poisoning",
                        "context_bias",
                        "interlocutor_bias",
                    ),
                    context_id=candidate.context_id,
                    interlocutor_id=candidate.interlocutor_id,
                )
            )
        if not any(record.split == DatasetSplit.TRAIN for record in records):
            first_included = next(
                (
                    record
                    for record in records
                    if record.disposition == DatasetDisposition.INCLUDED
                    and record.content_hash not in previous_assignments
                ),
                None,
            )
            if first_included is not None:
                records[records.index(first_included)] = replace(
                    first_included, split=DatasetSplit.TRAIN
                )
        return records

    def _exclusion_reasons(self, candidate: DatasetCandidate) -> list[str]:
        reasons: list[str] = []
        lowered_tags = {item.lower().replace("_", "-") for item in candidate.tags}
        if not candidate.training_included or {"do-not-train", "exclude-from-training"} & lowered_tags:
            reasons.append("do_not_train")
        if candidate.validation_status == ValidationStatus.REJECTED.value:
            reasons.append("rejected_source")
        if candidate.lifecycle_status == MemoryLifecycleStatus.REJECTED.value:
            reasons.append("rejected_source")
        if candidate.privacy.lower() == "private":
            reasons.append("private_record")
        if candidate.consent.lower() in {"denied", "withdrawn", "do_not_train", "none"}:
            reasons.append("training_consent_not_granted")
        return sorted(set(reasons))

    def _quarantine_reasons(
        self,
        candidate: DatasetCandidate,
        content_hash: str,
        accepted_content: list[tuple[str, str]],
    ) -> list[str]:
        text = candidate.input + "\n" + candidate.output
        try:
            reasons = list(self.sensitive_scanner(text))
        except Exception:
            return ["sensitive_scanner_failure"]
        normalized = _normalize(text)
        if any(content_hash == existing_hash for existing_hash, _ in accepted_content):
            reasons.append("exact_duplicate")
        elif any(
            SequenceMatcher(None, normalized, existing).ratio()
            >= self.near_duplicate_threshold
            for _digest, existing in accepted_content
        ):
            reasons.append("near_duplicate")
        if _looks_poisoned(text):
            reasons.append("possible_training_poisoning")
        return sorted(set(reasons))

    def _previous_assignments(
        self,
    ) -> tuple[
        dict[str, DatasetSplit], set[tuple[str, str]], dict[str, str]
    ]:
        assignments: dict[str, DatasetSplit] = {}
        sources: set[tuple[str, str]] = set()
        content: dict[str, str] = {}
        for manifest in self.list_revisions():
            revision = self.get_revision(str(manifest["revision"]))
            for record in revision.records:
                if record.split is None:
                    continue
                sources.add(
                    (record.provenance.source_kind, record.provenance.source_id)
                )
                content[record.content_hash] = _normalize(
                    record.input + "\n" + record.output
                )
                existing = assignments.setdefault(record.content_hash, record.split)
                if existing != record.split:
                    raise ValueError("Stored dataset revisions contain cross-split duplicates")
        return assignments, sources, content

    def _reject_cross_split_duplicates(
        self, records: Iterable[GovernedDatasetRecord]
    ) -> None:
        assignments: dict[str, DatasetSplit] = {}
        for record in records:
            if record.disposition != DatasetDisposition.INCLUDED or record.split is None:
                continue
            existing = assignments.setdefault(record.content_hash, record.split)
            if existing != record.split:
                raise ValueError("Duplicate record assigned across dataset splits")

    def _bias_findings(self, records: Iterable[GovernedDatasetRecord]) -> list[str]:
        included = [
            record for record in records if record.disposition == DatasetDisposition.INCLUDED
        ]
        findings: list[str] = []
        for field_name in ("context_id", "interlocutor_id"):
            values = [getattr(record, field_name) for record in included]
            values = [value for value in values if value]
            if len(values) < 5:
                continue
            value, count = Counter(values).most_common(1)[0]
            if count / len(values) > 0.8:
                findings.append(
                    f"{field_name.removesuffix('_id')}_bias:{value}:{count}/{len(values)}"
                )
        return findings

    def _revision_files(
        self, manifest: dict[str, Any], records: list[GovernedDatasetRecord]
    ) -> dict[str, bytes]:
        files = {
            "manifest.json": _json_bytes(manifest),
            "records.jsonl": b"".join(
                _json_bytes(record.to_json(), newline=True) for record in records
            ),
        }
        for split in DatasetSplit:
            files[f"{split.value}.jsonl"] = b"".join(
                _json_bytes(record.training_json(), newline=True)
                for record in records
                if record.disposition == DatasetDisposition.INCLUDED
                and record.split == split
            )
        return files

    def _atomic_finalize(self, final_path: Path, files: dict[str, bytes]) -> None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staging = final_path.parent / f".{final_path.name}.{uuid4()}.tmp"
        staging.mkdir()
        try:
            checksums = []
            for name, content in sorted(files.items()):
                path = staging / name
                with path.open("xb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                checksums.append(f"{_sha256(content)}  {name}\n")
            with (staging / "checksums.sha256").open("x", encoding="ascii") as output:
                output.writelines(checksums)
                output.flush()
                os.fsync(output.fileno())
            os.rename(staging, final_path)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _validate_checksums(self, path: Path) -> None:
        lines = (path / "checksums.sha256").read_text("ascii").splitlines()
        expected: dict[str, str] = {}
        for line in lines:
            digest, separator, name = line.partition("  ")
            if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Invalid dataset checksum manifest")
            expected[name] = digest
        actual_names = {
            item.name for item in path.iterdir() if item.name != "checksums.sha256"
        }
        if set(expected) != actual_names:
            raise ValueError("Dataset checksum inventory mismatch")
        for name, digest in expected.items():
            if _sha256((path / name).read_bytes()) != digest:
                raise ValueError(f"Dataset checksum mismatch: {name}")


def candidate_from_episode(episode: EpisodicMemoryRecord) -> DatasetCandidate:
    metadata = episode.metadata
    operator = episode.operator_metadata
    consent = str(operator.get("training_consent", metadata.get("training_consent", "runtime_training_allowed")))
    privacy = str(operator.get("privacy", metadata.get("privacy", "internal")))
    decision_ids = _string_tuple(metadata.get("decision_ids") or metadata.get("decision_id"))
    feedback_ids = tuple(
        sorted(
            set(episode.training_exclusion_refs)
            | set(_string_tuple(metadata.get("source_feedback_ids")))
            | set(_string_tuple(metadata.get("feedback_id")))
        )
    )
    interlocutor = metadata.get("interlocutor_id") or metadata.get("interlocutor_key")
    return DatasetCandidate(
        input=episode.user_input,
        output=episode.response,
        provenance=DatasetProvenance(
            source_kind="verified_episode",
            source_id=episode.id,
            source_event_ids=(() if episode.source_event_id is None else (episode.source_event_id,)),
            source_memory_ids=(episode.id,),
            source_decision_ids=decision_ids,
            source_feedback_ids=feedback_ids,
        ),
        consent=consent,
        privacy=privacy,
        training_included=episode.training_included,
        validation_status=episode.validation_status.value,
        lifecycle_status=episode.lifecycle_status.value,
        context_id=episode.context_id,
        interlocutor_id=None if interlocutor is None else str(interlocutor),
        tags=tuple(episode.tags),
        exclusion_refs=tuple(episode.training_exclusion_refs),
    )


def candidate_from_dream_json(value: dict[str, Any]) -> DatasetCandidate:
    source_id = str(value.get("source_id", ""))
    provenance = value.get("provenance")
    if isinstance(provenance, dict):
        dataset_provenance = DatasetProvenance(
            source_kind=str(provenance.get("source_kind", "verified_episode")),
            source_id=str(provenance.get("source_id", source_id)),
            source_event_ids=_string_tuple(provenance.get("source_event_ids")),
            source_memory_ids=_string_tuple(provenance.get("source_memory_ids")),
            source_decision_ids=_string_tuple(provenance.get("source_decision_ids")),
            source_feedback_ids=_string_tuple(provenance.get("source_feedback_ids")),
        )
    else:
        dataset_provenance = DatasetProvenance(
            source_kind=str(value.get("source_kind", "verified_episode")),
            source_id=source_id
            or _content_hash(str(value.get("input", "")), str(value.get("output", ""))),
            source_memory_ids=(() if not source_id else (source_id,)),
        )
    return DatasetCandidate(
        input=str(value.get("input", "")),
        output=str(value.get("output", "")),
        thought="",
        provenance=dataset_provenance,
        consent=str(value.get("consent", "runtime_training_allowed")),
        privacy=str(value.get("privacy", "internal")),
        validation_status=str(value.get("validation_status", "verified")),
    )


def detect_sensitive_content(text: str) -> list[str]:
    """Conservative local scanner; uncertain scanner execution is handled fail-closed."""

    patterns = {
        "credential": r"(?i)\b(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*[^\s]{4,}",
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "email_pii": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone_pii": r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)",
        "government_id_pii": r"\b\d{3}-\d{2}-\d{4}\b",
        "common_secret_token": r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,})\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]


def _looks_poisoned(text: str) -> bool:
    normalized = _normalize(text)
    instructions = (
        "ignore previous instructions",
        "ignore all previous",
        "system prompt",
        "training data poisoning",
        "always respond with",
        "you are now",
    )
    if any(item in normalized for item in instructions):
        return True
    words = normalized.split()
    return len(words) >= 20 and len(set(words)) / len(words) < 0.2


def _stable_split(content_hash: str) -> DatasetSplit:
    bucket = int(content_hash[:8], 16) % 100
    if bucket < 80:
        return DatasetSplit.TRAIN
    if bucket < 90:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def _content_hash(input_text: str, output_text: str) -> str:
    return _sha256((_normalize(input_text) + "\0" + _normalize(output_text)).encode())


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + suffix).encode()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _record_from_json(value: dict[str, Any]) -> GovernedDatasetRecord:
    value = dict(value)
    provenance_value = value.pop("provenance")
    provenance = DatasetProvenance(
        source_kind=str(provenance_value["source_kind"]),
        source_id=str(provenance_value["source_id"]),
        source_event_ids=_string_tuple(provenance_value.get("source_event_ids")),
        source_memory_ids=_string_tuple(provenance_value.get("source_memory_ids")),
        source_decision_ids=_string_tuple(
            provenance_value.get("source_decision_ids")
        ),
        source_feedback_ids=_string_tuple(
            provenance_value.get("source_feedback_ids")
        ),
    )
    disposition = DatasetDisposition(value.pop("disposition"))
    split_value = value.pop("split", None)
    quarantine_reasons = tuple(value.pop("quarantine_reasons", ()))
    exclusion_reasons = tuple(value.pop("exclusion_reasons", ()))
    quality_checks = tuple(value.pop("quality_checks", ()))
    return GovernedDatasetRecord(
        **value,
        provenance=provenance,
        disposition=disposition,
        split=None if split_value is None else DatasetSplit(split_value),
        quarantine_reasons=quarantine_reasons,
        exclusion_reasons=exclusion_reasons,
        quality_checks=quality_checks,
    )
