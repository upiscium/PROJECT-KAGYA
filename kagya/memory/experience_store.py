"""Private, append-only durable storage for Memory-owned Experience records."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from uuid import UUID, uuid4

from kagya.experience import (
    EXPERIENCE_MAX_REVISION,
    ExperienceAppraisalEvidence,
    ExperienceAppraisalReasonCode,
    ExperienceArousalContributions,
    ExperienceEmotionContributions,
    ExperienceEmotionProjection,
    ExperienceEmotionUpdateReasonCode,
    ExperienceLifecycle,
    ExperienceMeasurementEvidence,
    ExperienceMeasurementInvalidReason,
    ExperienceRecord,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
    ExperienceValenceContributions,
    experience_record_digest,
)
from kagya.identifiers import validate_identifier


EXPERIENCE_STORE_SCHEMA_VERSION = 2
EXPERIENCE_PENDING_SCHEMA_VERSION = 1
EXPERIENCE_RECEIPT_SCHEMA_VERSION = 1
EXPERIENCE_MAX_RECEIPTS = 1024
EXPERIENCE_MAX_FILE_BYTES = 4 * 1024 * 1024
_REVISION_NAME = re.compile(r"(?:0|[1-9][0-9]*)\.json\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TEMP_NAME = re.compile(
    r"\.(?P<target>[A-Za-z0-9_.-]+)\.publish-(?P<token>[0-9a-f-]{36})\.tmp\Z"
)
_COMPACTION_NAME = ".compaction.json"


class ExperienceStoreError(RuntimeError):
    """Base error for unavailable or inconsistent Experience storage."""


class ExperienceStoreUnavailable(ExperienceStoreError):
    """The private filesystem authority cannot currently be used."""


class ExperienceStoreCorrupt(ExperienceStoreError):
    """Stored bytes are malformed, unsafe, or inconsistent."""


class ExperienceStoreConflict(ExperienceStoreError):
    """An immutable artifact conflicts with the requested operation."""


class ExperienceStoredEntry:
    """Verified current record plus the operation evidence needed for recovery."""

    __slots__ = (
        "record",
        "operation_digest",
        "source_episode_operation_digest",
        "revision_evidence",
        "previous_record_digest",
        "anchor_revision",
        "anchor_record_digest",
    )

    def __init__(
        self,
        record: ExperienceRecord,
        operation_digest: str,
        source_episode_operation_digest: str,
        revision_evidence: ExperienceRevisionRecord | None = None,
        previous_record_digest: str | None = None,
        anchor_revision: int | None = None,
        anchor_record_digest: str | None = None,
    ) -> None:
        self.record = record
        self.operation_digest = operation_digest
        self.source_episode_operation_digest = source_episode_operation_digest
        self.revision_evidence = revision_evidence
        self.previous_record_digest = previous_record_digest
        self.anchor_revision = anchor_revision
        self.anchor_record_digest = anchor_record_digest


def _datetime_value(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ExperienceStoreCorrupt("Experience timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ExperienceStoreCorrupt("Experience timestamp is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExperienceStoreCorrupt("Experience timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _float_value(value: float) -> str:
    return value.hex()


def _parse_float(value: object) -> float:
    if not isinstance(value, str):
        raise ExperienceStoreCorrupt("Experience number is invalid")
    try:
        return float.fromhex(value)
    except ValueError:
        raise ExperienceStoreCorrupt("Experience number is invalid") from None


def _require_keys(value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ExperienceStoreCorrupt("Experience artifact shape is invalid")
    return value


def _digest_value(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ExperienceStoreCorrupt(f"{name} is invalid")
    return value


def _revision_to_dict(record: ExperienceRevisionRecord) -> dict[str, object]:
    return {
        "created_at": _datetime_value(record.created_at),
        "event_id": record.event_id,
        "event_sequence": record.event_sequence,
        "evidence_refs": list(record.evidence_refs),
        "experience_id": record.experience_id,
        "operation": record.operation.value,
        "previous_revision_digest": record.previous_revision_digest,
        "reason": record.reason.value,
        "record_digest": record.record_digest,
        "revision": record.revision,
    }


def _revision_from_dict(value: object) -> ExperienceRevisionRecord:
    payload = _require_keys(
        value,
        {
            "created_at",
            "event_id",
            "event_sequence",
            "evidence_refs",
            "experience_id",
            "operation",
            "previous_revision_digest",
            "reason",
            "record_digest",
            "revision",
        },
    )
    try:
        record = ExperienceRevisionRecord(
            experience_id=payload["experience_id"],
            revision=payload["revision"],
            operation=ExperienceRevisionOperation(payload["operation"]),
            reason=ExperienceRevisionReason(payload["reason"]),
            created_at=_parse_datetime(payload["created_at"]),
            event_id=payload["event_id"],
            event_sequence=payload["event_sequence"],
            evidence_refs=tuple(payload["evidence_refs"]),
            previous_revision_digest=payload["previous_revision_digest"],
        )
    except (TypeError, ValueError, KeyError):
        raise ExperienceStoreCorrupt("Experience revision is invalid") from None
    if payload["record_digest"] != record.record_digest:
        raise ExperienceStoreCorrupt("Experience revision digest is invalid")
    return record


def experience_record_to_dict(record: ExperienceRecord) -> dict[str, object]:
    """Serialize only the bounded U1 Experience contract, never request text."""

    if not isinstance(record, ExperienceRecord):
        raise TypeError("record must be ExperienceRecord")
    measurement = record.measurement
    appraisal = record.appraisal
    valence = record.emotion_contributions.valence_contributions
    arousal = record.emotion_contributions.arousal_contributions
    return {
        "appraisal": {
            "certainty": None if appraisal.certainty is None else _float_value(appraisal.certainty),
            "controllability": None
            if appraisal.controllability is None
            else _float_value(appraisal.controllability),
            "effort_cost": None
            if appraisal.effort_cost is None
            else _float_value(appraisal.effort_cost),
            "goal_progress": None
            if appraisal.goal_progress is None
            else _float_value(appraisal.goal_progress),
            "novelty": None if appraisal.novelty is None else _float_value(appraisal.novelty),
            "novelty_valid": appraisal.novelty_valid,
            "reason_codes": [item.value for item in appraisal.reason_codes],
            "social_relevance": None
            if appraisal.social_relevance is None
            else _float_value(appraisal.social_relevance),
            "threat": None if appraisal.threat is None else _float_value(appraisal.threat),
        },
        "context_id": record.context_id,
        "created_at": _datetime_value(record.created_at),
        "emotion_contributions": {
            "arousal": {
                "low_controllability": _float_value(arousal.low_controllability),
                "effort_cost": _float_value(arousal.effort_cost),
                "novelty": _float_value(arousal.novelty),
                "social_relevance": _float_value(arousal.social_relevance),
                "threat": _float_value(arousal.threat),
                "uncertainty": _float_value(arousal.uncertainty),
            },
            "valence": {
                "controllability": _float_value(valence.controllability),
                "effort_cost": _float_value(valence.effort_cost),
                "goal_progress": _float_value(valence.goal_progress),
                "threat": _float_value(valence.threat),
            },
        },
        "emotion_update_reasons": [item.value for item in record.emotion_update_reasons],
        "experience_id": record.experience_id,
        "history_anchor_digest": record.history_anchor_digest,
        "history_anchor_revision": record.history_anchor_revision,
        "lifecycle": record.lifecycle.value,
        "measurement": {
            "calibrated_novelty": None
            if measurement.calibrated_novelty is None
            else _float_value(measurement.calibrated_novelty),
            "invalid_reason": None
            if measurement.invalid_reason is None
            else measurement.invalid_reason.value,
            "model_key": measurement.model_key,
            "valid": measurement.valid,
        },
        "post_appraisal_emotion": {
            "arousal": _float_value(record.post_appraisal_emotion.arousal),
            "valence": _float_value(record.post_appraisal_emotion.valence),
        },
        "pre_appraisal_emotion": {
            "arousal": _float_value(record.pre_appraisal_emotion.arousal),
            "valence": _float_value(record.pre_appraisal_emotion.valence),
        },
        "revision": record.revision,
        "revision_history": [_revision_to_dict(item) for item in record.revision_history],
        "schema_version": record.schema_version,
        "source_episode_id": record.source_episode_id,
        "source_event_id": record.source_event_id,
        "source_event_sequence": record.source_event_sequence,
        "subjective_salience": _float_value(record.subjective_salience),
        "superseded_by_id": record.superseded_by_id,
        "temporal_update_reasons": [item.value for item in record.temporal_update_reasons],
    }


def experience_record_from_dict(value: object) -> ExperienceRecord:
    payload = _require_keys(
        value,
        {
            "appraisal",
            "context_id",
            "created_at",
            "emotion_contributions",
            "emotion_update_reasons",
            "experience_id",
            "history_anchor_digest",
            "history_anchor_revision",
            "lifecycle",
            "measurement",
            "post_appraisal_emotion",
            "pre_appraisal_emotion",
            "revision",
            "revision_history",
            "schema_version",
            "source_episode_id",
            "source_event_id",
            "source_event_sequence",
            "subjective_salience",
            "superseded_by_id",
            "temporal_update_reasons",
        },
    )
    measurement = _require_keys(
        payload["measurement"], {"calibrated_novelty", "invalid_reason", "model_key", "valid"}
    )
    appraisal = _require_keys(
        payload["appraisal"],
        {
            "certainty",
            "controllability",
            "effort_cost",
            "goal_progress",
            "novelty",
            "novelty_valid",
            "reason_codes",
            "social_relevance",
            "threat",
        },
    )
    contributions = _require_keys(
        payload["emotion_contributions"], {"arousal", "valence"}
    )
    arousal = _require_keys(
        contributions["arousal"],
        {
            "low_controllability",
            "effort_cost",
            "novelty",
            "social_relevance",
            "threat",
            "uncertainty",
        },
    )
    valence = _require_keys(
        contributions["valence"],
        {"controllability", "effort_cost", "goal_progress", "threat"},
    )

    def optional_float(item: object) -> float | None:
        return None if item is None else _parse_float(item)

    try:
        result = ExperienceRecord(
            experience_id=payload["experience_id"],
            revision=payload["revision"],
            lifecycle=ExperienceLifecycle(payload["lifecycle"]),
            source_event_id=payload["source_event_id"],
            source_event_sequence=payload["source_event_sequence"],
            source_episode_id=payload["source_episode_id"],
            context_id=payload["context_id"],
            measurement=ExperienceMeasurementEvidence(
                model_key=measurement["model_key"],
                valid=measurement["valid"],
                invalid_reason=(
                    None
                    if measurement["invalid_reason"] is None
                    else ExperienceMeasurementInvalidReason(measurement["invalid_reason"])
                ),
                calibrated_novelty=optional_float(measurement["calibrated_novelty"]),
            ),
            appraisal=ExperienceAppraisalEvidence(
                novelty=optional_float(appraisal["novelty"]),
                novelty_valid=appraisal["novelty_valid"],
                goal_progress=optional_float(appraisal["goal_progress"]),
                threat=optional_float(appraisal["threat"]),
                controllability=optional_float(appraisal["controllability"]),
                certainty=optional_float(appraisal["certainty"]),
                social_relevance=optional_float(appraisal["social_relevance"]),
                effort_cost=optional_float(appraisal["effort_cost"]),
                reason_codes=tuple(
                    ExperienceAppraisalReasonCode(item)
                    for item in appraisal["reason_codes"]
                ),
            ),
            pre_appraisal_emotion=ExperienceEmotionProjection(
                valence=_parse_float(
                    _require_keys(payload["pre_appraisal_emotion"], {"arousal", "valence"})[
                        "valence"
                    ]
                ),
                arousal=_parse_float(
                    _require_keys(payload["pre_appraisal_emotion"], {"arousal", "valence"})[
                        "arousal"
                    ]
                ),
            ),
            temporal_update_reasons=tuple(
                ExperienceEmotionUpdateReasonCode(item)
                for item in payload["temporal_update_reasons"]
            ),
            post_appraisal_emotion=ExperienceEmotionProjection(
                valence=_parse_float(
                    _require_keys(payload["post_appraisal_emotion"], {"arousal", "valence"})[
                        "valence"
                    ]
                ),
                arousal=_parse_float(
                    _require_keys(payload["post_appraisal_emotion"], {"arousal", "valence"})[
                        "arousal"
                    ]
                ),
            ),
            emotion_contributions=ExperienceEmotionContributions(
                valence_contributions=ExperienceValenceContributions(
                    goal_progress=_parse_float(valence["goal_progress"]),
                    threat=_parse_float(valence["threat"]),
                    effort_cost=_parse_float(valence["effort_cost"]),
                    controllability=_parse_float(valence["controllability"]),
                ),
                arousal_contributions=ExperienceArousalContributions(
                    novelty=_parse_float(arousal["novelty"]),
                    threat=_parse_float(arousal["threat"]),
                    effort_cost=_parse_float(arousal["effort_cost"]),
                    social_relevance=_parse_float(arousal["social_relevance"]),
                    uncertainty=_parse_float(arousal["uncertainty"]),
                    low_controllability=_parse_float(arousal["low_controllability"]),
                ),
            ),
            emotion_update_reasons=tuple(
                ExperienceEmotionUpdateReasonCode(item)
                for item in payload["emotion_update_reasons"]
            ),
            subjective_salience=_parse_float(payload["subjective_salience"]),
            created_at=_parse_datetime(payload["created_at"]),
            schema_version=payload["schema_version"],
            superseded_by_id=payload["superseded_by_id"],
            revision_history=tuple(
                _revision_from_dict(item) for item in payload["revision_history"]
            ),
            history_anchor_digest=payload["history_anchor_digest"],
            history_anchor_revision=payload["history_anchor_revision"],
        )
    except (TypeError, ValueError, KeyError, ExperienceStoreCorrupt):
        raise ExperienceStoreCorrupt("Experience record is invalid") from None
    return result


class ExperienceStore:
    """Own Experience payloads while exposing only verified bounded records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def from_memory_root(cls, memory_root: str | Path) -> ExperienceStore:
        return cls(Path(memory_root).parent / "experience")

    @property
    def records_root(self) -> Path:
        return self.root / "records"

    @property
    def pending_root(self) -> Path:
        return self.root / "pending"

    def pending_path(self, transaction_id: str) -> Path:
        self._validate_uuid(transaction_id)
        return self.pending_root / f"{transaction_id}.json"

    @property
    def receipts_root(self) -> Path:
        return self.root / "receipts"

    def receipt_path(self, transaction_id: str) -> Path:
        self._validate_uuid(transaction_id)
        return self.receipts_root / f"{transaction_id}.json"

    def record_path(self, experience_id: str, revision: int) -> Path:
        self._validate_identifier(experience_id)
        if type(revision) is not int or not 0 <= revision <= EXPERIENCE_MAX_REVISION:
            raise ValueError("Experience revision is invalid")
        return self.records_root / experience_id / f"{revision}.json"

    def load_pending(self, transaction_id: str) -> dict[str, object] | None:
        path = self.pending_path(transaction_id)
        return self._read_json(path, missing_ok=True)

    def write_pending(self, transaction_id: str, payload: dict[str, object]) -> None:
        path = self.pending_path(transaction_id)
        self._write_immutable_json(path, payload)

    def remove_pending(self, transaction_id: str) -> None:
        path = self.pending_path(transaction_id)
        directory_fd = -1
        try:
            self._secure_parent(self.pending_root, create=False)
            directory_fd = self._directory_fd(self.pending_root)
            os.unlink(path.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            return
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience pending storage is unavailable") from error
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)

    def load_receipt(self, transaction_id: str) -> dict[str, object] | None:
        """Load immutable Memory-owned operation evidence retained after commit."""

        return self._read_json(self.receipt_path(transaction_id), missing_ok=True)

    def write_receipt(self, transaction_id: str, payload: dict[str, object]) -> None:
        """Durably retain a bounded operation receipt before publication."""

        path = self.receipt_path(transaction_id)
        self._write_immutable_json(path, payload)

    def remove_receipt(self, transaction_id: str) -> None:
        path = self.receipt_path(transaction_id)
        if not self._path_exists(self.receipts_root):
            return
        directory_fd = -1
        try:
            self._secure_parent(self.receipts_root, create=False)
            directory_fd = self._directory_fd(self.receipts_root)
            os.unlink(path.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            return
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience receipt storage is unavailable") from error
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)

    def prune_receipts(self, protected_transaction_id: str | None = None) -> None:
        """Bound receipts while retaining only evidence of committed operations."""

        if not self._path_exists(self.receipts_root):
            return
        self._secure_parent(self.receipts_root, create=False)
        try:
            paths = tuple(self.receipts_root.iterdir())
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience receipts are unavailable") from error
        receipts: list[tuple[str, Path]] = []
        for path in paths:
            if path.suffix != ".json" or _REVISION_NAME.fullmatch(path.name) is not None:
                # Receipt filenames are UUIDs, not revision numbers.  Any other
                # directory entry is an untrusted artifact.
                if path.name != _COMPACTION_NAME:
                    raise ExperienceStoreCorrupt("Experience receipt file name is invalid")
                raise ExperienceStoreCorrupt("Experience receipt directory is invalid")
            transaction_id = path.stem
            try:
                parsed = UUID(transaction_id)
            except (TypeError, ValueError):
                raise ExperienceStoreCorrupt("Experience receipt file name is invalid") from None
            if str(parsed) != transaction_id:
                raise ExperienceStoreCorrupt("Experience receipt file name is invalid")
            self._secure_file(path)
            receipts.append((transaction_id, path))
        if len(receipts) <= EXPERIENCE_MAX_RECEIPTS:
            return
        for transaction_id, path in sorted(receipts):
            if len(receipts) <= EXPERIENCE_MAX_RECEIPTS:
                break
            if transaction_id == protected_transaction_id:
                continue
            if self._receipt_has_pending(transaction_id):
                continue
            if self._receipt_is_committed(path):
                self._unlink_receipt(path)
            else:
                # A receipt with neither a pending artifact nor a matching
                # current record is an aborted/stale preparation artifact.
                self._unlink_receipt(path)
            receipts = [item for item in receipts if item[0] != transaction_id]

    def _receipt_has_pending(self, transaction_id: str) -> bool:
        pending = self.pending_path(transaction_id)
        if not self._path_exists(pending):
            return False
        self._read_json(pending)
        return True

    def _receipt_is_committed(self, path: Path) -> bool:
        payload = self._read_json(path)
        value = _require_keys(
            payload,
            {
                "operation",
                "operation_digest",
                "participant_id",
                "schema_version",
                "transaction_id",
            },
        )
        if value["schema_version"] != EXPERIENCE_RECEIPT_SCHEMA_VERSION:
            raise ExperienceStoreCorrupt("Experience receipt schema is unsupported")
        operation = value["operation"]
        if not isinstance(operation, dict) or not isinstance(operation.get("record"), dict):
            raise ExperienceStoreCorrupt("Experience receipt operation is invalid")
        record = experience_record_from_dict(operation["record"])
        current = self.load_revision(record.experience_id, record.revision)
        return bool(
            current is not None
            and current.record == record
            and current.operation_digest == value["operation_digest"]
            and operation.get("source_episode_operation_digest")
            == current.source_episode_operation_digest
        )

    def _unlink_receipt(self, path: Path) -> None:
        self._secure_file(path)
        try:
            path.unlink()
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience receipt cleanup is unavailable") from error
        self._fsync_directory(path.parent)

    def load_current(self, experience_id: str) -> ExperienceStoredEntry | None:
        self._validate_identifier(experience_id)
        directory = self.records_root / experience_id
        if not self._path_exists(directory):
            return None
        self._secure_parent(directory, create=False)
        self._recover_publication_temps(directory)
        self._recover_compaction(directory, experience_id)
        entries = self._load_entries(directory, experience_id)
        if not entries:
            raise ExperienceStoreCorrupt("Experience record directory is empty")
        self._validate_revision_window(entries)
        return entries[max(entries)]

    def _load_entries(
        self, directory: Path, experience_id: str
    ) -> dict[int, ExperienceStoredEntry]:
        try:
            paths = tuple(directory.iterdir())
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience records are unavailable") from error
        result: dict[int, ExperienceStoredEntry] = {}
        for path in paths:
            if path.name == _COMPACTION_NAME:
                continue
            if _TEMP_NAME.fullmatch(path.name) is not None:
                raise ExperienceStoreCorrupt("Unreconciled Experience publication temp remains")
            if path.suffix != ".json" or _REVISION_NAME.fullmatch(path.name) is None:
                raise ExperienceStoreCorrupt("Experience revision file name is invalid")
            try:
                revision = int(path.stem)
            except ValueError:
                raise ExperienceStoreCorrupt("Experience revision file name is invalid") from None
            if revision in result:
                raise ExperienceStoreCorrupt("Duplicate Experience revision artifact")
            payload = self._read_json(path)
            result[revision] = self._entry_from_payload(payload, experience_id, revision)
        return result

    def _validate_revision_window(
        self, entries: dict[int, ExperienceStoredEntry]
    ) -> None:
        if not entries:
            return
        current_revision = max(entries)
        actual = set(entries)
        if current_revision < 32:
            expected = set(range(current_revision + 1))
        else:
            floor = current_revision - 32
            expected = set(range(floor, current_revision + 1))
            full = set(range(current_revision + 1))
            previous_window = set(range(max(0, floor - 1), current_revision + 1))
            if actual not in (expected, full, previous_window):
                raise ExperienceStoreCorrupt("Experience revision window is incomplete")
        if current_revision < 32 and actual != expected:
            raise ExperienceStoreCorrupt("Experience revision window is incomplete")
        for revision, entry in entries.items():
            evidence = entry.revision_evidence
            if evidence is None or evidence != entry.record.revision_history[-1]:
                raise ExperienceStoreCorrupt("Experience operation evidence is inconsistent")
            if revision == 0:
                if entry.previous_record_digest is not None:
                    raise ExperienceStoreCorrupt("Genesis artifact has a prior record")
                continue
            previous = entries.get(revision - 1)
            if previous is not None:
                if entry.previous_record_digest != experience_record_digest(previous.record):
                    raise ExperienceStoreCorrupt("Experience record chain is broken")
            elif _DIGEST.fullmatch(entry.previous_record_digest or "") is None:
                raise ExperienceStoreCorrupt("Compacted Experience predecessor is invalid")
        current = entries[current_revision]
        if current_revision >= 32:
            floor = current_revision - 32
            if current.record.history_anchor_revision != floor:
                raise ExperienceStoreCorrupt("Experience history anchor revision is invalid")
            if current.anchor_revision != floor:
                raise ExperienceStoreCorrupt("Experience artifact anchor revision is invalid")
            anchor = entries[floor]
            if current.anchor_record_digest != experience_record_digest(anchor.record):
                raise ExperienceStoreCorrupt("Experience artifact anchor is invalid")
            if (
                anchor.revision_evidence is None
                or current.record.history_anchor_digest
                != anchor.revision_evidence.record_digest
            ):
                raise ExperienceStoreCorrupt("Experience evidence anchor is invalid")
        elif any(entry.anchor_revision is not None for entry in entries.values()):
            raise ExperienceStoreCorrupt("Unexpected Experience history anchor")

    def _recover_publication_temps(self, directory: Path) -> None:
        try:
            paths = tuple(directory.iterdir())
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience records are unavailable") from error
        for temporary in paths:
            match = _TEMP_NAME.fullmatch(temporary.name)
            if match is None:
                continue
            target_name = match.group("target")
            token = match.group("token")
            try:
                parsed_token = UUID(token)
            except (TypeError, ValueError):
                raise ExperienceStoreCorrupt("Experience publication temp token is invalid") from None
            if str(parsed_token) != token:
                raise ExperienceStoreCorrupt("Experience publication temp token is invalid")
            if target_name != _COMPACTION_NAME and _REVISION_NAME.fullmatch(target_name) is None:
                raise ExperienceStoreCorrupt("Experience publication temp target is invalid")
            self._secure_file(temporary)
            target = directory / target_name
            temporary_payload = self._read_json(temporary)
            if self._path_exists(target):
                target_payload = self._read_json(target)
                if target_payload != temporary_payload:
                    raise ExperienceStoreCorrupt(
                        "Experience publication temp conflicts with its final artifact"
                    )
            try:
                temporary.unlink()
            except OSError as error:
                raise ExperienceStoreUnavailable(
                    "Experience publication temp cleanup is unavailable"
                ) from error
        if any(_TEMP_NAME.fullmatch(path.name) is not None for path in paths):
            self._fsync_directory(directory)

    def _recover_compaction(self, directory: Path, experience_id: str) -> None:
        marker = directory / _COMPACTION_NAME
        if not self._path_exists(marker):
            return
        payload = self._read_json(marker)
        value = _require_keys(
            payload,
            {
                "anchor_digest",
                "anchor_record_digest",
                "anchor_revision",
                "current_record_digest",
                "current_revision",
                "delete_revisions",
                "experience_id",
                "floor",
                "schema_version",
            },
        )
        if value["schema_version"] != 1 or value["experience_id"] != experience_id:
            raise ExperienceStoreCorrupt("Experience compaction marker is unsupported")
        try:
            current_revision = value["current_revision"]
            floor = value["floor"]
            anchor_revision = value["anchor_revision"]
            delete_revisions = value["delete_revisions"]
            if (
                type(current_revision) is not int
                or type(floor) is not int
                or type(anchor_revision) is not int
                or type(delete_revisions) is not list
                or delete_revisions != list(range(floor))
                or floor != current_revision - 32
                or anchor_revision != floor
            ):
                raise ValueError
            anchor_digest = _digest_value(value["anchor_digest"], "anchor_digest")
            anchor_record_digest = _digest_value(
                value["anchor_record_digest"], "anchor_record_digest"
            )
            current_record_digest = _digest_value(
                value["current_record_digest"], "current_record_digest"
            )
        except (TypeError, ValueError, ExperienceStoreCorrupt):
            raise ExperienceStoreCorrupt("Experience compaction marker is invalid") from None
        current_path = directory / f"{current_revision}.json"
        current_payload = self._read_json(current_path)
        current = self._entry_from_payload(current_payload, experience_id, current_revision)
        if (
            experience_record_digest(current.record) != current_record_digest
            or current.record.history_anchor_revision != anchor_revision
            or current.record.history_anchor_digest != anchor_digest
            or current.anchor_record_digest != anchor_record_digest
        ):
            raise ExperienceStoreCorrupt("Experience compaction marker conflicts")
        for revision in delete_revisions:
            path = directory / f"{revision}.json"
            try:
                self._secure_file(path)
            except FileNotFoundError:
                continue
            try:
                path.unlink()
            except OSError as error:
                raise ExperienceStoreUnavailable(
                    "Experience history compaction is unavailable"
                ) from error
        try:
            marker.unlink()
        except OSError as error:
            raise ExperienceStoreUnavailable(
                "Experience compaction marker cleanup is unavailable"
            ) from error
        self._fsync_directory(directory)

    def publish_create(
        self,
        record: ExperienceRecord,
        operation_digest: str,
        source_episode_operation_digest: str,
    ) -> ExperienceStoredEntry:
        if record.revision != 0:
            raise ExperienceStoreConflict("Experience create must publish revision zero")
        entry = ExperienceStoredEntry(
            record,
            operation_digest,
            source_episode_operation_digest,
            revision_evidence=record.revision_history[-1],
        )
        self._publish_entry(entry)
        return entry

    def publish_revision(
        self,
        record: ExperienceRecord,
        operation_digest: str,
        source_episode_operation_digest: str,
        *,
        expected_revision: int,
        expected_digest: str,
    ) -> ExperienceStoredEntry:
        current = self.load_current(record.experience_id)
        if current is None:
            raise ExperienceStoreUnavailable("Experience revision target is absent")
        if (
            current.record.revision != expected_revision
            or experience_record_digest(current.record) != expected_digest
        ):
            raise ExperienceStoreConflict("Experience revision target is stale")
        if record.revision != expected_revision + 1:
            raise ExperienceStoreConflict("Experience revision is not the next revision")
        anchor_revision = None
        anchor_record_digest = None
        if record.revision >= 32:
            anchor_revision = record.revision - 32
            if record.history_anchor_revision != anchor_revision:
                raise ExperienceStoreConflict("Experience history anchor is inconsistent")
            anchor_entry = self._load_revision_entry(record.experience_id, anchor_revision)
            if anchor_entry is None:
                raise ExperienceStoreUnavailable("Experience history anchor is absent")
            anchor_record_digest = experience_record_digest(anchor_entry.record)
        entry = ExperienceStoredEntry(
            record,
            operation_digest,
            source_episode_operation_digest,
            revision_evidence=record.revision_history[-1],
            previous_record_digest=experience_record_digest(current.record),
            anchor_revision=anchor_revision,
            anchor_record_digest=anchor_record_digest,
        )
        self._publish_entry(entry)
        return entry

    def _load_revision_entry(
        self, experience_id: str, revision: int
    ) -> ExperienceStoredEntry | None:
        path = self.record_path(experience_id, revision)
        payload = self._read_json(path, missing_ok=True)
        if payload is None:
            return None
        return self._entry_from_payload(payload, experience_id, revision)

    def load_revision(
        self, experience_id: str, revision: int
    ) -> ExperienceStoredEntry | None:
        """Load one immutable revision for historical transaction recovery."""

        self._validate_identifier(experience_id)
        if type(revision) is not int or not 0 <= revision <= EXPERIENCE_MAX_REVISION:
            raise ValueError("Experience revision is invalid")
        current = self.load_current(experience_id)
        if current is None or revision > current.record.revision:
            return None
        return self._load_revision_entry(experience_id, revision)

    def reconcile_prune(self, experience_id: str) -> None:
        """Finish an interrupted safe compaction without changing authority."""

        current = self.load_current(experience_id)
        if current is not None:
            self._prune(experience_id, current.record.revision)

    def _publish_entry(self, entry: ExperienceStoredEntry) -> None:
        if _DIGEST.fullmatch(entry.operation_digest) is None or _DIGEST.fullmatch(
            entry.source_episode_operation_digest
        ) is None:
            raise ExperienceStoreConflict("Experience operation digest is invalid")
        if entry.revision_evidence is None:
            raise ExperienceStoreConflict("Experience revision evidence is missing")
        if entry.revision_evidence != entry.record.revision_history[-1]:
            raise ExperienceStoreConflict("Experience revision evidence is inconsistent")
        if entry.record.revision == 0:
            if entry.previous_record_digest is not None or entry.anchor_revision is not None:
                raise ExperienceStoreConflict("Genesis artifact has predecessor evidence")
        elif _DIGEST.fullmatch(entry.previous_record_digest or "") is None:
            raise ExperienceStoreConflict("Experience predecessor evidence is missing")
        if entry.record.revision >= 32:
            if (
                entry.anchor_revision != entry.record.history_anchor_revision
                or _DIGEST.fullmatch(entry.anchor_record_digest or "") is None
            ):
                raise ExperienceStoreConflict("Experience anchor evidence is missing")
        elif entry.anchor_revision is not None or entry.anchor_record_digest is not None:
            raise ExperienceStoreConflict("Unexpected Experience anchor evidence")
        path = self.record_path(entry.record.experience_id, entry.record.revision)
        payload: dict[str, object] = {
            "anchor_record_digest": entry.anchor_record_digest,
            "anchor_revision": entry.anchor_revision,
            "experience_id": entry.record.experience_id,
            "operation_digest": entry.operation_digest,
            "previous_record_digest": entry.previous_record_digest,
            "record": experience_record_to_dict(entry.record),
            "record_digest": experience_record_digest(entry.record),
            "revision_evidence": _revision_to_dict(entry.revision_evidence),
            "schema_version": EXPERIENCE_STORE_SCHEMA_VERSION,
            "source_episode_operation_digest": entry.source_episode_operation_digest,
        }
        self._write_immutable_json(path, payload)
        self._prune(entry.record.experience_id, entry.record.revision)

    def _prune(self, experience_id: str, current_revision: int) -> None:
        directory = self.records_root / experience_id
        self._secure_parent(directory, create=False)
        self._recover_publication_temps(directory)
        self._recover_compaction(directory, experience_id)
        entries = self._load_entries(directory, experience_id)
        if not entries or max(entries) != current_revision:
            raise ExperienceStoreUnavailable("Experience publication is not current")
        self._validate_revision_window(entries)
        if current_revision <= 32:
            return
        current = entries[current_revision]
        floor = current_revision - 32
        marker_payload: dict[str, object] = {
            "anchor_digest": current.record.history_anchor_digest,
            "anchor_record_digest": current.anchor_record_digest,
            "anchor_revision": current.anchor_revision,
            "current_record_digest": experience_record_digest(current.record),
            "current_revision": current_revision,
            "delete_revisions": list(range(floor)),
            "experience_id": experience_id,
            "floor": floor,
            "schema_version": 1,
        }
        marker = directory / _COMPACTION_NAME
        self._write_immutable_json(marker, marker_payload)
        self._recover_compaction(directory, experience_id)

    def _entry_from_payload(
        self, payload: object, experience_id: str, revision: int
    ) -> ExperienceStoredEntry:
        value = _require_keys(
            payload,
            {
                "anchor_record_digest",
                "anchor_revision",
                "experience_id",
                "operation_digest",
                "previous_record_digest",
                "record",
                "record_digest",
                "revision_evidence",
                "schema_version",
                "source_episode_operation_digest",
            },
        )
        if (
            value["schema_version"] != EXPERIENCE_STORE_SCHEMA_VERSION
            or value["experience_id"] != experience_id
        ):
            raise ExperienceStoreCorrupt("Experience store schema is unsupported")
        operation_digest = _digest_value(value["operation_digest"], "operation_digest")
        source_digest = _digest_value(
            value["source_episode_operation_digest"],
            "source_episode_operation_digest",
        )
        record = experience_record_from_dict(value["record"])
        if (
            record.experience_id != experience_id
            or record.revision != revision
            or value["record_digest"] != experience_record_digest(record)
        ):
            raise ExperienceStoreCorrupt("Experience record digest is inconsistent")
        _digest_value(value["record_digest"], "record_digest")
        anchor_revision = value["anchor_revision"]
        if anchor_revision is not None and (
            type(anchor_revision) is not int or anchor_revision < 0
        ):
            raise ExperienceStoreCorrupt("Experience anchor revision is invalid")
        anchor_record_digest = value["anchor_record_digest"]
        if anchor_record_digest is not None:
            _digest_value(anchor_record_digest, "anchor_record_digest")
        previous_record_digest = value["previous_record_digest"]
        if previous_record_digest is not None:
            _digest_value(previous_record_digest, "previous_record_digest")
        revision_evidence = _revision_from_dict(value["revision_evidence"])
        if revision_evidence != record.revision_history[-1]:
            raise ExperienceStoreCorrupt("Experience operation evidence is inconsistent")
        return ExperienceStoredEntry(
            record,
            operation_digest,
            source_digest,
            revision_evidence=revision_evidence,
            previous_record_digest=previous_record_digest,
            anchor_revision=anchor_revision,
            anchor_record_digest=anchor_record_digest,
        )

    def _write_immutable_json(self, path: Path, payload: dict[str, object]) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
        if len(encoded) > EXPERIENCE_MAX_FILE_BYTES:
            raise ExperienceStoreConflict("Experience artifact is too large")
        self._secure_parent(path.parent, create=True)
        self._recover_publication_temps(path.parent)
        existing = self._read_json(path, missing_ok=True)
        if existing is not None:
            if existing != payload:
                raise ExperienceStoreConflict("Experience artifact conflicts")
            return
        parent_fd = self._directory_fd(path.parent)
        temporary = f".{path.name}.publish-{uuid4()}.tmp"
        linked = False
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "wb") as target:
                descriptor = -1
                target.write(encoded)
                target.flush()
                os.fsync(target.fileno())
            os.link(
                temporary,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            linked = True
            os.unlink(temporary, dir_fd=parent_fd)
            temporary = ""
            os.fsync(parent_fd)
        except FileExistsError:
            existing = self._read_json(path)
            if existing != payload:
                raise ExperienceStoreConflict("Experience artifact conflicts") from None
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience atomic publication is unavailable") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary and not linked:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except OSError:
                    raise ExperienceStoreUnavailable(
                        "Experience temporary cleanup is unavailable"
                    )
            os.close(parent_fd)

    def _read_json(self, path: Path, *, missing_ok: bool = False) -> dict[str, object] | None:
        try:
            self._secure_file(path)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ExperienceStoreUnavailable("Experience artifact is absent") from None
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid():
                raise ExperienceStoreCorrupt("Experience artifact is not private")
            if stat.S_IMODE(status.st_mode) != 0o600:
                raise ExperienceStoreCorrupt("Experience artifact mode is invalid")
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                payload = source.read(EXPERIENCE_MAX_FILE_BYTES + 1)
            if len(payload) > EXPERIENCE_MAX_FILE_BYTES:
                raise ExperienceStoreCorrupt("Experience artifact is oversized")
            loaded = json.loads(payload)
            if not isinstance(loaded, dict):
                raise ExperienceStoreCorrupt("Experience artifact is not an object")
            return loaded
        except ExperienceStoreError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ExperienceStoreCorrupt("Experience artifact cannot be read") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _secure_parent(self, path: Path, *, create: bool) -> None:
        absolute = path.absolute()
        root = self.root.absolute()
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            raise ExperienceStoreCorrupt("Experience path escapes its root") from None
        try:
            for ancestor in reversed(root.parents):
                status = ancestor.lstat()
                if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                    raise ExperienceStoreCorrupt("Experience path has an unsafe parent")
            current = root
            for component in (root, *relative.parts):
                if isinstance(component, str):
                    current /= component
                try:
                    status = current.lstat()
                except FileNotFoundError:
                    if not create:
                        raise ExperienceStoreUnavailable(
                            "Experience directory is absent"
                        ) from None
                    current.mkdir(mode=0o700)
                    self._fsync_directory(current.parent)
                    self._fsync_directory(current)
                    status = current.lstat()
                if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
                    raise ExperienceStoreCorrupt("Experience directory is unsafe")
                os.chmod(current, 0o700)
                if stat.S_IMODE(current.stat().st_mode) != 0o700:
                    raise ExperienceStoreCorrupt("Experience directory mode is invalid")
        except ExperienceStoreError:
            raise
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience directory is unavailable") from error

    def _secure_file(self, path: Path) -> None:
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid():
            raise ExperienceStoreCorrupt("Experience artifact is unsafe")
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise ExperienceStoreCorrupt("Experience artifact mode is invalid")

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            path.lstat()
            return True
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience path cannot be inspected") from error

    @staticmethod
    def _validate_uuid(value: str) -> None:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError):
            raise ValueError("Experience transaction identifier is invalid") from None
        if str(parsed) != value:
            raise ValueError("Experience transaction identifier is invalid")

    @staticmethod
    def _validate_identifier(value: str) -> None:
        try:
            validate_identifier(value)
        except (TypeError, ValueError):
            raise ValueError("Experience identifier is invalid") from None

    @staticmethod
    def _directory_fd(path: Path) -> int:
        try:
            return os.open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
            )
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience directory is unavailable") from error

    @classmethod
    def _fsync_directory(cls, path: Path) -> None:
        descriptor = cls._directory_fd(path)
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise ExperienceStoreUnavailable("Experience directory cannot be synced") from error
        finally:
            os.close(descriptor)
