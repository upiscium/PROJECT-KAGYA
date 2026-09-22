"""Private durable authority for R12 Semantic lifecycle revisions.

The store owns immutable Semantic revisions and the small amount of pending and
receipt evidence needed by the R07 participant.  It intentionally knows
nothing about Chroma, prompts, model providers, or runtime state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from uuid import UUID, uuid4

from kagya.identifiers import validate_identifier
from kagya.memory.semantic_lifecycle import (
    SEMANTIC_MAX_REVISION,
    SemanticLifecycle,
    SemanticProvenanceClass,
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticRevisionReason,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    validate_revision_digest,
)


SEMANTIC_STORE_SCHEMA_VERSION = 2
SEMANTIC_PENDING_SCHEMA_VERSION = 1
SEMANTIC_RECEIPT_SCHEMA_VERSION = 1
SEMANTIC_MAX_RECEIPTS = 1024
SEMANTIC_MAX_FILE_BYTES = 4 * 1024 * 1024
SEMANTIC_REVISION_RETENTION = 32

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_NAME = re.compile(r"(?:0|[1-9][0-9]*)\.json\Z")
_TEMP_NAME = re.compile(
    r"\.(?P<target>[A-Za-z0-9_.-]+)\.publish-(?P<token>[0-9a-f-]{36})\.tmp\Z"
)
_COMPACTION_NAME = ".compaction.json"


class SemanticStoreError(RuntimeError):
    """Base error for unavailable or inconsistent Semantic authority."""


class SemanticStoreUnavailable(SemanticStoreError):
    """The private Semantic filesystem cannot currently be used."""


class SemanticStoreCorrupt(SemanticStoreError):
    """Stored Semantic bytes are malformed, unsafe, or inconsistent."""


class SemanticStoreConflict(SemanticStoreError):
    """An immutable Semantic artifact conflicts with the requested operation."""


@dataclass(frozen=True, slots=True)
class SemanticStoredEntry:
    """One verified immutable revision plus its operation evidence."""

    revision: SemanticRevision
    operation_digest: str
    anchor_revision: int | None = None
    anchor_revision_digest: str | None = None
    batch_index: int | None = None
    expected_revision: int | None = None
    expected_revision_digest: str | None = None


def _datetime_value(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise SemanticStoreCorrupt("Semantic timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise SemanticStoreCorrupt("Semantic timestamp is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SemanticStoreCorrupt("Semantic timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _digest_value(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SemanticStoreCorrupt(f"{name} is invalid")
    return value


def _require_keys(value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise SemanticStoreCorrupt("Semantic artifact shape is invalid")
    return value


def _source_edge_to_dict(edge: SemanticSourceEdge) -> dict[str, object]:
    return {
        "captured_context_id": edge.captured_context_id,
        "source_id": edge.source_id,
        "source_kind": edge.source_kind.value,
        "source_revision": edge.source_revision,
        "source_status": edge.source_status.value,
    }


def _source_edge_from_dict(value: object) -> SemanticSourceEdge:
    payload = _require_keys(
        value,
        {
            "captured_context_id",
            "source_id",
            "source_kind",
            "source_revision",
            "source_status",
        },
    )
    try:
        return SemanticSourceEdge(
            source_kind=SemanticSourceKind(payload["source_kind"]),
            source_id=payload["source_id"],
            source_revision=payload["source_revision"],
            captured_context_id=payload["captured_context_id"],
            source_status=SemanticSourceStatus(payload["source_status"]),
        )
    except (TypeError, ValueError, KeyError):
        raise SemanticStoreCorrupt("Semantic source edge is invalid") from None


def semantic_revision_to_dict(revision: SemanticRevision) -> dict[str, object]:
    """Serialize the complete bounded U1 revision contract."""

    if not isinstance(revision, SemanticRevision):
        raise TypeError("revision must be SemanticRevision")
    provenance = revision.provenance
    return {
        "content_digest": revision.content_digest,
        "created_at": _datetime_value(revision.created_at),
        "event_id": revision.event_id,
        "event_sequence": revision.event_sequence,
        "lifecycle": revision.lifecycle.value,
        "operation": revision.operation.value,
        "previous_revision_digest": revision.previous_revision_digest,
        "provenance": {
            "classification": provenance.classification.value,
            "digest": provenance.digest,
            "incomplete_source_count": provenance.incomplete_source_count,
            "known_context_ids": list(provenance.known_context_ids),
            "source_count": provenance.source_count,
            "unknown_source_count": provenance.unknown_source_count,
        },
        "provenance_class": (
            None if revision.provenance_class is None else revision.provenance_class.value
        ),
        "reason": revision.reason.value,
        "revision": revision.revision,
        "revision_digest": revision.revision_digest,
        "schema_version": revision.schema_version,
        "semantic_content": revision.semantic_content,
        "semantic_id": revision.semantic_id,
        "source_edges": [_source_edge_to_dict(edge) for edge in revision.source_edges],
    }


def semantic_revision_from_dict(value: object) -> SemanticRevision:
    """Deserialize and cryptographically validate one Semantic revision."""

    payload = _require_keys(
        value,
        {
            "content_digest",
            "created_at",
            "event_id",
            "event_sequence",
            "lifecycle",
            "operation",
            "previous_revision_digest",
            "provenance",
            "provenance_class",
            "reason",
            "revision",
            "revision_digest",
            "schema_version",
            "semantic_content",
            "semantic_id",
            "source_edges",
        },
    )
    provenance = _require_keys(
        payload["provenance"],
        {
            "classification",
            "digest",
            "incomplete_source_count",
            "known_context_ids",
            "source_count",
            "unknown_source_count",
        },
    )
    edges_value = payload["source_edges"]
    if not isinstance(edges_value, list):
        raise SemanticStoreCorrupt("Semantic source edge list is invalid")
    try:
        edges = tuple(_source_edge_from_dict(item) for item in edges_value)
        revision = SemanticRevision(
            semantic_id=payload["semantic_id"],
            revision=payload["revision"],
            semantic_content=payload["semantic_content"],
            content_digest=payload["content_digest"],
            created_at=_parse_datetime(payload["created_at"]),
            schema_version=payload["schema_version"],
            lifecycle=SemanticLifecycle(payload["lifecycle"]),
            source_edges=edges,
            provenance_class=(
                None
                if payload["provenance_class"] is None
                else SemanticProvenanceClass(payload["provenance_class"])
            ),
            operation=SemanticRevisionOperation(payload["operation"]),
            reason=SemanticRevisionReason(payload["reason"]),
            previous_revision_digest=payload["previous_revision_digest"],
            event_id=payload["event_id"],
            event_sequence=payload["event_sequence"],
        )
    except (TypeError, ValueError, KeyError):
        raise SemanticStoreCorrupt("Semantic revision is invalid") from None
    try:
        if revision.provenance_class is None:
            raise SemanticStoreCorrupt("Semantic provenance class is absent")
        expected_provenance = {
            "classification": revision.provenance.classification.value,
            "digest": revision.provenance.digest,
            "incomplete_source_count": revision.provenance.incomplete_source_count,
            "known_context_ids": list(revision.provenance.known_context_ids),
            "source_count": revision.provenance.source_count,
            "unknown_source_count": revision.provenance.unknown_source_count,
        }
        if (
            provenance != expected_provenance
            or payload["provenance_class"] != revision.provenance_class.value
        ):
            raise SemanticStoreCorrupt("Semantic provenance evidence is inconsistent")
        if payload["revision_digest"] != revision.revision_digest:
            raise SemanticStoreCorrupt("Semantic revision digest is invalid")
        validate_revision_digest(revision)
    except (TypeError, ValueError):
        raise SemanticStoreCorrupt("Semantic revision digest is invalid") from None
    return revision


class SemanticStore:
    """Own immutable Semantic lifecycle payloads and recovery evidence."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def from_memory_root(cls, memory_root: str | Path) -> SemanticStore:
        return cls(Path(memory_root).parent / "semantic")

    @property
    def records_root(self) -> Path:
        return self.root / "records"

    @property
    def pending_root(self) -> Path:
        return self.root / "pending"

    @property
    def receipts_root(self) -> Path:
        return self.root / "receipts"

    def record_path(self, semantic_id: str, revision: int) -> Path:
        self._validate_identifier(semantic_id)
        if type(revision) is not int or not 0 <= revision <= SEMANTIC_MAX_REVISION:
            raise ValueError("Semantic revision is invalid")
        return self.records_root / semantic_id / f"{revision}.json"

    def pending_path(self, transaction_id: str) -> Path:
        self._validate_uuid(transaction_id)
        return self.pending_root / f"{transaction_id}.json"

    def receipt_path(self, transaction_id: str) -> Path:
        self._validate_uuid(transaction_id)
        return self.receipts_root / f"{transaction_id}.json"

    def load_pending(self, transaction_id: str) -> dict[str, object] | None:
        return self._read_json(self.pending_path(transaction_id), missing_ok=True)

    def write_pending(self, transaction_id: str, payload: dict[str, object]) -> None:
        self._write_immutable_json(self.pending_path(transaction_id), payload)

    def remove_pending(self, transaction_id: str) -> None:
        self._remove_file(self.pending_path(transaction_id), "Semantic pending")

    def load_receipt(self, transaction_id: str) -> dict[str, object] | None:
        return self._read_json(self.receipt_path(transaction_id), missing_ok=True)

    def write_receipt(self, transaction_id: str, payload: dict[str, object]) -> None:
        self._write_immutable_json(self.receipt_path(transaction_id), payload)

    def remove_receipt(self, transaction_id: str) -> None:
        self._remove_file(self.receipt_path(transaction_id), "Semantic receipt")

    def prune_receipts(self, safe_transaction_ids: Iterable[str] = ()) -> None:
        """Delete only receipts proven unnecessary by the Journal authority.

        The Semantic store cannot determine whether a transaction may still be
        recovered.  Callers therefore provide the transaction IDs for which a
        durable participant outcome has already been recorded by EventJournal.
        Every other receipt remains protected.  If the protected set is still
        over the bound, the store fails closed rather than guessing which
        recovery evidence may be discarded.
        """

        safe_transaction_ids = frozenset(safe_transaction_ids)
        for transaction_id in safe_transaction_ids:
            self._validate_uuid(transaction_id)

        if not self._path_exists(self.receipts_root):
            return
        self._secure_parent(self.receipts_root, create=False)
        try:
            paths = tuple(self.receipts_root.iterdir())
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic receipts are unavailable") from error
        receipts: list[tuple[str, Path]] = []
        for path in paths:
            if path.name == _COMPACTION_NAME or path.suffix != ".json":
                raise SemanticStoreCorrupt("Semantic receipt file name is invalid")
            transaction_id = path.stem
            self._validate_uuid(transaction_id)
            self._secure_file(path)
            receipts.append((transaction_id, path))
        retired = tuple(
            (transaction_id, path)
            for transaction_id, path in receipts
            if transaction_id in safe_transaction_ids
        )
        if retired:
            directory_fd = self._directory_fd(self.receipts_root)
            try:
                for _transaction_id, path in retired:
                    try:
                        os.unlink(path.name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        continue
                os.fsync(directory_fd)
            except OSError as error:
                raise SemanticStoreUnavailable(
                    "Semantic receipt retirement is unavailable"
                ) from error
            finally:
                os.close(directory_fd)
        remaining = len(receipts) - len(retired)
        if remaining > SEMANTIC_MAX_RECEIPTS:
            raise SemanticStoreUnavailable("Semantic receipt retention is exhausted")

    def iter_current(self) -> tuple[SemanticStoredEntry, ...]:
        """Return every verified current Semantic authority entry.

        This enumeration is intentionally rooted in the private authority
        filesystem.  DB2 is never used to discover Semantic identities.
        """

        if not self._path_exists(self.records_root):
            return ()
        self._secure_parent(self.records_root, create=False)
        try:
            paths = tuple(self.records_root.iterdir())
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic records are unavailable") from error
        entries: list[SemanticStoredEntry] = []
        for path in paths:
            try:
                status = path.lstat()
            except OSError as error:
                raise SemanticStoreUnavailable(
                    "Semantic record identity is unavailable"
                ) from error
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o700
            ):
                raise SemanticStoreCorrupt("Semantic record directory is unsafe")
            try:
                self._validate_identifier(path.name)
            except ValueError:
                raise SemanticStoreCorrupt("Semantic record identity is invalid") from None
            current = self.load_current(path.name)
            if current is None:
                raise SemanticStoreCorrupt("Semantic record authority is absent")
            entries.append(current)
        return tuple(sorted(entries, key=lambda entry: entry.revision.semantic_id))

    def iter_entries(self) -> tuple[SemanticStoredEntry, ...]:
        """Return every retained, verified revision in deterministic order."""

        if not self._path_exists(self.records_root):
            return ()
        self._secure_parent(self.records_root, create=False)
        try:
            paths = tuple(self.records_root.iterdir())
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic records are unavailable") from error
        entries: list[SemanticStoredEntry] = []
        for path in paths:
            try:
                status = path.lstat()
            except OSError as error:
                raise SemanticStoreUnavailable(
                    "Semantic record identity is unavailable"
                ) from error
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o700
            ):
                raise SemanticStoreCorrupt("Semantic record directory is unsafe")
            try:
                self._validate_identifier(path.name)
            except ValueError:
                raise SemanticStoreCorrupt("Semantic record identity is invalid") from None
            self._secure_parent(path, create=False)
            self._recover_publication_temps(path)
            self._recover_compaction(path, path.name)
            entries.extend(self._load_entries(path, path.name).values())
        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    entry.revision.semantic_id,
                    entry.revision.revision,
                ),
            )
        )

    def load_current(self, semantic_id: str) -> SemanticStoredEntry | None:
        self._validate_identifier(semantic_id)
        directory = self.records_root / semantic_id
        if not self._path_exists(directory):
            return None
        self._secure_parent(directory, create=False)
        self._recover_publication_temps(directory)
        self._recover_compaction(directory, semantic_id)
        entries = self._load_entries(directory, semantic_id)
        if not entries:
            raise SemanticStoreCorrupt("Semantic record directory is empty")
        try:
            self._validate_revision_window(entries)
        except SemanticStoreCorrupt:
            current_revision = max(entries)
            if current_revision <= SEMANTIC_REVISION_RETENTION or set(entries) != set(
                range(current_revision + 1)
            ):
                raise
            # A process may have published the next immutable revision and
            # crashed just before writing the compaction marker.  Re-enter the
            # same proof-bound compaction path rather than treating that safe
            # superset as corruption.
            self._prune(semantic_id, current_revision)
            entries = self._load_entries(directory, semantic_id)
            self._validate_revision_window(entries)
        return entries[max(entries)]

    def load_committed_current(self, semantic_id: str) -> SemanticStoredEntry:
        current = self.load_current(semantic_id)
        if current is None:
            raise SemanticStoreCorrupt("Committed Semantic record is absent")
        return current

    def load_revision(self, semantic_id: str, revision: int) -> SemanticStoredEntry | None:
        self._validate_identifier(semantic_id)
        if type(revision) is not int or not 0 <= revision <= SEMANTIC_MAX_REVISION:
            raise ValueError("Semantic revision is invalid")
        current = self.load_current(semantic_id)
        if current is None or revision > current.revision.revision:
            return None
        path = self.record_path(semantic_id, revision)
        payload = self._read_json(path, missing_ok=True)
        if payload is None:
            raise SemanticStoreCorrupt("Semantic revision artifact is absent")
        return self._entry_from_payload(payload, semantic_id, revision)

    def publish_create(
        self,
        revision: SemanticRevision,
        operation_digest: str,
        *,
        batch_index: int | None = None,
    ) -> SemanticStoredEntry:
        if revision.revision != 0:
            raise SemanticStoreConflict("Semantic create must publish revision zero")
        entry = SemanticStoredEntry(
            revision,
            self._validate_digest(operation_digest),
            batch_index=batch_index,
        )
        self._publish_entry(entry)
        return entry

    def publish_revision(
        self,
        revision: SemanticRevision,
        operation_digest: str,
        *,
        expected_revision: int,
        expected_digest: str,
        batch_index: int | None = None,
    ) -> SemanticStoredEntry:
        current = self.load_current(revision.semantic_id)
        if current is None:
            raise SemanticStoreUnavailable("Semantic revision target is absent")
        current_digest = current.revision.revision_digest
        if current.revision.revision == revision.revision and current_digest == revision.revision_digest:
            if current.operation_digest != operation_digest:
                raise SemanticStoreConflict("Semantic revision operation conflicts")
            return current
        if current.revision.revision != expected_revision or current_digest != expected_digest:
            raise SemanticStoreConflict("Semantic revision target is stale")
        if revision.revision != expected_revision + 1:
            raise SemanticStoreConflict("Semantic revision is not the next revision")
        anchor_revision: int | None = None
        anchor_digest: str | None = None
        if revision.revision >= SEMANTIC_REVISION_RETENTION + 1:
            anchor_revision = revision.revision - (SEMANTIC_REVISION_RETENTION + 1)
            anchor = self.load_revision(revision.semantic_id, anchor_revision)
            if anchor is None:
                raise SemanticStoreUnavailable("Semantic history anchor is absent")
            anchor_digest = anchor.revision.revision_digest
        entry = SemanticStoredEntry(
            revision,
            self._validate_digest(operation_digest),
            anchor_revision,
            anchor_digest,
            batch_index,
            expected_revision,
            expected_digest,
        )
        self._publish_entry(entry)
        return entry

    def reconcile_prune(self, semantic_id: str) -> None:
        current = self.load_current(semantic_id)
        if current is not None:
            self._prune(semantic_id, current.revision.revision)

    def _publish_entry(self, entry: SemanticStoredEntry) -> None:
        revision = entry.revision
        validate_revision_digest(revision)
        if revision.revision == 0:
            if entry.anchor_revision is not None or entry.anchor_revision_digest is not None:
                raise SemanticStoreConflict("Genesis Semantic artifact has anchor evidence")
        elif revision.revision < SEMANTIC_REVISION_RETENTION + 1:
            if entry.anchor_revision is not None or entry.anchor_revision_digest is not None:
                raise SemanticStoreConflict("Unexpected Semantic anchor evidence")
            if revision.previous_revision_digest is None:
                raise SemanticStoreConflict("Semantic predecessor evidence is missing")
        else:
            if (
                entry.anchor_revision != revision.revision - (SEMANTIC_REVISION_RETENTION + 1)
                or _DIGEST.fullmatch(entry.anchor_revision_digest or "") is None
            ):
                raise SemanticStoreConflict("Semantic history anchor is invalid")
        payload: dict[str, object] = {
            "anchor_revision": entry.anchor_revision,
            "anchor_revision_digest": entry.anchor_revision_digest,
            "batch_index": entry.batch_index,
            "operation_digest": entry.operation_digest,
            "revision": semantic_revision_to_dict(revision),
            "expected_revision": entry.expected_revision,
            "expected_revision_digest": entry.expected_revision_digest,
            "revision_digest": revision.revision_digest,
            "schema_version": SEMANTIC_STORE_SCHEMA_VERSION,
            "semantic_id": revision.semantic_id,
        }
        self._write_immutable_json(self.record_path(revision.semantic_id, revision.revision), payload)
        self._prune(revision.semantic_id, revision.revision)

    def _prune(self, semantic_id: str, current_revision: int) -> None:
        directory = self.records_root / semantic_id
        self._secure_parent(directory, create=False)
        self._recover_publication_temps(directory)
        self._recover_compaction(directory, semantic_id)
        entries = self._load_entries(directory, semantic_id)
        if not entries or max(entries) != current_revision:
            raise SemanticStoreUnavailable("Semantic publication is not current")
        if current_revision <= SEMANTIC_REVISION_RETENTION:
            self._validate_revision_window(entries)
            return
        floor = current_revision - SEMANTIC_REVISION_RETENTION
        current = entries[current_revision]
        if (
            current.anchor_revision != floor - 1
            or current.anchor_revision_digest is None
        ):
            raise SemanticStoreConflict("Semantic compaction anchor is missing")
        expected = set(range(floor, current_revision + 1))
        if set(entries) == expected:
            self._validate_revision_window(entries)
            return
        if not set(entries).issuperset(expected):
            raise SemanticStoreCorrupt("Semantic compaction suffix is incomplete")
        marker_payload: dict[str, object] = {
            "anchor_revision": current.anchor_revision,
            "anchor_revision_digest": current.anchor_revision_digest,
            "current_revision": current_revision,
            "current_revision_digest": current.revision.revision_digest,
            "delete_revisions": list(range(floor)),
            "floor": floor,
            "schema_version": 1,
            "semantic_id": semantic_id,
        }
        self._write_immutable_json(directory / _COMPACTION_NAME, marker_payload)
        self._recover_compaction(directory, semantic_id)
        final_entries = self._load_entries(directory, semantic_id)
        self._validate_revision_window(final_entries)

    def _load_entries(
        self, directory: Path, semantic_id: str
    ) -> dict[int, SemanticStoredEntry]:
        try:
            paths = tuple(directory.iterdir())
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic records are unavailable") from error
        result: dict[int, SemanticStoredEntry] = {}
        for path in paths:
            if path.name == _COMPACTION_NAME:
                continue
            if _TEMP_NAME.fullmatch(path.name) is not None:
                raise SemanticStoreCorrupt("Unreconciled Semantic publication temp remains")
            if path.suffix != ".json" or _REVISION_NAME.fullmatch(path.name) is None:
                raise SemanticStoreCorrupt("Semantic revision file name is invalid")
            try:
                revision = int(path.stem)
            except ValueError:
                raise SemanticStoreCorrupt("Semantic revision file name is invalid") from None
            if revision in result:
                raise SemanticStoreCorrupt("Duplicate Semantic revision artifact")
            payload = self._read_json(path)
            result[revision] = self._entry_from_payload(payload, semantic_id, revision)
        return result

    def _entry_from_payload(
        self, payload: object, semantic_id: str, revision: int
    ) -> SemanticStoredEntry:
        if not isinstance(payload, dict):
            raise SemanticStoreCorrupt("Semantic stored entry is invalid")
        legacy_keys = {
            "anchor_revision",
            "anchor_revision_digest",
            "operation_digest",
            "revision",
            "revision_digest",
            "schema_version",
            "semantic_id",
        }
        current_keys = legacy_keys | {
            "batch_index",
            "expected_revision",
            "expected_revision_digest",
        }
        if set(payload) not in (legacy_keys, current_keys):
            raise SemanticStoreCorrupt("Semantic stored entry shape is invalid")
        value = payload
        if (
            value["schema_version"] not in {1, SEMANTIC_STORE_SCHEMA_VERSION}
            or value["semantic_id"] != semantic_id
        ):
            raise SemanticStoreCorrupt("Semantic store schema is unsupported")
        operation_digest = _digest_value(value["operation_digest"], "operation_digest")
        record = semantic_revision_from_dict(value["revision"])
        if (
            record.semantic_id != semantic_id
            or record.revision != revision
            or value["revision_digest"] != record.revision_digest
        ):
            raise SemanticStoreCorrupt("Semantic revision digest is inconsistent")
        anchor_revision = value["anchor_revision"]
        anchor_digest = value["anchor_revision_digest"]
        if anchor_revision is not None:
            if type(anchor_revision) is not int or anchor_revision < 0:
                raise SemanticStoreCorrupt("Semantic anchor revision is invalid")
            _digest_value(anchor_digest, "anchor_revision_digest")
        elif anchor_digest is not None:
            raise SemanticStoreCorrupt("Semantic anchor digest is unexpected")
        batch_index = value.get("batch_index")
        expected_revision = value.get("expected_revision")
        expected_revision_digest = value.get("expected_revision_digest")
        if batch_index is not None and (
            type(batch_index) is not int or not 0 <= batch_index < 128
        ):
            raise SemanticStoreCorrupt("Semantic batch index is invalid")
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision < 0
        ):
            raise SemanticStoreCorrupt("Semantic expected revision is invalid")
        if expected_revision_digest is not None:
            _digest_value(expected_revision_digest, "expected_revision_digest")
        if record.revision == 0 and (
            expected_revision is not None or expected_revision_digest is not None
        ):
            raise SemanticStoreCorrupt("Genesis Semantic expected revision is invalid")
        if record.revision > 0 and (
            expected_revision is not None and expected_revision != record.revision - 1
        ):
            raise SemanticStoreCorrupt("Semantic expected revision is inconsistent")
        if record.revision > 0 and (
            expected_revision_digest is not None
            and expected_revision_digest != record.previous_revision_digest
        ):
            raise SemanticStoreCorrupt("Semantic expected predecessor is inconsistent")
        if revision >= SEMANTIC_REVISION_RETENTION + 1:
            if anchor_revision != revision - (SEMANTIC_REVISION_RETENTION + 1):
                raise SemanticStoreCorrupt("Semantic anchor revision is inconsistent")
        elif anchor_revision is not None:
            raise SemanticStoreCorrupt("Unexpected Semantic anchor evidence")
        return SemanticStoredEntry(
            record,
            operation_digest,
            anchor_revision,
            anchor_digest,
            batch_index,
            expected_revision,
            expected_revision_digest,
        )

    def _validate_revision_window(self, entries: dict[int, SemanticStoredEntry]) -> None:
        if not entries:
            raise SemanticStoreCorrupt("Semantic revision window is empty")
        current_revision = max(entries)
        floor = max(0, current_revision - SEMANTIC_REVISION_RETENTION)
        expected = set(range(floor, current_revision + 1))
        if set(entries) != expected:
            raise SemanticStoreCorrupt("Semantic revision window is incomplete")
        for revision in range(floor, current_revision + 1):
            entry = entries[revision]
            if revision == 0:
                if entry.revision.previous_revision_digest is not None:
                    raise SemanticStoreCorrupt("Genesis Semantic revision has a predecessor")
                continue
            previous = entries.get(revision - 1)
            if previous is not None:
                if entry.revision.previous_revision_digest != previous.revision.revision_digest:
                    raise SemanticStoreCorrupt("Semantic revision chain is broken")
            elif entry.revision.previous_revision_digest is None:
                raise SemanticStoreCorrupt("Compacted Semantic predecessor is absent")
        current = entries[current_revision]
        if current_revision > SEMANTIC_REVISION_RETENTION:
            if (
                current.anchor_revision != floor - 1
                or current.anchor_revision_digest != entries[floor].revision.previous_revision_digest
            ):
                raise SemanticStoreCorrupt("Semantic compaction anchor is inconsistent")
        elif current.anchor_revision is not None:
            raise SemanticStoreCorrupt("Unexpected Semantic compaction anchor")

    def _recover_publication_temps(self, directory: Path) -> None:
        try:
            paths = tuple(directory.iterdir())
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic records are unavailable") from error
        found = False
        for temporary in paths:
            match = _TEMP_NAME.fullmatch(temporary.name)
            if match is None:
                continue
            found = True
            try:
                token = UUID(match.group("token"))
            except (TypeError, ValueError):
                raise SemanticStoreCorrupt("Semantic temp token is invalid") from None
            if str(token) != match.group("token"):
                raise SemanticStoreCorrupt("Semantic temp token is invalid")
            target_name = match.group("target")
            if target_name != _COMPACTION_NAME and _REVISION_NAME.fullmatch(target_name) is None:
                raise SemanticStoreCorrupt("Semantic temp target is invalid")
            self._secure_file(temporary)
            target = directory / target_name
            temporary_payload = self._read_json(temporary)
            if self._path_exists(target) and self._read_json(target) != temporary_payload:
                raise SemanticStoreCorrupt("Semantic temp conflicts with final artifact")
            try:
                temporary.unlink()
            except OSError as error:
                raise SemanticStoreUnavailable("Semantic temp cleanup is unavailable") from error
        if found:
            self._fsync_directory(directory)

    def _recover_compaction(self, directory: Path, semantic_id: str) -> None:
        marker = directory / _COMPACTION_NAME
        if not self._path_exists(marker):
            return
        payload = self._read_json(marker)
        value = _require_keys(
            payload,
            {
                "anchor_revision",
                "anchor_revision_digest",
                "current_revision",
                "current_revision_digest",
                "delete_revisions",
                "floor",
                "schema_version",
                "semantic_id",
            },
        )
        if value["schema_version"] != 1 or value["semantic_id"] != semantic_id:
            raise SemanticStoreCorrupt("Semantic compaction marker is unsupported")
        current_revision = value["current_revision"]
        floor = value["floor"]
        delete_revisions = value["delete_revisions"]
        if (
            type(current_revision) is not int
            or type(floor) is not int
            or type(delete_revisions) is not list
            or delete_revisions != list(range(floor))
            or floor != current_revision - SEMANTIC_REVISION_RETENTION
        ):
            raise SemanticStoreCorrupt("Semantic compaction marker is invalid")
        anchor_revision = value["anchor_revision"]
        if anchor_revision != floor - 1:
            raise SemanticStoreCorrupt("Semantic compaction anchor is invalid")
        anchor_digest = _digest_value(value["anchor_revision_digest"], "anchor_revision_digest")
        current_digest = _digest_value(value["current_revision_digest"], "current_revision_digest")
        entries = self._load_entries(directory, semantic_id)
        suffix = set(range(floor, current_revision + 1))
        if not suffix.issubset(entries):
            raise SemanticStoreCorrupt("Semantic compaction suffix is incomplete")
        current = entries[current_revision]
        if (
            current.revision.revision_digest != current_digest
            or current.anchor_revision != anchor_revision
            or current.anchor_revision_digest != anchor_digest
        ):
            raise SemanticStoreCorrupt("Semantic compaction marker conflicts")
        first = entries[floor]
        if first.revision.previous_revision_digest != anchor_digest:
            raise SemanticStoreCorrupt("Semantic compaction predecessor conflicts")
        for revision in delete_revisions:
            path = directory / f"{revision}.json"
            if not self._path_exists(path):
                continue
            self._secure_file(path)
            try:
                path.unlink()
            except OSError as error:
                raise SemanticStoreUnavailable("Semantic history compaction is unavailable") from error
        try:
            marker.unlink()
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic compaction marker cleanup is unavailable") from error
        self._fsync_directory(directory)

    def _write_immutable_json(self, path: Path, payload: dict[str, object]) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
        if len(encoded) > SEMANTIC_MAX_FILE_BYTES:
            raise SemanticStoreConflict("Semantic artifact is too large")
        self._secure_parent(path.parent, create=True)
        self._recover_publication_temps(path.parent)
        existing = self._read_json(path, missing_ok=True)
        if existing is not None:
            if existing != payload:
                raise SemanticStoreConflict("Semantic artifact conflicts")
            return
        parent_fd = self._directory_fd(path.parent)
        temporary = f".{path.name}.publish-{uuid4()}.tmp"
        descriptor = -1
        linked = False
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
            os.link(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
            linked = True
            os.unlink(temporary, dir_fd=parent_fd)
            temporary = ""
            os.fsync(parent_fd)
        except FileExistsError:
            existing = self._read_json(path)
            if existing != payload:
                raise SemanticStoreConflict("Semantic artifact conflicts") from None
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic atomic publication is unavailable") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary and not linked:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except OSError as error:
                    raise SemanticStoreUnavailable("Semantic temporary cleanup is unavailable") from error
            os.close(parent_fd)

    def _read_json(self, path: Path, *, missing_ok: bool = False) -> dict[str, object] | None:
        try:
            self._secure_file(path)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise SemanticStoreUnavailable("Semantic artifact is absent") from None
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) != 0o600:
                raise SemanticStoreCorrupt("Semantic artifact is not private")
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                encoded = source.read(SEMANTIC_MAX_FILE_BYTES + 1)
            if len(encoded) > SEMANTIC_MAX_FILE_BYTES:
                raise SemanticStoreCorrupt("Semantic artifact is oversized")
            loaded = json.loads(encoded)
            if not isinstance(loaded, dict):
                raise SemanticStoreCorrupt("Semantic artifact is not an object")
            return loaded
        except SemanticStoreError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise SemanticStoreCorrupt("Semantic artifact cannot be read") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _remove_file(self, path: Path, label: str) -> None:
        if not self._path_exists(path):
            return
        self._secure_parent(path.parent, create=False)
        directory_fd = self._directory_fd(path.parent)
        try:
            self._secure_file(path)
            os.unlink(path.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            return
        except OSError as error:
            raise SemanticStoreUnavailable(f"{label} storage is unavailable") from error
        finally:
            os.close(directory_fd)

    def _secure_parent(self, path: Path, *, create: bool) -> None:
        absolute = path.absolute()
        root = self.root.absolute()
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            raise SemanticStoreCorrupt("Semantic path escapes its root") from None
        try:
            for ancestor in reversed(root.parents):
                status = ancestor.lstat()
                if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                    raise SemanticStoreCorrupt("Semantic path has an unsafe parent")
            try:
                root_status = root.lstat()
            except FileNotFoundError:
                if not create:
                    raise SemanticStoreUnavailable("Semantic directory is absent") from None
                root.mkdir(mode=0o700)
                self._fsync_directory(root.parent)
                root_status = root.lstat()
            if not stat.S_ISDIR(root_status.st_mode) or root_status.st_uid != os.geteuid():
                raise SemanticStoreCorrupt("Semantic directory is unsafe")
            os.chmod(root, 0o700)
            current = root
            for component in relative.parts:
                current /= component
                try:
                    status = current.lstat()
                except FileNotFoundError:
                    if not create:
                        raise SemanticStoreUnavailable("Semantic directory is absent") from None
                    current.mkdir(mode=0o700)
                    self._fsync_directory(current.parent)
                    status = current.lstat()
                if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
                    raise SemanticStoreCorrupt("Semantic directory is unsafe")
                os.chmod(current, 0o700)
                if stat.S_IMODE(current.stat().st_mode) != 0o700:
                    raise SemanticStoreCorrupt("Semantic directory mode is invalid")
        except SemanticStoreError:
            raise
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic directory is unavailable") from error

    def _secure_file(self, path: Path) -> None:
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid():
            raise SemanticStoreCorrupt("Semantic artifact is unsafe")
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise SemanticStoreCorrupt("Semantic artifact mode is invalid")

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            path.lstat()
            return True
        except FileNotFoundError:
            return False
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic path cannot be inspected") from error

    @staticmethod
    def _directory_fd(path: Path) -> int:
        try:
            return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW)
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic directory is unavailable") from error

    @classmethod
    def _fsync_directory(cls, path: Path) -> None:
        descriptor = cls._directory_fd(path)
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise SemanticStoreUnavailable("Semantic directory cannot be synced") from error
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_uuid(value: str) -> None:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError):
            raise ValueError("Semantic transaction identifier is invalid") from None
        if str(parsed) != value:
            raise ValueError("Semantic transaction identifier is invalid")

    @staticmethod
    def _validate_identifier(value: str) -> None:
        try:
            validate_identifier(value)
        except (TypeError, ValueError):
            raise ValueError("Semantic identifier is invalid") from None

    @staticmethod
    def _validate_digest(value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("Semantic operation digest is invalid")
        return value


__all__ = [
    "SEMANTIC_MAX_FILE_BYTES",
    "SEMANTIC_MAX_RECEIPTS",
    "SEMANTIC_PENDING_SCHEMA_VERSION",
    "SEMANTIC_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_REVISION_RETENTION",
    "SEMANTIC_STORE_SCHEMA_VERSION",
    "SemanticStoredEntry",
    "SemanticStore",
    "SemanticStoreConflict",
    "SemanticStoreCorrupt",
    "SemanticStoreError",
    "SemanticStoreUnavailable",
    "semantic_revision_from_dict",
    "semantic_revision_to_dict",
]
