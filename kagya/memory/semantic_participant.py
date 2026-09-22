"""R07 batch participant for Memory-owned Semantic lifecycle mutations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import re
from uuid import UUID, uuid5

from kagya.memory.dual_memory_system import (
    DualMemorySystem,
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
    SemanticMemoryFormatError,
    SemanticMemoryReadError,
    SemanticProjectionStatus,
)
from kagya.memory.semantic_lifecycle import (
    SEMANTIC_MAX_REVISION,
    SemanticLifecycle,
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticSourceKind,
    SemanticSourceStatus,
)
from kagya.memory.semantic_store import (
    SEMANTIC_PENDING_SCHEMA_VERSION,
    SEMANTIC_RECEIPT_SCHEMA_VERSION,
    SemanticStore,
    SemanticStoreConflict,
    SemanticStoreCorrupt,
    SemanticStoreError,
    SemanticStoreUnavailable,
    SemanticStoredEntry,
    semantic_revision_from_dict,
    semantic_revision_to_dict,
)
from kagya.runtime.event_journal import (
    AbortOutcome,
    ParticipantCapability,
    ParticipantOutcome,
    StartupParticipantOutcome,
)
from kagya.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    UnsupportedParticipantReconciliationError,
    validate_transaction_binding,
)
from kagya.identifiers import validate_identifier


MEMORY_SEMANTIC_PARTICIPANT_ID = "memory.semantic"
SEMANTIC_MAX_BATCH_ENTRIES = 128
_SEMANTIC_ID_NAMESPACE = UUID("4a64ff28-55e0-5e7d-9f6d-08f743ab4b47")
_OPERATION_DOMAIN = b"PROJECT-KAGYA:R12:SEMANTIC-PARTICIPANT:V1\0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class SemanticMutationKind(StrEnum):
    CREATE = "create"
    REVISION = "revision"


def semantic_id_for_batch_entry(transaction_id: str, batch_index: int) -> str:
    """Return the deterministic new-format Semantic identity for one batch slot."""

    try:
        parsed = UUID(transaction_id)
    except (TypeError, ValueError):
        raise ValueError("Semantic transaction identity is invalid") from None
    if str(parsed) != transaction_id:
        raise ValueError("Semantic transaction identity is invalid")
    if type(batch_index) is not int or not 0 <= batch_index < SEMANTIC_MAX_BATCH_ENTRIES:
        raise ValueError("Semantic batch index is invalid")
    canonical = json.dumps(
        [transaction_id, batch_index],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"semantic-{uuid5(_SEMANTIC_ID_NAMESPACE, canonical)}"


@dataclass(frozen=True, slots=True)
class SemanticCreateIntent:
    revision: SemanticRevision

    def __post_init__(self) -> None:
        if not isinstance(self.revision, SemanticRevision):
            raise TypeError("Semantic create revision is invalid")
        if (
            self.revision.revision != 0
            or self.revision.lifecycle is not SemanticLifecycle.ACTIVE
            or self.revision.operation is not SemanticRevisionOperation.CREATE
        ):
            raise ValueError("Semantic create must be active revision zero")

    @property
    def kind(self) -> SemanticMutationKind:
        return SemanticMutationKind.CREATE

    def canonical_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "revision": semantic_revision_to_dict(self.revision)}


@dataclass(frozen=True, slots=True)
class SemanticRevisionIntent:
    revision: SemanticRevision
    expected_revision: int
    expected_revision_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.revision, SemanticRevision):
            raise TypeError("Semantic revision is invalid")
        if self.revision.revision <= 0:
            raise ValueError("Semantic revision must be positive")
        if type(self.expected_revision) is not int or not 0 <= self.expected_revision <= SEMANTIC_MAX_REVISION:
            raise ValueError("Expected Semantic revision is invalid")
        if self.revision.revision != self.expected_revision + 1:
            raise ValueError("Semantic revision does not advance the expected target")
        if _DIGEST.fullmatch(self.expected_revision_digest) is None:
            raise ValueError("Expected Semantic revision digest is invalid")
        if self.revision.operation is SemanticRevisionOperation.CREATE:
            raise ValueError("Semantic revision cannot be a creation")
        if self.revision.previous_revision_digest != self.expected_revision_digest:
            raise ValueError("Semantic revision predecessor is not the expected target")

    @property
    def kind(self) -> SemanticMutationKind:
        return SemanticMutationKind.REVISION

    def canonical_dict(self) -> dict[str, object]:
        return {
            "expected_revision": self.expected_revision,
            "expected_revision_digest": self.expected_revision_digest,
            "kind": self.kind.value,
            "revision": semantic_revision_to_dict(self.revision),
        }


SemanticMutation = SemanticCreateIntent | SemanticRevisionIntent


@dataclass(frozen=True, slots=True)
class SemanticBatchEntry:
    batch_index: int
    mutation: SemanticMutation

    def __post_init__(self) -> None:
        if type(self.batch_index) is not int or not 0 <= self.batch_index < SEMANTIC_MAX_BATCH_ENTRIES:
            raise ValueError("Semantic batch index is invalid")
        if not isinstance(self.mutation, (SemanticCreateIntent, SemanticRevisionIntent)):
            raise TypeError("Semantic batch mutation is invalid")

    @property
    def semantic_id(self) -> str:
        return self.mutation.revision.semantic_id

    def canonical_dict(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index,
            "mutation": self.mutation.canonical_dict(),
        }


@dataclass(frozen=True, slots=True)
class SemanticBatchOperation:
    transaction_id: str
    entries: tuple[SemanticBatchEntry, ...]

    def __post_init__(self) -> None:
        try:
            parsed = UUID(self.transaction_id)
        except (TypeError, ValueError):
            raise ValueError("Semantic transaction identity is invalid") from None
        if str(parsed) != self.transaction_id:
            raise ValueError("Semantic transaction identity is invalid")
        if type(self.entries) is not tuple:
            raise TypeError("Semantic batch entries must be a tuple")
        if not self.entries or len(self.entries) > SEMANTIC_MAX_BATCH_ENTRIES:
            raise ValueError("Semantic batch size is outside its bound")
        expected_indices = tuple(range(len(self.entries)))
        actual_indices = tuple(entry.batch_index for entry in self.entries)
        if actual_indices != expected_indices:
            raise ValueError("Semantic batch indices must be contiguous")
        semantic_ids = tuple(entry.semantic_id for entry in self.entries)
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("Semantic batch contains duplicate identities")
        for entry in self.entries:
            validate_identifier(entry.semantic_id)
            if isinstance(entry.mutation, SemanticCreateIntent):
                if entry.semantic_id != semantic_id_for_batch_entry(
                    self.transaction_id, entry.batch_index
                ):
                    raise ValueError("Semantic create identity is not deterministic")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.canonical_dict() for entry in self.entries],
            "transaction_id": self.transaction_id,
        }


def semantic_batch_operation_digest(operation: SemanticBatchOperation) -> str:
    return hashlib.sha256(
        _OPERATION_DOMAIN + _canonical_json(operation.canonical_dict())
    ).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _mutation_from_dict(value: object) -> SemanticMutation:
    if not isinstance(value, dict) or set(value) not in (
        {"kind", "revision"},
        {"expected_revision", "expected_revision_digest", "kind", "revision"},
    ):
        raise ParticipantDivergedError("Semantic batch mutation is invalid")
    kind = value.get("kind")
    try:
        revision = semantic_revision_from_dict(value["revision"])
        if kind == SemanticMutationKind.CREATE.value:
            if set(value) != {"kind", "revision"}:
                raise ValueError
            return SemanticCreateIntent(revision)
        if kind != SemanticMutationKind.REVISION.value:
            raise ValueError
        return SemanticRevisionIntent(
            revision,
            value["expected_revision"],
            value["expected_revision_digest"],
        )
    except (KeyError, TypeError, ValueError, SemanticStoreError):
        raise ParticipantDivergedError("Semantic batch mutation is invalid") from None


def _operation_from_dict(value: object) -> SemanticBatchOperation:
    if not isinstance(value, dict) or set(value) != {"entries", "transaction_id"}:
        raise ParticipantDivergedError("Semantic batch operation is invalid")
    entries_value = value["entries"]
    if not isinstance(entries_value, list):
        raise ParticipantDivergedError("Semantic batch entries are invalid")
    entries: list[SemanticBatchEntry] = []
    try:
        for item in entries_value:
            if not isinstance(item, dict) or set(item) != {"batch_index", "mutation"}:
                raise ValueError
            entries.append(
                SemanticBatchEntry(item["batch_index"], _mutation_from_dict(item["mutation"]))
            )
        operation = SemanticBatchOperation(value["transaction_id"], tuple(entries))
    except (KeyError, TypeError, ValueError, SemanticStoreError):
        raise ParticipantDivergedError("Semantic batch operation is invalid") from None
    return operation


class MemorySemanticParticipant:
    """Prepare/finalize/abort authority for one bounded Semantic batch."""

    participant_id = MEMORY_SEMANTIC_PARTICIPANT_ID
    capabilities = (
        ParticipantCapability.ABORT,
        ParticipantCapability.IDEMPOTENT_FINALIZE,
        ParticipantCapability.INSPECT_RECONCILE,
        ParticipantCapability.PREPARE,
    )

    def __init__(
        self,
        memory: DualMemorySystem,
        store: SemanticStore,
        operation: SemanticBatchOperation,
    ) -> None:
        self.memory = memory
        self.store = store
        self.operation = operation
        self.operation_digest = semantic_batch_operation_digest(operation)

    @classmethod
    def from_pending(
        cls,
        memory: DualMemorySystem,
        store: SemanticStore,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
        *,
        event_id: str | None = None,
        processing_sequence: int | None = None,
    ) -> MemorySemanticParticipant:
        if (
            participant_id != MEMORY_SEMANTIC_PARTICIPANT_ID
            or _DIGEST.fullmatch(operation_digest) is None
        ):
            raise ParticipantDivergedError("Semantic binding is invalid")
        pending = cls._load_pending(store, transaction_id)
        receipt = cls._load_receipt(store, transaction_id)
        payload = pending or receipt
        if payload is None:
            raise UnsupportedParticipantReconciliationError(
                "Semantic operation evidence is absent"
            )
        operation = cls._operation_from_evidence(
            payload, transaction_id, participant_id, operation_digest
        )
        participant = cls(memory, store, operation)
        if event_id is not None or processing_sequence is not None:
            if event_id is None or processing_sequence is None:
                raise ParticipantDivergedError("Semantic event binding is incomplete")
            participant._validate_event_identity(event_id, processing_sequence)
        return participant

    @staticmethod
    def _load_pending(
        store: SemanticStore, transaction_id: str
    ) -> dict[str, object] | None:
        try:
            return store.load_pending(transaction_id)
        except (SemanticStoreCorrupt, SemanticStoreConflict) as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None

    @staticmethod
    def _load_receipt(
        store: SemanticStore, transaction_id: str
    ) -> dict[str, object] | None:
        try:
            return store.load_receipt(transaction_id)
        except (SemanticStoreCorrupt, SemanticStoreConflict) as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None

    @staticmethod
    def _operation_from_evidence(
        payload: dict[str, object],
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
    ) -> SemanticBatchOperation:
        expected_schema = (
            SEMANTIC_PENDING_SCHEMA_VERSION
            if payload.get("schema_version") == SEMANTIC_PENDING_SCHEMA_VERSION
            else SEMANTIC_RECEIPT_SCHEMA_VERSION
        )
        if (
            set(payload)
            != {"operation", "operation_digest", "participant_id", "schema_version", "transaction_id"}
            or payload["schema_version"] != expected_schema
            or payload["transaction_id"] != transaction_id
            or payload["participant_id"] != participant_id
            or payload["operation_digest"] != operation_digest
        ):
            raise ParticipantDivergedError("Semantic evidence conflicts")
        operation = _operation_from_dict(payload["operation"])
        if (
            operation.transaction_id != transaction_id
            or semantic_batch_operation_digest(operation) != operation_digest
        ):
            raise ParticipantDivergedError("Semantic evidence digest conflicts")
        return operation

    def prepare(self, binding: TransactionBinding) -> None:
        self._validate_binding(binding)
        self._validate_event_identity(binding.event_id, binding.processing_sequence)
        self._validate_sources()
        expected = self._artifact(binding, SEMANTIC_PENDING_SCHEMA_VERSION)
        pending = self._load_pending(self.store, binding.transaction_id)
        receipt = self._load_receipt(self.store, binding.transaction_id)
        if pending is not None and pending != expected:
            raise ParticipantDivergedError("Semantic pending artifact conflicts")
        if receipt is not None:
            self._operation_from_evidence(
                receipt,
                binding.transaction_id,
                binding.participant_id,
                binding.operation_digest,
            )
        self._validate_revision_preconditions()
        if pending is None and receipt is None:
            try:
                self.store.write_pending(binding.transaction_id, expected)
            except (SemanticStoreCorrupt, SemanticStoreConflict) as error:
                raise ParticipantDivergedError(str(error)) from None
            except SemanticStoreUnavailable as error:
                raise ParticipantUnavailableError(str(error)) from None

    @classmethod
    def abort_pending(
        cls, memory: DualMemorySystem, store: SemanticStore, binding: TransactionBinding
    ) -> AbortOutcome:
        if (
            not validate_transaction_binding(binding)
            or binding.participant_id != MEMORY_SEMANTIC_PARTICIPANT_ID
            or _DIGEST.fullmatch(binding.operation_digest) is None
        ):
            raise ParticipantDivergedError("Semantic transaction binding is invalid")
        pending = cls._load_pending(store, binding.transaction_id)
        receipt = cls._load_receipt(store, binding.transaction_id)
        if receipt is not None:
            cls._operation_from_evidence(
                receipt,
                binding.transaction_id,
                binding.participant_id,
                binding.operation_digest,
            )
            raise ParticipantDivergedError("Committed Semantic batch cannot be aborted")
        if pending is None:
            return AbortOutcome.ALREADY_ABSENT
        participant = cls.from_pending(
            memory,
            store,
            binding.transaction_id,
            binding.participant_id,
            binding.operation_digest,
            event_id=binding.event_id,
            processing_sequence=binding.processing_sequence,
        )
        return participant.abort(binding)

    def finalize(self, binding: TransactionBinding) -> ParticipantOutcome:
        self._validate_binding(binding)
        self._validate_event_identity(binding.event_id, binding.processing_sequence)
        self._validate_sources()
        expected = self._artifact(binding, SEMANTIC_PENDING_SCHEMA_VERSION)
        pending = self._load_pending(self.store, binding.transaction_id)
        receipt = self._load_receipt(self.store, binding.transaction_id)
        if pending is not None and pending != expected:
            raise ParticipantDivergedError("Semantic pending artifact conflicts")
        if receipt is not None:
            self._operation_from_evidence(
                receipt,
                binding.transaction_id,
                binding.participant_id,
                binding.operation_digest,
            )
        if pending is None and receipt is None:
            raise ParticipantUnavailableError("Semantic pending artifact is absent")
        already_consistent = receipt is not None
        for entry in self.operation.entries:
            if not self._publish_entry(entry):
                already_consistent = False
        self._ensure_receipt(binding)
        for entry in self.operation.entries:
            self._project_entry(entry)
        self._remove_pending(binding.transaction_id)
        return (
            ParticipantOutcome.ALREADY_CONSISTENT
            if already_consistent
            else ParticipantOutcome.FINALIZED
        )

    def abort(self, binding: TransactionBinding) -> AbortOutcome:
        self._validate_binding(binding)
        self._validate_event_identity(binding.event_id, binding.processing_sequence)
        pending = self._load_pending(self.store, binding.transaction_id)
        receipt = self._load_receipt(self.store, binding.transaction_id)
        if receipt is not None:
            raise ParticipantDivergedError("Committed Semantic batch cannot be aborted")
        if pending is None:
            return AbortOutcome.ALREADY_ABSENT
        expected = self._artifact(binding, SEMANTIC_PENDING_SCHEMA_VERSION)
        if pending != expected:
            raise ParticipantDivergedError("Semantic pending artifact conflicts")
        if self._any_target_committed():
            raise ParticipantDivergedError("Published Semantic revision cannot be aborted")
        self._remove_pending(binding.transaction_id)
        return AbortOutcome.ABORTED

    def inspect_reconciliation(
        self, binding: TransactionBinding
    ) -> StartupParticipantOutcome:
        self._validate_binding(binding)
        self._validate_event_identity(binding.event_id, binding.processing_sequence)
        pending = self._load_pending(self.store, binding.transaction_id)
        receipt = self._load_receipt(self.store, binding.transaction_id)
        expected = self._artifact(binding, SEMANTIC_PENDING_SCHEMA_VERSION)
        if pending is not None and pending != expected:
            raise ParticipantDivergedError("Semantic pending artifact conflicts")
        if receipt is not None:
            self._operation_from_evidence(
                receipt,
                binding.transaction_id,
                binding.participant_id,
                binding.operation_digest,
            )
        elif pending is None:
            raise UnsupportedParticipantReconciliationError(
                "Semantic operation evidence is absent"
            )
        self._validate_sources()
        all_lifecycle = all(self._target_committed(entry) for entry in self.operation.entries)
        if not all_lifecycle:
            raise ParticipantUnavailableError("Semantic lifecycle roll-forward is required")
        for entry in self.operation.entries:
            self._inspect_projection(entry)
        return StartupParticipantOutcome.VERIFIED_CONSISTENT

    def reconcile(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        try:
            outcome = self.inspect_reconciliation(binding)
        except ParticipantUnavailableError:
            self.finalize(binding)
            return StartupParticipantOutcome.ROLLED_FORWARD
        self.finalize(binding)
        return outcome

    def _validate_binding(self, binding: TransactionBinding) -> None:
        if (
            not validate_transaction_binding(binding)
            or binding.participant_id != self.participant_id
            or binding.operation_digest != self.operation_digest
        ):
            raise ParticipantDivergedError("Semantic transaction binding is invalid")

    def _validate_event_identity(self, event_id: str, processing_sequence: int) -> None:
        if type(processing_sequence) is not int or processing_sequence <= 0:
            raise ParticipantDivergedError("Semantic event sequence is invalid")
        for entry in self.operation.entries:
            revision = entry.mutation.revision
            if revision.event_id != event_id or revision.event_sequence != processing_sequence:
                raise ParticipantDivergedError("Semantic revision event binding conflicts")

    def _validate_sources(self) -> None:
        for entry in self.operation.entries:
            revision = entry.mutation.revision
            for edge in revision.source_edges:
                if isinstance(entry.mutation, SemanticCreateIntent):
                    if edge.source_kind is SemanticSourceKind.EPISODIC and edge.source_status is not SemanticSourceStatus.AVAILABLE:
                        raise ParticipantDivergedError("Semantic create source is not available")
                if edge.source_kind is SemanticSourceKind.EPISODIC and edge.source_status is SemanticSourceStatus.AVAILABLE:
                    try:
                        committed = self.memory.get_committed_episodic(edge.source_id)
                    except EpisodicMemoryReadError:
                        raise ParticipantUnavailableError("Committed Semantic source is unavailable") from None
                    except EpisodicMemoryFormatError:
                        raise ParticipantDivergedError("Committed Semantic source is invalid") from None
                    if committed is None:
                        raise ParticipantUnavailableError("Committed Semantic source is absent")
                    if committed.record.context_id != edge.captured_context_id:
                        raise ParticipantDivergedError("Semantic source Context binding conflicts")

    def _validate_revision_preconditions(self) -> None:
        for entry in self.operation.entries:
            current = self._current(entry.semantic_id)
            mutation = entry.mutation
            if isinstance(mutation, SemanticCreateIntent):
                if current is None:
                    continue
                if self._entry_matches_target(current, mutation.revision):
                    continue
                if self._historical_entry_matches(entry):
                    continue
                raise ParticipantDivergedError("Semantic create identity conflicts")
            if current is None:
                raise ParticipantUnavailableError("Semantic revision target is absent")
            if self._entry_matches_target(current, mutation.revision):
                continue
            if current.revision.revision == mutation.expected_revision:
                if current.revision.revision_digest != mutation.expected_revision_digest:
                    raise ParticipantDivergedError("Semantic revision target is stale")
                if current.revision.lifecycle is not SemanticLifecycle.ACTIVE:
                    raise ParticipantDivergedError("Only active Semantic revisions may advance")
                continue
            if self._historical_entry_matches(entry):
                continue
            raise ParticipantDivergedError("Semantic revision target is stale")

    def _current(self, semantic_id: str) -> SemanticStoredEntry | None:
        try:
            return self.store.load_current(semantic_id)
        except SemanticStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _historical_entry_matches(self, entry: SemanticBatchEntry) -> bool:
        try:
            retained = self.store.load_revision(entry.semantic_id, entry.mutation.revision.revision)
        except SemanticStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None
        return retained is not None and self._entry_matches_target(
            retained, entry.mutation.revision
        ) and retained.operation_digest == self.operation_digest

    def _entry_matches_target(
        self, current: SemanticStoredEntry, revision: SemanticRevision
    ) -> bool:
        return (
            current.revision == revision
            and current.revision.revision_digest == revision.revision_digest
            and current.operation_digest == self.operation_digest
        )

    def _target_committed(self, entry: SemanticBatchEntry) -> bool:
        current = self._current(entry.semantic_id)
        if current is not None and self._entry_matches_target(current, entry.mutation.revision):
            return True
        return self._historical_entry_matches(entry)

    def _any_target_committed(self) -> bool:
        return any(self._target_committed(entry) for entry in self.operation.entries)

    def _publish_entry(self, entry: SemanticBatchEntry) -> bool:
        if self._target_committed(entry):
            return True
        mutation = entry.mutation
        try:
            if isinstance(mutation, SemanticCreateIntent):
                self.store.publish_create(mutation.revision, self.operation_digest)
            else:
                self.store.publish_revision(
                    mutation.revision,
                    self.operation_digest,
                    expected_revision=mutation.expected_revision,
                    expected_digest=mutation.expected_revision_digest,
                )
        except (SemanticStoreConflict, SemanticStoreCorrupt) as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None
        return False

    def _ensure_receipt(self, binding: TransactionBinding) -> None:
        expected = self._artifact(binding, SEMANTIC_RECEIPT_SCHEMA_VERSION)
        existing = self._load_receipt(self.store, binding.transaction_id)
        if existing is not None:
            if existing != expected:
                raise ParticipantDivergedError("Semantic receipt conflicts")
            return
        try:
            self.store.write_receipt(binding.transaction_id, expected)
        except (SemanticStoreCorrupt, SemanticStoreConflict) as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _project_entry(self, entry: SemanticBatchEntry) -> None:
        try:
            self.memory.project_semantic_revision(entry.mutation.revision, self.store)
        except SemanticMemoryFormatError as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticMemoryReadError as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _inspect_projection(self, entry: SemanticBatchEntry) -> None:
        try:
            inspection = self.memory.inspect_semantic_projection(
                entry.semantic_id, entry.mutation.revision, self.store
            )
        except SemanticMemoryFormatError as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticMemoryReadError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if inspection.status is not SemanticProjectionStatus.EXACT:
            raise ParticipantUnavailableError("Semantic DB2 projection repair is required")

    def _artifact(self, binding: TransactionBinding, schema_version: int) -> dict[str, object]:
        return {
            "operation": self.operation.canonical_dict(),
            "operation_digest": self.operation_digest,
            "participant_id": self.participant_id,
            "schema_version": schema_version,
            "transaction_id": binding.transaction_id,
        }

    def _remove_pending(self, transaction_id: str) -> None:
        try:
            self.store.remove_pending(transaction_id)
        except (SemanticStoreCorrupt, SemanticStoreConflict) as error:
            raise ParticipantDivergedError(str(error)) from None
        except SemanticStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None

__all__ = [
    "MEMORY_SEMANTIC_PARTICIPANT_ID",
    "SEMANTIC_MAX_BATCH_ENTRIES",
    "MemorySemanticParticipant",
    "SemanticBatchEntry",
    "SemanticBatchOperation",
    "SemanticCreateIntent",
    "SemanticMutationKind",
    "SemanticRevisionIntent",
    "semantic_batch_operation_digest",
    "semantic_id_for_batch_entry",
]
