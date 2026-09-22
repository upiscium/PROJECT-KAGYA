"""R07 participant for Memory-owned Experience evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import re
from uuid import UUID, uuid5

from kagya.experience import (
    ExperienceLifecycle,
    ExperienceRecord,
    ExperienceRevisionRecord,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    experience_record_digest,
)
from kagya.memory.dual_memory_system import (
    DualMemorySystem,
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
)
from kagya.memory.episodic_participant import (
    MEMORY_EPISODIC_PARTICIPANT_ID,
    MemoryEpisodicParticipant,
    episodic_episode_id,
)
from kagya.memory.experience_store import (
    EXPERIENCE_PENDING_SCHEMA_VERSION,
    EXPERIENCE_RECEIPT_SCHEMA_VERSION,
    ExperienceStore,
    ExperienceStoreConflict,
    ExperienceStoreCorrupt,
    ExperienceStoreError,
    ExperienceStoreUnavailable,
    ExperienceStoredEntry,
    experience_record_from_dict,
    experience_record_to_dict,
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


MEMORY_EXPERIENCE_PARTICIPANT_ID = "memory.experience"
_EXPERIENCE_ID_NAMESPACE = UUID("0f23fc3b-5b8a-5ea1-9701-e4c8f7a5a5ee")
_OPERATION_DOMAIN = b"PROJECT-KAGYA:R12:EXPERIENCE-PARTICIPANT:V1\0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def experience_id_for_event(event_id: str, processing_sequence: int) -> str:
    """Derive exactly one stable Experience identity for a subject event."""

    validate_identifier(event_id)
    if type(processing_sequence) is not int or processing_sequence <= 0:
        raise ValueError("Experience event sequence is invalid")
    canonical = json.dumps(
        [event_id, processing_sequence],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"experience-{uuid5(_EXPERIENCE_ID_NAMESPACE, canonical)}"


def _datetime_value(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ParticipantDivergedError("Experience operation timestamp is invalid")
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        raise ParticipantDivergedError("Experience operation timestamp is invalid") from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ParticipantDivergedError("Experience operation timestamp is invalid")
    return result.astimezone(UTC)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class ExperienceCreateIntent:
    record: ExperienceRecord
    source_episode_operation_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.record, ExperienceRecord):
            raise TypeError("Experience create record is invalid")
        if self.record.revision != 0 or self.record.lifecycle is not ExperienceLifecycle.ACTIVE:
            raise ValueError("Experience create must be an active revision zero")
        if _DIGEST.fullmatch(self.source_episode_operation_digest) is None:
            raise ValueError("Source episode operation digest is invalid")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "kind": "create",
            "record": experience_record_to_dict(self.record),
            "source_episode_operation_digest": self.source_episode_operation_digest,
        }


@dataclass(frozen=True, slots=True)
class ExperienceRevisionIntent:
    record: ExperienceRecord
    revision_record: ExperienceRevisionRecord
    expected_revision: int
    expected_record_digest: str
    source_episode_operation_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.record, ExperienceRecord):
            raise TypeError("Experience revision record is invalid")
        if not isinstance(self.revision_record, ExperienceRevisionRecord):
            raise TypeError("Experience revision evidence is invalid")
        if self.record.revision <= 0:
            raise ValueError("Experience revision must be positive")
        if self.revision_record.experience_id != self.record.experience_id:
            raise ValueError("Experience revision identity is inconsistent")
        if self.revision_record.revision != self.record.revision:
            raise ValueError("Experience revision evidence number is inconsistent")
        if type(self.expected_revision) is not int or self.expected_revision < 0:
            raise ValueError("Expected Experience revision is invalid")
        if self.record.revision != self.expected_revision + 1:
            raise ValueError("Experience revision does not advance the expected target")
        if _DIGEST.fullmatch(self.expected_record_digest) is None:
            raise ValueError("Expected Experience digest is invalid")
        if _DIGEST.fullmatch(self.source_episode_operation_digest) is None:
            raise ValueError("Source episode operation digest is invalid")
        if (
            len(self.record.revision_history) < 2
            or self.revision_record.previous_revision_digest
            != self.record.revision_history[-2].record_digest
        ):
            raise ValueError("Experience revision evidence is not linked to history")
        if self.record.revision_history[-1] != self.revision_record:
            raise ValueError("Experience revision evidence is not the committed revision")
        expected_lifecycle = {
            ExperienceRevisionOperation.REASSESS: ExperienceLifecycle.ACTIVE,
            ExperienceRevisionOperation.CORRECT: ExperienceLifecycle.ACTIVE,
            ExperienceRevisionOperation.SUPERSEDE: ExperienceLifecycle.SUPERSEDED,
            ExperienceRevisionOperation.RETRACT: ExperienceLifecycle.RETRACTED,
        }[self.revision_record.operation]
        if self.record.lifecycle is not expected_lifecycle:
            raise ValueError("Experience revision lifecycle does not match its operation")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "expected_record_digest": self.expected_record_digest,
            "expected_revision": self.expected_revision,
            "kind": "revision",
            "record": experience_record_to_dict(self.record),
            "revision_record": {
                "created_at": _datetime_value(self.revision_record.created_at),
                "event_id": self.revision_record.event_id,
                "event_sequence": self.revision_record.event_sequence,
                "evidence_refs": list(self.revision_record.evidence_refs),
                "experience_id": self.revision_record.experience_id,
                "operation": self.revision_record.operation.value,
                "previous_revision_digest": self.revision_record.previous_revision_digest,
                "reason": self.revision_record.reason.value,
                "record_digest": self.revision_record.record_digest,
                "revision": self.revision_record.revision,
            },
            "source_episode_operation_digest": self.source_episode_operation_digest,
        }


ExperienceOperation = ExperienceCreateIntent | ExperienceRevisionIntent


def experience_operation_digest(operation: ExperienceOperation) -> str:
    return hashlib.sha256(_OPERATION_DOMAIN + _canonical_json(operation.canonical_dict())).hexdigest()


def _operation_from_dict(value: object) -> ExperienceOperation:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ParticipantDivergedError("Experience pending operation is invalid")
    kind = value["kind"]
    if kind == "create":
        if set(value) != {"kind", "record", "source_episode_operation_digest"}:
            raise ParticipantDivergedError("Experience pending operation is invalid")
        try:
            return ExperienceCreateIntent(
                experience_record_from_dict(value["record"]),
                value["source_episode_operation_digest"],
            )
        except (TypeError, ValueError, ExperienceStoreError):
            raise ParticipantDivergedError("Experience pending operation is invalid") from None
    if kind != "revision":
        raise ParticipantDivergedError("Experience pending operation is invalid")
    if set(value) != {
        "expected_record_digest",
        "expected_revision",
        "kind",
        "record",
        "revision_record",
        "source_episode_operation_digest",
    }:
        raise ParticipantDivergedError("Experience pending operation is invalid")
    revision = value["revision_record"]
    if not isinstance(revision, dict):
        raise ParticipantDivergedError("Experience pending operation is invalid")
    try:
        revision_record = ExperienceRevisionRecord(
            experience_id=revision["experience_id"],
            revision=revision["revision"],
            operation=ExperienceRevisionOperation(revision["operation"]),
            reason=ExperienceRevisionReason(revision["reason"]),
            created_at=_parse_datetime(revision["created_at"]),
            event_id=revision["event_id"],
            event_sequence=revision["event_sequence"],
            evidence_refs=tuple(revision["evidence_refs"]),
            previous_revision_digest=revision["previous_revision_digest"],
        )
        if revision["record_digest"] != revision_record.record_digest:
            raise ValueError
        return ExperienceRevisionIntent(
            record=experience_record_from_dict(value["record"]),
            revision_record=revision_record,
            expected_revision=value["expected_revision"],
            expected_record_digest=value["expected_record_digest"],
            source_episode_operation_digest=value["source_episode_operation_digest"],
        )
    except (KeyError, TypeError, ValueError, ExperienceStoreError):
        raise ParticipantDivergedError("Experience pending operation is invalid") from None


class MemoryExperienceParticipant:
    """Prepare/finalize/abort authority for one Experience operation."""

    participant_id = MEMORY_EXPERIENCE_PARTICIPANT_ID
    capabilities = (
        ParticipantCapability.ABORT,
        ParticipantCapability.IDEMPOTENT_FINALIZE,
        ParticipantCapability.INSPECT_RECONCILE,
        ParticipantCapability.PREPARE,
    )

    def __init__(
        self,
        memory: DualMemorySystem,
        store: ExperienceStore,
        operation: ExperienceOperation,
    ) -> None:
        self.memory = memory
        self.store = store
        self.operation = operation
        self.operation_digest = experience_operation_digest(operation)

    @classmethod
    def from_pending(
        cls,
        memory: DualMemorySystem,
        store: ExperienceStore,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
        *,
        event_id: str | None = None,
        processing_sequence: int | None = None,
    ) -> MemoryExperienceParticipant:
        if (
            participant_id != MEMORY_EXPERIENCE_PARTICIPANT_ID
            or _DIGEST.fullmatch(operation_digest) is None
        ):
            raise ParticipantDivergedError("Experience binding is invalid")
        try:
            UUID(transaction_id)
        except (TypeError, ValueError):
            raise ParticipantDivergedError("Experience transaction identity is invalid") from None
        pending = cls._load_pending(store, transaction_id)
        if pending is not None:
            operation = cls._operation_from_pending_payload(
                pending, transaction_id, participant_id, operation_digest
            )
            participant = cls(memory, store, operation)
            if participant.operation_digest != operation_digest:
                raise ParticipantDivergedError("Experience pending digest conflicts")
            if event_id is not None and processing_sequence is not None:
                participant._validate_record_identity(
                    transaction_id, event_id, processing_sequence
                )
            current = participant._current_entry()
            if current is not None:
                if isinstance(participant.operation, ExperienceRevisionIntent):
                    if not participant._operation_matches_current(current):
                        if current.record.revision == participant.operation.expected_revision:
                            participant._validate_revision_target(current)
                        else:
                            participant._validate_committed_artifact()
                else:
                    if current.record.revision == 0:
                        participant._ensure_current_matches(current)
                    else:
                        participant._validate_committed_artifact()
            return participant
        receipt = cls._load_receipt(store, transaction_id)
        if receipt is not None:
            operation = cls._operation_from_receipt(
                receipt, transaction_id, participant_id, operation_digest
            )
            participant = cls(memory, store, operation)
            if event_id is not None and processing_sequence is not None:
                participant._validate_record_identity(
                    transaction_id, event_id, processing_sequence
                )
            participant._validate_committed_artifact()
            return participant
        if event_id is None or processing_sequence is None:
            raise UnsupportedParticipantReconciliationError(
                "Experience operation evidence is absent"
            )
        experience_id = experience_id_for_event(event_id, processing_sequence)
        try:
            current = store.load_current(experience_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if current is None:
            raise UnsupportedParticipantReconciliationError(
                "Experience operation evidence is absent"
            )
        if current.operation_digest != operation_digest:
            raise ParticipantDivergedError("Committed Experience digest conflicts")
        if current.record.revision != 0:
            raise UnsupportedParticipantReconciliationError(
                "Committed Experience revision cannot reconstruct a create"
            )
        operation = ExperienceCreateIntent(
            current.record, current.source_episode_operation_digest
        )
        participant = cls(memory, store, operation)
        participant._validate_record_identity(
            transaction_id, event_id, processing_sequence
        )
        participant._ensure_current_matches(current)
        return participant

    @staticmethod
    def _load_pending(
        store: ExperienceStore, transaction_id: str
    ) -> dict[str, object] | None:
        try:
            return store.load_pending(transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    @staticmethod
    def _load_receipt(
        store: ExperienceStore, transaction_id: str
    ) -> dict[str, object] | None:
        try:
            return store.load_receipt(transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    @staticmethod
    def _operation_from_pending_payload(
        payload: dict[str, object],
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
    ) -> ExperienceOperation:
        expected = {
            "operation",
            "operation_digest",
            "participant_id",
            "schema_version",
            "transaction_id",
        }
        if (
            set(payload) != expected
            or payload["schema_version"] != EXPERIENCE_PENDING_SCHEMA_VERSION
            or payload["transaction_id"] != transaction_id
            or payload["participant_id"] != participant_id
            or payload["operation_digest"] != operation_digest
        ):
            raise ParticipantDivergedError("Experience pending artifact conflicts")
        operation = _operation_from_dict(payload["operation"])
        if experience_operation_digest(operation) != operation_digest:
            raise ParticipantDivergedError("Experience pending digest conflicts")
        return operation

    @staticmethod
    def _operation_from_receipt(
        payload: dict[str, object],
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
    ) -> ExperienceOperation:
        expected = {
            "operation",
            "operation_digest",
            "participant_id",
            "schema_version",
            "transaction_id",
        }
        if (
            set(payload) != expected
            or payload["schema_version"] != EXPERIENCE_RECEIPT_SCHEMA_VERSION
            or payload["transaction_id"] != transaction_id
            or payload["participant_id"] != participant_id
            or payload["operation_digest"] != operation_digest
        ):
            raise ParticipantDivergedError("Experience receipt conflicts")
        operation = _operation_from_dict(payload["operation"])
        if experience_operation_digest(operation) != operation_digest:
            raise ParticipantDivergedError("Experience receipt digest conflicts")
        return operation

    def prepare(self, binding: TransactionBinding) -> None:
        self._validate_binding(binding)
        self._validate_record_identity(binding.transaction_id, binding.event_id, binding.processing_sequence)
        self._validate_source_operation(binding)
        expected = self._artifact(binding)
        try:
            current = self._current_entry()
            self._ensure_receipt(binding)
            if current is not None:
                if self._operation_matches_current(current):
                    pass
                elif isinstance(self.operation, ExperienceRevisionIntent):
                    if current.record.revision == self.operation.expected_revision:
                        self._validate_revision_target(current)
                    else:
                        self._validate_committed_artifact()
                else:
                    if current.record.revision == 0:
                        self._ensure_current_matches(current)
                    else:
                        self._validate_committed_artifact()
                pending = self.store.load_pending(binding.transaction_id)
                if pending is not None and pending != expected:
                    raise ParticipantDivergedError("Experience pending artifact conflicts")
                if (
                    pending is None
                    and isinstance(self.operation, ExperienceRevisionIntent)
                    and not self._operation_matches_current(current)
                ):
                    self.store.write_pending(binding.transaction_id, expected)
                return
            if isinstance(self.operation, ExperienceRevisionIntent):
                raise ParticipantUnavailableError("Experience revision target is absent")
            pending = self.store.load_pending(binding.transaction_id)
            if pending is not None and pending != expected:
                raise ParticipantDivergedError("Experience pending artifact conflicts")
            if pending is None:
                self.store.write_pending(binding.transaction_id, expected)
        except (ExperienceStoreCorrupt, ExperienceStoreConflict) as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreUnavailable as error:
            raise ParticipantUnavailableError(str(error)) from None

    @classmethod
    def abort_pending(
        cls,
        memory: DualMemorySystem,
        store: ExperienceStore,
        binding: TransactionBinding,
    ) -> AbortOutcome:
        if (
            not validate_transaction_binding(binding)
            or binding.participant_id != MEMORY_EXPERIENCE_PARTICIPANT_ID
            or _DIGEST.fullmatch(binding.operation_digest) is None
        ):
            raise ParticipantDivergedError("Experience transaction binding is invalid")
        try:
            pending = store.load_pending(binding.transaction_id)
            receipt = cls._load_receipt(store, binding.transaction_id)
            current = None
            if receipt is not None:
                operation = cls._operation_from_receipt(
                    receipt,
                    binding.transaction_id,
                    binding.participant_id,
                    binding.operation_digest,
                )
                current = store.load_current(operation.record.experience_id)
            else:
                current = store.load_current(
                    experience_id_for_event(binding.event_id, binding.processing_sequence)
                )
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if current is not None and pending is None:
            raise ParticipantDivergedError("Committed Experience cannot be aborted")
        if pending is None:
            if receipt is not None:
                try:
                    store.remove_receipt(binding.transaction_id)
                except ExperienceStoreCorrupt as error:
                    raise ParticipantDivergedError(str(error)) from None
                except ExperienceStoreError as error:
                    raise ParticipantUnavailableError(str(error)) from None
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
        self._validate_record_identity(binding.transaction_id, binding.event_id, binding.processing_sequence)
        self._validate_source_committed(binding)
        expected = self._artifact(binding)
        try:
            current = self._current_entry()
            pending = self.store.load_pending(binding.transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if current is not None:
            if self._operation_matches_current(current):
                if pending is not None and pending != expected:
                    raise ParticipantDivergedError("Experience pending artifact conflicts")
                self._ensure_receipt(binding)
                self._reconcile_prune()
                if pending is not None:
                    self._remove_pending(binding.transaction_id)
                self._prune_receipts(binding.transaction_id)
                return ParticipantOutcome.ALREADY_CONSISTENT
            if not isinstance(self.operation, ExperienceRevisionIntent):
                if current.record.revision == 0:
                    self._ensure_current_matches(current)
                else:
                    self._validate_committed_artifact()
                    if pending is not None and pending != expected:
                        raise ParticipantDivergedError("Experience pending artifact conflicts")
                    self._reconcile_prune()
                    if pending is not None:
                        self._remove_pending(binding.transaction_id)
                    self._prune_receipts(binding.transaction_id)
                    return ParticipantOutcome.ALREADY_CONSISTENT
            elif current.record.revision == self.operation.expected_revision:
                self._validate_revision_target(current)
            else:
                self._validate_committed_artifact()
                if pending is not None and pending != expected:
                    raise ParticipantDivergedError("Experience pending artifact conflicts")
                self._reconcile_prune()
                if pending is not None:
                    self._remove_pending(binding.transaction_id)
                self._prune_receipts(binding.transaction_id)
                return ParticipantOutcome.ALREADY_CONSISTENT
        if pending != expected:
            if pending is None:
                raise ParticipantUnavailableError("Experience pending artifact is absent")
            raise ParticipantDivergedError("Experience pending artifact conflicts")
        self._ensure_receipt(binding)
        try:
            if isinstance(self.operation, ExperienceCreateIntent):
                self.store.publish_create(
                    self.operation.record,
                    self.operation_digest,
                    self.operation.source_episode_operation_digest,
                )
            else:
                self.store.publish_revision(
                    self.operation.record,
                    self.operation_digest,
                    self.operation.source_episode_operation_digest,
                    expected_revision=self.operation.expected_revision,
                    expected_digest=self.operation.expected_record_digest,
                )
        except ExperienceStoreConflict as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        self._remove_pending(binding.transaction_id)
        self._prune_receipts(binding.transaction_id)
        return ParticipantOutcome.FINALIZED

    def abort(self, binding: TransactionBinding) -> AbortOutcome:
        self._validate_binding(binding)
        self._validate_record_identity(binding.transaction_id, binding.event_id, binding.processing_sequence)
        try:
            current = self._current_entry()
            pending = self.store.load_pending(binding.transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if current is not None:
            if not isinstance(self.operation, ExperienceRevisionIntent):
                raise ParticipantDivergedError("Committed Experience cannot be aborted")
            if self._operation_matches_current(current):
                raise ParticipantDivergedError("Committed Experience cannot be aborted")
            if current.record.revision == self.operation.expected_revision:
                self._validate_revision_target(current)
            else:
                self._validate_committed_artifact()
                raise ParticipantDivergedError("Committed Experience cannot be aborted")
        if pending is None:
            self._remove_receipt(binding.transaction_id)
            return AbortOutcome.ALREADY_ABSENT
        if pending != self._artifact(binding):
            raise ParticipantDivergedError("Experience pending artifact conflicts")
        self._remove_pending(binding.transaction_id)
        self._remove_receipt(binding.transaction_id)
        return AbortOutcome.ABORTED

    def inspect_reconciliation(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        self._validate_binding(binding)
        self._validate_record_identity(binding.transaction_id, binding.event_id, binding.processing_sequence)
        try:
            current = self._current_entry()
            pending = self.store.load_pending(binding.transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if current is not None:
            if isinstance(self.operation, ExperienceRevisionIntent):
                if not self._operation_matches_current(current):
                    if current.record.revision == self.operation.expected_revision:
                        self._validate_revision_target(current)
                    else:
                        self._validate_committed_artifact()
            else:
                if current.record.revision == 0:
                    self._ensure_current_matches(current)
                else:
                    self._validate_committed_artifact()
            if pending is not None and pending != self._artifact(binding):
                raise ParticipantDivergedError("Experience pending artifact conflicts")
            return StartupParticipantOutcome.VERIFIED_CONSISTENT
        if pending == self._artifact(binding):
            raise ParticipantUnavailableError("Experience roll-forward is required")
        if pending is None:
            raise UnsupportedParticipantReconciliationError(
                "Experience operation evidence is absent"
            )
        raise ParticipantDivergedError("Experience pending artifact conflicts")

    def reconcile(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        outcome = self.inspect_reconciliation(binding)
        if outcome is StartupParticipantOutcome.VERIFIED_CONSISTENT:
            pending = self.store.load_pending(binding.transaction_id)
            self._reconcile_prune()
            if pending is not None:
                self._remove_pending(binding.transaction_id)
            self._prune_receipts(binding.transaction_id)
            return outcome
        self.finalize(binding)
        return StartupParticipantOutcome.ROLLED_FORWARD

    def _validate_binding(self, binding: TransactionBinding) -> None:
        if (
            not validate_transaction_binding(binding)
            or binding.participant_id != self.participant_id
            or binding.operation_digest != self.operation_digest
        ):
            raise ParticipantDivergedError("Experience transaction binding is invalid")

    def _validate_record_identity(
        self,
        transaction_id: str,
        event_id: str,
        processing_sequence: int | None,
    ) -> None:
        record = self.operation.record
        if processing_sequence is None:
            return
        if isinstance(self.operation, ExperienceCreateIntent):
            if (
                record.source_event_id != event_id
                or record.source_event_sequence != processing_sequence
                or record.experience_id
                != experience_id_for_event(event_id, processing_sequence)
            ):
                raise ParticipantDivergedError("Experience event identity is invalid")
            expected_episode = episodic_episode_id(
                transaction_id,
                MEMORY_EPISODIC_PARTICIPANT_ID,
                self.operation.source_episode_operation_digest,
            )
        else:
            if (
                self.operation.revision_record.event_id != event_id
                or self.operation.revision_record.event_sequence != processing_sequence
            ):
                raise ParticipantDivergedError("Experience revision event identity is invalid")
            expected_episode = record.source_episode_id
        if record.source_episode_id != expected_episode:
            raise ParticipantDivergedError("Experience source episode identity is invalid")

    def _validate_source_operation(self, binding: TransactionBinding) -> None:
        if isinstance(self.operation, ExperienceRevisionIntent):
            return
        source_digest = self.operation.source_episode_operation_digest
        try:
            episodic = MemoryEpisodicParticipant.from_pending(
                self.memory,
                binding.transaction_id,
                MEMORY_EPISODIC_PARTICIPANT_ID,
                source_digest,
            )
        except (ParticipantDivergedError, UnsupportedParticipantReconciliationError):
            raise
        except ParticipantUnavailableError:
            raise
        if episodic.operation_digest != source_digest:
            raise ParticipantDivergedError("Source episode operation digest conflicts")
        if episodic.episode_id(binding.transaction_id) != self.operation.record.source_episode_id:
            raise ParticipantDivergedError("Source episode identity conflicts")

    def _validate_source_committed(self, binding: TransactionBinding) -> None:
        if isinstance(self.operation, ExperienceRevisionIntent):
            try:
                committed = self.memory.get_committed_episodic(
                    self.operation.record.source_episode_id
                )
            except EpisodicMemoryReadError:
                raise ParticipantUnavailableError(
                    "Committed source episode is unavailable"
                ) from None
            except EpisodicMemoryFormatError:
                raise ParticipantDivergedError(
                    "Committed source episode is invalid"
                ) from None
            if committed is None:
                raise ParticipantUnavailableError("Committed source episode is absent")
            return
        self._validate_source_operation(binding)
        try:
            committed = self.memory.get_committed_episodic(self.operation.record.source_episode_id)
        except EpisodicMemoryReadError:
            raise ParticipantUnavailableError("Committed source episode is unavailable") from None
        except EpisodicMemoryFormatError:
            raise ParticipantDivergedError("Committed source episode is invalid") from None
        if committed is None or committed.record.id != self.operation.record.source_episode_id:
            raise ParticipantUnavailableError("Committed source episode is absent")

    def _artifact(self, binding: TransactionBinding) -> dict[str, object]:
        return {
            "operation": self.operation.canonical_dict(),
            "operation_digest": self.operation_digest,
            "participant_id": self.participant_id,
            "schema_version": EXPERIENCE_PENDING_SCHEMA_VERSION,
            "transaction_id": binding.transaction_id,
        }

    def _ensure_receipt(self, binding: TransactionBinding) -> None:
        expected = {
            "operation": self.operation.canonical_dict(),
            "operation_digest": self.operation_digest,
            "participant_id": self.participant_id,
            "schema_version": EXPERIENCE_RECEIPT_SCHEMA_VERSION,
            "transaction_id": binding.transaction_id,
        }
        try:
            existing = self.store.load_receipt(binding.transaction_id)
            if existing is not None and existing != expected:
                raise ParticipantDivergedError("Experience receipt conflicts")
            if existing is None:
                self.store.write_receipt(binding.transaction_id, expected)
        except ParticipantDivergedError:
            raise
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreConflict as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _current_entry(self) -> ExperienceStoredEntry | None:
        try:
            return self.store.load_current(self.operation.record.experience_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _ensure_current_matches(self, current: ExperienceStoredEntry) -> None:
        if (
            current.record != self.operation.record
            or current.operation_digest != self.operation_digest
            or current.source_episode_operation_digest
            != self.operation.source_episode_operation_digest
        ):
            raise ParticipantDivergedError("Committed Experience conflicts")

    def _operation_matches_current(self, current: ExperienceStoredEntry) -> bool:
        return self._operation_matches_entry(current)

    def _operation_matches_entry(self, entry: ExperienceStoredEntry) -> bool:
        return (
            entry.record == self.operation.record
            and entry.operation_digest == self.operation_digest
            and entry.source_episode_operation_digest
            == self.operation.source_episode_operation_digest
        )

    def _validate_committed_artifact(self) -> ExperienceStoredEntry:
        try:
            entry = self.store.load_revision(
                self.operation.record.experience_id,
                self.operation.record.revision,
            )
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
        if entry is None:
            raise UnsupportedParticipantReconciliationError(
                "Committed Experience revision artifact is absent"
            )
        if not self._operation_matches_entry(entry):
            raise ParticipantDivergedError("Committed Experience revision conflicts")
        return entry

    def _validate_revision_target(self, current: ExperienceStoredEntry) -> None:
        if not isinstance(self.operation, ExperienceRevisionIntent):
            return
        if (
            current.record.revision != self.operation.expected_revision
            or experience_record_digest(current.record)
            != self.operation.expected_record_digest
        ):
            raise ParticipantDivergedError("Experience revision target is stale")
        if current.record.lifecycle is not ExperienceLifecycle.ACTIVE:
            raise ParticipantDivergedError("Only active Experiences may be revised")
        revised = self.operation.record
        if (
            revised.experience_id != current.record.experience_id
            or revised.source_event_id != current.record.source_event_id
            or revised.source_event_sequence != current.record.source_event_sequence
            or revised.source_episode_id != current.record.source_episode_id
            or revised.context_id != current.record.context_id
            or revised.created_at != current.record.created_at
            or self.operation.source_episode_operation_digest
            != current.source_episode_operation_digest
        ):
            raise ParticipantDivergedError("Experience source lineage conflicts")

    def _remove_pending(self, transaction_id: str) -> None:
        try:
            self.store.remove_pending(transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _remove_receipt(self, transaction_id: str) -> None:
        try:
            self.store.remove_receipt(transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _prune_receipts(self, protected_transaction_id: str) -> None:
        try:
            self.store.prune_receipts(protected_transaction_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None

    def _reconcile_prune(self) -> None:
        try:
            self.store.reconcile_prune(self.operation.record.experience_id)
        except ExperienceStoreCorrupt as error:
            raise ParticipantDivergedError(str(error)) from None
        except ExperienceStoreError as error:
            raise ParticipantUnavailableError(str(error)) from None
