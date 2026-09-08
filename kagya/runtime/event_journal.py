"""Durable, integrity-chained lifecycle evidence for agent events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from kagya.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType


CURRENT_EVENT_JOURNAL_SCHEMA_VERSION: Literal[3] = 3
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN_V1 = b"PROJECT-KAGYA:event-journal:v1\0"
_HASH_DOMAIN_V2 = b"PROJECT-KAGYA:event-journal:v2\0"
_HASH_DOMAIN_V3 = b"PROJECT-KAGYA:event-journal:v3\0"
_PARTICIPANT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9]+)*$")


class EventLifecycle(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    PREPARED = "prepared"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERY_CLASSIFIED = "recovery_classified"
    CHECKPOINT = "checkpoint"
    RECOVERY_PREPARED = "recovery_prepared"
    RECOVERY_COMPLETED = "recovery_completed"
    TRANSACTION_PREPARED = "transaction_prepared"
    PARTICIPANT_FINALIZED = "participant_finalized"
    TRANSACTION_COMPLETED = "transaction_completed"
    TRANSACTION_RECONCILIATION_REQUIRED = "transaction_reconciliation_required"
    TRANSACTION_RECONCILED = "transaction_reconciled"
    PARTICIPANT_ABORTED = "participant_aborted"
    TRANSACTION_ABORT_REQUIRED = "transaction_abort_required"
    TRANSACTION_ABORTED = "transaction_aborted"
    STARTUP_RECONCILIATION_PREPARED = "startup_reconciliation_prepared"
    STARTUP_PARTICIPANT_RECONCILED = "startup_participant_reconciled"
    STARTUP_RECONCILIATION_COMPLETED = "startup_reconciliation_completed"


class TransactionKind(str, Enum):
    EVENT_MUTATION = "event_mutation"


class ParticipantOutcome(str, Enum):
    FINALIZED = "finalized"
    ALREADY_CONSISTENT = "already_consistent"


class ParticipantCapability(str, Enum):
    PREPARE = "prepare"
    ABORT = "abort"
    IDEMPOTENT_FINALIZE = "idempotent_finalize"
    INSPECT_RECONCILE = "inspect_reconcile"


class AbortOutcome(str, Enum):
    ABORTED = "aborted"
    ALREADY_ABSENT = "already_absent"


class ReconciliationReason(str, Enum):
    PARTICIPANT_FINALIZATION_INCOMPLETE = "participant_finalization_incomplete"
    PARTICIPANT_DIVERGED = "participant_diverged"
    PARTICIPANT_UNAVAILABLE = "participant_unavailable"
    UNSUPPORTED_RECONCILIATION = "unsupported_reconciliation"


class StartupParticipantOutcome(str, Enum):
    VERIFIED_CONSISTENT = "verified_consistent"
    ROLLED_FORWARD = "rolled_forward"


class EventFailureCategory(str, Enum):
    HANDLER_FAILURE = "handler_failure"
    ACCEPTED_NOT_STARTED = "accepted_not_started"
    UNCOMMITTED_AFTER_CRASH = "uncommitted_after_crash"
    COMMITTED_BEFORE_CRASH = "committed_before_crash"


class EventRecoveryCategory(str, Enum):
    EXACT_CURRENT = "exact_current"
    TRUE_ROLLBACK = "true_rollback"
    UNCOMMITTED_TAIL = "uncommitted_tail"


class EventJournalAppendStage(str, Enum):
    VALIDATE = "validate"
    WRITE = "write"
    FILE_FSYNC = "file_fsync"
    PARENT_FSYNC = "parent_fsync"
    ROTATION = "rotation"


class _JournalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ParticipantRequirement(_JournalModel):
    participant_id: str = Field(min_length=1, max_length=64)
    operation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: tuple[ParticipantCapability, ...]

    @field_validator("participant_id")
    @classmethod
    def validate_participant_id(cls, value: str) -> str:
        if _PARTICIPANT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("participant identifier is invalid")
        return value

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(
        cls, value: tuple[ParticipantCapability, ...]
    ) -> tuple[ParticipantCapability, ...]:
        if (
            not value
            or value != tuple(sorted(value, key=lambda item: item.value))
            or len(set(value)) != len(value)
            or ParticipantCapability.PREPARE not in value
            or ParticipantCapability.IDEMPOTENT_FINALIZE not in value
        ):
            raise ValueError("participant capabilities are invalid")
        return value


class EventJournalRecord(_JournalModel):
    schema_version: Literal[1, 2, 3]
    record_id: str = Field(min_length=1)
    timestamp: datetime
    lifecycle: EventLifecycle
    event_id: str | None = None
    event_type: AgentEventType | None = None
    source: AgentEventSource | None = None
    processing_sequence: int | None = Field(default=None, ge=0)
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    snapshot_sequence: int | None = Field(default=None, ge=0)
    snapshot_hash: str | None = None
    failure_category: EventFailureCategory | None = None
    previous_record_hash: str | None = None
    record_hash: str
    wal_generation_id: str | None = None
    wal_record_id: str | None = None
    wal_record_hash: str | None = None
    migration_previous_v1_hash: str | None = None
    journal_lineage_id: str | None = None
    recovery_id: str | None = None
    recovery_processing_high_water: int | None = Field(default=None, ge=0)
    recovery_category: EventRecoveryCategory | None = None
    external_reconciliation_required: bool | None = None
    migration_previous_v2_hash: str | None = None
    v3_migration_anchor_hash: str | None = None
    transaction_id: str | None = None
    transaction_kind: TransactionKind | None = None
    required_participants: tuple[ParticipantRequirement, ...] | None = None
    participant_id: str | None = None
    operation_digest: str | None = None
    participant_outcome: ParticipantOutcome | None = None
    abort_outcome: AbortOutcome | None = None
    abort_reason: ReconciliationReason | None = None
    reconciliation_id: str | None = None
    startup_participant_outcome: StartupParticipantOutcome | None = None
    reconciliation_reason: ReconciliationReason | None = None
    unresolved_participants: tuple[str, ...] | None = None

    @field_validator("record_id", "event_id")
    @classmethod
    def require_canonical_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError("identifier must be a UUID") from None
        if str(parsed) != value:
            raise ValueError("identifier must use canonical UUID form")
        return value

    @field_validator("participant_id")
    @classmethod
    def validate_record_participant_id(cls, value: str | None) -> str | None:
        if value is not None and _PARTICIPANT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("participant identifier is invalid")
        return value

    @model_validator(mode="after")
    def validate_record_shape(self) -> EventJournalRecord:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        for value in (
            self.state_hash_before,
            self.state_hash_after,
            self.snapshot_hash,
            self.previous_record_hash,
            self.record_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")

        identity = (self.event_id, self.event_type, self.source)
        has_identity = all(value is not None for value in identity)
        no_identity = all(value is None for value in identity)
        if not has_identity and not no_identity:
            raise ValueError("event identity must be complete")

        for value in (
            self.wal_generation_id,
            self.wal_record_id,
            self.recovery_id,
            self.reconciliation_id,
            self.journal_lineage_id,
        ):
            if value is not None:
                try:
                    parsed = UUID(value)
                except ValueError:
                    raise ValueError("identifier must be a UUID") from None
                if str(parsed) != value:
                    raise ValueError("identifier must use canonical UUID form")
        if self.transaction_id is not None:
            try:
                parsed = UUID(self.transaction_id)
            except ValueError:
                raise ValueError("identifier must be a UUID") from None
            if str(parsed) != self.transaction_id:
                raise ValueError("identifier must use canonical UUID form")
        for value in (
            self.operation_digest,
            self.migration_previous_v2_hash,
            self.v3_migration_anchor_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")
        if self.required_participants is not None:
            ids = tuple(item.participant_id for item in self.required_participants)
            if not ids or ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
                raise ValueError("required participants must be unique and sorted")
        if self.unresolved_participants is not None:
            if (
                not self.unresolved_participants
                or any(
                    _PARTICIPANT_ID_PATTERN.fullmatch(x) is None
                    for x in self.unresolved_participants
                )
                or self.unresolved_participants
                != tuple(sorted(self.unresolved_participants))
                or len(set(self.unresolved_participants))
                != len(self.unresolved_participants)
            ):
                raise ValueError("unresolved participants must be unique and sorted")

        v3_fields = (
            self.migration_previous_v2_hash,
            self.v3_migration_anchor_hash,
            self.transaction_id,
            self.transaction_kind,
            self.required_participants,
            self.participant_id,
            self.operation_digest,
            self.participant_outcome,
            self.abort_outcome,
            self.abort_reason,
            self.reconciliation_reason,
            self.unresolved_participants,
            self.reconciliation_id,
            self.startup_participant_outcome,
        )
        if self.schema_version in {1, 2} and any(
            value is not None for value in v3_fields
        ):
            raise ValueError("v3 fields require schema 3")
        for value in (
            self.wal_record_hash,
            self.migration_previous_v1_hash,
        ):
            if value is not None and _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("hash must be lowercase SHA-256")

        if self.schema_version == 1 and any(
            value is not None
            for value in (
                self.wal_generation_id,
                self.wal_record_id,
                self.wal_record_hash,
                self.migration_previous_v1_hash,
                self.journal_lineage_id,
                self.recovery_id,
                self.recovery_processing_high_water,
                self.recovery_category,
                self.external_reconciliation_required,
            )
        ):
            raise ValueError("v1 has no v2 fields")
        transaction_lifecycles = {
            EventLifecycle.TRANSACTION_PREPARED,
            EventLifecycle.PARTICIPANT_FINALIZED,
            EventLifecycle.TRANSACTION_COMPLETED,
            EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
            EventLifecycle.TRANSACTION_RECONCILED,
            EventLifecycle.PARTICIPANT_ABORTED,
            EventLifecycle.TRANSACTION_ABORT_REQUIRED,
            EventLifecycle.TRANSACTION_ABORTED,
        }
        if self.lifecycle in transaction_lifecycles:
            if (
                self.schema_version != 3
                or not has_identity
                or self.processing_sequence is None
            ):
                raise ValueError("transaction record identity is incomplete")
            allowed_transaction_fields = {
                EventLifecycle.TRANSACTION_PREPARED: {
                    "transaction_id",
                    "transaction_kind",
                    "required_participants",
                },
                EventLifecycle.PARTICIPANT_FINALIZED: {
                    "transaction_id",
                    "participant_id",
                    "operation_digest",
                    "participant_outcome",
                },
                EventLifecycle.TRANSACTION_COMPLETED: {"transaction_id"},
                EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED: {
                    "transaction_id",
                    "reconciliation_reason",
                    "unresolved_participants",
                },
                EventLifecycle.TRANSACTION_RECONCILED: {"transaction_id"},
                EventLifecycle.PARTICIPANT_ABORTED: {
                    "transaction_id",
                    "participant_id",
                    "operation_digest",
                    "abort_outcome",
                },
                EventLifecycle.TRANSACTION_ABORT_REQUIRED: {
                    "transaction_id",
                    "abort_reason",
                    "unresolved_participants",
                },
                EventLifecycle.TRANSACTION_ABORTED: {"transaction_id"},
            }[self.lifecycle]
            self._forbid_irrelevant_fields(
                {"processing_sequence", *allowed_transaction_fields}
            )
            if self.lifecycle is EventLifecycle.TRANSACTION_PREPARED and (
                self.transaction_id is None
                or self.transaction_kind is None
                or not self.required_participants
            ):
                raise ValueError("transaction preparation is incomplete")
            if self.lifecycle is EventLifecycle.PARTICIPANT_FINALIZED and (
                self.transaction_id is None
                or self.participant_id is None
                or self.operation_digest is None
                or self.participant_outcome is None
            ):
                raise ValueError("participant finalization is incomplete")
            if self.lifecycle is EventLifecycle.PARTICIPANT_ABORTED and (
                self.transaction_id is None
                or self.participant_id is None
                or self.operation_digest is None
                or self.abort_outcome is None
            ):
                raise ValueError("participant abort is incomplete")
            if (
                self.lifecycle
                in {
                    EventLifecycle.TRANSACTION_COMPLETED,
                    EventLifecycle.TRANSACTION_RECONCILED,
                }
                and self.transaction_id is None
            ):
                raise ValueError("transaction completion is incomplete")
            if (
                self.lifecycle is EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED
                and (
                    self.transaction_id is None
                    or self.reconciliation_reason is None
                    or not self.unresolved_participants
                )
            ):
                raise ValueError("transaction reconciliation is incomplete")
            if self.lifecycle is EventLifecycle.TRANSACTION_ABORT_REQUIRED and (
                self.transaction_id is None
                or self.abort_reason is None
                or not self.unresolved_participants
            ):
                raise ValueError("transaction abort is incomplete")
            if (
                self.lifecycle is EventLifecycle.TRANSACTION_ABORTED
                and self.transaction_id is None
            ):
                raise ValueError("transaction abort completion is incomplete")
            return self
        startup_lifecycles = {
            EventLifecycle.STARTUP_RECONCILIATION_PREPARED,
            EventLifecycle.STARTUP_PARTICIPANT_RECONCILED,
            EventLifecycle.STARTUP_RECONCILIATION_COMPLETED,
        }
        if self.lifecycle in startup_lifecycles:
            if (
                self.schema_version != 3
                or not no_identity
                or self.processing_sequence is not None
                or self.reconciliation_id is None
                or self.recovery_id is None
                or self.snapshot_sequence is None
                or self.snapshot_hash is None
                or self.recovery_processing_high_water is None
                or self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
                or self.journal_lineage_id is None
            ):
                raise ValueError("startup reconciliation binding is incomplete")
            allowed = {
                EventLifecycle.STARTUP_RECONCILIATION_PREPARED: {
                    "reconciliation_id",
                    "recovery_id",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "recovery_processing_high_water",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "journal_lineage_id",
                    "required_participants",
                },
                EventLifecycle.STARTUP_PARTICIPANT_RECONCILED: {
                    "reconciliation_id",
                    "recovery_id",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "recovery_processing_high_water",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "journal_lineage_id",
                    "participant_id",
                    "operation_digest",
                    "startup_participant_outcome",
                },
                EventLifecycle.STARTUP_RECONCILIATION_COMPLETED: {
                    "reconciliation_id",
                    "recovery_id",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "recovery_processing_high_water",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "journal_lineage_id",
                },
            }[self.lifecycle]
            self._forbid_irrelevant_fields(allowed)
            if self.lifecycle is EventLifecycle.STARTUP_RECONCILIATION_PREPARED:
                if not self.required_participants:
                    raise ValueError("startup preparation participants are missing")
            elif self.lifecycle is EventLifecycle.STARTUP_PARTICIPANT_RECONCILED:
                if (
                    self.participant_id is None
                    or self.operation_digest is None
                    or self.startup_participant_outcome is None
                ):
                    raise ValueError("startup participant evidence is incomplete")
            return self
        if self.lifecycle is EventLifecycle.CHECKPOINT:
            if not no_identity or self.processing_sequence is None:
                raise ValueError("checkpoint identity is invalid")
            self._require_snapshot()
            self._forbid_state_and_failure()
            if self.schema_version == 2 and self.journal_lineage_id is None:
                raise ValueError("v2 checkpoint lineage is missing")
            if self.schema_version >= 2 and self.journal_lineage_id is None:
                raise ValueError("checkpoint lineage is missing")
            if self.schema_version >= 2 and (
                self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
                or self.external_reconciliation_required is None
            ):
                raise ValueError("v2 checkpoint WAL identity is incomplete")
            if self.schema_version == 3 and self.v3_migration_anchor_hash is None:
                raise ValueError("v3 checkpoint migration anchor is missing")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                    "migration_previous_v1_hash",
                    "migration_previous_v2_hash",
                    "v3_migration_anchor_hash",
                    "journal_lineage_id",
                    "external_reconciliation_required",
                }
            )
        elif not has_identity and self.lifecycle not in {
            EventLifecycle.RECOVERY_PREPARED,
            EventLifecycle.RECOVERY_COMPLETED,
        }:
            raise ValueError("event lifecycle requires identity")
        elif self.lifecycle is EventLifecycle.ACCEPTED:
            self._require_only()
            self._forbid_irrelevant_fields(set())
        elif self.lifecycle is EventLifecycle.STARTED:
            if self.processing_sequence is None:
                raise ValueError("started requires processing sequence")
            self._forbid_state_snapshot_failure()
            self._forbid_irrelevant_fields({"processing_sequence"})
        elif self.lifecycle is EventLifecycle.PREPARED:
            if (
                self.processing_sequence is None
                or self.state_hash_before is None
                or self.state_hash_after is None
            ):
                raise ValueError("prepared requires sequence and state hashes")
            if any(
                value is not None
                for value in (
                    self.snapshot_sequence,
                    self.snapshot_hash,
                    self.failure_category,
                )
            ):
                raise ValueError("prepared has forbidden fields")
            if self.schema_version >= 2 and (self.wal_generation_id is None):
                raise ValueError("v2 prepared WAL identity is incomplete")
            if self.schema_version >= 2 and (
                self.wal_record_id is not None or self.wal_record_hash is not None
            ):
                raise ValueError("v2 prepared cannot bind appended WAL record")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "state_hash_before",
                    "state_hash_after",
                    "wal_generation_id",
                }
            )
        elif self.lifecycle is EventLifecycle.COMPLETED:
            if self.processing_sequence is None:
                raise ValueError("completed requires processing sequence")
            self._require_snapshot()
            self._forbid_state_and_failure()
            if self.schema_version >= 2 and (
                self.wal_generation_id is None
                or self.wal_record_id is None
                or self.wal_record_hash is None
            ):
                raise ValueError("v2 completed WAL identity is incomplete")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "wal_record_id",
                    "wal_record_hash",
                }
            )
        elif self.lifecycle is EventLifecycle.FAILED:
            if self.processing_sequence is None:
                raise ValueError("failed requires processing sequence")
            self._require_snapshot()
            if self.failure_category is not EventFailureCategory.HANDLER_FAILURE:
                raise ValueError("failed category is invalid")
            self._forbid_state()
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "failure_category",
                }
            )
        elif self.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
            self._require_snapshot()
            self._forbid_state()
            if self.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED:
                if self.processing_sequence is not None:
                    raise ValueError("accepted-only recovery cannot consume sequence")
            elif self.failure_category not in {
                EventFailureCategory.UNCOMMITTED_AFTER_CRASH,
                EventFailureCategory.COMMITTED_BEFORE_CRASH,
            }:
                raise ValueError("recovery category is invalid")
            elif self.processing_sequence is None:
                raise ValueError("processing recovery requires sequence")
            self._forbid_irrelevant_fields(
                {
                    "processing_sequence",
                    "snapshot_sequence",
                    "snapshot_hash",
                    "failure_category",
                }
            )
        elif self.lifecycle is EventLifecycle.RECOVERY_PREPARED:
            if self.schema_version < 2 or self.recovery_id is None or not no_identity:
                raise ValueError("v2 recovery preparation is invalid")
            self._require_snapshot()
            if self.recovery_category is None or self.wal_generation_id is None:
                raise ValueError("v2 recovery preparation is incomplete")
            if self.recovery_processing_high_water is None:
                raise ValueError("recovery preparation high-water is missing")
            if self.external_reconciliation_required is not None:
                raise ValueError("recovery preparation has forbidden fields")
            self._forbid_state()
            self._forbid_irrelevant_fields(
                {
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "recovery_id",
                    "recovery_processing_high_water",
                    "recovery_category",
                }
            )
        elif self.lifecycle is EventLifecycle.RECOVERY_COMPLETED:
            if self.schema_version < 2 or self.recovery_id is None or not no_identity:
                raise ValueError("v2 recovery completion is invalid")
            self._require_snapshot()
            if self.recovery_category is None or self.wal_generation_id is None:
                raise ValueError("v2 recovery completion is incomplete")
            if self.recovery_processing_high_water is None:
                raise ValueError("recovery completion high-water is missing")
            if self.external_reconciliation_required is None:
                raise ValueError("recovery completion requires reconciliation flag")
            if self.wal_record_id is None or self.wal_record_hash is None:
                raise ValueError("recovery completion WAL identity is incomplete")
            self._forbid_state()
            self._forbid_irrelevant_fields(
                {
                    "snapshot_sequence",
                    "snapshot_hash",
                    "wal_generation_id",
                    "recovery_id",
                    "recovery_processing_high_water",
                    "recovery_category",
                    "external_reconciliation_required",
                    "wal_record_id",
                    "wal_record_hash",
                }
            )
        return self

    def _require_only(self) -> None:
        if any(
            value is not None
            for value in (
                self.processing_sequence,
                self.state_hash_before,
                self.state_hash_after,
                self.snapshot_sequence,
                self.snapshot_hash,
                self.failure_category,
            )
        ):
            raise ValueError("accepted has forbidden fields")

    def _require_snapshot(self) -> None:
        if self.snapshot_sequence is None or self.snapshot_hash is None:
            raise ValueError("snapshot identity is required")

    def _forbid_state(self) -> None:
        if self.state_hash_before is not None or self.state_hash_after is not None:
            raise ValueError("state hashes are forbidden")

    def _forbid_irrelevant_fields(self, allowed: set[str]) -> None:
        allowed = allowed | {"event_id", "event_type", "source"}
        fields = (
            "event_id",
            "event_type",
            "source",
            "processing_sequence",
            "state_hash_before",
            "state_hash_after",
            "snapshot_sequence",
            "snapshot_hash",
            "failure_category",
            "wal_generation_id",
            "wal_record_id",
            "wal_record_hash",
            "migration_previous_v1_hash",
            "journal_lineage_id",
            "recovery_id",
            "recovery_processing_high_water",
            "recovery_category",
            "external_reconciliation_required",
            "migration_previous_v2_hash",
            "v3_migration_anchor_hash",
            "transaction_id",
            "transaction_kind",
            "required_participants",
            "participant_id",
            "operation_digest",
            "participant_outcome",
            "abort_outcome",
            "abort_reason",
            "reconciliation_id",
            "startup_participant_outcome",
            "reconciliation_reason",
            "unresolved_participants",
        )
        if any(
            getattr(self, field) is not None for field in fields if field not in allowed
        ):
            raise ValueError(f"{self.lifecycle.value} has forbidden fields")

    def _forbid_state_and_failure(self) -> None:
        self._forbid_state()
        if self.failure_category is not None:
            raise ValueError("failure category is forbidden")

    def _forbid_state_snapshot_failure(self) -> None:
        self._forbid_state_and_failure()
        if self.snapshot_sequence is not None or self.snapshot_hash is not None:
            raise ValueError("snapshot identity is forbidden")


class EventJournalRecovery(_JournalModel):
    processing_high_water: int = Field(ge=0)
    snapshot_sequence: int = Field(ge=0)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class EventJournalError(Exception):
    """Base class for bounded Journal failures."""


class EventJournalLoadError(EventJournalError):
    """Journal artifacts cannot be loaded safely."""


class EventJournalIntegrityError(EventJournalLoadError):
    """Journal evidence is internally inconsistent."""


class UnsupportedEventJournalVersion(EventJournalLoadError):
    """A record uses an unsupported schema version."""


class EventJournalAppendError(EventJournalError):
    """A lifecycle record did not reach confirmed durable success."""

    def __init__(self, stage: EventJournalAppendStage, *, published: bool) -> None:
        self.stage = stage
        self.published = published
        super().__init__(
            "EventJournal append failed "
            f"at {stage.value}; published={str(published).lower()}"
        )


@dataclass(frozen=True, slots=True)
class _EventState:
    event_id: str
    event_type: AgentEventType
    source: AgentEventSource
    requested_at: datetime
    lifecycle: EventLifecycle
    processing_sequence: int | None
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    wal_generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventJournalTransaction:
    transaction_id: str
    event_id: str
    event_type: AgentEventType
    source: AgentEventSource
    processing_sequence: int
    kind: TransactionKind
    required_participants: tuple[ParticipantRequirement, ...]
    participant_outcomes: tuple[tuple[str, ParticipantOutcome], ...]
    abort_outcomes: tuple[tuple[str, AbortOutcome], ...] = ()
    reconciliation_reason: ReconciliationReason | None = None
    abort_reason: ReconciliationReason | None = None
    unresolved_participants: tuple[str, ...] = ()
    terminal_lifecycle: EventLifecycle | None = None

    @property
    def known_outcomes(self) -> tuple[tuple[str, ParticipantOutcome], ...]:
        return self.participant_outcomes


@dataclass(frozen=True, slots=True)
class EventJournalStartupReconciliation:
    reconciliation_id: str
    recovery_id: str
    snapshot_sequence: int
    snapshot_hash: str
    recovery_processing_high_water: int
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str
    journal_lineage_id: str
    required_participants: tuple[ParticipantRequirement, ...]
    participant_outcomes: tuple[tuple[str, str, StartupParticipantOutcome], ...]
    completed: bool

    @property
    def known_outcomes(
        self,
    ) -> tuple[tuple[str, str, StartupParticipantOutcome], ...]:
        return self.participant_outcomes


@dataclass(frozen=True, slots=True)
class _VerifiedJournal:
    records: tuple[EventJournalRecord, ...]
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    open_events: tuple[_EventState, ...]
    journal_lineage_id: str | None
    external_reconciliation_required: bool
    wal_snapshot_sequence: int | None
    wal_snapshot_hash: str | None
    open_recoveries: tuple[EventJournalRecord, ...]
    wal_generation_id: str | None
    wal_record_id: str | None
    wal_record_hash: str | None
    transactions: tuple[EventJournalTransaction, ...] = ()
    startup_reconciliations: tuple[EventJournalStartupReconciliation, ...] = ()


@dataclass(frozen=True, slots=True)
class EventJournalOpenEvent:
    event_id: str
    lifecycle: EventLifecycle
    processing_sequence: int | None
    classification: str


@dataclass(frozen=True, slots=True)
class EventJournalOpenRecovery:
    recovery_id: str
    category: EventRecoveryCategory
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str


@dataclass(frozen=True, slots=True)
class EventJournalInspection:
    """Read-only, bounded evidence for a parent authority to evaluate."""

    records: tuple[EventJournalRecord, ...]
    tail_record_id: str | None
    tail_record_hash: str | None
    schema_version: int | None
    processing_high_water: int
    snapshot_sequence: int
    snapshot_hash: str
    open_events: tuple[EventJournalOpenEvent, ...]
    classification_plan: tuple[EventJournalOpenEvent, ...]
    open_recoveries: tuple[EventJournalOpenRecovery, ...]
    sequence_evidence: tuple[int, ...]
    journal_lineage_id: str | None
    external_reconciliation_required: bool
    wal_snapshot_sequence: int | None
    wal_snapshot_hash: str | None
    wal_generation_id: str | None
    wal_record_id: str | None
    wal_record_hash: str | None
    open_transactions: tuple[EventJournalTransaction, ...] = ()
    reconciliation_required_transactions: tuple[EventJournalTransaction, ...] = ()
    completed_transactions: tuple[EventJournalTransaction, ...] = ()
    reconciled_transactions: tuple[EventJournalTransaction, ...] = ()
    abort_required_transactions: tuple[EventJournalTransaction, ...] = ()
    aborted_transactions: tuple[EventJournalTransaction, ...] = ()
    open_startup_reconciliations: tuple[EventJournalStartupReconciliation, ...] = ()
    completed_startup_reconciliations: tuple[
        EventJournalStartupReconciliation, ...
    ] = ()


class EventJournalLease:
    """Exclusive process lease protecting Journal and Snapshot startup authority."""

    def __init__(self, journal_path: str | Path) -> None:
        self.path = Path(journal_path)
        self._descriptor: int | None = None
        lease_failure: EventJournalLoadError | None = None
        descriptor: int | None = None
        parent_descriptor: int | None = None
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_descriptor = os.open(
                self.path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_status = os.fstat(parent_descriptor)
            if (
                not stat.S_ISDIR(parent_status.st_mode)
                or parent_status.st_uid != os.geteuid()
            ):
                raise OSError("journal directory is not private")
            if parent_status.st_mode & 0o077:
                os.fchmod(parent_descriptor, 0o700)
                if os.fstat(parent_descriptor).st_mode & 0o077:
                    raise OSError("journal directory hardening failed")
            descriptor = os.open(
                f".{self.path.name}.lock",
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            lock_status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock_status.st_mode)
                or lock_status.st_uid != os.geteuid()
            ):
                raise OSError("journal lock is not regular")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._descriptor = descriptor
            descriptor = None
            os.fsync(parent_descriptor)
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            lease_failure = EventJournalLoadError(
                "EventJournal exclusive authority is unavailable"
            )
        finally:
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
        if lease_failure is not None:
            raise lease_failure

    @property
    def held(self) -> bool:
        return self._descriptor is not None

    def close(self) -> None:
        if self._descriptor is not None:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
                self._descriptor = None

    def _transfer(self) -> EventJournalLease:
        if self._descriptor is None:
            raise ValueError("EventJournal lease is not held")
        adopted = object.__new__(EventJournalLease)
        adopted.path = self.path
        adopted._descriptor = self._descriptor
        self._descriptor = None
        return adopted

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _fsync_parent(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class EventJournal:
    """Append and verify durable lifecycle records without event payloads."""

    def __init__(
        self,
        path: str | Path,
        max_bytes: int,
        retained_files: int,
        *,
        clock: Callable[[], datetime] | None = None,
        append_stage_hook: (
            Callable[[EventLifecycle, EventJournalAppendStage], None] | None
        ) = None,
        lease: EventJournalLease | None = None,
    ) -> None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(retained_files, bool)
            or not isinstance(retained_files, int)
            or retained_files < 2
        ):
            raise ValueError("retained_files must be at least two")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.retained_files = retained_files
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._append_stage_hook = append_stage_hook
        self._lock = RLock()
        if lease is not None and (lease.path != self.path or not lease.held):
            raise ValueError("EventJournal lease does not match configured path")
        self._lease = (
            lease._transfer() if lease is not None else EventJournalLease(self.path)
        )
        with self._lock:
            try:
                records = self._read_records_unlocked()
                if records:
                    self._verify_records(records)
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Release this process's exclusive journal authority."""

        with self._lock:
            self._lease.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def records(self) -> tuple[EventJournalRecord, ...]:
        with self._lock:
            self._require_authority()
            return self._read_records_unlocked()

    @property
    def has_exclusive_authority(self) -> bool:
        return self._lease.held

    @property
    def has_history(self) -> bool:
        with self._lock:
            self._require_authority()
            return bool(self._read_records_unlocked())

    def inspect(
        self,
        snapshot_sequence: int | None = None,
        snapshot_hash: str | None = None,
    ) -> EventJournalInspection:
        """Return verified evidence without appending, rotating, or migrating."""
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                return EventJournalInspection(
                    records=(),
                    tail_record_id=None,
                    tail_record_hash=None,
                    schema_version=None,
                    processing_high_water=0,
                    snapshot_sequence=0,
                    snapshot_hash="0" * 64,
                    open_events=(),
                    classification_plan=(),
                    open_recoveries=(),
                    sequence_evidence=(),
                    journal_lineage_id=None,
                    external_reconciliation_required=False,
                    wal_snapshot_sequence=None,
                    wal_snapshot_hash=None,
                    wal_generation_id=None,
                    wal_record_id=None,
                    wal_record_hash=None,
                    open_transactions=(),
                    reconciliation_required_transactions=(),
                    completed_transactions=(),
                    reconciled_transactions=(),
                    abort_required_transactions=(),
                    aborted_transactions=(),
                    open_startup_reconciliations=(),
                    completed_startup_reconciliations=(),
                )
            verified = self._verify_records(records)
            classification_sequence = (
                verified.snapshot_sequence
                if snapshot_sequence is None
                else snapshot_sequence
            )
            classification_hash = (
                verified.snapshot_hash if snapshot_hash is None else snapshot_hash
            )
            self._validate_snapshot_identity(
                classification_sequence, classification_hash
            )
            open_events = self._plan_reconciliation(
                verified, classification_sequence, classification_hash
            )
            return EventJournalInspection(
                records=records,
                tail_record_id=records[-1].record_id,
                tail_record_hash=records[-1].record_hash,
                schema_version=records[-1].schema_version,
                processing_high_water=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                open_events=open_events,
                classification_plan=open_events,
                open_recoveries=tuple(
                    EventJournalOpenRecovery(
                        recovery_id=record.recovery_id or "",
                        category=record.recovery_category
                        or EventRecoveryCategory.EXACT_CURRENT,
                        processing_high_water=record.recovery_processing_high_water
                        or 0,
                        snapshot_sequence=record.snapshot_sequence or 0,
                        snapshot_hash=record.snapshot_hash or "0" * 64,
                        wal_generation_id=record.wal_generation_id or "",
                    )
                    for record in verified.open_recoveries
                ),
                sequence_evidence=tuple(
                    record.processing_sequence
                    for record in records
                    if record.processing_sequence is not None
                ),
                journal_lineage_id=verified.journal_lineage_id,
                external_reconciliation_required=(
                    verified.external_reconciliation_required
                ),
                wal_snapshot_sequence=verified.wal_snapshot_sequence,
                wal_snapshot_hash=verified.wal_snapshot_hash,
                wal_generation_id=verified.wal_generation_id,
                wal_record_id=verified.wal_record_id,
                wal_record_hash=verified.wal_record_hash,
                open_transactions=tuple(
                    t for t in verified.transactions if t.terminal_lifecycle is None
                ),
                reconciliation_required_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is None
                    and t.reconciliation_reason is not None
                ),
                completed_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is EventLifecycle.TRANSACTION_COMPLETED
                ),
                reconciled_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is EventLifecycle.TRANSACTION_RECONCILED
                ),
                abort_required_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.abort_reason is not None and t.terminal_lifecycle is None
                ),
                aborted_transactions=tuple(
                    t
                    for t in verified.transactions
                    if t.terminal_lifecycle is EventLifecycle.TRANSACTION_ABORTED
                ),
                open_startup_reconciliations=tuple(
                    item
                    for item in verified.startup_reconciliations
                    if not item.completed
                ),
                completed_startup_reconciliations=tuple(
                    item for item in verified.startup_reconciliations if item.completed
                ),
            )

    def apply_planned_reconciliation(
        self, snapshot_sequence: int, snapshot_hash: str
    ) -> EventJournalRecovery:
        """Apply the legacy reconciliation plan; migration remains explicit."""
        return self.verify_and_reconcile(snapshot_sequence, snapshot_hash)

    @staticmethod
    def _plan_reconciliation(
        verified: _VerifiedJournal,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> tuple[EventJournalOpenEvent, ...]:
        """Validate a candidate identity and return a read-only crash plan."""

        processing = tuple(
            event
            for event in verified.open_events
            if event.lifecycle in {EventLifecycle.STARTED, EventLifecycle.PREPARED}
        )
        if len(processing) > 1:
            raise EventJournalIntegrityError(
                "EventJournal has multiple interrupted handlers"
            )
        canonical_matches = (
            snapshot_sequence == verified.snapshot_sequence
            and snapshot_hash == verified.snapshot_hash
        )
        committed_event_id: str | None = None
        if processing:
            interrupted = processing[0]
            if interrupted.lifecycle is EventLifecycle.PREPARED:
                if (
                    snapshot_sequence == interrupted.processing_sequence
                    and snapshot_hash == interrupted.state_hash_after
                ):
                    committed_event_id = interrupted.event_id
                elif not (
                    canonical_matches and snapshot_hash == interrupted.state_hash_before
                ):
                    raise EventJournalIntegrityError(
                        "Prepared transition does not match canonical snapshot"
                    )
            elif not canonical_matches:
                raise EventJournalIntegrityError(
                    "Started event does not match canonical snapshot"
                )
        elif not canonical_matches:
            raise EventJournalIntegrityError(
                "Journal and canonical snapshot are inconsistent"
            )
        return tuple(
            EventJournalOpenEvent(
                event_id=item.event_id,
                lifecycle=item.lifecycle,
                processing_sequence=item.processing_sequence,
                classification=(
                    "accepted_only"
                    if item.lifecycle is EventLifecycle.ACCEPTED
                    else "prepared_committed_before_crash"
                    if item.event_id == committed_event_id
                    else "started_or_prepared_uncommitted"
                ),
            )
            for item in verified.open_events
        )

    def append_v2_migration_checkpoint(
        self,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
    ) -> None:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError("migration requires v1 history")
            verified = self._verify_records(records)
            if any(record.schema_version == 2 for record in records):
                raise EventJournalIntegrityError("EventJournal is already v2")
            if (
                verified.snapshot_sequence != snapshot_sequence
                or verified.snapshot_hash != snapshot_hash
            ):
                raise EventJournalIntegrityError("migration snapshot is inconsistent")
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                schema_version=2,
                migration_previous_v1_hash=records[-1].record_hash,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=str(uuid4()),
                external_reconciliation_required=False,
            )
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    # Short name retained for callers that model this as a state transition.
    migrate_v2 = append_v2_migration_checkpoint

    def append_v3_migration_checkpoint(self) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) != 2:
                raise EventJournalIntegrityError("v3 migration requires v2 history")
            verified = self._verify_records(records)
            if (
                verified.open_events
                or verified.open_recoveries
                or verified.transactions
                or any(not item.completed for item in verified.startup_reconciliations)
            ):
                raise EventJournalIntegrityError(
                    "v3 migration requires a closed v2 journal"
                )
            prior = records[-1]
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=prior.record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                schema_version=3,
                migration_previous_v2_hash=prior.record_hash,
                v3_migration_anchor_hash=prior.record_hash,
                wal_generation_id=verified.wal_generation_id,
                wal_record_id=verified.wal_record_id,
                wal_record_hash=verified.wal_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
                external_reconciliation_required=verified.external_reconciliation_required,
            )
            self._verify_records((*records, checkpoint))
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    migrate_v3 = append_v3_migration_checkpoint

    def _append_transaction(
        self, lifecycle: EventLifecycle, event: AgentEvent, **fields: object
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) != 3:
                raise EventJournalIntegrityError(
                    "transaction evidence requires schema 3"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    event=event,
                    schema_version=3,
                    processing_sequence=event.processing_sequence,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def append_transaction_prepared(
        self,
        event: AgentEvent,
        transaction_id: str,
        kind: TransactionKind,
        required_participants: tuple[ParticipantRequirement, ...],
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_PREPARED,
            event,
            transaction_id=transaction_id,
            transaction_kind=kind,
            required_participants=required_participants,
        )

    def append_participant_finalized(
        self,
        event: AgentEvent,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
        outcome: ParticipantOutcome,
    ) -> None:
        self._append_transaction(
            EventLifecycle.PARTICIPANT_FINALIZED,
            event,
            transaction_id=transaction_id,
            participant_id=participant_id,
            operation_digest=operation_digest,
            participant_outcome=outcome,
        )

    def append_transaction_completed(
        self, event: AgentEvent, transaction_id: str
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_COMPLETED, event, transaction_id=transaction_id
        )

    def append_transaction_reconciliation_required(
        self,
        event: AgentEvent,
        transaction_id: str,
        reason: ReconciliationReason,
        unresolved_participants: tuple[str, ...],
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
            event,
            transaction_id=transaction_id,
            reconciliation_reason=reason,
            unresolved_participants=unresolved_participants,
        )

    def append_transaction_reconciled(
        self, event: AgentEvent, transaction_id: str
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_RECONCILED, event, transaction_id=transaction_id
        )

    def append_participant_aborted(
        self,
        event: AgentEvent,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
        outcome: AbortOutcome,
    ) -> None:
        self._append_transaction(
            EventLifecycle.PARTICIPANT_ABORTED,
            event,
            transaction_id=transaction_id,
            participant_id=participant_id,
            operation_digest=operation_digest,
            abort_outcome=outcome,
        )

    def append_transaction_abort_required(
        self,
        event: AgentEvent,
        transaction_id: str,
        reason: ReconciliationReason,
        unresolved_participants: tuple[str, ...],
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_ABORT_REQUIRED,
            event,
            transaction_id=transaction_id,
            abort_reason=reason,
            unresolved_participants=unresolved_participants,
        )

    def append_transaction_aborted(
        self, event: AgentEvent, transaction_id: str
    ) -> None:
        self._append_transaction(
            EventLifecycle.TRANSACTION_ABORTED, event, transaction_id=transaction_id
        )

    def _append_startup_reconciliation(
        self, lifecycle: EventLifecycle, **fields: object
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) != 3:
                raise EventJournalIntegrityError(
                    "startup reconciliation evidence requires schema 3"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                if (
                    lifecycle is EventLifecycle.STARTUP_RECONCILIATION_PREPARED
                    and not any(
                        record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
                        and record.recovery_id == fields.get("recovery_id")
                        for record in records
                    )
                ):
                    raise EventJournalIntegrityError(
                        "startup reconciliation recovery evidence is unknown"
                    )
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    schema_version=3,
                    event=None,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def append_startup_reconciliation_prepared(
        self,
        reconciliation_id: str,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        recovery_processing_high_water: int,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        journal_lineage_id: str,
        required_participants: tuple[ParticipantRequirement, ...],
    ) -> None:
        self._append_startup_reconciliation(
            EventLifecycle.STARTUP_RECONCILIATION_PREPARED,
            reconciliation_id=reconciliation_id,
            recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=recovery_processing_high_water,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            journal_lineage_id=journal_lineage_id,
            required_participants=required_participants,
        )

    def append_startup_participant_reconciled(
        self,
        reconciliation_id: str,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        recovery_processing_high_water: int,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        journal_lineage_id: str,
        participant_id: str,
        operation_digest: str,
        outcome: StartupParticipantOutcome,
    ) -> None:
        self._append_startup_reconciliation(
            EventLifecycle.STARTUP_PARTICIPANT_RECONCILED,
            reconciliation_id=reconciliation_id,
            recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=recovery_processing_high_water,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            journal_lineage_id=journal_lineage_id,
            participant_id=participant_id,
            operation_digest=operation_digest,
            startup_participant_outcome=outcome,
        )

    def append_startup_reconciliation_completed(
        self,
        reconciliation_id: str,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        recovery_processing_high_water: int,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        journal_lineage_id: str,
    ) -> None:
        self._append_startup_reconciliation(
            EventLifecycle.STARTUP_RECONCILIATION_COMPLETED,
            reconciliation_id=reconciliation_id,
            recovery_id=recovery_id,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            recovery_processing_high_water=recovery_processing_high_water,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
            journal_lineage_id=journal_lineage_id,
        )

    def append_v2_bootstrap_checkpoint(
        self,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
        processing_high_water: int | None = None,
    ) -> None:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        if processing_high_water is None:
            processing_high_water = snapshot_sequence
        if processing_high_water < snapshot_sequence:
            raise ValueError("processing high-water precedes snapshot")
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if records:
                raise EventJournalIntegrityError(
                    "v2 bootstrap requires an empty journal"
                )
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=None,
                processing_sequence=processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                schema_version=2,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=str(uuid4()),
                external_reconciliation_required=False,
            )
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    def append_v2_current_checkpoint(
        self,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        wal_record_id: str,
        wal_record_hash: str,
    ) -> None:
        """Bind reconciled current state to exact WAL evidence."""

        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or records[-1].schema_version < 2:
                raise EventJournalIntegrityError(
                    "current checkpoint requires v2 history"
                )
            verified = self._verify_records(records)
            if (
                verified.open_events
                or verified.snapshot_sequence != snapshot_sequence
                or verified.snapshot_hash != snapshot_hash
                or verified.journal_lineage_id is None
            ):
                raise EventJournalIntegrityError("current checkpoint is inconsistent")
            schema_version = records[-1].schema_version
            v3_migration_anchor_hash = (
                next(
                    (
                        record.v3_migration_anchor_hash
                        for record in reversed(records)
                        if record.schema_version == 3
                        and record.lifecycle is EventLifecycle.CHECKPOINT
                        and record.v3_migration_anchor_hash is not None
                    ),
                    None,
                )
                if schema_version == 3
                else None
            )
            if schema_version == 3 and v3_migration_anchor_hash is None:
                raise EventJournalIntegrityError("v3 Journal lacks migration authority")
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
                schema_version=schema_version,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=verified.journal_lineage_id,
                v3_migration_anchor_hash=v3_migration_anchor_hash,
                external_reconciliation_required=(
                    verified.external_reconciliation_required
                ),
            )
            self._verify_records((*records, checkpoint))
            self._append_record_unlocked(checkpoint)
            self._maybe_rotate_unlocked()

    def append_accepted(self, event: AgentEvent) -> None:
        self._append_event(EventLifecycle.ACCEPTED, event)

    def append_started(self, event: AgentEvent) -> None:
        self._append_event(
            EventLifecycle.STARTED,
            event,
            processing_sequence=event.processing_sequence,
        )

    def append_prepared(
        self,
        event: AgentEvent,
        state_hash_before: str,
        state_hash_after: str,
        wal_generation_id: str | None = None,
    ) -> None:
        self._append_event(
            EventLifecycle.PREPARED,
            event,
            processing_sequence=event.processing_sequence,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            wal_generation_id=wal_generation_id,
        )

    def append_completed(
        self,
        event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str | None = None,
        wal_record_id: str | None = None,
        wal_record_hash: str | None = None,
    ) -> None:
        self._append_event(
            EventLifecycle.COMPLETED,
            event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            wal_generation_id=wal_generation_id,
            wal_record_id=wal_record_id,
            wal_record_hash=wal_record_hash,
        )

    def append_recovery_prepared(
        self,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        category: EventRecoveryCategory,
        processing_high_water: int | None = None,
    ) -> None:
        self._append_v2_recovery(
            EventLifecycle.RECOVERY_PREPARED,
            recovery_id,
            snapshot_sequence,
            snapshot_hash,
            wal_generation_id,
            category,
            None,
            processing_high_water,
        )

    def append_recovery_completed(
        self,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        category: EventRecoveryCategory,
        external_reconciliation_required: bool,
        processing_high_water: int | None = None,
        *,
        wal_record_id: str,
        wal_record_hash: str,
    ) -> None:
        self._append_v2_recovery(
            EventLifecycle.RECOVERY_COMPLETED,
            recovery_id,
            snapshot_sequence,
            snapshot_hash,
            wal_generation_id,
            category,
            external_reconciliation_required,
            processing_high_water,
            wal_record_id,
            wal_record_hash,
        )

    def append_failed(
        self,
        event: AgentEvent,
        snapshot_sequence: int,
        snapshot_hash: str,
    ) -> None:
        self._append_event(
            EventLifecycle.FAILED,
            event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            failure_category=EventFailureCategory.HANDLER_FAILURE,
        )

    def verify_and_reconcile(
        self, snapshot_sequence: int, snapshot_hash: str
    ) -> EventJournalRecovery:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                checkpoint = self._make_record(
                    EventLifecycle.CHECKPOINT,
                    previous_hash=None,
                    processing_sequence=snapshot_sequence,
                    snapshot_sequence=snapshot_sequence,
                    snapshot_hash=snapshot_hash,
                    schema_version=1,
                )
                self._append_record_unlocked(checkpoint)
                return EventJournalRecovery(
                    processing_high_water=snapshot_sequence,
                    snapshot_sequence=snapshot_sequence,
                    snapshot_hash=snapshot_hash,
                )

            verified = self._verify_records(records)
            self._plan_reconciliation(verified, snapshot_sequence, snapshot_hash)
            processing = tuple(
                event
                for event in verified.open_events
                if event.lifecycle in {EventLifecycle.STARTED, EventLifecycle.PREPARED}
            )
            if len(processing) > 1:
                raise EventJournalIntegrityError(
                    "EventJournal has multiple interrupted handlers"
                )

            canonical_matches = (
                snapshot_sequence == verified.snapshot_sequence
                and snapshot_hash == verified.snapshot_hash
            )
            if processing:
                interrupted = processing[0]
                category = EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                if interrupted.lifecycle is EventLifecycle.PREPARED:
                    if (
                        snapshot_sequence == interrupted.processing_sequence
                        and snapshot_hash == interrupted.state_hash_after
                    ):
                        category = EventFailureCategory.COMMITTED_BEFORE_CRASH
                    elif not (
                        canonical_matches
                        and snapshot_hash == interrupted.state_hash_before
                    ):
                        raise EventJournalIntegrityError(
                            "Prepared transition does not match canonical snapshot"
                        )
                elif not canonical_matches:
                    raise EventJournalIntegrityError(
                        "Started event does not match canonical snapshot"
                    )
                self._append_recovery_unlocked(
                    interrupted,
                    snapshot_sequence,
                    snapshot_hash,
                    category,
                )
                records = self._read_records_unlocked()
                verified = self._verify_records(records)
            elif not canonical_matches:
                raise EventJournalIntegrityError(
                    "Journal and canonical snapshot are inconsistent"
                )

            for accepted in tuple(
                event
                for event in verified.open_events
                if event.lifecycle is EventLifecycle.ACCEPTED
            ):
                self._append_recovery_unlocked(
                    accepted,
                    snapshot_sequence,
                    snapshot_hash,
                    EventFailureCategory.ACCEPTED_NOT_STARTED,
                )
                verified = self._verify_records(self._read_records_unlocked())

            return EventJournalRecovery(
                processing_high_water=verified.processing_high_water,
                snapshot_sequence=snapshot_sequence,
                snapshot_hash=snapshot_hash,
            )

    def _append_event(
        self,
        lifecycle: EventLifecycle,
        event: AgentEvent,
        **fields: Any,
    ) -> None:
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records:
                raise EventJournalIntegrityError(
                    "EventJournal requires startup reconciliation"
                )
            validation_failure: EventJournalAppendError | None = None
            try:
                fields.setdefault("schema_version", self._active_schema(records))
                record = self._make_record(
                    lifecycle,
                    previous_hash=records[-1].record_hash,
                    event=event,
                    **fields,
                )
                self._verify_records((*records, record))
            except (ValidationError, ValueError, EventJournalIntegrityError):
                validation_failure = EventJournalAppendError(
                    EventJournalAppendStage.VALIDATE, published=False
                )
            if validation_failure is not None:
                raise validation_failure
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def _append_v2_recovery(
        self,
        lifecycle: EventLifecycle,
        recovery_id: str,
        snapshot_sequence: int,
        snapshot_hash: str,
        wal_generation_id: str,
        category: EventRecoveryCategory,
        external: bool | None,
        processing_high_water: int | None,
        wal_record_id: str | None = None,
        wal_record_hash: str | None = None,
    ) -> None:
        self._validate_snapshot_identity(snapshot_sequence, snapshot_hash)
        with self._lock:
            self._require_authority()
            records = self._read_records_unlocked()
            if not records or self._active_schema(records) < 2:
                raise EventJournalIntegrityError("v2 recovery requires migration")
            verified = self._verify_records(records)
            if processing_high_water is None:
                processing_high_water = verified.processing_high_water
            if processing_high_water != verified.processing_high_water:
                raise EventJournalIntegrityError("recovery high-water changed")
            current = (verified.snapshot_sequence, verified.snapshot_hash)
            target = (snapshot_sequence, snapshot_hash)
            if category in {
                EventRecoveryCategory.EXACT_CURRENT,
                EventRecoveryCategory.UNCOMMITTED_TAIL,
            }:
                if target != current or (
                    external is not None
                    and external is not verified.external_reconciliation_required
                ):
                    raise EventJournalIntegrityError(
                        "current recovery anchor is invalid"
                    )
            elif category is EventRecoveryCategory.TRUE_ROLLBACK:
                if snapshot_sequence >= verified.snapshot_sequence or (
                    external is not None and external is not True
                ):
                    raise EventJournalIntegrityError(
                        "rollback recovery anchor is invalid"
                    )
            if lifecycle is EventLifecycle.RECOVERY_COMPLETED and (
                wal_record_id is None or wal_record_hash is None
            ):
                raise EventJournalIntegrityError("recovery WAL identity is incomplete")
            kwargs: dict[str, Any] = {
                "schema_version": self._active_schema(records),
                "recovery_id": recovery_id,
                "snapshot_sequence": snapshot_sequence,
                "snapshot_hash": snapshot_hash,
                "wal_generation_id": wal_generation_id,
                "recovery_category": category,
                "recovery_processing_high_water": processing_high_water,
            }
            if external is not None:
                kwargs["external_reconciliation_required"] = external
            if wal_record_id is not None:
                kwargs["wal_record_id"] = wal_record_id
            if wal_record_hash is not None:
                kwargs["wal_record_hash"] = wal_record_hash
            record = self._make_record(
                lifecycle, previous_hash=records[-1].record_hash, **kwargs
            )
            self._verify_records((*records, record))
            self._append_record_unlocked(record)
            self._maybe_rotate_unlocked()

    def _append_recovery_unlocked(
        self,
        event: _EventState,
        snapshot_sequence: int,
        snapshot_hash: str,
        category: EventFailureCategory,
    ) -> None:
        records = self._read_records_unlocked()
        journal_event = AgentEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            source=event.source,
            requested_at=event.requested_at,
            processing_sequence=event.processing_sequence,
        )
        record = self._make_record(
            EventLifecycle.RECOVERY_CLASSIFIED,
            previous_hash=records[-1].record_hash,
            event=journal_event,
            processing_sequence=event.processing_sequence,
            snapshot_sequence=snapshot_sequence,
            snapshot_hash=snapshot_hash,
            failure_category=category,
            schema_version=self._active_schema(records),
        )
        self._verify_records((*records, record))
        self._append_record_unlocked(record)
        self._maybe_rotate_unlocked()

    def _make_record(
        self,
        lifecycle: EventLifecycle,
        *,
        previous_hash: str | None,
        event: AgentEvent | None = None,
        schema_version: int | None = None,
        **fields: object,
    ) -> EventJournalRecord:
        validation_failure: EventJournalAppendError | None = None
        try:
            timestamp = self._clock()
            unsigned = EventJournalRecord.model_validate(
                {
                    "schema_version": schema_version or 1,
                    "record_id": str(uuid4()),
                    "timestamp": timestamp,
                    "lifecycle": lifecycle,
                    "event_id": event.event_id if event is not None else None,
                    "event_type": event.event_type if event is not None else None,
                    "source": event.source if event is not None else None,
                    "previous_record_hash": previous_hash,
                    "record_hash": "0" * 64,
                    **fields,
                }
            )
            record = unsigned.model_copy(
                update={"record_hash": self._record_hash(unsigned)}
            )
        except Exception:
            validation_failure = EventJournalAppendError(
                EventJournalAppendStage.VALIDATE, published=False
            )
        if validation_failure is not None:
            raise validation_failure
        return record

    @staticmethod
    def _record_hash(record: EventJournalRecord) -> str:
        excluded = (
            {
                "wal_generation_id",
                "wal_record_id",
                "wal_record_hash",
                "migration_previous_v1_hash",
                "journal_lineage_id",
                "recovery_id",
                "recovery_category",
                "recovery_processing_high_water",
                "external_reconciliation_required",
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
            }
            if record.schema_version == 1
            else {
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
            }
            if record.schema_version in {1, 2}
            else set()
        )
        canonical = json.dumps(
            record.model_dump(mode="json", exclude={"record_hash", *excluded}),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        domain = (
            _HASH_DOMAIN_V1
            if record.schema_version == 1
            else _HASH_DOMAIN_V2
            if record.schema_version == 2
            else _HASH_DOMAIN_V3
        )
        return hashlib.sha256(domain + canonical).hexdigest()

    @staticmethod
    def _record_bytes(record: EventJournalRecord) -> bytes:
        excluded = (
            {
                "wal_generation_id",
                "wal_record_id",
                "wal_record_hash",
                "migration_previous_v1_hash",
                "journal_lineage_id",
                "recovery_id",
                "recovery_category",
                "recovery_processing_high_water",
                "external_reconciliation_required",
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
            }
            if record.schema_version == 1
            else {
                "migration_previous_v2_hash",
                "v3_migration_anchor_hash",
                "transaction_id",
                "transaction_kind",
                "required_participants",
                "participant_id",
                "operation_digest",
                "participant_outcome",
                "abort_outcome",
                "abort_reason",
                "reconciliation_reason",
                "unresolved_participants",
                "reconciliation_id",
                "startup_participant_outcome",
            }
            if record.schema_version in {1, 2}
            else set()
        )
        return (
            json.dumps(
                record.model_dump(mode="json", exclude=excluded),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def _append_record_unlocked(self, record: EventJournalRecord) -> None:
        stage = EventJournalAppendStage.WRITE
        published = False
        append_failure: EventJournalAppendError | None = None
        descriptor: int | None = None
        try:
            self._run_hook(record.lifecycle, stage)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("journal is not a regular file")
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab") as journal_file:
                descriptor = None
                payload = self._record_bytes(record)
                if journal_file.write(payload) != len(payload):
                    raise OSError("incomplete journal write")
                journal_file.flush()
                stage = EventJournalAppendStage.FILE_FSYNC
                self._run_hook(record.lifecycle, stage)
                os.fsync(journal_file.fileno())
                published = True
            stage = EventJournalAppendStage.PARENT_FSYNC
            self._run_hook(record.lifecycle, stage)
            self._fsync_parent()
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            append_failure = EventJournalAppendError(stage, published=published)
        if append_failure is not None:
            raise append_failure

    def _maybe_rotate_unlocked(self) -> None:
        failure: EventJournalAppendError | None
        try:
            size = self.path.stat().st_size
        except OSError:
            failure = EventJournalAppendError(
                EventJournalAppendStage.ROTATION, published=True
            )
        else:
            failure = None
        if failure is not None:
            raise failure
        if size <= self.max_bytes:
            return
        verified = self._verify_records(self._read_records_unlocked())
        if (
            verified.open_events
            or verified.open_recoveries
            or any(t.terminal_lifecycle is None for t in verified.transactions)
            or any(not item.completed for item in verified.startup_reconciliations)
        ):
            return
        if verified.records[-1].schema_version >= 2 and not any(
            record.lifecycle
            in {
                EventLifecycle.CHECKPOINT,
                EventLifecycle.COMPLETED,
                EventLifecycle.RECOVERY_COMPLETED,
            }
            and record.schema_version >= 2
            and record.snapshot_sequence == verified.snapshot_sequence
            and record.snapshot_hash == verified.snapshot_hash
            and record.wal_generation_id is not None
            and record.wal_record_id is not None
            and record.wal_record_hash is not None
            for record in reversed(verified.records)
        ):
            return
        self._rotate_unlocked(verified)

    def _rotate_unlocked(self, verified: _VerifiedJournal) -> None:
        failure: EventJournalAppendError | None = None
        try:
            rotated = self._rotated_paths_unlocked()
            next_index = rotated[-1][0] + 1 if rotated else 0
            archived = self._segment_path(next_index)
            wal_generation_id: str | None = None
            wal_record_id: str | None = None
            wal_record_hash: str | None = None
            journal_lineage_id: str | None = None
            v3_migration_anchor_hash: str | None = None
            if verified.records[-1].schema_version >= 2:
                anchor = next(
                    (
                        record
                        for record in reversed(verified.records)
                        if record.lifecycle
                        in {
                            EventLifecycle.CHECKPOINT,
                            EventLifecycle.COMPLETED,
                            EventLifecycle.RECOVERY_COMPLETED,
                        }
                        and record.schema_version >= 2
                        and record.wal_generation_id is not None
                        and record.wal_record_id is not None
                        and record.wal_record_hash is not None
                    ),
                    None,
                )
                if anchor is None:
                    raise EventJournalIntegrityError("v2 Journal lacks WAL anchor")
                if (
                    anchor.snapshot_sequence != verified.snapshot_sequence
                    or anchor.snapshot_hash != verified.snapshot_hash
                ):
                    raise EventJournalIntegrityError("v2 Journal anchor is stale")
                wal_generation_id = anchor.wal_generation_id
                wal_record_id = anchor.wal_record_id
                wal_record_hash = anchor.wal_record_hash
                journal_lineage_id = verified.journal_lineage_id
                if verified.records[-1].schema_version == 3:
                    v3_migration_anchor_hash = next(
                        (
                            record.v3_migration_anchor_hash
                            for record in verified.records
                            if record.schema_version == 3
                            and record.lifecycle is EventLifecycle.CHECKPOINT
                            and record.v3_migration_anchor_hash is not None
                        ),
                        None,
                    )
                    if v3_migration_anchor_hash is None:
                        raise EventJournalIntegrityError(
                            "v3 Journal lacks migration authority"
                        )
            checkpoint = self._make_record(
                EventLifecycle.CHECKPOINT,
                previous_hash=verified.records[-1].record_hash,
                processing_sequence=verified.processing_high_water,
                snapshot_sequence=verified.snapshot_sequence,
                snapshot_hash=verified.snapshot_hash,
                schema_version=verified.records[-1].schema_version,
                wal_generation_id=wal_generation_id,
                wal_record_id=wal_record_id,
                wal_record_hash=wal_record_hash,
                journal_lineage_id=journal_lineage_id,
                v3_migration_anchor_hash=v3_migration_anchor_hash,
                external_reconciliation_required=(
                    verified.external_reconciliation_required
                    if verified.records[-1].schema_version >= 2
                    else None
                ),
            )
            temporary = self.path.with_name(f".{self.path.name}.rotation.tmp")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as segment_file:
                segment_file.write(self._record_bytes(checkpoint))
                segment_file.flush()
                os.fsync(segment_file.fileno())
            os.replace(self.path, archived)
            self._fsync_parent()
            os.replace(temporary, self.path)
            self._fsync_parent()
            rotated = self._rotated_paths_unlocked()
            while len(rotated) + 1 > self.retained_files:
                rotated[0][1].unlink()
                self._fsync_parent()
                rotated = rotated[1:]
        except Exception:
            failure = EventJournalAppendError(
                EventJournalAppendStage.ROTATION, published=True
            )
        if failure is not None:
            raise failure

    def _read_records_unlocked(self) -> tuple[EventJournalRecord, ...]:
        artifacts = self._journal_artifacts_unlocked()
        if not artifacts:
            return ()
        records: list[EventJournalRecord] = []
        previous_tail: str | None = None
        for segment_index, path in artifacts:
            segment = self._read_segment(path)
            if not segment:
                raise EventJournalLoadError("EventJournal segment is empty")
            if segment[0].lifecycle is not EventLifecycle.CHECKPOINT:
                raise EventJournalIntegrityError(
                    "EventJournal segment lacks checkpoint anchor"
                )
            if (
                previous_tail is not None
                and segment[0].previous_record_hash != previous_tail
            ):
                raise EventJournalIntegrityError("EventJournal segment chain is broken")
            if (
                previous_tail is None
                and segment_index is None
                and segment[0].previous_record_hash is not None
            ):
                raise EventJournalIntegrityError(
                    "EventJournal rotated predecessor is missing"
                )
            if segment_index is not None and path == self.path:
                raise EventJournalIntegrityError(
                    "EventJournal segment identity is invalid"
                )
            for index, record in enumerate(segment):
                if self._record_hash(record) != record.record_hash:
                    raise EventJournalIntegrityError(
                        "EventJournal record hash mismatch"
                    )
                if (
                    index > 0
                    and record.previous_record_hash != segment[index - 1].record_hash
                ):
                    raise EventJournalIntegrityError(
                        "EventJournal record chain is broken"
                    )
            records.extend(segment)
            previous_tail = segment[-1].record_hash
        return tuple(records)

    def _read_segment(self, path: Path) -> tuple[EventJournalRecord, ...]:
        read_failure: EventJournalLoadError | None = None
        descriptor: int | None = None
        try:
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode):
                raise OSError("journal segment is not regular")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("journal segment is not regular")
            with os.fdopen(descriptor, "rb") as segment_file:
                descriptor = None
                raw = segment_file.read()
        except OSError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            read_failure = EventJournalLoadError("EventJournal segment cannot be read")
        if read_failure is not None:
            raise read_failure
        if not raw or not raw.endswith(b"\n"):
            raise EventJournalLoadError("EventJournal contains a partial record")

        records: list[EventJournalRecord] = []
        parse_failure: EventJournalLoadError | None = None
        for line in raw.splitlines():
            try:
                value = json.loads(
                    line.decode("utf-8"),
                    parse_constant=self._reject_json_constant,
                )
                if not isinstance(value, dict):
                    raise ValueError("record root is invalid")
                if "schema_version" not in value:
                    raise ValueError("record schema version is missing")
                version = value["schema_version"]
                if (
                    isinstance(version, int)
                    and not isinstance(version, bool)
                    and version not in {1, 2, 3}
                ):
                    raise UnsupportedEventJournalVersion(
                        "EventJournal schema version is unsupported"
                    )
                records.append(EventJournalRecord.model_validate_json(line))
            except UnsupportedEventJournalVersion:
                raise
            except (UnicodeError, json.JSONDecodeError, ValueError, ValidationError):
                parse_failure = EventJournalLoadError(
                    "EventJournal contains an invalid record"
                )
                break
        if parse_failure is not None:
            raise parse_failure
        return tuple(records)

    @staticmethod
    def _active_schema(records: tuple[EventJournalRecord, ...]) -> int:
        return records[-1].schema_version

    def _journal_artifacts_unlocked(self) -> list[tuple[int | None, Path]]:
        scan_failure: EventJournalLoadError | None = None
        try:
            candidates = tuple(self.path.parent.glob(f"{self.path.name}.*"))
        except OSError:
            candidates = ()
            scan_failure = EventJournalLoadError(
                "EventJournal artifacts cannot be inspected"
            )
        if scan_failure is not None:
            raise scan_failure

        rotated: list[tuple[int, Path]] = []
        interrupted_failure: EventJournalLoadError | None = None
        try:
            interrupted = tuple(self.path.parent.glob(f".{self.path.name}.rotation*"))
        except OSError:
            interrupted = ()
            interrupted_failure = EventJournalLoadError(
                "EventJournal artifacts cannot be inspected"
            )
        if interrupted_failure is not None:
            raise interrupted_failure
        if interrupted:
            temporary = self.path.with_name(f".{self.path.name}.rotation.tmp")
            try:
                active_exists = self.path.lstat() is not None
            except FileNotFoundError:
                active_exists = False
            if (
                len(interrupted) == 1
                and interrupted[0] == temporary
                and not active_exists
            ):
                try:
                    status = temporary.lstat()
                    if (
                        not stat.S_ISREG(status.st_mode)
                        or status.st_uid != os.geteuid()
                        or stat.S_IMODE(status.st_mode) != 0o600
                    ):
                        raise OSError
                    os.replace(temporary, self.path)
                    self._fsync_parent()
                except OSError:
                    raise EventJournalIntegrityError(
                        "EventJournal interrupted rotation is unsafe"
                    ) from None
            else:
                raise EventJournalIntegrityError(
                    "EventJournal has an interrupted rotation artifact"
                )
        for candidate in candidates:
            suffix = candidate.name.removeprefix(f"{self.path.name}.")
            if not re.fullmatch(r"\d{8}", suffix):
                raise EventJournalIntegrityError(
                    "EventJournal has an interrupted rotation artifact"
                )
            rotated.append((int(suffix), candidate))
        rotated.sort()
        indexes = [index for index, _ in rotated]
        if indexes and indexes != list(range(indexes[0], indexes[-1] + 1)):
            raise EventJournalIntegrityError("EventJournal rotated segment is missing")

        active_failure: EventJournalLoadError | None = None
        try:
            active_exists = self.path.lstat() is not None
        except FileNotFoundError:
            active_exists = False
        except OSError:
            active_exists = False
            active_failure = EventJournalLoadError("EventJournal cannot be inspected")
        if active_failure is not None:
            raise active_failure
        if rotated and not active_exists:
            raise EventJournalIntegrityError("EventJournal active segment is missing")
        artifacts: list[tuple[int | None, Path]] = [*rotated]
        if active_exists:
            artifacts.append((None, self.path))
        return artifacts

    def _verify_records(
        self, records: tuple[EventJournalRecord, ...]
    ) -> _VerifiedJournal:
        if (
            not records
            or records[0].lifecycle is not EventLifecycle.CHECKPOINT
            or records[0].schema_version not in {1, 2, 3}
        ):
            raise EventJournalIntegrityError("EventJournal lacks checkpoint authority")
        checkpoint = records[0]
        assert checkpoint.processing_sequence is not None
        assert checkpoint.snapshot_sequence is not None
        assert checkpoint.snapshot_hash is not None
        high_water = checkpoint.processing_sequence
        snapshot_sequence = checkpoint.snapshot_sequence
        snapshot_hash = checkpoint.snapshot_hash
        journal_lineage_id = checkpoint.journal_lineage_id
        wal_generation_id = checkpoint.wal_generation_id
        wal_record_id = checkpoint.wal_record_id
        wal_record_hash = checkpoint.wal_record_hash
        wal_snapshot_sequence = snapshot_sequence
        wal_snapshot_hash = snapshot_hash
        external_reconciliation_required = bool(
            checkpoint.external_reconciliation_required
        )
        if checkpoint.schema_version >= 2 and (
            checkpoint.wal_generation_id is None
            or checkpoint.wal_record_id is None
            or checkpoint.wal_record_hash is None
            or checkpoint.journal_lineage_id is None
            or checkpoint.migration_previous_v1_hash is not None
            or checkpoint.migration_previous_v2_hash is not None
        ):
            raise EventJournalIntegrityError("WAL-bound checkpoint anchor is invalid")
        if snapshot_sequence > high_water:
            raise EventJournalIntegrityError("Checkpoint sequence is impossible")
        open_events: dict[str, _EventState] = {}
        accepted_queue: list[str] = []
        processing_event_id: str | None = None
        seen_event_ids: set[str] = set()
        seen_record_ids: set[str] = set()
        recovery_open: dict[str, EventJournalRecord] = {}
        seen_recovery_ids: set[str] = set()
        completed_recoveries: dict[str, EventJournalRecord] = {}
        startup_open: dict[str, dict[str, Any]] = {}
        startup_seen_ids: set[str] = set()
        startup_seen_recovery_ids: set[str] = set()
        startup_completed: list[dict[str, Any]] = []
        v2_seen = checkpoint.schema_version == 2
        v3_seen = checkpoint.schema_version == 3
        v3_migration_anchor_hash = checkpoint.v3_migration_anchor_hash
        transactions: dict[str, dict[str, Any]] = {}

        for record in records:
            if record.record_id in seen_record_ids:
                raise EventJournalIntegrityError("Record identifier is duplicated")
            seen_record_ids.add(record.record_id)

        for index, record in enumerate(records[1:], start=1):
            if record.schema_version == 3:
                if not v3_seen and (
                    record.lifecycle is not EventLifecycle.CHECKPOINT
                    or record.migration_previous_v2_hash
                    != records[index - 1].record_hash
                    or records[index - 1].schema_version != 2
                    or record.v3_migration_anchor_hash
                    != record.migration_previous_v2_hash
                ):
                    raise EventJournalIntegrityError("v3 migration boundary is invalid")
                if v3_seen and record.migration_previous_v2_hash is not None:
                    raise EventJournalIntegrityError("v3 migration is duplicated")
                v3_seen = True
            elif record.schema_version == 2:
                if v3_seen:
                    raise EventJournalIntegrityError("schema downgrade is invalid")
                if not v2_seen and (
                    record.lifecycle is not EventLifecycle.CHECKPOINT
                    or record.migration_previous_v1_hash
                    != records[index - 1].record_hash
                    or records[index - 1].schema_version != 1
                ):
                    raise EventJournalIntegrityError("v2 migration boundary is invalid")
                if v2_seen and record.migration_previous_v1_hash is not None:
                    raise EventJournalIntegrityError("v2 migration is duplicated")
                v2_seen = True
            elif v2_seen or v3_seen:
                raise EventJournalIntegrityError("v1 record follows v2 migration")
            if record.lifecycle is EventLifecycle.CHECKPOINT:
                if open_events:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open event lifecycle"
                    )
                if recovery_open:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open recovery lifecycle"
                    )
                if any(t["terminal"] is None for t in transactions.values()):
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open transaction lifecycle"
                    )
                if startup_open:
                    raise EventJournalIntegrityError(
                        "Checkpoint cannot hide open startup reconciliation"
                    )
                if (
                    record.processing_sequence != high_water
                    or record.snapshot_sequence != snapshot_sequence
                    or record.snapshot_hash != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Checkpoint continuity is invalid")
                if (
                    record.schema_version == 3
                    and record.migration_previous_v2_hash is not None
                    and (
                        records[index - 1].schema_version != 2
                        or record.migration_previous_v2_hash
                        != records[index - 1].record_hash
                    )
                ):
                    raise EventJournalIntegrityError("v3 migration anchor is invalid")
                if record.schema_version == 3:
                    if v3_migration_anchor_hash is None:
                        v3_migration_anchor_hash = record.v3_migration_anchor_hash
                    elif record.v3_migration_anchor_hash != v3_migration_anchor_hash:
                        raise EventJournalIntegrityError(
                            "v3 migration authority changed"
                        )
                if (
                    record.schema_version in {2, 3}
                    and record.migration_previous_v1_hash is None
                    and record.snapshot_sequence == wal_snapshot_sequence
                    and record.snapshot_hash == wal_snapshot_hash
                    and (
                        record.wal_generation_id != wal_generation_id
                        or record.wal_record_id != wal_record_id
                        or record.wal_record_hash != wal_record_hash
                        or record.external_reconciliation_required
                        is not external_reconciliation_required
                    )
                ):
                    raise EventJournalIntegrityError("v2 checkpoint anchor changed")
                if (
                    record.schema_version in {2, 3}
                    and record.migration_previous_v1_hash is not None
                ):
                    if (
                        record.migration_previous_v1_hash
                        != records[index - 1].record_hash
                        or records[index - 1].schema_version != 1
                        or record.wal_generation_id is None
                        or record.wal_record_id is None
                        or record.wal_record_hash is None
                    ):
                        raise EventJournalIntegrityError(
                            "v2 migration anchor is invalid"
                        )
                if record.schema_version in {2, 3}:
                    if journal_lineage_id is None:
                        journal_lineage_id = record.journal_lineage_id
                    elif record.journal_lineage_id != journal_lineage_id:
                        raise EventJournalIntegrityError("v2 Journal lineage changed")
                    wal_generation_id = record.wal_generation_id
                    wal_record_id = record.wal_record_id
                    wal_record_hash = record.wal_record_hash
                    wal_snapshot_sequence = record.snapshot_sequence
                    wal_snapshot_hash = record.snapshot_hash
                continue
            if record.lifecycle in {
                EventLifecycle.RECOVERY_PREPARED,
                EventLifecycle.RECOVERY_COMPLETED,
            }:
                if record.schema_version < 2 or record.recovery_id is None:
                    raise EventJournalIntegrityError("v2 recovery identity is invalid")
                if record.recovery_processing_high_water != high_water:
                    raise EventJournalIntegrityError("recovery high-water is invalid")
                prior = recovery_open.get(record.recovery_id)
                if record.lifecycle is EventLifecycle.RECOVERY_PREPARED:
                    if prior is not None or record.recovery_id in seen_recovery_ids:
                        raise EventJournalIntegrityError("recovery is duplicated")
                    if record.recovery_category in {
                        EventRecoveryCategory.EXACT_CURRENT,
                        EventRecoveryCategory.UNCOMMITTED_TAIL,
                    } and (
                        record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "prepared current recovery semantics are invalid"
                        )
                    assert record.snapshot_sequence is not None
                    if (
                        record.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
                        and record.snapshot_sequence >= snapshot_sequence
                    ):
                        raise EventJournalIntegrityError(
                            "prepared rollback semantics are invalid"
                        )
                    recovery_open[record.recovery_id] = record
                    seen_recovery_ids.add(record.recovery_id)
                else:
                    if prior is None or (
                        prior.wal_generation_id != record.wal_generation_id
                        or (
                            prior.wal_record_id is not None
                            and (
                                prior.wal_record_id != record.wal_record_id
                                or prior.wal_record_hash != record.wal_record_hash
                            )
                        )
                        or prior.recovery_processing_high_water
                        != record.recovery_processing_high_water
                        or prior.snapshot_sequence != record.snapshot_sequence
                        or prior.snapshot_hash != record.snapshot_hash
                        or prior.recovery_category != record.recovery_category
                    ):
                        raise EventJournalIntegrityError("recovery pair is invalid")
                    if record.recovery_category in {
                        EventRecoveryCategory.EXACT_CURRENT,
                        EventRecoveryCategory.UNCOMMITTED_TAIL,
                    } and (
                        record.external_reconciliation_required
                        is not external_reconciliation_required
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "current recovery semantics are invalid"
                        )
                    assert record.snapshot_sequence is not None
                    if (
                        record.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
                        and (
                            record.external_reconciliation_required is not True
                            or record.snapshot_sequence >= snapshot_sequence
                        )
                    ):
                        raise EventJournalIntegrityError(
                            "rollback recovery semantics are invalid"
                        )
                    assert record.snapshot_sequence is not None
                    assert record.snapshot_hash is not None
                    snapshot_sequence = record.snapshot_sequence
                    snapshot_hash = record.snapshot_hash
                    external_reconciliation_required = bool(
                        record.external_reconciliation_required
                    )
                    wal_generation_id = record.wal_generation_id
                    wal_record_id = record.wal_record_id
                    wal_record_hash = record.wal_record_hash
                    wal_snapshot_sequence = record.snapshot_sequence
                    wal_snapshot_hash = record.snapshot_hash
                    completed_recoveries[record.recovery_id] = record
                    del recovery_open[record.recovery_id]
                continue
            if record.lifecycle in {
                EventLifecycle.STARTUP_RECONCILIATION_PREPARED,
                EventLifecycle.STARTUP_PARTICIPANT_RECONCILED,
                EventLifecycle.STARTUP_RECONCILIATION_COMPLETED,
            }:
                rid = record.reconciliation_id
                assert rid is not None
                if record.lifecycle is EventLifecycle.STARTUP_RECONCILIATION_PREPARED:
                    recovery = completed_recoveries.get(record.recovery_id or "")
                    recovery_binding = (
                        (
                            recovery.external_reconciliation_required,
                            recovery.snapshot_sequence,
                            recovery.snapshot_hash,
                            recovery.recovery_processing_high_water,
                            recovery.wal_generation_id,
                            recovery.wal_record_id,
                            recovery.wal_record_hash,
                        )
                        if recovery is not None
                        else (
                            external_reconciliation_required,
                            snapshot_sequence,
                            snapshot_hash,
                            high_water,
                            wal_generation_id,
                            wal_record_id,
                            wal_record_hash,
                        )
                    )
                    if (
                        recovery_binding
                        != (
                            True,
                            record.snapshot_sequence,
                            record.snapshot_hash,
                            record.recovery_processing_high_water,
                            record.wal_generation_id,
                            record.wal_record_id,
                            record.wal_record_hash,
                        )
                        or record.journal_lineage_id != journal_lineage_id
                    ):
                        raise EventJournalIntegrityError(
                            "startup reconciliation binding is invalid"
                        )
                    if (
                        rid in startup_seen_ids
                        or rid in startup_open
                        or record.recovery_id in startup_seen_recovery_ids
                    ):
                        raise EventJournalIntegrityError(
                            "startup reconciliation is duplicated"
                        )
                    assert record.required_participants is not None
                    startup_open[rid] = {
                        "record": record,
                        "required": record.required_participants,
                        "outcomes": {},
                        "completed": False,
                    }
                    startup_seen_ids.add(rid)
                    assert record.recovery_id is not None
                    startup_seen_recovery_ids.add(record.recovery_id)
                else:
                    state = startup_open.get(rid)
                    if state is None:
                        raise EventJournalIntegrityError(
                            "startup reconciliation is not open"
                        )
                    prior = state["record"]
                    common = (
                        record.recovery_id,
                        record.snapshot_sequence,
                        record.snapshot_hash,
                        record.recovery_processing_high_water,
                        record.wal_generation_id,
                        record.wal_record_id,
                        record.wal_record_hash,
                        record.journal_lineage_id,
                    )
                    expected = (
                        prior.recovery_id,
                        prior.snapshot_sequence,
                        prior.snapshot_hash,
                        prior.recovery_processing_high_water,
                        prior.wal_generation_id,
                        prior.wal_record_id,
                        prior.wal_record_hash,
                        prior.journal_lineage_id,
                    )
                    if common != expected:
                        raise EventJournalIntegrityError(
                            "startup reconciliation binding changed"
                        )
                    if (
                        record.lifecycle
                        is EventLifecycle.STARTUP_PARTICIPANT_RECONCILED
                    ):
                        assert record.participant_id is not None
                        assert record.operation_digest is not None
                        required = {
                            item.participant_id: item.operation_digest
                            for item in state["required"]
                        }
                        if (
                            record.participant_id not in required
                            or record.operation_digest
                            != required[record.participant_id]
                            or record.participant_id in state["outcomes"]
                        ):
                            raise EventJournalIntegrityError(
                                "startup participant evidence is invalid"
                            )
                        assert record.startup_participant_outcome is not None
                        state["outcomes"][record.participant_id] = (
                            record.operation_digest,
                            record.startup_participant_outcome,
                        )
                    else:
                        required_ids = {
                            item.participant_id for item in state["required"]
                        }
                        if set(state["outcomes"]) != required_ids:
                            raise EventJournalIntegrityError(
                                "startup reconciliation completion is premature"
                            )
                        state["completed"] = True
                        startup_completed.append(state)
                        del startup_open[rid]
                continue
            if record.lifecycle in {
                EventLifecycle.TRANSACTION_PREPARED,
                EventLifecycle.PARTICIPANT_FINALIZED,
                EventLifecycle.TRANSACTION_COMPLETED,
                EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
                EventLifecycle.TRANSACTION_RECONCILED,
                EventLifecycle.PARTICIPANT_ABORTED,
                EventLifecycle.TRANSACTION_ABORT_REQUIRED,
                EventLifecycle.TRANSACTION_ABORTED,
            }:
                if (
                    record.schema_version != 3
                    or record.event_id is None
                    or record.event_type is None
                    or record.source is None
                    or record.transaction_id is None
                ):
                    raise EventJournalIntegrityError("transaction identity is invalid")
                current = open_events.get(record.event_id)
                if (
                    current is None
                    or record.processing_sequence != current.processing_sequence
                ):
                    raise EventJournalIntegrityError(
                        "transaction event anchor is invalid"
                    )
                tx = transactions.get(record.transaction_id)
                if record.lifecycle is EventLifecycle.TRANSACTION_PREPARED:
                    if (
                        tx is not None
                        or current.lifecycle is not EventLifecycle.STARTED
                        or record.transaction_kind is None
                        or record.required_participants is None
                    ):
                        raise EventJournalIntegrityError(
                            "transaction preparation is invalid"
                        )
                    transactions[record.transaction_id] = {
                        "event_id": record.event_id,
                        "event_type": record.event_type,
                        "source": record.source,
                        "sequence": record.processing_sequence,
                        "kind": record.transaction_kind,
                        "required": record.required_participants,
                        "outcomes": {},
                        "reason": None,
                        "unresolved": (),
                        "abort_outcomes": {},
                        "abort_reason": None,
                        "branch": None,
                        "terminal": None,
                    }
                else:
                    if (
                        tx is None
                        or current.lifecycle
                        not in {EventLifecycle.PREPARED, EventLifecycle.STARTED}
                        or (
                            tx["event_id"],
                            tx["event_type"],
                            tx["source"],
                            tx["sequence"],
                        )
                        != (
                            record.event_id,
                            record.event_type,
                            record.source,
                            record.processing_sequence,
                        )
                    ):
                        raise EventJournalIntegrityError("transaction identity changed")
                    required = {
                        item.participant_id: item.operation_digest
                        for item in tx["required"]
                    }
                    if (
                        record.lifecycle
                        in {
                            EventLifecycle.PARTICIPANT_FINALIZED,
                            EventLifecycle.TRANSACTION_COMPLETED,
                            EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
                            EventLifecycle.TRANSACTION_RECONCILED,
                        }
                        and current.lifecycle is not EventLifecycle.PREPARED
                    ):
                        raise EventJournalIntegrityError(
                            "finalization evidence is before internal prepare"
                        )
                    if (
                        record.lifecycle
                        in {
                            EventLifecycle.PARTICIPANT_ABORTED,
                            EventLifecycle.TRANSACTION_ABORT_REQUIRED,
                            EventLifecycle.TRANSACTION_ABORTED,
                        }
                        and current.lifecycle is not EventLifecycle.STARTED
                    ):
                        raise EventJournalIntegrityError(
                            "abort evidence is after internal prepare"
                        )
                    if record.lifecycle is EventLifecycle.PARTICIPANT_FINALIZED:
                        if tx["branch"] == "abort":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "finalize"
                        if (
                            record.participant_id is None
                            or record.operation_digest
                            != required.get(record.participant_id)
                            or record.participant_id in tx["outcomes"]
                            or tx["terminal"] is not None
                        ):
                            raise EventJournalIntegrityError(
                                "participant finalization is invalid"
                            )
                        assert record.participant_outcome is not None
                        tx["outcomes"][record.participant_id] = (
                            record.participant_outcome
                        )
                        if tx["reason"] is not None:
                            tx["unresolved"] = tuple(
                                sorted(set(required) - set(tx["outcomes"]))
                            )
                    elif (
                        record.lifecycle
                        is EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED
                    ):
                        if tx["branch"] == "abort":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "finalize"
                        missing = tuple(sorted(set(required) - set(tx["outcomes"])))
                        if (
                            tx["reason"] is not None
                            or tuple(record.unresolved_participants or ()) != missing
                            or not missing
                        ):
                            raise EventJournalIntegrityError(
                                "transaction reconciliation is invalid"
                            )
                        tx["reason"] = record.reconciliation_reason
                        tx["unresolved"] = missing
                    elif record.lifecycle is EventLifecycle.TRANSACTION_COMPLETED:
                        if (
                            tx["branch"] == "abort"
                            or tx["terminal"] is not None
                            or tx["reason"] is not None
                            or set(tx["outcomes"]) != set(required)
                        ):
                            raise EventJournalIntegrityError(
                                "transaction completion is premature"
                            )
                        tx["terminal"] = EventLifecycle.TRANSACTION_COMPLETED
                    elif record.lifecycle is EventLifecycle.PARTICIPANT_ABORTED:
                        if tx["branch"] == "finalize":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "abort"
                        requirements = {
                            item.participant_id: item for item in tx["required"]
                        }
                        requirement = requirements.get(record.participant_id or "")
                        if (
                            requirement is None
                            or ParticipantCapability.ABORT
                            not in requirement.capabilities
                            or record.operation_digest != requirement.operation_digest
                            or record.participant_id in tx["abort_outcomes"]
                            or tx["terminal"] is not None
                        ):
                            raise EventJournalIntegrityError(
                                "participant abort is invalid"
                            )
                        assert record.abort_outcome is not None
                        tx["abort_outcomes"][record.participant_id] = (
                            record.abort_outcome
                        )
                        if tx["abort_reason"] is not None:
                            tx["unresolved"] = tuple(
                                sorted(
                                    participant
                                    for participant, item in requirements.items()
                                    if ParticipantCapability.ABORT in item.capabilities
                                    and participant not in tx["abort_outcomes"]
                                )
                            )
                    elif record.lifecycle is EventLifecycle.TRANSACTION_ABORT_REQUIRED:
                        if tx["branch"] == "finalize":
                            raise EventJournalIntegrityError(
                                "transaction branches are mixed"
                            )
                        tx["branch"] = "abort"
                        requirements = {
                            item.participant_id: item for item in tx["required"]
                        }
                        missing = tuple(
                            sorted(
                                participant
                                for participant, item in requirements.items()
                                if ParticipantCapability.ABORT in item.capabilities
                                and participant not in tx["abort_outcomes"]
                            )
                        )
                        if (
                            tx["abort_reason"] is not None
                            or tuple(record.unresolved_participants or ()) != missing
                            or not missing
                        ):
                            raise EventJournalIntegrityError(
                                "transaction abort is invalid"
                            )
                        tx["abort_reason"] = record.abort_reason
                        tx["unresolved"] = missing
                    elif record.lifecycle is EventLifecycle.TRANSACTION_ABORTED:
                        if tx["branch"] != "abort":
                            raise EventJournalIntegrityError(
                                "transaction abort branch is invalid"
                            )
                        abort_ids = {
                            item.participant_id
                            for item in tx["required"]
                            if ParticipantCapability.ABORT in item.capabilities
                        }
                        if (
                            tx["terminal"] is not None
                            or set(tx["abort_outcomes"]) != abort_ids
                        ):
                            raise EventJournalIntegrityError(
                                "transaction abort is incomplete"
                            )
                        tx["unresolved"] = ()
                        tx["terminal"] = EventLifecycle.TRANSACTION_ABORTED
                    elif (
                        tx["reason"] is None
                        or set(tx["outcomes"]) != set(required)
                        or tx["terminal"] is not None
                        or tx["branch"] == "abort"
                    ):
                        raise EventJournalIntegrityError(
                            "transaction reconciliation is incomplete"
                        )
                    elif tx["branch"] != "abort":
                        tx["terminal"] = EventLifecycle.TRANSACTION_RECONCILED
                    else:
                        raise EventJournalIntegrityError(
                            "transaction branch is invalid"
                        )
                continue
            assert record.event_id is not None
            assert record.event_type is not None
            assert record.source is not None
            current = open_events.get(record.event_id)
            if record.lifecycle is EventLifecycle.ACCEPTED:
                if record.event_id in seen_event_ids:
                    raise EventJournalIntegrityError("Event identifier is duplicated")
                seen_event_ids.add(record.event_id)
                open_events[record.event_id] = _EventState(
                    record.event_id,
                    record.event_type,
                    record.source,
                    record.timestamp,
                    EventLifecycle.ACCEPTED,
                    None,
                )
                accepted_queue.append(record.event_id)
                continue
            if current is None:
                raise EventJournalIntegrityError("Event lifecycle lacks acceptance")
            if (record.event_type, record.source) != (
                current.event_type,
                current.source,
            ):
                raise EventJournalIntegrityError("Event metadata changed")

            if record.lifecycle is EventLifecycle.STARTED:
                if (
                    current.lifecycle is not EventLifecycle.ACCEPTED
                    or processing_event_id is not None
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                ):
                    raise EventJournalIntegrityError("Started lifecycle is impossible")
                if record.processing_sequence != high_water + 1:
                    raise EventJournalIntegrityError("Processing sequence has a gap")
                high_water = record.processing_sequence
                processing_event_id = record.event_id
                open_events[record.event_id] = _EventState(
                    current.event_id,
                    current.event_type,
                    current.source,
                    current.requested_at,
                    EventLifecycle.STARTED,
                    record.processing_sequence,
                )
            elif record.lifecycle is EventLifecycle.PREPARED:
                if (
                    processing_event_id != record.event_id
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                    or current.lifecycle is not EventLifecycle.STARTED
                    or record.processing_sequence != current.processing_sequence
                    or record.state_hash_before != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Prepared lifecycle is impossible")
                if any(
                    transaction["event_id"] == record.event_id
                    and transaction["terminal"] is EventLifecycle.TRANSACTION_ABORTED
                    for transaction in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "aborted transaction cannot cross internal prepare"
                    )
                open_events[record.event_id] = _EventState(
                    current.event_id,
                    current.event_type,
                    current.source,
                    current.requested_at,
                    EventLifecycle.PREPARED,
                    current.processing_sequence,
                    record.state_hash_before,
                    record.state_hash_after,
                    record.wal_generation_id,
                )
            elif record.lifecycle is EventLifecycle.COMPLETED:
                if (
                    processing_event_id != record.event_id
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                    or current.lifecycle is not EventLifecycle.PREPARED
                    or record.processing_sequence != current.processing_sequence
                    or record.snapshot_sequence != current.processing_sequence
                    or record.snapshot_hash != current.state_hash_after
                    or (
                        record.schema_version >= 2
                        and record.wal_generation_id != current.wal_generation_id
                    )
                ):
                    raise EventJournalIntegrityError(
                        "Completed lifecycle is impossible"
                    )
                if any(
                    t["event_id"] == record.event_id and t["terminal"] is None
                    for t in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "event completed with open transaction evidence"
                    )
                assert record.snapshot_sequence is not None
                assert record.snapshot_hash is not None
                snapshot_sequence = record.snapshot_sequence
                snapshot_hash = record.snapshot_hash
                wal_generation_id = record.wal_generation_id
                wal_record_id = record.wal_record_id
                wal_record_hash = record.wal_record_hash
                wal_snapshot_sequence = record.snapshot_sequence
                wal_snapshot_hash = record.snapshot_hash
                del open_events[record.event_id]
                accepted_queue.pop(0)
                processing_event_id = None
            elif record.lifecycle is EventLifecycle.FAILED:
                if (
                    processing_event_id != record.event_id
                    or not accepted_queue
                    or accepted_queue[0] != record.event_id
                    or current.lifecycle is not EventLifecycle.STARTED
                    or record.processing_sequence != current.processing_sequence
                    or record.snapshot_sequence != snapshot_sequence
                    or record.snapshot_hash != snapshot_hash
                ):
                    raise EventJournalIntegrityError("Failed lifecycle is impossible")
                if any(
                    t["event_id"] == record.event_id and t["terminal"] is None
                    for t in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "event failed with open transaction evidence"
                    )
                del open_events[record.event_id]
                accepted_queue.pop(0)
                processing_event_id = None
            elif record.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
                if record.snapshot_sequence is None or record.snapshot_hash is None:
                    raise EventJournalIntegrityError("Recovery snapshot is missing")
                if any(
                    transaction["event_id"] == record.event_id
                    and transaction["terminal"] is None
                    for transaction in transactions.values()
                ):
                    raise EventJournalIntegrityError(
                        "recovery cannot hide open transaction evidence"
                    )
                if record.failure_category is EventFailureCategory.ACCEPTED_NOT_STARTED:
                    if (
                        current.lifecycle is not EventLifecycle.ACCEPTED
                        or processing_event_id is not None
                        or not accepted_queue
                        or accepted_queue[0] != record.event_id
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "Accepted recovery is impossible"
                        )
                elif (
                    record.failure_category
                    is EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                ):
                    if (
                        processing_event_id != record.event_id
                        or not accepted_queue
                        or accepted_queue[0] != record.event_id
                        or current.lifecycle
                        not in {
                            EventLifecycle.STARTED,
                            EventLifecycle.PREPARED,
                        }
                        or record.processing_sequence != current.processing_sequence
                        or record.snapshot_sequence != snapshot_sequence
                        or record.snapshot_hash != snapshot_hash
                    ):
                        raise EventJournalIntegrityError(
                            "Uncommitted recovery is impossible"
                        )
                elif (
                    record.failure_category
                    is EventFailureCategory.COMMITTED_BEFORE_CRASH
                ):
                    if (
                        processing_event_id != record.event_id
                        or not accepted_queue
                        or accepted_queue[0] != record.event_id
                        or current.lifecycle is not EventLifecycle.PREPARED
                        or record.processing_sequence != current.processing_sequence
                        or record.snapshot_sequence != current.processing_sequence
                        or record.snapshot_hash != current.state_hash_after
                    ):
                        raise EventJournalIntegrityError(
                            "Committed recovery is impossible"
                        )
                    snapshot_sequence = record.snapshot_sequence
                    snapshot_hash = record.snapshot_hash
                else:
                    raise EventJournalIntegrityError("Recovery category is impossible")
                del open_events[record.event_id]
                accepted_queue.pop(0)
                processing_event_id = None
            else:
                raise EventJournalIntegrityError("Lifecycle is impossible")

        transaction_views = tuple(
            EventJournalTransaction(
                transaction_id=transaction_id,
                event_id=value["event_id"],
                event_type=value["event_type"],
                source=value["source"],
                processing_sequence=value["sequence"],
                kind=value["kind"],
                required_participants=value["required"],
                participant_outcomes=tuple(sorted(value["outcomes"].items())),
                abort_outcomes=tuple(sorted(value["abort_outcomes"].items())),
                reconciliation_reason=value["reason"],
                abort_reason=value["abort_reason"],
                unresolved_participants=value["unresolved"],
                terminal_lifecycle=value["terminal"],
            )
            for transaction_id, value in transactions.items()
        )
        startup_views = tuple(
            EventJournalStartupReconciliation(
                reconciliation_id=rid,
                recovery_id=state["record"].recovery_id or "",
                snapshot_sequence=state["record"].snapshot_sequence or 0,
                snapshot_hash=state["record"].snapshot_hash or "0" * 64,
                recovery_processing_high_water=(
                    state["record"].recovery_processing_high_water or 0
                ),
                wal_generation_id=state["record"].wal_generation_id or "",
                wal_record_id=state["record"].wal_record_id or "",
                wal_record_hash=state["record"].wal_record_hash or "0" * 64,
                journal_lineage_id=state["record"].journal_lineage_id or "",
                required_participants=state["required"],
                participant_outcomes=tuple(
                    (participant, digest, outcome)
                    for participant, (digest, outcome) in sorted(
                        state["outcomes"].items()
                    )
                ),
                completed=state["completed"],
            )
            for rid, state in [
                *startup_open.items(),
                *[
                    (item["record"].reconciliation_id, item)
                    for item in startup_completed
                ],
            ]
        )
        return _VerifiedJournal(
            records,
            high_water,
            snapshot_sequence,
            snapshot_hash,
            tuple(open_events.values()),
            journal_lineage_id,
            external_reconciliation_required,
            wal_snapshot_sequence,
            wal_snapshot_hash,
            tuple(recovery_open.values()),
            wal_generation_id,
            wal_record_id,
            wal_record_hash,
            transaction_views,
            startup_views,
        )

    def _rotated_paths_unlocked(self) -> list[tuple[int, Path]]:
        return [
            (index, path)
            for index, path in self._journal_artifacts_unlocked()
            if index is not None
        ]

    def _segment_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index:08d}")

    def _fsync_parent(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _run_hook(
        self, lifecycle: EventLifecycle, stage: EventJournalAppendStage
    ) -> None:
        if self._append_stage_hook is not None:
            self._append_stage_hook(lifecycle, stage)

    def _require_authority(self) -> None:
        if not self._lease.held:
            raise EventJournalLoadError(
                "EventJournal exclusive authority is unavailable"
            )

    @staticmethod
    def _validate_snapshot_identity(sequence: int, snapshot_hash: str) -> None:
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or not isinstance(snapshot_hash, str)
            or _HASH_PATTERN.fullmatch(snapshot_hash) is None
        ):
            raise ValueError("canonical snapshot identity is invalid")

    @staticmethod
    def _reject_json_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")
