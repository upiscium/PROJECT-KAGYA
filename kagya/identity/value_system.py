"""Immutable value primitives and bounded process-local Value authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Final, TypeVar, cast

from kagya.identifiers import validate_identifier
from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
)

__all__ = [
    "IdentityOrigin",
    "OriginActor",
    "OriginInputKind",
    "ValueAdmissionStatus",
    "ValueConflictDefinition",
    "ValueEvidence",
    "ValueDomainError",
    "ValueEventResult",
    "ValueMutationEvidence",
    "ValueMutationReason",
    "ValueMutationResult",
    "ValueMutationStatus",
    "ValueNotFound",
    "ValueOriginReviewDecision",
    "ValueProposal",
    "ValueReason",
    "ValueRevisionOperation",
    "ValueRevisionRecord",
    "ValueRevisionHistory",
    "ValueSeedDeclaration",
    "ValueSelfAdmission",
    "ValueScope",
    "ValueState",
    "ValueSystemSnapshot",
    "ValueSystem",
    "canonical_evidence_ledger_payload",
    "canonical_revision_record_payload",
    "canonical_seed_payload",
    "canonical_value_state_payload",
    "evidence_ledger_digest",
    "recompute_revision_record_digest",
    "recompute_seed_contract_digest",
    "validate_seed_contract_digest",
    "validate_revision_record_digest",
    "value_state_digest",
]


class ValueScope(str, Enum):
    SUBJECT = "subject"
    CONTEXT = "context"


class ValueReason(str, Enum):
    OBSERVATION = "observation"
    FEEDBACK = "feedback"
    PROPOSAL = "proposal"
    CONFLICT = "conflict"
    SEED = "seed"


class ValueMutationReason(str, Enum):
    SELF_ADMISSION = "self_admission"
    ADMITTED_UPDATE = "admitted_update"


class ValueRevisionOperation(str, Enum):
    ADMISSION = "admission"
    UPDATE = "update"
    FREEZE = "freeze"
    UNFREEZE = "unfreeze"
    ROLLBACK = "rollback"
    ORIGIN_REVIEW = "origin_review"


class ValueOriginReviewDecision(str, Enum):
    ACCEPT_PROVENANCE = "accept_provenance"
    REJECT = "reject"


class ValueMutationStatus(str, Enum):
    APPLIED = "applied"
    NO_CHANGE = "no_change"
    IDEMPOTENT = "idempotent"
    FROZEN = "frozen"


_NAME_LIMIT: Final = 128
_CONCEPT_LIMIT: Final = 2048
_MAX_REFS: Final = 16
_MAX_AUTHORITATIVE_VALUES: Final = 128
_MAX_REVISION_RECORDS: Final = 32
_MAX_APPLIED_EVIDENCE_REFS: Final = 512
_MAX_OPPOSITION_COUNT: Final = 6
_EVENT_UPDATE_BUDGET: Final = 0.10
_SEED_DOMAIN: Final = "kagya.identity.value-seed/v1"
_STATE_DOMAIN: Final = "kagya.identity.value-state/v1"
_RECORD_DOMAIN: Final = "kagya.identity.value-revision/v1"
_GENESIS_DOMAIN: Final = "kagya.identity.value-genesis/v1"
_LEDGER_DOMAIN: Final = "kagya.identity.value-evidence-ledger/v1"


_ValueEnum = TypeVar("_ValueEnum", bound=Enum)


def _enum(value: object, enum_type: type[_ValueEnum], name: str) -> _ValueEnum:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return cast(_ValueEnum, value)


def _text(value: object, name: str, limit: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value or len(value) > limit or any(
        ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise ValueError(f"{name} must be nonempty, bounded, and single-line")
    return value


def _fraction(value: object, name: str, *, nonzero: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite float")
    if not 0.0 <= value <= 1.0 or (nonzero and value <= 0.0):
        raise ValueError(f"{name} must be in the permitted range")
    return value


def _refs(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of identifiers")
    if len(value) > _MAX_REFS:
        raise ValueError(f"{name} has too many entries")
    checked = tuple(validate_identifier(item) for item in value)
    if checked != tuple(sorted(set(checked))) or len(set(checked)) != len(checked):
        raise ValueError(f"{name} must be sorted and unique")
    return checked


def _ledger_refs(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of identifiers")
    if len(value) > _MAX_APPLIED_EVIDENCE_REFS:
        raise ValueError(f"{name} has too many entries")
    checked = tuple(validate_identifier(item) for item in value)
    if checked != tuple(sorted(set(checked))) or len(set(checked)) != len(checked):
        raise ValueError(f"{name} must be sorted and unique")
    return checked


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return validate_identifier(value)


def _canonical_contexts(
    value: object, scope: ValueScope, name: str
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of identifiers")
    contexts = tuple(validate_identifier(item) for item in value)
    if contexts != tuple(sorted(set(contexts))) or len(set(contexts)) != len(contexts):
        raise ValueError(f"{name} must be sorted and unique")
    if scope is ValueScope.SUBJECT and contexts:
        raise ValueError("subject values cannot have contexts")
    if scope is ValueScope.CONTEXT and not 1 <= len(contexts) <= _MAX_REFS:
        raise ValueError("context values require one to sixteen contexts")
    return contexts


def _optional_text(value: object, name: str, limit: int) -> str | None:
    if value is None:
        return None
    return _text(value, name, limit)


def _validate_seed_digest_format(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError("seed_contract_digest must be a lowercase SHA-256 digest")
    return value


def _validate_sha256_digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_utc_datetime(value: object, name: str = "recorded_at") -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be an exact datetime")
    if value.tzinfo is not timezone.utc or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use canonical UTC")
    return value


def _signed_fraction(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite float")
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between -1 and 1")
    return value


@dataclass(frozen=True, slots=True)
class ValueSeedDeclaration:
    """Immutable seed semantics, independent of admission or runtime state."""

    value_id: str
    name: str
    concept: str | None
    scope: ValueScope
    context_ids: tuple[str, ...]
    polarity: int
    initial_strength: float
    confidence: float
    stability: float
    protectedness: float
    negotiability: float
    allowed_update_rate: float

    def __post_init__(self) -> None:
        validate_identifier(self.value_id)
        _text(self.name, "name", _NAME_LIMIT)
        _optional_text(self.concept, "concept", _CONCEPT_LIMIT)
        scope = _enum(self.scope, ValueScope, "scope")
        object.__setattr__(self, "scope", scope)
        contexts = _canonical_contexts(self.context_ids, scope, "context_ids")
        object.__setattr__(self, "context_ids", contexts)
        if type(self.polarity) is not int or self.polarity not in (-1, 1):
            raise TypeError("polarity must be exactly -1 or 1")
        _fraction(self.initial_strength, "initial_strength")
        for field_name in (
            "confidence",
            "stability",
            "protectedness",
            "negotiability",
        ):
            _fraction(getattr(self, field_name), field_name)
        _fraction(self.allowed_update_rate, "allowed_update_rate", nonzero=True)

@dataclass(frozen=True, slots=True)
class ValueState:
    value_id: str
    revision: int
    name: str
    concept: str | None
    scope: ValueScope
    context_ids: tuple[str, ...]
    polarity: int
    strength: float
    confidence: float
    stability: float
    protectedness: float
    negotiability: float
    allowed_update_rate: float
    frozen: bool
    origin: IdentityOrigin
    evidence_refs: tuple[str, ...]
    seed_contract_digest: str | None = None
    opposition_count: int = 0

    def __post_init__(self) -> None:
        validate_identifier(self.value_id)
        if type(self.revision) is not int or self.revision < 0:
            raise TypeError("revision must be a nonnegative exact integer")
        _text(self.name, "name", _NAME_LIMIT)
        _optional_text(self.concept, "concept", _CONCEPT_LIMIT)
        scope = _enum(self.scope, ValueScope, "scope")
        object.__setattr__(self, "scope", scope)
        contexts = _canonical_contexts(self.context_ids, scope, "context_ids")
        object.__setattr__(self, "context_ids", contexts)
        if type(self.polarity) is not int or self.polarity not in (-1, 1):
            raise TypeError("polarity must be exactly -1 or 1")
        for field_name in ("strength", "confidence", "stability", "protectedness", "negotiability"):
            _fraction(getattr(self, field_name), field_name)
        _fraction(self.allowed_update_rate, "allowed_update_rate", nonzero=True)
        if type(self.frozen) is not bool:
            raise TypeError("frozen must be an exact boolean")
        if type(self.opposition_count) is not int or not 0 <= self.opposition_count <= _MAX_OPPOSITION_COUNT:
            raise ValueError("opposition_count must be an exact integer between 0 and 6")
        if not isinstance(self.origin, IdentityOrigin):
            raise TypeError("origin must be an IdentityOrigin")
        evidence = _refs(self.evidence_refs, "evidence_refs")
        object.__setattr__(self, "evidence_refs", evidence)
        if self.seed_contract_digest is not None:
            _validate_seed_digest_format(self.seed_contract_digest)
        if self.origin.admission is ValueAdmissionStatus.SYSTEM_AUTHORIZED:
            if self.seed_contract_digest is None:
                raise ValueError("system-authorized values require a seed contract digest")
        elif self.seed_contract_digest is not None:
            raise ValueError("only system-authorized values may carry a seed contract digest")

    def is_active(self) -> bool:
        """Return whether this declaration is admitted for consideration."""

        return self.origin.admission in {
            ValueAdmissionStatus.SELF_ENDORSED,
            ValueAdmissionStatus.SYSTEM_AUTHORIZED,
        }

    def applies_to(self, context_id: str | None) -> bool:
        """Read applicability without changing the immutable declaration."""

        if context_id is not None:
            validate_identifier(context_id)
        if self.scope is ValueScope.SUBJECT:
            return True
        if context_id is None:
            return False
        return context_id in self.context_ids


def _origin_fields(origin: IdentityOrigin) -> dict[str, object]:
    return {
        "actor": origin.actor.value,
        "admission": origin.admission.value,
        "confidence": origin.confidence,
        "context_id": origin.context_id,
        "event_id": origin.event_id,
        "event_sequence": origin.event_sequence,
        "input_kind": origin.input_kind.value,
        "origin_id": origin.origin_id,
        "source_ref": origin.source_ref,
    }


def _state_fields(value: ValueState) -> dict[str, object]:
    return {
        "allowed_update_rate": value.allowed_update_rate,
        "confidence": value.confidence,
        "concept": value.concept,
        "context_ids": list(value.context_ids),
        "evidence_refs": list(value.evidence_refs),
        "frozen": value.frozen,
        "name": value.name,
        "opposition_count": value.opposition_count,
        "origin": _origin_fields(value.origin),
        "polarity": value.polarity,
        "protectedness": value.protectedness,
        "negotiability": value.negotiability,
        "revision": value.revision,
        "scope": value.scope.value,
        "seed_contract_digest": value.seed_contract_digest,
        "stability": value.stability,
        "strength": value.strength,
        "value_id": value.value_id,
    }


def canonical_value_state_payload(value: ValueState) -> bytes:
    """Return canonical, domain-separated bytes for one current Value state."""

    if not isinstance(value, ValueState):
        raise TypeError("value must be a ValueState")
    encoded = json.dumps(
        _state_fields(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return _STATE_DOMAIN.encode("ascii") + b"\0" + encoded


def value_state_digest(value: ValueState) -> str:
    """Return the content address of all authoritative current Value fields."""

    return hashlib.sha256(canonical_value_state_payload(value)).hexdigest()


def canonical_evidence_ledger_payload(
    value_id: str, evidence_refs: tuple[str, ...]
) -> bytes:
    """Return canonical bytes authenticating one complete evidence ledger."""

    checked_value_id = validate_identifier(value_id)
    checked_refs = _ledger_refs(evidence_refs, "evidence_refs")
    encoded = json.dumps(
        {"evidence_refs": list(checked_refs), "value_id": checked_value_id},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _LEDGER_DOMAIN.encode("ascii") + b"\0" + encoded


def evidence_ledger_digest(value_id: str, evidence_refs: tuple[str, ...]) -> str:
    """Return the content address of one complete bounded evidence ledger."""

    return hashlib.sha256(
        canonical_evidence_ledger_payload(value_id, evidence_refs)
    ).hexdigest()


class ValueDomainError(ValueError):
    """A fail-closed domain error raised by the process-local Value authority."""


class ValueNotFound(ValueDomainError):
    """The requested stored Value does not exist."""


@dataclass(frozen=True, slots=True)
class ValueMutationEvidence:
    """Explicit event identity supplied by the future runtime integration."""

    event_id: str
    event_sequence: int
    recorded_at: datetime

    def __post_init__(self) -> None:
        validate_identifier(self.event_id)
        if type(self.event_sequence) is not int or self.event_sequence < 0:
            raise TypeError("event_sequence must be a nonnegative exact integer")
        _validate_utc_datetime(self.recorded_at)


@dataclass(frozen=True, slots=True)
class ValueSelfAdmission:
    """Strict self-origin intent accepted by the U2 Value authority."""

    target_value_id: str
    subject_origin: IdentityOrigin
    evidence_refs: tuple[str, ...]
    requested_delta: float
    confidence: float
    reason: ValueMutationReason

    def __post_init__(self) -> None:
        validate_identifier(self.target_value_id)
        if not isinstance(self.subject_origin, IdentityOrigin):
            raise TypeError("subject_origin must be an IdentityOrigin")
        if not (
            self.subject_origin.actor is OriginActor.SELF
            and self.subject_origin.input_kind is OriginInputKind.INTERNAL_STATE
            and self.subject_origin.admission is ValueAdmissionStatus.SELF_ENDORSED
            and self.subject_origin.event_id is not None
            and self.subject_origin.event_sequence is not None
        ):
            raise ValueError("subject_origin must be self/internal_state/self_endorsed")
        refs = _refs(self.evidence_refs, "evidence_refs")
        if not refs:
            raise ValueError("evidence_refs must be nonempty")
        object.__setattr__(self, "evidence_refs", refs)
        _signed_fraction(self.requested_delta, "requested_delta")
        _fraction(self.confidence, "confidence")
        _enum(self.reason, ValueMutationReason, "reason")


def _seed_fields(seed: ValueSeedDeclaration) -> dict[str, object]:
    return {
        "allowed_update_rate": seed.allowed_update_rate,
        "confidence": seed.confidence,
        "concept": seed.concept,
        "context_ids": list(seed.context_ids),
        "initial_strength": seed.initial_strength,
        "negotiability": seed.negotiability,
        "name": seed.name,
        "polarity": seed.polarity,
        "protectedness": seed.protectedness,
        "scope": seed.scope.value,
        "stability": seed.stability,
        "value_id": seed.value_id,
    }


def canonical_seed_payload(seed: ValueSeedDeclaration) -> bytes:
    """Return the domain-separated canonical seed declaration bytes."""

    if not isinstance(seed, ValueSeedDeclaration):
        raise TypeError("seed must be a ValueSeedDeclaration")
    encoded = json.dumps(
        _seed_fields(seed), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return _SEED_DOMAIN.encode("ascii") + b"\0" + encoded


def recompute_seed_contract_digest(seed: ValueSeedDeclaration) -> str:
    return hashlib.sha256(canonical_seed_payload(seed)).hexdigest()


def validate_seed_contract_digest(seed: ValueSeedDeclaration, digest: str) -> str:
    _validate_seed_digest_format(digest)
    expected = recompute_seed_contract_digest(seed)
    if digest != expected:
        raise ValueError("seed_contract_digest does not match the seed declaration")
    return digest


@dataclass(frozen=True, slots=True)
class ValueConflictDefinition:
    left_value_id: str
    right_value_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.left_value_id)
        validate_identifier(self.right_value_id)
        if self.left_value_id == self.right_value_id:
            raise ValueError("a value cannot conflict with itself")
        if self.left_value_id > self.right_value_id:
            raise ValueError("conflict IDs must be in canonical pair order")


@dataclass(frozen=True, slots=True)
class ValueEvidence:
    evidence_ref: str
    origin: IdentityOrigin
    reason: ValueReason
    target_value_id: str | None = None
    confidence: float = 1.0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.evidence_ref)
        if not isinstance(self.origin, IdentityOrigin):
            raise TypeError("origin must be an IdentityOrigin")
        _enum(self.reason, ValueReason, "reason")
        _optional_identifier(self.target_value_id, "target_value_id")
        _fraction(self.confidence, "confidence")
        _refs(self.evidence_refs, "evidence_refs")


@dataclass(frozen=True, slots=True)
class ValueProposal(ValueEvidence):
    """A bounded observation; no method here applies it to ValueState."""


def _genesis_digest(value_id: str) -> str:
    validate_identifier(value_id)
    return hashlib.sha256(
        _GENESIS_DOMAIN.encode("ascii") + b"\0" + value_id.encode("ascii")
    ).hexdigest()


def _canonical_record_fields(record: ValueRevisionRecord) -> dict[str, object]:
    return {
        "after_digest": record.after_digest,
        "after_state_projection": _state_fields(record.after_state_projection),
        "before_digest": record.before_digest,
        "event_id": record.event_id,
        "event_sequence": record.event_sequence,
        "evidence_refs": list(record.evidence_refs),
        "from_revision": record.from_revision,
        "operation": record.operation.value,
        "origin_id": record.origin_id,
        "previous_record_digest": record.previous_record_digest,
        "recorded_at": record.recorded_at.isoformat(timespec="microseconds"),
        "target_revision": record.target_revision,
        "to_revision": record.to_revision,
        "value_id": record.value_id,
    }


def canonical_revision_record_payload(record: ValueRevisionRecord) -> bytes:
    """Return canonical, domain-separated bytes for a revision record."""

    if not isinstance(record, ValueRevisionRecord):
        raise TypeError("record must be a ValueRevisionRecord")
    encoded = json.dumps(
        _canonical_record_fields(record),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _RECORD_DOMAIN.encode("ascii") + b"\0" + encoded


def recompute_revision_record_digest(record: ValueRevisionRecord) -> str:
    return hashlib.sha256(canonical_revision_record_payload(record)).hexdigest()


def validate_revision_record_digest(record: ValueRevisionRecord) -> str:
    expected = recompute_revision_record_digest(record)
    if record.record_digest != expected:
        raise ValueError("record_digest does not match the immutable revision record")
    return record.record_digest


@dataclass(frozen=True, slots=True)
class ValueRevisionRecord:
    """One immutable link in a per-Value revision chain."""

    value_id: str
    from_revision: int
    to_revision: int
    before_digest: str
    after_state_projection: ValueState
    after_digest: str
    operation: ValueRevisionOperation
    origin_id: str
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    recorded_at: datetime
    previous_record_digest: str | None = None
    target_revision: int | None = None
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        validate_identifier(self.value_id)
        if type(self.from_revision) is not int:
            raise TypeError("from_revision must be an exact integer")
        if type(self.to_revision) is not int:
            raise TypeError("to_revision must be an exact integer")
        _enum(self.operation, ValueRevisionOperation, "operation")
        if self.operation is ValueRevisionOperation.ADMISSION:
            if (self.from_revision, self.to_revision) != (-1, 0):
                raise ValueError("admission records must begin at revision zero")
        elif self.from_revision < 0 or self.to_revision != self.from_revision + 1:
            raise ValueError("ordinary records must increment revision exactly once")
        if self.operation is ValueRevisionOperation.ROLLBACK:
            if type(self.target_revision) is not int or self.target_revision < 0:
                raise TypeError("rollback records require a nonnegative target revision")
        elif self.target_revision is not None:
            raise ValueError("only rollback records may name a target revision")
        if not isinstance(self.after_state_projection, ValueState):
            raise TypeError("after_state_projection must be a ValueState")
        if self.after_state_projection.value_id != self.value_id:
            raise ValueError("record value_id must match its after-state projection")
        if self.after_state_projection.revision != self.to_revision:
            raise ValueError("record to_revision must match its after-state projection")
        _validate_sha256_digest(self.before_digest, "before_digest")
        _validate_sha256_digest(self.after_digest, "after_digest")
        if (
            self.operation is ValueRevisionOperation.ADMISSION
            and self.before_digest != _genesis_digest(self.value_id)
        ):
            raise ValueError("admission records must use their deterministic genesis digest")
        if self.after_digest != value_state_digest(self.after_state_projection):
            raise ValueError("after_digest does not match the after-state projection")
        _validate_sha256_digest(self.origin_id, "origin_id")
        refs = _refs(self.evidence_refs, "evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        validate_identifier(self.event_id)
        if type(self.event_sequence) is not int or self.event_sequence < 0:
            raise TypeError("event_sequence must be a nonnegative exact integer")
        _validate_utc_datetime(self.recorded_at)
        if self.previous_record_digest is not None:
            _validate_sha256_digest(self.previous_record_digest, "previous_record_digest")
        object.__setattr__(
            self,
            "record_digest",
            hashlib.sha256(canonical_revision_record_payload(self)).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class ValueRevisionHistory:
    """A bounded retained suffix and its immutable compaction anchor."""

    value_id: str
    history_anchor_revision: int | None = None
    history_anchor_digest: str | None = None
    history_anchor_state_digest: str | None = None
    records: tuple[ValueRevisionRecord, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.value_id)
        anchor_values = (
            self.history_anchor_revision,
            self.history_anchor_digest,
            self.history_anchor_state_digest,
        )
        if any(value is None for value in anchor_values) and not all(
            value is None for value in anchor_values
        ):
            raise ValueError("history anchor fields must be all present or all absent")
        if self.history_anchor_revision is not None:
            if type(self.history_anchor_revision) is not int or self.history_anchor_revision < 0:
                raise TypeError("history_anchor_revision must be a nonnegative exact integer")
            _validate_sha256_digest(self.history_anchor_digest, "history_anchor_digest")
            _validate_sha256_digest(
                self.history_anchor_state_digest, "history_anchor_state_digest"
            )
        if type(self.records) is not tuple:
            raise TypeError("records must be a tuple")
        if len(self.records) > _MAX_REVISION_RECORDS:
            raise ValueError("revision history exceeds the retained record bound")
        if self.history_anchor_revision is not None and not self.records:
            raise ValueError("a compacted history must retain at least one record")

        expected_previous = self.history_anchor_digest
        expected_before = self.history_anchor_state_digest
        expected_from = self.history_anchor_revision
        previous_record: ValueRevisionRecord | None = None
        for index, record in enumerate(self.records):
            if not isinstance(record, ValueRevisionRecord):
                raise TypeError("records must contain ValueRevisionRecord values")
            if record.value_id != self.value_id:
                raise ValueError("revision history cannot mix Value IDs")
            validate_revision_record_digest(record)
            if record.previous_record_digest != expected_previous:
                raise ValueError("revision history has a broken previous-record link")
            if expected_from is not None and record.from_revision != expected_from:
                raise ValueError("first retained revision does not follow its anchor")
            if expected_before is not None and record.before_digest != expected_before:
                raise ValueError("first retained record does not follow its anchor state")
            if previous_record is not None and record.from_revision != previous_record.to_revision:
                raise ValueError("revision history has a discontinuous revision sequence")
            if previous_record is not None and record.before_digest != previous_record.after_digest:
                raise ValueError("revision history has a discontinuous state digest")
            if previous_record is None and self.history_anchor_revision is None:
                if record.operation is ValueRevisionOperation.ADMISSION:
                    if record.before_digest != _genesis_digest(self.value_id):
                        raise ValueError("admission record does not begin at its genesis digest")
                elif record.from_revision != 0:
                    raise ValueError("ordinary history must begin at revision zero")
            expected_previous = record.record_digest
            expected_from = record.to_revision
            expected_before = record.after_digest
            previous_record = record

    def validate(self) -> ValueRevisionHistory:
        """Re-run immutable chain validation and return this history."""

        self.__post_init__()
        return self

    @property
    def last_record(self) -> ValueRevisionRecord | None:
        return self.records[-1] if self.records else None

    def append(self, record: ValueRevisionRecord) -> ValueRevisionHistory:
        if record.value_id != self.value_id:
            raise ValueError("record belongs to a different Value")
        records = self.records + (record,)
        anchor_revision = self.history_anchor_revision
        anchor_digest = self.history_anchor_digest
        anchor_state_digest = self.history_anchor_state_digest
        if len(records) > _MAX_REVISION_RECORDS:
            removed = records[0]
            anchor_revision = removed.to_revision
            anchor_digest = removed.record_digest
            anchor_state_digest = removed.after_digest
            records = records[1:]
        return ValueRevisionHistory(
            value_id=self.value_id,
            history_anchor_revision=anchor_revision,
            history_anchor_digest=anchor_digest,
            history_anchor_state_digest=anchor_state_digest,
            records=records,
        )


@dataclass(frozen=True, slots=True)
class ValueSystemSnapshot:
    """I/O-free, exact domain snapshot used by durable AgentState."""

    schema_version: int
    values: tuple[ValueState, ...]
    conflicts: tuple[ValueConflictDefinition, ...]
    histories: tuple[ValueRevisionHistory, ...]
    evidence_ledgers: tuple[tuple[str, tuple[str, ...]], ...]
    evidence_ledger_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("ValueSystem snapshot schema version must be 1")
        if type(self.values) is not tuple:
            raise TypeError("snapshot values must be a tuple")
        if type(self.conflicts) is not tuple:
            raise TypeError("snapshot conflicts must be a tuple")
        if type(self.histories) is not tuple:
            raise TypeError("snapshot histories must be a tuple")
        if type(self.evidence_ledgers) is not tuple:
            raise TypeError("snapshot evidence_ledgers must be a tuple")
        if type(self.evidence_ledger_digests) is not tuple:
            raise TypeError("snapshot evidence_ledger_digests must be a tuple")
        for value in self.values:
            if not isinstance(value, ValueState):
                raise TypeError("snapshot values must contain ValueState values")
        for conflict in self.conflicts:
            if not isinstance(conflict, ValueConflictDefinition):
                raise TypeError(
                    "snapshot conflicts must contain ValueConflictDefinition values"
                )
        for history in self.histories:
            if not isinstance(history, ValueRevisionHistory):
                raise TypeError(
                    "snapshot histories must contain ValueRevisionHistory values"
                )
        for value_id, ledger in self.evidence_ledgers:
            validate_identifier(value_id)
            _ledger_refs(ledger, "snapshot evidence ledger")
        for value_id, digest in self.evidence_ledger_digests:
            validate_identifier(value_id)
            _validate_sha256_digest(digest, "evidence_ledger_digest")


@dataclass(frozen=True, slots=True)
class ValueMutationResult:
    """Typed outcome of one domain mutation attempt."""

    status: ValueMutationStatus
    value_id: str
    value: ValueState
    applied_delta: float = 0.0
    revision_record: ValueRevisionRecord | None = None

    def __post_init__(self) -> None:
        _enum(self.status, ValueMutationStatus, "status")
        validate_identifier(self.value_id)
        if not isinstance(self.value, ValueState) or self.value.value_id != self.value_id:
            raise TypeError("value must match value_id")
        _signed_fraction(self.applied_delta, "applied_delta")


@dataclass(frozen=True, slots=True)
class ValueEventResult:
    """Deterministic results for one event's canonical target order."""

    results: tuple[ValueMutationResult, ...]

    def __post_init__(self) -> None:
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple")
        ids = tuple(result.value_id for result in self.results)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError("event results must be sorted and unique")

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> ValueMutationResult:
        return self.results[index]

    @property
    def total_applied_delta(self) -> float:
        return math.fsum(result.applied_delta for result in self.results)


def _project_evidence_refs(
    current: tuple[str, ...], incoming: tuple[str, ...]
) -> tuple[str, ...]:
    """Retain every new ref and fill the bounded projection deterministically."""

    combined = tuple(sorted(set(current) | set(incoming)))
    if len(combined) <= _MAX_REFS:
        return combined
    incoming_set = set(incoming)
    retained = list(incoming)
    retained.extend(ref for ref in combined if ref not in incoming_set)
    return tuple(sorted(retained[:_MAX_REFS]))


def _validate_governance_origin(
    origin: IdentityOrigin,
    evidence: ValueMutationEvidence,
    expected_source: str,
) -> None:
    if not isinstance(origin, IdentityOrigin):
        raise TypeError("governance_origin must be an IdentityOrigin")
    if (
        origin.actor is not OriginActor.OPERATOR
        or origin.input_kind is not OriginInputKind.CONSTRAINT
        or origin.admission is not ValueAdmissionStatus.PENDING
    ):
        raise ValueDomainError("governance origin is not an operator constraint")
    if origin.source_ref != expected_source:
        raise ValueDomainError("governance origin source is not allowlisted")
    if origin.event_id != evidence.event_id or origin.event_sequence != evidence.event_sequence:
        raise ValueDomainError("governance origin is not bound to event evidence")


class ValueSystem:
    """Process-local authority for bounded Value mutation and revision history."""

    EVENT_UPDATE_BUDGET: Final[float] = _EVENT_UPDATE_BUDGET
    MAX_AUTHORITATIVE_VALUES: Final[int] = _MAX_AUTHORITATIVE_VALUES

    def __init__(
        self,
        values: Iterable[ValueState] | Mapping[str, ValueState] = (),
        conflicts: Iterable[ValueConflictDefinition] = (),
    ) -> None:
        raw_values = values.values() if isinstance(values, Mapping) else values
        value_items = tuple(raw_values)
        value_map: dict[str, ValueState] = {}
        for value in value_items:
            if not isinstance(value, ValueState):
                raise TypeError("values must contain ValueState values")
            if value.revision != 0:
                raise ValueDomainError("initial Values must begin at revision zero")
            if value.value_id in value_map:
                raise ValueDomainError("Value IDs must be unique")
            value_map[value.value_id] = value
        if len(value_map) > _MAX_AUTHORITATIVE_VALUES:
            raise ValueDomainError("ValueSystem exceeds the authoritative Value bound")

        conflict_items = tuple(conflicts)
        conflict_map: dict[tuple[str, str], ValueConflictDefinition] = {}
        for conflict in conflict_items:
            if not isinstance(conflict, ValueConflictDefinition):
                raise TypeError("conflicts must contain ValueConflictDefinition values")
            if conflict.left_value_id not in value_map or conflict.right_value_id not in value_map:
                raise ValueDomainError("conflicts must reference known Values")
            pair = (conflict.left_value_id, conflict.right_value_id)
            if pair in conflict_map:
                raise ValueDomainError("conflict pairs must be unique")
            conflict_map[pair] = conflict

        self._values = {value_id: value_map[value_id] for value_id in sorted(value_map)}
        self._conflicts = tuple(conflict_map[key] for key in sorted(conflict_map))
        self._histories = {
            value_id: ValueRevisionHistory(value_id)
            for value_id in sorted(self._values)
        }
        self._evidence_ledgers = {
            value_id: tuple(value.evidence_refs)
            for value_id, value in self._values.items()
        }
        self._evidence_ledger_digests = {
            value_id: evidence_ledger_digest(value_id, ledger)
            for value_id, ledger in self._evidence_ledgers.items()
        }
        self.validate()

    @classmethod
    def from_seed_declarations(
        cls,
        seeds: Iterable[ValueSeedDeclaration],
        conflicts: Iterable[ValueConflictDefinition] = (),
    ) -> ValueSystem:
        """Bootstrap authoritative Values from immutable configuration seeds."""

        seed_items = tuple(seeds)
        values: list[ValueState] = []
        for seed in seed_items:
            if not isinstance(seed, ValueSeedDeclaration):
                raise TypeError("seeds must contain ValueSeedDeclaration values")
            values.append(
                ValueState(
                    value_id=seed.value_id,
                    revision=0,
                    name=seed.name,
                    concept=seed.concept,
                    scope=seed.scope,
                    context_ids=seed.context_ids,
                    polarity=seed.polarity,
                    strength=seed.initial_strength,
                    confidence=seed.confidence,
                    stability=seed.stability,
                    protectedness=seed.protectedness,
                    negotiability=seed.negotiability,
                    allowed_update_rate=seed.allowed_update_rate,
                    frozen=False,
                    origin=IdentityOrigin(
                        OriginActor.SYSTEM,
                        OriginInputKind.CONFIG_SEED,
                        ValueAdmissionStatus.SYSTEM_AUTHORIZED,
                        source_ref=f"config-seed:{seed.value_id}",
                    ),
                    evidence_refs=(),
                    seed_contract_digest=recompute_seed_contract_digest(seed),
                    opposition_count=0,
                )
            )
        return cls(values, conflicts)

    @classmethod
    def restore(
        cls,
        *,
        values: Mapping[str, ValueState],
        conflicts: Iterable[ValueConflictDefinition] = (),
        histories: Mapping[str, ValueRevisionHistory],
        evidence_ledgers: Mapping[str, tuple[str, ...]],
        evidence_ledger_digests: Mapping[str, str] | None = None,
    ) -> ValueSystem:
        """Restore a complete domain snapshot without I/O or replay."""

        if not isinstance(values, Mapping):
            raise TypeError("restore values must be a mapping")
        if not isinstance(histories, Mapping):
            raise TypeError("restore histories must be a mapping")
        if not isinstance(evidence_ledgers, Mapping):
            raise TypeError("restore evidence_ledgers must be a mapping")

        value_map: dict[str, ValueState] = {}
        for key, value in values.items():
            validate_identifier(key)
            if not isinstance(value, ValueState):
                raise TypeError("restore values must contain ValueState values")
            if key != value.value_id:
                raise ValueDomainError("restore Value keys must match Value IDs")
            value_map[key] = value
        if len(value_map) > _MAX_AUTHORITATIVE_VALUES:
            raise ValueDomainError("ValueSystem exceeds the authoritative Value bound")

        if set(value_map) != set(histories) or set(value_map) != set(evidence_ledgers):
            raise ValueDomainError("restore Value, history, and ledger keys must match exactly")

        history_map: dict[str, ValueRevisionHistory] = {}
        for key, history in histories.items():
            validate_identifier(key)
            if not isinstance(history, ValueRevisionHistory):
                raise TypeError("restore histories must contain ValueRevisionHistory values")
            if key != history.value_id:
                raise ValueDomainError("restore history keys must match Value IDs")
            history.validate()
            history_map[key] = history

        ledger_map: dict[str, tuple[str, ...]] = {}
        for key, ledger in evidence_ledgers.items():
            validate_identifier(key)
            try:
                ledger_map[key] = _ledger_refs(ledger, "restore evidence ledger")
            except (TypeError, ValueError) as exc:
                raise ValueDomainError("restore evidence ledger is invalid") from exc

        if evidence_ledger_digests is None:
            digest_map = {
                value_id: evidence_ledger_digest(value_id, ledger)
                for value_id, ledger in ledger_map.items()
            }
        else:
            if not isinstance(evidence_ledger_digests, Mapping):
                raise TypeError("restore evidence_ledger_digests must be a mapping")
            if set(value_map) != set(evidence_ledger_digests):
                raise ValueDomainError(
                    "restore Value and ledger-digest keys must match exactly"
                )
            digest_map = {}
            for key, digest in evidence_ledger_digests.items():
                validate_identifier(key)
                try:
                    checked_digest = _validate_sha256_digest(
                        digest, "evidence_ledger_digest"
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueDomainError("restore evidence ledger digest is invalid") from exc
                expected_digest = evidence_ledger_digest(key, ledger_map[key])
                if checked_digest != expected_digest:
                    raise ValueDomainError(
                        "restore evidence ledger digest does not match its ledger"
                    )
                digest_map[key] = checked_digest

        conflict_items = tuple(conflicts)
        conflict_map: dict[tuple[str, str], ValueConflictDefinition] = {}
        for conflict in conflict_items:
            if not isinstance(conflict, ValueConflictDefinition):
                raise TypeError("conflicts must contain ValueConflictDefinition values")
            if conflict.left_value_id not in value_map or conflict.right_value_id not in value_map:
                raise ValueDomainError("conflicts must reference stored Values")
            pair = (conflict.left_value_id, conflict.right_value_id)
            if pair in conflict_map:
                raise ValueDomainError("conflict pairs must be unique")
            conflict_map[pair] = conflict

        system = cls.__new__(cls)
        system._values = {value_id: value_map[value_id] for value_id in sorted(value_map)}
        system._conflicts = tuple(conflict_map[key] for key in sorted(conflict_map))
        system._histories = {
            value_id: history_map[value_id] for value_id in sorted(history_map)
        }
        system._evidence_ledgers = {
            value_id: ledger_map[value_id] for value_id in sorted(ledger_map)
        }
        system._evidence_ledger_digests = {
            value_id: digest_map[value_id] for value_id in sorted(digest_map)
        }
        system.validate()
        return system

    def snapshot(self) -> ValueSystemSnapshot:
        """Return the complete domain state without replay or I/O."""

        return ValueSystemSnapshot(
            schema_version=1,
            values=self.values,
            conflicts=self.conflicts,
            histories=tuple(
                self._histories[value_id] for value_id in sorted(self._histories)
            ),
            evidence_ledgers=tuple(
                (value_id, self._evidence_ledgers[value_id])
                for value_id in sorted(self._evidence_ledgers)
            ),
            evidence_ledger_digests=tuple(
                (value_id, self._evidence_ledger_digests[value_id])
                for value_id in sorted(self._evidence_ledger_digests)
            ),
        )

    @classmethod
    def restore_snapshot(cls, snapshot: ValueSystemSnapshot) -> ValueSystem:
        """Restore an exact domain snapshot through the normal validator."""

        if not isinstance(snapshot, ValueSystemSnapshot):
            raise TypeError("snapshot must be a ValueSystemSnapshot")
        values = {value.value_id: value for value in snapshot.values}
        histories = {history.value_id: history for history in snapshot.histories}
        evidence_ledgers = dict(snapshot.evidence_ledgers)
        evidence_ledger_digests = dict(snapshot.evidence_ledger_digests)
        if len(values) != len(snapshot.values) or len(histories) != len(snapshot.histories):
            raise ValueDomainError("snapshot contains duplicate Value IDs")
        if len(evidence_ledgers) != len(snapshot.evidence_ledgers):
            raise ValueDomainError("snapshot contains duplicate evidence-ledger IDs")
        if len(evidence_ledger_digests) != len(snapshot.evidence_ledger_digests):
            raise ValueDomainError("snapshot contains duplicate ledger-digest IDs")
        return cls.restore(
            values=values,
            conflicts=snapshot.conflicts,
            histories=histories,
            evidence_ledgers=evidence_ledgers,
            evidence_ledger_digests=evidence_ledger_digests,
        )

    @property
    def values(self) -> tuple[ValueState, ...]:
        return tuple(self._values[value_id] for value_id in sorted(self._values))

    @property
    def value_map(self) -> Mapping[str, ValueState]:
        return MappingProxyType({value.value_id: value for value in self.values})

    @property
    def conflicts(self) -> tuple[ValueConflictDefinition, ...]:
        return self._conflicts

    @property
    def histories(self) -> Mapping[str, ValueRevisionHistory]:
        return MappingProxyType(
            {value_id: self._histories[value_id] for value_id in sorted(self._histories)}
        )

    @property
    def evidence_ledgers(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                value_id: self._evidence_ledgers[value_id]
                for value_id in sorted(self._evidence_ledgers)
            }
        )

    @property
    def evidence_ledger_digests(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                value_id: self._evidence_ledger_digests[value_id]
                for value_id in sorted(self._evidence_ledger_digests)
            }
        )

    def get(self, value_id: str) -> ValueState:
        validate_identifier(value_id)
        try:
            return self._values[value_id]
        except KeyError as exc:
            raise ValueNotFound("unknown Value ID") from exc

    def history(self, value_id: str) -> ValueRevisionHistory:
        validate_identifier(value_id)
        try:
            return self._histories[value_id]
        except KeyError as exc:
            raise ValueNotFound("unknown Value ID") from exc

    def revision_history(self, value_id: str) -> ValueRevisionHistory:
        return self.history(value_id)

    def applied_evidence(self, value_id: str) -> tuple[str, ...]:
        validate_identifier(value_id)
        try:
            return self._evidence_ledgers[value_id]
        except KeyError as exc:
            raise ValueNotFound("unknown Value ID") from exc

    @staticmethod
    def _validate_event(evidence: ValueMutationEvidence) -> None:
        if not isinstance(evidence, ValueMutationEvidence):
            raise TypeError("evidence must be a ValueMutationEvidence")

    @staticmethod
    def _validate_binding(
        admission: ValueSelfAdmission, evidence: ValueMutationEvidence
    ) -> None:
        if (
            admission.subject_origin.event_id != evidence.event_id
            or admission.subject_origin.event_sequence != evidence.event_sequence
        ):
            raise ValueDomainError("self-admission origin does not match event evidence")

    def _require_active(self, value_id: str) -> ValueState:
        value = self.get(value_id)
        if not value.is_active():
            raise ValueDomainError("Value is not active")
        return value

    def _record(
        self,
        before: ValueState,
        after: ValueState,
        operation: ValueRevisionOperation,
        origin_id: str,
        evidence_refs: tuple[str, ...],
        evidence: ValueMutationEvidence,
        target_revision: int | None = None,
    ) -> ValueRevisionRecord:
        history = self._histories[before.value_id]
        previous = history.last_record
        return ValueRevisionRecord(
            value_id=before.value_id,
            from_revision=before.revision,
            to_revision=after.revision,
            before_digest=value_state_digest(before),
            after_state_projection=after,
            after_digest=value_state_digest(after),
            operation=operation,
            origin_id=origin_id,
            evidence_refs=evidence_refs,
            event_id=evidence.event_id,
            event_sequence=evidence.event_sequence,
            recorded_at=evidence.recorded_at,
            previous_record_digest=(
                previous.record_digest if previous is not None else history.history_anchor_digest
            ),
            target_revision=target_revision,
        )

    def _commit_record(
        self, value: ValueState, record: ValueRevisionRecord
    ) -> None:
        history = self._histories[value.value_id].append(record)
        self._values[value.value_id] = value
        self._histories[value.value_id] = history

    def admit_self_value(
        self,
        candidate: ValueState,
        admission: ValueSelfAdmission,
        evidence: ValueMutationEvidence,
    ) -> ValueMutationResult:
        self._validate_event(evidence)
        if not isinstance(candidate, ValueState):
            raise TypeError("candidate must be a ValueState")
        if not isinstance(admission, ValueSelfAdmission):
            raise TypeError("admission must be a ValueSelfAdmission")
        if admission.reason is not ValueMutationReason.SELF_ADMISSION:
            raise ValueDomainError("new Value admission requires the self-admission reason")
        self._validate_binding(admission, evidence)
        if candidate.value_id != admission.target_value_id:
            raise ValueDomainError("candidate Value ID does not match the admission")
        if candidate.revision != 0:
            raise ValueDomainError("new self-admitted Values must begin at revision zero")
        if candidate.origin != admission.subject_origin:
            raise ValueDomainError("candidate origin must equal the self-admission origin")
        if candidate.seed_contract_digest is not None:
            raise ValueDomainError("self-admitted Values cannot carry a seed digest")
        if candidate.evidence_refs != admission.evidence_refs:
            raise ValueDomainError("candidate evidence refs must equal the admission refs")
        if candidate.frozen or candidate.opposition_count != 0:
            raise ValueDomainError("new self-admitted Values must start unfrozen and unopposed")
        existing = self._values.get(candidate.value_id)
        if existing is not None:
            if (
                existing == candidate
                and set(admission.evidence_refs)
                <= set(self._evidence_ledgers[candidate.value_id])
            ):
                return ValueMutationResult(
                    status=ValueMutationStatus.IDEMPOTENT,
                    value_id=candidate.value_id,
                    value=existing,
                )
            raise ValueDomainError("Value ID is already admitted")
        if len(self._values) >= _MAX_AUTHORITATIVE_VALUES:
            raise ValueDomainError("ValueSystem has reached its authoritative Value bound")

        record = ValueRevisionRecord(
            value_id=candidate.value_id,
            from_revision=-1,
            to_revision=0,
            before_digest=_genesis_digest(candidate.value_id),
            after_state_projection=candidate,
            after_digest=value_state_digest(candidate),
            operation=ValueRevisionOperation.ADMISSION,
            origin_id=admission.subject_origin.origin_id,
            evidence_refs=admission.evidence_refs,
            event_id=evidence.event_id,
            event_sequence=evidence.event_sequence,
            recorded_at=evidence.recorded_at,
        )
        self._values[candidate.value_id] = candidate
        self._histories[candidate.value_id] = ValueRevisionHistory(
            candidate.value_id, records=(record,)
        )
        self._evidence_ledgers[candidate.value_id] = candidate.evidence_refs
        self._evidence_ledger_digests[candidate.value_id] = evidence_ledger_digest(
            candidate.value_id, candidate.evidence_refs
        )
        return ValueMutationResult(
            status=ValueMutationStatus.APPLIED,
            value_id=candidate.value_id,
            value=candidate,
            revision_record=record,
        )

    def admit_value(
        self,
        candidate: ValueState,
        admission: ValueSelfAdmission,
        evidence: ValueMutationEvidence,
    ) -> ValueMutationResult:
        return self.admit_self_value(candidate, admission, evidence)

    def admit(
        self,
        candidate: ValueState,
        admission: ValueSelfAdmission,
        evidence: ValueMutationEvidence,
    ) -> ValueMutationResult:
        return self.admit_self_value(candidate, admission, evidence)

    @staticmethod
    def _candidate_delta(value: ValueState, admission: ValueSelfAdmission) -> float:
        raw_signal = admission.requested_delta * admission.confidence
        per_value_cap = (
            value.allowed_update_rate
            * (1.0 - value.stability)
            * (1.0 - value.protectedness)
            * value.negotiability
        )
        return max(-per_value_cap, min(raw_signal, per_value_cap))

    @staticmethod
    def _updated_state(
        value: ValueState, admission: ValueSelfAdmission, applied_delta: float
    ) -> ValueState:
        magnitude = abs(applied_delta)
        supporting = admission.requested_delta >= 0.0
        if supporting:
            strength = min(1.0, value.strength + magnitude)
            opposition_count = 0
            polarity = value.polarity
        else:
            threshold = 3 + math.ceil(3.0 * value.protectedness)
            if value.strength == 0.0 and value.opposition_count >= threshold:
                polarity = -value.polarity
                strength = magnitude
                opposition_count = 0
            else:
                polarity = value.polarity
                strength = max(0.0, value.strength - magnitude)
                opposition_count = min(
                    _MAX_OPPOSITION_COUNT, value.opposition_count + 1
                )
        confidence_step = min(0.05, magnitude) * admission.confidence
        confidence = (
            min(1.0, value.confidence + confidence_step)
            if supporting
            else max(0.0, value.confidence - confidence_step)
        )
        return replace(
            value,
            revision=value.revision + 1,
            polarity=polarity,
            strength=strength,
            confidence=confidence,
            opposition_count=opposition_count,
            evidence_refs=_project_evidence_refs(
                value.evidence_refs, admission.evidence_refs
            ),
        )

    def apply_updates(
        self,
        evidence: ValueMutationEvidence,
        admissions: Iterable[ValueSelfAdmission],
    ) -> ValueEventResult:
        self._validate_event(evidence)
        try:
            intents = tuple(admissions)
        except TypeError as exc:
            raise TypeError("admissions must be iterable") from exc
        if any(not isinstance(intent, ValueSelfAdmission) for intent in intents):
            raise TypeError("admissions must contain ValueSelfAdmission values")
        if not intents:
            return ValueEventResult(())

        target_ids = tuple(intent.target_value_id for intent in intents)
        if len(set(target_ids)) != len(target_ids):
            raise ValueDomainError("an event cannot contain duplicate target Value IDs")
        ordered = tuple(sorted(intents, key=lambda intent: intent.target_value_id))
        for intent in ordered:
            if intent.reason is not ValueMutationReason.ADMITTED_UPDATE:
                raise ValueDomainError("existing Value updates require the admitted-update reason")
            self._validate_binding(intent, evidence)
            self._require_active(intent.target_value_id)

        if any(
            set(intent.evidence_refs) & set(self._evidence_ledgers[intent.target_value_id])
            for intent in ordered
        ):
            return ValueEventResult(
                tuple(
                    ValueMutationResult(
                        status=ValueMutationStatus.IDEMPOTENT,
                        value_id=intent.target_value_id,
                        value=self._values[intent.target_value_id],
                    )
                    for intent in ordered
                )
            )

        plans: list[
            tuple[
                ValueSelfAdmission,
                ValueState,
                ValueState | None,
                float,
                tuple[str, ...] | None,
            ]
        ] = []
        remaining_budget = _EVENT_UPDATE_BUDGET
        applied_magnitudes: list[float] = []
        for intent in ordered:
            current = self._values[intent.target_value_id]
            if current.frozen:
                plans.append((intent, current, None, 0.0, None))
                continue
            candidate_delta = self._candidate_delta(current, intent)
            remaining_budget = max(0.0, _EVENT_UPDATE_BUDGET - math.fsum(applied_magnitudes))
            applied_magnitude = min(abs(candidate_delta), remaining_budget)
            applied_delta = math.copysign(applied_magnitude, candidate_delta)
            if applied_magnitude == 0.0:
                new_ledger = tuple(
                    sorted(set(self._evidence_ledgers[current.value_id]) | set(intent.evidence_refs))
                )
                if len(new_ledger) > _MAX_APPLIED_EVIDENCE_REFS:
                    raise ValueDomainError("applied evidence ledger would exceed its bound")
                plans.append((intent, current, current, 0.0, new_ledger))
                continue
            computed_state = self._updated_state(current, intent, applied_delta)
            new_ledger = tuple(
                sorted(set(self._evidence_ledgers[current.value_id]) | set(intent.evidence_refs))
            )
            if len(new_ledger) > _MAX_APPLIED_EVIDENCE_REFS:
                raise ValueDomainError("applied evidence ledger would exceed its bound")
            plans.append((intent, current, computed_state, applied_delta, new_ledger))
            applied_magnitudes.append(applied_magnitude)

        results: list[ValueMutationResult] = []
        for intent, current, planned_state, applied_delta, planned_ledger in plans:
            if planned_state is None:
                results.append(
                    ValueMutationResult(
                        status=ValueMutationStatus.FROZEN,
                        value_id=current.value_id,
                        value=current,
                    )
                )
                continue
            if applied_delta == 0.0:
                self._evidence_ledgers[current.value_id] = planned_ledger or ()
                self._evidence_ledger_digests[current.value_id] = evidence_ledger_digest(
                    current.value_id, planned_ledger or ()
                )
                results.append(
                    ValueMutationResult(
                        status=ValueMutationStatus.NO_CHANGE,
                        value_id=current.value_id,
                        value=current,
                    )
                )
                continue
            assert planned_ledger is not None
            record = self._record(
                current,
                planned_state,
                ValueRevisionOperation.UPDATE,
                intent.subject_origin.origin_id,
                intent.evidence_refs,
                evidence,
            )
            self._commit_record(planned_state, record)
            self._evidence_ledgers[current.value_id] = planned_ledger
            self._evidence_ledger_digests[current.value_id] = evidence_ledger_digest(
                current.value_id, planned_ledger
            )
            results.append(
                ValueMutationResult(
                    status=ValueMutationStatus.APPLIED,
                    value_id=current.value_id,
                    value=planned_state,
                    applied_delta=applied_delta,
                    revision_record=record,
                )
            )
        return ValueEventResult(tuple(results))

    def apply_update(
        self,
        admission: ValueSelfAdmission,
        evidence: ValueMutationEvidence,
    ) -> ValueMutationResult:
        return self.apply_updates(evidence, (admission,))[0]

    def update(
        self,
        admission: ValueSelfAdmission,
        evidence: ValueMutationEvidence,
    ) -> ValueMutationResult:
        return self.apply_update(admission, evidence)

    def _set_frozen(
        self,
        value_id: str,
        frozen: bool,
        evidence: ValueMutationEvidence,
        governance_origin: IdentityOrigin,
    ) -> ValueMutationResult:
        self._validate_event(evidence)
        _validate_governance_origin(
            governance_origin,
            evidence,
            "api.values.freeze" if frozen else "api.values.unfreeze",
        )
        value = self.get(value_id)
        if value.frozen is frozen:
            return ValueMutationResult(
                status=ValueMutationStatus.IDEMPOTENT,
                value_id=value_id,
                value=value,
            )
        updated = replace(value, revision=value.revision + 1, frozen=frozen)
        operation = (
            ValueRevisionOperation.FREEZE
            if frozen
            else ValueRevisionOperation.UNFREEZE
        )
        record = self._record(
            value,
            updated,
            operation,
            governance_origin.origin_id,
            (),
            evidence,
        )
        self._commit_record(updated, record)
        return ValueMutationResult(
            status=ValueMutationStatus.APPLIED,
            value_id=value_id,
            value=updated,
            revision_record=record,
        )

    def freeze(
        self,
        value_id: str,
        evidence: ValueMutationEvidence,
        *,
        governance_origin: IdentityOrigin,
    ) -> ValueMutationResult:
        return self._set_frozen(value_id, True, evidence, governance_origin)

    def unfreeze(
        self,
        value_id: str,
        evidence: ValueMutationEvidence,
        *,
        governance_origin: IdentityOrigin,
    ) -> ValueMutationResult:
        return self._set_frozen(value_id, False, evidence, governance_origin)

    def rollback(
        self,
        value_id: str,
        target_revision: int,
        evidence: ValueMutationEvidence,
        *,
        governance_origin: IdentityOrigin,
    ) -> ValueMutationResult:
        self._validate_event(evidence)
        _validate_governance_origin(
            governance_origin, evidence, "api.values.rollback"
        )
        if type(target_revision) is not int or target_revision < 0:
            raise TypeError("target_revision must be a nonnegative exact integer")
        current = self.get(value_id)
        history = self._histories[value_id]
        target_record = next(
            (
                record
                for record in history.records
                if record.to_revision == target_revision
            ),
            None,
        )
        if target_record is None:
            raise ValueDomainError("rollback target is not in retained revision history")
        target = target_record.after_state_projection
        restored = replace(
            current,
            revision=current.revision + 1,
            polarity=target.polarity,
            strength=target.strength,
            confidence=target.confidence,
            frozen=target.frozen,
            opposition_count=target.opposition_count,
            evidence_refs=target.evidence_refs,
        )
        record = self._record(
            current,
            restored,
            ValueRevisionOperation.ROLLBACK,
            governance_origin.origin_id,
            target.evidence_refs,
            evidence,
            target_revision=target_revision,
        )
        self._commit_record(restored, record)
        return ValueMutationResult(
            status=ValueMutationStatus.APPLIED,
            value_id=value_id,
            value=restored,
            revision_record=record,
        )

    def adopt_seed(
        self,
        seed: ValueSeedDeclaration,
        evidence: ValueMutationEvidence,
        *,
        governance_origin: IdentityOrigin,
    ) -> ValueMutationResult:
        self._validate_event(evidence)
        if not isinstance(seed, ValueSeedDeclaration):
            raise TypeError("seed must be a ValueSeedDeclaration")
        _validate_governance_origin(
            governance_origin, evidence, "api.values.seed_adopt"
        )
        seed_digest = recompute_seed_contract_digest(seed)
        existing = self._values.get(seed.value_id)
        if existing is not None:
            if (
                existing.origin.actor is OriginActor.SYSTEM
                and existing.origin.input_kind is OriginInputKind.CONFIG_SEED
                and existing.origin.admission is ValueAdmissionStatus.SYSTEM_AUTHORIZED
                and existing.seed_contract_digest == seed_digest
            ):
                return ValueMutationResult(
                    status=ValueMutationStatus.IDEMPOTENT,
                    value_id=seed.value_id,
                    value=existing,
                )
            raise ValueDomainError("Value ID collides with existing state")
        if len(self._values) >= _MAX_AUTHORITATIVE_VALUES:
            raise ValueDomainError("ValueSystem has reached its authoritative Value bound")

        candidate = ValueState(
            value_id=seed.value_id,
            revision=0,
            name=seed.name,
            concept=seed.concept,
            scope=seed.scope,
            context_ids=seed.context_ids,
            polarity=seed.polarity,
            strength=seed.initial_strength,
            confidence=seed.confidence,
            stability=seed.stability,
            protectedness=seed.protectedness,
            negotiability=seed.negotiability,
            allowed_update_rate=seed.allowed_update_rate,
            frozen=False,
            origin=IdentityOrigin(
                OriginActor.SYSTEM,
                OriginInputKind.CONFIG_SEED,
                ValueAdmissionStatus.SYSTEM_AUTHORIZED,
                source_ref=f"config-seed:{seed.value_id}",
            ),
            evidence_refs=(),
            seed_contract_digest=seed_digest,
            opposition_count=0,
        )
        record = ValueRevisionRecord(
            value_id=candidate.value_id,
            from_revision=-1,
            to_revision=0,
            before_digest=_genesis_digest(candidate.value_id),
            after_state_projection=candidate,
            after_digest=value_state_digest(candidate),
            operation=ValueRevisionOperation.ADMISSION,
            origin_id=governance_origin.origin_id,
            evidence_refs=(),
            event_id=evidence.event_id,
            event_sequence=evidence.event_sequence,
            recorded_at=evidence.recorded_at,
        )
        self._values[candidate.value_id] = candidate
        self._histories[candidate.value_id] = ValueRevisionHistory(
            candidate.value_id, records=(record,)
        )
        self._evidence_ledgers[candidate.value_id] = ()
        self._evidence_ledger_digests[candidate.value_id] = evidence_ledger_digest(
            candidate.value_id, ()
        )
        self._values = {value_id: self._values[value_id] for value_id in sorted(self._values)}
        self._histories = {
            value_id: self._histories[value_id] for value_id in sorted(self._histories)
        }
        self._evidence_ledgers = {
            value_id: self._evidence_ledgers[value_id]
            for value_id in sorted(self._evidence_ledgers)
        }
        self._evidence_ledger_digests = {
            value_id: self._evidence_ledger_digests[value_id]
            for value_id in sorted(self._evidence_ledger_digests)
        }
        return ValueMutationResult(
            status=ValueMutationStatus.APPLIED,
            value_id=candidate.value_id,
            value=candidate,
            revision_record=record,
        )

    def review_origin(
        self,
        value_id: str,
        decision: ValueOriginReviewDecision,
        evidence: ValueMutationEvidence,
        *,
        governance_origin: IdentityOrigin,
    ) -> ValueMutationResult:
        self._validate_event(evidence)
        _enum(decision, ValueOriginReviewDecision, "decision")
        _validate_governance_origin(
            governance_origin, evidence, "api.values.origin_review"
        )
        value = self.get(value_id)
        if value.origin.actor not in {OriginActor.INHERITED, OriginActor.UNKNOWN}:
            raise ValueDomainError("only inherited or unknown Values may be reviewed")

        current_admission = value.origin.admission
        if decision is ValueOriginReviewDecision.ACCEPT_PROVENANCE:
            if current_admission is ValueAdmissionStatus.PENDING:
                return ValueMutationResult(
                    status=ValueMutationStatus.IDEMPOTENT,
                    value_id=value_id,
                    value=value,
                )
            if current_admission is not ValueAdmissionStatus.UNCERTAIN:
                raise ValueDomainError("Value origin review transition is invalid")
            admission = ValueAdmissionStatus.PENDING
        else:
            if current_admission is ValueAdmissionStatus.REJECTED:
                return ValueMutationResult(
                    status=ValueMutationStatus.IDEMPOTENT,
                    value_id=value_id,
                    value=value,
                )
            if current_admission not in {
                ValueAdmissionStatus.UNCERTAIN,
                ValueAdmissionStatus.PENDING,
            }:
                raise ValueDomainError("Value origin review transition is invalid")
            admission = ValueAdmissionStatus.REJECTED

        reviewed_origin = IdentityOrigin(
            value.origin.actor,
            value.origin.input_kind,
            admission,
            source_ref=value.origin.source_ref,
            event_id=value.origin.event_id,
            context_id=value.origin.context_id,
            event_sequence=value.origin.event_sequence,
            confidence=value.origin.confidence,
        )
        if reviewed_origin.origin_id != value.origin.origin_id:
            raise ValueDomainError("origin review changed the provenance identity")
        updated = replace(value, revision=value.revision + 1, origin=reviewed_origin)
        record = self._record(
            value,
            updated,
            ValueRevisionOperation.ORIGIN_REVIEW,
            governance_origin.origin_id,
            (),
            evidence,
        )
        self._commit_record(updated, record)
        return ValueMutationResult(
            status=ValueMutationStatus.APPLIED,
            value_id=value_id,
            value=updated,
            revision_record=record,
        )

    def validate(self) -> None:
        """Validate current state, retained chains, and exact ledger witnesses."""

        value_ids = set(self._values)
        if len(value_ids) > _MAX_AUTHORITATIVE_VALUES:
            raise ValueDomainError("ValueSystem exceeds the authoritative Value bound")
        if value_ids != set(self._histories) or value_ids != set(self._evidence_ledgers):
            raise ValueDomainError("Value, history, and evidence-ledger keys must match")
        if value_ids != set(self._evidence_ledger_digests):
            raise ValueDomainError("Value and evidence-ledger digest keys must match")
        if tuple(self._values) != tuple(sorted(self._values)):
            raise ValueDomainError("Value mapping must be canonically sorted")
        if tuple(self._histories) != tuple(sorted(self._histories)):
            raise ValueDomainError("history mapping must be canonically sorted")
        if tuple(self._evidence_ledgers) != tuple(sorted(self._evidence_ledgers)):
            raise ValueDomainError("evidence-ledger mapping must be canonically sorted")

        for conflict in self._conflicts:
            if (
                conflict.left_value_id not in value_ids
                or conflict.right_value_id not in value_ids
            ):
                raise ValueDomainError("conflicts must reference stored Values")

        for value_id, value in self._values.items():
            if value_id != value.value_id:
                raise ValueDomainError("Value mapping key must match Value ID")
            history = self._histories[value_id]
            history.validate()
            ledger = self._evidence_ledgers[value_id]
            if len(ledger) > _MAX_APPLIED_EVIDENCE_REFS:
                raise ValueDomainError("applied evidence ledger exceeds its bound")
            if ledger != tuple(sorted(set(ledger))):
                raise ValueDomainError("applied evidence ledger is not canonical")
            for evidence_ref in ledger:
                validate_identifier(evidence_ref)
            expected_ledger_digest = evidence_ledger_digest(value_id, ledger)
            if self._evidence_ledger_digests[value_id] != expected_ledger_digest:
                raise ValueDomainError("evidence ledger digest does not match its ledger")
            if not set(value.evidence_refs) <= set(ledger):
                raise ValueDomainError("current Value evidence is missing from its ledger")

            if history.records:
                last = history.last_record
                assert last is not None
                if value.revision != last.to_revision:
                    raise ValueDomainError("current revision does not match retained history")
                if value != last.after_state_projection:
                    raise ValueDomainError(
                        "current Value does not match the last retained state projection"
                    )
                if value_state_digest(value) != last.after_digest:
                    raise ValueDomainError("current Value digest does not match retained history")
                for record in history.records:
                    if not set(record.evidence_refs) <= set(ledger):
                        raise ValueDomainError(
                            "retained revision evidence is missing from its ledger"
                        )
            elif value.revision != 0:
                raise ValueDomainError("a Value with no history must be at revision zero")
