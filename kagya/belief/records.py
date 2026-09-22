"""Pure R12 Belief proposition and authority-boundary contracts.

These immutable values describe bounded current subject-state evidence.  They
do not adopt claims, consult Memory, invoke a model, or provide a mutable
Belief authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import ClassVar, Final

from kagya.identifiers import validate_identifier


class _ClosedStrEnum(str, Enum):
    pass


class BeliefLifecycle(_ClosedStrEnum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"


class BeliefEpistemicStatus(_ClosedStrEnum):
    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"
    PROBABLE = "probable"
    ESTABLISHED = "established"


class BeliefEvidenceType(_ClosedStrEnum):
    EXPERIENCE = "experience"
    SEMANTIC_MEMORY = "semantic_memory"
    EPISODIC_MEMORY = "episodic_memory"
    EXTERNAL_CLAIM = "external_claim"
    MODEL_INFERENCE = "model_inference"
    OPERATOR_CORRECTION = "operator_correction"


class BeliefSubjectAdmissionReason(_ClosedStrEnum):
    SUBJECT_ENDORSEMENT = "subject_endorsement"
    SUBJECT_CORRECTION = "subject_correction"
    SUBJECT_REVIEW = "subject_review"


class BeliefRevisionOperation(_ClosedStrEnum):
    CREATE = "create"
    ADOPT = "adopt"
    CORRECT = "correct"
    SUPERSEDE = "supersede"
    RETRACT = "retract"
    EXPIRE = "expire"


class BeliefRevisionReason(_ClosedStrEnum):
    CREATION = "creation"
    SUBJECT_ADMISSION = "subject_admission"
    CORRECTION = "correction"
    SUPERSESSION = "supersession"
    RETRACTION = "retraction"
    EXPIRATION = "expiration"


# Short names are retained for ergonomic imports without widening vocabularies.
EpistemicStatus = BeliefEpistemicStatus
EvidenceType = BeliefEvidenceType
AdmissionReason = BeliefSubjectAdmissionReason


BELIEF_SCHEMA_VERSION: Final = 1
BELIEF_MAX_CONTEXTS: Final = 16
BELIEF_MAX_EVIDENCE: Final = 32
BELIEF_MAX_REVISIONS: Final = 32
BELIEF_MAX_PROPOSITION_CODEPOINTS: Final = 2_000
BELIEF_MAX_COMPONENT_CODEPOINTS: Final = 256
BELIEF_MAX_REVISION: Final = 2**31 - 1
BELIEF_MAX_EVENT_SEQUENCE: Final = 2**63 - 1

# Short constant names are compatibility spellings, not separate bounds.
MAX_CONTEXTS: Final = BELIEF_MAX_CONTEXTS
MAX_EVIDENCE: Final = BELIEF_MAX_EVIDENCE
MAX_REVISIONS: Final = BELIEF_MAX_REVISIONS
MAX_PROPOSITION_LENGTH: Final = BELIEF_MAX_PROPOSITION_CODEPOINTS
MAX_COMPONENT_LENGTH: Final = BELIEF_MAX_COMPONENT_CODEPOINTS

BELIEF_PROPOSITION_DOMAIN: Final = b"PROJECT-KAGYA:R12:BELIEF-PROPOSITION:V1\0"
BELIEF_ADMISSION_DOMAIN: Final = b"PROJECT-KAGYA:R12:BELIEF-ADMISSION:V1\0"
BELIEF_REVISION_DOMAIN: Final = b"PROJECT-KAGYA:R12:BELIEF-REVISION:V1\0"
BELIEF_RECORD_DOMAIN: Final = b"PROJECT-KAGYA:R12:BELIEF-RECORD:V1\0"
_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a bounded non-negative exact integer")
    return value


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a positive bounded exact integer")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _normalize_text(value: object, name: str, maximum: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    for character in value:
        codepoint = ord(character)
        if codepoint == 0 or codepoint == 127 or (
            codepoint < 32 and character not in "\t\n\r"
        ):
            raise ValueError(f"{name} contains a prohibited control character")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds its code-point bound")
    return normalized


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    try:
        return validate_identifier(value)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{name} must be a valid identifier") from error


def _utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _datetime_value(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class BeliefStructuredProjection:
    """Optional typed S/P/O projection; it is not proposition identity."""

    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject",
            _normalize_text(self.subject, "subject", BELIEF_MAX_COMPONENT_CODEPOINTS),
        )
        object.__setattr__(
            self,
            "predicate",
            _normalize_text(self.predicate, "predicate", BELIEF_MAX_COMPONENT_CODEPOINTS),
        )
        object.__setattr__(
            self,
            "object",
            _normalize_text(self.object, "object", BELIEF_MAX_COMPONENT_CODEPOINTS),
        )


@dataclass(frozen=True, slots=True)
class BeliefProposition:
    canonical_text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    proposition_digest: str = field(init=False)

    def __post_init__(self) -> None:
        canonical_text = _normalize_text(
            self.canonical_text,
            "canonical_text",
            BELIEF_MAX_PROPOSITION_CODEPOINTS,
        )
        values = (self.subject, self.predicate, self.object)
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("subject, predicate, and object are all-or-none")
        if all(value is not None for value in values):
            projection = BeliefStructuredProjection(
                values[0], values[1], values[2]  # type: ignore[arg-type]
            )
            object.__setattr__(self, "subject", projection.subject)
            object.__setattr__(self, "predicate", projection.predicate)
            object.__setattr__(self, "object", projection.object)
        object.__setattr__(self, "canonical_text", canonical_text)
        object.__setattr__(
            self,
            "proposition_digest",
            hashlib.sha256(
                BELIEF_PROPOSITION_DOMAIN + canonical_text.encode("utf-8")
            ).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        canonical_text: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> BeliefProposition:
        return cls(canonical_text, subject, predicate, object)

    @property
    def normalized(self) -> str:
        return self.canonical_text

    @property
    def text(self) -> str:
        return self.canonical_text

    @property
    def digest(self) -> str:
        return self.proposition_digest

    @property
    def structured_projection(self) -> BeliefStructuredProjection | None:
        if self.subject is None:
            return None
        return BeliefStructuredProjection(self.subject, self.predicate, self.object)  # type: ignore[arg-type]


Proposition = BeliefProposition


def recompute_proposition_digest(proposition: BeliefProposition) -> str:
    if not isinstance(proposition, BeliefProposition):
        raise TypeError("proposition must be BeliefProposition")
    return hashlib.sha256(
        BELIEF_PROPOSITION_DOMAIN + proposition.canonical_text.encode("utf-8")
    ).hexdigest()


def validate_proposition_digest(proposition: BeliefProposition) -> str:
    expected = recompute_proposition_digest(proposition)
    if proposition.proposition_digest != expected:
        raise ValueError("proposition_digest does not match canonical_text")
    return expected


@dataclass(frozen=True, slots=True)
class BeliefEvidence:
    evidence_ref: str
    evidence_type: BeliefEvidenceType

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ref", validate_identifier(self.evidence_ref))
        _enum(self.evidence_type, BeliefEvidenceType, "evidence_type")

    @property
    def reference(self) -> str:
        return self.evidence_ref


BeliefEvidenceReference = BeliefEvidence
EvidenceReference = BeliefEvidence


def canonicalize_evidence(
    evidence: tuple[BeliefEvidence, ...],
) -> tuple[BeliefEvidence, ...]:
    if type(evidence) is not tuple:
        raise TypeError("evidence must be a tuple")
    if len(evidence) > BELIEF_MAX_EVIDENCE:
        raise ValueError("evidence exceeds its bound")
    for item in evidence:
        if not isinstance(item, BeliefEvidence):
            raise TypeError("evidence must contain BeliefEvidence values")
    refs = [item.evidence_ref for item in evidence]
    if len(set(refs)) != len(refs):
        raise ValueError("evidence references must be unique")
    return tuple(sorted(evidence, key=lambda item: (item.evidence_ref, item.evidence_type.value)))


@dataclass(frozen=True, slots=True)
class BeliefSubjectAdmission:
    """Typed proof for a later subject-admission primitive."""

    proposition_digest: str
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    reason: BeliefSubjectAdmissionReason
    admission_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.proposition_digest, "proposition_digest")
        if type(self.evidence_refs) is not tuple or not self.evidence_refs:
            raise TypeError("evidence_refs must be a non-empty tuple")
        refs = tuple(validate_identifier(reference) for reference in self.evidence_refs)
        if len(refs) > BELIEF_MAX_EVIDENCE:
            raise ValueError("evidence_refs exceeds its bound")
        if refs != tuple(sorted(set(refs))):
            raise ValueError("evidence_refs must be sorted and unique")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        object.__setattr__(
            self,
            "event_sequence",
            _positive_int(
                self.event_sequence,
                "event_sequence",
                maximum=BELIEF_MAX_EVENT_SEQUENCE,
            ),
        )
        _enum(self.reason, BeliefSubjectAdmissionReason, "reason")
        payload = {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_refs": list(self.evidence_refs),
            "proposition_digest": self.proposition_digest,
            "reason": self.reason.value,
        }
        object.__setattr__(
            self,
            "admission_digest",
            hashlib.sha256(BELIEF_ADMISSION_DOMAIN + _canonical_json(payload)).hexdigest(),
        )

    @property
    def digest(self) -> str:
        return self.admission_digest


BeliefAdmissionProof = BeliefSubjectAdmission
SubjectAdmissionProof = BeliefSubjectAdmission
BeliefAdmissionReason = BeliefSubjectAdmissionReason


@dataclass(frozen=True, slots=True)
class BeliefConflictCandidate:
    proposition_digests: tuple[str, str]
    shared_context_scope: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.proposition_digests) is not tuple or len(self.proposition_digests) != 2:
            raise TypeError("proposition_digests must contain exactly two digests")
        digests: tuple[str, str] = (
            _digest(self.proposition_digests[0], "proposition_digest"),
            _digest(self.proposition_digests[1], "proposition_digest"),
        )
        if digests[0] == digests[1] or digests != tuple(sorted(digests)):
            raise ValueError("conflict proposition digests must be distinct and canonical")
        object.__setattr__(self, "proposition_digests", digests)
        if type(self.shared_context_scope) is not tuple:
            raise TypeError("shared_context_scope must be a tuple")
        contexts = tuple(validate_identifier(value) for value in self.shared_context_scope)
        if contexts != tuple(sorted(set(contexts))):
            raise ValueError("shared_context_scope must be sorted and unique")
        if len(contexts) > BELIEF_MAX_CONTEXTS:
            raise ValueError("shared_context_scope exceeds its bound")
        object.__setattr__(self, "shared_context_scope", contexts)

    @property
    def left_proposition_digest(self) -> str:
        return self.proposition_digests[0]

    @property
    def right_proposition_digest(self) -> str:
        return self.proposition_digests[1]


def _context_scope(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(value) > BELIEF_MAX_CONTEXTS:
        raise ValueError(f"{name} exceeds its bound")
    contexts = tuple(validate_identifier(item) for item in value)
    if contexts != tuple(sorted(set(contexts))):
        raise ValueError(f"{name} must be sorted and unique")
    return contexts


def build_conflict_candidate(
    left: BeliefProposition,
    right: BeliefProposition,
    left_context_scope: tuple[str, ...] = (),
    right_context_scope: tuple[str, ...] = (),
) -> BeliefConflictCandidate | None:
    if not isinstance(left, BeliefProposition) or not isinstance(right, BeliefProposition):
        raise TypeError("conflict candidates require BeliefProposition values")
    if left.proposition_digest == right.proposition_digest:
        return None
    left_contexts = _context_scope(left_context_scope, "left_context_scope")
    right_contexts = _context_scope(right_context_scope, "right_context_scope")
    if not all(
        value is not None
        for value in (
            left.subject,
            left.predicate,
            left.object,
            right.subject,
            right.predicate,
            right.object,
        )
    ):
        return None
    if left.subject != right.subject or left.predicate != right.predicate:
        return None
    if left.object == right.object:
        return None
    if left_contexts and right_contexts and not set(left_contexts).intersection(right_contexts):
        return None
    shared = tuple(sorted(set(left_contexts).intersection(right_contexts)))
    ordered_digests = tuple(sorted((left.proposition_digest, right.proposition_digest)))
    if len(ordered_digests) != 2:
        raise AssertionError("distinct conflict propositions must have two digests")
    return BeliefConflictCandidate(
        (ordered_digests[0], ordered_digests[1]),
        shared,
    )


conflict_candidate = build_conflict_candidate


def is_conflict_candidate(
    left: BeliefProposition,
    right: BeliefProposition,
    left_context_scope: tuple[str, ...] = (),
    right_context_scope: tuple[str, ...] = (),
) -> bool:
    return (
        build_conflict_candidate(
            left, right, left_context_scope, right_context_scope
        )
        is not None
    )


@dataclass(frozen=True, slots=True)
class BeliefRevisionRecord:
    belief_id: str
    revision: int
    operation: BeliefRevisionOperation
    reason: BeliefRevisionReason
    created_at: datetime
    previous_revision_digest: str | None = None
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief_id", validate_identifier(self.belief_id))
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", maximum=BELIEF_MAX_REVISION),
        )
        _enum(self.operation, BeliefRevisionOperation, "operation")
        _enum(self.reason, BeliefRevisionReason, "reason")
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        if self.revision == 0 and self.previous_revision_digest is not None:
            raise ValueError("genesis revision cannot have a previous digest")
        if self.revision > 0 and self.previous_revision_digest is None:
            raise ValueError("non-genesis revision requires a previous digest")
        if self.previous_revision_digest is not None:
            _digest(self.previous_revision_digest, "previous_revision_digest")
        object.__setattr__(self, "record_digest", recompute_revision_digest(self))


def _revision_fields(record: BeliefRevisionRecord) -> dict[str, object]:
    return {
        "belief_id": record.belief_id,
        "created_at": _datetime_value(record.created_at),
        "operation": record.operation.value,
        "previous_revision_digest": record.previous_revision_digest,
        "reason": record.reason.value,
        "revision": record.revision,
    }


def canonical_revision_payload(record: BeliefRevisionRecord) -> bytes:
    if not isinstance(record, BeliefRevisionRecord):
        raise TypeError("record must be BeliefRevisionRecord")
    return BELIEF_REVISION_DOMAIN + _canonical_json(_revision_fields(record))


def recompute_revision_digest(record: BeliefRevisionRecord) -> str:
    return hashlib.sha256(canonical_revision_payload(record)).hexdigest()


def validate_revision_digest(record: BeliefRevisionRecord) -> str:
    expected = recompute_revision_digest(record)
    if record.record_digest != expected:
        raise ValueError("record_digest does not match immutable revision")
    return record.record_digest


@dataclass(frozen=True, slots=True)
class BeliefRecord:
    belief_id: str
    proposition: BeliefProposition
    lifecycle: BeliefLifecycle
    epistemic_status: BeliefEpistemicStatus
    confidence: float
    context_scope: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    evidence: tuple[BeliefEvidence, ...] = ()
    subject_admission: BeliefSubjectAdmission | None = None
    supersedes_id: str | None = None
    superseded_by_id: str | None = None
    revision: int = 0
    revision_history: tuple[BeliefRevisionRecord, ...] = ()
    history_anchor_digest: str | None = None
    schema_version: int = BELIEF_SCHEMA_VERSION

    MAX_EVENT_SEQUENCE: ClassVar[int] = BELIEF_MAX_EVENT_SEQUENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief_id", validate_identifier(self.belief_id))
        if not isinstance(self.proposition, BeliefProposition):
            raise TypeError("proposition must be BeliefProposition")
        if type(self.schema_version) is not int or self.schema_version != BELIEF_SCHEMA_VERSION:
            raise ValueError("unsupported Belief schema version")
        _enum(self.lifecycle, BeliefLifecycle, "lifecycle")
        _enum(self.epistemic_status, BeliefEpistemicStatus, "epistemic_status")
        object.__setattr__(self, "confidence", _fraction(self.confidence, "confidence"))
        object.__setattr__(
            self,
            "context_scope",
            _context_scope(self.context_scope, "context_scope"),
        )
        if self.valid_from is not None:
            object.__setattr__(self, "valid_from", _utc_datetime(self.valid_from, "valid_from"))
        if self.valid_until is not None:
            object.__setattr__(self, "valid_until", _utc_datetime(self.valid_until, "valid_until"))
        if self.valid_from is not None and self.valid_until is not None:
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until cannot precede valid_from")
        object.__setattr__(self, "evidence", canonicalize_evidence(self.evidence))
        evidence_refs = tuple(item.evidence_ref for item in self.evidence)
        if self.subject_admission is not None:
            if not isinstance(self.subject_admission, BeliefSubjectAdmission):
                raise TypeError("subject_admission must be BeliefSubjectAdmission")
            if self.subject_admission.proposition_digest != self.proposition.proposition_digest:
                raise ValueError("subject_admission proposition mismatch")
            if evidence_refs != self.subject_admission.evidence_refs:
                raise ValueError("subject admission evidence does not match record evidence")
        if self.lifecycle is BeliefLifecycle.ADOPTED and self.subject_admission is None:
            raise ValueError("adopted Beliefs require subject admission")
        object.__setattr__(
            self,
            "supersedes_id",
            _optional_identifier(self.supersedes_id, "supersedes_id"),
        )
        object.__setattr__(
            self,
            "superseded_by_id",
            _optional_identifier(self.superseded_by_id, "superseded_by_id"),
        )
        if self.supersedes_id == self.belief_id or self.superseded_by_id == self.belief_id:
            raise ValueError("a Belief cannot supersede itself")
        if self.lifecycle is BeliefLifecycle.SUPERSEDED:
            if self.superseded_by_id is None:
                raise ValueError("superseded Beliefs require superseded_by_id")
        elif self.superseded_by_id is not None:
            raise ValueError("only superseded Beliefs may name a successor")
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", maximum=BELIEF_MAX_REVISION),
        )
        if type(self.revision_history) is not tuple:
            raise TypeError("revision_history must be a tuple")
        if len(self.revision_history) > BELIEF_MAX_REVISIONS:
            raise ValueError("revision_history exceeds its bound")
        if self.revision == 0 and self.revision_history:
            raise ValueError("revision zero cannot retain prior revisions")
        if self.revision == 0 and self.history_anchor_digest is not None:
            raise ValueError("revision zero cannot retain a history anchor")
        if self.revision > 0 and not self.revision_history and self.history_anchor_digest is None:
            raise ValueError("nonzero revision requires history or an anchor")
        previous: BeliefRevisionRecord | None = None
        for item in self.revision_history:
            if not isinstance(item, BeliefRevisionRecord):
                raise TypeError("revision_history contains an invalid record")
            if item.belief_id != self.belief_id:
                raise ValueError("revision_history cannot mix Belief IDs")
            if previous is not None:
                if item.revision != previous.revision + 1:
                    raise ValueError("revision_history is non-monotonic")
                if item.previous_revision_digest != previous.record_digest:
                    raise ValueError("revision_history has a broken digest link")
            elif item.revision > 0:
                if self.history_anchor_digest is None:
                    raise ValueError("first retained revision requires a history anchor")
                if item.previous_revision_digest != self.history_anchor_digest:
                    raise ValueError("first retained revision does not link to its anchor")
            elif item.previous_revision_digest is not None:
                raise ValueError("genesis revision cannot link to a prior digest")
            if item.revision > self.revision:
                raise ValueError("revision_history contains a future revision")
            previous = item
        if previous is not None and previous.revision != self.revision - 1:
            raise ValueError("revision_history does not reach the current revision")
        if self.history_anchor_digest is not None:
            _digest(self.history_anchor_digest, "history_anchor_digest")

    @property
    def contexts(self) -> tuple[str, ...]:
        return self.context_scope

    @property
    def admission(self) -> BeliefSubjectAdmission | None:
        return self.subject_admission

    @property
    def history(self) -> tuple[BeliefRevisionRecord, ...]:
        return self.revision_history

    @property
    def is_active(self) -> bool:
        return self.lifecycle is BeliefLifecycle.ADOPTED and self.epistemic_status in {
            BeliefEpistemicStatus.PROBABLE,
            BeliefEpistemicStatus.ESTABLISHED,
        }

    def is_ordinary_active(
        self,
        *,
        at: datetime | None = None,
        context_id: str | None = None,
    ) -> bool:
        """Pure ordinary-active projection; no clock or state mutation is used."""

        if not self.is_active:
            return False
        if context_id is not None:
            context_id = validate_identifier(context_id)
        elif self.context_scope:
            return False
        if self.context_scope and context_id not in self.context_scope:
            return False
        if (self.valid_from is not None or self.valid_until is not None) and at is None:
            return False
        if at is not None:
            timestamp = _utc_datetime(at, "at")
            if self.valid_from is not None and timestamp < self.valid_from:
                return False
            if self.valid_until is not None and timestamp > self.valid_until:
                return False
        return True


def _record_fields(record: BeliefRecord) -> dict[str, object]:
    return {
        "belief_id": record.belief_id,
        "confidence": record.confidence.hex(),
        "context_scope": list(record.context_scope),
        "epistemic_status": record.epistemic_status.value,
        "evidence": [
            {"evidence_ref": item.evidence_ref, "evidence_type": item.evidence_type.value}
            for item in record.evidence
        ],
        "history_anchor_digest": record.history_anchor_digest,
        "lifecycle": record.lifecycle.value,
        "proposition_digest": record.proposition.proposition_digest,
        "structured_projection": None
        if record.proposition.structured_projection is None
        else {
            "object": record.proposition.object,
            "predicate": record.proposition.predicate,
            "subject": record.proposition.subject,
        },
        "revision": record.revision,
        "revision_history": [item.record_digest for item in record.revision_history],
        "schema_version": record.schema_version,
        "subject_admission": None
        if record.subject_admission is None
        else record.subject_admission.admission_digest,
        "superseded_by_id": record.superseded_by_id,
        "supersedes_id": record.supersedes_id,
        "valid_from": _datetime_value(record.valid_from),
        "valid_until": _datetime_value(record.valid_until),
    }


def canonical_belief_record_payload(record: BeliefRecord) -> bytes:
    if not isinstance(record, BeliefRecord):
        raise TypeError("record must be BeliefRecord")
    return BELIEF_RECORD_DOMAIN + _canonical_json(_record_fields(record))


def belief_record_digest(record: BeliefRecord) -> str:
    return hashlib.sha256(canonical_belief_record_payload(record)).hexdigest()


# Descriptive aliases used by callers and tests.
BeliefRecordRevision = BeliefRevisionRecord
canonical_belief_revision_payload = canonical_revision_payload
belief_revision_digest = recompute_revision_digest
validate_belief_revision_digest = validate_revision_digest


__all__ = [
    "AdmissionReason",
    "BeliefAdmissionReason",
    "BELIEF_ADMISSION_DOMAIN",
    "BELIEF_MAX_EVENT_SEQUENCE",
    "BELIEF_MAX_COMPONENT_CODEPOINTS",
    "BELIEF_MAX_CONTEXTS",
    "BELIEF_MAX_EVIDENCE",
    "BELIEF_MAX_PROPOSITION_CODEPOINTS",
    "BELIEF_MAX_REVISIONS",
    "BELIEF_PROPOSITION_DOMAIN",
    "BELIEF_RECORD_DOMAIN",
    "BELIEF_REVISION_DOMAIN",
    "BELIEF_SCHEMA_VERSION",
    "BeliefAdmissionProof",
    "BeliefConflictCandidate",
    "BeliefEpistemicStatus",
    "BeliefEvidence",
    "BeliefEvidenceReference",
    "BeliefEvidenceType",
    "BeliefLifecycle",
    "BeliefProposition",
    "BeliefRecord",
    "BeliefRecordRevision",
    "BeliefRevisionOperation",
    "BeliefRevisionReason",
    "BeliefRevisionRecord",
    "BeliefStructuredProjection",
    "BeliefSubjectAdmission",
    "BeliefSubjectAdmissionReason",
    "EpistemicStatus",
    "EvidenceReference",
    "EvidenceType",
    "Proposition",
    "SubjectAdmissionProof",
    "belief_record_digest",
    "belief_revision_digest",
    "build_conflict_candidate",
    "canonical_belief_record_payload",
    "canonical_belief_revision_payload",
    "canonicalize_evidence",
    "conflict_candidate",
    "is_conflict_candidate",
    "MAX_COMPONENT_LENGTH",
    "MAX_CONTEXTS",
    "MAX_EVIDENCE",
    "MAX_PROPOSITION_LENGTH",
    "MAX_REVISIONS",
    "recompute_proposition_digest",
    "validate_belief_revision_digest",
    "validate_proposition_digest",
    "validate_revision_digest",
]
