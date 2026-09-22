"""Pure R12 Semantic Memory lifecycle contracts.

The values in this module are deliberately independent of DB2/Chroma,
ContextRegistry, R07, and runtime state.  They describe the bounded evidence
that a later Memory-owned lifecycle authority may publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Final

from kagya.identifiers import validate_identifier


class _ClosedStrEnum(str, Enum):
    """Base for vocabularies whose values are closed at the domain boundary."""


class SemanticLifecycle(_ClosedStrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    QUARANTINED = "quarantined"
    ARCHIVED = "archived"


class SemanticSourceKind(_ClosedStrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class SemanticSourceStatus(_ClosedStrEnum):
    AVAILABLE = "available"
    UNKNOWN = "unknown"
    MISSING = "missing"
    RETRACTED = "retracted"
    SUPERSEDED = "superseded"


class SemanticProvenanceClass(_ClosedStrEnum):
    SINGLE_CONTEXT = "single_context"
    MULTI_CONTEXT = "multi_context"
    UNKNOWN = "unknown"
    INCOMPLETE = "incomplete"


class SemanticRevisionOperation(_ClosedStrEnum):
    CREATE = "create"
    CORRECT = "correct"
    SUPERSEDE = "supersede"
    RETRACT = "retract"
    QUARANTINE = "quarantine"
    ARCHIVE = "archive"


class SemanticRevisionReason(_ClosedStrEnum):
    CREATION = "creation"
    CORRECTION = "correction"
    SUPERSESSION = "supersession"
    RETRACTION = "retraction"
    QUARANTINE = "quarantine"
    ARCHIVAL = "archival"


# Compatibility aliases retain descriptive names used by early U1 consumers;
# they do not introduce additional enum values or authorities.
SourceEdgeStatus = SemanticSourceStatus
SemanticRevisionLifecycle = SemanticLifecycle


SEMANTIC_SCHEMA_VERSION: Final = 1
SEMANTIC_MAX_CONTENT_CODEPOINTS: Final = 16_384
SEMANTIC_MAX_SOURCE_EDGES: Final = 64
SEMANTIC_MAX_REVISION: Final = 2**31 - 1

SEMANTIC_CONTENT_DOMAIN: Final = b"PROJECT-KAGYA:R12:SEMANTIC-CONTENT:V1\0"
SEMANTIC_PROVENANCE_DOMAIN: Final = b"PROJECT-KAGYA:R12:SEMANTIC-PROVENANCE:V1\0"
SEMANTIC_REVISION_DOMAIN: Final = b"PROJECT-KAGYA:R12:SEMANTIC-REVISION:V1\0"


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return value


def _nonnegative_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a bounded non-negative exact integer")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _datetime_value(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def normalize_semantic_content(value: object) -> str:
    """Normalize only line endings and outer whitespace for new R12 content."""

    if type(value) is not str:
        raise TypeError("semantic_content must be an exact string")
    if "\x00" in value:
        raise ValueError("semantic_content must not contain NUL")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("semantic_content must be non-empty")
    if len(normalized) > SEMANTIC_MAX_CONTENT_CODEPOINTS:
        raise ValueError("semantic_content exceeds its code-point bound")
    return normalized


def semantic_content_digest(value: object) -> str:
    content = normalize_semantic_content(value)
    return hashlib.sha256(SEMANTIC_CONTENT_DOMAIN + content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticSourceEdge:
    """One bounded, canonical source relationship for a Semantic revision."""

    source_kind: SemanticSourceKind
    source_id: str
    source_revision: int | None = None
    captured_context_id: str | None = None
    source_status: SemanticSourceStatus = SemanticSourceStatus.AVAILABLE

    def __post_init__(self) -> None:
        _enum(self.source_kind, SemanticSourceKind, "source_kind")
        object.__setattr__(self, "source_id", validate_identifier(self.source_id))
        if self.source_revision is not None:
            object.__setattr__(
                self,
                "source_revision",
                _nonnegative_int(
                    self.source_revision,
                    "source_revision",
                    maximum=SEMANTIC_MAX_REVISION,
                ),
            )
        object.__setattr__(
            self,
            "captured_context_id",
            _optional_identifier(self.captured_context_id, "captured_context_id"),
        )
        _enum(self.source_status, SemanticSourceStatus, "source_status")

    @property
    def identity(self) -> tuple[str, str, int | None]:
        """Return identity excluding mutable evidence attributes."""

        return (self.source_kind.value, self.source_id, self.source_revision)

    # Read-only compatibility spellings for early callers.
    @property
    def kind(self) -> SemanticSourceKind:
        return self.source_kind

    @property
    def context_id(self) -> str | None:
        return self.captured_context_id

    @property
    def status(self) -> SemanticSourceStatus:
        return self.source_status


def canonicalize_source_edges(
    edges: tuple[SemanticSourceEdge, ...],
) -> tuple[SemanticSourceEdge, ...]:
    """Return sorted unique edges, rejecting conflicting duplicate identities."""

    if type(edges) is not tuple:
        raise TypeError("source_edges must be a tuple")
    if len(edges) > SEMANTIC_MAX_SOURCE_EDGES:
        raise ValueError("source_edges exceeds its bound")
    by_identity: dict[tuple[str, str, int | None], SemanticSourceEdge] = {}
    for edge in edges:
        if not isinstance(edge, SemanticSourceEdge):
            raise TypeError("source_edges must contain SemanticSourceEdge values")
        prior = by_identity.get(edge.identity)
        if prior is not None and prior != edge:
            raise ValueError("conflicting duplicate source edge")
        by_identity[edge.identity] = edge
    return tuple(
        by_identity[key]
        for key in sorted(
            by_identity,
            key=lambda identity: (
                identity[0],
                identity[1],
                -1 if identity[2] is None else identity[2],
            ),
        )
    )


canonical_source_edges = canonicalize_source_edges


@dataclass(frozen=True, slots=True)
class SemanticProvenance:
    """Bounded classification evidence retained beside a source-edge set."""

    classification: SemanticProvenanceClass
    source_count: int
    known_context_ids: tuple[str, ...] = ()
    unknown_source_count: int = 0
    incomplete_source_count: int = 0
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _enum(self.classification, SemanticProvenanceClass, "classification")
        _nonnegative_int(
            self.source_count,
            "source_count",
            maximum=SEMANTIC_MAX_SOURCE_EDGES,
        )
        if type(self.known_context_ids) is not tuple:
            raise TypeError("known_context_ids must be a tuple")
        contexts = tuple(validate_identifier(value) for value in self.known_context_ids)
        if contexts != tuple(sorted(set(contexts))):
            raise ValueError("known_context_ids must be sorted and unique")
        object.__setattr__(self, "known_context_ids", contexts)
        for name in ("unknown_source_count", "incomplete_source_count"):
            _nonnegative_int(
                getattr(self, name),
                name,
                maximum=SEMANTIC_MAX_SOURCE_EDGES,
            )
        if self.unknown_source_count > self.source_count:
            raise ValueError("unknown_source_count exceeds source_count")
        if self.incomplete_source_count > self.source_count:
            raise ValueError("incomplete_source_count exceeds source_count")
        object.__setattr__(self, "digest", provenance_digest(self))


def canonical_provenance_payload(provenance: SemanticProvenance) -> bytes:
    if not isinstance(provenance, SemanticProvenance):
        raise TypeError("provenance must be SemanticProvenance")
    return SEMANTIC_PROVENANCE_DOMAIN + _canonical_json(
        {
            "classification": provenance.classification.value,
            "incomplete_source_count": provenance.incomplete_source_count,
            "known_context_ids": list(provenance.known_context_ids),
            "source_count": provenance.source_count,
            "unknown_source_count": provenance.unknown_source_count,
        }
    )


def provenance_digest(provenance: SemanticProvenance) -> str:
    return hashlib.sha256(canonical_provenance_payload(provenance)).hexdigest()


def provenance_for_edges(
    edges: tuple[SemanticSourceEdge, ...],
) -> SemanticProvenance:
    canonical = canonicalize_source_edges(edges)
    known_contexts = tuple(
        sorted(
            {
                edge.captured_context_id
                for edge in canonical
                if edge.source_status is SemanticSourceStatus.AVAILABLE
                and edge.captured_context_id is not None
            }
        )
    )
    unknown_count = sum(
        edge.source_status is not SemanticSourceStatus.AVAILABLE
        or edge.captured_context_id is None
        for edge in canonical
    )
    incomplete_count = sum(
        edge.source_status
        in {
            SemanticSourceStatus.MISSING,
            SemanticSourceStatus.RETRACTED,
            SemanticSourceStatus.SUPERSEDED,
        }
        for edge in canonical
    )
    if not canonical:
        classification = SemanticProvenanceClass.UNKNOWN
    elif incomplete_count or (known_contexts and unknown_count):
        classification = SemanticProvenanceClass.INCOMPLETE
    elif not known_contexts:
        classification = SemanticProvenanceClass.UNKNOWN
    elif len(known_contexts) == 1:
        classification = SemanticProvenanceClass.SINGLE_CONTEXT
    else:
        classification = SemanticProvenanceClass.MULTI_CONTEXT
    return SemanticProvenance(
        classification=classification,
        source_count=len(canonical),
        known_context_ids=known_contexts,
        unknown_source_count=unknown_count,
        incomplete_source_count=incomplete_count,
    )


def classify_provenance(
    edges: tuple[SemanticSourceEdge, ...],
) -> SemanticProvenanceClass:
    return provenance_for_edges(edges).classification


@dataclass(frozen=True, slots=True)
class SemanticRevision:
    """One immutable new-format Semantic lifecycle revision."""

    semantic_id: str
    revision: int
    semantic_content: str
    content_digest: str
    created_at: datetime
    schema_version: int = SEMANTIC_SCHEMA_VERSION
    lifecycle: SemanticLifecycle = SemanticLifecycle.ACTIVE
    source_edges: tuple[SemanticSourceEdge, ...] = ()
    provenance_class: SemanticProvenanceClass | None = None
    operation: SemanticRevisionOperation = SemanticRevisionOperation.CREATE
    reason: SemanticRevisionReason = SemanticRevisionReason.CREATION
    previous_revision_digest: str | None = None
    event_id: str | None = None
    event_sequence: int | None = None
    provenance: SemanticProvenance = field(init=False)
    revision_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_id", validate_identifier(self.semantic_id))
        object.__setattr__(
            self,
            "revision",
            _nonnegative_int(self.revision, "revision", maximum=SEMANTIC_MAX_REVISION),
        )
        if type(self.schema_version) is not int or self.schema_version != SEMANTIC_SCHEMA_VERSION:
            raise ValueError("unsupported Semantic schema version")
        object.__setattr__(self, "semantic_content", normalize_semantic_content(self.semantic_content))
        _digest(self.content_digest, "content_digest")
        if self.content_digest != semantic_content_digest(self.semantic_content):
            raise ValueError("content_digest does not match semantic_content")
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        _enum(self.lifecycle, SemanticLifecycle, "lifecycle")
        _enum(self.operation, SemanticRevisionOperation, "operation")
        _enum(self.reason, SemanticRevisionReason, "reason")
        edges = canonicalize_source_edges(self.source_edges)
        object.__setattr__(self, "source_edges", edges)
        provenance = provenance_for_edges(edges)
        if self.provenance_class is None:
            object.__setattr__(self, "provenance_class", provenance.classification)
        else:
            _enum(self.provenance_class, SemanticProvenanceClass, "provenance_class")
            if self.provenance_class is not provenance.classification:
                raise ValueError("provenance_class does not match source_edges")
        object.__setattr__(self, "provenance", provenance)
        if self.previous_revision_digest is not None:
            _digest(self.previous_revision_digest, "previous_revision_digest")
        object.__setattr__(self, "event_id", _optional_identifier(self.event_id, "event_id"))
        if self.event_sequence is not None:
            object.__setattr__(
                self,
                "event_sequence",
                _nonnegative_int(
                    self.event_sequence,
                    "event_sequence",
                    maximum=SEMANTIC_MAX_REVISION,
                ),
            )
        object.__setattr__(self, "revision_digest", recompute_revision_digest(self))


def _revision_fields(revision: SemanticRevision) -> dict[str, object]:
    assert revision.provenance_class is not None
    return {
        "content_digest": revision.content_digest,
        "created_at": _datetime_value(revision.created_at),
        "event_id": revision.event_id,
        "event_sequence": revision.event_sequence,
        "lifecycle": revision.lifecycle.value,
        "operation": revision.operation.value,
        "previous_revision_digest": revision.previous_revision_digest,
        "provenance": {
            "classification": revision.provenance.classification.value,
            "digest": revision.provenance.digest,
            "incomplete_source_count": revision.provenance.incomplete_source_count,
            "known_context_ids": list(revision.provenance.known_context_ids),
            "source_count": revision.provenance.source_count,
            "unknown_source_count": revision.provenance.unknown_source_count,
        },
        "provenance_class": revision.provenance_class.value,
        "reason": revision.reason.value,
        "revision": revision.revision,
        "schema_version": revision.schema_version,
        "semantic_content": revision.semantic_content,
        "semantic_id": revision.semantic_id,
        "source_edges": [
            {
                "captured_context_id": edge.captured_context_id,
                "source_id": edge.source_id,
                "source_kind": edge.source_kind.value,
                "source_revision": edge.source_revision,
                "source_status": edge.source_status.value,
            }
            for edge in revision.source_edges
        ],
    }


def canonical_revision_payload(revision: SemanticRevision) -> bytes:
    if not isinstance(revision, SemanticRevision):
        raise TypeError("revision must be SemanticRevision")
    return SEMANTIC_REVISION_DOMAIN + _canonical_json(_revision_fields(revision))


def recompute_revision_digest(revision: SemanticRevision) -> str:
    return hashlib.sha256(canonical_revision_payload(revision)).hexdigest()


def validate_revision_digest(revision: SemanticRevision) -> str:
    expected = recompute_revision_digest(revision)
    if revision.revision_digest != expected:
        raise ValueError("revision_digest does not match immutable revision")
    return revision.revision_digest


# Explicit aliases make the boundary discoverable without adding alternate
# implementations.
canonical_semantic_revision_payload = canonical_revision_payload
semantic_revision_digest = recompute_revision_digest
validate_semantic_revision_digest = validate_revision_digest
SemanticMemoryRevision = SemanticRevision


__all__ = [
    "SEMANTIC_CONTENT_DOMAIN",
    "SEMANTIC_MAX_CONTENT_CODEPOINTS",
    "SEMANTIC_MAX_REVISION",
    "SEMANTIC_MAX_SOURCE_EDGES",
    "SEMANTIC_PROVENANCE_DOMAIN",
    "SEMANTIC_REVISION_DOMAIN",
    "SEMANTIC_SCHEMA_VERSION",
    "SemanticLifecycle",
    "SemanticMemoryRevision",
    "SemanticProvenance",
    "SemanticProvenanceClass",
    "SemanticRevision",
    "SemanticRevisionLifecycle",
    "SemanticRevisionOperation",
    "SemanticRevisionReason",
    "SemanticSourceEdge",
    "SemanticSourceKind",
    "SemanticSourceStatus",
    "SourceEdgeStatus",
    "canonical_provenance_payload",
    "canonical_revision_payload",
    "canonical_semantic_revision_payload",
    "canonical_source_edges",
    "canonicalize_source_edges",
    "classify_provenance",
    "normalize_semantic_content",
    "provenance_digest",
    "provenance_for_edges",
    "recompute_revision_digest",
    "semantic_content_digest",
    "semantic_revision_digest",
    "validate_revision_digest",
    "validate_semantic_revision_digest",
]
