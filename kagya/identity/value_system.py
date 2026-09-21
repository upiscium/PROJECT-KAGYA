"""Immutable value declarations and non-authoritative value observations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
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
    "ValueProposal",
    "ValueReason",
    "ValueSeedDeclaration",
    "ValueScope",
    "ValueState",
    "canonical_seed_payload",
    "recompute_seed_contract_digest",
    "validate_seed_contract_digest",
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


_NAME_LIMIT: Final = 128
_CONCEPT_LIMIT: Final = 2048
_MAX_REFS: Final = 16
_SEED_DOMAIN: Final = "kagya.identity.value-seed/v1"


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

    @classmethod
    def from_value(cls, value: ValueState) -> ValueSeedDeclaration:
        """Project immutable seed fields from a Value without admission data."""

        if not isinstance(value, ValueState):
            raise TypeError("value must be a ValueState")
        return cls(
            value_id=value.value_id,
            name=value.name,
            concept=value.concept,
            scope=value.scope,
            context_ids=value.context_ids,
            polarity=value.polarity,
            initial_strength=value.strength,
            confidence=value.confidence,
            stability=value.stability,
            protectedness=value.protectedness,
            negotiability=value.negotiability,
            allowed_update_rate=value.allowed_update_rate,
        )


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
        if not isinstance(self.origin, IdentityOrigin):
            raise TypeError("origin must be an IdentityOrigin")
        evidence = _refs(self.evidence_refs, "evidence_refs")
        object.__setattr__(self, "evidence_refs", evidence)
        if self.seed_contract_digest is not None:
            if type(self.seed_contract_digest) is not str or len(self.seed_contract_digest) != 64:
                raise ValueError("seed_contract_digest must be a lowercase SHA-256 digest")
            if any(char not in "0123456789abcdef" for char in self.seed_contract_digest):
                raise ValueError("seed_contract_digest must be a lowercase SHA-256 digest")
        if self.origin.admission is ValueAdmissionStatus.SYSTEM_AUTHORIZED:
            if self.seed_contract_digest is None:
                raise ValueError("system-authorized values require a seed contract digest")
            validate_seed_contract_digest(
                ValueSeedDeclaration.from_value(self), self.seed_contract_digest
            )
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
    if type(digest) is not str or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ValueError("seed_contract_digest must be a lowercase SHA-256 digest")
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
