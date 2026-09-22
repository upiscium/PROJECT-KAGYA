"""Immutable, content-addressed provenance for identity inputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, TypeVar, cast

from kagya.identifiers import validate_identifier


class _ClosedStrEnum(str, Enum):
    """String enum base which does not accept values outside its members."""


class OriginActor(_ClosedStrEnum):
    SELF = "self"
    USER = "user"
    OPERATOR = "operator"
    SYSTEM = "system"
    EXTERNAL_SOURCE = "external_source"
    MODEL_INFERENCE = "model_inference"
    INHERITED = "inherited"
    UNKNOWN = "unknown"


class OriginInputKind(_ClosedStrEnum):
    INTERNAL_STATE = "internal_state"
    REQUEST = "request"
    SUGGESTION = "suggestion"
    CONSTRAINT = "constraint"
    FEEDBACK = "feedback"
    EVIDENCE = "evidence"
    CONFIG_SEED = "config_seed"
    LEGACY = "legacy"


class ValueAdmissionStatus(_ClosedStrEnum):
    PENDING = "pending"
    SELF_ENDORSED = "self_endorsed"
    SYSTEM_AUTHORIZED = "system_authorized"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


_ORIGIN_DOMAIN: Final = "kagya.identity.origin/v1"
_ACTIVE: Final = frozenset(
    {ValueAdmissionStatus.SELF_ENDORSED, ValueAdmissionStatus.SYSTEM_AUTHORIZED}
)


_OriginEnum = TypeVar("_OriginEnum", bound=_ClosedStrEnum)


def _enum(
    value: object, enum_type: type[_OriginEnum], name: str
) -> _OriginEnum:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return cast(_OriginEnum, value)


def _canonical_provenance(origin: IdentityOrigin) -> bytes:
    # Admission is deliberately absent: changing a decision cannot create a new
    # provenance identity. JSON is used only as a deterministic byte encoding.
    fields = {
        "actor": origin.actor.value,
        "input_kind": origin.input_kind.value,
        "source_ref": origin.source_ref,
        "event_id": origin.event_id,
        "context_id": origin.context_id,
        "event_sequence": origin.event_sequence,
        "confidence": origin.confidence.hex(),
    }
    encoded = json.dumps(
        fields, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return _ORIGIN_DOMAIN.encode("ascii") + b"\0" + encoded


def recompute_origin_id(origin: IdentityOrigin) -> str:
    """Recompute the content address from an origin's immutable provenance."""

    if not isinstance(origin, IdentityOrigin):
        raise TypeError("origin must be an IdentityOrigin")
    return hashlib.sha256(_canonical_provenance(origin)).hexdigest()


def validate_origin_id(origin: IdentityOrigin) -> str:
    """Validate and return an origin's stored content address."""

    expected = recompute_origin_id(origin)
    if origin.origin_id != expected:
        raise ValueError("origin_id does not match immutable provenance")
    return origin.origin_id


@dataclass(frozen=True, slots=True)
class IdentityOrigin:
    """Strict immutable provenance record with a deterministic origin identity."""

    actor: OriginActor
    input_kind: OriginInputKind
    admission: ValueAdmissionStatus
    source_ref: str | None = None
    event_id: str | None = None
    context_id: str | None = None
    event_sequence: int | None = None
    confidence: float = 1.0
    origin_id: str = field(init=False)

    def __post_init__(self) -> None:
        actor = _enum(self.actor, OriginActor, "actor")
        input_kind = _enum(self.input_kind, OriginInputKind, "input_kind")
        admission = _enum(self.admission, ValueAdmissionStatus, "admission")
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "input_kind", input_kind)
        object.__setattr__(self, "admission", admission)

        for name in ("source_ref", "event_id", "context_id"):
            value = getattr(self, name)
            if value is not None:
                validate_identifier(value)
        if self.event_sequence is not None:
            if type(self.event_sequence) is not int or self.event_sequence < 0:
                raise ValueError("event_sequence must be a nonnegative exact integer")
        if type(self.confidence) is not float or not math.isfinite(self.confidence):
            raise TypeError("confidence must be a finite float")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        self._validate_admission(
            actor, input_kind, admission, self.event_id, self.event_sequence
        )
        object.__setattr__(self, "origin_id", recompute_origin_id(self))

    @staticmethod
    def _validate_admission(
        actor: OriginActor,
        input_kind: OriginInputKind,
        admission: ValueAdmissionStatus,
        event_id: str | None,
        event_sequence: int | None,
    ) -> None:
        self_endorsed_evidence = (
            actor is OriginActor.SELF
            and input_kind is OriginInputKind.INTERNAL_STATE
            and event_id is not None
            and event_sequence is not None
        )
        system_authorized_evidence = (
            actor is OriginActor.SYSTEM
            and input_kind is OriginInputKind.CONFIG_SEED
        )
        if (
            admission is ValueAdmissionStatus.SELF_ENDORSED
            and not self_endorsed_evidence
        ):
            raise ValueError("self_endorsed requires self/internal_state provenance")
        if (
            admission is ValueAdmissionStatus.SYSTEM_AUTHORIZED
            and not system_authorized_evidence
        ):
            raise ValueError("system_authorized requires system/config_seed provenance")
        if actor in (OriginActor.INHERITED, OriginActor.UNKNOWN) and admission in _ACTIVE:
            raise ValueError("inherited and unknown origins cannot be active")
        if input_kind is OriginInputKind.CONSTRAINT and actor in (
            OriginActor.SYSTEM,
            OriginActor.OPERATOR,
        ) and admission in _ACTIVE:
            raise ValueError("system/operator constraints cannot be active")
