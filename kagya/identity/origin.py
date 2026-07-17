"""Shared provenance boundary between self-originated and external inputs."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
import re
from typing import Any
from uuid import uuid4


class OriginActor(StrEnum):
    SELF = "self"
    USER = "user"
    OPERATOR = "operator"
    SYSTEM = "system"
    EXTERNAL_SOURCE = "external_source"
    MODEL_INFERENCE = "model_inference"
    INHERITED = "inherited"
    UNKNOWN = "unknown"


class OriginInputKind(StrEnum):
    INTERNAL_STATE = "internal_state"
    REQUEST = "request"
    SUGGESTION = "suggestion"
    CONSTRAINT = "constraint"
    FEEDBACK = "feedback"
    EVIDENCE = "evidence"
    OBSERVATION = "observation"
    CONFIG_SEED = "config_seed"
    DECISION_OUTCOME = "decision_outcome"
    LEGACY = "legacy"


class EndorsementStatus(StrEnum):
    PENDING = "pending"
    ENDORSED = "endorsed"
    REJECTED = "rejected"
    IMPOSED = "imposed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class IdentityOrigin:
    origin_id: str
    actor: OriginActor
    input_kind: OriginInputKind
    endorsement: EndorsementStatus
    source_ref: str | None
    event_id: str | None
    event_sequence: int | None
    context_id: str | None
    confidence: float
    created_at: str
    endorsement_ref: str | None = None
    endorsed_by_event_id: str | None = None
    endorsed_by_event_sequence: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported identity origin schema version: {self.schema_version}"
            )
        for label, value in (
            ("origin_id", self.origin_id),
            ("source_ref", self.source_ref),
            ("event_id", self.event_id),
            ("context_id", self.context_id),
            ("endorsement_ref", self.endorsement_ref),
            ("endorsed_by_event_id", self.endorsed_by_event_id),
        ):
            if value is not None and re.fullmatch(
                r"[A-Za-z0-9._:@/-]{1,160}", value
            ) is None:
                raise ValueError(f"{label} must be an opaque safe reference")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("origin confidence must be finite and between zero and one")
        if datetime.fromisoformat(self.created_at).tzinfo is None:
            raise ValueError("origin created_at must include a timezone")
        if self.event_sequence is not None and self.event_sequence < 0:
            raise ValueError("origin event sequence must not be negative")
        if (
            self.endorsed_by_event_sequence is not None
            and self.endorsed_by_event_sequence < 0
        ):
            raise ValueError("endorsement event sequence must not be negative")
        if self.endorsement == EndorsementStatus.ENDORSED and (
            self.actor
            in {
                OriginActor.USER,
                OriginActor.OPERATOR,
                OriginActor.EXTERNAL_SOURCE,
                OriginActor.MODEL_INFERENCE,
                OriginActor.INHERITED,
                OriginActor.UNKNOWN,
            }
            and self.endorsement_ref is None
        ):
            raise ValueError("external origin requires explicit subject endorsement")
        if self.endorsement == EndorsementStatus.IMPOSED and self.actor not in {
            OriginActor.OPERATOR,
            OriginActor.SYSTEM,
        }:
            raise ValueError("only operator or system constraints may be imposed")

    def endorse(
        self,
        endorsement_ref: str,
        *,
        event_id: str | None,
        event_sequence: int | None,
    ) -> "IdentityOrigin":
        if not endorsement_ref:
            raise ValueError("endorsement reference must not be empty")
        return replace(
            self,
            endorsement=EndorsementStatus.ENDORSED,
            endorsement_ref=endorsement_ref,
            endorsed_by_event_id=event_id,
            endorsed_by_event_sequence=event_sequence,
        )

    def reject(
        self,
        endorsement_ref: str,
        *,
        event_id: str | None,
        event_sequence: int | None,
    ) -> "IdentityOrigin":
        return replace(
            self,
            endorsement=EndorsementStatus.REJECTED,
            endorsement_ref=endorsement_ref,
            endorsed_by_event_id=event_id,
            endorsed_by_event_sequence=event_sequence,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def new_identity_origin(
    actor: OriginActor,
    input_kind: OriginInputKind,
    *,
    source_ref: str | None = None,
    event_id: str | None = None,
    event_sequence: int | None = None,
    context_id: str | None = None,
    confidence: float = 1.0,
    endorsement: EndorsementStatus | None = None,
) -> IdentityOrigin:
    resolved_endorsement = endorsement
    if resolved_endorsement is None:
        if actor == OriginActor.SELF and input_kind == OriginInputKind.INTERNAL_STATE:
            resolved_endorsement = EndorsementStatus.ENDORSED
        elif (
            actor in {OriginActor.OPERATOR, OriginActor.SYSTEM}
            and input_kind == OriginInputKind.CONSTRAINT
        ):
            resolved_endorsement = EndorsementStatus.IMPOSED
        else:
            resolved_endorsement = EndorsementStatus.PENDING
    return IdentityOrigin(
        origin_id=str(uuid4()),
        actor=actor,
        input_kind=input_kind,
        endorsement=resolved_endorsement,
        source_ref=source_ref,
        event_id=event_id,
        event_sequence=event_sequence,
        context_id=context_id,
        confidence=confidence,
        created_at=datetime.now(UTC).isoformat(),
        endorsement_ref=(
            "self_generated"
            if resolved_endorsement == EndorsementStatus.ENDORSED
            and actor == OriginActor.SELF
            else None
        ),
    )


def legacy_identity_origin(
    source_ref: str = "legacy",
    *,
    event_id: str | None = None,
    event_sequence: int | None = None,
) -> IdentityOrigin:
    return new_identity_origin(
        OriginActor.INHERITED,
        OriginInputKind.LEGACY,
        source_ref=_safe_or_default(source_ref, "legacy"),
        event_id=event_id,
        event_sequence=event_sequence,
        confidence=0.0,
        endorsement=EndorsementStatus.UNCERTAIN,
    )


def identity_origin_from_json(
    payload: object, *, fallback_source: str = "legacy"
) -> IdentityOrigin:
    if not isinstance(payload, dict):
        return legacy_identity_origin(fallback_source)
    data = dict(payload)
    data["actor"] = OriginActor(data["actor"])
    data["input_kind"] = OriginInputKind(data["input_kind"])
    data["endorsement"] = EndorsementStatus(data["endorsement"])
    return IdentityOrigin(**data)


def _safe_or_default(value: str, default: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9._:@/-]{1,160}", value) else default
