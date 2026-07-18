"""Structured provenance and trade-off records for dynamic values."""

from dataclasses import dataclass
from enum import StrEnum
import math

from kagya.identity import IdentityOrigin


class ValueScope(StrEnum):
    SUBJECT = "subject"
    CONTEXT = "context"


class EvidenceDirection(StrEnum):
    SUPPORTING = "supporting"
    OPPOSING = "opposing"


@dataclass(frozen=True)
class ValueEvidenceRecord:
    evidence_id: str
    value_id: str
    direction: EvidenceDirection
    strength: float
    confidence: float
    experience_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    decision_id: str | None
    context_id: str | None
    source: str
    identity_origin: IdentityOrigin
    event_id: str | None
    event_sequence: int | None
    created_at: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.value_id or not self.source:
            raise ValueError("Value evidence identifiers and source must not be empty")
        for value, name in (
            (self.strength, "strength"),
            (self.confidence, "confidence"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Value evidence {name} must be between zero and one")


@dataclass(frozen=True)
class ValueRevisionDiff:
    from_revision: int
    to_revision: int
    changed_fields: dict[str, tuple[object, object]]
    reason_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValueTradeoffRecord:
    tradeoff_id: str
    value_ids: tuple[str, ...]
    value_revision_refs: dict[str, int]
    option_id: str
    context_id: str | None
    contribution_by_value: dict[str, float]
    conflict_names: tuple[str, ...]
    reasoning_codes: tuple[str, ...]
    decision_id: str | None
    created_at: str


@dataclass(frozen=True)
class ValueReassessmentRecord:
    reassessment_id: str
    decision_id: str
    outcome_utility: float
    prediction_error: float
    regret: float
    value_update_ids: tuple[str, ...]
    created_at: str
