"""Versioned structured feedback records and idempotent audit history."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
from typing import Any, Iterable
from uuid import uuid4


class FeedbackSignal(StrEnum):
    GOOD = "good"
    BAD = "bad"
    FACTUAL_ERROR = "factual_error"
    STYLE_PROBLEM = "style_problem"
    UNSAFE_BEHAVIOR = "unsafe_behavior"
    REMEMBER = "remember"
    DO_NOT_REMEMBER = "do_not_remember"
    CORRECTION = "correction"
    EXPECTED_ANSWER = "expected_answer"
    EXCLUDE_FROM_TRAINING = "exclude_from_training"


class FeedbackTargetType(StrEnum):
    RESPONSE = "response"
    EPISODE = "episode"
    MEMORY = "memory"
    DECISION = "decision"
    CONTEXT = "context"


class FeedbackStatus(StrEnum):
    ACTIVE = "active"
    WITHDRAWN = "withdrawn"


class TrainingDisposition(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass(frozen=True)
class FeedbackTarget:
    target_type: FeedbackTargetType
    target_id: str
    episode_id: str | None = None
    experience_id: str | None = None
    decision_id: str | None = None
    context_id: str | None = None

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("Feedback target ID must not be empty")


@dataclass(frozen=True)
class FeedbackProvenance:
    actor_type: str
    actor_id: str | None
    source: str
    event_id: str | None
    event_sequence: int | None
    submitted_at: str

    def __post_init__(self) -> None:
        if self.actor_type not in {"user", "operator"}:
            raise ValueError("Feedback actor type must be user or operator")
        if not self.source:
            raise ValueError("Feedback source must not be empty")


@dataclass(frozen=True)
class ValueEvidenceProposal:
    proposal_id: str
    direction: str
    strength: float
    reason_codes: tuple[str, ...]
    target_refs: tuple[str, ...]
    value_impacts: dict[str, float]
    status: str = "proposed"

    def __post_init__(self) -> None:
        if self.direction not in {"supporting", "opposing"}:
            raise ValueError("Value evidence direction is invalid")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Value evidence strength must be between zero and one")
        if any(not -1.0 <= value <= 1.0 for value in self.value_impacts.values()):
            raise ValueError("Value evidence impacts must be between minus one and one")


@dataclass(frozen=True)
class FeedbackPropagation:
    memory_id: str | None
    correction_memory_id: str | None
    memory_before: dict[str, str]
    memory_after: dict[str, str]
    decision_id: str | None
    decision_outcome_applied: bool
    prediction_error: float | None
    value_evidence: ValueEvidenceProposal | None
    training_disposition: TrainingDisposition
    exclusion_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackRevision:
    revision: int
    status: FeedbackStatus
    signals: tuple[FeedbackSignal, ...]
    target: FeedbackTarget
    provenance: FeedbackProvenance
    correction_memory_id: str | None
    expected_answer_memory_id: str | None
    propagation: FeedbackPropagation
    supersedes_revision: int | None
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("Feedback revision must be positive")
        if self.status == FeedbackStatus.ACTIVE and not self.signals:
            raise ValueError("Active feedback requires at least one signal")
        if len(self.signals) != len(set(self.signals)):
            raise ValueError("Feedback signals must be unique")
        if self.schema_version != 1:
            raise ValueError("Unsupported feedback revision schema version")


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    current_revision: int
    revisions: tuple[FeedbackRevision, ...]
    created_at: str
    updated_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.feedback_id or not self.revisions:
            raise ValueError("Feedback record ID and revisions must not be empty")
        if self.schema_version != 1:
            raise ValueError("Unsupported feedback record schema version")
        if self.current_revision != self.revisions[-1].revision:
            raise ValueError("Feedback current revision does not match its history")

    @property
    def current(self) -> FeedbackRevision:
        return self.revisions[-1]


class FeedbackStore:
    """Append-only logical feedback records with operation idempotency."""

    def __init__(self) -> None:
        self.records: dict[str, FeedbackRecord] = {}
        self.idempotency: dict[str, tuple[str, str]] = {}

    def idempotent_result(
        self, idempotency_key: str, fingerprint: str
    ) -> FeedbackRecord | None:
        existing = self.idempotency.get(idempotency_key)
        if existing is None:
            return None
        existing_fingerprint, feedback_id = existing
        if existing_fingerprint != fingerprint:
            raise ValueError("Idempotency key was already used for another operation")
        return self.get(feedback_id)

    def create(
        self,
        *,
        signals: tuple[FeedbackSignal, ...],
        target: FeedbackTarget,
        provenance: FeedbackProvenance,
        correction_memory_id: str | None,
        expected_answer_memory_id: str | None,
        propagation: FeedbackPropagation,
        idempotency_key: str,
        fingerprint: str,
        feedback_id: str | None = None,
    ) -> FeedbackRecord:
        identifier = feedback_id or f"feedback-{uuid4()}"
        if identifier in self.records:
            raise ValueError(f"Feedback already exists: {identifier}")
        now = _now()
        revision = FeedbackRevision(
            revision=1,
            status=FeedbackStatus.ACTIVE,
            signals=signals,
            target=target,
            provenance=provenance,
            correction_memory_id=correction_memory_id,
            expected_answer_memory_id=expected_answer_memory_id,
            propagation=propagation,
            supersedes_revision=None,
            created_at=now,
        )
        record = FeedbackRecord(identifier, 1, (revision,), now, now)
        self.records[identifier] = record
        self.idempotency[idempotency_key] = (fingerprint, identifier)
        return record

    def revise(
        self,
        feedback_id: str,
        *,
        expected_revision: int,
        signals: tuple[FeedbackSignal, ...],
        target: FeedbackTarget,
        provenance: FeedbackProvenance,
        correction_memory_id: str | None,
        expected_answer_memory_id: str | None,
        propagation: FeedbackPropagation,
        idempotency_key: str,
        fingerprint: str,
        status: FeedbackStatus = FeedbackStatus.ACTIVE,
    ) -> FeedbackRecord:
        record = self.get(feedback_id)
        if record.current_revision != expected_revision:
            raise ValueError(
                f"Feedback revision conflict: expected {expected_revision}, "
                f"current {record.current_revision}"
            )
        now = _now()
        revision = FeedbackRevision(
            revision=expected_revision + 1,
            status=status,
            signals=signals,
            target=target,
            provenance=provenance,
            correction_memory_id=correction_memory_id,
            expected_answer_memory_id=expected_answer_memory_id,
            propagation=propagation,
            supersedes_revision=expected_revision,
            created_at=now,
        )
        updated = FeedbackRecord(
            feedback_id=record.feedback_id,
            current_revision=revision.revision,
            revisions=(*record.revisions, revision),
            created_at=record.created_at,
            updated_at=now,
        )
        self.records[feedback_id] = updated
        self.idempotency[idempotency_key] = (fingerprint, feedback_id)
        return updated

    def get(self, feedback_id: str) -> FeedbackRecord:
        try:
            return self.records[feedback_id]
        except KeyError as exc:
            raise ValueError(f"Unknown feedback: {feedback_id}") from exc

    def list_records(self) -> list[FeedbackRecord]:
        return sorted(self.records.values(), key=lambda item: item.created_at)

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("schema_version", 1) != 1:
            raise ValueError("Unsupported feedback store schema version")
        self.records = {}
        for raw in payload.get("records", []):
            if not isinstance(raw, dict):
                continue
            record = _record_from_json(raw)
            self.records[record.feedback_id] = record
        self.idempotency = {
            str(key): (str(value[0]), str(value[1]))
            for key, value in dict(payload.get("idempotency", {})).items()
            if isinstance(value, list) and len(value) == 2
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": [asdict(record) for record in self.list_records()],
            "idempotency": {
                key: list(value) for key, value in sorted(self.idempotency.items())
            },
        }


def feedback_fingerprint(operation: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"operation": operation, "payload": payload},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_signals(signals: Iterable[FeedbackSignal]) -> tuple[FeedbackSignal, ...]:
    normalized = tuple(dict.fromkeys(signals))
    if (
        FeedbackSignal.REMEMBER in normalized
        and FeedbackSignal.DO_NOT_REMEMBER in normalized
    ):
        raise ValueError("remember and do_not_remember are mutually exclusive")
    if FeedbackSignal.GOOD in normalized and FeedbackSignal.BAD in normalized:
        raise ValueError("good and bad are mutually exclusive")
    return normalized


def _record_from_json(payload: dict[str, Any]) -> FeedbackRecord:
    revisions: list[FeedbackRevision] = []
    for raw in payload.get("revisions", []):
        target = FeedbackTarget(
            **{
                **raw["target"],
                "target_type": FeedbackTargetType(raw["target"]["target_type"]),
            }
        )
        propagation_raw = dict(raw["propagation"])
        proposal_raw = propagation_raw.get("value_evidence")
        propagation_raw["value_evidence"] = (
            None
            if proposal_raw is None
            else ValueEvidenceProposal(
                **{
                    **proposal_raw,
                    "reason_codes": tuple(proposal_raw.get("reason_codes", ())),
                    "target_refs": tuple(proposal_raw.get("target_refs", ())),
                    "value_impacts": dict(proposal_raw.get("value_impacts", {})),
                }
            )
        )
        propagation_raw["training_disposition"] = TrainingDisposition(
            propagation_raw["training_disposition"]
        )
        propagation_raw["exclusion_refs"] = tuple(
            propagation_raw.get("exclusion_refs", ())
        )
        propagation_raw["reason_codes"] = tuple(propagation_raw.get("reason_codes", ()))
        revisions.append(
            FeedbackRevision(
                **{
                    **raw,
                    "status": FeedbackStatus(raw["status"]),
                    "signals": tuple(
                        FeedbackSignal(item) for item in raw.get("signals", ())
                    ),
                    "target": target,
                    "provenance": FeedbackProvenance(**raw["provenance"]),
                    "propagation": FeedbackPropagation(**propagation_raw),
                }
            )
        )
    return FeedbackRecord(
        feedback_id=str(payload["feedback_id"]),
        current_revision=int(payload["current_revision"]),
        revisions=tuple(revisions),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        schema_version=int(payload.get("schema_version", 1)),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
