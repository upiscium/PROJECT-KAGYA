"""Versioned beliefs separated from observed and remembered records."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
import re
from typing import Any
from uuid import uuid4

from kagya.identity import (
    EndorsementStatus,
    IdentityOrigin,
    identity_origin_from_json,
    OriginActor,
)


class BeliefLifecycle(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"
    REJECTED = "rejected"


class EpistemicStatus(StrEnum):
    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"
    PROBABLE = "probable"
    ESTABLISHED = "established"


@dataclass(frozen=True)
class Proposition:
    normalized: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None

    def __post_init__(self) -> None:
        if not self.normalized.strip() or len(self.normalized) > 2000:
            raise ValueError(
                "normalized proposition must contain at most 2000 characters"
            )
        structured = (self.subject, self.predicate, self.object)
        if any(item is not None for item in structured) and not all(
            item is not None and item.strip() for item in structured
        ):
            raise ValueError(
                "structured proposition requires subject, predicate, and object"
            )

    @classmethod
    def create(
        cls,
        text: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> "Proposition":
        return cls(
            normalized=" ".join(text.strip().split()),
            subject=_normalized_component(subject),
            predicate=_normalized_component(predicate),
            object=_normalized_component(object),
        )


@dataclass(frozen=True)
class BeliefEvidence:
    reference: str
    evidence_type: str
    source_trust: float
    observed_at: str

    def __post_init__(self) -> None:
        _safe_ref(self.reference, "belief evidence reference")
        _safe_ref(self.evidence_type, "belief evidence type")
        _unit(self.source_trust, "source trust")
        if datetime.fromisoformat(self.observed_at).tzinfo is None:
            raise ValueError("belief evidence timestamp must include a timezone")


@dataclass(frozen=True)
class BeliefRevision:
    revision_id: str
    from_revision: int
    to_revision: int
    operation: str
    reason_code: str
    evidence_refs: tuple[str, ...]
    before: dict[str, Any]
    after: dict[str, Any]
    event_id: str | None
    event_sequence: int | None
    created_at: str
    reviewer_id: str | None = None
    reviewer_authority: str | None = None

    def __post_init__(self) -> None:
        _safe_ref(self.revision_id, "belief revision ID")
        _safe_ref(self.operation, "belief revision operation")
        _safe_ref(self.reason_code, "belief revision reason")
        for reference in self.evidence_refs:
            _safe_ref(reference, "belief revision evidence")
        if self.from_revision < 0 or self.to_revision != self.from_revision + 1:
            raise ValueError("belief revisions must be sequential")
        if datetime.fromisoformat(self.created_at).tzinfo is None:
            raise ValueError("belief revision timestamp must include a timezone")


@dataclass(frozen=True)
class BeliefRecord:
    belief_id: str
    proposition: Proposition
    confidence: float
    epistemic_status: EpistemicStatus
    lifecycle: BeliefLifecycle
    identity_origin: IdentityOrigin
    evidence: tuple[BeliefEvidence, ...]
    context_scope: tuple[str, ...]
    valid_from: str | None
    valid_until: str | None
    contradiction_ids: tuple[str, ...]
    supersedes_id: str | None
    superseded_by_id: str | None
    created_at: str
    updated_at: str
    revision: int = 0
    revisions: tuple[BeliefRevision, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported belief schema version: {self.schema_version}"
            )
        _safe_ref(self.belief_id, "belief ID")
        _unit(self.confidence, "belief confidence")
        if (
            self.epistemic_status == EpistemicStatus.ESTABLISHED
            and self.confidence < 0.8
        ):
            raise ValueError("established belief confidence must be at least 0.8")
        if (
            self.lifecycle == BeliefLifecycle.ACTIVE
            and self.identity_origin.endorsement != EndorsementStatus.ENDORSED
        ):
            raise ValueError("active belief requires explicit subject endorsement")
        for value in self.context_scope:
            _safe_ref(value, "belief context scope")
        for value in self.contradiction_ids:
            _safe_ref(value, "belief contradiction ID")
        valid_from = _parse_time(self.valid_from)
        valid_until = _parse_time(self.valid_until)
        if (
            valid_from is not None
            and valid_until is not None
            and valid_until <= valid_from
        ):
            raise ValueError("belief valid_until must be after valid_from")

    def to_json(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class BeliefStore:
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self.records: dict[str, BeliefRecord] = {}

    def propose(
        self,
        proposition: Proposition,
        *,
        identity_origin: IdentityOrigin,
        evidence: tuple[BeliefEvidence, ...],
        confidence: float,
        context_scope: tuple[str, ...] = (),
        valid_from: str | None = None,
        valid_until: str | None = None,
        belief_id: str | None = None,
    ) -> BeliefRecord:
        if not evidence:
            raise ValueError("belief proposal requires evidence")
        identifier = belief_id or f"belief-{uuid4()}"
        if identifier in self.records:
            raise ValueError(f"Belief already exists: {identifier}")
        now = _now()
        record = BeliefRecord(
            belief_id=identifier,
            proposition=proposition,
            confidence=confidence,
            epistemic_status=EpistemicStatus.UNKNOWN
            if confidence == 0.0
            else EpistemicStatus.UNCERTAIN,
            lifecycle=BeliefLifecycle.PROPOSED,
            identity_origin=identity_origin,
            evidence=evidence,
            context_scope=context_scope,
            valid_from=valid_from,
            valid_until=valid_until,
            contradiction_ids=(),
            supersedes_id=None,
            superseded_by_id=None,
            created_at=now,
            updated_at=now,
        )
        contradictions = [
            existing
            for existing in self.records.values()
            if _contradicts(record, existing)
            and existing.lifecycle
            not in {
                BeliefLifecycle.RETRACTED,
                BeliefLifecycle.EXPIRED,
                BeliefLifecycle.REJECTED,
                BeliefLifecycle.SUPERSEDED,
            }
        ]
        if contradictions:
            record = replace(
                record,
                lifecycle=BeliefLifecycle.DISPUTED,
                contradiction_ids=tuple(item.belief_id for item in contradictions),
            )
            for existing in contradictions:
                updated = replace(
                    existing,
                    lifecycle=BeliefLifecycle.DISPUTED,
                    contradiction_ids=tuple(
                        dict.fromkeys((*existing.contradiction_ids, identifier))
                    ),
                    updated_at=now,
                )
                self.records[existing.belief_id] = self._with_revision(
                    existing,
                    updated,
                    operation="contradiction_detected",
                    reason_code="structured_proposition_conflict",
                    evidence_refs=tuple(item.reference for item in evidence),
                    event_id=None,
                    event_sequence=None,
                )
        self.records[identifier] = record
        return record

    def resolve(
        self,
        belief_id: str,
        *,
        accept: bool,
        confidence: float,
        epistemic_status: EpistemicStatus,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
        reviewer_id: str | None = None,
        reviewer_authority: str | None = None,
    ) -> BeliefRecord:
        current = self.get(belief_id)
        if current.lifecycle not in {
            BeliefLifecycle.PROPOSED,
            BeliefLifecycle.DISPUTED,
        }:
            raise ValueError("Only proposed or disputed beliefs can be resolved")
        if not evidence_refs:
            raise ValueError("belief resolution requires evidence references")
        if accept and epistemic_status == EpistemicStatus.UNKNOWN:
            raise ValueError("unknown belief cannot be accepted as active")
        if current.identity_origin.actor in {
            OriginActor.UNKNOWN,
            OriginActor.INHERITED,
        }:
            if reviewer_authority not in {"subject", "operator"} or not reviewer_id:
                raise ValueError(
                    "unknown-origin belief requires an authorized reviewer"
                )
            _safe_ref(reviewer_id, "belief reviewer ID")
        origin = (
            current.identity_origin.endorse(
                "belief_review", event_id=event_id, event_sequence=event_sequence
            )
            if accept
            else current.identity_origin.reject(
                "belief_rejected", event_id=event_id, event_sequence=event_sequence
            )
        )
        lifecycle = BeliefLifecycle.ACTIVE if accept else BeliefLifecycle.REJECTED
        if accept and current.contradiction_ids:
            lifecycle = BeliefLifecycle.DISPUTED
        updated = replace(
            current,
            confidence=confidence,
            epistemic_status=epistemic_status,
            lifecycle=lifecycle,
            identity_origin=origin,
            updated_at=_now(),
        )
        resolved = self._with_revision(
            current,
            updated,
            operation="accept" if accept else "reject",
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=event_id,
            event_sequence=event_sequence,
            reviewer_id=reviewer_id,
            reviewer_authority=reviewer_authority,
        )
        self.records[belief_id] = resolved
        if not accept:
            self._release_contradictions(
                resolved,
                reason_code=reason_code,
                evidence_refs=evidence_refs,
                event_id=event_id,
                event_sequence=event_sequence,
            )
        return resolved

    def supersede(
        self,
        old_belief_id: str,
        new_belief_id: str,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> tuple[BeliefRecord, BeliefRecord]:
        if not evidence_refs:
            raise ValueError("belief supersession requires evidence references")
        old = self.get(old_belief_id)
        new = self.get(new_belief_id)
        if new.lifecycle not in {BeliefLifecycle.ACTIVE, BeliefLifecycle.DISPUTED}:
            raise ValueError("Superseding belief must be reviewed")
        if new.identity_origin.endorsement != EndorsementStatus.ENDORSED:
            raise ValueError("Superseding belief must be reviewed")
        updated_old = self._with_revision(
            old,
            replace(
                old,
                lifecycle=BeliefLifecycle.SUPERSEDED,
                superseded_by_id=new_belief_id,
                updated_at=_now(),
            ),
            operation="supersede",
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )
        remaining_contradictions = tuple(
            identifier
            for identifier in new.contradiction_ids
            if identifier != old_belief_id
            and self.get(identifier).lifecycle
            not in {
                BeliefLifecycle.SUPERSEDED,
                BeliefLifecycle.RETRACTED,
                BeliefLifecycle.EXPIRED,
                BeliefLifecycle.REJECTED,
            }
        )
        updated_new = self._with_revision(
            new,
            replace(
                new,
                lifecycle=BeliefLifecycle.ACTIVE
                if not remaining_contradictions
                else BeliefLifecycle.DISPUTED,
                contradiction_ids=remaining_contradictions,
                supersedes_id=old_belief_id,
                updated_at=_now(),
            ),
            operation="supersedes",
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )
        self.records[old_belief_id] = updated_old
        self.records[new_belief_id] = updated_new
        return updated_old, updated_new

    def retract(
        self,
        belief_id: str,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> BeliefRecord:
        if not evidence_refs:
            raise ValueError("belief retraction requires evidence references")
        current = self.get(belief_id)
        updated = self._with_revision(
            current,
            replace(current, lifecycle=BeliefLifecycle.RETRACTED, updated_at=_now()),
            operation="retract",
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )
        self.records[belief_id] = updated
        self._release_contradictions(
            updated,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )
        return updated

    def expire(self, now: datetime | None = None) -> list[BeliefRecord]:
        current_time = now or datetime.now(UTC)
        expired: list[BeliefRecord] = []
        for record in list(self.records.values()):
            if record.lifecycle == BeliefLifecycle.ACTIVE and not _temporally_valid(
                record, current_time
            ):
                updated = self._with_revision(
                    record,
                    replace(
                        record, lifecycle=BeliefLifecycle.EXPIRED, updated_at=_now()
                    ),
                    operation="expire",
                    reason_code="validity_window_ended",
                    evidence_refs=(),
                    event_id=None,
                    event_sequence=None,
                )
                self.records[record.belief_id] = updated
                expired.append(updated)
        return expired

    def active(
        self, *, context_id: str | None = None, now: datetime | None = None
    ) -> list[BeliefRecord]:
        current_time = now or datetime.now(UTC)
        return [
            record
            for record in sorted(self.records.values(), key=lambda item: item.belief_id)
            if record.lifecycle == BeliefLifecycle.ACTIVE
            and record.identity_origin.endorsement == EndorsementStatus.ENDORSED
            and not (
                record.identity_origin.actor
                in {OriginActor.UNKNOWN, OriginActor.INHERITED}
                and not any(revision.reviewer_id for revision in record.revisions)
            )
            and _temporally_valid(record, current_time)
            and (
                not record.context_scope
                or context_id is not None
                and context_id in record.context_scope
            )
        ]

    def get(self, belief_id: str) -> BeliefRecord:
        record = self.records.get(belief_id)
        if record is None:
            raise ValueError(f"Unknown belief: {belief_id}")
        return record

    def list_records(self) -> list[BeliefRecord]:
        return sorted(self.records.values(), key=lambda item: item.belief_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "records": [record.to_json() for record in self.list_records()],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.records = {}
            return
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported belief store schema version: {payload.get('schema_version')}"
            )
        records = [
            _belief_from_json(item)
            for item in payload.get("records", [])
            if isinstance(item, dict)
        ]
        if len(records) != len({record.belief_id for record in records}):
            raise ValueError("Belief identifiers must be unique")
        self.records = {record.belief_id: record for record in records}

    def _with_revision(
        self,
        before: BeliefRecord,
        after: BeliefRecord,
        *,
        operation: str,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
        reviewer_id: str | None = None,
        reviewer_authority: str | None = None,
    ) -> BeliefRecord:
        revision = BeliefRevision(
            revision_id=f"belief-revision-{uuid4()}",
            from_revision=before.revision,
            to_revision=before.revision + 1,
            operation=operation,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            before=_state(before),
            after=_state(after),
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
            reviewer_id=reviewer_id,
            reviewer_authority=reviewer_authority,
        )
        return replace(
            after,
            revision=revision.to_revision,
            revisions=(*before.revisions, revision),
        )

    def _release_contradictions(
        self,
        removed: BeliefRecord,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> None:
        for belief_id in removed.contradiction_ids:
            other = self.get(belief_id)
            remaining = tuple(
                identifier
                for identifier in other.contradiction_ids
                if identifier != removed.belief_id
                and self.get(identifier).lifecycle
                not in {
                    BeliefLifecycle.SUPERSEDED,
                    BeliefLifecycle.RETRACTED,
                    BeliefLifecycle.EXPIRED,
                    BeliefLifecycle.REJECTED,
                }
            )
            lifecycle = BeliefLifecycle.DISPUTED
            if not remaining:
                lifecycle = (
                    BeliefLifecycle.ACTIVE
                    if other.identity_origin.endorsement == EndorsementStatus.ENDORSED
                    else BeliefLifecycle.PROPOSED
                )
            updated = self._with_revision(
                other,
                replace(
                    other,
                    lifecycle=lifecycle,
                    contradiction_ids=remaining,
                    updated_at=_now(),
                ),
                operation="contradiction_resolved",
                reason_code=reason_code,
                evidence_refs=evidence_refs,
                event_id=event_id,
                event_sequence=event_sequence,
            )
            self.records[belief_id] = updated


def _state(record: BeliefRecord) -> dict[str, Any]:
    return {
        "confidence": record.confidence,
        "epistemic_status": record.epistemic_status.value,
        "lifecycle": record.lifecycle.value,
        "contradiction_ids": list(record.contradiction_ids),
        "supersedes_id": record.supersedes_id,
        "superseded_by_id": record.superseded_by_id,
    }


def _contradicts(left: BeliefRecord, right: BeliefRecord) -> bool:
    a = left.proposition
    b = right.proposition
    return (
        a.subject is not None
        and a.subject == b.subject
        and a.predicate == b.predicate
        and a.object != b.object
        and bool(
            set(left.context_scope).intersection(right.context_scope)
            if left.context_scope and right.context_scope
            else True
        )
    )


def _temporally_valid(record: BeliefRecord, now: datetime) -> bool:
    valid_from = _parse_time(record.valid_from)
    valid_until = _parse_time(record.valid_until)
    return (valid_from is None or valid_from <= now) and (
        valid_until is None or now < valid_until
    )


def _belief_from_json(payload: dict[str, Any]) -> BeliefRecord:
    data = dict(payload)
    data["proposition"] = Proposition(**data["proposition"])
    data["identity_origin"] = identity_origin_from_json(data.get("identity_origin"))
    data["epistemic_status"] = EpistemicStatus(data["epistemic_status"])
    data["lifecycle"] = BeliefLifecycle(data["lifecycle"])
    data["evidence"] = tuple(
        BeliefEvidence(**item) for item in data.get("evidence", ())
    )
    for name in ("context_scope", "contradiction_ids"):
        data[name] = tuple(data.get(name, ()))
    data["revisions"] = tuple(
        BeliefRevision(
            **{
                **item,
                "evidence_refs": tuple(item.get("evidence_refs", ())),
            }
        )
        for item in data.get("revisions", ())
    )
    if (
        data["lifecycle"] == BeliefLifecycle.ACTIVE
        and data["identity_origin"].actor
        in {OriginActor.UNKNOWN, OriginActor.INHERITED}
        and not any(revision.reviewer_id for revision in data["revisions"])
    ):
        data["lifecycle"] = BeliefLifecycle.PROPOSED
        data["identity_origin"] = replace(
            data["identity_origin"],
            endorsement=EndorsementStatus.UNCERTAIN,
            endorsement_ref=None,
            endorsed_by_event_id=None,
            endorsed_by_event_sequence=None,
        )
    return BeliefRecord(**data)


def _normalized_component(value: str | None) -> str | None:
    return None if value is None else " ".join(value.strip().lower().split())


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("belief validity timestamps must include a timezone")
    return parsed


def _safe_ref(value: str, name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", value) is None:
        raise ValueError(f"{name} must be an opaque safe reference")


def _unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()
