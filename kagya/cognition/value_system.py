"""Persistent, rate-limited values and structured action evaluation."""

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any, Iterable
from uuid import uuid4

from kagya.cognition.appraisal import AppraisalResult
from kagya.identity import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    identity_origin_from_json,
    legacy_identity_origin,
    new_identity_origin,
)


class ValueUpdateKind(StrEnum):
    OBSERVATION = "observation"
    OUTCOME = "outcome"
    REFLECTION = "reflection"
    ADMIN = "admin"


@dataclass(frozen=True)
class ValueState:
    value_id: str
    name: str
    weight: float
    confidence: float
    stability: float
    source: str
    origin: str
    last_updated_at: str
    allowed_update_rate: float
    origin_provenance: IdentityOrigin | None = None
    revision: int = 0
    frozen: bool = False
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.value_id or not self.name:
            raise ValueError("value_id and name must not be empty")
        for field_name in ("weight", "confidence", "stability"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and between zero and one")
        if not math.isfinite(self.allowed_update_rate) or self.allowed_update_rate <= 0:
            raise ValueError("allowed_update_rate must be finite and greater than zero")
        if self.schema_version not in {1, 2}:
            raise ValueError(f"Unsupported value schema version: {self.schema_version}")
        if self.origin_provenance is None:
            object.__setattr__(
                self,
                "origin_provenance",
                legacy_identity_origin(self.origin),
            )
        if self.schema_version == 1:
            object.__setattr__(self, "schema_version", 2)


@dataclass(frozen=True)
class ValueConflictDefinition:
    left_value_id: str
    right_value_id: str
    name: str


@dataclass(frozen=True)
class ValueEvidence:
    event_id: str | None = None
    event_sequence: int | None = None
    memory_ids: tuple[str, ...] = ()
    source: str = "runtime"
    identity_origin: IdentityOrigin = field(
        default_factory=lambda: legacy_identity_origin("value_evidence")
    )


@dataclass(frozen=True)
class ValueUpdateProposal:
    proposal_id: str
    kind: ValueUpdateKind
    value_id: str
    requested_delta: float
    confidence: float
    reason_codes: tuple[str, ...]
    evidence: ValueEvidence


@dataclass(frozen=True)
class ValueUpdateRecord:
    update_id: str
    operation: str
    value_id: str
    event_id: str | None
    event_sequence: int | None
    memory_ids: tuple[str, ...]
    kind: str
    reason_codes: tuple[str, ...]
    identity_origin: IdentityOrigin
    requested_delta: float
    applied_delta: float
    before: dict[str, Any]
    after: dict[str, Any]
    created_at: str
    rollback_target_revision: int | None = None


@dataclass(frozen=True)
class ValueContribution:
    value_id: str
    effect: float
    weight: float
    confidence: float
    contribution: float


@dataclass(frozen=True)
class ActionScore:
    option_id: str
    total_score: float
    contributions: tuple[ValueContribution, ...]
    conflicts: tuple[str, ...]


class ValueSystem:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        seeds: Iterable[ValueState],
        conflicts: Iterable[ValueConflictDefinition] = (),
        max_update_per_event: float = 0.05,
        max_total_update_per_event: float = 0.1,
    ) -> None:
        seed_values = tuple(seeds)
        self._seeds = {seed.value_id: seed for seed in seed_values}
        if len(self._seeds) != len(seed_values):
            raise ValueError("Value seed identifiers must be unique")
        self.values = dict(self._seeds)
        self.conflicts = tuple(conflicts)
        for conflict in self.conflicts:
            if conflict.left_value_id == conflict.right_value_id:
                raise ValueError("A value cannot conflict with itself")
            if conflict.left_value_id not in self.values:
                raise ValueError(f"Unknown value: {conflict.left_value_id}")
            if conflict.right_value_id not in self.values:
                raise ValueError(f"Unknown value: {conflict.right_value_id}")
        self.history: list[ValueUpdateRecord] = []
        self.applied_proposals: set[str] = set()
        self.max_update_per_event = max_update_per_event
        self.max_total_update_per_event = max_total_update_per_event

    def proposals_from_appraisal(
        self,
        appraisal: AppraisalResult,
        impacts: dict[str, float],
        *,
        kind: ValueUpdateKind,
        evidence: ValueEvidence,
        proposal_id: str | None = None,
    ) -> list[ValueUpdateProposal]:
        confidence = appraisal.certainty
        if not appraisal.novelty_valid and kind == ValueUpdateKind.OBSERVATION:
            confidence *= 0.5
        reason = {
            ValueUpdateKind.OBSERVATION: "observation_appraised",
            ValueUpdateKind.OUTCOME: "outcome_appraised",
            ValueUpdateKind.REFLECTION: "reflection_appraised",
            ValueUpdateKind.ADMIN: "admin_requested",
        }[kind]
        unknown = set(impacts) - self.values.keys()
        if unknown:
            raise ValueError(f"Unknown value: {sorted(unknown)[0]}")
        if any(not math.isfinite(impact) for impact in impacts.values()):
            raise ValueError("Value impacts must be finite")
        return [
            ValueUpdateProposal(
                proposal_id=(
                    str(uuid4())
                    if proposal_id is None
                    else f"{proposal_id}:{value_id}"
                ),
                kind=kind,
                value_id=value_id,
                requested_delta=max(-1.0, min(1.0, impact)),
                confidence=confidence,
                reason_codes=(reason, *appraisal.reasons),
                evidence=evidence,
            )
            for value_id, impact in impacts.items()
        ]

    def apply(self, proposals: Iterable[ValueUpdateProposal]) -> list[ValueUpdateRecord]:
        grouped: dict[str, list[ValueUpdateProposal]] = {}
        for proposal in proposals:
            if proposal.proposal_id in self.applied_proposals:
                continue
            if proposal.evidence.identity_origin.input_kind in {
                OriginInputKind.REQUEST,
                OriginInputKind.SUGGESTION,
            }:
                raise ValueError(
                    "Requests and suggestions cannot directly update subject values"
                )
            self._require(proposal.value_id)
            grouped.setdefault(proposal.value_id, []).append(proposal)
        records: list[ValueUpdateRecord] = []
        remaining_budget = self.max_total_update_per_event
        for value_id in sorted(grouped):
            candidates = grouped[value_id]
            for proposal in candidates:
                self.applied_proposals.add(proposal.proposal_id)
            state = self._require(value_id)
            requested = sum(
                proposal.requested_delta * proposal.confidence
                for proposal in candidates
            )
            cap = min(self.max_update_per_event, state.allowed_update_rate)
            cap *= 1.0 - state.stability
            cap = min(cap, remaining_budget)
            applied = 0.0 if state.frozen else max(-cap, min(cap, requested))
            remaining_budget = max(0.0, remaining_budget - abs(applied))
            before = _revision_snapshot(state)
            after = state
            if not state.frozen:
                after = replace(
                    state,
                    weight=max(0.0, min(1.0, state.weight + applied)),
                    confidence=max(
                        0.0,
                        min(
                            1.0,
                            state.confidence
                            + 0.05
                            * (
                                sum(p.confidence for p in candidates) / len(candidates)
                                - state.confidence
                            ),
                        ),
                    ),
                    source=candidates[-1].evidence.source,
                    last_updated_at=_now(),
                    revision=state.revision + 1,
                )
            self.values[value_id] = after
            record = ValueUpdateRecord(
                update_id=str(uuid4()),
                operation="rejected" if state.frozen else "update",
                value_id=value_id,
                event_id=candidates[-1].evidence.event_id,
                event_sequence=candidates[-1].evidence.event_sequence,
                memory_ids=candidates[-1].evidence.memory_ids,
                kind=candidates[-1].kind.value,
                reason_codes=("value_frozen",)
                if state.frozen
                else tuple(code for item in candidates for code in item.reason_codes),
                identity_origin=candidates[-1].evidence.identity_origin,
                requested_delta=requested,
                applied_delta=applied,
                before=before,
                after=_revision_snapshot(after),
                created_at=_now(),
            )
            self.history.append(record)
            records.append(record)
        return records

    def freeze(self, value_id: str, frozen: bool) -> ValueState:
        state = self._require(value_id)
        updated = replace(
            state,
            frozen=frozen,
            revision=state.revision + 1,
            last_updated_at=_now(),
            source="admin",
        )
        self.values[value_id] = updated
        self._administrative_record("freeze" if frozen else "unfreeze", state, updated)
        return updated

    def rollback(self, value_id: str, target_revision: int) -> ValueState:
        current = self._require(value_id)
        snapshots = [
            record.before
            for record in self.history
            if record.value_id == value_id
            and int(record.before.get("revision", -1)) == target_revision
        ]
        snapshots.extend(
            record.after
            for record in self.history
            if record.value_id == value_id
            and int(record.after.get("revision", -1)) == target_revision
        )
        if not snapshots:
            raise ValueError(f"Unknown value revision: {target_revision}")
        target = snapshots[-1]
        restored = replace(
            current,
            weight=float(target["weight"]),
            confidence=float(target["confidence"]),
            stability=float(target["stability"]),
            frozen=bool(target["frozen"]),
            revision=current.revision + 1,
            source="admin.rollback",
            last_updated_at=_now(),
        )
        self.values[value_id] = restored
        self._administrative_record(
            "rollback", current, restored, rollback_target_revision=target_revision
        )
        return restored

    def reset(self, value_ids: tuple[str, ...] | None = None) -> list[ValueState]:
        before = dict(self.values)
        selected = tuple(self._seeds) if value_ids is None else value_ids
        if value_ids is None:
            self.values = {}
        for value_id in selected:
            seed = self._seeds.get(value_id)
            if seed is None:
                raise ValueError(f"Value has no configured seed: {value_id}")
            self.values[value_id] = replace(
                seed,
                revision=before.get(value_id, seed).revision + 1,
                last_updated_at=_now(),
            )
        for value_id in selected:
            state = self.values[value_id]
            self._administrative_record("reset", before.get(value_id, state), state)
        return [self.values[value_id] for value_id in selected]

    def evaluate(self, options: dict[str, dict[str, float]]) -> list[ActionScore]:
        scores: list[ActionScore] = []
        for option_id, effects in options.items():
            unknown = set(effects) - self.values.keys()
            if unknown:
                raise ValueError(f"Unknown value: {sorted(unknown)[0]}")
            if any(not math.isfinite(effect) for effect in effects.values()):
                raise ValueError("Value effects must be finite")
            contributions = tuple(
                ValueContribution(
                    value_id=value_id,
                    effect=max(-1.0, min(1.0, effect)),
                    weight=self.values[value_id].weight,
                    confidence=self.values[value_id].confidence,
                    contribution=max(-1.0, min(1.0, effect))
                    * self.values[value_id].weight
                    * self.values[value_id].confidence,
                )
                for value_id, effect in sorted(effects.items())
                if value_id in self.values
            )
            contribution_map = {item.value_id: item.contribution for item in contributions}
            conflicts = tuple(
                definition.name
                for definition in self.conflicts
                if contribution_map.get(definition.left_value_id, 0.0)
                * contribution_map.get(definition.right_value_id, 0.0)
                < 0
            )
            scores.append(
                ActionScore(
                    option_id=option_id,
                    total_score=sum(item.contribution for item in contributions),
                    contributions=contributions,
                    conflicts=conflicts,
                )
            )
        return scores

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "values": {key: asdict(value) for key, value in self.values.items()},
            "conflicts": [asdict(item) for item in self.conflicts],
            "history": [asdict(item) for item in self.history],
            "applied_proposals": sorted(self.applied_proposals),
        }

    def get(self, value_id: str) -> ValueState:
        return self._require(value_id)

    def list_values(self) -> list[ValueState]:
        return [self.values[value_id] for value_id in sorted(self.values)]

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.values = dict(self._seeds)
            self.history = []
            self.applied_proposals = set()
            return
        if payload.get("schema_version") != 1:
            if isinstance(payload, dict) and payload and all(
                isinstance(value, (int, float)) for value in payload.values()
            ):
                for key, weight in payload.items():
                    if key in self.values:
                        self.values[key] = replace(self.values[key], weight=float(weight))
                    else:
                        self.values[key] = ValueState(
                            value_id=key,
                            name=key.replace("_", " ").title(),
                            weight=float(weight),
                            confidence=0.5,
                            stability=0.5,
                            source="legacy_snapshot",
                            origin="legacy_snapshot",
                            last_updated_at=_now(),
                            allowed_update_rate=self.max_update_per_event,
                        )
            return
        values = payload.get("values")
        if isinstance(values, dict):
            self.values = {
                key: _value_state_from_json(value)
                for key, value in values.items()
                if isinstance(key, str) and isinstance(value, dict)
            }
        self.history = [
            _value_update_record_from_json(record)
            for record in payload.get("history", [])
            if isinstance(record, dict)
        ]
        self.applied_proposals = set(payload.get("applied_proposals", []))
    def _require(self, value_id: str) -> ValueState:
        state = self.values.get(value_id)
        if state is None:
            raise ValueError(f"Unknown value: {value_id}")
        return state

    def _administrative_record(
        self,
        operation: str,
        before: ValueState,
        after: ValueState,
        *,
        rollback_target_revision: int | None = None,
    ) -> None:
        self.history.append(
            ValueUpdateRecord(
                update_id=str(uuid4()),
                operation=operation,
                value_id=after.value_id,
                event_id=None,
                event_sequence=None,
                memory_ids=(),
                kind=ValueUpdateKind.ADMIN.value,
                reason_codes=(operation,),
                identity_origin=new_identity_origin(
                    OriginActor.OPERATOR,
                    OriginInputKind.CONSTRAINT,
                    source_ref="value_admin",
                ),
                requested_delta=0.0,
                applied_delta=after.weight - before.weight,
                before=_revision_snapshot(before),
                after=_revision_snapshot(after),
                created_at=_now(),
                rollback_target_revision=rollback_target_revision,
            )
        )


def _value_state_from_json(payload: dict[str, Any]) -> ValueState:
    data = dict(payload)
    data["origin_provenance"] = identity_origin_from_json(
        data.get("origin_provenance"),
        fallback_source=str(data.get("origin", "legacy_value")),
    )
    data["schema_version"] = 2
    return ValueState(**data)


def _value_update_record_from_json(payload: dict[str, Any]) -> ValueUpdateRecord:
    data = dict(payload)
    data["memory_ids"] = tuple(data.get("memory_ids", ()))
    data["reason_codes"] = tuple(data.get("reason_codes", ()))
    data["identity_origin"] = identity_origin_from_json(
        data.get("identity_origin"), fallback_source="legacy_value_update"
    )
    return ValueUpdateRecord(**data)


def _revision_snapshot(state: ValueState) -> dict[str, Any]:
    return {
        "revision": state.revision,
        "weight": state.weight,
        "confidence": state.confidence,
        "stability": state.stability,
        "frozen": state.frozen,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
