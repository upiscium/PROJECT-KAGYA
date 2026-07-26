"""Persistent, rate-limited values and structured action evaluation."""

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
import re
import hashlib
import json
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

from kagya.cognition.appraisal import AppraisalResult
from kagya.cognition.value_records import (
    EvidenceDirection,
    ValueEvidenceRecord,
    ValueReassessmentRecord,
    ValueRevisionDiff,
    ValueScope,
    ValueTradeoffRecord,
)
from kagya.identity import (
    EndorsementStatus,
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    identity_origin_from_json,
    legacy_identity_origin,
    new_identity_origin,
)

if TYPE_CHECKING:
    from kagya.decision import DecisionRecord
    from kagya.experience import ExperienceRecord


class ValueUpdateKind(StrEnum):
    OBSERVATION = "observation"
    OUTCOME = "outcome"
    REFLECTION = "reflection"
    ADMIN = "admin"


@dataclass(frozen=True)
class OriginReviewRecord:
    value_id: str
    origin_id: str
    origin_actor: str
    origin_input_kind: str
    origin_source_ref: str | None
    reviewed_revision: int
    reviewed_state_digest: str
    reviewer_id: str
    reviewer_authority: str
    decision: str
    evidence_refs: tuple[str, ...]
    reason_code: str
    event_id: str
    event_sequence: int
    record_hash: str

    def __post_init__(self) -> None:
        if self.reviewer_authority not in {"subject", "operator"}:
            raise ValueError("Origin review requires allowed reviewer authority")
        if self.decision not in {"accept", "reject"}:
            raise ValueError("Origin review decision is invalid")
        if not self.event_id or self.event_sequence < 1:
            raise ValueError("Origin review requires an AgentRuntime event")
        if self.record_hash != _origin_review_hash(
            self.value_id,
            self.origin_id,
            self.origin_actor,
            self.origin_input_kind,
            self.origin_source_ref,
            self.reviewed_revision,
            self.reviewed_state_digest,
            self.reviewer_id,
            self.reviewer_authority,
            self.decision,
            self.evidence_refs,
            self.reason_code,
            self.event_id,
            self.event_sequence,
        ):
            raise ValueError("Origin review record hash mismatch")


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
    concept: str | None = None
    scope: ValueScope = ValueScope.SUBJECT
    context_ids: tuple[str, ...] = ()
    polarity: int = 1
    protectedness: float = 0.0
    negotiability: float = 1.0
    origin_experience_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    opposing_evidence_ids: tuple[str, ...] = ()
    related_drive_ids: tuple[str, ...] = ()
    related_goal_ids: tuple[str, ...] = ()
    related_commitment_ids: tuple[str, ...] = ()
    last_reinforced_at: str | None = None
    last_challenged_at: str | None = None
    last_reviewed_at: str | None = None
    change_reason: str | None = None
    origin_review: OriginReviewRecord | None = None
    schema_version: int = 3

    def __post_init__(self) -> None:
        if not self.value_id or not self.name:
            raise ValueError("value_id and name must not be empty")
        for field_name in ("weight", "confidence", "stability"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be finite and between zero and one"
                )
        if not math.isfinite(self.allowed_update_rate) or self.allowed_update_rate <= 0:
            raise ValueError("allowed_update_rate must be finite and greater than zero")
        if self.schema_version not in {1, 2, 3}:
            raise ValueError(f"Unsupported value schema version: {self.schema_version}")
        if self.origin_provenance is None:
            object.__setattr__(
                self,
                "origin_provenance",
                legacy_identity_origin(self.origin),
            )
        if self.schema_version == 1:
            object.__setattr__(self, "schema_version", 3)
        if self.schema_version == 2:
            object.__setattr__(self, "schema_version", 3)
        if self.concept is None:
            object.__setattr__(self, "concept", self.name)
        if self.polarity not in {-1, 1}:
            raise ValueError("polarity must be minus one or one")
        for field_name in ("protectedness", "negotiability"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be finite and between zero and one"
                )
        if self.scope == ValueScope.CONTEXT and not self.context_ids:
            raise ValueError("Context-scoped values require context IDs")
        if self.origin_review is not None:
            provenance = self.origin_provenance
            assert provenance is not None
            if (
                self.origin_review.value_id != self.value_id
                or self.origin_review.reviewed_revision >= self.revision
                or self.origin_review.origin_id != provenance.origin_id
                or self.origin_review.origin_actor != provenance.actor.value
                or self.origin_review.origin_input_kind != provenance.input_kind.value
                or self.origin_review.origin_source_ref != provenance.source_ref
            ):
                raise ValueError("Origin review is not bound to this Value state")


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
    evidence_id: str | None = None
    experience_ids: tuple[str, ...] = ()
    decision_id: str | None = None
    context_id: str | None = None
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
    evidence_ids: tuple[str, ...] = ()
    revision_diff: ValueRevisionDiff | None = None
    reviewer_id: str | None = None
    reviewer_authority: str | None = None


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
    SCHEMA_VERSION = 3

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
        self.evidence: dict[str, ValueEvidenceRecord] = {}
        self.tradeoffs: list[ValueTradeoffRecord] = []
        self.reassessments: list[ValueReassessmentRecord] = []
        self.held_proposals: dict[str, ValueUpdateProposal] = {}
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
                    str(uuid4()) if proposal_id is None else f"{proposal_id}:{value_id}"
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

    def proposals_from_experience(
        self,
        experience: "ExperienceRecord",
        impacts: dict[str, float],
        *,
        proposal_id: str | None = None,
    ) -> list[ValueUpdateProposal]:
        appraisal = AppraisalResult(
            novelty=experience.appraisal.novelty,
            goal_progress=experience.appraisal.goal_progress,
            threat=experience.appraisal.threat,
            controllability=experience.appraisal.controllability,
            certainty=experience.appraisal.certainty,
            social_relevance=experience.appraisal.social_relevance,
            effort_cost=experience.appraisal.effort_cost,
            novelty_valid=experience.appraisal.novelty_valid,
            reasons=("experience_value_evidence", *experience.appraisal.reason_codes),
        )
        return self.proposals_from_appraisal(
            appraisal,
            impacts,
            kind=ValueUpdateKind.OBSERVATION,
            evidence=ValueEvidence(
                event_id=experience.source_event_id,
                event_sequence=experience.source_event_sequence,
                experience_ids=(experience.experience_id,),
                context_id=experience.context_id,
                source="experience.value_evidence",
                identity_origin=experience.identity_origin,
            ),
            proposal_id=proposal_id or f"experience:{experience.experience_id}",
        )

    def proposals_from_decision_outcome(
        self,
        decision: "DecisionRecord",
        *,
        experience_ids: tuple[str, ...] = (),
    ) -> list[ValueUpdateProposal]:
        if decision.actual_outcome is None or decision.prediction_error is None:
            raise ValueError("Value reassessment requires a resolved decision outcome")
        selected = next(
            item
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        outcome = decision.actual_outcome
        outcome_signal = outcome.utility
        if not outcome.success:
            outcome_signal = min(outcome_signal, -0.25)
        impacts = {
            value_id: effect * outcome_signal
            for value_id, effect in selected.candidate.value_effects.items()
        }
        if not impacts:
            return []
        return [
            ValueUpdateProposal(
                proposal_id=f"decision:{decision.decision_id}:{value_id}",
                kind=ValueUpdateKind.OUTCOME,
                value_id=value_id,
                requested_delta=max(-1.0, min(1.0, impact)),
                confidence=max(0.2, min(1.0, abs(decision.prediction_error))),
                reason_codes=(
                    "decision_outcome_reassessment",
                    "regret_signal"
                    if decision.prediction_error < 0
                    else "expectation_confirmed",
                ),
                evidence=ValueEvidence(
                    event_id=outcome.observed_event_id,
                    event_sequence=outcome.observed_event_sequence,
                    decision_id=decision.decision_id,
                    experience_ids=experience_ids,
                    context_id=decision.context_id,
                    source="decision.outcome",
                ),
            )
            for value_id, impact in impacts.items()
        ]

    def apply(
        self, proposals: Iterable[ValueUpdateProposal]
    ) -> list[ValueUpdateRecord]:
        grouped: dict[str, list[ValueUpdateProposal]] = {}
        for proposal in proposals:
            if proposal.proposal_id in self.applied_proposals:
                continue
            self._require(proposal.value_id)
            if not self._is_active(self._require(proposal.value_id)):
                self.held_proposals[proposal.proposal_id] = proposal
                continue
            grouped.setdefault(proposal.value_id, []).append(proposal)
        records: list[ValueUpdateRecord] = []
        remaining_budget = self.max_total_update_per_event
        for value_id in sorted(grouped):
            candidates = grouped[value_id]
            for proposal in candidates:
                self.applied_proposals.add(proposal.proposal_id)
            state = self._require(value_id)
            known_evidence_ids = set(
                (*state.supporting_evidence_ids, *state.opposing_evidence_ids)
            )
            novel = [
                (proposal, self._record_evidence(proposal)) for proposal in candidates
            ]
            novel = [
                item for item in novel if item[1].evidence_id not in known_evidence_ids
            ]
            if not novel:
                continue
            candidates = [item[0] for item in novel]
            evidence_records = [item[1] for item in novel]
            direct_evidence = [
                proposal
                for proposal in candidates
                if proposal.evidence.identity_origin.input_kind
                not in {OriginInputKind.REQUEST, OriginInputKind.SUGGESTION}
            ]
            if not direct_evidence:
                records.append(
                    self._record_evidence_only_update(
                        state, candidates, evidence_records
                    )
                )
                continue
            requested = sum(
                proposal.requested_delta * proposal.confidence
                for proposal in direct_evidence
            )
            cap = min(self.max_update_per_event, state.allowed_update_rate)
            cap *= 1.0 - state.stability
            cap *= state.negotiability * (1.0 - 0.8 * state.protectedness)
            direction = (
                EvidenceDirection.SUPPORTING
                if requested >= 0
                else EvidenceDirection.OPPOSING
            )
            prior_count = len(
                state.supporting_evidence_ids
                if direction == EvidenceDirection.SUPPORTING
                else state.opposing_evidence_ids
            )
            reinforcement = min(1.0, 0.35 + 0.15 * prior_count)
            cap = min(cap * reinforcement, remaining_budget)
            applied = 0.0 if state.frozen else max(-cap, min(cap, requested))
            remaining_budget = max(0.0, remaining_budget - abs(applied))
            before = _revision_snapshot(state)
            after = state
            if not state.frozen:
                now = _now()
                evidence_ids = tuple(item.evidence_id for item in evidence_records)
                supporting_ids = tuple(
                    item.evidence_id
                    for item in evidence_records
                    if item.direction == EvidenceDirection.SUPPORTING
                )
                opposing_ids = tuple(
                    item.evidence_id
                    for item in evidence_records
                    if item.direction == EvidenceDirection.OPPOSING
                )
                next_strength = state.weight + applied
                next_polarity = state.polarity
                if next_strength < 0.0:
                    reversal_threshold = 3 + math.ceil(3 * state.protectedness)
                    if prior_count + len(evidence_records) < reversal_threshold:
                        next_strength = 0.0
                    else:
                        next_strength = abs(next_strength)
                        next_polarity = -state.polarity
                actual_delta = -state.weight if next_strength == 0.0 else applied
                after = replace(
                    state,
                    weight=max(0.0, min(1.0, next_strength)),
                    polarity=next_polarity,
                    confidence=max(
                        0.0,
                        min(
                            1.0,
                            state.confidence
                            + (0.05 if requested >= 0 else -0.05)
                            * sum(p.confidence for p in direct_evidence)
                            / len(direct_evidence),
                        ),
                    ),
                    source=candidates[-1].evidence.source,
                    origin_experience_ids=tuple(
                        dict.fromkeys(
                            (
                                *state.origin_experience_ids,
                                *(
                                    item
                                    for record in evidence_records
                                    for item in record.experience_ids
                                ),
                            )
                        )
                    ),
                    supporting_evidence_ids=(
                        *state.supporting_evidence_ids,
                        *supporting_ids,
                    ),
                    opposing_evidence_ids=(
                        *state.opposing_evidence_ids,
                        *opposing_ids,
                    ),
                    last_reinforced_at=now
                    if direction == EvidenceDirection.SUPPORTING
                    else state.last_reinforced_at,
                    last_challenged_at=now
                    if direction == EvidenceDirection.OPPOSING
                    else state.last_challenged_at,
                    last_reviewed_at=now,
                    change_reason=candidates[-1].reason_codes[0],
                    last_updated_at=now,
                    revision=state.revision + 1,
                )
                applied = actual_delta
            self.values[value_id] = after
            reason_codes = (
                ("value_frozen",)
                if state.frozen
                else tuple(code for item in candidates for code in item.reason_codes)
            )
            evidence_ids = tuple(item.evidence_id for item in evidence_records)
            record = ValueUpdateRecord(
                update_id=str(uuid4()),
                operation="rejected" if state.frozen else "update",
                value_id=value_id,
                event_id=candidates[-1].evidence.event_id,
                event_sequence=candidates[-1].evidence.event_sequence,
                memory_ids=candidates[-1].evidence.memory_ids,
                kind=candidates[-1].kind.value,
                reason_codes=reason_codes,
                identity_origin=candidates[-1].evidence.identity_origin,
                requested_delta=requested,
                applied_delta=applied,
                before=before,
                after=_revision_snapshot(after),
                created_at=_now(),
                evidence_ids=evidence_ids,
                revision_diff=ValueRevisionDiff(
                    from_revision=state.revision,
                    to_revision=after.revision,
                    changed_fields=_snapshot_diff(before, _revision_snapshot(after)),
                    reason_codes=reason_codes,
                    evidence_ids=evidence_ids,
                ),
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
            polarity=int(target.get("polarity", current.polarity)),
            protectedness=float(target.get("protectedness", current.protectedness)),
            negotiability=float(target.get("negotiability", current.negotiability)),
            scope=ValueScope(target.get("scope", current.scope)),
            context_ids=tuple(target.get("context_ids", current.context_ids)),
            supporting_evidence_ids=tuple(
                target.get("supporting_evidence_ids", current.supporting_evidence_ids)
            ),
            opposing_evidence_ids=tuple(
                target.get("opposing_evidence_ids", current.opposing_evidence_ids)
            ),
            change_reason="rollback",
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

    def evaluate(
        self, options: dict[str, dict[str, float]], *, context_id: str | None = None
    ) -> list[ActionScore]:
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
                and self._is_active(self.values[value_id])
                and self._applies(self.values[value_id], context_id)
            )
            contribution_map = {
                item.value_id: item.contribution for item in contributions
            }
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

    def record_tradeoffs(
        self,
        scores: Iterable[ActionScore],
        *,
        context_id: str | None,
        decision_id: str | None = None,
    ) -> list[ValueTradeoffRecord]:
        records: list[ValueTradeoffRecord] = []
        for score in scores:
            if not score.conflicts:
                continue
            contributions = {
                item.value_id: item.contribution for item in score.contributions
            }
            record = ValueTradeoffRecord(
                tradeoff_id=str(uuid4()),
                value_ids=tuple(contributions),
                value_revision_refs={
                    key: self.values[key].revision for key in contributions
                },
                option_id=score.option_id,
                context_id=context_id,
                contribution_by_value=contributions,
                conflict_names=score.conflicts,
                reasoning_codes=("simultaneous_values_retained", "tradeoff_required"),
                decision_id=decision_id,
                created_at=_now(),
            )
            self.tradeoffs.append(record)
            records.append(record)
        return records

    def record_reassessment(
        self, decision: "DecisionRecord", updates: Iterable[ValueUpdateRecord]
    ) -> ValueReassessmentRecord:
        if decision.actual_outcome is None or decision.prediction_error is None:
            raise ValueError("Value reassessment requires a resolved decision")
        record = ValueReassessmentRecord(
            reassessment_id=str(uuid4()),
            decision_id=decision.decision_id,
            outcome_utility=decision.actual_outcome.utility,
            prediction_error=decision.prediction_error,
            regret=max(0.0, -decision.prediction_error),
            value_update_ids=tuple(item.update_id for item in updates),
            created_at=_now(),
        )
        self.reassessments.append(record)
        return record

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "values": {key: asdict(value) for key, value in self.values.items()},
            "conflicts": [asdict(item) for item in self.conflicts],
            "history": [asdict(item) for item in self.history],
            "applied_proposals": sorted(self.applied_proposals),
            "evidence": [asdict(item) for item in self.evidence.values()],
            "tradeoffs": [asdict(item) for item in self.tradeoffs],
            "reassessments": [asdict(item) for item in self.reassessments],
            "held_proposals": [asdict(item) for item in self.held_proposals.values()],
        }

    def get(self, value_id: str) -> ValueState:
        return self._require(value_id)

    def list_values(self) -> list[ValueState]:
        return [self.values[value_id] for value_id in sorted(self.values)]

    def active_values(self) -> list[ValueState]:
        return [item for item in self.list_values() if self._is_active(item)]

    def review_origin(
        self,
        value_id: str,
        *,
        accept: bool,
        reviewer_id: str,
        reviewer_authority: str,
        evidence_refs: tuple[str, ...],
        reason_code: str,
        event_id: str,
        event_sequence: int,
    ) -> ValueState:
        current = self._require(value_id)
        if current.origin_provenance is None or current.origin_provenance.actor not in {
            OriginActor.UNKNOWN,
            OriginActor.INHERITED,
        }:
            raise ValueError("Only unknown or inherited Value origins can be reviewed")
        if current.origin_provenance.endorsement not in {
            EndorsementStatus.PENDING,
            EndorsementStatus.UNCERTAIN,
        }:
            raise ValueError("Value origin is not awaiting review")
        if reviewer_authority not in {"subject", "operator"}:
            raise ValueError("Value review requires subject or operator authority")
        if (
            not reviewer_id
            or not evidence_refs
            or not reason_code
            or not event_id
            or event_sequence < 1
        ):
            raise ValueError("Value review requires reviewer, evidence, and reason")
        if any(
            re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", reference) is None
            for reference in (reviewer_id, reason_code, *evidence_refs)
        ):
            raise ValueError("Value review provenance must use opaque safe references")
        origin = (
            current.origin_provenance.endorse(
                f"value_review:{reviewer_authority}",
                event_id=event_id,
                event_sequence=event_sequence,
            )
            if accept
            else current.origin_provenance.reject(
                f"value_review:{reviewer_authority}",
                event_id=event_id,
                event_sequence=event_sequence,
            )
        )
        review = OriginReviewRecord(
            value_id=current.value_id,
            origin_id=current.origin_provenance.origin_id,
            origin_actor=current.origin_provenance.actor.value,
            origin_input_kind=current.origin_provenance.input_kind.value,
            origin_source_ref=current.origin_provenance.source_ref,
            reviewed_revision=current.revision,
            reviewed_state_digest=_reviewed_value_digest(current),
            reviewer_id=reviewer_id,
            reviewer_authority=reviewer_authority,
            decision="accept" if accept else "reject",
            evidence_refs=evidence_refs,
            reason_code=reason_code,
            event_id=event_id,
            event_sequence=event_sequence,
            record_hash=_origin_review_hash(
                current.value_id,
                current.origin_provenance.origin_id,
                current.origin_provenance.actor.value,
                current.origin_provenance.input_kind.value,
                current.origin_provenance.source_ref,
                current.revision,
                _reviewed_value_digest(current),
                reviewer_id,
                reviewer_authority,
                "accept" if accept else "reject",
                evidence_refs,
                reason_code,
                event_id,
                event_sequence,
            ),
        )
        updated = replace(
            current,
            origin_provenance=origin,
            origin_review=review,
            revision=current.revision + 1,
            last_reviewed_at=_now(),
            change_reason=reason_code,
            last_updated_at=_now(),
        )
        self.values[value_id] = updated
        self.history.append(
            ValueUpdateRecord(
                update_id=str(uuid4()),
                operation="origin_accept" if accept else "origin_reject",
                value_id=value_id,
                event_id=event_id,
                event_sequence=event_sequence,
                memory_ids=(),
                kind=ValueUpdateKind.ADMIN.value,
                reason_codes=(reason_code,),
                identity_origin=origin,
                requested_delta=0.0,
                applied_delta=0.0,
                before=_revision_snapshot(current),
                after=_revision_snapshot(updated),
                created_at=_now(),
                evidence_ids=evidence_refs,
                revision_diff=ValueRevisionDiff(
                    from_revision=current.revision,
                    to_revision=updated.revision,
                    changed_fields={
                        "origin_provenance": (
                            current.origin_provenance.to_json(),
                            origin.to_json(),
                        ),
                        "last_reviewed_at": (
                            current.last_reviewed_at,
                            updated.last_reviewed_at,
                        ),
                    },
                    reason_codes=(reason_code,),
                    evidence_ids=evidence_refs,
                ),
                reviewer_id=reviewer_id,
                reviewer_authority=reviewer_authority,
            )
        )
        if accept:
            held = [
                item
                for item in self.held_proposals.values()
                if item.value_id == value_id
            ]
            for item in held:
                self.held_proposals.pop(item.proposal_id, None)
            if held:
                self.apply(held)
        return updated

    def endorse_system_seed(
        self,
        value_id: str,
        *,
        reviewer_authority: str,
        event_id: str,
        event_sequence: int,
    ) -> ValueState:
        current = self._require(value_id)
        if (
            current.origin_provenance is None
            or current.origin_provenance.actor != OriginActor.SYSTEM
            or current.origin_provenance.input_kind != OriginInputKind.CONFIG_SEED
        ):
            raise ValueError("Subject endorsement applies only to SYSTEM config seeds")
        if reviewer_authority != "subject":
            raise ValueError("Only the subject can endorse a SYSTEM config seed")
        if not event_id or event_sequence < 1:
            raise ValueError("Subject endorsement requires an AgentRuntime event")
        updated = replace(
            current,
            origin_provenance=current.origin_provenance.endorse(
                "subject_endorsement",
                event_id=event_id,
                event_sequence=event_sequence,
            ),
            revision=current.revision + 1,
            last_reviewed_at=_now(),
            change_reason="subject_endorsement",
            last_updated_at=_now(),
        )
        self.values[value_id] = updated
        return updated

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.values = dict(self._seeds)
            self.history = []
            self.applied_proposals = set()
            self.evidence = {}
            self.tradeoffs = []
            self.reassessments = []
            self.held_proposals = {}
            return
        if payload.get("schema_version") not in {1, 2, 3}:
            if (
                isinstance(payload, dict)
                and payload
                and all(isinstance(value, (int, float)) for value in payload.values())
            ):
                for key, weight in payload.items():
                    legacy_id = f"legacy:{key}"
                    self.values[legacy_id] = ValueState(
                            value_id=legacy_id,
                            name=key.replace("_", " ").title(),
                            weight=float(weight),
                            confidence=0.5,
                            stability=0.5,
                            source="legacy_snapshot",
                            origin="legacy_snapshot",
                            last_updated_at=_now(),
                            allowed_update_rate=self.max_update_per_event,
                            origin_provenance=new_identity_origin(
                                OriginActor.INHERITED,
                                OriginInputKind.LEGACY,
                                endorsement=EndorsementStatus.UNCERTAIN,
                                source_ref=f"legacy:{key}",
                            ),
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
        self.evidence = {
            item.evidence_id: item
            for raw in payload.get("evidence", [])
            if isinstance(raw, dict)
            for item in [_value_evidence_record_from_json(raw)]
        }
        self.tradeoffs = [
            _value_tradeoff_from_json(item)
            for item in payload.get("tradeoffs", [])
            if isinstance(item, dict)
        ]
        self.reassessments = [
            _value_reassessment_from_json(item)
            for item in payload.get("reassessments", [])
            if isinstance(item, dict)
        ]
        self.held_proposals = {
            proposal.proposal_id: proposal
            for raw in payload.get("held_proposals", [])
            if isinstance(raw, dict)
            for proposal in [_value_update_proposal_from_json(raw)]
        }

    def revisions(self, value_id: str) -> list[ValueUpdateRecord]:
        self._require(value_id)
        return [record for record in self.history if record.value_id == value_id]

    def _record_evidence(self, proposal: ValueUpdateProposal) -> ValueEvidenceRecord:
        identifier = proposal.evidence.evidence_id or f"proposal:{proposal.proposal_id}"
        existing = self.evidence.get(identifier)
        if existing is not None and existing.value_id != proposal.value_id:
            identifier = f"{identifier}:{proposal.value_id}"
            existing = self.evidence.get(identifier)
        if existing is not None:
            return existing
        record = ValueEvidenceRecord(
            evidence_id=identifier,
            value_id=proposal.value_id,
            direction=EvidenceDirection.SUPPORTING
            if proposal.requested_delta >= 0
            else EvidenceDirection.OPPOSING,
            strength=abs(proposal.requested_delta),
            confidence=proposal.confidence,
            experience_ids=proposal.evidence.experience_ids,
            memory_ids=proposal.evidence.memory_ids,
            decision_id=proposal.evidence.decision_id,
            context_id=proposal.evidence.context_id,
            source=proposal.evidence.source,
            identity_origin=proposal.evidence.identity_origin,
            event_id=proposal.evidence.event_id,
            event_sequence=proposal.evidence.event_sequence,
            created_at=_now(),
        )
        self.evidence[identifier] = record
        return record

    def _record_evidence_only_update(
        self,
        state: ValueState,
        proposals: list[ValueUpdateProposal],
        evidence_records: list[ValueEvidenceRecord],
    ) -> ValueUpdateRecord:
        now = _now()
        before = _revision_snapshot(state)
        supporting = tuple(
            item.evidence_id
            for item in evidence_records
            if item.direction == EvidenceDirection.SUPPORTING
        )
        opposing = tuple(
            item.evidence_id
            for item in evidence_records
            if item.direction == EvidenceDirection.OPPOSING
        )
        after = replace(
            state,
            origin_experience_ids=tuple(
                dict.fromkeys(
                    (
                        *state.origin_experience_ids,
                        *(
                            experience_id
                            for item in evidence_records
                            for experience_id in item.experience_ids
                        ),
                    )
                )
            ),
            supporting_evidence_ids=(*state.supporting_evidence_ids, *supporting),
            opposing_evidence_ids=(*state.opposing_evidence_ids, *opposing),
            last_reviewed_at=now,
            change_reason="external_evidence_recorded",
            last_updated_at=now,
            revision=state.revision + 1,
        )
        self.values[state.value_id] = after
        evidence_ids = tuple(item.evidence_id for item in evidence_records)
        reason_codes = ("external_evidence_only",)
        record = ValueUpdateRecord(
            update_id=str(uuid4()),
            operation="evidence_only",
            value_id=state.value_id,
            event_id=proposals[-1].evidence.event_id,
            event_sequence=proposals[-1].evidence.event_sequence,
            memory_ids=proposals[-1].evidence.memory_ids,
            kind=proposals[-1].kind.value,
            reason_codes=reason_codes,
            identity_origin=proposals[-1].evidence.identity_origin,
            requested_delta=sum(
                proposal.requested_delta * proposal.confidence for proposal in proposals
            ),
            applied_delta=0.0,
            before=before,
            after=_revision_snapshot(after),
            created_at=now,
            evidence_ids=evidence_ids,
            revision_diff=ValueRevisionDiff(
                from_revision=state.revision,
                to_revision=after.revision,
                changed_fields=_snapshot_diff(before, _revision_snapshot(after)),
                reason_codes=reason_codes,
                evidence_ids=evidence_ids,
            ),
        )
        self.history.append(record)
        return record

    @staticmethod
    def _applies(state: ValueState, context_id: str | None) -> bool:
        return state.scope == ValueScope.SUBJECT or (
            context_id is not None and context_id in state.context_ids
        )

    @staticmethod
    def _is_active(state: ValueState) -> bool:
        origin = state.origin_provenance
        if origin is None or origin.endorsement != EndorsementStatus.ENDORSED:
            return False
        if origin.actor in {OriginActor.SELF, OriginActor.SYSTEM}:
            return True
        review = state.origin_review
        return (
            origin.actor in {OriginActor.UNKNOWN, OriginActor.INHERITED}
            and review is not None
            and review.decision == "accept"
        )

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
                revision_diff=ValueRevisionDiff(
                    from_revision=before.revision,
                    to_revision=after.revision,
                    changed_fields=_snapshot_diff(
                        _revision_snapshot(before), _revision_snapshot(after)
                    ),
                    reason_codes=(operation,),
                    evidence_ids=(),
                ),
            )
        )


def _value_state_from_json(payload: dict[str, Any]) -> ValueState:
    data = dict(payload)
    data["origin_provenance"] = identity_origin_from_json(
        data.get("origin_provenance"),
        fallback_source=str(data.get("origin", "legacy_value")),
    )
    data["scope"] = ValueScope(data.get("scope", ValueScope.SUBJECT))
    if isinstance(data.get("origin_review"), dict):
        review = dict(data["origin_review"])
        review["evidence_refs"] = tuple(review.get("evidence_refs", ()))
        try:
            bound_review = OriginReviewRecord(**review)
            origin = data["origin_provenance"]
            if (
                bound_review.value_id != data.get("value_id")
                or bound_review.reviewed_revision >= int(data.get("revision", 0))
                or bound_review.origin_id != origin.origin_id
                or bound_review.origin_actor != origin.actor.value
                or bound_review.origin_input_kind != origin.input_kind.value
                or bound_review.origin_source_ref != origin.source_ref
            ):
                raise ValueError("Origin review binding mismatch")
            data["origin_review"] = bound_review
        except (TypeError, ValueError):
            data["origin_review"] = None
            origin = data["origin_provenance"]
            data["origin_provenance"] = replace(
                origin,
                endorsement=EndorsementStatus.UNCERTAIN,
                endorsement_ref=None,
                endorsed_by_event_id=None,
                endorsed_by_event_sequence=None,
            )
    for name in (
        "context_ids",
        "origin_experience_ids",
        "supporting_evidence_ids",
        "opposing_evidence_ids",
        "related_drive_ids",
        "related_goal_ids",
        "related_commitment_ids",
    ):
        data[name] = tuple(data.get(name, ()))
    data["schema_version"] = 3
    return ValueState(**data)


def _origin_review_hash(
    value_id: str,
    origin_id: str,
    origin_actor: str,
    origin_input_kind: str,
    origin_source_ref: str | None,
    reviewed_revision: int,
    reviewed_state_digest: str,
    reviewer_id: str,
    reviewer_authority: str,
    decision: str,
    evidence_refs: tuple[str, ...],
    reason_code: str,
    event_id: str,
    event_sequence: int,
) -> str:
    payload = {
        "value_id": value_id,
        "origin_id": origin_id,
        "origin_actor": origin_actor,
        "origin_input_kind": origin_input_kind,
        "origin_source_ref": origin_source_ref,
        "reviewed_revision": reviewed_revision,
        "reviewed_state_digest": reviewed_state_digest,
        "reviewer_id": reviewer_id,
        "reviewer_authority": reviewer_authority,
        "decision": decision,
        "evidence_refs": evidence_refs,
        "reason_code": reason_code,
        "event_id": event_id,
        "event_sequence": event_sequence,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reviewed_value_digest(value: ValueState) -> str:
    payload = asdict(replace(value, origin_review=None))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _value_update_record_from_json(payload: dict[str, Any]) -> ValueUpdateRecord:
    data = dict(payload)
    data["memory_ids"] = tuple(data.get("memory_ids", ()))
    data["reason_codes"] = tuple(data.get("reason_codes", ()))
    data["identity_origin"] = identity_origin_from_json(
        data.get("identity_origin"), fallback_source="legacy_value_update"
    )
    data["evidence_ids"] = tuple(data.get("evidence_ids", ()))
    if isinstance(data.get("revision_diff"), dict):
        diff = dict(data["revision_diff"])
        diff["reason_codes"] = tuple(diff.get("reason_codes", ()))
        diff["evidence_ids"] = tuple(diff.get("evidence_ids", ()))
        diff["changed_fields"] = {
            key: tuple(value) for key, value in diff.get("changed_fields", {}).items()
        }
        data["revision_diff"] = ValueRevisionDiff(**diff)
    return ValueUpdateRecord(**data)


def _value_evidence_record_from_json(payload: dict[str, Any]) -> ValueEvidenceRecord:
    data = dict(payload)
    data["direction"] = EvidenceDirection(data["direction"])
    data["experience_ids"] = tuple(data.get("experience_ids", ()))
    data["memory_ids"] = tuple(data.get("memory_ids", ()))
    data["identity_origin"] = identity_origin_from_json(
        data.get("identity_origin"), fallback_source="legacy_value_evidence"
    )
    return ValueEvidenceRecord(**data)


def _value_update_proposal_from_json(payload: dict[str, Any]) -> ValueUpdateProposal:
    data = dict(payload)
    data["kind"] = ValueUpdateKind(data["kind"])
    data["reason_codes"] = tuple(data.get("reason_codes", ()))
    evidence = dict(data["evidence"])
    for name in ("memory_ids", "experience_ids"):
        evidence[name] = tuple(evidence.get(name, ()))
    evidence["identity_origin"] = identity_origin_from_json(
        evidence.get("identity_origin"), fallback_source="held_value_proposal"
    )
    data["evidence"] = ValueEvidence(**evidence)
    return ValueUpdateProposal(**data)


def _value_tradeoff_from_json(payload: dict[str, Any]) -> ValueTradeoffRecord:
    data = dict(payload)
    for name in ("value_ids", "conflict_names", "reasoning_codes"):
        data[name] = tuple(data.get(name, ()))
    return ValueTradeoffRecord(**data)


def _value_reassessment_from_json(
    payload: dict[str, Any],
) -> ValueReassessmentRecord:
    data = dict(payload)
    data["value_update_ids"] = tuple(data.get("value_update_ids", ()))
    return ValueReassessmentRecord(**data)


def _revision_snapshot(state: ValueState) -> dict[str, Any]:
    return {
        "revision": state.revision,
        "weight": state.weight,
        "confidence": state.confidence,
        "stability": state.stability,
        "frozen": state.frozen,
        "polarity": state.polarity,
        "protectedness": state.protectedness,
        "negotiability": state.negotiability,
        "scope": state.scope.value,
        "context_ids": state.context_ids,
        "supporting_evidence_ids": state.supporting_evidence_ids,
        "opposing_evidence_ids": state.opposing_evidence_ids,
        "change_reason": state.change_reason,
    }


def _snapshot_diff(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, tuple[object, object]]:
    return {
        key: (before.get(key), after.get(key))
        for key in before.keys() | after.keys()
        if before.get(key) != after.get(key)
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
