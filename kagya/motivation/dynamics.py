"""Persistent internal motivation signals and bounded goal formation."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any, Callable
from uuid import uuid4

from kagya.experience import ExperienceRecord
from kagya.identity import OriginActor, OriginInputKind


class MotivationKind(StrEnum):
    DRIVE = "drive"
    INTEREST = "interest"
    DESIRE = "desire"
    AVERSION = "aversion"


class MotivationSource(StrEnum):
    CURIOSITY = "curiosity"
    CLOSURE = "closure"
    DELIBERATION = "deliberation"
    SOCIAL = "social"
    LEARNING = "learning"
    HOMEOSTATIC = "homeostatic"


class MotivationStatus(StrEnum):
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    SATISFIED = "satisfied"
    FAILED = "failed"
    DECAYED = "decayed"


@dataclass(frozen=True)
class MotivationEvidence:
    evidence_ref: str
    source_state_ref: str
    observed_at: str
    origin_actor: OriginActor
    origin_input_kind: OriginInputKind
    measurements: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if datetime.fromisoformat(self.observed_at).tzinfo is None:
            raise ValueError("motivation evidence time must include a timezone")
        for name, value in self.measurements:
            _unit(value, f"motivation evidence {name}")


@dataclass(frozen=True)
class MotivationRevision:
    revision_id: str
    operation: str
    before: dict[str, Any]
    after: dict[str, Any]
    evidence_refs: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class MotivationRecord:
    motivation_id: str
    kind: MotivationKind
    source: MotivationSource
    target_ref: str
    source_refs: tuple[str, ...]
    strength: float
    persistence: float
    satiation: float
    uncertainty: float
    decay_per_hour: float
    conflict_ids: tuple[str, ...]
    related_value_ids: tuple[str, ...]
    related_experience_ids: tuple[str, ...]
    related_goal_ids: tuple[str, ...]
    evidence_count: int
    status: MotivationStatus
    created_at: str
    updated_at: str
    next_review_at: str | None = None
    revision: int = 0
    revisions: tuple[MotivationRevision, ...] = ()
    evidence: tuple[MotivationEvidence, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(
                f"Unsupported motivation schema version: {self.schema_version}"
            )
        if not self.motivation_id or not self.target_ref or not self.source_refs:
            raise ValueError("motivation requires ID, target, and source references")
        for name in ("strength", "persistence", "satiation", "uncertainty"):
            _unit(getattr(self, name), name)
        if not math.isfinite(self.decay_per_hour) or self.decay_per_hour <= 0.0:
            raise ValueError("motivation decay must be finite and positive")
        if self.evidence_count <= 0 or self.revision < 0:
            raise ValueError("motivation evidence count must be positive")
        if (
            self.next_review_at is not None
            and datetime.fromisoformat(self.next_review_at).tzinfo is None
        ):
            raise ValueError("motivation next review must include a timezone")

    def to_json(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class GoalFormationCandidate:
    motivation_id: str
    description: str
    target_ref: str
    priority: float
    urgency: float
    confidence: float
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class MotivationEpisode:
    episode_id: str
    evaluated_motivation_ids: tuple[str, ...]
    selected_motivation_ids: tuple[str, ...]
    held_conflict_ids: tuple[str, ...]
    generated_goal_ids: tuple[str, ...]
    budget: int
    event_id: str | None
    event_sequence: int | None
    created_at: str


class MotivationDynamics:
    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        max_goal_proposals_per_cycle: int = 2,
        min_strength: float = 0.5,
        min_persistence: float = 0.4,
        min_evidence_count: int = 2,
        min_persistence_seconds: float = 60.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_goal_proposals_per_cycle <= 0 or min_evidence_count <= 0:
            raise ValueError("motivation budgets must be positive")
        if not math.isfinite(min_persistence_seconds) or min_persistence_seconds < 0:
            raise ValueError(
                "minimum persistence seconds must be finite and non-negative"
            )
        self.max_goal_proposals_per_cycle = max_goal_proposals_per_cycle
        self.min_strength = min_strength
        self.min_persistence = min_persistence
        self.min_evidence_count = min_evidence_count
        self.min_persistence_seconds = min_persistence_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._now_datetime()
        self.records: dict[str, MotivationRecord] = {}
        self.episodes: list[MotivationEpisode] = []

    def observe_experience(
        self, experience: ExperienceRecord
    ) -> list[MotivationRecord]:
        experience_ref = f"experience:{experience.experience_id}"
        if any(
            experience.experience_id in record.related_experience_ids
            for record in self.records.values()
        ):
            return []
        target_ref = f"context:{experience.context_id}"
        generated: list[MotivationRecord] = []
        novelty = experience.appraisal.novelty or 0.0
        prediction_error = (
            0.0
            if experience.prediction_error is None
            else 1.0 - math.exp(-experience.prediction_error)
        )
        curiosity_allowed = self._curiosity_allowed(experience, target_ref)
        if curiosity_allowed and (
            (experience.appraisal.novelty_valid and novelty >= 0.35)
            or prediction_error >= 0.35
        ):
            curiosity_signal = min(
                1.0,
                0.45 * (novelty if experience.appraisal.novelty_valid else 0.0)
                + 0.25 * prediction_error
                + 0.15 * experience.subjective_salience
                + 0.15 * (1.0 - experience.appraisal.certainty),
            )
            generated.append(
                self._reinforce(
                    MotivationKind.INTEREST,
                    MotivationSource.CURIOSITY,
                    target_ref,
                    signal=curiosity_signal,
                    uncertainty=1.0 - experience.appraisal.certainty,
                    source_ref=experience_ref,
                    experience_id=experience.experience_id,
                    value_ids=(),
                    evidence=_experience_evidence(
                        experience,
                        experience_ref,
                        signal=curiosity_signal,
                        measurements=(
                            (
                                "novelty",
                                novelty if experience.appraisal.novelty_valid else 0.0,
                            ),
                            ("prediction_error", prediction_error),
                            ("uncertainty", 1.0 - experience.appraisal.certainty),
                        ),
                    ),
                )
            )
        internal_allowed = _is_self_internal(experience)
        if internal_allowed and experience.unresolved_tension >= 0.4:
            generated.append(
                self._reinforce(
                    MotivationKind.DRIVE,
                    MotivationSource.CLOSURE,
                    target_ref,
                    signal=experience.unresolved_tension,
                    uncertainty=1.0 - experience.appraisal.certainty,
                    source_ref=experience_ref,
                    experience_id=experience.experience_id,
                    value_ids=(),
                    evidence=_experience_evidence(
                        experience,
                        experience_ref,
                        signal=experience.unresolved_tension,
                        measurements=(("tension", experience.unresolved_tension),),
                    ),
                )
            )
        if internal_allowed and experience.appraisal.threat >= 0.5:
            generated.append(
                self._reinforce(
                    MotivationKind.AVERSION,
                    MotivationSource.DELIBERATION,
                    target_ref,
                    signal=experience.appraisal.threat,
                    uncertainty=1.0 - experience.appraisal.certainty,
                    source_ref=experience_ref,
                    experience_id=experience.experience_id,
                    value_ids=(),
                    evidence=_experience_evidence(
                        experience,
                        experience_ref,
                        signal=experience.appraisal.threat,
                        measurements=(("threat", experience.appraisal.threat),),
                    ),
                )
            )
        return generated

    def observe_future_self_gap(
        self,
        projection_id: str,
        *,
        gap: float,
        uncertainty: float,
        related_value_ids: tuple[str, ...] = (),
    ) -> MotivationRecord | None:
        """Connect a durable desired/current self discrepancy to motivation."""
        _unit(gap, "future-self gap")
        _unit(uncertainty, "future-self uncertainty")
        if gap <= 0.0:
            return None
        source_ref = f"future-self:{projection_id}"
        if any(source_ref in record.source_refs for record in self.records.values()):
            return next(
                record
                for record in self.records.values()
                if source_ref in record.source_refs
            )
        return self._reinforce(
            MotivationKind.DESIRE,
            MotivationSource.LEARNING,
            source_ref,
            signal=gap,
            uncertainty=uncertainty,
            source_ref=source_ref,
            experience_id=None,
            value_ids=related_value_ids,
            evidence=_structured_evidence(
                source_ref,
                observed_at=self._now(),
                signal=gap,
                uncertainty=uncertainty,
                measurements=(("gap", gap),),
            ),
        )

    def observe_structured_signal(
        self,
        kind: MotivationKind,
        source: MotivationSource,
        target_ref: str,
        *,
        signal: float,
        uncertainty: float,
        source_refs: tuple[str, ...],
        value_ids: tuple[str, ...] = (),
        observed_at: str | None = None,
        measurements: tuple[tuple[str, float], ...] = (),
    ) -> MotivationRecord:
        """Record durable structured evidence without model interpretation."""
        _unit(signal, "motivation signal")
        _unit(uncertainty, "motivation uncertainty")
        evidence_refs = tuple(dict.fromkeys(source_refs))
        if not evidence_refs:
            raise ValueError("structured motivation requires evidence references")
        evidence = tuple(
            _structured_evidence(
                reference,
                observed_at=observed_at or self._now(),
                signal=signal,
                uncertainty=uncertainty,
                measurements=measurements,
            )
            for reference in evidence_refs
        )
        matching = [
            record
            for record in self.records.values()
            if record.source == source and record.target_ref == target_ref
        ]
        existing = next(
            (record for record in matching if record.status == MotivationStatus.ACTIVE),
            None,
        )
        if existing is None:
            terminal = next(
                (
                    record
                    for record in matching
                    if set(evidence_refs).issubset(record.source_refs)
                ),
                None,
            )
            if terminal is not None:
                return terminal
            now = self._now()
            record = MotivationRecord(
                motivation_id=f"motivation-{uuid4()}",
                kind=kind,
                source=source,
                target_ref=target_ref,
                source_refs=evidence_refs,
                strength=min(1.0, 0.35 + 0.4 * signal),
                persistence=max(0.4, signal),
                satiation=0.0,
                uncertainty=uncertainty,
                decay_per_hour=0.05,
                conflict_ids=(),
                related_value_ids=tuple(dict.fromkeys(value_ids)),
                related_experience_ids=(),
                related_goal_ids=(),
                evidence_count=len(evidence_refs),
                status=MotivationStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                evidence=evidence,
            )
            self.records[record.motivation_id] = record
            return record
        novel_refs = tuple(
            reference
            for reference in evidence_refs
            if reference not in existing.source_refs
        )
        novel_values = tuple(
            value_id
            for value_id in value_ids
            if value_id not in existing.related_value_ids
        )
        if not novel_refs and not novel_values:
            return existing
        updated = self._revise(
            existing,
            replace(
                existing,
                source_refs=(*existing.source_refs, *novel_refs),
                related_value_ids=(*existing.related_value_ids, *novel_values),
                strength=max(existing.strength, min(1.0, 0.35 + 0.4 * signal)),
                persistence=max(existing.persistence, signal, 0.4),
                uncertainty=min(existing.uncertainty, uncertainty),
                evidence_count=existing.evidence_count + len(novel_refs),
                updated_at=self._now(),
                evidence=(
                    *existing.evidence,
                    *(item for item in evidence if item.evidence_ref in novel_refs),
                ),
            ),
            "structured_evidence",
            novel_refs,
        )
        self.records[existing.motivation_id] = updated
        return updated

    def retire_structured_signal(
        self,
        source: MotivationSource,
        target_ref: str,
        *,
        source_state_ref: str,
    ) -> MotivationRecord | None:
        current = next(
            (
                record
                for record in self.records.values()
                if record.source == source
                and record.target_ref == target_ref
                and record.status == MotivationStatus.ACTIVE
            ),
            None,
        )
        if current is None:
            return None
        updated = self._revise(
            current,
            replace(
                current,
                status=MotivationStatus.DECAYED,
                strength=0.0,
                updated_at=self._now(),
            ),
            "source_signal_retired",
            (source_state_ref,),
        )
        self.records[current.motivation_id] = updated
        return updated

    def register_conflict(self, left_id: str, right_id: str) -> None:
        if left_id == right_id:
            raise ValueError("motivation cannot conflict with itself")
        left = self.get(left_id)
        right = self.get(right_id)
        if right_id in left.conflict_ids and left_id in right.conflict_ids:
            return
        self.records[left_id] = self._revise(
            left,
            replace(
                left,
                conflict_ids=tuple(dict.fromkeys((*left.conflict_ids, right_id))),
                updated_at=self._now(),
            ),
            "conflict_registered",
            (f"motivation:{right_id}",),
        )
        self.records[right_id] = self._revise(
            right,
            replace(
                right,
                conflict_ids=tuple(dict.fromkeys((*right.conflict_ids, left_id))),
                updated_at=self._now(),
            ),
            "conflict_registered",
            (f"motivation:{left_id}",),
        )

    def goal_candidates(
        self, max_candidates: int | None = None, *, review_at: datetime | None = None
    ) -> tuple[list[GoalFormationCandidate], tuple[str, ...]]:
        limit = self.max_goal_proposals_per_cycle
        if max_candidates is not None:
            if max_candidates < 0:
                raise ValueError("goal candidate limit must not be negative")
            limit = min(limit, max_candidates)
        reviewed_at = review_at or self._now_datetime()
        if reviewed_at.tzinfo is None:
            raise ValueError("motivation review time must include a timezone")
        eligible = [
            record
            for record in self.records.values()
            if record.status == MotivationStatus.ACTIVE
            and record.strength >= self.min_strength
            and record.persistence >= self.min_persistence
            and record.evidence_count >= self.min_evidence_count
            and record.satiation < 0.9
            and not record.related_goal_ids
            and (
                reviewed_at - datetime.fromisoformat(record.created_at)
            ).total_seconds()
            >= self.min_persistence_seconds
        ]
        eligible.sort(
            key=lambda item: (item.strength * item.persistence, item.motivation_id),
            reverse=True,
        )
        held: set[str] = set()
        selected: list[GoalFormationCandidate] = []
        eligible_ids = {record.motivation_id for record in eligible}
        for record in eligible:
            blocking = eligible_ids.intersection(record.conflict_ids)
            if blocking:
                held.add(record.motivation_id)
                held.update(blocking)
                continue
            if len(selected) >= limit:
                break
            selected.append(_goal_candidate(record))
        return selected, tuple(sorted(held))

    def link_goal(self, motivation_id: str, goal_id: str) -> MotivationRecord:
        current = self.get(motivation_id)
        if goal_id in current.related_goal_ids:
            return current
        updated = self._revise(
            current,
            replace(
                current,
                kind=MotivationKind.DESIRE,
                related_goal_ids=(*current.related_goal_ids, goal_id),
                updated_at=self._now(),
            ),
            "goal_proposed",
            (f"goal:{goal_id}",),
        )
        self.records[motivation_id] = updated
        return updated

    def resolve_goal(self, goal_id: str, *, success: bool) -> list[MotivationRecord]:
        updated_records: list[MotivationRecord] = []
        for current in list(self.records.values()):
            if goal_id not in current.related_goal_ids:
                continue
            status = MotivationStatus.SATISFIED if success else MotivationStatus.FAILED
            updated = self._revise(
                current,
                replace(
                    current,
                    status=status,
                    strength=0.0 if success else max(0.0, current.strength - 0.25),
                    satiation=1.0 if success else current.satiation,
                    updated_at=self._now(),
                ),
                "goal_satisfied" if success else "goal_failed",
                (f"goal:{goal_id}",),
            )
            self.records[current.motivation_id] = updated
            updated_records.append(updated)
        return updated_records

    def decay(self, elapsed_hours: float) -> list[MotivationRecord]:
        if not math.isfinite(elapsed_hours) or elapsed_hours <= 0.0:
            raise ValueError("elapsed hours must be finite and positive")
        updated_records: list[MotivationRecord] = []
        for current in list(self.records.values()):
            if current.status != MotivationStatus.ACTIVE:
                continue
            strength = max(
                0.0, current.strength - current.decay_per_hour * elapsed_hours
            )
            satiation = max(0.0, current.satiation - 0.05 * elapsed_hours)
            status = MotivationStatus.DECAYED if strength < 0.1 else current.status
            updated = self._revise(
                current,
                replace(
                    current,
                    strength=strength,
                    satiation=satiation,
                    status=status,
                    updated_at=self._now(),
                ),
                "time_decay",
                (),
            )
            self.records[current.motivation_id] = updated
            updated_records.append(updated)
        return updated_records

    def decay_record(
        self, motivation_id: str, elapsed_hours: float
    ) -> MotivationRecord | None:
        if not math.isfinite(elapsed_hours) or elapsed_hours <= 0.0:
            raise ValueError("elapsed hours must be finite and positive")
        current = self.get(motivation_id)
        if current.status != MotivationStatus.ACTIVE:
            return None
        strength = max(0.0, current.strength - current.decay_per_hour * elapsed_hours)
        satiation = max(0.0, current.satiation - 0.05 * elapsed_hours)
        status = MotivationStatus.DECAYED if strength < 0.1 else current.status
        updated = self._revise(
            current,
            replace(
                current,
                strength=strength,
                satiation=satiation,
                status=status,
                updated_at=self._now(),
            ),
            "time_decay",
            (),
        )
        self.records[motivation_id] = updated
        return updated

    def schedule_next_reviews(self, review_at: datetime) -> list[MotivationRecord]:
        if review_at.tzinfo is None:
            raise ValueError("motivation review time must include a timezone")
        updated_records: list[MotivationRecord] = []
        for current in list(self.records.values()):
            if current.status != MotivationStatus.ACTIVE:
                continue
            review_value = review_at.isoformat()
            if current.next_review_at == review_value:
                continue
            updated = self._revise(
                current,
                replace(current, next_review_at=review_value),
                "review_scheduled",
                (),
            )
            self.records[current.motivation_id] = updated
            updated_records.append(updated)
        return updated_records

    def record_episode(
        self,
        *,
        selected_ids: tuple[str, ...],
        held_ids: tuple[str, ...],
        generated_goal_ids: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
        budget: int | None = None,
    ) -> MotivationEpisode:
        resolved_budget = self.max_goal_proposals_per_cycle
        if budget is not None:
            if budget < 0:
                raise ValueError("motivation episode budget must not be negative")
            resolved_budget = min(resolved_budget, budget)
        episode = MotivationEpisode(
            episode_id=f"motivation-episode-{uuid4()}",
            evaluated_motivation_ids=tuple(sorted(self.records)),
            selected_motivation_ids=selected_ids,
            held_conflict_ids=held_ids,
            generated_goal_ids=generated_goal_ids,
            budget=resolved_budget,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=self._now(),
        )
        self.episodes.append(episode)
        return episode

    def get(self, motivation_id: str) -> MotivationRecord:
        record = self.records.get(motivation_id)
        if record is None:
            raise ValueError(f"Unknown motivation: {motivation_id}")
        return record

    def list_records(self) -> list[MotivationRecord]:
        return sorted(self.records.values(), key=lambda item: item.motivation_id)

    def reassess_experience(
        self, experience: ExperienceRecord
    ) -> list[MotivationRecord]:
        evidence_ref = f"experience:{experience.experience_id}@{experience.revision}"
        updated: list[MotivationRecord] = []
        for current in tuple(self.records.values()):
            if experience.experience_id not in current.related_experience_ids:
                continue
            if evidence_ref in current.source_refs:
                continue
            signal = {
                MotivationKind.INTEREST: experience.appraisal.novelty or 0.0,
                MotivationKind.DRIVE: experience.unresolved_tension,
                MotivationKind.AVERSION: experience.appraisal.threat,
                MotivationKind.DESIRE: max(0.0, experience.appraisal.goal_progress),
            }[current.kind]
            desired = min(1.0, 0.35 + 0.4 * signal)
            strength = current.strength + max(
                -0.1, min(0.1, desired - current.strength)
            )
            before = current.to_json()
            now = self._now()
            revision = MotivationRevision(
                revision_id=f"motivation-revision-{uuid4()}",
                operation="experience_reassessment",
                before=before,
                after={},
                evidence_refs=(evidence_ref,),
                created_at=now,
            )
            record = replace(
                current,
                strength=strength,
                uncertainty=1.0 - experience.appraisal.certainty,
                source_refs=(*current.source_refs, evidence_ref),
                updated_at=now,
                revision=current.revision + 1,
            )
            revision = replace(revision, after=record.to_json())
            record = replace(record, revisions=(*current.revisions, revision))
            self.records[record.motivation_id] = record
            updated.append(record)
        return updated

    def _curiosity_allowed(self, experience: ExperienceRecord, target_ref: str) -> bool:
        origin = experience.identity_origin
        if _is_self_internal(experience):
            return True
        if origin.input_kind in {OriginInputKind.OBSERVATION, OriginInputKind.EVIDENCE}:
            return True
        if origin.input_kind != OriginInputKind.REQUEST:
            return False
        return any(
            record.source == MotivationSource.CURIOSITY
            and record.target_ref == target_ref
            and any(
                item.origin_actor == OriginActor.SELF
                and item.origin_input_kind == OriginInputKind.INTERNAL_STATE
                for item in record.evidence
            )
            for record in self.records.values()
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "records": [record.to_json() for record in self.list_records()],
            "episodes": [_json_value(asdict(item)) for item in self.episodes],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.records = {}
            self.episodes = []
            return
        version = payload.get("schema_version")
        if version not in {1, self.SCHEMA_VERSION}:
            raise ValueError("Unsupported motivation dynamics schema version")
        records = [
            _record_from_json(_migrate_record_v1(item) if version == 1 else item)
            for item in payload.get("records", [])
        ]
        if len(records) != len({record.motivation_id for record in records}):
            raise ValueError("Motivation identifiers must be unique")
        self.records = {record.motivation_id: record for record in records}
        self.episodes = [
            MotivationEpisode(
                **{
                    **item,
                    "evaluated_motivation_ids": tuple(item["evaluated_motivation_ids"]),
                    "selected_motivation_ids": tuple(item["selected_motivation_ids"]),
                    "held_conflict_ids": tuple(item["held_conflict_ids"]),
                    "generated_goal_ids": tuple(item["generated_goal_ids"]),
                }
            )
            for item in payload.get("episodes", [])
        ]

    def _reinforce(
        self,
        kind: MotivationKind,
        source: MotivationSource,
        target_ref: str,
        *,
        signal: float,
        uncertainty: float,
        source_ref: str,
        experience_id: str | None,
        value_ids: tuple[str, ...],
        evidence: MotivationEvidence,
    ) -> MotivationRecord:
        matching = [
            record
            for record in self.records.values()
            if record.source == source and record.target_ref == target_ref
        ]
        existing = next(
            (record for record in matching if record.status == MotivationStatus.ACTIVE),
            None,
        )
        if existing is None:
            terminal = next(
                (record for record in matching if source_ref in record.source_refs),
                None,
            )
            if terminal is not None:
                return terminal
            now = self._now()
            record = MotivationRecord(
                motivation_id=f"motivation-{uuid4()}",
                kind=kind,
                source=source,
                target_ref=target_ref,
                source_refs=(source_ref,),
                strength=min(1.0, 0.35 + 0.4 * signal),
                persistence=0.2,
                satiation=0.0,
                uncertainty=uncertainty,
                decay_per_hour=0.05,
                conflict_ids=(),
                related_value_ids=value_ids,
                related_experience_ids=()
                if experience_id is None
                else (experience_id,),
                related_goal_ids=(),
                evidence_count=1,
                status=MotivationStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                evidence=(evidence,),
            )
            self.records[record.motivation_id] = record
            return record
        satiation = min(1.0, existing.satiation + 0.1)
        strength = min(
            1.0,
            existing.strength + 0.25 * signal * (1.0 - satiation),
        )
        updated = self._revise(
            existing,
            replace(
                existing,
                source_refs=(*existing.source_refs, source_ref),
                strength=strength,
                persistence=min(1.0, existing.persistence + 0.25),
                satiation=satiation,
                uncertainty=(existing.uncertainty + uncertainty) / 2.0,
                related_experience_ids=tuple(
                    dict.fromkeys(
                        (
                            *existing.related_experience_ids,
                            *(() if experience_id is None else (experience_id,)),
                        )
                    )
                ),
                related_value_ids=tuple(
                    dict.fromkeys((*existing.related_value_ids, *value_ids))
                ),
                evidence_count=existing.evidence_count + 1,
                updated_at=self._now(),
                evidence=(*existing.evidence, evidence),
            ),
            "experience_reinforcement",
            (source_ref,),
        )
        self.records[existing.motivation_id] = updated
        return updated

    def _revise(
        self,
        before: MotivationRecord,
        after: MotivationRecord,
        operation: str,
        evidence_refs: tuple[str, ...],
    ) -> MotivationRecord:
        revision = MotivationRevision(
            revision_id=f"motivation-revision-{uuid4()}",
            operation=operation,
            before=_state(before),
            after=_state(after),
            evidence_refs=evidence_refs,
            created_at=self._now(),
        )
        return replace(
            after,
            revision=before.revision + 1,
            revisions=(*before.revisions, revision),
        )

    def _now_datetime(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("motivation clock must return a timezone-aware datetime")
        return now

    def _now(self) -> str:
        return self._now_datetime().isoformat()


def _goal_candidate(record: MotivationRecord) -> GoalFormationCandidate:
    action = {
        MotivationSource.CURIOSITY: "Investigate unresolved novelty",
        MotivationSource.CLOSURE: "Resolve outstanding tension",
        MotivationSource.DELIBERATION: "Clarify a potential conflict",
        MotivationSource.SOCIAL: "Review an active social obligation",
        MotivationSource.LEARNING: "Reduce a known capability gap",
        MotivationSource.HOMEOSTATIC: "Restore a sustained internal balance",
    }[record.source]
    return GoalFormationCandidate(
        motivation_id=record.motivation_id,
        description=f"{action} for {record.target_ref}",
        target_ref=record.target_ref,
        priority=record.strength,
        urgency=max(0.2, record.strength * (1.0 - record.satiation)),
        confidence=max(0.0, 1.0 - record.uncertainty),
        source_refs=record.source_refs,
    )


def _state(record: MotivationRecord) -> dict[str, Any]:
    return {
        "kind": record.kind.value,
        "strength": record.strength,
        "persistence": record.persistence,
        "satiation": record.satiation,
        "uncertainty": record.uncertainty,
        "status": record.status.value,
        "evidence_count": record.evidence_count,
        "related_goal_ids": list(record.related_goal_ids),
        "next_review_at": record.next_review_at,
    }


def _record_from_json(payload: dict[str, Any]) -> MotivationRecord:
    data = dict(payload)
    data["kind"] = MotivationKind(data["kind"])
    data["source"] = MotivationSource(data["source"])
    data["status"] = MotivationStatus(data["status"])
    for name in (
        "source_refs",
        "conflict_ids",
        "related_value_ids",
        "related_experience_ids",
        "related_goal_ids",
    ):
        data[name] = tuple(data.get(name, ()))
    data["revisions"] = tuple(
        MotivationRevision(
            **{**item, "evidence_refs": tuple(item.get("evidence_refs", ()))}
        )
        for item in data.get("revisions", ())
    )
    data["evidence"] = tuple(
        MotivationEvidence(
            **{
                **item,
                "origin_actor": OriginActor(item["origin_actor"]),
                "origin_input_kind": OriginInputKind(item["origin_input_kind"]),
                "measurements": tuple(
                    tuple(value) for value in item.get("measurements", ())
                ),
            }
        )
        for item in data.get("evidence", ())
    )
    return MotivationRecord(**data)


def _migrate_record_v1(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "schema_version": 2, "evidence": []}


def _is_self_internal(experience: ExperienceRecord) -> bool:
    return (
        experience.identity_origin.actor == OriginActor.SELF
        and experience.identity_origin.input_kind == OriginInputKind.INTERNAL_STATE
    )


def _experience_evidence(
    experience: ExperienceRecord,
    evidence_ref: str,
    *,
    signal: float,
    measurements: tuple[tuple[str, float], ...],
) -> MotivationEvidence:
    return MotivationEvidence(
        evidence_ref=evidence_ref,
        source_state_ref=f"{evidence_ref}@{experience.revision}",
        observed_at=experience.created_at,
        origin_actor=experience.identity_origin.actor,
        origin_input_kind=experience.identity_origin.input_kind,
        measurements=(("signal", signal), *measurements),
    )


def _structured_evidence(
    evidence_ref: str,
    *,
    observed_at: str,
    signal: float,
    uncertainty: float,
    measurements: tuple[tuple[str, float], ...],
) -> MotivationEvidence:
    return MotivationEvidence(
        evidence_ref=evidence_ref,
        source_state_ref=evidence_ref,
        observed_at=observed_at,
        origin_actor=OriginActor.SELF,
        origin_input_kind=OriginInputKind.INTERNAL_STATE,
        measurements=(("signal", signal), ("uncertainty", uncertainty), *measurements),
    )


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
