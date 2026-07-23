"""Bounded, revisioned counterfactual comparisons from structured evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import re
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


COUNTERFACTUAL_STATE_KEY = "counterfactual_simulation"
_STRUCTURED_CODE = r"^[A-Za-z0-9_.:@/-]+$"
_PRIVATE_ALIASES = {
    "hiddenthought",
    "prompt",
    "rawprompt",
    "chainofthought",
    "reasoning",
    "selfreport",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceStatus(StrEnum):
    EVIDENCE_BACKED = "evidence_backed"
    SPECULATIVE = "speculative"


class CounterfactualSignal(StrEnum):
    REGRET = "regret"
    RELIEF = "relief"
    MISSED_OPPORTUNITY = "missed_opportunity"
    NONE = "none"


class CounterfactualTarget(StrEnum):
    DECISION_CALIBRATION = "decision_calibration"
    VALUE = "value"
    MOTIVATION = "motivation"
    PLAN_STRATEGY = "plan_strategy"
    EMOTION = "emotion"
    METACOGNITION = "metacognition"


class AlternativeOutcome(_StrictModel):
    candidate_id: str = Field(min_length=1, max_length=128, pattern=_STRUCTURED_CODE)
    candidate_type: str = Field(min_length=1, max_length=64, pattern=_STRUCTURED_CODE)
    plausible_utility: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=0.8, allow_inf_nan=False)
    evidence_status: EvidenceStatus
    assumption_codes: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_alternative(self) -> "AlternativeOutcome":
        if len(self.assumption_codes) != len(set(self.assumption_codes)):
            raise ValueError("counterfactual assumptions must be unique")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("counterfactual evidence references must be unique")
        for value in (
            self.candidate_id,
            self.candidate_type,
            *self.assumption_codes,
            *self.evidence_refs,
        ):
            _validate_structured_value(value)
        if self.evidence_status == EvidenceStatus.SPECULATIVE and self.confidence > 0.6:
            raise ValueError("speculative counterfactual confidence cannot exceed 0.6")
        return self


class CounterfactualSimulation(_StrictModel):
    schema_version: Literal[1] = 1
    simulation_id: str = Field(min_length=1, max_length=128, pattern=_STRUCTURED_CODE)
    revision: int = Field(ge=1)
    decision_id: str = Field(min_length=1, max_length=128, pattern=_STRUCTURED_CODE)
    selected_candidate_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    action_intent_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    execution_receipt_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    outcome_ref: str = Field(min_length=1, max_length=256, pattern=_STRUCTURED_CODE)
    agency_attribution_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    agency_attribution_revision: int = Field(ge=1)
    observed_utility: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    alternatives: tuple[AlternativeOutcome, ...] = Field(min_length=1, max_length=8)
    signal: CounterfactualSignal
    signal_magnitude: float = Field(ge=0.0, le=0.5, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=0.8, allow_inf_nan=False)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    reason_codes: tuple[str, ...] = Field(min_length=1, max_length=16)
    created_at: datetime
    updated_at: datetime
    supersedes_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_record(self) -> "CounterfactualSimulation":
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("counterfactual timestamps must include timezones")
        if self.revision == 1 and self.supersedes_revision is not None:
            raise ValueError("initial counterfactual cannot supersede a revision")
        if self.revision > 1 and self.supersedes_revision != self.revision - 1:
            raise ValueError("counterfactual revisions must be contiguous")
        candidate_ids = [item.candidate_id for item in self.alternatives]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("counterfactual alternatives must be unique")
        if self.selected_candidate_id in candidate_ids:
            raise ValueError(
                "selected candidate cannot be a counterfactual alternative"
            )
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("counterfactual evidence references must be unique")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("counterfactual reason codes must be unique")
        for value in (
            self.simulation_id,
            self.decision_id,
            self.selected_candidate_id,
            self.action_intent_id,
            self.execution_receipt_id,
            self.outcome_ref,
            self.agency_attribution_id,
            *self.evidence_refs,
            *self.reason_codes,
        ):
            _validate_structured_value(value)
        if self.signal == CounterfactualSignal.NONE and self.signal_magnitude != 0.0:
            raise ValueError("no counterfactual signal must have zero magnitude")
        if self.signal != CounterfactualSignal.NONE and self.signal_magnitude == 0.0:
            raise ValueError("counterfactual affect requires non-zero magnitude")
        gap = (
            max(item.plausible_utility for item in self.alternatives)
            - self.observed_utility
        )
        expected_signal = (
            CounterfactualSignal.REGRET
            if gap > 0.05 and self.observed_utility < 0.0
            else CounterfactualSignal.MISSED_OPPORTUNITY
            if gap > 0.05
            else CounterfactualSignal.RELIEF
            if gap < -0.05
            else CounterfactualSignal.NONE
        )
        if self.signal != expected_signal:
            raise ValueError(
                "counterfactual signal must follow the bounded utility comparison"
            )
        if self.confidence > max(item.confidence for item in self.alternatives):
            raise ValueError("counterfactual confidence exceeds alternative evidence")
        if self.signal_magnitude > min(0.5, abs(gap) * self.confidence) + 1e-9:
            raise ValueError(
                "counterfactual affect exceeds confidence-weighted difference"
            )
        return self

    @property
    def reference(self) -> str:
        return f"counterfactual:{self.simulation_id}@{self.revision}"


class CounterfactualProjection(_StrictModel):
    schema_version: Literal[1] = 1
    simulation_id: str = Field(min_length=1, max_length=128, pattern=_STRUCTURED_CODE)
    simulation_revision: int = Field(ge=1)
    target: CounterfactualTarget
    subject_ref: str = Field(min_length=1, max_length=256, pattern=_STRUCTURED_CODE)
    applied_delta: float = Field(ge=-0.05, le=0.05, allow_inf_nan=False)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    applied_at: datetime

    @model_validator(mode="after")
    def validate_projection(self) -> "CounterfactualProjection":
        if self.applied_at.tzinfo is None:
            raise ValueError(
                "counterfactual projection timestamp must include timezone"
            )
        for value in (self.simulation_id, self.subject_ref, *self.evidence_refs):
            _validate_structured_value(value)
        return self


class CounterfactualState(_StrictModel):
    schema_version: Literal[1] = 1
    records: tuple[CounterfactualSimulation, ...] = ()
    projections: tuple[CounterfactualProjection, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> "CounterfactualState":
        seen: set[tuple[str, int]] = set()
        latest: dict[str, int] = {}
        for record in self.records:
            key = (record.simulation_id, record.revision)
            if (
                key in seen
                or record.revision != latest.get(record.simulation_id, 0) + 1
            ):
                raise ValueError(
                    "counterfactual history is duplicate or non-contiguous"
                )
            seen.add(key)
            latest[record.simulation_id] = record.revision
        projection_keys = [
            (
                item.simulation_id,
                item.simulation_revision,
                item.target,
                item.subject_ref,
            )
            for item in self.projections
        ]
        if len(projection_keys) != len(set(projection_keys)):
            raise ValueError("duplicate counterfactual projection")
        if any(
            (item.simulation_id, item.simulation_revision) not in seen
            for item in self.projections
        ):
            raise ValueError("counterfactual projection references an unknown revision")
        return self


class CounterfactualStore:
    """Append-only simulations with deduplicated, bounded learning projections."""

    MAX_REVISIONS = 4

    def __init__(
        self,
        *,
        load: Callable[[], object | None],
        save: Callable[[dict[str, object]], None],
        validate_chain: Callable[[CounterfactualSimulation], None],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._save = save
        self._validate_chain = validate_chain
        self.clock = clock or (lambda: datetime.now(UTC))
        raw = load()
        self.state = (
            CounterfactualState()
            if raw is None
            else CounterfactualState.model_validate(raw)
        )
        for record in self.state.records:
            self._validate_chain(record)
        self._persist()

    def list_current(self) -> tuple[CounterfactualSimulation, ...]:
        latest: dict[str, CounterfactualSimulation] = {}
        for record in self.state.records:
            latest[record.simulation_id] = record
        return tuple(
            sorted(
                latest.values(), key=lambda item: (item.created_at, item.simulation_id)
            )
        )

    def history(self, simulation_id: str) -> tuple[CounterfactualSimulation, ...]:
        records = tuple(
            item for item in self.state.records if item.simulation_id == simulation_id
        )
        if not records:
            raise ValueError(f"Unknown counterfactual simulation: {simulation_id}")
        return records

    def current_for_decision(self, decision_id: str) -> CounterfactualSimulation | None:
        return next(
            (item for item in self.list_current() if item.decision_id == decision_id),
            None,
        )

    def create(
        self,
        *,
        decision_id: str,
        selected_candidate_id: str,
        action_intent_id: str,
        execution_receipt_id: str,
        outcome_ref: str,
        agency_attribution_id: str,
        agency_attribution_revision: int,
        observed_utility: float,
        alternatives: tuple[AlternativeOutcome, ...],
        signal: CounterfactualSignal,
        signal_magnitude: float,
        confidence: float,
        evidence_refs: tuple[str, ...],
        reason_codes: tuple[str, ...],
        simulation_id: str | None = None,
    ) -> CounterfactualSimulation:
        existing = self.current_for_decision(decision_id)
        if existing is not None:
            return existing
        now = self.clock()
        record = CounterfactualSimulation(
            simulation_id=simulation_id or str(uuid4()),
            revision=1,
            decision_id=decision_id,
            selected_candidate_id=selected_candidate_id,
            action_intent_id=action_intent_id,
            execution_receipt_id=execution_receipt_id,
            outcome_ref=outcome_ref,
            agency_attribution_id=agency_attribution_id,
            agency_attribution_revision=agency_attribution_revision,
            observed_utility=observed_utility,
            alternatives=alternatives,
            signal=signal,
            signal_magnitude=signal_magnitude,
            confidence=confidence,
            evidence_refs=evidence_refs,
            reason_codes=reason_codes,
            created_at=now,
            updated_at=now,
        )
        self._append(record)
        return record

    def revise(
        self,
        simulation_id: str,
        *,
        expected_revision: int,
        agency_attribution_revision: int,
        alternatives: tuple[AlternativeOutcome, ...],
        signal: CounterfactualSignal,
        signal_magnitude: float,
        confidence: float,
        evidence_refs: tuple[str, ...],
        reason_code: str,
    ) -> CounterfactualSimulation:
        current = self.history(simulation_id)[-1]
        if current.revision != expected_revision:
            raise ValueError("counterfactual simulation revision conflict")
        if current.revision >= self.MAX_REVISIONS:
            raise ValueError("counterfactual simulation revision budget exhausted")
        novel = tuple(
            item for item in evidence_refs if item not in current.evidence_refs
        )
        if (
            not novel
            and agency_attribution_revision == current.agency_attribution_revision
        ):
            raise ValueError("counterfactual revision requires new structured evidence")
        record = CounterfactualSimulation(
            **current.model_dump(
                exclude={
                    "revision",
                    "agency_attribution_revision",
                    "alternatives",
                    "signal",
                    "signal_magnitude",
                    "confidence",
                    "evidence_refs",
                    "reason_codes",
                    "updated_at",
                    "supersedes_revision",
                }
            ),
            revision=current.revision + 1,
            agency_attribution_revision=agency_attribution_revision,
            alternatives=alternatives,
            signal=signal,
            signal_magnitude=signal_magnitude,
            confidence=confidence,
            evidence_refs=tuple(dict.fromkeys((*current.evidence_refs, *novel))),
            reason_codes=tuple(dict.fromkeys((*current.reason_codes, reason_code))),
            updated_at=self.clock(),
            supersedes_revision=current.revision,
        )
        self._append(record)
        return record

    def record_projection(
        self,
        record: CounterfactualSimulation,
        target: CounterfactualTarget,
        *,
        subject_ref: str,
        applied_delta: float,
        evidence_refs: tuple[str, ...],
    ) -> CounterfactualProjection:
        existing = next(
            (
                item
                for item in self.state.projections
                if item.simulation_id == record.simulation_id
                and item.simulation_revision == record.revision
                and item.target == target
                and item.subject_ref == subject_ref
            ),
            None,
        )
        if existing is not None:
            return existing
        projection = CounterfactualProjection(
            simulation_id=record.simulation_id,
            simulation_revision=record.revision,
            target=target,
            subject_ref=subject_ref,
            applied_delta=max(-0.05, min(0.05, applied_delta)),
            evidence_refs=tuple(dict.fromkeys((record.reference, *evidence_refs))),
            applied_at=self.clock(),
        )
        self.state = self.state.model_copy(
            update={"projections": (*self.state.projections, projection)}
        )
        self._persist()
        return projection

    def calibration(self, target: CounterfactualTarget, subject_ref: str) -> float:
        return max(
            -0.1,
            min(
                0.1,
                sum(
                    item.applied_delta
                    for item in self.state.projections
                    if item.target == target and item.subject_ref == subject_ref
                ),
            ),
        )

    def _append(self, record: CounterfactualSimulation) -> None:
        self._validate_chain(record)
        self.state = self.state.model_copy(
            update={"records": (*self.state.records, record)}
        )
        self._persist()

    def _persist(self) -> None:
        self._save(self.state.model_dump(mode="json"))


def _validate_structured_value(value: str) -> None:
    if re.fullmatch(_STRUCTURED_CODE, value) is None:
        raise ValueError("counterfactual fields must contain structured codes only")
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    if any(alias in normalized for alias in _PRIVATE_ALIASES):
        raise ValueError("counterfactual simulation cannot contain private reasoning")
