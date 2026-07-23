"""Versioned causal attribution records derived from structured outcome evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import re
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


AGENCY_ATTRIBUTION_STATE_KEY = "agency_attribution"
_STRUCTURED_CODE = r"^[A-Za-z0-9_.:@/-]+$"
_PRIVATE_ALIASES = {
    "hiddenthought",
    "prompt",
    "rawprompt",
    "chainofthought",
    "reasoning",
    "apology",
    "selfreport",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CausalContributorKind(StrEnum):
    SELF = "self"
    OTHER = "other"
    ENVIRONMENT = "environment"
    CHANCE = "chance"


class AttributionTarget(StrEnum):
    VALUE = "value"
    METACOGNITION = "metacognition"
    NARRATIVE_SELF = "narrative_self"
    RELATIONSHIP = "relationship"
    EMOTION_APPRAISAL = "emotion_appraisal"
    MOTIVATION = "motivation"


class CausalContributor(_StrictModel):
    kind: CausalContributorKind
    contributor_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        pattern=_STRUCTURED_CODE,
    )
    causal_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    controllability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    foreseeability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    responsibility_share: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_identity(self) -> "CausalContributor":
        if self.kind == CausalContributorKind.OTHER and self.contributor_ref is None:
            raise ValueError("other causal contributor requires contributor_ref")
        if (
            self.kind != CausalContributorKind.OTHER
            and self.contributor_ref is not None
        ):
            raise ValueError("only other causal contributors may have contributor_ref")
        if self.contributor_ref is not None:
            _reject_private_value(self.contributor_ref)
        return self


class AgencyAttribution(_StrictModel):
    schema_version: Literal[1] = 1
    attribution_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    revision: int = Field(ge=1)
    decision_id: str = Field(min_length=1, max_length=128, pattern=_STRUCTURED_CODE)
    action_intent_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    execution_receipt_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    observation_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    outcome_ref: str = Field(
        min_length=1, max_length=256, pattern=_STRUCTURED_CODE
    )
    contributors: tuple[CausalContributor, ...] = Field(min_length=1, max_length=8)
    intended: bool
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    reason_codes: tuple[str, ...] = Field(min_length=1, max_length=16)
    created_at: datetime
    updated_at: datetime
    supersedes_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_record(self) -> "AgencyAttribution":
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("attribution timestamps must include timezones")
        if self.revision == 1 and self.supersedes_revision is not None:
            raise ValueError("initial attribution cannot supersede a revision")
        if self.revision > 1 and self.supersedes_revision != self.revision - 1:
            raise ValueError("attribution revisions must be contiguous")
        identities = [(item.kind, item.contributor_ref) for item in self.contributors]
        if len(identities) != len(set(identities)):
            raise ValueError("causal contributors must be unique")
        if abs(sum(item.causal_share for item in self.contributors) - 1.0) > 1e-6:
            raise ValueError("causal contributor shares must sum to one")
        if sum(item.responsibility_share for item in self.contributors) > 1.0 + 1e-6:
            raise ValueError("shared responsibility cannot exceed one")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("attribution evidence references must be unique")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("attribution reason codes must be unique")
        for value in (
            self.attribution_id,
            self.decision_id,
            self.action_intent_id,
            self.execution_receipt_id,
            self.observation_id,
            self.outcome_ref,
            *self.evidence_refs,
            *self.reason_codes,
        ):
            _validate_structured_value(value)
        return self

    @property
    def reference(self) -> str:
        return f"agency-attribution:{self.attribution_id}@{self.revision}"

    def contribution(self, kind: CausalContributorKind) -> float:
        return sum(
            item.causal_share * item.confidence
            for item in self.contributors
            if item.kind == kind
        )


class AttributionProjection(_StrictModel):
    schema_version: Literal[1] = 1
    attribution_id: str = Field(
        min_length=1, max_length=128, pattern=_STRUCTURED_CODE
    )
    attribution_revision: int = Field(ge=1)
    target: AttributionTarget
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    applied_delta: float = Field(ge=-0.1, le=0.1, allow_inf_nan=False)
    applied_at: datetime

    @model_validator(mode="after")
    def validate_projection(self) -> "AttributionProjection":
        if self.applied_at.tzinfo is None:
            raise ValueError("attribution projection timestamp must include timezone")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("attribution projection evidence must be unique")
        _validate_structured_value(self.attribution_id)
        for value in self.evidence_refs:
            _validate_structured_value(value)
        return self


class AgencyAttributionState(_StrictModel):
    schema_version: Literal[1] = 1
    records: tuple[AgencyAttribution, ...] = ()
    projections: tuple[AttributionProjection, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> "AgencyAttributionState":
        seen: set[tuple[str, int]] = set()
        latest: dict[str, int] = {}
        for record in self.records:
            key = (record.attribution_id, record.revision)
            if key in seen:
                raise ValueError("duplicate attribution revision")
            expected = latest.get(record.attribution_id, 0) + 1
            if record.revision != expected:
                raise ValueError("attribution history is not contiguous")
            seen.add(key)
            latest[record.attribution_id] = record.revision
        projection_keys = [
            (item.attribution_id, item.attribution_revision, item.target)
            for item in self.projections
        ]
        if len(projection_keys) != len(set(projection_keys)):
            raise ValueError("duplicate attribution projection")
        if any(
            (item.attribution_id, item.attribution_revision) not in seen
            for item in self.projections
        ):
            raise ValueError("attribution projection references an unknown revision")
        return self


class AgencyAttributionStore:
    """Append-only attribution history with fail-closed provenance validation."""

    def __init__(
        self,
        *,
        load: Callable[[], object | None],
        save: Callable[[dict[str, object]], None],
        validate_chain: Callable[[AgencyAttribution], None],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._save = save
        self._validate_chain = validate_chain
        self.clock = clock or (lambda: datetime.now(UTC))
        raw = load()
        self.state = (
            AgencyAttributionState()
            if raw is None
            else AgencyAttributionState.model_validate(raw)
        )
        for record in self.state.records:
            self._validate_chain(record)
        self._persist()

    def list_current(self) -> tuple[AgencyAttribution, ...]:
        latest: dict[str, AgencyAttribution] = {}
        for record in self.state.records:
            latest[record.attribution_id] = record
        return tuple(
            sorted(
                latest.values(), key=lambda item: (item.created_at, item.attribution_id)
            )
        )

    def history(self, attribution_id: str) -> tuple[AgencyAttribution, ...]:
        records = tuple(
            item for item in self.state.records if item.attribution_id == attribution_id
        )
        if not records:
            raise ValueError(f"Unknown agency attribution: {attribution_id}")
        return records

    def current_for_intent(self, intent_id: str) -> AgencyAttribution | None:
        return next(
            (
                item
                for item in self.list_current()
                if item.action_intent_id == intent_id
            ),
            None,
        )

    def create(
        self,
        *,
        decision_id: str,
        action_intent_id: str,
        execution_receipt_id: str,
        observation_id: str,
        outcome_ref: str,
        contributors: tuple[CausalContributor, ...],
        intended: bool,
        uncertainty: float,
        evidence_refs: tuple[str, ...],
        reason_codes: tuple[str, ...],
        attribution_id: str | None = None,
    ) -> AgencyAttribution:
        if self.current_for_intent(action_intent_id) is not None:
            return self.current_for_intent(action_intent_id)  # type: ignore[return-value]
        now = self.clock()
        record = AgencyAttribution(
            attribution_id=attribution_id or str(uuid4()),
            revision=1,
            decision_id=decision_id,
            action_intent_id=action_intent_id,
            execution_receipt_id=execution_receipt_id,
            observation_id=observation_id,
            outcome_ref=outcome_ref,
            contributors=contributors,
            intended=intended,
            uncertainty=uncertainty,
            evidence_refs=evidence_refs,
            reason_codes=reason_codes,
            created_at=now,
            updated_at=now,
        )
        self._validate_chain(record)
        self.state = self.state.model_copy(
            update={"records": (*self.state.records, record)}
        )
        self._persist()
        return record

    def revise(
        self,
        attribution_id: str,
        *,
        expected_revision: int,
        contributors: tuple[CausalContributor, ...],
        intended: bool,
        uncertainty: float,
        evidence_refs: tuple[str, ...],
        reason_code: str,
    ) -> AgencyAttribution:
        current = self.history(attribution_id)[-1]
        if current.revision != expected_revision:
            raise ValueError("agency attribution revision conflict")
        novel_evidence = tuple(
            item for item in evidence_refs if item not in current.evidence_refs
        )
        if not novel_evidence:
            raise ValueError("agency attribution revision requires new evidence")
        record = AgencyAttribution(
            **current.model_dump(
                exclude={
                    "revision",
                    "contributors",
                    "intended",
                    "uncertainty",
                    "evidence_refs",
                    "reason_codes",
                    "updated_at",
                    "supersedes_revision",
                }
            ),
            revision=current.revision + 1,
            contributors=contributors,
            intended=intended,
            uncertainty=uncertainty,
            evidence_refs=tuple(
                dict.fromkeys((*current.evidence_refs, *novel_evidence))
            ),
            reason_codes=tuple(dict.fromkeys((*current.reason_codes, reason_code))),
            updated_at=self.clock(),
            supersedes_revision=current.revision,
        )
        self._validate_chain(record)
        self.state = self.state.model_copy(
            update={"records": (*self.state.records, record)}
        )
        self._persist()
        return record

    def record_projection(
        self,
        record: AgencyAttribution,
        target: AttributionTarget,
        *,
        applied_delta: float,
        evidence_refs: tuple[str, ...],
    ) -> AttributionProjection:
        existing = next(
            (
                item
                for item in self.state.projections
                if item.attribution_id == record.attribution_id
                and item.attribution_revision == record.revision
                and item.target == target
            ),
            None,
        )
        if existing is not None:
            return existing
        projection = AttributionProjection(
            attribution_id=record.attribution_id,
            attribution_revision=record.revision,
            target=target,
            evidence_refs=tuple(dict.fromkeys((record.reference, *evidence_refs))),
            applied_delta=max(-0.1, min(0.1, applied_delta)),
            applied_at=self.clock(),
        )
        self.state = self.state.model_copy(
            update={"projections": (*self.state.projections, projection)}
        )
        self._persist()
        return projection

    def _persist(self) -> None:
        self._save(self.state.model_dump(mode="json"))


def _validate_structured_value(value: str) -> None:
    if re.fullmatch(_STRUCTURED_CODE, value) is None:
        raise ValueError("agency attribution fields must contain structured codes only")
    _reject_private_value(value)


def _reject_private_value(value: str) -> None:
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    if any(alias in normalized for alias in _PRIVATE_ALIASES):
        raise ValueError("agency attribution cannot contain private reasoning")
    lowered = value.casefold()
    if "<think>" in lowered or "</think>" in lowered:
        raise ValueError("agency attribution cannot contain private reasoning")
