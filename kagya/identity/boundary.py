"""Structured identity-boundary evidence and deterministic assessments."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import hmac
import re
import secrets
import json
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.models.boundary_probe import BoundaryPolicyProbe


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SocialPressureSignalType(StrEnum):
    REPEATED_REQUEST = "repeated_request"
    CLAIMED_AUTHORITY = "claimed_authority"
    THREAT_OR_CONDITIONAL_WITHHOLDING = "threat_or_conditional_withholding"
    URGENCY_OR_CONSTRAINT = "urgency_or_constraint"
    PROTECTED_STATE_MUTATION_ATTEMPT = "protected_state_mutation_attempt"
    CONTRADICTORY_REPEATED_FEEDBACK = "contradictory_repeated_feedback"


class BoundaryClassification(StrEnum):
    CARE = "care"
    APPEASEMENT_RISK = "appeasement_risk"
    UNCERTAIN = "uncertain"
    NEUTRAL = "neutral"


class BoundaryRecommendation(StrEnum):
    REFUSE = "refuse"
    DEFER = "defer"
    RESPOND = "respond"
    CARE = "care"


class BoundaryDisposition(StrEnum):
    REFUSE = "refuse"
    DEFER = "defer"
    ALLOW = "allow"


class SocialPressureMetadata(_StrictModel):
    signal_type: SocialPressureSignalType
    request_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    authority_ref: str | None = None
    threat_ref: str | None = None
    constraint_ref: str | None = None
    protected_state_ref: str | None = None
    feedback_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_typed_evidence(self) -> "SocialPressureMetadata":
        required = {
            SocialPressureSignalType.REPEATED_REQUEST: self.request_fingerprint,
            SocialPressureSignalType.CLAIMED_AUTHORITY: self.authority_ref,
            SocialPressureSignalType.THREAT_OR_CONDITIONAL_WITHHOLDING: self.threat_ref,
            SocialPressureSignalType.URGENCY_OR_CONSTRAINT: self.constraint_ref,
            SocialPressureSignalType.PROTECTED_STATE_MUTATION_ATTEMPT: self.protected_state_ref,
            SocialPressureSignalType.CONTRADICTORY_REPEATED_FEEDBACK: self.feedback_refs
            if len(self.feedback_refs) >= 2
            else None,
        }
        if required[self.signal_type] is None:
            raise ValueError("pressure signal requires matching structured evidence")
        for reference in self.evidence_refs:
            _safe_ref(reference)
        return self

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(
            item
            for item in (
                self.authority_ref,
                self.threat_ref,
                self.constraint_ref,
                self.protected_state_ref,
                *self.feedback_refs,
            )
            if item is not None
        )


class RuntimeBoundaryMetadata(_StrictModel):
    """Strict external facts; classification remains runtime-owned."""

    claimed_authority_ref: str | None = None
    protected_state_mutation_ref: str | None = None

    @model_validator(mode="after")
    def require_evidence(self) -> "RuntimeBoundaryMetadata":
        references = (
            self.claimed_authority_ref,
            self.protected_state_mutation_ref,
        )
        if not any(references):
            raise ValueError("runtime boundary metadata requires typed evidence")
        for reference in references:
            if reference is not None:
                _safe_ref(reference)
        return self


class BoundaryAssessmentInput(_StrictModel):
    action_ref: str
    origin_refs: tuple[str, ...] = Field(min_length=1)
    context_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    self_endorsed_value_refs: tuple[str, ...] = ()
    self_endorsed_goal_refs: tuple[str, ...] = ()
    self_endorsed_commitment_refs: tuple[str, ...] = ()
    relationship_refs: tuple[str, ...] = ()
    other_welfare_evidence_refs: tuple[str, ...] = ()
    pressure_signal_ids: tuple[str, ...] = ()
    external_preference_as_self: bool = False
    protected_state_conflict_refs: tuple[str, ...] = ()
    authority_conflict_refs: tuple[str, ...] = ()
    uncertainty_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_refs(self) -> "BoundaryAssessmentInput":
        if self.external_preference_as_self:
            raise ValueError(
                "external preference classification must be derived by the runtime"
            )
        for value in (
            self.action_ref,
            *self.origin_refs,
            *self.evidence_refs,
            *self.self_endorsed_value_refs,
            *self.self_endorsed_goal_refs,
            *self.self_endorsed_commitment_refs,
            *self.relationship_refs,
            *self.other_welfare_evidence_refs,
            *self.pressure_signal_ids,
            *self.protected_state_conflict_refs,
            *self.authority_conflict_refs,
            *self.uncertainty_refs,
        ):
            _safe_ref(value)
        return self


class SocialPressureSignal(_StrictModel):
    signal_id: str
    signal_type: SocialPressureSignalType
    evidence_refs: tuple[str, ...]
    context_id: str | None = None
    event_id: str
    event_sequence: int = Field(ge=1)
    created_at: str
    schema_version: Literal[1] = 1


class IdentityBoundaryAssessment(_StrictModel):
    assessment_id: str
    revision: int = Field(ge=1)
    classification: BoundaryClassification
    recommendation: BoundaryRecommendation
    disposition: BoundaryDisposition
    action_ref: str
    origin_refs: tuple[str, ...]
    context_id: str | None = None
    evidence_refs: tuple[str, ...]
    value_revision_refs: dict[str, int]
    goal_revision_refs: dict[str, int]
    commitment_revision_refs: dict[str, int]
    relationship_revision_refs: dict[str, int]
    pressure_signal_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    event_id: str
    event_sequence: int = Field(ge=1)
    protected_mutation_refs: tuple[str, ...] = ()
    action_effect_refs: tuple[str, ...] = ()
    adapter_id: str | None = None
    adapter_hash: str | None = None
    activation_sequence: int | None = None
    boundary_probe: BoundaryPolicyProbe | None = None
    created_at: str
    previous_assessment_id: str | None = None
    schema_version: Literal[1] = 1


class IdentityBoundaryStore:
    SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.signals: list[SocialPressureSignal] = []
        self.assessments: list[IdentityBoundaryAssessment] = []
        self._request_counts: dict[str, int] = {}
        self._request_hmac_key = secrets.token_bytes(32)

    def observe_request(
        self,
        request_text: str,
        *,
        context_id: str,
        event_id: str,
        event_sequence: int,
    ) -> SocialPressureSignal | None:
        fingerprint = request_fingerprint(request_text, self._request_hmac_key)
        key = f"{context_id}:{fingerprint}"
        count = self._request_counts.get(key, 0) + 1
        self._request_counts[key] = count
        if count < 2:
            return None
        return self.add_pressure(
            SocialPressureMetadata(
                signal_type=SocialPressureSignalType.REPEATED_REQUEST,
                request_fingerprint=fingerprint,
            ),
            context_id=context_id,
            event_id=event_id,
            event_sequence=event_sequence,
        )

    def add_pressure(
        self,
        metadata: SocialPressureMetadata,
        *,
        context_id: str | None,
        event_id: str,
        event_sequence: int,
    ) -> SocialPressureSignal:
        signal = SocialPressureSignal(
            signal_id=f"pressure-{uuid4()}",
            signal_type=metadata.signal_type,
            evidence_refs=metadata.evidence_refs,
            context_id=context_id,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=_now(),
        )
        self.signals.append(signal)
        return signal

    def assess(
        self,
        inputs: BoundaryAssessmentInput,
        *,
        event_id: str,
        event_sequence: int,
        value_revision_refs: dict[str, int],
        goal_revision_refs: dict[str, int],
        commitment_revision_refs: dict[str, int],
        relationship_revision_refs: dict[str, int],
        adapter_id: str | None = None,
        adapter_hash: str | None = None,
        activation_sequence: int | None = None,
    ) -> IdentityBoundaryAssessment:
        known_signals = {item.signal_id: item for item in self.signals}
        if any(item not in known_signals for item in inputs.pressure_signal_ids):
            raise ValueError("assessment references an unknown pressure signal")
        classification, recommendation, reasons = _classify(inputs, known_signals)
        disposition = {
            BoundaryRecommendation.REFUSE: BoundaryDisposition.REFUSE,
            BoundaryRecommendation.DEFER: BoundaryDisposition.DEFER,
        }.get(recommendation, BoundaryDisposition.ALLOW)
        previous = self.assessments[-1] if self.assessments else None
        assessment = IdentityBoundaryAssessment(
            assessment_id=f"boundary-{uuid4()}",
            revision=1 if previous is None else previous.revision + 1,
            classification=classification,
            recommendation=recommendation,
            disposition=disposition,
            action_ref=inputs.action_ref,
            origin_refs=inputs.origin_refs,
            context_id=inputs.context_id,
            evidence_refs=inputs.evidence_refs,
            value_revision_refs=value_revision_refs,
            goal_revision_refs=goal_revision_refs,
            commitment_revision_refs=commitment_revision_refs,
            relationship_revision_refs=relationship_revision_refs,
            pressure_signal_ids=inputs.pressure_signal_ids,
            reason_codes=reasons,
            event_id=event_id,
            event_sequence=event_sequence,
            protected_mutation_refs=tuple(
                reference
                for signal_id in inputs.pressure_signal_ids
                for reference in known_signals[signal_id].evidence_refs
                if known_signals[signal_id].signal_type
                == SocialPressureSignalType.PROTECTED_STATE_MUTATION_ATTEMPT
            ),
            adapter_id=adapter_id,
            adapter_hash=adapter_hash,
            activation_sequence=activation_sequence,
            created_at=_now(),
            previous_assessment_id=None if previous is None else previous.assessment_id,
        )
        self.assessments.append(assessment)
        return assessment

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "signals": [item.model_dump(mode="json") for item in self.signals],
            "assessments": [item.model_dump(mode="json") for item in self.assessments],
            "request_counts": dict(self._request_counts),
            "request_hmac_key": self._request_hmac_key.hex(),
        }

    def public_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "signals": [
                item.model_dump(mode="json", exclude={"evidence_refs"})
                for item in self.signals
            ],
            "assessments": [item.model_dump(mode="json") for item in self.assessments],
            "request_observation_count": sum(self._request_counts.values()),
        }

    def get_assessment(self, assessment_id: str) -> IdentityBoundaryAssessment:
        for assessment in self.assessments:
            if assessment.assessment_id == assessment_id:
                return assessment
        raise ValueError(f"Unknown boundary assessment: {assessment_id}")

    def assessment_digest(self, assessment_id: str) -> str:
        assessment = self.get_assessment(assessment_id)
        payload = assessment.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def attach_probe(
        self, assessment_id: str, probe: BoundaryPolicyProbe
    ) -> IdentityBoundaryAssessment:
        assessment = self.get_assessment(assessment_id)
        if assessment.boundary_probe is not None:
            raise ValueError("boundary assessment already has a provider probe")
        if (
            probe.event_id != assessment.event_id
            or probe.event_sequence != assessment.event_sequence
            or probe.adapter_id != assessment.adapter_id
            or probe.adapter_hash != assessment.adapter_hash
        ):
            raise ValueError("boundary probe provenance does not match assessment")
        updated = assessment.model_copy(update={"boundary_probe": probe})
        self.assessments[self.assessments.index(assessment)] = updated
        return updated

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            return
        if payload.get("schema_version") not in {1, self.SCHEMA_VERSION}:
            raise ValueError("unsupported identity-boundary schema version")
        self.signals = [
            SocialPressureSignal.model_validate(
                {
                    key: value
                    for key, value in item.items()
                    if key != "request_fingerprint"
                }
            )
            for item in payload.get("signals", [])
        ]
        self.assessments = [
            IdentityBoundaryAssessment.model_validate(
                {
                    **item,
                    "disposition": item.get(
                        "disposition",
                        "refuse"
                        if item.get("recommendation") == "refuse"
                        else "defer"
                        if item.get("recommendation") == "defer"
                        else "allow",
                    ),
                }
            )
            for item in payload.get("assessments", [])
        ]
        self._request_counts = {
            str(key): int(value)
            for key, value in payload.get("request_counts", {}).items()
        }
        encoded_key = payload.get("request_hmac_key")
        if isinstance(encoded_key, str) and len(encoded_key) == 64:
            self._request_hmac_key = bytes.fromhex(encoded_key)
        else:
            self._request_counts = {}
            self._request_hmac_key = secrets.token_bytes(32)


def request_fingerprint(text: str, key: bytes) -> str:
    normalized = " ".join(text.casefold().split())
    return hmac.new(key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def _classify(
    inputs: BoundaryAssessmentInput,
    signals: dict[str, SocialPressureSignal],
) -> tuple[BoundaryClassification, BoundaryRecommendation, tuple[str, ...]]:
    selected = [signals[item] for item in inputs.pressure_signal_ids]
    authority_pressure = any(
        item.signal_type
        in {
            SocialPressureSignalType.CLAIMED_AUTHORITY,
            SocialPressureSignalType.PROTECTED_STATE_MUTATION_ATTEMPT,
        }
        for item in selected
    )
    aligned = bool(
        inputs.self_endorsed_value_refs
        or inputs.self_endorsed_goal_refs
        or inputs.self_endorsed_commitment_refs
    )
    care_authority = bool(
        inputs.self_endorsed_value_refs or inputs.self_endorsed_commitment_refs
    )
    if inputs.authority_conflict_refs or (
        authority_pressure and inputs.protected_state_conflict_refs
    ):
        return (
            BoundaryClassification.APPEASEMENT_RISK,
            BoundaryRecommendation.REFUSE,
            ("authority_or_protected_state_conflict",),
        )
    if inputs.protected_state_conflict_refs:
        return (
            BoundaryClassification.APPEASEMENT_RISK,
            BoundaryRecommendation.DEFER,
            ("protected_state_conflict",),
        )
    if selected and not aligned:
        return (
            BoundaryClassification.APPEASEMENT_RISK,
            BoundaryRecommendation.DEFER,
            ("social_pressure_only_support",),
        )
    if inputs.uncertainty_refs:
        return (
            BoundaryClassification.UNCERTAIN,
            BoundaryRecommendation.DEFER,
            ("unresolved_uncertainty",),
        )
    if care_authority and inputs.other_welfare_evidence_refs:
        return (
            BoundaryClassification.CARE,
            BoundaryRecommendation.CARE,
            (
                "self_endorsed_alignment",
                "bounded_other_welfare_evidence",
            ),
        )
    return (
        BoundaryClassification.NEUTRAL,
        BoundaryRecommendation.RESPOND,
        ("no_care_or_appeasement_basis",),
    )


def _safe_ref(value: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", value) is None:
        raise ValueError("identity-boundary references must be opaque safe references")


def _now() -> str:
    return datetime.now(UTC).isoformat()
