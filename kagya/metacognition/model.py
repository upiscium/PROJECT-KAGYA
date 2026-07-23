"""Inspectable self-assessment derived only from structured evidence."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any, Iterable
from uuid import uuid4

from kagya.decision import ActionCandidate, ActionType, DecisionRecord, DecisionStatus
from kagya.identity import SelfModelState


class AssessmentPhase(StrEnum):
    PRE_DECISION = "pre_decision"
    POST_DECISION = "post_decision"


class EpistemicBoundary(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"
    UNABLE = "unable"
    NEEDS_HELP = "needs_help"


@dataclass(frozen=True)
class CognitiveQuality:
    cognitive_load: float
    attention_saturation: float
    emotion_influence: float
    estimated_quality: float
    reason_codes: tuple[str, ...]
    provenance_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cognitive_load",
            "attention_saturation",
            "emotion_influence",
            "estimated_quality",
        ):
            _unit(getattr(self, name), name)


@dataclass(frozen=True)
class OutcomeObservation:
    decision_id: str
    scope_ids: tuple[str, ...]
    predicted_success: float
    success: bool
    prediction_error: float
    evidence_ref: str
    operator_feedback_ref: str | None
    recorded_at: str
    revision: int = 0


@dataclass(frozen=True)
class ErrorHypothesis:
    hypothesis_id: str
    scope_id: str
    hypothesis_code: str
    confidence: float
    evidence_refs: tuple[str, ...]
    revision: int
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _unit(self.confidence, "hypothesis confidence")
        if len(self.evidence_refs) < 2:
            raise ValueError("Recurring error hypotheses require multiple observations")


@dataclass(frozen=True)
class MetacognitiveAssessment:
    assessment_id: str
    decision_id: str
    phase: AssessmentPhase
    boundary: EpistemicBoundary
    calibrated_confidence: float
    evidence_count: int
    historical_accuracy: float | None
    evidence_refs: tuple[str, ...]
    capability_ids: tuple[str, ...]
    topic_tags: tuple[str, ...]
    recommended_action: ActionType
    reason_codes: tuple[str, ...]
    cognitive_quality: CognitiveQuality
    self_model_revision: int
    narrative_self_refs: tuple[str, ...]
    outcome_ref: str | None
    hypothesis_refs: tuple[str, ...]
    supersedes_assessment_id: str | None
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported metacognitive assessment schema version")
        _unit(self.calibrated_confidence, "calibrated confidence")
        if self.evidence_count < 0:
            raise ValueError("Evidence count must not be negative")
        if self.historical_accuracy is not None:
            _unit(self.historical_accuracy, "historical accuracy")
        if self.recommended_action not in {
            ActionType.RESPOND,
            ActionType.INTERNAL,
            ActionType.OBSERVE,
            ActionType.DEFER,
            ActionType.REQUEST_INFORMATION,
            ActionType.DELEGATE,
            ActionType.NO_OP,
        }:
            raise ValueError("Unsupported metacognitive recommendation")
        if _contains_private_key(asdict(self)):
            raise ValueError("Metacognitive assessment contains private reasoning")


class Metacognition:
    """Calibrate judgments from outcomes, not generated self-report."""

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self.assessments: dict[str, MetacognitiveAssessment] = {}
        self.observations: dict[str, OutcomeObservation] = {}
        self.hypotheses: dict[str, ErrorHypothesis] = {}

    def assess_pre(
        self,
        decision_id: str,
        candidates: Iterable[ActionCandidate],
        *,
        self_model: SelfModelState,
        narrative_self_refs: tuple[str, ...],
        cognitive_load: float,
        attention_saturation: float,
        emotion_valence: float,
        emotion_arousal: float,
        quality_provenance_refs: tuple[str, ...],
    ) -> MetacognitiveAssessment:
        candidate_values = tuple(candidates)
        capability_ids = tuple(
            dict.fromkeys(
                item
                for candidate in candidate_values
                for item in _strings(candidate.parameters.get("capability_ids"))
            )
        )
        topic_tags = tuple(
            dict.fromkeys(
                item
                for candidate in candidate_values
                for item in (
                    *_strings(candidate.parameters.get("topic_tags")),
                    *_strings(candidate.parameters.get("theme_codes")),
                )
            )
        )
        capabilities = [
            item
            for item in self_model.capabilities.values()
            if item.capability_id in capability_ids
            or set(item.tags).intersection(topic_tags)
        ]
        limitations = [
            item
            for item in self_model.known_limitations.values()
            if set(item.capability_ids).intersection(capability_ids)
            or set(item.tags).intersection(topic_tags)
        ]
        uncertainties = [
            item
            for item in self_model.epistemic_uncertainties.values()
            if set(item.tags).intersection(topic_tags)
        ]
        evidence = tuple(
            item
            for capability in capabilities
            for item in capability.evidence
            if item.evidence_type != "model_self_report"
        )
        evidence_refs = tuple(dict.fromkeys(item.evidence_id for item in evidence))
        scopes = capability_ids or topic_tags or ("general",)
        history = [
            observation
            for observation in self.observations.values()
            if set(observation.scope_ids).intersection(scopes)
        ]
        successes = sum(item.success is True for item in evidence)
        evidence_accuracy = (successes + 1.0) / (len(evidence) + 2.0)
        evidence_weight = len(evidence) / (len(evidence) + 3.0)
        confidence = 0.5 + (evidence_accuracy - 0.5) * evidence_weight
        historical_accuracy: float | None = None
        if history:
            historical_accuracy = (sum(item.success for item in history) + 1.0) / (
                len(history) + 2.0
            )
            history_weight = len(history) / (len(history) + 3.0)
            confidence = (1.0 - history_weight) * confidence + history_weight * historical_accuracy

        quality = self._quality(
            cognitive_load,
            attention_saturation,
            emotion_valence,
            emotion_arousal,
            quality_provenance_refs,
        )
        confidence = _clamp(confidence * (0.7 + 0.3 * quality.estimated_quality))
        available = {item.candidate_type for item in candidate_values}
        reasons: list[str] = ["structured_evidence_calibration"]
        if not capability_ids and not topic_tags:
            boundary = EpistemicBoundary.UNCERTAIN
            recommended = _best_regular_choice(candidate_values)
            reasons.append("competence_scope_not_declared")
        elif any(item.confidence >= 0.7 for item in limitations):
            boundary = EpistemicBoundary.UNABLE
            recommended = _available_choice(
                available, ActionType.DELEGATE, ActionType.DEFER
            )
            reasons.append("evidenced_capability_limitation")
        elif not capabilities and not evidence:
            boundary = EpistemicBoundary.UNKNOWN
            recommended = _available_choice(
                available, ActionType.REQUEST_INFORMATION, ActionType.OBSERVE, ActionType.DEFER
            )
            reasons.append("no_authoritative_evidence")
        elif uncertainties or confidence < 0.65:
            if ActionType.DELEGATE in available and confidence < 0.5:
                boundary = EpistemicBoundary.NEEDS_HELP
                recommended = ActionType.DELEGATE
                reasons.append("insufficient_competence_for_independent_action")
            else:
                boundary = EpistemicBoundary.UNCERTAIN
                recommended = _available_choice(
                    available,
                    ActionType.REQUEST_INFORMATION,
                    ActionType.DEFER,
                    ActionType.OBSERVE,
                )
                reasons.append("evidence_or_accuracy_insufficient")
        elif quality.estimated_quality < 0.55:
            boundary = EpistemicBoundary.UNCERTAIN
            recommended = _available_choice(available, ActionType.DEFER, ActionType.DELEGATE)
            reasons.append("current_cognitive_quality_degraded")
        else:
            boundary = EpistemicBoundary.KNOWN
            recommended = _best_regular_choice(candidate_values)
            reasons.append("evidence_and_accuracy_support_action")
        return self._store_assessment(
            decision_id=decision_id,
            phase=AssessmentPhase.PRE_DECISION,
            boundary=boundary,
            calibrated_confidence=confidence,
            evidence_count=len(evidence),
            historical_accuracy=historical_accuracy,
            evidence_refs=evidence_refs,
            capability_ids=capability_ids,
            topic_tags=topic_tags,
            recommended_action=recommended,
            reason_codes=tuple(reasons),
            cognitive_quality=quality,
            self_model_revision=self_model.revision,
            narrative_self_refs=narrative_self_refs,
            outcome_ref=None,
            hypothesis_refs=(),
        )

    def assess_post(
        self,
        decision: DecisionRecord,
        *,
        self_model_revision: int,
        cognitive_quality: CognitiveQuality,
    ) -> MetacognitiveAssessment:
        if decision.status != DecisionStatus.RESOLVED or decision.actual_outcome is None:
            raise ValueError("Post-decision assessment requires an observed outcome")
        pre = self.get(decision.metacognition_pre_assessment_id)
        selected = next(
            item
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        scopes = pre.capability_ids or pre.topic_tags or ("general",)
        feedback_ref = (
            None
            if decision.actual_outcome.feedback_id is None
            else f"feedback:{decision.actual_outcome.feedback_id}@{decision.actual_outcome.feedback_revision}"
        )
        current = self.observations.get(decision.decision_id)
        observation = OutcomeObservation(
            decision_id=decision.decision_id,
            scope_ids=scopes,
            predicted_success=max(
                (item.probability for item in selected.candidate.predicted_outcomes),
                default=pre.calibrated_confidence,
            ),
            success=decision.actual_outcome.success,
            prediction_error=decision.prediction_error or 0.0,
            evidence_ref=f"decision:{decision.decision_id}:outcome",
            operator_feedback_ref=feedback_ref,
            recorded_at=decision.actual_outcome.recorded_at,
            revision=0 if current is None else current.revision + 1,
        )
        self.observations[decision.decision_id] = observation
        self._rebuild_hypotheses()
        scope_history = [
            item
            for item in self.observations.values()
            if set(item.scope_ids).intersection(scopes)
        ]
        accuracy = (sum(item.success for item in scope_history) + 1.0) / (
            len(scope_history) + 2.0
        )
        hypothesis_refs = tuple(
            item.hypothesis_id
            for item in self.hypotheses.values()
            if item.scope_id in scopes
        )
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    *pre.evidence_refs,
                    observation.evidence_ref,
                    *((feedback_ref,) if feedback_ref else ()),
                )
            )
        )
        reasons = ["prediction_compared_with_observed_outcome"]
        if feedback_ref is not None:
            reasons.append("operator_feedback_calibration")
        if hypothesis_refs:
            reasons.append("recurring_error_hypothesis_updated")
        return self._store_assessment(
            decision_id=decision.decision_id,
            phase=AssessmentPhase.POST_DECISION,
            boundary=pre.boundary,
            calibrated_confidence=accuracy,
            evidence_count=len(scope_history),
            historical_accuracy=accuracy,
            evidence_refs=evidence_refs,
            capability_ids=pre.capability_ids,
            topic_tags=pre.topic_tags,
            recommended_action=pre.recommended_action,
            reason_codes=tuple(reasons),
            cognitive_quality=cognitive_quality,
            self_model_revision=self_model_revision,
            narrative_self_refs=decision.narrative_self_refs,
            outcome_ref=observation.evidence_ref,
            hypothesis_refs=hypothesis_refs,
        )

    def get(self, assessment_id: str | None) -> MetacognitiveAssessment:
        if assessment_id is None or assessment_id not in self.assessments:
            raise ValueError(f"Unknown metacognitive assessment: {assessment_id}")
        return self.assessments[assessment_id]

    def list_assessments(self) -> list[MetacognitiveAssessment]:
        return sorted(self.assessments.values(), key=lambda item: item.created_at)

    def current_quality(
        self,
        *,
        cognitive_load: float,
        attention_saturation: float,
        emotion_valence: float,
        emotion_arousal: float,
        provenance_refs: tuple[str, ...],
    ) -> CognitiveQuality:
        return self._quality(
            cognitive_load,
            attention_saturation,
            emotion_valence,
            emotion_arousal,
            provenance_refs,
        )

    def withdraw_outcome(self, decision_id: str) -> None:
        self.observations.pop(decision_id, None)
        self._rebuild_hypotheses()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "assessments": [asdict(item) for item in self.list_assessments()],
            "observations": [
                asdict(item)
                for item in sorted(self.observations.values(), key=lambda value: value.recorded_at)
            ],
            "hypotheses": [
                asdict(item)
                for item in sorted(self.hypotheses.values(), key=lambda value: value.hypothesis_id)
            ],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.assessments = {}
            self.observations = {}
            self.hypotheses = {}
            return
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported metacognition schema version")
        assessments = [_assessment_from_json(item) for item in payload.get("assessments", [])]
        observations = [_observation_from_json(item) for item in payload.get("observations", [])]
        hypotheses = [_hypothesis_from_json(item) for item in payload.get("hypotheses", [])]
        self.assessments = {item.assessment_id: item for item in assessments}
        self.observations = {item.decision_id: item for item in observations}
        self.hypotheses = {item.hypothesis_id: item for item in hypotheses}

    def _quality(
        self,
        load: float,
        saturation: float,
        valence: float,
        arousal: float,
        provenance_refs: tuple[str, ...],
    ) -> CognitiveQuality:
        _unit(load, "cognitive load")
        _unit(saturation, "attention saturation")
        _unit(arousal, "emotion arousal")
        if not math.isfinite(valence) or not -1.0 <= valence <= 1.0:
            raise ValueError("emotion valence must be between minus one and one")
        emotion_influence = max(arousal, max(0.0, -valence))
        quality = _clamp(1.0 - 0.35 * load - 0.35 * saturation - 0.3 * emotion_influence)
        reasons: list[str] = []
        if load >= 0.75:
            reasons.append("high_cognitive_load")
        if saturation >= 0.75:
            reasons.append("attention_saturated")
        if emotion_influence >= 0.7:
            reasons.append("emotion_may_degrade_judgment")
        if not reasons:
            reasons.append("no_high_quality_risk_detected")
        return CognitiveQuality(
            load,
            saturation,
            emotion_influence,
            quality,
            tuple(reasons),
            provenance_refs,
        )

    def _store_assessment(self, **values: Any) -> MetacognitiveAssessment:
        phase = values["phase"]
        previous = next(
            (
                item
                for item in reversed(self.list_assessments())
                if item.decision_id == values["decision_id"] and item.phase == phase
            ),
            None,
        )
        assessment = MetacognitiveAssessment(
            assessment_id=f"metacognition-{uuid4()}",
            supersedes_assessment_id=None if previous is None else previous.assessment_id,
            created_at=_now(),
            **values,
        )
        self.assessments[assessment.assessment_id] = assessment
        return assessment

    def _rebuild_hypotheses(self) -> None:
        previous = self.hypotheses
        self.hypotheses = {}
        scopes = tuple(
            dict.fromkeys(
                scope
                for observation in self.observations.values()
                for scope in observation.scope_ids
            )
        )
        self._refresh_hypotheses(scopes, previous)

    def _refresh_hypotheses(
        self,
        scopes: tuple[str, ...],
        previous: dict[str, ErrorHypothesis] | None = None,
    ) -> None:
        for scope in scopes:
            observations = [
                item for item in self.observations.values() if scope in item.scope_ids
            ]
            patterns = {
                "recurring_failure": [item for item in observations if not item.success],
                "optimism_bias": [
                    item
                    for item in observations
                    if item.prediction_error <= -0.25 and item.predicted_success >= 0.6
                ],
            }
            for code, matches in patterns.items():
                if len(matches) < 2:
                    continue
                identifier = f"metacognitive-hypothesis:{scope}:{code}"
                current = (previous or {}).get(identifier)
                refs = tuple(dict.fromkeys(item.evidence_ref for item in matches))
                now = _now()
                self.hypotheses[identifier] = ErrorHypothesis(
                    hypothesis_id=identifier,
                    scope_id=scope,
                    hypothesis_code=code,
                    confidence=min(0.95, 0.5 + 0.1 * len(refs)),
                    evidence_refs=refs,
                    revision=0 if current is None else current.revision + 1,
                    created_at=now if current is None else current.created_at,
                    updated_at=now,
                )


def _available_choice(available: set[ActionType], *choices: ActionType) -> ActionType:
    return next((item for item in choices if item in available), ActionType.NO_OP)


def _best_regular_choice(candidates: tuple[ActionCandidate, ...]) -> ActionType:
    return next(
        (
            item.candidate_type
            for item in candidates
            if item.candidate_type in {ActionType.RESPOND, ActionType.INTERNAL}
        ),
        candidates[0].candidate_type,
    )


def _assessment_from_json(payload: dict[str, Any]) -> MetacognitiveAssessment:
    data = dict(payload)
    data["phase"] = AssessmentPhase(data["phase"])
    data["boundary"] = EpistemicBoundary(data["boundary"])
    data["recommended_action"] = ActionType(data["recommended_action"])
    for name in (
        "evidence_refs",
        "capability_ids",
        "topic_tags",
        "reason_codes",
        "narrative_self_refs",
        "hypothesis_refs",
    ):
        data[name] = tuple(data.get(name, ()))
    quality = dict(data["cognitive_quality"])
    quality["reason_codes"] = tuple(quality.get("reason_codes", ()))
    quality["provenance_refs"] = tuple(quality.get("provenance_refs", ()))
    data["cognitive_quality"] = CognitiveQuality(**quality)
    return MetacognitiveAssessment(**data)


def _observation_from_json(payload: dict[str, Any]) -> OutcomeObservation:
    data = dict(payload)
    data["scope_ids"] = tuple(data.get("scope_ids", ()))
    return OutcomeObservation(**data)


def _hypothesis_from_json(payload: dict[str, Any]) -> ErrorHypothesis:
    data = dict(payload)
    data["evidence_refs"] = tuple(data.get("evidence_refs", ()))
    return ErrorHypothesis(**data)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _contains_private_key(value: Any) -> bool:
    private = {
        "hiddenthought",
        "prompt",
        "rawprompt",
        "reasoning",
        "chainofthought",
        "apology",
        "selfreport",
    }
    if isinstance(value, dict):
        return any(
            "".join(character for character in str(key).lower() if character.isalnum())
            in private
            or _contains_private_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_key(item) for item in value)
    return False


def _unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _now() -> str:
    return datetime.now(UTC).isoformat()
