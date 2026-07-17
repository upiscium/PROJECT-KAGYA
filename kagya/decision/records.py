"""Versioned decision records independent of free-form model reasoning."""

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import json
import math
from typing import Any, Callable, Iterable
from uuid import uuid4


class ActionType(StrEnum):
    RESPOND = "respond"
    INTERNAL = "internal"
    NO_OP = "no_op"
    DEFER = "defer"
    OBSERVE = "observe"
    REQUEST_INFORMATION = "request_information"


class DecisionStatus(StrEnum):
    AWAITING_OUTCOME = "awaiting_outcome"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class PredictedOutcome:
    outcome_id: str
    description: str
    probability: float
    utility: float

    def __post_init__(self) -> None:
        if not self.outcome_id or not self.description:
            raise ValueError("Predicted outcome ID and description must not be empty")
        _bounded(self.probability, "probability", minimum=0.0, maximum=1.0)
        _bounded(self.utility, "utility", minimum=-1.0, maximum=1.0)


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    candidate_type: ActionType
    proposed_action: str
    parameters: dict[str, Any]
    prerequisites: tuple[str, ...]
    predicted_outcomes: tuple[PredictedOutcome, ...]
    uncertainty: float
    estimated_cost: float
    estimated_risk: float
    value_effects: dict[str, float]
    appraisal_contributions: dict[str, float]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.proposed_action:
            raise ValueError("Candidate ID and proposed action must not be empty")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported action candidate schema version: {self.schema_version}"
            )
        if len(self.prerequisites) != len(set(self.prerequisites)):
            raise ValueError("Candidate prerequisites must be unique")
        if len({item.outcome_id for item in self.predicted_outcomes}) != len(
            self.predicted_outcomes
        ):
            raise ValueError("Predicted outcome IDs must be unique")
        _bounded(self.uncertainty, "uncertainty", minimum=0.0, maximum=1.0)
        _bounded(self.estimated_cost, "estimated_cost", minimum=0.0, maximum=1.0)
        _bounded(self.estimated_risk, "estimated_risk", minimum=0.0, maximum=1.0)
        _bounded_mapping(self.value_effects, "value_effects")
        _bounded_mapping(
            self.appraisal_contributions, "appraisal_contributions"
        )
        if _contains_private_key(asdict(self)):
            raise ValueError("Action candidate contains a private reasoning field")


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: ActionCandidate
    eligible: bool
    predicted_utility: float
    value_contributions: dict[str, float]
    appraisal_contributions: dict[str, float]
    self_model_contributions: dict[str, float]
    total_score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ActualOutcome:
    description: str
    utility: float
    success: bool
    observed_event_id: str | None
    observed_event_sequence: int | None
    recorded_at: str

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("Actual outcome description must not be empty")
        _bounded(self.utility, "utility", minimum=-1.0, maximum=1.0)
        if _contains_private_key(asdict(self)):
            raise ValueError("Actual outcome contains a private reasoning field")


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    triggering_event_id: str | None
    triggering_event_sequence: int | None
    context_id: str | None
    active_goal_ids: tuple[str, ...]
    value_revision_refs: dict[str, int]
    emotion_snapshot: dict[str, float]
    considered_candidates: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str
    selection_reasons: tuple[str, ...]
    selection_confidence: float
    status: DecisionStatus
    actual_outcome: ActualOutcome | None
    prediction_error: float | None
    created_at: str
    updated_at: str
    adapter_id: str | None = None
    adapter_hash: str | None = None
    activation_sequence: int | None = None
    identity_origin_refs: dict[str, str] = field(default_factory=dict)
    experience_refs: tuple[str, ...] = ()
    schema_version: int = 5

    def __post_init__(self) -> None:
        if not self.decision_id or not self.selected_candidate_id:
            raise ValueError("Decision and selected candidate IDs must not be empty")
        if self.schema_version not in {1, 2, 3, 4, 5}:
            raise ValueError(
                f"Unsupported decision record schema version: {self.schema_version}"
            )
        candidate_ids = [item.candidate.candidate_id for item in self.considered_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Considered candidate IDs must be unique")
        if self.selected_candidate_id not in candidate_ids:
            raise ValueError("Selected candidate was not considered")
        _bounded(
            self.selection_confidence,
            "selection_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        if self.status == DecisionStatus.AWAITING_OUTCOME:
            if self.actual_outcome is not None or self.prediction_error is not None:
                raise ValueError("Awaiting decision must not contain an outcome")
        elif self.actual_outcome is None or self.prediction_error is None:
            raise ValueError("Resolved decision requires outcome and prediction error")
        if _contains_private_key(asdict(self)):
            raise ValueError("Decision record contains a private reasoning field")


ValueEvaluator = Callable[[dict[str, dict[str, float]]], dict[str, dict[str, float]]]
SelfModelEvaluator = Callable[
    [tuple[ActionCandidate, ...]], dict[str, dict[str, float]]
]


class DecisionStore:
    def __init__(self) -> None:
        self.records: dict[str, DecisionRecord] = {}

    def create(
        self,
        candidates: Iterable[ActionCandidate],
        *,
        triggering_event_id: str | None,
        triggering_event_sequence: int | None,
        context_id: str | None,
        active_goal_ids: tuple[str, ...],
        value_revision_refs: dict[str, int],
        emotion_snapshot: dict[str, float],
        satisfied_prerequisites: set[str] | None = None,
        value_evaluator: ValueEvaluator | None = None,
        self_model_evaluator: SelfModelEvaluator | None = None,
        decision_id: str | None = None,
        adapter_id: str | None = None,
        adapter_hash: str | None = None,
        activation_sequence: int | None = None,
        identity_origin_refs: dict[str, str] | None = None,
        experience_refs: tuple[str, ...] = (),
    ) -> DecisionRecord:
        identifier = decision_id or str(uuid4())
        if identifier in self.records:
            raise ValueError(f"Decision already exists: {identifier}")
        candidate_values = tuple(candidates)
        if not candidate_values:
            raise ValueError("Decision requires at least one action candidate")
        if len({item.candidate_id for item in candidate_values}) != len(
            candidate_values
        ):
            raise ValueError("Action candidate IDs must be unique")
        if not any(
            item.candidate_type
            in {
                ActionType.NO_OP,
                ActionType.DEFER,
                ActionType.OBSERVE,
                ActionType.REQUEST_INFORMATION,
            }
            for item in candidate_values
        ):
            raise ValueError("Decision requires a non-action fallback candidate")
        value_contributions = (
            value_evaluator(
                {
                    item.candidate_id: item.value_effects
                    for item in candidate_values
                    if item.value_effects
                }
            )
            if value_evaluator is not None
            else {}
        )
        self_model_contributions = (
            self_model_evaluator(candidate_values)
            if self_model_evaluator is not None
            else {}
        )
        evaluations = tuple(
            _evaluate_candidate(
                candidate,
                satisfied_prerequisites or set(),
                value_contributions.get(candidate.candidate_id, {}),
                self_model_contributions.get(candidate.candidate_id, {}),
            )
            for candidate in candidate_values
        )
        eligible = [item for item in evaluations if item.eligible]
        if not eligible:
            raise ValueError("Decision has no eligible candidate")
        selected = max(
            eligible,
            key=lambda item: (
                item.total_score if item.total_score is not None else -math.inf,
                item.candidate.candidate_id,
            ),
        )
        confidence = _selection_confidence(selected, eligible)
        now = _now()
        record = DecisionRecord(
            decision_id=identifier,
            triggering_event_id=triggering_event_id,
            triggering_event_sequence=triggering_event_sequence,
            context_id=context_id,
            active_goal_ids=active_goal_ids,
            value_revision_refs=dict(value_revision_refs),
            emotion_snapshot=dict(emotion_snapshot),
            considered_candidates=evaluations,
            selected_candidate_id=selected.candidate.candidate_id,
            selection_reasons=selected.reasons,
            selection_confidence=confidence,
            status=DecisionStatus.AWAITING_OUTCOME,
            actual_outcome=None,
            prediction_error=None,
            created_at=now,
            updated_at=now,
            adapter_id=adapter_id,
            adapter_hash=adapter_hash,
            activation_sequence=activation_sequence,
            identity_origin_refs=dict(identity_origin_refs or {}),
            experience_refs=experience_refs,
        )
        self.records[identifier] = record
        return record

    def get(self, decision_id: str) -> DecisionRecord:
        record = self.records.get(decision_id)
        if record is None:
            raise ValueError(f"Unknown decision: {decision_id}")
        return record

    def list_records(
        self, status: DecisionStatus | None = None
    ) -> list[DecisionRecord]:
        return [
            record
            for record in sorted(self.records.values(), key=lambda item: item.created_at)
            if status is None or record.status == status
        ]

    def record_outcome(
        self,
        decision_id: str,
        *,
        description: str,
        utility: float,
        success: bool,
        observed_event_id: str | None,
        observed_event_sequence: int | None,
    ) -> DecisionRecord:
        record = self.get(decision_id)
        if record.status == DecisionStatus.RESOLVED:
            raise ValueError("Decision outcome is already recorded")
        selected = next(
            item
            for item in record.considered_candidates
            if item.candidate.candidate_id == record.selected_candidate_id
        )
        outcome = ActualOutcome(
            description=description,
            utility=utility,
            success=success,
            observed_event_id=observed_event_id,
            observed_event_sequence=observed_event_sequence,
            recorded_at=_now(),
        )
        updated = replace(
            record,
            status=DecisionStatus.RESOLVED,
            actual_outcome=outcome,
            prediction_error=utility - selected.predicted_utility,
            updated_at=outcome.recorded_at,
        )
        self.records[decision_id] = updated
        return updated

    def restore(self, payloads: Iterable[dict[str, Any]]) -> None:
        restored: dict[str, DecisionRecord] = {}
        for payload in payloads:
            record = _record_from_json(payload)
            if record.decision_id in restored:
                raise ValueError(f"Duplicate decision: {record.decision_id}")
            restored[record.decision_id] = record
        self.records = restored

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.list_records()]


@dataclass(frozen=True)
class DecisionDatasetRecord:
    source_id: str
    input: dict[str, Any]
    selected_action: dict[str, Any]
    outcome: dict[str, Any]
    prediction_error: float
    schema_version: int = 1

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class DecisionDatasetGenerator:
    def generate(
        self, records: Iterable[DecisionRecord]
    ) -> list[DecisionDatasetRecord]:
        dataset: list[DecisionDatasetRecord] = []
        for record in records:
            if record.status != DecisionStatus.RESOLVED:
                continue
            selected = next(
                item.candidate
                for item in record.considered_candidates
                if item.candidate.candidate_id == record.selected_candidate_id
            )
            if record.actual_outcome is None or record.prediction_error is None:
                continue
            dataset.append(
                DecisionDatasetRecord(
                    source_id=record.decision_id,
                    input={
                        "context_id": record.context_id,
                        "active_goal_ids": list(record.active_goal_ids),
                        "value_revision_refs": record.value_revision_refs,
                        "emotion_snapshot": record.emotion_snapshot,
                        "adapter_id": record.adapter_id,
                        "adapter_hash": record.adapter_hash,
                        "activation_sequence": record.activation_sequence,
                        "candidates": [
                            asdict(item.candidate)
                            for item in record.considered_candidates
                        ],
                    },
                    selected_action={
                        "candidate_id": selected.candidate_id,
                        "candidate_type": selected.candidate_type.value,
                        "proposed_action": selected.proposed_action,
                        "parameters": selected.parameters,
                    },
                    outcome=asdict(record.actual_outcome),
                    prediction_error=record.prediction_error,
                )
            )
        return dataset


def parse_candidate_output(value: str | dict[str, Any]) -> list[ActionCandidate]:
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ValueError("Candidate output must contain only a candidates field")
    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("Candidate output candidates must be a list")
    return [_candidate_from_json(item) for item in candidates]


def schema_candidate_prompt(situation: str) -> str:
    schema = {
        "candidates": [
            {
                "candidate_id": "string",
                "candidate_type": [item.value for item in ActionType],
                "proposed_action": "string",
                "parameters": {},
                "prerequisites": ["string"],
                "predicted_outcomes": [
                    {
                        "outcome_id": "string",
                        "description": "string",
                        "probability": "0..1",
                        "utility": "-1..1",
                    }
                ],
                "uncertainty": "0..1",
                "estimated_cost": "0..1",
                "estimated_risk": "0..1",
                "value_effects": {"value_id": "-1..1"},
                "appraisal_contributions": {"signal": "-1..1"},
            }
        ]
    }
    return (
        "Generate action candidates as strict JSON. Do not include reasoning, "
        "hidden thoughts, markdown, or fields outside this schema.\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Situation: {situation}"
    )


def _evaluate_candidate(
    candidate: ActionCandidate,
    satisfied_prerequisites: set[str],
    value_contributions: dict[str, float],
    self_model_contributions: dict[str, float],
) -> CandidateEvaluation:
    missing = tuple(
        item for item in candidate.prerequisites if item not in satisfied_prerequisites
    )
    predicted_utility = sum(
        item.probability * item.utility for item in candidate.predicted_outcomes
    )
    if missing:
        return CandidateEvaluation(
            candidate=candidate,
            eligible=False,
            predicted_utility=predicted_utility,
            value_contributions=dict(value_contributions),
            appraisal_contributions=dict(candidate.appraisal_contributions),
            self_model_contributions=dict(self_model_contributions),
            total_score=None,
            reasons=("prerequisites_missing", *missing),
        )
    total = (
        predicted_utility
        + sum(value_contributions.values())
        + 0.25 * sum(candidate.appraisal_contributions.values())
        + sum(self_model_contributions.values())
        - 0.25 * candidate.uncertainty
        - 0.25 * candidate.estimated_cost
        - 0.5 * candidate.estimated_risk
    )
    return CandidateEvaluation(
        candidate=candidate,
        eligible=True,
        predicted_utility=predicted_utility,
        value_contributions=dict(value_contributions),
        appraisal_contributions=dict(candidate.appraisal_contributions),
        self_model_contributions=dict(self_model_contributions),
        total_score=total,
        reasons=(
            "prerequisites_satisfied",
            "predicted_outcomes_scored",
            "value_contributions_scored",
            "appraisal_contributions_scored",
            "self_model_contributions_scored",
            "cost_risk_uncertainty_applied",
        ),
    )


def _selection_confidence(
    selected: CandidateEvaluation, eligible: list[CandidateEvaluation]
) -> float:
    selected_score = selected.total_score or 0.0
    alternatives = [
        item.total_score or 0.0
        for item in eligible
        if item.candidate.candidate_id != selected.candidate.candidate_id
    ]
    margin = 1.0 if not alternatives else max(0.0, selected_score - max(alternatives))
    confidence = (1.0 - selected.candidate.uncertainty) * (
        0.5 + 0.5 * min(1.0, margin)
    )
    return max(0.0, min(1.0, confidence))


def _candidate_from_json(payload: Any) -> ActionCandidate:
    if not isinstance(payload, dict):
        raise ValueError("Action candidate must be an object")
    allowed = {
        "candidate_id",
        "candidate_type",
        "proposed_action",
        "parameters",
        "prerequisites",
        "predicted_outcomes",
        "uncertainty",
        "estimated_cost",
        "estimated_risk",
        "value_effects",
        "appraisal_contributions",
        "schema_version",
    }
    if set(payload) - allowed:
        raise ValueError("Action candidate contains unknown fields")
    data = dict(payload)
    data["candidate_type"] = ActionType(data["candidate_type"])
    data["parameters"] = dict(data.get("parameters", {}))
    data["prerequisites"] = tuple(data.get("prerequisites", ()))
    data["predicted_outcomes"] = tuple(
        PredictedOutcome(**item) for item in data.get("predicted_outcomes", ())
    )
    data["value_effects"] = dict(data.get("value_effects", {}))
    data["appraisal_contributions"] = dict(
        data.get("appraisal_contributions", {})
    )
    return ActionCandidate(**data)


def _record_from_json(payload: dict[str, Any]) -> DecisionRecord:
    data = dict(payload)
    data.setdefault("adapter_id", None)
    data.setdefault("adapter_hash", None)
    data.setdefault("activation_sequence", None)
    data.setdefault("identity_origin_refs", {})
    data["experience_refs"] = tuple(data.get("experience_refs", ()))
    evaluations = []
    for raw in data.get("considered_candidates", ()):
        item = dict(raw)
        item["candidate"] = _candidate_from_json(item["candidate"])
        item["reasons"] = tuple(item.get("reasons", ()))
        item["self_model_contributions"] = dict(
            item.get("self_model_contributions", {})
        )
        evaluations.append(CandidateEvaluation(**item))
    data["considered_candidates"] = tuple(evaluations)
    data["active_goal_ids"] = tuple(data.get("active_goal_ids", ()))
    data["selection_reasons"] = tuple(data.get("selection_reasons", ()))
    data["status"] = DecisionStatus(data["status"])
    if data.get("actual_outcome") is not None:
        data["actual_outcome"] = ActualOutcome(**data["actual_outcome"])
    return DecisionRecord(**data)


def _bounded(value: float, name: str, *, minimum: float, maximum: float) -> None:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


def _bounded_mapping(values: dict[str, float], name: str) -> None:
    for key, value in values.items():
        if not key:
            raise ValueError(f"{name} keys must not be empty")
        _bounded(value, name, minimum=-1.0, maximum=1.0)


def _contains_private_key(value: Any) -> bool:
    private = {"hidden_thought", "prompt", "raw_prompt", "reasoning", "chain_of_thought"}
    if isinstance(value, dict):
        return any(
            key in private or _contains_private_key(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_key(item) for item in value)
    return False


def _now() -> str:
    return datetime.now(UTC).isoformat()
