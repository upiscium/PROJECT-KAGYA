"""Public-safe, revisioned explanations derived from frozen DecisionRecords."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.decision.records import ActionType, DecisionRecord, DecisionStatus


MAX_ALTERNATIVES = 5
MAX_CONTRIBUTIONS = 24
MAX_REFS = 32
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PRIVATE_MARKERS = (
    "<think",
    "hidden_thought",
    "raw_prompt",
    "chain_of_thought",
    "credential",
    "secret",
    "password",
    "private_key",
    "attachment_body",
    "file://",
    "/home/",
    "/tmp/",
)
Compatibility = Literal["compatible", "context_filtered", "interlocutor_filtered"]
SourceType = Literal["value", "goal", "commitment", "belief"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExplanationDisposition(StrEnum):
    SELECTED_ACTION = "selected_action"
    NO_OP = "no_op"
    REFUSE = "refuse"
    DEFER = "defer"
    REQUEST_INFORMATION = "request_information"
    UNABLE = "unable"
    AWAITING_APPROVAL = "awaiting_approval"
    BLOCKED_POLICY = "blocked_policy"
    ACTION_FAILED = "action_failed"
    REPLAN = "replan"


class ReferenceAvailability(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    STALE = "stale"
    PRIVATE = "private"
    INCOMPATIBLE = "incompatible"


class RendererState(StrEnum):
    DETERMINISTIC = "deterministic"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CandidateProjection(_StrictModel):
    candidate_id: str
    action_type: str
    eligible: bool
    score: float | None = Field(default=None, allow_inf_nan=False)
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    risk: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    disposition_code: str
    reason_codes: tuple[str, ...] = Field(max_length=16)


class SourceContribution(_StrictModel):
    source_type: SourceType
    source_id: str
    source_revision: int = Field(ge=0)
    contribution: float | None = Field(default=None, ge=-1.0, le=1.0)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=MAX_REFS)
    origin_ref: str | None = None
    availability: Literal[ReferenceAvailability.AVAILABLE] = ReferenceAvailability.AVAILABLE


class UncertaintyProjection(_StrictModel):
    code: str
    severity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    refs: tuple[str, ...] = Field(default=(), max_length=MAX_REFS)


class RiskProjection(_StrictModel):
    risk_class: str
    policy_status: str
    approval_status: str
    policy_ref: str | None = None
    approval_ref: str | None = None
    action_intent_ref: str | None = None
    validation_ref: str | None = None
    receipt_ref: str | None = None
    observation_ref: str | None = None
    verification_ref: str | None = None
    policy_reason_codes: tuple[str, ...] = Field(default=(), max_length=16)


class BoundaryProjection(_StrictModel):
    assessment_id: str
    assessment_revision: int = Field(ge=1)
    classification: str
    recommendation: str
    disposition: str
    reason_codes: tuple[str, ...] = Field(max_length=16)


class OutcomeProjection(_StrictModel):
    status: Literal["pending", "succeeded", "failed", "compensated"]
    utility: float | None = Field(default=None, ge=-1.0, le=1.0)
    prediction_error: float | None = Field(default=None, ge=-2.0, le=2.0)
    observed_event_ref: str | None = None
    post_assessment_ref: str | None = None


class ChangeProjection(_StrictModel):
    previous_explanation_revision: int | None = Field(default=None, ge=1)
    changed_fields: tuple[str, ...] = Field(default=(), max_length=32)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=16)


class RendererProjection(_StrictModel):
    state: RendererState
    deterministic_template: str
    offered_clause_ids: tuple[str, ...] = Field(max_length=16)
    ordered_clause_ids: tuple[str, ...] = Field(max_length=16)
    visible_explanation: str
    failure_code: str | None = None


class PublicDecisionExplanation(_StrictModel):
    schema_version: Literal[1] = 1
    explanation_id: str
    revision: int = Field(ge=1)
    decision_id: str
    decision_revision: int = Field(ge=1)
    decision_status: str
    selected: CandidateProjection
    disposition: ExplanationDisposition
    major_alternatives: tuple[CandidateProjection, ...] = Field(
        default=(), max_length=MAX_ALTERNATIVES
    )
    contributions: tuple[SourceContribution, ...] = Field(
        default=(), max_length=MAX_CONTRIBUTIONS
    )
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=MAX_REFS)
    uncertainty: tuple[UncertaintyProjection, ...] = Field(default=(), max_length=16)
    information_gap_codes: tuple[str, ...] = Field(default=(), max_length=16)
    omitted_reference_count: int = Field(default=0, ge=0, le=MAX_REFS)
    risk: RiskProjection
    tradeoff_refs: tuple[str, ...] = Field(default=(), max_length=MAX_REFS)
    conflict_codes: tuple[str, ...] = Field(default=(), max_length=16)
    boundary: BoundaryProjection | None = None
    reason_codes: tuple[str, ...] = Field(default=(), max_length=32)
    outcome: OutcomeProjection
    change: ChangeProjection
    created_event_id: str | None
    created_event_sequence: int | None = Field(ge=1)
    context_id: str | None = None
    interlocutor_id: str | None = None
    compatibility: Compatibility
    renderer: RendererProjection
    created_at: str

    @model_validator(mode="after")
    def validate_public_contract(self) -> "PublicDecisionExplanation":
        _validate_safe_tree(self.model_dump(mode="json"))
        if self.decision_status not in {item.value for item in DecisionStatus}:
            raise ValueError("unsupported decision status")
        return self

    def public_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class NaturalExplanationOutput(_StrictModel):
    explanation_id: str
    explanation_revision: int = Field(ge=1)
    ordered_clause_ids: tuple[str, ...] = Field(min_length=1, max_length=16)


_CLAUSE_TEMPLATES = {
    **{
        f"disposition.{item.value}.v1": f"Disposition: {item.value.replace('_', ' ')}."
        for item in ExplanationDisposition
    },
    **{
        f"decision_status.{item.value}.v1": f"Decision status: {item.value.replace('_', ' ')}."
        for item in DecisionStatus
    },
    "information_gaps.present.v1": "Some public information is unavailable.",
    "information_gaps.none.v1": "No public information gaps were recorded.",
    "outcome.pending.v1": "Outcome status: pending.",
    "outcome.succeeded.v1": "Outcome status: succeeded.",
    "outcome.failed.v1": "Outcome status: failed.",
    "outcome.compensated.v1": "Outcome status: compensated.",
}


class DecisionExplanationStore:
    """Append-only explanation history with event/input idempotency."""

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._records: dict[str, list[PublicDecisionExplanation]] = {}
        self._idempotency: dict[str, tuple[str, str, int]] = {}

    def append(
        self,
        explanation: PublicDecisionExplanation,
        *,
        idempotency_key: str,
        input_digest: str,
    ) -> PublicDecisionExplanation:
        existing = self._idempotency.get(idempotency_key)
        if existing is not None:
            digest, explanation_id, revision = existing
            if digest != input_digest:
                raise ValueError("Explanation idempotency key has different input")
            return self.get(explanation_id, revision)
        history = self._records.setdefault(explanation.explanation_id, [])
        expected_revision = len(history) + 1
        if explanation.revision != expected_revision:
            raise ValueError("Explanation revision is not the next immutable revision")
        history.append(explanation)
        self._idempotency[idempotency_key] = (
            input_digest,
            explanation.explanation_id,
            explanation.revision,
        )
        return explanation

    def get(
        self, explanation_id: str, revision: int | None = None
    ) -> PublicDecisionExplanation:
        history = self._records.get(explanation_id)
        if not history:
            raise ValueError(f"Unknown decision explanation: {explanation_id}")
        if revision is None:
            return history[-1]
        if revision < 1 or revision > len(history):
            raise ValueError("Unknown decision explanation revision")
        return history[revision - 1]

    def history(self, explanation_id: str) -> tuple[PublicDecisionExplanation, ...]:
        self.get(explanation_id)
        return tuple(self._records[explanation_id])

    def list_latest(
        self, *, decision_id: str | None = None
    ) -> tuple[PublicDecisionExplanation, ...]:
        values = (history[-1] for history in self._records.values())
        return tuple(
            sorted(
                (
                    item
                    for item in values
                    if decision_id is None or item.decision_id == decision_id
                ),
                key=lambda item: (
                    item.created_event_sequence or 0,
                    item.explanation_id,
                ),
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "records": {
                key: [item.model_dump(mode="json") for item in history]
                for key, history in self._records.items()
            },
            "idempotency": {
                key: list(value) for key, value in self._idempotency.items()
            },
        }

    def restore(self, payload: object) -> None:
        if payload in (None, {}):
            self._records = {}
            self._idempotency = {}
            return
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported decision explanation state")
        records: dict[str, list[PublicDecisionExplanation]] = {}
        raw_records = payload.get("records", {})
        if not isinstance(raw_records, dict):
            raise ValueError("Invalid decision explanation records")
        for explanation_id, values in raw_records.items():
            if not isinstance(values, list):
                raise ValueError("Invalid decision explanation history")
            history = [
                PublicDecisionExplanation.model_validate(
                    _migrate_explanation_payload(item)
                )
                for item in values
            ]
            if any(
                item.explanation_id != explanation_id or item.revision != index
                for index, item in enumerate(history, 1)
            ):
                raise ValueError("Invalid decision explanation revision chain")
            records[str(explanation_id)] = history
        idempotency: dict[str, tuple[str, str, int]] = {}
        raw_idempotency = payload.get("idempotency", {})
        if not isinstance(raw_idempotency, dict):
            raise ValueError("Invalid decision explanation idempotency state")
        for key, value in raw_idempotency.items():
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("Invalid decision explanation idempotency entry")
            idempotency[str(key)] = (str(value[0]), str(value[1]), int(value[2]))
        self._records = records
        self._idempotency = idempotency

    def get_idempotent(
        self, idempotency_key: str, input_digest: str
    ) -> PublicDecisionExplanation | None:
        existing = self._idempotency.get(idempotency_key)
        if existing is None:
            return None
        digest, explanation_id, revision = existing
        if digest != input_digest:
            raise ValueError("Explanation idempotency key has different input")
        return self.get(explanation_id, revision)


def build_explanation(
    main_loop: Any,
    decision: DecisionRecord,
    *,
    event_id: str,
    event_sequence: int,
    explanation_id: str | None = None,
    previous: PublicDecisionExplanation | None = None,
    context_id: str | None = None,
    interlocutor_id: str | None = None,
) -> PublicDecisionExplanation:
    selected_eval = next(
        item
        for item in decision.considered_candidates
        if item.candidate.candidate_id == decision.selected_candidate_id
    )
    selected = _candidate_projection(selected_eval, selected=True)
    alternatives = tuple(
        _candidate_projection(item, selected=False)
        for item in sorted(
            (
                item
                for item in decision.considered_candidates
                if item.candidate.candidate_id != decision.selected_candidate_id
            ),
            key=lambda item: (
                item.total_score is not None,
                item.total_score if item.total_score is not None else -math.inf,
                item.candidate.candidate_id,
            ),
            reverse=True,
        )[:MAX_ALTERNATIVES]
    )
    compatibility = _compatibility(main_loop, decision, context_id, interlocutor_id)
    contributions, unavailable = _contributions(
        main_loop, decision, selected_eval, compatibility
    )
    risk, action_disposition = _risk(main_loop, decision)
    boundary = _boundary(main_loop, decision, compatibility)
    if (
        compatibility == "compatible"
        and decision.boundary_assessment_id is not None
        and boundary is None
    ):
        unavailable = (*unavailable, "boundary_reference_unavailable")
    disposition = (
        action_disposition or _disposition(selected_eval.candidate.candidate_type, boundary)
        if compatibility == "compatible"
        else _disposition(selected_eval.candidate.candidate_type, None)
    )
    uncertainty = [
        UncertaintyProjection(
            code="selected_candidate_uncertainty",
            severity=selected_eval.candidate.uncertainty,
            refs=(selected_eval.candidate.candidate_id,)
            if compatibility == "compatible"
            else (),
        )
    ]
    information_gaps: list[str] = []
    if selected_eval.candidate.uncertainty >= 0.5:
        information_gaps.append("high_candidate_uncertainty")
    for code in unavailable:
        uncertainty.append(UncertaintyProjection(code=code, severity=1.0))
        information_gaps.append(code)
    if compatibility != "compatible":
        information_gaps.append(compatibility)
    outcome = _outcome(decision, compatibility)
    changed_fields = _changed_fields(previous, decision, outcome, contributions)
    reason_codes = tuple(
        dict.fromkeys(
            (
                *decision.selection_reasons,
                f"disposition_{disposition.value}",
                *(boundary.reason_codes if boundary is not None else ()),
            )
        )
    )[:32]
    template = _deterministic_template(disposition, decision.status, outcome.status)
    clauses = _deterministic_clauses(
        disposition, decision.status, outcome.status, bool(information_gaps)
    )
    visible = _render_clauses(clauses)
    filtered = compatibility != "compatible"
    if filtered:
        selected = selected.model_copy(update={"candidate_id": "filtered"})
        alternatives = ()
        risk = _empty_risk()
        boundary = None
        reason_codes = (f"compatibility_{compatibility}",)
    return PublicDecisionExplanation(
        explanation_id=explanation_id or f"explanation-{uuid4()}",
        revision=1 if previous is None else previous.revision + 1,
        decision_id=decision.decision_id,
        decision_revision=decision.revision,
        decision_status=decision.status.value,
        selected=selected,
        disposition=disposition,
        major_alternatives=alternatives,
        contributions=contributions,
        evidence_refs=()
        if filtered
        else tuple(
            dict.fromkeys(
                reference
                for contribution in contributions
                for reference in contribution.evidence_refs
            )
        )[:MAX_REFS],
        uncertainty=tuple(uncertainty[:16]),
        information_gap_codes=tuple(dict.fromkeys(information_gaps))[:16],
        omitted_reference_count=min(len(unavailable), MAX_REFS),
        risk=risk,
        tradeoff_refs=() if filtered else _tradeoff_refs(main_loop, decision),
        conflict_codes=() if filtered else _conflict_codes(selected_eval),
        boundary=boundary,
        reason_codes=reason_codes,
        outcome=outcome,
        change=ChangeProjection(
            previous_explanation_revision=None
            if previous is None
            else previous.revision,
            changed_fields=changed_fields,
            reason_codes=() if previous is None else ("decision_projection_revised",),
        ),
        created_event_id=None if filtered else event_id,
        created_event_sequence=None if filtered else event_sequence,
        context_id=None if filtered else context_id or decision.context_id,
        interlocutor_id=None if filtered else interlocutor_id,
        compatibility=compatibility,
        renderer=RendererProjection(
            state=RendererState.DETERMINISTIC,
            deterministic_template=template,
            offered_clause_ids=clauses,
            ordered_clause_ids=clauses,
            visible_explanation=visible,
        ),
        created_at=datetime.now(UTC).isoformat(),
    )


def render_natural(
    explanation: PublicDecisionExplanation,
    generate: Callable[[str], str],
) -> PublicDecisionExplanation:
    deterministic_order = explanation.renderer.offered_clause_ids
    public_payload = explanation.public_json()
    prompt = (
        "Select and order only offered_clause_ids. Return strict JSON with exactly "
        "explanation_id, explanation_revision, ordered_clause_ids. Do not write prose.\n"
        + json.dumps(public_payload, ensure_ascii=True, sort_keys=True)
    )
    try:
        raw = generate(prompt)
        parsed = json.loads(raw)
        output = NaturalExplanationOutput.model_validate(parsed)
        if (
            output.explanation_id != explanation.explanation_id
            or output.explanation_revision != explanation.revision
        ):
            raise ValueError("renderer identity mismatch")
        if (
            len(output.ordered_clause_ids) != len(set(output.ordered_clause_ids))
            or not set(output.ordered_clause_ids) <= set(deterministic_order)
        ):
            raise ValueError("renderer clause selection is invalid")
        renderer = RendererProjection(
            state=RendererState.SUCCEEDED,
            deterministic_template=explanation.renderer.deterministic_template,
            offered_clause_ids=deterministic_order,
            ordered_clause_ids=output.ordered_clause_ids,
            visible_explanation=_render_clauses(output.ordered_clause_ids),
        )
    except Exception:
        renderer = RendererProjection(
            state=RendererState.FAILED,
            deterministic_template=explanation.renderer.deterministic_template,
            offered_clause_ids=deterministic_order,
            ordered_clause_ids=deterministic_order,
            visible_explanation=_render_clauses(deterministic_order),
            failure_code="renderer_failed",
        )
    return explanation.model_copy(
        update={
            "revision": explanation.revision + 1,
            "change": ChangeProjection(
                previous_explanation_revision=explanation.revision,
                changed_fields=("renderer",),
                reason_codes=("renderer_updated",),
            ),
            "renderer": renderer,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )


def explanation_input_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_projection(evaluation: Any, *, selected: bool) -> CandidateProjection:
    reasons = tuple(_safe_reason(item) for item in evaluation.reasons[:16])
    return CandidateProjection(
        candidate_id=evaluation.candidate.candidate_id,
        action_type=evaluation.candidate.candidate_type.value,
        eligible=evaluation.eligible,
        score=evaluation.total_score,
        uncertainty=evaluation.candidate.uncertainty,
        risk=evaluation.candidate.estimated_risk,
        disposition_code="selected"
        if selected
        else "rejected"
        if evaluation.eligible
        else "blocked",
        reason_codes=reasons,
    )


def _contributions(
    main_loop: Any,
    decision: DecisionRecord,
    selected_eval: Any,
    compatibility: str,
) -> tuple[tuple[SourceContribution, ...], tuple[str, ...]]:
    if compatibility != "compatible":
        return (), ("source_context_incompatible",)
    candidate = selected_eval.candidate
    values: list[SourceContribution] = []
    unavailable: list[str] = []
    for value_id, contribution in selected_eval.value_contributions.items():
        if not _public_ref(value_id):
            unavailable.append("value_reference_private")
            continue
        revision = decision.value_revision_refs.get(value_id)
        current = main_loop.value_system.values.get(value_id)
        availability = _availability(current, revision, compatibility)
        if (
            availability == ReferenceAvailability.AVAILABLE
            and getattr(getattr(current, "scope", None), "value", None) == "context"
            and decision.context_id not in getattr(current, "context_ids", ())
        ):
            availability = ReferenceAvailability.INCOMPATIBLE
        if availability != ReferenceAvailability.AVAILABLE:
            unavailable.append(f"value_reference_{availability.value}")
            continue
        assert revision is not None
        values.append(
            SourceContribution(
                source_type="value",
                source_id=value_id,
                source_revision=revision,
                contribution=contribution,
                evidence_refs=tuple(
                    item
                    for item in (
                        *(
                            current.supporting_evidence_ids
                            if current is not None
                            else ()
                        ),
                        *(current.opposing_evidence_ids if current is not None else ()),
                    )
                    if item in candidate.evidence_refs and _public_ref(item)
                )[:MAX_REFS],
                origin_ref=_public_optional_ref(
                    decision.identity_origin_refs.get(f"value:{value_id}")
                ),
            )
        )
    source_groups: tuple[
        tuple[SourceType, tuple[str, ...], dict[str, int], dict[str, Any]], ...
    ] = (
        (
            "goal",
            candidate.goal_refs,
            decision.goal_revision_refs,
            main_loop.goal_manager.goals,
        ),
        (
            "commitment",
            candidate.commitment_refs,
            decision.commitment_revision_refs,
            main_loop.commitment_store.commitments,
        ),
        (
            "belief",
            candidate.belief_refs,
            decision.belief_revision_refs,
            main_loop.belief_store.records,
        ),
    )
    for source_type, refs, decision_refs, store in source_groups:
        for source_id in refs:
            if not _public_ref(source_id):
                unavailable.append(f"{source_type}_reference_private")
                continue
            revision = decision_refs.get(source_id)
            current = store.get(source_id)
            availability = _availability(current, revision, compatibility)
            if (
                source_type == "belief"
                and current is not None
                and current.context_scope
            ):
                if decision.context_id not in current.context_scope:
                    availability = ReferenceAvailability.INCOMPATIBLE
            if availability != ReferenceAvailability.AVAILABLE:
                unavailable.append(f"{source_type}_reference_{availability.value}")
                continue
            assert revision is not None
            evidence = ()
            if source_type == "belief" and current is not None:
                known = {item.reference for item in current.evidence}
                evidence = tuple(
                    item
                    for item in candidate.evidence_refs
                    if item in known and _public_ref(item)
                )
            values.append(
                SourceContribution(
                    source_type=source_type,
                    source_id=source_id,
                    source_revision=revision,
                    contribution=None,
                    evidence_refs=evidence[:MAX_REFS],
                    origin_ref=_public_optional_ref(
                        decision.identity_origin_refs.get(f"{source_type}:{source_id}")
                    ),
                )
            )
    return tuple(values[:MAX_CONTRIBUTIONS]), tuple(dict.fromkeys(unavailable))


def _availability(
    current: Any, expected_revision: int | None, compatibility: str
) -> ReferenceAvailability:
    if compatibility != "compatible":
        return ReferenceAvailability.INCOMPATIBLE
    if current is None or expected_revision is None:
        return ReferenceAvailability.MISSING
    current_revision = getattr(
        current, "revision", len(getattr(current, "transitions", ())) + 1
    )
    if current_revision != expected_revision:
        return ReferenceAvailability.STALE
    origin = getattr(current, "identity_origin", None) or getattr(
        current, "origin_provenance", None
    )
    if origin is None or getattr(
        getattr(origin, "endorsement", None), "value", None
    ) != "endorsed":
        return ReferenceAvailability.PRIVATE
    return ReferenceAvailability.AVAILABLE


def _risk(
    main_loop: Any, decision: DecisionRecord
) -> tuple[RiskProjection, ExplanationDisposition | None]:
    execution = getattr(main_loop, "_action_execution", None)
    if execution is None:
        return RiskProjection(
            risk_class="unclassified",
            policy_status="not_evaluated",
            approval_status="not_required",
        ), None
    intents = [
        item
        for item in execution.list_intents()
        if item.provenance.decision_id == decision.decision_id
        and item.provenance.candidate_id == decision.selected_candidate_id
    ]
    validations = execution.list_validation_records()
    receipts = execution.list_receipts()
    attempts: list[tuple[int, int, str, Any]] = []
    for intent in intents:
        sequences = [
            item.validated_event_sequence
            for item in validations
            if item.validation_id == intent.validation_record_id
        ]
        sequences.extend(
            item.event_sequence
            for item in receipts
            if item.intent_id == intent.intent_id and item.event_sequence is not None
        )
        attempts.append((max(sequences, default=0), intent.revision, "intent", intent))
    attempts.extend(
        (item.event_sequence, 0, "blocked", item)
        for item in getattr(execution, "list_policy_rejections", lambda: ())()
        if item.decision_id == decision.decision_id
        and item.candidate_id == decision.selected_candidate_id
    )
    attempts.extend(
        (item.validated_event_sequence, 0, "invalid", item)
        for item in validations
        if item.decision_id == decision.decision_id and not item.arguments_valid
    )
    canonical = max(attempts, key=lambda item: item[:2], default=None)
    if canonical is None:
        return (
            RiskProjection(
                risk_class="unclassified",
                policy_status="not_evaluated",
                approval_status="not_required",
            ),
            None,
        )
    _sequence, _revision, attempt_type, attempt = canonical
    if attempt_type == "blocked":
        return RiskProjection(
            risk_class=attempt.risk_class.value,
            policy_status="blocked",
            approval_status="not_required",
            policy_ref=_public_optional_ref(attempt.rejection_id),
            policy_reason_codes=(attempt.reason_code,),
        ), ExplanationDisposition.BLOCKED_POLICY
    if attempt_type == "invalid":
        return RiskProjection(
            risk_class="unclassified",
            policy_status="invalid",
            approval_status="not_required",
            validation_ref=_public_optional_ref(attempt.validation_id),
        ), ExplanationDisposition.UNABLE
    intent = attempt
    receipt = next(
        (
            item
            for item in receipts
            if item.receipt_id == intent.receipt_id and item.intent_id == intent.intent_id
        ),
        None,
    )
    observation = next(
        (
            item
            for item in execution.list_observations()
            if receipt is not None
            and item.observation_id == receipt.observation_id
            and item.receipt_id == receipt.receipt_id
            and item.intent_id == intent.intent_id
        ),
        None,
    )
    verification = next(
        (
            item
            for item in execution.list_verifications()
            if receipt is not None
            and observation is not None
            and item.verification_id == receipt.verification_id
            and item.observation_id == observation.observation_id
            and item.intent_id == intent.intent_id
        ),
        None,
    )
    approval = next(
        (
            item
            for item in execution.list_approvals()
            if item.approval_id == intent.approval_id
        ),
        None,
    )
    disposition = None
    if intent.status.value == "awaiting_approval":
        disposition = ExplanationDisposition.AWAITING_APPROVAL
    elif intent.failure_code is not None or (
        receipt is not None and receipt.status.value == "failed"
    ):
        disposition = ExplanationDisposition.ACTION_FAILED
    return RiskProjection(
        risk_class=intent.risk_class.value,
        policy_status="allowed" if intent.policy.allowed else "blocked",
        approval_status="not_required"
        if intent.approval_id is None
        else "pending"
        if approval is None
        else approval.status,
        policy_ref=_public_optional_ref(intent.policy.evaluation_id),
        approval_ref=_public_optional_ref(intent.approval_id),
        action_intent_ref=_public_optional_ref(intent.intent_id),
        validation_ref=_public_optional_ref(intent.validation_record_id),
        receipt_ref=None
        if receipt is None
        else _public_optional_ref(receipt.receipt_id),
        observation_ref=None
        if observation is None
        else _public_optional_ref(observation.observation_id),
        verification_ref=None
        if verification is None
        else _public_optional_ref(verification.verification_id),
        policy_reason_codes=tuple(
            _safe_reason(item) for item in intent.policy.reasons[:16]
        ),
    ), disposition


def _boundary(
    main_loop: Any, decision: DecisionRecord, compatibility: str
) -> BoundaryProjection | None:
    if (
        decision.boundary_assessment_id is None
        or compatibility != "compatible"
        or not _public_ref(decision.boundary_assessment_id)
    ):
        return None
    try:
        assessment = main_loop.identity_boundary_store.get_assessment(
            decision.boundary_assessment_id
        )
    except ValueError:
        return None
    if (
        not main_loop.identity_boundary_store.assessments
        or main_loop.identity_boundary_store.assessments[-1].assessment_id
        != assessment.assessment_id
        or assessment.revision != decision.boundary_assessment_revision
        or main_loop.identity_boundary_store.assessment_digest(assessment.assessment_id)
        != decision.boundary_assessment_digest
    ):
        return None
    return BoundaryProjection(
        assessment_id=assessment.assessment_id,
        assessment_revision=assessment.revision,
        classification=assessment.classification.value,
        recommendation=assessment.recommendation.value,
        disposition=assessment.disposition.value,
        reason_codes=tuple(_safe_reason(item) for item in assessment.reason_codes),
    )


def _outcome(decision: DecisionRecord, compatibility: str) -> OutcomeProjection:
    if decision.actual_outcome is None:
        return OutcomeProjection(status="pending")
    status: Literal["succeeded", "failed", "compensated"] = (
        "compensated"
        if decision.actual_outcome.compensated
        else "succeeded"
        if decision.actual_outcome.success
        else "failed"
    )
    event_ref = None
    if compatibility == "compatible" and decision.actual_outcome.observed_event_id is not None:
        event_ref = decision.actual_outcome.observed_event_id
    return OutcomeProjection(
        status=status,
        utility=decision.actual_outcome.utility,
        prediction_error=decision.prediction_error,
        observed_event_ref=event_ref,
        post_assessment_ref=decision.metacognition_post_assessment_id
        if compatibility == "compatible"
        else None,
    )


def _disposition(
    action_type: ActionType, boundary: BoundaryProjection | None
) -> ExplanationDisposition:
    if boundary is not None and boundary.recommendation == "refuse":
        return ExplanationDisposition.REFUSE
    if boundary is not None and boundary.recommendation == "defer":
        return ExplanationDisposition.DEFER
    return {
        ActionType.NO_OP: ExplanationDisposition.NO_OP,
        ActionType.DEFER: ExplanationDisposition.DEFER,
        ActionType.REQUEST_INFORMATION: ExplanationDisposition.REQUEST_INFORMATION,
        ActionType.REFUSE: ExplanationDisposition.REFUSE,
        ActionType.UNABLE: ExplanationDisposition.UNABLE,
        ActionType.REPLAN: ExplanationDisposition.REPLAN,
    }.get(action_type, ExplanationDisposition.SELECTED_ACTION)


def _compatibility(
    main_loop: Any,
    decision: DecisionRecord,
    context_id: str | None,
    interlocutor_id: str | None,
) -> Compatibility:
    requested_context = context_id or decision.context_id
    if requested_context != decision.context_id:
        return "context_filtered"
    if decision.context_id is None:
        return "compatible"
    context = main_loop.context_registry.get(decision.context_id)
    if context is None:
        return "interlocutor_filtered"
    if context.participant_ids and (
        interlocutor_id is None or interlocutor_id not in context.participant_ids
    ):
        return "interlocutor_filtered"
    return "compatible"


def _conflict_codes(selected_eval: Any) -> tuple[str, ...]:
    return tuple(
        _safe_reason(item) for item in selected_eval.reasons if "conflict" in item
    )[:16]


def _tradeoff_refs(main_loop: Any, decision: DecisionRecord) -> tuple[str, ...]:
    known = {
        item.tradeoff_id
        for item in getattr(main_loop.value_system, "tradeoffs", ())
        if item.decision_id == decision.decision_id
        and item.option_id == decision.selected_candidate_id
        and item.context_id == decision.context_id
        and item.value_revision_refs == decision.value_revision_refs
        and _public_ref(item.tradeoff_id)
    }
    return tuple(item for item in decision.value_tradeoff_refs if item in known)[:MAX_REFS]


def _public_optional_ref(value: str | None) -> str | None:
    return value if value is not None and _public_ref(value) else None


def _public_ref(value: str) -> bool:
    lowered = value.lower()
    return bool(_SAFE_ID.fullmatch(value)) and not any(
        marker in lowered for marker in _PRIVATE_MARKERS
    )


def _changed_fields(
    previous: PublicDecisionExplanation | None,
    decision: DecisionRecord,
    outcome: OutcomeProjection,
    contributions: tuple[SourceContribution, ...],
) -> tuple[str, ...]:
    if previous is None:
        return ()
    fields: list[str] = []
    if previous.decision_revision != decision.revision:
        fields.append("decision_revision")
    if previous.decision_status != decision.status.value:
        fields.append("decision_status")
    if previous.outcome != outcome:
        fields.append("outcome")
    if previous.contributions != contributions:
        fields.append("contributions")
    return tuple(fields)


def _deterministic_template(
    disposition: ExplanationDisposition, status: DecisionStatus, outcome: str
) -> str:
    return f"decision_explanation.{disposition.value}.{status.value}.{outcome}.v1"


def _migrate_explanation_payload(value: object) -> object:
    if not isinstance(value, dict):
        return value
    renderer = value.get("renderer")
    if not isinstance(renderer, dict) or "offered_clause_ids" in renderer:
        return value
    try:
        clauses = _deterministic_clauses(
            ExplanationDisposition(str(value["disposition"])),
            DecisionStatus(str(value["decision_status"])),
            str(value["outcome"]["status"]),
            bool(value.get("information_gap_codes")),
        )
    except (KeyError, TypeError, ValueError):
        return value
    migrated = dict(value)
    migrated["renderer"] = {
        "state": RendererState.DETERMINISTIC.value,
        "deterministic_template": renderer.get(
            "deterministic_template",
            _deterministic_template(
                ExplanationDisposition(str(value["disposition"])),
                DecisionStatus(str(value["decision_status"])),
                str(value["outcome"]["status"]),
            ),
        ),
        "offered_clause_ids": clauses,
        "ordered_clause_ids": clauses,
        "visible_explanation": _render_clauses(clauses),
        "failure_code": None,
    }
    return migrated


def _deterministic_clauses(
    disposition: ExplanationDisposition,
    status: DecisionStatus,
    outcome: str,
    has_gaps: bool,
) -> tuple[str, ...]:
    return (
        f"disposition.{disposition.value}.v1",
        f"decision_status.{status.value}.v1",
        f"information_gaps.{'present' if has_gaps else 'none'}.v1",
        f"outcome.{outcome}.v1",
    )


def _render_clauses(clause_ids: tuple[str, ...]) -> str:
    return " ".join(_CLAUSE_TEMPLATES[item] for item in clause_ids)


def _empty_risk() -> RiskProjection:
    return RiskProjection(
        risk_class="filtered",
        policy_status="filtered",
        approval_status="filtered",
    )


def _safe_reason(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not _SAFE_CODE.fullmatch(normalized):
        return "unavailable_reason_code"
    return normalized


def _validate_safe_tree(value: Any, key: str | None = None) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(child_key).lower())
            if normalized in {
                "hiddenthought",
                "rawprompt",
                "prompt",
                "reasoning",
                "chainofthought",
                "secret",
                "credential",
                "attachmentbody",
                "attachmentpath",
                "memorypayload",
            }:
                raise ValueError("private field is forbidden in public explanation")
            _validate_safe_tree(child, str(child_key))
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_safe_tree(child, key)
        return
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _PRIVATE_MARKERS):
            raise ValueError("private marker is forbidden in public explanation")
        if key == "visible_explanation":
            return
        if key in {"created_at"}:
            datetime.fromisoformat(value)
            return
        if key in {"deterministic_template", "offered_clause_ids", "ordered_clause_ids"}:
            if not re.fullmatch(r"[a-z0-9_.]+", value):
                raise ValueError("invalid deterministic template")
            return
        if key and (
            key.endswith("code")
            or key.endswith("codes")
            or key.endswith("status")
            or key
            in {
                "state",
                "compatibility",
                "classification",
                "recommendation",
                "disposition",
                "source_type",
                "action_type",
                "risk_class",
                "policy_status",
                "approval_status",
            }
        ):
            if not _SAFE_CODE.fullmatch(value):
                raise ValueError("invalid safe explanation code")
            return
        if (
            key
            and (key.endswith("id") or key.endswith("ref") or key.endswith("refs"))
            and not _SAFE_ID.fullmatch(value)
        ):
            raise ValueError("invalid safe explanation reference")
