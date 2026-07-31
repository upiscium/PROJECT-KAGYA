"""Structured decision record administration routes."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.decision import ActionCandidate, ActionType, DecisionStatus, PredictedOutcome
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictedOutcomeRequest(_RequestModel):
    outcome_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    utility: float = Field(ge=-1.0, le=1.0)


class ActionCandidateRequest(_RequestModel):
    candidate_id: str = Field(min_length=1)
    candidate_type: ActionType
    proposed_action: str = Field(min_length=1)
    parameters: dict[str, object] = Field(default_factory=dict)
    prerequisites: list[str] = Field(default_factory=list)
    predicted_outcomes: list[PredictedOutcomeRequest] = Field(default_factory=list)
    uncertainty: float = Field(ge=0.0, le=1.0)
    estimated_cost: float = Field(ge=0.0, le=1.0)
    estimated_risk: float = Field(ge=0.0, le=1.0)
    value_effects: dict[str, float] = Field(default_factory=dict)
    appraisal_contributions: dict[str, float] = Field(default_factory=dict)
    plan_id: str | None = None
    plan_revision: int | None = Field(default=None, ge=1)
    step_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)
    goal_refs: list[str] = Field(default_factory=list, max_length=32)
    commitment_refs: list[str] = Field(default_factory=list, max_length=32)
    belief_refs: list[str] = Field(default_factory=list, max_length=32)

    def to_domain(self) -> ActionCandidate:
        return ActionCandidate(
            candidate_id=self.candidate_id,
            candidate_type=self.candidate_type,
            proposed_action=self.proposed_action,
            parameters=self.parameters,
            prerequisites=tuple(self.prerequisites),
            predicted_outcomes=tuple(
                PredictedOutcome(**item.model_dump())
                for item in self.predicted_outcomes
            ),
            uncertainty=self.uncertainty,
            estimated_cost=self.estimated_cost,
            estimated_risk=self.estimated_risk,
            value_effects=self.value_effects,
            appraisal_contributions=self.appraisal_contributions,
            plan_id=self.plan_id,
            plan_revision=self.plan_revision,
            step_id=self.step_id,
            evidence_refs=tuple(self.evidence_refs),
            goal_refs=tuple(self.goal_refs),
            commitment_refs=tuple(self.commitment_refs),
            belief_refs=tuple(self.belief_refs),
        )


class DecisionRequest(_RequestModel):
    decision_id: str | None = Field(default=None, min_length=1)
    context_id: str | None = Field(default=None, min_length=1)
    boundary_assessment_id: str | None = Field(default=None, min_length=1)
    candidates: list[ActionCandidateRequest] = Field(min_length=1)
    satisfied_prerequisites: list[str] = Field(default_factory=list)


class CandidateGenerationRequest(_RequestModel):
    situation: str = Field(min_length=1)


class DecisionOutcomeRequest(_RequestModel):
    description: str = Field(min_length=1)
    utility: float = Field(ge=-1.0, le=1.0)
    success: bool


class ExplanationCreateRequest(_RequestModel):
    explanation_id: str | None = Field(default=None, min_length=1, max_length=128)
    context_id: str | None = Field(default=None, min_length=1, max_length=128)
    interlocutor_id: str | None = Field(default=None, min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)


class ExplanationReviseRequest(_RequestModel):
    expected_revision: int = Field(ge=1)
    context_id: str | None = Field(default=None, min_length=1, max_length=128)
    interlocutor_id: str | None = Field(default=None, min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)


class ExplanationRenderRequest(_RequestModel):
    expected_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=128)


router = APIRouter(
    prefix="/api/decisions",
    tags=["decisions"],
)


@router.get("")
def inspect_decisions(
    request: Request,
    status: DecisionStatus | None = None,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    coordinator = get_main_loop(request).plan_decision_coordinator
    records = execute_agent_event(
        runtime,
        AgentEventType.DECISION_READ,
        source="api.decisions.inspect",
        handler=lambda: coordinator.list_decisions(status),
    ).value
    return {"decisions": [asdict(record) for record in records]}


@router.post("")
def create_decision(
    body: DecisionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.DECISION_UPDATE,
            source="api.decisions.create",
            handler=lambda: main_loop.create_decision(
                [candidate.to_domain() for candidate in body.candidates],
                context_id=body.context_id,
                satisfied_prerequisites=set(body.satisfied_prerequisites),
                decision_id=body.decision_id,
                boundary_assessment_id=body.boundary_assessment_id,
            ),
            payload={"decision_id": body.decision_id, "context_id": body.context_id},
            correlation_id=body.decision_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(record)


@router.post("/generate")
def generate_candidates(
    body: CandidateGenerationRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        candidates = execute_agent_event(
            runtime,
            AgentEventType.DECISION_GENERATE,
            source="api.decisions.generate",
            handler=lambda: main_loop.generate_decision_candidates(body.situation),
        ).value
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail="Model output did not match the candidate schema"
        ) from exc
    return {"candidates": [asdict(candidate) for candidate in candidates]}


@router.post("/{decision_id}/outcome")
def record_outcome(
    decision_id: str,
    body: DecisionOutcomeRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.DECISION_UPDATE,
            source="api.decisions.outcome",
            handler=lambda: main_loop.record_decision_outcome(
                decision_id,
                description=body.description,
                utility=body.utility,
                success=body.success,
            ),
            payload={"decision_id": decision_id},
            correlation_id=decision_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(record)


@router.get("/explanations")
def list_explanations(
    request: Request,
    decision_id: str | None = None,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    records = execute_agent_event(
        runtime,
        AgentEventType.DECISION_EXPLANATION_READ,
        source="api.decisions.explanations.list",
        handler=lambda: main_loop.list_decision_explanations(decision_id),
    ).value
    return {"explanations": [item.public_json() for item in records]}


@router.get("/explanations/{explanation_id}")
def get_explanation(
    explanation_id: str,
    request: Request,
    revision: int | None = None,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.DECISION_EXPLANATION_READ,
            source="api.decisions.explanations.get",
            handler=lambda: main_loop.get_decision_explanation(
                explanation_id, revision
            ),
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.public_json()


@router.post("/{decision_id}/explanations")
def create_explanation(
    decision_id: str,
    body: ExplanationCreateRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.DECISION_EXPLANATION_CREATE,
            source="api.decisions.explanations.create",
            handler=lambda: main_loop.create_decision_explanation(
                decision_id,
                context_id=body.context_id,
                interlocutor_id=body.interlocutor_id,
                explanation_id=body.explanation_id,
                idempotency_key=body.idempotency_key,
            ),
            payload={"decision_id": decision_id},
            correlation_id=decision_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.public_json()


@router.post("/explanations/{explanation_id}/revisions")
def revise_explanation(
    explanation_id: str,
    body: ExplanationReviseRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.DECISION_EXPLANATION_REVISE,
            source="api.decisions.explanations.revise",
            handler=lambda: main_loop.revise_decision_explanation(
                explanation_id,
                expected_revision=body.expected_revision,
                context_id=body.context_id,
                interlocutor_id=body.interlocutor_id,
                idempotency_key=body.idempotency_key,
            ),
            payload={"explanation_id": explanation_id},
            correlation_id=explanation_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.public_json()


@router.post("/explanations/{explanation_id}/render")
def render_explanation(
    explanation_id: str,
    body: ExplanationRenderRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.DECISION_EXPLANATION_RENDER,
            source="api.decisions.explanations.render",
            handler=lambda: main_loop.render_decision_explanation(
                explanation_id,
                expected_revision=body.expected_revision,
                idempotency_key=body.idempotency_key,
            ),
            payload={"explanation_id": explanation_id},
            correlation_id=explanation_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.public_json()


@router.get("/dataset")
def decision_dataset(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    records = execute_agent_event(
        runtime,
        AgentEventType.DECISION_READ,
        source="api.decisions.dataset",
        handler=main_loop.decision_dataset,
    ).value
    return {"records": [record.to_json() for record in records]}
