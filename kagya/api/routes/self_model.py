"""Self-model inspection and evidence-bound updates."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.identity import EpistemicUncertainty, KnownLimitation
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CapabilityEvidenceRequest(_RequestModel):
    capability_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class CapabilityCorrectionRequest(_RequestModel):
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class LimitationRequest(_RequestModel):
    limitation_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    capability_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class UncertaintyRequest(_RequestModel):
    uncertainty_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class IdentityProposalRequest(_RequestModel):
    proposal_id: str | None = Field(default=None, min_length=1)
    proposed_summary: str | None = Field(default=None, min_length=1)
    proposed_traits: dict[str, float] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    source: str = Field(min_length=1)


class IdentityResolutionRequest(_RequestModel):
    apply: bool
    reason: str = Field(min_length=1)


class SelfModelRollbackRequest(_RequestModel):
    target_revision: int = Field(ge=0)
    reason: str = Field(min_length=1)


router = APIRouter(
    prefix="/api/self-model",
    tags=["self-model"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def inspect_self_model(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    model = get_main_loop(request).self_model
    return execute_agent_event(
        runtime,
        AgentEventType.SELF_MODEL_READ,
        source="api.self_model.inspect",
        handler=lambda: model.to_json(),
    ).value


@router.post("/capabilities/from-decision")
def update_capability_from_decision(
    body: CapabilityEvidenceRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.self_model.capability_evidence",
            handler=lambda: main_loop.update_capability_from_decision(
                body.capability_id,
                body.description,
                body.decision_id,
                tags=tuple(body.tags),
            ),
            payload={
                "capability_id": body.capability_id,
                "decision_id": body.decision_id,
            },
            correlation_id=body.decision_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(state)


@router.post("/capabilities/{capability_id}/correction")
def correct_capability(
    capability_id: str,
    body: CapabilityCorrectionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.self_model.capability_correction",
            handler=lambda: main_loop.manual_correct_capability(
                capability_id,
                body.description,
                body.confidence,
                reason=body.reason,
                tags=tuple(body.tags),
            ),
            payload={"capability_id": capability_id},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(state)


@router.post("/limitations")
def add_limitation(
    body: LimitationRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    limitation = KnownLimitation(
        limitation_id=body.limitation_id,
        description=body.description,
        confidence=body.confidence,
        capability_ids=tuple(body.capability_ids),
        tags=tuple(body.tags),
        evidence_refs=tuple(body.evidence_refs),
    )
    state = execute_agent_event(
        runtime,
        AgentEventType.SELF_MODEL_UPDATE,
        source="api.self_model.limitation",
        handler=lambda: main_loop.add_self_limitation(limitation, reason=body.reason),
        payload={"limitation_id": body.limitation_id},
    ).value
    return asdict(state)


@router.post("/uncertainties")
def add_uncertainty(
    body: UncertaintyRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    uncertainty = EpistemicUncertainty(
        uncertainty_id=body.uncertainty_id,
        description=body.description,
        confidence=body.confidence,
        tags=tuple(body.tags),
        evidence_refs=tuple(body.evidence_refs),
    )
    state = execute_agent_event(
        runtime,
        AgentEventType.SELF_MODEL_UPDATE,
        source="api.self_model.uncertainty",
        handler=lambda: main_loop.add_self_uncertainty(
            uncertainty, reason=body.reason
        ),
        payload={"uncertainty_id": body.uncertainty_id},
    ).value
    return asdict(state)


@router.post("/identity/proposals")
def propose_identity_revision(
    body: IdentityProposalRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        proposal = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.self_model.identity_proposal",
            handler=lambda: main_loop.propose_identity_revision(
                proposed_summary=body.proposed_summary,
                proposed_traits=body.proposed_traits,
                evidence_refs=tuple(body.evidence_refs),
                source=body.source,
                proposal_id=body.proposal_id,
            ),
            payload={"proposal_id": body.proposal_id, "source": body.source},
            correlation_id=body.proposal_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(proposal)


@router.post("/identity/proposals/{proposal_id}/resolve")
def resolve_identity_revision(
    proposal_id: str,
    body: IdentityResolutionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        proposal = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.self_model.identity_resolution",
            handler=lambda: main_loop.resolve_identity_revision(
                proposal_id, apply=body.apply, reason=body.reason
            ),
            payload={"proposal_id": proposal_id, "apply": body.apply},
            correlation_id=proposal_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(proposal)


@router.post("/rollback")
def rollback_self_model(
    body: SelfModelRollbackRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = get_main_loop(request)
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source="api.self_model.rollback",
            handler=lambda: main_loop.rollback_self_model(
                body.target_revision, reason=body.reason
            ),
            payload={"target_revision": body.target_revision},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(state)
