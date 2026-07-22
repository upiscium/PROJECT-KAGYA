"""Admin-only relationship inspection, correction, and identity mapping."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.relationship import PerceivedAttribute
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttributeRequest(_RequestModel):
    value: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


class RelationshipCorrectionRequest(_RequestModel):
    reason: str = Field(min_length=1, max_length=200)
    evidence_refs: list[str] = Field(min_length=1)
    axes: dict[str, float] | None = None
    perceived_identity: AttributeRequest | None = None
    perceived_role: AttributeRequest | None = None
    expectations: dict[str, AttributeRequest] | None = None
    boundaries: dict[str, AttributeRequest] | None = None
    other_values: dict[str, AttributeRequest] | None = None
    other_beliefs: dict[str, AttributeRequest] | None = None
    reciprocity: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    commitment_refs: list[str] | None = None
    unresolved_matter_refs: list[str] | None = None
    conflict_refs: list[str] | None = None
    repair_refs: list[str] | None = None


class AliasRequest(_RequestModel):
    interlocutor_key: str = Field(min_length=1, max_length=200)
    evidence_refs: list[str] = Field(min_length=2)


class SplitRequest(_RequestModel):
    interlocutor_key: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=200)
    evidence_refs: list[str] = Field(min_length=1)


router = APIRouter(
    prefix="/api/relationships",
    tags=["relationships"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def list_relationships(
    request: Request, runtime: AgentRuntime = Depends(get_agent_runtime)
) -> dict[str, object]:
    store = get_main_loop(request).relationship_store
    relationships = execute_agent_event(
        runtime,
        AgentEventType.RELATIONSHIP_READ,
        source="api.relationships.list",
        handler=store.list_relationships,
    ).value
    return {
        "schema_version": store.SCHEMA_VERSION,
        "relationships": [item.to_json() for item in relationships],
    }


@router.get("/{relationship_id}")
def inspect_relationship(
    relationship_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.RELATIONSHIP_READ,
            source="api.relationships.inspect",
            handler=lambda: get_main_loop(request).relationship_store.get(
                relationship_id
            ),
            correlation_id=relationship_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return state.to_json()


@router.post("/{relationship_id}/corrections")
def correct_relationship(
    relationship_id: str,
    body: RelationshipCorrectionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    updates = {
        "axes": body.axes,
        "perceived_identity": _attribute(body.perceived_identity),
        "perceived_role": _attribute(body.perceived_role),
        "expectations": _attributes(body.expectations),
        "boundaries": _attributes(body.boundaries),
        "other_values": _attributes(body.other_values),
        "other_beliefs": _attributes(body.other_beliefs),
        "reciprocity": body.reciprocity,
        "uncertainty": body.uncertainty,
        "commitment_refs": None
        if body.commitment_refs is None
        else tuple(body.commitment_refs),
        "unresolved_matter_refs": None
        if body.unresolved_matter_refs is None
        else tuple(body.unresolved_matter_refs),
        "conflict_refs": None
        if body.conflict_refs is None
        else tuple(body.conflict_refs),
        "repair_refs": None if body.repair_refs is None else tuple(body.repair_refs),
    }
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.RELATIONSHIP_UPDATE,
            source="api.relationships.correct",
            handler=lambda: get_main_loop(request).correct_relationship(
                relationship_id,
                reason=body.reason,
                evidence_refs=tuple(body.evidence_refs),
                **updates,
            ),
            payload={"relationship_id": relationship_id},
            correlation_id=relationship_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.to_json()


@router.post("/{relationship_id}/aliases")
def attach_alias(
    relationship_id: str,
    body: AliasRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.RELATIONSHIP_UPDATE,
            source="api.relationships.alias",
            handler=lambda: get_main_loop(request).attach_relationship_alias(
                relationship_id,
                body.interlocutor_key,
                evidence_refs=tuple(body.evidence_refs),
            ),
            payload={"relationship_id": relationship_id},
            correlation_id=relationship_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.to_json()


@router.post("/{relationship_id}/split")
def split_relationship(
    relationship_id: str,
    body: SplitRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    try:
        state = execute_agent_event(
            runtime,
            AgentEventType.RELATIONSHIP_UPDATE,
            source="api.relationships.split",
            handler=lambda: get_main_loop(request).split_relationship_alias(
                relationship_id,
                body.interlocutor_key,
                reason=body.reason,
                evidence_refs=tuple(body.evidence_refs),
            ),
            payload={"relationship_id": relationship_id},
            correlation_id=relationship_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.to_json()


def _attribute(value: AttributeRequest | None) -> PerceivedAttribute | None:
    if value is None:
        return None
    return PerceivedAttribute(
        value=value.value,
        confidence=value.confidence,
        evidence_refs=tuple(value.evidence_refs),
    )


def _attributes(
    values: dict[str, AttributeRequest] | None,
) -> dict[str, PerceivedAttribute] | None:
    if values is None:
        return None
    return {
        key: PerceivedAttribute(
            value=value.value,
            confidence=value.confidence,
            evidence_refs=tuple(value.evidence_refs),
        )
        for key, value in values.items()
    }
