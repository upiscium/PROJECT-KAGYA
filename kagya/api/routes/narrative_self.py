"""Admin inspection and revision routes for autobiographical continuity."""

from dataclasses import asdict
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
    require_admin,
)
from kagya.identity import IdentityClaimKind, IdentityClaimStatus
from kagya.runtime import AgentEventType, AgentRuntime


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeRequest(_RequestModel):
    experience_id: str = Field(min_length=1)


class ChapterRequest(_RequestModel):
    chapter_id: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1)
    theme_codes: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(min_length=2)


class ClaimRequest(_RequestModel):
    claim_id: str | None = Field(default=None, min_length=1)
    kind: IdentityClaimKind
    statement: str = Field(min_length=1)
    polarity: int = Field(ge=-1, le=1)
    theme_codes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(min_length=1)
    related_experience_ids: list[str] = Field(default_factory=list)
    related_value_refs: list[str] = Field(default_factory=list)
    related_goal_refs: list[str] = Field(default_factory=list)
    related_decision_refs: list[str] = Field(default_factory=list)


class ClaimRevisionRequest(_RequestModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    counterevidence_refs: list[str] = Field(default_factory=list)
    status: IdentityClaimStatus | None = None


class ContinuityRequest(_RequestModel):
    link_id: str | None = Field(default=None, min_length=1)
    earlier_ref: str = Field(min_length=1)
    later_ref: str = Field(min_length=1)
    relation_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class FutureSelfRequest(_RequestModel):
    projection_id: str | None = Field(default=None, min_length=1)
    description: str = Field(min_length=1)
    theme_codes: list[str] = Field(default_factory=list)
    desired_level: float = Field(ge=0.0, le=1.0)
    current_level: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(min_length=1)


router = APIRouter(
    prefix="/api/narrative-self",
    tags=["narrative-self"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
def inspect_narrative_self(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = get_main_loop(request).narrative_self
    return execute_agent_event(
        runtime,
        AgentEventType.SELF_MODEL_READ,
        source="api.narrative_self.inspect",
        handler=store.to_json,
    ).value


@router.post("/episodes/from-experience")
def form_episode(
    body: EpisodeRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute(
        runtime,
        source="api.narrative_self.episode",
        handler=lambda: get_main_loop(request).form_autobiographical_episode(
            body.experience_id
        ),
        payload={"experience_id": body.experience_id},
    )


@router.post("/chapters")
def create_chapter(
    body: ChapterRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute(
        runtime,
        source="api.narrative_self.chapter",
        handler=lambda: get_main_loop(request).create_narrative_chapter(
            title=body.title,
            theme_codes=tuple(body.theme_codes),
            episode_ids=tuple(body.episode_ids),
            chapter_id=body.chapter_id,
        ),
        payload={"chapter_id": body.chapter_id},
    )


@router.post("/claims")
def propose_claim(
    body: ClaimRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute(
        runtime,
        source="api.narrative_self.claim",
        handler=lambda: get_main_loop(request).propose_narrative_claim(
            kind=body.kind,
            statement=body.statement,
            polarity=body.polarity,
            theme_codes=tuple(body.theme_codes),
            confidence=body.confidence,
            stability=body.stability,
            evidence_refs=tuple(body.evidence_refs),
            related_experience_ids=tuple(body.related_experience_ids),
            related_value_refs=tuple(body.related_value_refs),
            related_goal_refs=tuple(body.related_goal_refs),
            related_decision_refs=tuple(body.related_decision_refs),
            claim_id=body.claim_id,
        ),
        payload={"claim_id": body.claim_id, "kind": body.kind.value},
    )


@router.post("/claims/{claim_id}/revise")
def revise_claim(
    claim_id: str,
    body: ClaimRevisionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute(
        runtime,
        source="api.narrative_self.claim_revision",
        handler=lambda: get_main_loop(request).revise_narrative_claim(
            claim_id,
            confidence=body.confidence,
            reason_code=body.reason_code,
            evidence_refs=tuple(body.evidence_refs),
            counterevidence_refs=tuple(body.counterevidence_refs),
            status=body.status,
        ),
        payload={"claim_id": claim_id},
    )


@router.post("/continuity-links")
def create_continuity_link(
    body: ContinuityRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute(
        runtime,
        source="api.narrative_self.continuity",
        handler=lambda: get_main_loop(request).create_narrative_continuity_link(
            body.earlier_ref,
            body.later_ref,
            relation_code=body.relation_code,
            evidence_refs=tuple(body.evidence_refs),
            confidence=body.confidence,
            link_id=body.link_id,
        ),
        payload={"link_id": body.link_id},
    )


@router.post("/future-self")
def set_future_self(
    body: FutureSelfRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    return _execute(
        runtime,
        source="api.narrative_self.future_self",
        handler=lambda: get_main_loop(request).set_future_self_projection(
            description=body.description,
            theme_codes=tuple(body.theme_codes),
            desired_level=body.desired_level,
            current_level=body.current_level,
            evidence_refs=tuple(body.evidence_refs),
            projection_id=body.projection_id,
        ),
        payload={"projection_id": body.projection_id},
    )


def _execute(
    runtime: AgentRuntime,
    *,
    source: str,
    handler: Callable[[], Any],
    payload: dict[str, object],
) -> dict[str, object]:
    try:
        value = execute_agent_event(
            runtime,
            AgentEventType.SELF_MODEL_UPDATE,
            source=source,
            handler=handler,
            payload=payload,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(value)
