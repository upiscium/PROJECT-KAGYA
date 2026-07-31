"""Admin read and revision API for structured agency attribution."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from typing import Protocol, cast

from kagya.agency import AgencyAttribution, AgencyAttributionStore, CausalContributor
from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.runtime import AgentEventType, AgentRuntime


class _AttributionRuntime(Protocol):
    agency_attribution_store: AgencyAttributionStore

    def revise_agency_attribution(
        self,
        attribution_id: str,
        *,
        expected_revision: int,
        contributors: tuple[CausalContributor, ...],
        intended: bool,
        uncertainty: float,
        evidence_refs: tuple[str, ...],
        reason_code: str,
    ) -> AgencyAttribution: ...


def _main_loop(request: Request) -> _AttributionRuntime:
    return cast(_AttributionRuntime, get_main_loop(request))


class AttributionRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    contributors: tuple[CausalContributor, ...] = Field(min_length=1, max_length=8)
    intended: bool
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    reason_code: str = Field(pattern=r"^[A-Za-z0-9_.:@/-]+$", max_length=128)


router = APIRouter(
    prefix="/api/attributions",
    tags=["agency-attribution"],
)


@router.get("")
def list_attributions(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = _main_loop(request).agency_attribution_store
    records = execute_agent_event(
        runtime,
        AgentEventType.ATTRIBUTION_READ,
        source="api.attributions.list",
        handler=store.list_current,
    ).value
    return {"attributions": [item.model_dump(mode="json") for item in records]}


@router.get("/{attribution_id}")
def attribution_history(
    attribution_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = _main_loop(request).agency_attribution_store
    try:
        records = execute_agent_event(
            runtime,
            AgentEventType.ATTRIBUTION_READ,
            source="api.attributions.history",
            handler=lambda: store.history(attribution_id),
            correlation_id=attribution_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"revisions": [item.model_dump(mode="json") for item in records]}


@router.post("/{attribution_id}/revisions")
def revise_attribution(
    attribution_id: str,
    body: AttributionRevisionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = _main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.ATTRIBUTION_REVISE,
            source="api.attributions.revise",
            handler=lambda: main_loop.revise_agency_attribution(
                attribution_id,
                expected_revision=body.expected_revision,
                contributors=body.contributors,
                intended=body.intended,
                uncertainty=body.uncertainty,
                evidence_refs=body.evidence_refs,
                reason_code=body.reason_code,
            ),
            payload={
                "attribution_id": attribution_id,
                "expected_revision": body.expected_revision,
            },
            correlation_id=attribution_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.model_dump(mode="json")
