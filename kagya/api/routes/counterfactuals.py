"""Admin API for structured counterfactual simulation history."""

from typing import Protocol, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_main_loop,
)
from kagya.counterfactual import (
    AlternativeOutcome,
    CounterfactualSignal,
    CounterfactualSimulation,
    CounterfactualStore,
)
from kagya.runtime import AgentEventType, AgentRuntime


class _CounterfactualRuntime(Protocol):
    counterfactual_store: CounterfactualStore

    def revise_counterfactual(
        self,
        simulation_id: str,
        *,
        expected_revision: int,
        agency_attribution_revision: int,
        alternatives: tuple[AlternativeOutcome, ...],
        signal: CounterfactualSignal,
        signal_magnitude: float,
        confidence: float,
        evidence_refs: tuple[str, ...],
        reason_code: str,
    ) -> CounterfactualSimulation: ...


def _main_loop(request: Request) -> _CounterfactualRuntime:
    return cast(_CounterfactualRuntime, get_main_loop(request))


class CounterfactualRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=1)
    agency_attribution_revision: int = Field(ge=1)
    alternatives: tuple[AlternativeOutcome, ...] = Field(min_length=1, max_length=8)
    signal: CounterfactualSignal
    signal_magnitude: float = Field(ge=0.0, le=0.5, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=0.8, allow_inf_nan=False)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    reason_code: str = Field(pattern=r"^[A-Za-z0-9_.:@/-]+$", max_length=128)


router = APIRouter(
    prefix="/api/counterfactuals",
    tags=["counterfactual-simulation"],
)


@router.get("")
def list_counterfactuals(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = _main_loop(request).counterfactual_store
    records = execute_agent_event(
        runtime,
        AgentEventType.COUNTERFACTUAL_READ,
        source="api.counterfactuals.list",
        handler=store.list_current,
    ).value
    return {"counterfactuals": [item.model_dump(mode="json") for item in records]}


@router.get("/{simulation_id}")
def counterfactual_history(
    simulation_id: str,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    store = _main_loop(request).counterfactual_store
    try:
        records = execute_agent_event(
            runtime,
            AgentEventType.COUNTERFACTUAL_READ,
            source="api.counterfactuals.history",
            handler=lambda: store.history(simulation_id),
            correlation_id=simulation_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"revisions": [item.model_dump(mode="json") for item in records]}


@router.post("/{simulation_id}/revisions")
def revise_counterfactual(
    simulation_id: str,
    body: CounterfactualRevisionRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> dict[str, object]:
    main_loop = _main_loop(request)
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.COUNTERFACTUAL_REVISE,
            source="api.counterfactuals.revise",
            handler=lambda: main_loop.revise_counterfactual(
                simulation_id,
                expected_revision=body.expected_revision,
                agency_attribution_revision=body.agency_attribution_revision,
                alternatives=body.alternatives,
                signal=body.signal,
                signal_magnitude=body.signal_magnitude,
                confidence=body.confidence,
                evidence_refs=body.evidence_refs,
                reason_code=body.reason_code,
            ),
            payload={
                "simulation_id": simulation_id,
                "expected_revision": body.expected_revision,
            },
            correlation_id=simulation_id,
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.model_dump(mode="json")
