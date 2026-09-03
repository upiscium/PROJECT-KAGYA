"""Adapter lifecycle routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import (
    get_adapter_registry,
    get_agent_runtime,
    get_api_settings,
    get_model_provider,
    require_admin,
)
from kagya.api.runtime_execution import execute
from kagya.api.schemas.adapter import (
    AdapterEvaluateRequest,
    AdapterEvaluateResponse,
    AdapterListResponse,
    AdapterResponse,
)
from kagya.config import Settings
from kagya.learning import (
    AdapterEntry,
    AdapterEvaluator,
    AdapterRegistry,
    AdapterStatus,
)
from kagya.models import ModelProvider
from kagya.runtime import AgentEventSource, AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/adapters", tags=["adapters"], dependencies=[Depends(require_admin)]
)


@router.get("", response_model=AdapterListResponse)
def list_adapters(
    registry: AdapterRegistry = Depends(get_adapter_registry),
) -> AdapterListResponse:
    return AdapterListResponse(
        adapters=[adapter_response(entry) for entry in registry.list()]
    )


@router.post("/{adapter_id}/evaluate", response_model=AdapterEvaluateResponse)
def evaluate_adapter(
    adapter_id: str,
    request: AdapterEvaluateRequest,
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
    provider: ModelProvider = Depends(get_model_provider),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterEvaluateResponse:
    def evaluate_and_read_entry():
        result = AdapterEvaluator(settings, registry).evaluate(
            adapter_id,
            provider,
            deterministic_score=request.deterministic_score,
        )
        return result, registry.lookup(adapter_id)

    try:
        result, entry = execute(
            runtime,
            AgentEventType.ADAPTER_EVALUATE,
            AgentEventSource.API_ADAPTER_EVALUATE,
            evaluate_and_read_entry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AdapterEvaluateResponse(
        adapter_id=result.adapter_id,
        score=result.score,
        decision=result.decision.value,
        result_path=result.result_path,
        status="" if entry is None else entry.status.value,
    )


@router.post("/{adapter_id}/trial", response_model=AdapterResponse)
def trial_adapter(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterResponse:
    return _transition(runtime, registry, adapter_id, AdapterStatus.TRIAL_ACTIVE)


@router.post("/{adapter_id}/approve", response_model=AdapterResponse)
def approve_adapter(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterResponse:
    try:
        return adapter_response(
            execute(
                runtime,
                AgentEventType.ADAPTER_UPDATE,
                AgentEventSource.API_ADAPTER_APPROVE,
                lambda: registry.approve(adapter_id),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{adapter_id}/activate", response_model=AdapterResponse)
def activate_adapter(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterResponse:
    try:
        return adapter_response(
            execute(
                runtime,
                AgentEventType.ADAPTER_UPDATE,
                AgentEventSource.API_ADAPTER_ACTIVATE,
                lambda: registry.activate(adapter_id),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{adapter_id}/reject", response_model=AdapterResponse)
def reject_adapter(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterResponse:
    return _transition(runtime, registry, adapter_id, AdapterStatus.REJECTED)


def _transition(
    runtime: AgentRuntime,
    registry: AdapterRegistry,
    adapter_id: str,
    status: AdapterStatus,
) -> AdapterResponse:
    source = (
        AgentEventSource.API_ADAPTER_TRIAL
        if status is AdapterStatus.TRIAL_ACTIVE
        else AgentEventSource.API_ADAPTER_REJECT
    )
    try:
        return adapter_response(
            execute(
                runtime,
                AgentEventType.ADAPTER_UPDATE,
                source,
                lambda: registry.transition(adapter_id, status),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def adapter_response(entry: AdapterEntry) -> AdapterResponse:
    return AdapterResponse(
        adapter_id=entry.adapter_id,
        base_model=entry.base_model,
        path=entry.path,
        status=entry.status.value,
        dataset_path=entry.dataset_path,
        dataset_hash=entry.dataset_hash,
        eval_score=entry.eval_score,
        eval_result_path=entry.eval_result_path,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        notes=entry.notes,
    )
