"""Adapter lifecycle routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_adapter_registry,
    get_api_settings,
    get_model_provider,
    get_runtime_event_log,
    require_admin,
    sync_main_loop_to_active_adapter,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.adapter import (
    AdapterEvaluateRequest,
    AdapterEvaluateResponse,
    AdapterListResponse,
    AdapterResponse,
)
from kagya.config import Settings
from kagya.learning import (
    AdapterEntry,
    AdapterEvaluationResult,
    AdapterEvaluator,
    AdapterRegistry,
    AdapterStatus,
)
from kagya.models import ModelProvider, load_model_provider
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/adapters", tags=["adapters"], dependencies=[Depends(require_admin)]
)


@router.get("", response_model=AdapterListResponse)
def list_adapters(
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterListResponse:
    entries = execute_agent_event(
        runtime,
        AgentEventType.ADAPTER_READ,
        source="api.adapters.list",
        handler=registry.list,
    ).value
    return AdapterListResponse(
        adapters=[adapter_response(entry) for entry in entries]
    )


@router.post("/{adapter_id}/evaluate", response_model=AdapterEvaluateResponse)
def evaluate_adapter(
    adapter_id: str,
    request: AdapterEvaluateRequest,
    http_request: Request,
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> AdapterEvaluateResponse:
    try:
        result = execute_agent_event(
            runtime,
            AgentEventType.ADAPTER_UPDATE,
            source="api.adapters.evaluate",
            handler=lambda: _evaluate_candidate(
                adapter_id,
                request,
                settings,
                registry,
                get_model_provider(http_request),
            ),
            payload={"adapter_id": adapter_id},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    entry = registry.lookup(adapter_id)
    event_log.record(
        category="adapter",
        event_type="evaluated",
        message="Adapter evaluation completed",
        metadata={
            "adapter_id": result.adapter_id,
            "score": result.score,
            "baseline_score": result.baseline_score,
            "candidate_score": result.candidate_score,
            "score_delta": result.score_delta,
            "decision": result.decision.value,
            "status": None if entry is None else entry.status.value,
        },
    )
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
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> AdapterResponse:
    return execute_agent_event(
        runtime,
        AgentEventType.ADAPTER_UPDATE,
        source="api.adapters.trial",
        handler=lambda: _transition(
            registry, adapter_id, AdapterStatus.TRIAL_ACTIVE, event_log
        ),
        payload={"adapter_id": adapter_id},
    ).value


@router.post("/{adapter_id}/approve", response_model=AdapterResponse)
def approve_adapter(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> AdapterResponse:
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.ADAPTER_UPDATE,
            source="api.adapters.approve",
            handler=lambda: _approve(registry, adapter_id, event_log),
            payload={"adapter_id": adapter_id},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{adapter_id}/activate", response_model=AdapterResponse)
def activate_adapter(
    adapter_id: str,
    request: Request,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> AdapterResponse:
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.ADAPTER_UPDATE,
            source="api.adapters.activate",
            handler=lambda: _activate(
                request, registry, adapter_id, event_log
            ),
            payload={"adapter_id": adapter_id},
        ).value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{adapter_id}/reject", response_model=AdapterResponse)
def reject_adapter(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> AdapterResponse:
    return execute_agent_event(
        runtime,
        AgentEventType.ADAPTER_UPDATE,
        source="api.adapters.reject",
        handler=lambda: _transition(
            registry, adapter_id, AdapterStatus.REJECTED, event_log
        ),
        payload={"adapter_id": adapter_id},
    ).value


def _transition(
    registry: AdapterRegistry,
    adapter_id: str,
    status: AdapterStatus,
    event_log: RuntimeEventLog,
) -> AdapterResponse:
    try:
        entry = registry.transition(adapter_id, status)
        _record_adapter_transition(event_log, entry, status.value)
        return adapter_response(entry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _evaluate_candidate(
    adapter_id: str,
    request: AdapterEvaluateRequest,
    settings: Settings,
    registry: AdapterRegistry,
    runtime_provider: ModelProvider,
) -> AdapterEvaluationResult:
    if request.deterministic_score is not None:
        return AdapterEvaluator(settings, registry).evaluate(
            adapter_id,
            runtime_provider,
            deterministic_score=request.deterministic_score,
        )
    entry = registry.lookup(adapter_id)
    if entry is None:
        raise ValueError(f"Unknown adapter: {adapter_id}")
    baseline = load_model_provider(settings)
    candidate = load_model_provider(
        settings,
        adapter_path=entry.path,
        allow_candidate_adapter=True,
    )
    return AdapterEvaluator(settings, registry).evaluate(
        adapter_id,
        candidate,
        baseline_provider=baseline,
    )


def _approve(
    registry: AdapterRegistry,
    adapter_id: str,
    event_log: RuntimeEventLog,
) -> AdapterResponse:
    entry = registry.approve(adapter_id)
    _record_adapter_transition(event_log, entry, "approved")
    return adapter_response(entry)


def _activate(
    request: Request,
    registry: AdapterRegistry,
    adapter_id: str,
    event_log: RuntimeEventLog,
) -> AdapterResponse:
    entry = registry.activate(adapter_id)
    sync_main_loop_to_active_adapter(request)
    _record_adapter_transition(event_log, entry, "activated")
    return adapter_response(entry)


def _record_adapter_transition(
    event_log: RuntimeEventLog, entry: AdapterEntry, event_type: str
) -> None:
    event_log.record(
        category="adapter",
        event_type=event_type,
        message="Adapter lifecycle transition completed",
        metadata={
            "adapter_id": entry.adapter_id,
            "status": entry.status.value,
            "eval_score": entry.eval_score,
        },
    )


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
