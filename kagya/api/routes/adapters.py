"""Adapter lifecycle routes."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_adapter_registry,
    get_adapter_runtime_manager,
    get_api_settings,
    get_model_provider,
    get_runtime_event_log,
    require_admin,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.adapter import (
    AdapterEvaluateRequest,
    AdapterCanaryRequest,
    AdapterEvaluateResponse,
    AdapterListResponse,
    AdapterResponse,
    AdapterActivationResponse,
    AdapterRuntimeStateResponse,
)
from kagya.config import Settings
from kagya.learning import (
    AdapterEntry,
    AdapterEvaluationResult,
    AdapterEvaluator,
    AdapterRegistry,
    AdapterRuntimeManager,
    AdapterStatus,
)
from kagya.models import ModelProvider, load_model_provider
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/adapters", tags=["adapters"], dependencies=[Depends(require_admin)]
)


@router.get("/runtime", response_model=AdapterRuntimeStateResponse)
def adapter_runtime_state(
    settings: Settings = Depends(get_api_settings),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    manager: AdapterRuntimeManager = Depends(get_adapter_runtime_manager),
) -> AdapterRuntimeStateResponse:
    state = execute_agent_event(
        runtime,
        AgentEventType.ADAPTER_READ,
        source="api.adapters.runtime",
        handler=manager.current,
    ).value
    return AdapterRuntimeStateResponse(
        base_model=settings.model.primary_id,
        adapter_id=state.adapter_id,
        adapter_hash=state.adapter_hash,
        activation_sequence=state.activation_sequence,
    )


@router.get("/{adapter_id}/provenance")
def adapter_provenance(
    adapter_id: str,
    registry: AdapterRegistry = Depends(get_adapter_registry),
    manager: AdapterRuntimeManager = Depends(get_adapter_runtime_manager),
) -> dict:
    entry = registry.lookup(adapter_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown adapter: {adapter_id}")
    return {
        "adapter": asdict(entry),
        "lineage": [asdict(item) for item in registry.lineage(adapter_id)],
        "activation_history": [
            asdict(record) for record in manager.history(adapter_id)
        ],
    }


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
    return AdapterListResponse(adapters=[adapter_response(entry) for entry in entries])


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
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
    manager: AdapterRuntimeManager = Depends(get_adapter_runtime_manager),
) -> AdapterResponse:
    try:
        manager.stage(adapter_id)
        manager.verify(adapter_id)
        return execute_agent_event(
            runtime,
            AgentEventType.ADAPTER_UPDATE,
            source="api.adapters.activate",
            handler=lambda: _activate(manager, registry, adapter_id, event_log),
            payload={"adapter_id": adapter_id},
        ).value
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/rollback", response_model=AdapterActivationResponse)
def rollback_adapter(
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
    manager: AdapterRuntimeManager = Depends(get_adapter_runtime_manager),
) -> AdapterActivationResponse:
    try:
        record = execute_agent_event(
            runtime,
            AgentEventType.ADAPTER_UPDATE,
            source="api.adapters.rollback",
            handler=manager.rollback,
        ).value
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_log.record(
        category="adapter",
        event_type="rollback",
        message="Adapter runtime rollback completed",
        metadata=record.__dict__,
    )
    return AdapterActivationResponse.model_validate(record.__dict__)


@router.post("/{adapter_id}/canary")
def report_adapter_canary(
    adapter_id: str,
    request: AdapterCanaryRequest,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    manager: AdapterRuntimeManager = Depends(get_adapter_runtime_manager),
) -> dict:
    current = manager.current()
    if current.adapter_id != adapter_id:
        raise HTTPException(status_code=400, detail="Adapter is not the active canary")
    try:
        rollback = execute_agent_event(
            runtime,
            AgentEventType.ADAPTER_UPDATE,
            source="api.adapters.canary",
            handler=lambda: manager.report_canary(success=request.success),
            payload={"adapter_id": adapter_id, "success": request.success},
        ).value
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "adapter_id": adapter_id,
        "success": request.success,
        "automatic_rollback": rollback is not None,
        "rollback": None if rollback is None else asdict(rollback),
    }


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
            deterministic_dimensions=request.deterministic_dimensions,
            deterministic_baselines=request.deterministic_baselines,
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
    manager: AdapterRuntimeManager,
    registry: AdapterRegistry,
    adapter_id: str,
    event_log: RuntimeEventLog,
) -> AdapterResponse:
    record = manager.activate_at_event_boundary(adapter_id)
    entry = registry.lookup(adapter_id)
    if entry is None:
        raise RuntimeError("Activated adapter disappeared from registry")
    _record_adapter_transition(event_log, entry, "activated")
    event_log.record(
        category="adapter",
        event_type="runtime_activated",
        message="Adapter runtime activation completed",
        metadata=record.__dict__,
    )
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
        base_model_revision=entry.base_model_revision,
        adapter_hash=entry.adapter_hash,
        parent_adapter_id=entry.parent_adapter_id,
        parent_adapter_hash=entry.parent_adapter_hash,
        activation_sequence=entry.activation_sequence,
        dataset_repetition_count=entry.dataset_repetition_count,
        dataset_overlap_count=entry.dataset_overlap_count,
        dataset_overlap_ratio=entry.dataset_overlap_ratio,
        holdout_score=entry.holdout_score,
        holdout_baseline_score=entry.holdout_baseline_score,
        holdout_regression=entry.holdout_regression,
        drift_scores=entry.drift_scores,
        activation_gate_passed=entry.activation_gate_passed,
        behavioral_evaluation_id=entry.behavioral_evaluation_id,
        behavioral_evaluation_path=entry.behavioral_evaluation_path,
        behavioral_gate_passed=entry.behavioral_gate_passed,
        rollout_state=entry.rollout_state,
        canary_failures=entry.canary_failures,
        rollback_target_id=entry.rollback_target_id,
    )
