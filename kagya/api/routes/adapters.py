"""Adapter lifecycle routes."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException

from kagya.artifact_provenance import build_adapter_artifact_manifest
from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_adapter_registry,
    get_adapter_runtime_manager,
    get_api_settings,
    get_runtime_event_log,
    require_admin,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.adapter import (
    AdapterEvaluateRequest,
    AdapterCanaryRequest,
    AdapterEvaluateResponse,
    AdapterBehavioralEvaluateRequest,
    AdapterBehavioralEvaluateResponse,
    AdapterBehavioralStatusResponse,
    AdapterListResponse,
    AdapterResponse,
    AdapterActivationResponse,
    AdapterRuntimeStateResponse,
)
from kagya.config import BehavioralActivationPolicy, ProjectEnvironment, Settings
from kagya.learning import (
    AdapterEntry,
    AdapterEvaluationResult,
    AdapterEvaluator,
    AdapterRegistry,
    AdapterRuntimeManager,
    AdapterStatus,
    BehavioralArtifactStore,
    BehavioralArtifactBusyError,
    BehavioralArtifactRecord,
    BehavioralArtifactStatus,
    run_deterministic_runtime_evaluation,
    run_real_model_runtime_evaluation,
)
from kagya.models import load_model_provider
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/adapters", tags=["adapters"], dependencies=[Depends(require_admin)]
)

ArtifactStatusValue = Literal[
    "not_run", "prepared", "valid", "hash_mismatch", "corrupt", "orphan"
]
ArtifactHashMatch = Literal["passed", "failed", "not_run"]
BehavioralPolicyValue = Literal[
    "real_model_required", "deterministic_runtime_only", "disabled"
]


@router.post(
    "/{adapter_id}/behavioral-evaluate",
    response_model=AdapterBehavioralEvaluateResponse,
)
def behavioral_evaluate_adapter(
    adapter_id: str,
    request: AdapterBehavioralEvaluateRequest,
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterBehavioralEvaluateResponse:
    entry = registry.lookup(adapter_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown adapter: {adapter_id}")
    if entry.adapter_hash is None or entry.base_model_revision is None:
        raise HTTPException(
            status_code=409,
            detail="Adapter lacks cryptographic runtime evaluation provenance",
        )

    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    reserved = False
    try:
        with store.adapter_lock(adapter_id, blocking=False):
            store.begin(request.evaluation_id, adapter_key=adapter_id)
            reserved = True
            store.mark_running(request.evaluation_id)
            result, artifact_status = _run_and_bind_behavioral_evaluation(
                settings=settings,
                registry=registry,
                runtime=runtime,
                store=store,
                adapter_id=adapter_id,
                request=request,
                entry_path=Path(entry.path),
                adapter_hash=entry.adapter_hash,
                base_model_revision=entry.base_model_revision,
            )
    except BehavioralArtifactBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        if reserved:
            store.fail(request.evaluation_id, "policy_rejected")
        raise
    except (ValueError, RuntimeError) as exc:
        if reserved:
            store.fail(request.evaluation_id, "evaluation_failed")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    manifest = result.manifest
    assert manifest is not None
    eligibility = registry.activation_eligibility(adapter_id)
    return AdapterBehavioralEvaluateResponse(
        evaluation_id=result.evaluation_id,
        adapter_id=adapter_id,
        runtime_kind=result.runtime_kind.value,
        activation_gate_passed=result.activation_gate_passed,
        deterministic_runtime_gate_passed=result.deterministic_runtime_gate_passed,
        real_model_runtime_gate_passed=result.real_model_runtime_gate_passed,
        source_commit_sha=manifest.source_commit_sha,
        adapter_hash=manifest.candidate_adapter_hash,
        base_model_revision=manifest.base_model_revision,
        fixture_set_hash=manifest.fixture_set_hash,
        activation_eligibility=eligibility.reason.value,
        candidate_score=result.candidate.aggregate_score,
        hard_gate_failures=[item.value for item in result.candidate.hard_gate_failures],
        regression_dimensions=[item.value for item in result.regression_dimensions],
        artifact_status=artifact_status,
        artifact_path=f"behavioral/{request.evaluation_id}.json",
    )


def _run_and_bind_behavioral_evaluation(
    *,
    settings: Settings,
    registry: AdapterRegistry,
    runtime: AgentRuntime,
    store: BehavioralArtifactStore,
    adapter_id: str,
    request: AdapterBehavioralEvaluateRequest,
    entry_path: Path,
    adapter_hash: str,
    base_model_revision: str,
):
    policy = settings.adapter_registry.behavioral_activation_policy
    if policy == BehavioralActivationPolicy.DISABLED:
        raise HTTPException(
            status_code=403, detail="Behavioral evaluation is disabled by policy"
        )
    runtime_kind = (
        "real_model_runtime"
        if policy == BehavioralActivationPolicy.REAL_MODEL_REQUIRED
        else "deterministic_runtime"
    )
    if request.runtime_kind is not None and request.runtime_kind != runtime_kind:
        raise HTTPException(
            status_code=(
                403
                if settings.project.environment == ProjectEnvironment.PRODUCTION
                else 409
            ),
            detail=f"Behavioral runtime kind is fixed by policy to {runtime_kind}",
        )
    evaluation_runner = (
        run_real_model_runtime_evaluation
        if runtime_kind == "real_model_runtime"
        else run_deterministic_runtime_evaluation
    )
    result, artifact_status = evaluation_runner(
        settings,
        request.evaluation_id,
        baseline_id=request.baseline_id,
        candidate_id=adapter_id,
        candidate_adapter_path=entry_path,
        candidate_adapter_hash=adapter_hash,
        base_model_revision=base_model_revision,
        subject_revision=request.subject_revision,
    )
    if artifact_status != "prepared":
        raise ValueError("Behavioral artifact did not enter prepared state")

    def bind_and_finalize() -> str:
        registry.prepare_behavioral_evaluation(
            adapter_id,
            evaluation_id=request.evaluation_id,
            prepared_path=store.prepared_path(request.evaluation_id),
            final_path=store.final_path(request.evaluation_id),
        )
        store.finalize(request.evaluation_id)
        registry.finalize_behavioral_evaluation(
            adapter_id, evaluation_id=request.evaluation_id
        )
        reconciled = store.reconcile(registry)
        status = next(
            item.status.value
            for item in reconciled
            if item.evaluation_id == request.evaluation_id
        )
        if status != "valid":
            raise ValueError("Behavioral artifact cross-reconciliation failed")
        registry.mark_behavioral_evaluation_reconciled(
            adapter_id, evaluation_id=request.evaluation_id
        )
        store.mark_reconciled(request.evaluation_id)
        return status

    artifact_status = execute_agent_event(
        runtime,
        AgentEventType.BEHAVIORAL_EVALUATE,
        source="api.adapters.behavioral_evaluation_binding",
        handler=bind_and_finalize,
        payload={"adapter_id": adapter_id, "evaluation_id": request.evaluation_id},
    ).value
    return result, artifact_status


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


@router.get(
    "/{adapter_id}/behavioral-evaluation-status",
    response_model=AdapterBehavioralStatusResponse,
)
def behavioral_evaluation_status(
    adapter_id: str,
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
) -> AdapterBehavioralStatusResponse:
    entry = registry.lookup(adapter_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown adapter: {adapter_id}")
    eligibility = registry.activation_eligibility(adapter_id)
    artifacts = {
        item.evaluation_id: item
        for item in BehavioralArtifactStore(
            settings.adapter_registry.eval_result_dir
        ).reconcile(registry)
    }
    return AdapterBehavioralStatusResponse(
        adapter_id=adapter_id,
        policy=settings.adapter_registry.behavioral_activation_policy,
        ordinary_gates={
            "quality": _gate_status(entry.quality_gate_passed),
            "holdout": _gate_status(entry.holdout_gate_passed),
            "drift": _gate_status(entry.drift_gate_passed),
        },
        deterministic_status=eligibility.deterministic_status,
        deterministic_coverage=_coverage_status(entry.deterministic_coverage_complete),
        deterministic_artifact=_artifact_status(
            entry.behavioral_evaluation_id,
            entry.behavioral_artifact_state,
            artifacts,
        ),
        real_status=eligibility.real_model_status,
        real_coverage=_coverage_status(entry.real_model_coverage_complete),
        real_required=eligibility.real_model_required,
        real_artifact=_artifact_status(
            entry.real_model_behavioral_evaluation_id,
            entry.real_model_behavioral_artifact_state,
            artifacts,
        ),
        activation_eligible=eligibility.eligible,
        activation_reason=eligibility.reason,
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
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AdapterListResponse:
    entries = execute_agent_event(
        runtime,
        AgentEventType.ADAPTER_READ,
        source="api.adapters.list",
        handler=registry.list,
    ).value
    artifacts = {
        item.evaluation_id: item
        for item in BehavioralArtifactStore(
            settings.adapter_registry.eval_result_dir
        ).reconcile(registry)
    }
    return AdapterListResponse(
        adapters=[
            adapter_response(
                entry,
                registry.activation_eligibility(entry.adapter_id),
                artifacts=artifacts,
            )
            for entry in entries
        ]
    )


@router.post("/{adapter_id}/evaluate", response_model=AdapterEvaluateResponse)
def evaluate_adapter(
    adapter_id: str,
    _request: AdapterEvaluateRequest,
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
                settings,
                registry,
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
        return adapter_response(entry, registry.activation_eligibility(adapter_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _evaluate_candidate(
    adapter_id: str,
    settings: Settings,
    registry: AdapterRegistry,
) -> AdapterEvaluationResult:
    entry = registry.lookup(adapter_id)
    if entry is None:
        raise ValueError(f"Unknown adapter: {adapter_id}")
    baseline = load_model_provider(settings)
    if settings.model.provider.lower() == "transformers":
        manifest = build_adapter_artifact_manifest(
            Path(entry.path),
            base_model_name=entry.base_model,
            base_model_revision=entry.base_model_revision,
        )
        candidate = load_model_provider(
            settings,
            adapter_path=entry.path,
            allow_candidate_adapter=True,
            expected_adapter_hash=entry.adapter_hash,
            expected_adapter_manifest=manifest,
        )
    else:
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
    return adapter_response(entry, registry.activation_eligibility(adapter_id))


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
    return adapter_response(entry, registry.activation_eligibility(adapter_id))


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


def adapter_response(
    entry: AdapterEntry,
    eligibility: object | None = None,
    *,
    artifacts: dict[str, BehavioralArtifactRecord] | None = None,
) -> AdapterResponse:
    deterministic_artifact = _artifact_status(
        entry.behavioral_evaluation_id,
        entry.behavioral_artifact_state,
        artifacts,
    )
    real_artifact = _artifact_status(
        entry.real_model_behavioral_evaluation_id,
        entry.real_model_behavioral_artifact_state,
        artifacts,
    )
    reason = str(getattr(getattr(eligibility, "reason", ""), "value", ""))
    if deterministic_artifact == "valid":
        if reason in {
            "behavioral_result_missing",
        }:
            deterministic_artifact = "orphan"
        elif reason in {
            "behavioral_result_corrupt",
            "behavioral_result_schema_invalid",
        }:
            deterministic_artifact = "corrupt"
        elif reason in {
            "behavioral_result_stale",
            "behavioral_result_tampered",
            "behavioral_binding_mismatch",
            "adapter_artifact_mismatch",
        }:
            deterministic_artifact = "hash_mismatch"
    real_validation = str(getattr(eligibility, "real_model_status", "not_run"))
    if real_artifact == "valid" and real_validation == "corrupt":
        real_artifact = "corrupt"
    elif real_artifact == "valid" and real_validation == "stale":
        real_artifact = "hash_mismatch"
    evaluated_artifacts = [
        status
        for status, evaluation_id in (
            (deterministic_artifact, entry.behavioral_evaluation_id),
            (real_artifact, entry.real_model_behavioral_evaluation_id),
        )
        if evaluation_id is not None
    ]
    hash_match: ArtifactHashMatch = (
        "not_run"
        if not evaluated_artifacts
        else "failed"
        if any(
            status in {"hash_mismatch", "corrupt", "orphan"}
            for status in evaluated_artifacts
        )
        else "passed"
        if all(status == "valid" for status in evaluated_artifacts)
        else "not_run"
    )
    return AdapterResponse(
        adapter_id=entry.adapter_id,
        base_model=entry.base_model,
        path=_safe_path_reference(entry.path) or "",
        status=entry.status.value,
        dataset_path=_safe_path_reference(entry.dataset_path) or "",
        dataset_hash=entry.dataset_hash,
        eval_score=entry.eval_score,
        eval_result_path=_safe_path_reference(entry.eval_result_path),
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
        quality_gate_passed=entry.quality_gate_passed,
        holdout_gate_passed=entry.holdout_gate_passed,
        drift_gate_passed=entry.drift_gate_passed,
        activation_gate_passed=(
            entry.activation_gate_passed
            if eligibility is None
            else bool(getattr(eligibility, "eligible", False))
        ),
        behavioral_evaluation_id=entry.behavioral_evaluation_id,
        behavioral_evaluation_path=_safe_path_reference(
            entry.behavioral_evaluation_path
        ),
        behavioral_result_hash=entry.behavioral_result_hash,
        behavioral_gate_passed=entry.behavioral_gate_passed,
        behavioral_candidate_adapter_hash=entry.behavioral_candidate_adapter_hash,
        behavioral_base_model_revision=entry.behavioral_base_model_revision,
        subject_revision=entry.subject_revision,
        fixture_set_hash=entry.fixture_set_hash,
        behavioral_artifact_state=entry.behavioral_artifact_state,
        deterministic_coverage_status=_coverage_status(
            entry.deterministic_coverage_complete
        ),
        deterministic_behavioral_artifact_status=deterministic_artifact,
        real_model_behavioral_evaluation_id=entry.real_model_behavioral_evaluation_id,
        real_model_behavioral_gate_passed=entry.real_model_behavioral_gate_passed,
        real_model_behavioral_artifact_state=entry.real_model_behavioral_artifact_state,
        real_model_coverage_status=_coverage_status(entry.real_model_coverage_complete),
        real_model_behavioral_artifact_status=real_artifact,
        behavioral_artifact_hash_match=hash_match,
        activation_eligibility_reason=reason,
        real_model_behavioral_required=bool(
            getattr(eligibility, "real_model_required", False)
        ),
        behavioral_activation_policy=cast(
            BehavioralPolicyValue,
            getattr(
                eligibility,
                "policy",
                BehavioralActivationPolicy.REAL_MODEL_REQUIRED,
            ).value,
        ),
        legacy_activation_warning=entry.legacy_activation_warning,
        rollout_state=entry.rollout_state,
        canary_failures=entry.canary_failures,
        rollback_target_id=entry.rollback_target_id,
    )


def _coverage_status(
    value: bool | None,
) -> Literal["complete", "incomplete", "not_evaluated"]:
    if value is None:
        return "not_evaluated"
    return "complete" if value else "incomplete"


def _artifact_status(
    evaluation_id: str | None,
    binding_state: str,
    artifacts: dict[str, BehavioralArtifactRecord] | None,
) -> ArtifactStatusValue:
    if evaluation_id is None:
        return "not_run"
    if artifacts is None:
        binding_statuses: dict[str, ArtifactStatusValue] = {
            "prepared": "prepared",
            "finalized": "valid",
            "reconciled": "valid",
            "quarantined": "corrupt",
        }
        return binding_statuses.get(binding_state, "orphan")
    record = artifacts.get(evaluation_id)
    if record is None:
        return "orphan"
    reconciled_statuses: dict[BehavioralArtifactStatus, ArtifactStatusValue] = {
        BehavioralArtifactStatus.VALID: "valid",
        BehavioralArtifactStatus.PREPARED: "prepared",
        BehavioralArtifactStatus.HASH_MISMATCH: "hash_mismatch",
        BehavioralArtifactStatus.CORRUPT: "corrupt",
        BehavioralArtifactStatus.ORPHAN_RESULT: "orphan",
        BehavioralArtifactStatus.ORPHAN_REGISTRY_REFERENCE: "orphan",
    }
    return reconciled_statuses[record.status]


def _safe_path_reference(value: str | None) -> str | None:
    return None if value is None else Path(value).name


def _gate_status(value: bool | None) -> Literal["passed", "failed", "not_run"]:
    return "passed" if value is True else "failed" if value is False else "not_run"
