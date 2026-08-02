"""Administrative agent-state snapshot operations."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_api_settings,
    get_adapter_runtime_manager,
    get_dataset_governance,
    get_main_loop,
    get_operator_restore_service,
    get_sleep_coordinator,
    get_tool_executor,
    get_tool_registry,
)
from kagya.config import Settings
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    OperatorRestoreService,
    RestoreCommitRequest,
    RestoreCommitResponse,
    RestoreContractError,
    RestoreErrorCode,
    RestorePreview as OperatorRestorePreview,
    RestoreSummary,
    event_id_for_operation,
)
from kagya.security.backup import (
    BackupError,
    BackupManager,
    BackupStatus,
    RestorePreview,
)
from kagya.security.crypto import EncryptionError


router = APIRouter(prefix="/api/state", tags=["state"])
_OPERATOR_ACTOR = "subject-cockpit-operator"


class BackupCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_backup_id: str | None = None


class BackupRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_backup_id: str
    expected_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkingMemorySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    item_capacity: int = Field(gt=0)
    token_capacity: int = Field(gt=0)


def _restore_error(error: RestoreContractError) -> HTTPException:
    status_code = (
        status.HTTP_404_NOT_FOUND
        if error.code == RestoreErrorCode.TARGET_NOT_RETAINED.value
        else status.HTTP_409_CONFLICT
    )
    return HTTPException(status_code=status_code, detail={"code": error.code})


def _restore_lock(request: Request):
    lock = getattr(request.app.state, "operator_restore_lock", None)
    if lock is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": RestoreErrorCode.NOT_AUTHORITATIVE.value},
        )
    return lock


@router.get("/operator-restore/summary", response_model=RestoreSummary)
def operator_restore_summary(
    request: Request,
    limit: int = Query(default=100, ge=1, le=100),
    service: OperatorRestoreService = Depends(get_operator_restore_service),
) -> RestoreSummary:
    try:
        with _restore_lock(request):
            return service.summary(limit, _OPERATOR_ACTOR)
    except RestoreContractError as exc:
        raise _restore_error(exc) from exc


@router.post(
    "/operator-restore/preview/{target_sequence}",
    response_model=OperatorRestorePreview,
)
def preview_operator_restore(
    target_sequence: int,
    request: Request,
    service: OperatorRestoreService = Depends(get_operator_restore_service),
) -> OperatorRestorePreview:
    try:
        with _restore_lock(request):
            return service.preview(target_sequence, _OPERATOR_ACTOR)
    except RestoreContractError as exc:
        raise _restore_error(exc) from exc


@router.post("/operator-restore/commit", response_model=RestoreCommitResponse)
def commit_operator_restore(
    body: RestoreCommitRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    service: OperatorRestoreService = Depends(get_operator_restore_service),
) -> RestoreCommitResponse:
    reserved_preview: OperatorRestorePreview | None = None
    try:
        reserved_preview = service.preflight(body, _OPERATOR_ACTOR)
        outcome = execute_agent_event(
            runtime,
            AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.operator_restore.commit",
            handler=lambda: service.apply_commit(
                get_main_loop(request), body, _OPERATOR_ACTOR
            ),
            event_id=event_id_for_operation(reserved_preview.operation_id),
            correlation_id=reserved_preview.preview_digest,
            payload={
                "operation_id": reserved_preview.operation_id,
                "target_sequence": reserved_preview.target_sequence,
                "journal_target": (
                    f"restore:{reserved_preview.target_sequence}:"
                    f"{reserved_preview.target_snapshot_hash}"
                ),
            },
        )
        with _restore_lock(request):
            response = service.build_completion_response(outcome.value.operation_id)
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": RestoreErrorCode.COMMIT_INDETERMINATE.value},
            )
        return response
    except RestoreContractError as exc:
        if reserved_preview is not None:
            service.release(reserved_preview.preview_digest)
        raise _restore_error(exc) from exc
    except HTTPException:
        if reserved_preview is not None:
            service.release(reserved_preview.preview_digest)
        raise
    except Exception:
        if reserved_preview is not None:
            service.release(reserved_preview.preview_digest)
        raise


@router.get("/working-memory", response_model=WorkingMemorySummaryResponse)
def working_memory_summary(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> WorkingMemorySummaryResponse:
    working_memory = get_main_loop(request).working_memory
    return execute_agent_event(
        runtime,
        AgentEventType.MEMORY_READ,
        source="api.state.working_memory",
        handler=lambda: WorkingMemorySummaryResponse(
            item_count=len(working_memory.items),
            token_count=working_memory.token_count,
            item_capacity=working_memory.item_capacity,
            token_capacity=working_memory.token_capacity,
        ),
    ).value


@router.post("/snapshot")
def create_snapshot() -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "private_state_projection_unavailable"},
    )


@router.get("/export")
def export_snapshot() -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "private_state_projection_unavailable"},
    )


@router.post("/restore")
def restore_snapshot() -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "operator_restore_contract_required"},
    )


@router.post("/reset")
def reset_snapshot() -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "operator_restore_contract_required"},
    )


@router.get("/reconstruct/{sequence}")
def reconstruct_snapshot(sequence: int) -> None:
    del sequence
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "raw_state_replay_disabled"},
    )


@router.post("/restore/{sequence}/dry-run")
def dry_run_restore(sequence: int) -> None:
    del sequence
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "raw_state_replay_disabled"},
    )


@router.get("/external-transactions")
def external_transaction_audit() -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "raw_state_replay_disabled"},
    )


@router.post("/external-transactions/reconcile")
def reconcile_external_transactions() -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "raw_state_replay_disabled"},
    )


@router.get("/restore/{sequence}/external-diff")
def external_restore_diff(
    sequence: int,
) -> None:
    del sequence
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "raw_state_replay_disabled"},
    )


@router.post("/restore/{sequence}")
def restore_point_in_time(
    sequence: int,
) -> None:
    del sequence
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": "raw_state_restore_disabled"},
    )


@router.post("/backups", response_model=RestorePreview)
def create_encrypted_backup(
    body: BackupCreateRequest,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    manager = BackupManager(settings)
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.BACKUP_CREATE,
            source="api.state.backup.create",
            handler=lambda: manager.create(base_backup_id=body.base_backup_id),
        ).value
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/backups", response_model=list[BackupStatus])
def list_encrypted_backups(
    limit: int = 50,
    settings: Settings = Depends(get_api_settings),
) -> list[BackupStatus]:
    try:
        return BackupManager(settings).list(limit)
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/backups/{backup_id}/preview", response_model=RestorePreview)
def preview_encrypted_backup(
    backup_id: str,
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    try:
        return BackupManager(settings).preview(backup_id)
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/verify", response_model=RestorePreview)
def verify_encrypted_backup(
    backup_id: str,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    manager = BackupManager(settings)
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.BACKUP_VERIFY,
            source="api.state.backup.verify",
            handler=lambda: manager.verify(backup_id),
        ).value
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/rotate", response_model=RestorePreview)
def rotate_encrypted_backup(
    backup_id: str,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    manager = BackupManager(settings)
    try:
        return execute_agent_event(
            runtime,
            AgentEventType.BACKUP_ROTATE,
            source="api.state.backup.rotate",
            handler=lambda: manager.rotate(backup_id),
        ).value
    except (BackupError, EncryptionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/restore", response_model=RestorePreview)
def commit_encrypted_restore(
    backup_id: str,
    body: BackupRestoreRequest,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_api_settings),
) -> RestorePreview:
    if settings.failover.enabled:
        raise HTTPException(
            status_code=409,
            detail="online backup restore is unavailable while subject failover is enabled",
        )
    if body.expected_backup_id != backup_id:
        raise HTTPException(status_code=409, detail="expected backup ID does not match")
    manager = BackupManager(settings)
    initialized_caches = {
        name
        for name in (
            "sleep_coordinator",
            "dataset_governance",
            "adapter_runtime_manager",
            "tool_registry",
            "tool_executor",
        )
        if getattr(request.app.state, name, None) is not None
    }
    # Serialize all accepted work before the offline-style authoritative swap.
    execute_agent_event(
        runtime,
        AgentEventType.BACKUP_RESTORE,
        source="api.state.backup.restore.prepare",
        handler=lambda: manager.verify(backup_id),
    )
    try:
        restored = manager.restore(
            backup_id,
            expected_manifest_hash=body.expected_manifest_hash,
            prepare=lambda _root: _teardown_subject_runtime(request),
            after_publish=lambda: _build_subject_runtime_offline(
                request, settings, initialized_caches
            ),
            activate=lambda: _activate_subject_runtime(request),
            after_rollback=lambda: _reset_build_and_activate_subject_runtime(
                request, settings, initialized_caches
            ),
        )
        return restored
    except Exception as exc:
        if isinstance(exc, (BackupError, EncryptionError)):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise


def _build_subject_runtime_offline(
    request: Request,
    settings: Settings,
    initialized_caches: set[str],
) -> None:
    from kagya.api.server import _build_subject_runtime

    _build_subject_runtime(
        request.app, settings, reconcile=False, allow_restore_marker=True
    )
    if "dataset_governance" in initialized_caches:
        get_dataset_governance(request)
    if "sleep_coordinator" in initialized_caches:
        get_sleep_coordinator(request)
    if "adapter_runtime_manager" in initialized_caches:
        get_adapter_runtime_manager(request)
    if "tool_registry" in initialized_caches:
        get_tool_registry(request)
    if "tool_executor" in initialized_caches:
        get_tool_executor(request)


def _activate_subject_runtime(request: Request) -> None:
    from kagya.api.server import _activate_subject_runtime as activate

    try:
        activate(request.app)
    except Exception:
        _teardown_subject_runtime(request)
        raise


def _teardown_subject_runtime(request: Request) -> None:
    from kagya.api.server import teardown_subject_runtime

    teardown_subject_runtime(request.app)


def _reset_build_and_activate_subject_runtime(
    request: Request, settings: Settings, initialized_caches: set[str]
) -> None:
    _teardown_subject_runtime(request)
    _build_subject_runtime_offline(request, settings, initialized_caches)
    _activate_subject_runtime(request)
