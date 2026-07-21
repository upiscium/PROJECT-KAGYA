"""Administrative agent-state snapshot operations."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_agent_state_store,
    get_api_settings,
    get_main_loop,
    get_state_wal,
    require_admin,
)
from kagya.config import Settings
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentStateSnapshot,
    AgentStateStore,
    StateDryRun,
    StateReconstruction,
    StateWAL,
    StateWalIntegrityError,
)
from kagya.runtime.agent_state import default_agent_state_snapshot


router = APIRouter(
    prefix="/api/state", tags=["state"], dependencies=[Depends(require_admin)]
)


@router.post("/snapshot", response_model=AgentStateSnapshot)
def create_snapshot(
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
) -> AgentStateSnapshot:
    execute_agent_event(
        runtime,
        AgentEventType.STATE_SNAPSHOT,
        source="api.state.snapshot",
        handler=lambda: None,
    )
    return _current_snapshot(store)


@router.get("/export", response_model=AgentStateSnapshot)
def export_snapshot(
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
) -> AgentStateSnapshot:
    execute_agent_event(
        runtime,
        AgentEventType.STATE_EXPORT,
        source="api.state.export",
        handler=lambda: None,
    )
    return _current_snapshot(store)


@router.post("/restore", response_model=AgentStateSnapshot)
def restore_snapshot(
    snapshot: AgentStateSnapshot,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
) -> AgentStateSnapshot:
    execute_agent_event(
        runtime,
        AgentEventType.STATE_RESTORE,
        source="api.state.restore",
        handler=lambda: store.restore_into(get_main_loop(request), snapshot),
    )
    return _current_snapshot(store)


@router.post("/reset", response_model=AgentStateSnapshot)
def reset_snapshot(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
    settings: Settings = Depends(get_api_settings),
) -> AgentStateSnapshot:
    def reset() -> None:
        main_loop = get_main_loop(request)
        store.restore_into(
            main_loop,
            default_agent_state_snapshot(settings.emotion.baseline_surprisal),
        )
        main_loop.session_state.turns.clear()
        main_loop.working_memory.clear()
        main_loop.context_registry.clear()

    execute_agent_event(
        runtime,
        AgentEventType.STATE_RESET,
        source="api.state.reset",
        handler=reset,
    )
    return _current_snapshot(store)


@router.get("/reconstruct/{sequence}", response_model=StateReconstruction)
def reconstruct_snapshot(
    sequence: int,
    wal: StateWAL = Depends(get_state_wal),
) -> StateReconstruction:
    try:
        return wal.reconstruct(sequence)
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/restore/{sequence}/dry-run", response_model=StateDryRun)
def dry_run_restore(
    sequence: int,
    store: AgentStateStore = Depends(get_agent_state_store),
    wal: StateWAL = Depends(get_state_wal),
) -> StateDryRun:
    try:
        return wal.dry_run(_current_snapshot(store), sequence)
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.post("/restore/{sequence}", response_model=AgentStateSnapshot)
def restore_point_in_time(
    sequence: int,
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    store: AgentStateStore = Depends(get_agent_state_store),
    wal: StateWAL = Depends(get_state_wal),
) -> AgentStateSnapshot:
    def restore() -> None:
        target = wal.reconstruct(sequence)
        store.restore_into(get_main_loop(request), target.snapshot)

    try:
        execute_agent_event(
            runtime,
            AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.point_in_time_restore",
            handler=restore,
        )
    except StateWalIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return _current_snapshot(store)


def _current_snapshot(store: AgentStateStore) -> AgentStateSnapshot:
    if store.last_snapshot is None:
        raise RuntimeError("Agent state snapshot is unavailable")
    return store.last_snapshot
