"""Administrative agent-state snapshot operations."""

from fastapi import APIRouter, Depends, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_agent_state_store,
    get_api_settings,
    get_main_loop,
    require_admin,
)
from kagya.config import Settings
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentStateSnapshot,
    AgentStateStore,
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


def _current_snapshot(store: AgentStateStore) -> AgentStateSnapshot:
    if store.last_snapshot is None:
        raise RuntimeError("Agent state snapshot is unavailable")
    return store.last_snapshot
