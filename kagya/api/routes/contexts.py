"""Administrative context lifecycle and relation routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_agent_runtime, get_main_loop, require_admin
from kagya.api.runtime_execution import execute
from kagya.api.schemas.context import (
    ContextFrameResponse,
    ContextListResponse,
    ContextRelationRequest,
    ContextRelationResponse,
)
from kagya.identifiers import validate_identifier
from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    ContextFrame,
    ContextNotFound,
    ContextRegistry,
    KagyaMainLoop,
)


router = APIRouter(
    prefix="/api/contexts", tags=["contexts"], dependencies=[Depends(require_admin)]
)


def _registry(main_loop: KagyaMainLoop) -> ContextRegistry:
    return main_loop.context_registry


def _context_id(value: str) -> str:
    try:
        return validate_identifier(value)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=[{"type": "value_error", "loc": ["path"], "msg": "Invalid request"}],
        ) from None


def _frame(frame: ContextFrame, current_context_id: str | None) -> ContextFrameResponse:
    return ContextFrameResponse(
        context_id=frame.context_id,
        context_type=frame.context_type.value,
        source_channel=frame.source_channel,
        source_session_id=frame.source_session_id,
        participant_refs=frame.participant_refs,
        parent_context_id=frame.parent_context_id,
        related_context_ids=frame.related_context_ids,
        status=frame.status.value,
        created_revision=frame.created_revision,
        last_modified_revision=frame.last_modified_revision,
        started_at=frame.started_at,
        last_active_at=frame.last_active_at,
        is_current=frame.context_id == current_context_id,
    )


@router.get("", response_model=ContextListResponse)
def list_contexts(
    main_loop: KagyaMainLoop = Depends(get_main_loop),
) -> ContextListResponse:
    state = _registry(main_loop).state
    return ContextListResponse(
        current_context_id=state.current_context_id,
        contexts=tuple(
            _frame(frame, state.current_context_id) for frame in state.frames
        ),
    )


@router.get("/{context_id}", response_model=ContextFrameResponse)
def get_context(
    context_id: str, main_loop: KagyaMainLoop = Depends(get_main_loop)
) -> ContextFrameResponse:
    checked = _context_id(context_id)
    registry = _registry(main_loop)
    try:
        frame = registry.get(checked)
    except ContextNotFound as exc:
        raise HTTPException(status_code=404, detail="Context not found") from exc
    return _frame(frame, registry.current_context_id)


def _transition(
    context_id: str,
    main_loop: KagyaMainLoop,
    runtime: AgentRuntime,
    source: AgentEventSource,
    operation: str,
) -> ContextFrameResponse:
    checked = _context_id(context_id)
    registry = _registry(main_loop)
    handler = getattr(registry, operation)
    frame = execute(
        runtime, AgentEventType.CONTEXT_UPDATE, source, lambda: handler(checked)
    )
    return _frame(frame, registry.current_context_id)


@router.post("/{context_id}/suspend", response_model=ContextFrameResponse)
def suspend_context(
    context_id: str,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextFrameResponse:
    return _transition(
        context_id, main_loop, runtime, AgentEventSource.API_CONTEXT_SUSPEND, "suspend"
    )


@router.post("/{context_id}/resume", response_model=ContextFrameResponse)
def resume_context(
    context_id: str,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextFrameResponse:
    return _transition(
        context_id, main_loop, runtime, AgentEventSource.API_CONTEXT_RESUME, "resume"
    )


@router.post("/{context_id}/close", response_model=ContextFrameResponse)
def close_context(
    context_id: str,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextFrameResponse:
    return _transition(
        context_id, main_loop, runtime, AgentEventSource.API_CONTEXT_CLOSE, "close"
    )


@router.post("/{context_id}/relations", response_model=ContextRelationResponse)
def relate_context(
    context_id: str,
    request: ContextRelationRequest,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ContextRelationResponse:
    checked = _context_id(context_id)
    registry = _registry(main_loop)

    def relate_and_read() -> tuple[ContextFrame, ContextFrame]:
        registry.relate(checked, request.related_context_id)
        return registry.get(checked), registry.get(request.related_context_id)

    left, right = execute(
        runtime,
        AgentEventType.CONTEXT_UPDATE,
        AgentEventSource.API_CONTEXT_RELATE,
        relate_and_read,
    )
    current = registry.current_context_id
    ordered = tuple(sorted((left, right), key=lambda frame: frame.context_id))
    return ContextRelationResponse(
        contexts=tuple(_frame(frame, current) for frame in ordered)
    )
