"""Small boundary between HTTP handlers and the serialized agent runtime."""

from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException, status

from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    AgentStateSaveError,
)


T = TypeVar("T")


def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], T],
) -> T:
    """Submit one request handler, translating only runtime-boundary failures."""

    try:
        return runtime.submit(event_type, source, handler).result().value
    except (AgentRuntimeQueueFull, AgentRuntimeStopped) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime is temporarily unavailable",
        ) from exc
    except AgentRuntimeExecutionError as exc:
        if isinstance(exc.__cause__, AgentStateSaveError):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent state checkpoint could not be saved; outcome is indeterminate",
            ) from exc
        if exc.__cause__ is not None:
            raise exc.__cause__
        raise
