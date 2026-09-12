"""Small boundary between HTTP handlers and the serialized agent runtime."""

from collections.abc import Callable
from typing import TypeVar, overload

from fastapi import HTTPException, status

from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    CoordinatedResult,
)


T = TypeVar("T")


@overload
def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], CoordinatedResult[T]],
) -> T: ...


@overload
def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], T],
) -> T: ...


def execute(
    runtime: AgentRuntime,
    event_type: AgentEventType,
    source: AgentEventSource,
    handler: Callable[[], object],
) -> object:
    """Submit one request handler, translating only runtime-boundary failures."""

    try:
        return runtime.submit(event_type, source, handler).result().value
    except (
        AgentRuntimeAdmissionBlocked,
        AgentRuntimeQueueFull,
        AgentRuntimeStopped,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime is temporarily unavailable",
        ) from exc
    except AgentRuntimeDurabilityError as exc:
        if exc.outcome_indeterminate:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent mutation durability is indeterminate",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime durability is temporarily unavailable",
        ) from exc
    except AgentRuntimeExecutionError as exc:
        if exc.__cause__ is not None:
            raise exc.__cause__
        raise
