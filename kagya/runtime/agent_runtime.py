"""Single-consumer runtime for ordered agent events."""

from __future__ import annotations

from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from queue import Full, Queue
from threading import Lock, Thread
from typing import Any, Callable, Generic, Protocol, TypeVar
from uuid import uuid4


T = TypeVar("T")
_STOP = object()
_current_event: ContextVar[AgentEvent | None] = ContextVar("kagya_agent_event", default=None)


class AgentEventType(StrEnum):
    CHAT = "chat"
    DEBUG_CHAT = "debug_chat"
    SLEEP = "sleep"
    MEMORY_READ = "memory_read"
    MEMORY_UPDATE = "memory_update"
    ADAPTER_READ = "adapter_read"
    ADAPTER_UPDATE = "adapter_update"
    STATE_SNAPSHOT = "state_snapshot"
    STATE_EXPORT = "state_export"
    STATE_RESTORE = "state_restore"
    STATE_RESET = "state_reset"
    CONTEXT_UPDATE = "context_update"


@dataclass(frozen=True)
class AgentEvent:
    event_id: str
    event_type: AgentEventType
    source: str
    observed_at: datetime
    requested_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    processing_sequence: int | None = None
    causation_id: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True)
class AgentEventOutcome(Generic[T]):
    event: AgentEvent
    value: T


@dataclass(frozen=True)
class _Envelope(Generic[T]):
    event: AgentEvent
    handler: Callable[[], T]
    future: Future[AgentEventOutcome[T]]


class EventRecorder(Protocol):
    def record(
        self,
        *,
        category: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> object: ...


class EventCompletionHook(Protocol):
    def __call__(self, event: AgentEvent) -> None: ...


class EventFailureHook(Protocol):
    def __call__(self, event: AgentEvent, exception: Exception) -> None: ...


class AgentRuntimeQueueFull(RuntimeError):
    """Raised when an event cannot be accepted without blocking."""


class AgentRuntimeStopped(RuntimeError):
    """Raised when an event is submitted after draining starts."""


class AgentRuntime:
    """Execute accepted agent events in one process-local causal order."""

    def __init__(
        self,
        *,
        queue_capacity: int,
        event_recorder: EventRecorder | None = None,
        initial_sequence: int = 0,
        completion_hook: EventCompletionHook | None = None,
        failure_hook: EventFailureHook | None = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be greater than zero")
        if initial_sequence < 0:
            raise ValueError("initial_sequence must not be negative")
        self._queue: Queue[_Envelope[Any] | object] = Queue(maxsize=queue_capacity)
        self._event_recorder = event_recorder
        self._completion_hook = completion_hook
        self._failure_hook = failure_hook
        self._state_lock = Lock()
        self._state = "created"
        self._sequence = initial_sequence
        self._worker: Thread | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._state == "accepting":
                return
            if self._state != "created":
                raise AgentRuntimeStopped("Agent runtime cannot be restarted")
            self._state = "accepting"
            self._worker = Thread(
                target=self._consume,
                name="kagya-agent-runtime",
                daemon=True,
            )
            self._worker.start()

    def submit(
        self,
        event_type: AgentEventType,
        *,
        source: str,
        handler: Callable[[], T],
        payload: dict[str, Any] | None = None,
        requested_at: datetime | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> Future[AgentEventOutcome[T]]:
        now = datetime.now(UTC)
        event = AgentEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            source=source,
            observed_at=now,
            requested_at=requested_at or now,
            payload=dict(payload or {}),
            causation_id=causation_id,
            correlation_id=correlation_id,
        )
        future: Future[AgentEventOutcome[T]] = Future()
        envelope = _Envelope(event=event, handler=handler, future=future)
        with self._state_lock:
            if self._state != "accepting":
                raise AgentRuntimeStopped("Agent runtime is draining or stopped")
            try:
                self._queue.put_nowait(envelope)
            except Full as exc:
                raise AgentRuntimeQueueFull("Agent event queue is full") from exc
        return future

    def execute(
        self,
        event_type: AgentEventType,
        *,
        source: str,
        handler: Callable[[], T],
        payload: dict[str, Any] | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> AgentEventOutcome[T]:
        return self.submit(
            event_type,
            source=source,
            handler=handler,
            payload=payload,
            causation_id=causation_id,
            correlation_id=correlation_id,
        ).result()

    def shutdown(self) -> None:
        with self._state_lock:
            if self._state == "stopped":
                return
            if self._state == "created":
                self._state = "stopped"
                return
            owns_shutdown = self._state == "accepting"
            self._state = "draining"
            worker = self._worker
        if not owns_shutdown:
            if worker is not None:
                worker.join()
            return
        self._queue.join()
        self._queue.put(_STOP)
        if worker is not None:
            worker.join()
        with self._state_lock:
            self._state = "stopped"

    @property
    def is_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def is_accepting(self) -> bool:
        with self._state_lock:
            return self._state == "accepting"

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                envelope = item
                self._sequence += 1
                event = replace(
                    envelope.event, processing_sequence=self._sequence
                )
                can_deliver = envelope.future.set_running_or_notify_cancel()
                self._record(event, "started")
                event_token = _current_event.set(event)
                try:
                    value = envelope.handler()
                except Exception as exc:
                    if self._failure_hook is not None:
                        try:
                            self._failure_hook(event, exc)
                        except Exception:
                            pass
                    self._record(event, "failed", exception_type=type(exc).__name__)
                    if can_deliver:
                        envelope.future.set_exception(exc)
                else:
                    try:
                        if self._completion_hook is not None:
                            self._completion_hook(event)
                    except Exception as exc:
                        self._record(
                            event, "failed", exception_type=type(exc).__name__
                        )
                        if can_deliver:
                            envelope.future.set_exception(exc)
                    else:
                        self._record(event, "completed")
                        if can_deliver:
                            envelope.future.set_result(
                                AgentEventOutcome(event=event, value=value)
                            )
                finally:
                    _current_event.reset(event_token)
            finally:
                self._queue.task_done()

    def _record(
        self,
        event: AgentEvent,
        status: str,
        *,
        exception_type: str | None = None,
    ) -> None:
        if self._event_recorder is None:
            return
        metadata = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "source": event.source,
            "processing_sequence": event.processing_sequence,
            "causation_id": event.causation_id,
            "correlation_id": event.correlation_id,
        }
        if exception_type is not None:
            metadata["exception_type"] = exception_type
        try:
            self._event_recorder.record(
                category="agent",
                event_type=status,
                message=f"Agent event {status}",
                metadata=metadata,
            )
        except Exception:
            # Observability must never terminate the subject's event consumer.
            return


def current_agent_event() -> AgentEvent | None:
    return _current_event.get()
