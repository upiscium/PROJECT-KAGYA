"""Process-local, bounded agent event runtime."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from threading import Condition, Thread, current_thread
from typing import Callable, Generic, TypeVar, cast
from uuid import uuid4


class AgentEventType(str, Enum):
    CHAT = "chat"
    DEBUG_CHAT = "debug_chat"
    SLEEP = "sleep"
    ADAPTER_EVALUATE = "adapter_evaluate"
    ADAPTER_UPDATE = "adapter_update"


class AgentEventSource(str, Enum):
    """Allowlisted origins that cannot become arbitrary request metadata."""

    API_CHAT = "api.chat"
    API_CHAT_DEBUG = "api.chat.debug"
    API_SLEEP_RUN = "api.sleep.run"
    API_ADAPTER_EVALUATE = "api.adapters.evaluate"
    API_ADAPTER_TRIAL = "api.adapters.trial"
    API_ADAPTER_APPROVE = "api.adapters.approve"
    API_ADAPTER_ACTIVATE = "api.adapters.activate"
    API_ADAPTER_REJECT = "api.adapters.reject"


class AgentRuntimeStatus(str, Enum):
    CREATED = "created"
    ACCEPTING = "accepting"
    DRAINING = "draining"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_id: str
    event_type: AgentEventType
    source: AgentEventSource
    requested_at: datetime
    processing_sequence: int | None = None


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AgentEventOutcome(Generic[T]):
    event: AgentEvent
    value: T


class _AgentRuntimeEventError(RuntimeError):
    """Base for admission and execution errors that expose only event metadata."""

    def __init__(self, event: AgentEvent) -> None:
        super().__init__("Agent event could not be processed")
        self.event = event


class AgentRuntimeQueueFull(_AgentRuntimeEventError):
    """The bounded pending queue has no admission capacity."""


class AgentRuntimeStopped(_AgentRuntimeEventError):
    """The runtime is not accepting new events."""


class AgentRuntimeExecutionError(_AgentRuntimeEventError):
    """A handler or completion checkpoint failed; the consumer remains available."""


@dataclass(slots=True)
class _PendingEvent:
    event: AgentEvent
    handler: Callable[[], object]
    future: Future[AgentEventOutcome[object]]


class AgentRuntime:
    """Execute accepted handlers in FIFO order on one process-local thread."""

    def __init__(
        self,
        queue_capacity: int,
        *,
        initial_sequence: int = 0,
        completion_checkpoint: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be greater than zero")
        if (
            isinstance(initial_sequence, bool)
            or not isinstance(initial_sequence, int)
            or initial_sequence < 0
        ):
            raise ValueError("initial_sequence must be a non-negative integer")
        self._queue_capacity = queue_capacity
        self._completion_checkpoint = completion_checkpoint
        self._condition = Condition()
        self._pending: deque[_PendingEvent] = deque()
        self._status = AgentRuntimeStatus.CREATED
        self._worker: Thread | None = None
        self._sequence = initial_sequence

    @property
    def status(self) -> AgentRuntimeStatus:
        with self._condition:
            return self._status

    def start(self) -> None:
        with self._condition:
            if self._status in {
                AgentRuntimeStatus.DRAINING,
                AgentRuntimeStatus.STOPPED,
            }:
                raise RuntimeError("AgentRuntime cannot be restarted")
            if self._status is AgentRuntimeStatus.ACCEPTING:
                return
            self._status = AgentRuntimeStatus.ACCEPTING
            self._worker = Thread(
                target=self._consume,
                name="kagya-agent-runtime",
                daemon=True,
            )
            self._worker.start()

    def submit(
        self,
        event_type: AgentEventType,
        source: AgentEventSource,
        handler: Callable[[], T],
    ) -> Future[AgentEventOutcome[T]]:
        if not isinstance(event_type, AgentEventType):
            raise TypeError("event_type must be an AgentEventType")
        if not isinstance(source, AgentEventSource):
            raise TypeError("source must be an AgentEventSource")
        event = AgentEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            source=source,
            requested_at=datetime.now(timezone.utc),
        )
        future: Future[AgentEventOutcome[T]] = Future()
        pending = _PendingEvent(
            event,
            cast(Callable[[], object], handler),
            cast(Future[AgentEventOutcome[object]], future),
        )
        with self._condition:
            if self._status is not AgentRuntimeStatus.ACCEPTING:
                raise AgentRuntimeStopped(event)
            if len(self._pending) >= self._queue_capacity:
                raise AgentRuntimeQueueFull(event)
            self._pending.append(pending)
            self._condition.notify()
        return future

    def shutdown(self) -> None:
        """Stop admission and wait until every accepted event has been handled."""
        with self._condition:
            if self._status is AgentRuntimeStatus.CREATED:
                self._status = AgentRuntimeStatus.STOPPED
                return
            if self._status is AgentRuntimeStatus.ACCEPTING:
                self._status = AgentRuntimeStatus.DRAINING
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker is not current_thread():
            worker.join()

    def _consume(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._pending and self._status is AgentRuntimeStatus.ACCEPTING
                ):
                    self._condition.wait()
                if not self._pending:
                    self._status = AgentRuntimeStatus.STOPPED
                    self._condition.notify_all()
                    return
                pending = self._pending.popleft()
                self._sequence += 1
                event = replace(pending.event, processing_sequence=self._sequence)
            try:
                value = pending.handler()
            except Exception as error:
                wrapped = AgentRuntimeExecutionError(event)
                wrapped.__cause__ = error
                try:
                    pending.future.set_exception(wrapped)
                except InvalidStateError:
                    pass
            else:
                try:
                    if self._completion_checkpoint is not None:
                        self._completion_checkpoint(event)
                except Exception as error:
                    wrapped = AgentRuntimeExecutionError(event)
                    wrapped.__cause__ = error
                    try:
                        pending.future.set_exception(wrapped)
                    except InvalidStateError:
                        pass
                else:
                    try:
                        pending.future.set_result(AgentEventOutcome(event, value))
                    except InvalidStateError:
                        pass
