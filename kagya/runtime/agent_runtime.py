"""Process-local, bounded agent event runtime."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from threading import Condition, Thread, current_thread
from typing import Any, Callable, Generic, TypeVar, cast
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
    FAILED = "failed"


class AgentRuntimeDurabilityPhase(str, Enum):
    ADMISSION = "admission"
    STARTED = "started"
    HANDLER_FAILURE = "handler_failure"
    INTERNAL_COMMIT = "internal_commit"
    FINALIZATION = "finalization"
    TERMINAL_COMPLETION = "terminal_completion"


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
    """A handler failed before internal commit; the consumer remains available."""


class AgentRuntimeDurabilityError(_AgentRuntimeEventError):
    """A lifecycle checkpoint failed and the runtime entered fail-stop mode.

    ``outcome_indeterminate`` describes the overall operation response boundary; the
    precise durability boundary is represented by ``phase``.
    """

    def __init__(
        self,
        event: AgentEvent,
        phase: AgentRuntimeDurabilityPhase,
        *,
        outcome_indeterminate: bool,
        failure_type: str | None = None,
        published: bool | None = None,
    ) -> None:
        super().__init__(event)
        self.phase = phase
        self.outcome_indeterminate = outcome_indeterminate
        self.failure_type = failure_type
        self.published = published


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
        admission_checkpoint: Callable[[AgentEvent], None] | None = None,
        started_checkpoint: Callable[[AgentEvent], None] | None = None,
        internal_commit_checkpoint: Callable[[AgentEvent], None] | None = None,
        finalization_checkpoint: Callable[[AgentEvent], None] | None = None,
        terminal_completion_checkpoint: Callable[[AgentEvent], None] | None = None,
        failure_checkpoint: Callable[[AgentEvent], None] | None = None,
        allow_volatile: bool = False,
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
        self._admission_checkpoint = admission_checkpoint
        self._started_checkpoint = started_checkpoint
        self._internal_commit_checkpoint = internal_commit_checkpoint
        self._finalization_checkpoint = finalization_checkpoint
        self._terminal_completion_checkpoint = terminal_completion_checkpoint
        self._failure_checkpoint = failure_checkpoint
        self._condition = Condition()
        self._pending: deque[_PendingEvent] = deque()
        self._status = AgentRuntimeStatus.CREATED
        self._worker: Thread | None = None
        self._sequence = initial_sequence
        self._active: tuple[_PendingEvent, AgentEvent] | None = None
        self._failed_phase: AgentRuntimeDurabilityPhase | None = None
        self._allow_volatile = allow_volatile
        self._durability_configured = all(
            callback is not None
            for callback in (
                admission_checkpoint,
                started_checkpoint,
                internal_commit_checkpoint,
                finalization_checkpoint,
                terminal_completion_checkpoint,
                failure_checkpoint,
            )
        )

    @property
    def status(self) -> AgentRuntimeStatus:
        with self._condition:
            return self._status

    def start(self) -> None:
        with self._condition:
            if self._status in {
                AgentRuntimeStatus.DRAINING,
                AgentRuntimeStatus.STOPPED,
                AgentRuntimeStatus.FAILED,
            }:
                raise RuntimeError("AgentRuntime cannot be restarted")
            if self._status is AgentRuntimeStatus.ACCEPTING:
                return
            if not self._durability_configured and not self._allow_volatile:
                raise RuntimeError(
                    "AgentRuntime durability lifecycle is not configured"
                )
            self._status = AgentRuntimeStatus.ACCEPTING
            self._worker = Thread(
                target=self._consume,
                name="kagya-agent-runtime",
                daemon=True,
            )
            self._worker.start()

    def configure_durability(
        self,
        *,
        initial_sequence: int,
        admission_checkpoint: Callable[[AgentEvent], None],
        started_checkpoint: Callable[[AgentEvent], None],
        internal_commit_checkpoint: Callable[[AgentEvent], None],
        finalization_checkpoint: Callable[[AgentEvent], None],
        terminal_completion_checkpoint: Callable[[AgentEvent], None],
        failure_checkpoint: Callable[[AgentEvent], None],
    ) -> None:
        """Bind mandatory durable lifecycle collaborators before startup."""

        if (
            isinstance(initial_sequence, bool)
            or not isinstance(initial_sequence, int)
            or initial_sequence < 0
        ):
            raise ValueError("initial_sequence must be a non-negative integer")
        with self._condition:
            if self._status is not AgentRuntimeStatus.CREATED:
                raise RuntimeError(
                    "AgentRuntime durability must be configured before start"
                )
            self._sequence = initial_sequence
            self._admission_checkpoint = admission_checkpoint
            self._started_checkpoint = started_checkpoint
            self._internal_commit_checkpoint = internal_commit_checkpoint
            self._finalization_checkpoint = finalization_checkpoint
            self._terminal_completion_checkpoint = terminal_completion_checkpoint
            self._failure_checkpoint = failure_checkpoint
            self._durability_configured = True

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
            if self._admission_checkpoint is not None:
                durability_error: AgentRuntimeDurabilityError | None = None
                try:
                    self._admission_checkpoint(event)
                except Exception as error:
                    durability_error = self._durability_error(
                        event, AgentRuntimeDurabilityPhase.ADMISSION, error, False
                    )
                if durability_error is not None:
                    self._fail_stop_locked(phase=AgentRuntimeDurabilityPhase.ADMISSION)
                    raise durability_error
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
                    if self._status is AgentRuntimeStatus.FAILED:
                        self._condition.notify_all()
                        return
                    self._status = AgentRuntimeStatus.STOPPED
                    self._condition.notify_all()
                    return
                pending = self._pending.popleft()
                self._sequence += 1
                event = replace(pending.event, processing_sequence=self._sequence)
                self._active = (pending, event)
            if self._started_checkpoint is not None:
                try:
                    self._started_checkpoint(event)
                except Exception as error:
                    durability_error = self._durability_error(
                        event, AgentRuntimeDurabilityPhase.STARTED, error, False
                    )
                    with self._condition:
                        self._fail_stop_locked(durability_error)
                    return
            with self._condition:
                if self._status is AgentRuntimeStatus.FAILED:
                    self._set_exception(
                        pending.future,
                        AgentRuntimeDurabilityError(
                            event,
                            self._failed_phase or AgentRuntimeDurabilityPhase.ADMISSION,
                            outcome_indeterminate=False,
                        ),
                    )
                    self._finish_active_locked()
                    return
            try:
                value = pending.handler()
            except Exception as error:
                if self._failure_checkpoint is not None:
                    try:
                        self._failure_checkpoint(event)
                    except Exception as checkpoint_error:
                        durability_error = self._durability_error(
                            event,
                            AgentRuntimeDurabilityPhase.HANDLER_FAILURE,
                            checkpoint_error,
                            True,
                        )
                        with self._condition:
                            self._fail_stop_locked(durability_error)
                        return
                wrapped = AgentRuntimeExecutionError(event)
                wrapped.__cause__ = error
                self._set_exception(pending.future, wrapped)
                self._finish_active()
            else:
                checkpoints = (
                    (
                        AgentRuntimeDurabilityPhase.INTERNAL_COMMIT,
                        self._internal_commit_checkpoint,
                    ),
                    (
                        AgentRuntimeDurabilityPhase.FINALIZATION,
                        self._finalization_checkpoint,
                    ),
                    (
                        AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION,
                        self._terminal_completion_checkpoint,
                    ),
                )
                for phase, checkpoint in checkpoints:
                    try:
                        if checkpoint is not None:
                            checkpoint(event)
                    except Exception as error:
                        durability_error = self._durability_error(
                            event, phase, error, True
                        )
                        with self._condition:
                            self._fail_stop_locked(durability_error)
                        return
                self._set_result(pending.future, AgentEventOutcome(event, value))
                self._finish_active()

    @staticmethod
    def _durability_error(
        event: AgentEvent,
        phase: AgentRuntimeDurabilityPhase,
        error: Exception,
        outcome_indeterminate: bool,
    ) -> AgentRuntimeDurabilityError:
        raw_failure_type = type(error).__name__
        failure_type = raw_failure_type if raw_failure_type.isidentifier() else None
        published = getattr(error, "published", None)
        if not isinstance(published, bool):
            published = None
        return AgentRuntimeDurabilityError(
            event,
            phase,
            outcome_indeterminate=outcome_indeterminate,
            failure_type=failure_type,
            published=published,
        )

    def _fail_stop_locked(
        self,
        current_error: AgentRuntimeDurabilityError | None = None,
        *,
        phase: AgentRuntimeDurabilityPhase | None = None,
    ) -> None:
        self._status = AgentRuntimeStatus.FAILED
        failure_phase = phase or (
            current_error.phase
            if current_error is not None
            else AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION
        )
        self._failed_phase = failure_phase
        if self._active is not None and current_error is not None:
            pending, _event = self._active
            self._set_exception(pending.future, current_error)
        while self._pending:
            pending = self._pending.popleft()
            self._set_exception(
                pending.future,
                AgentRuntimeDurabilityError(
                    pending.event,
                    failure_phase,
                    outcome_indeterminate=False,
                ),
            )
        self._condition.notify_all()

    def _finish_active_locked(self) -> None:
        self._active = None
        self._condition.notify_all()

    def _finish_active(self) -> None:
        with self._condition:
            self._finish_active_locked()

    @staticmethod
    def _set_exception(future: Future[Any], error: BaseException) -> None:
        try:
            future.set_exception(error)
        except InvalidStateError:
            pass

    @staticmethod
    def _set_result(future: Future[Any], result: object) -> None:
        try:
            future.set_result(result)
        except InvalidStateError:
            pass
