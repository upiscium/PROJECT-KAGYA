"""Single-consumer runtime for ordered agent events."""

from __future__ import annotations

from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, Callable, cast, Generic, Protocol, TypeVar
from uuid import uuid4


T = TypeVar("T")
_STOP = object()
_current_event: ContextVar[AgentEvent | None] = ContextVar(
    "kagya_agent_event", default=None
)
_rollback_callbacks: ContextVar[tuple[Callable[[], None], ...]] = ContextVar(
    "kagya_event_rollback_callbacks", default=()
)


def register_event_rollback(callback: Callable[[], None]) -> None:
    if _current_event.get() is None:
        raise RuntimeError("rollback callback requires an AgentRuntime event")
    _rollback_callbacks.set((*_rollback_callbacks.get(), callback))


class AgentEventType(StrEnum):
    CHAT = "chat"
    DEBUG_CHAT = "debug_chat"
    SLEEP = "sleep"
    MEMORY_READ = "memory_read"
    MEMORY_UPDATE = "memory_update"
    ADAPTER_READ = "adapter_read"
    ADAPTER_UPDATE = "adapter_update"
    BEHAVIORAL_EVALUATE = "behavioral_evaluate"
    STATE_SNAPSHOT = "state_snapshot"
    STATE_EXPORT = "state_export"
    STATE_RESTORE = "state_restore"
    STATE_POINT_IN_TIME_RESTORE = "state_point_in_time_restore"
    STATE_RESET = "state_reset"
    CONTEXT_UPDATE = "context_update"
    EMOTION_TICK = "emotion_tick"
    VALUE_READ = "value_read"
    VALUE_UPDATE = "value_update"
    GOAL_READ = "goal_read"
    GOAL_UPDATE = "goal_update"
    GOAL_REEVALUATE = "goal_reevaluate"
    PLAN_READ = "plan_read"
    PLAN_UPDATE = "plan_update"
    PLAN_REPLAN = "plan_replan"
    STEP_UPDATE = "step_update"
    DECISION_READ = "decision_read"
    DECISION_UPDATE = "decision_update"
    DECISION_GENERATE = "decision_generate"
    DECISION_EXPLANATION_READ = "decision_explanation_read"
    DECISION_EXPLANATION_CREATE = "decision_explanation_create"
    DECISION_EXPLANATION_REVISE = "decision_explanation_revise"
    DECISION_EXPLANATION_RENDER = "decision_explanation_render"
    SELF_MODEL_READ = "self_model_read"
    SELF_MODEL_UPDATE = "self_model_update"
    EXPERIENCE_READ = "experience_read"
    EXPERIENCE_UPDATE = "experience_update"
    BELIEF_READ = "belief_read"
    BELIEF_UPDATE = "belief_update"
    MOTIVATION_READ = "motivation_read"
    MOTIVATION_UPDATE = "motivation_update"
    MOTIVATION_REEVALUATE = "motivation_reevaluate"
    ATTENTION_READ = "attention_read"
    ATTENTION_UPDATE = "attention_update"
    ATTENTION_COMPETE = "attention_compete"
    AUTONOMY_SCHEDULE = "autonomy_schedule"
    AUTONOMY_WAKE = "autonomy_wake"
    FEEDBACK_READ = "feedback_read"
    FEEDBACK_UPDATE = "feedback_update"
    RELATIONSHIP_READ = "relationship_read"
    RELATIONSHIP_UPDATE = "relationship_update"
    INTRINSIC_GOAL_PROPOSE = "intrinsic_goal_propose"
    INTRINSIC_GOAL_DELIBERATE = "intrinsic_goal_deliberate"
    INTRINSIC_GOAL_ADOPT = "intrinsic_goal_adopt"
    PLAN_GENERATE = "plan_generate"
    ACTION_READ = "action_read"
    ACTION_INTENT = "action_intent"
    ACTION_APPROVAL = "action_approval"
    ACTION_EXECUTE = "action_execute"
    ACTION_CANCEL = "action_cancel"
    ACTION_COMPENSATE = "action_compensate"
    ATTRIBUTION_READ = "attribution_read"
    ATTRIBUTION_APPLY = "attribution_apply"
    ATTRIBUTION_REVISE = "attribution_revise"
    COUNTERFACTUAL_READ = "counterfactual_read"
    COUNTERFACTUAL_APPLY = "counterfactual_apply"
    COUNTERFACTUAL_REVISE = "counterfactual_revise"
    OUTBOX_READ = "outbox_read"
    OUTBOX_ENQUEUE = "outbox_enqueue"
    OUTBOX_DELIVER = "outbox_deliver"
    OUTBOX_RESPONSE = "outbox_response"
    OUTBOX_FAILURE = "outbox_failure"


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
    def __call__(self, event: AgentEvent) -> str | None: ...


class EventFailureHook(Protocol):
    def __call__(self, event: AgentEvent, exception: Exception) -> str | None: ...


class DurableEventJournal(Protocol):
    def accepted(self, event: AgentEvent) -> object: ...

    def started(self, event: AgentEvent) -> object: ...

    def completed(self, event: AgentEvent, snapshot_hash: str) -> object: ...

    def failed(
        self,
        event: AgentEvent,
        failure_category: str,
        snapshot_hash: str | None,
    ) -> object: ...


class RuntimeTelemetry(Protocol):
    def event_accepted(self, event: AgentEvent, queue_depth: int) -> None: ...

    def event_started(self, event: AgentEvent, queue_depth: int) -> None: ...

    def event_finished(
        self, event: AgentEvent, status: str, queue_depth: int
    ) -> None: ...


class AgentRuntimeQueueFull(RuntimeError):
    """Raised when an event cannot be accepted without blocking."""


class AgentRuntimeStopped(RuntimeError):
    """Raised when an event is submitted after draining starts."""


class AgentRuntimeJournalError(RuntimeError):
    """Raised when durable event lifecycle persistence fails."""


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
        event_journal: DurableEventJournal | None = None,
        telemetry: RuntimeTelemetry | None = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be greater than zero")
        if initial_sequence < 0:
            raise ValueError("initial_sequence must not be negative")
        self._queue: Queue[_Envelope[Any] | object] = Queue(maxsize=queue_capacity)
        self._event_recorder = event_recorder
        self._completion_hook = completion_hook
        self._failure_hook = failure_hook
        self._event_journal = event_journal
        self._telemetry = telemetry
        self._state_lock = Lock()
        self._state = "created"
        self._sequence = initial_sequence
        self._worker: Thread | None = None
        self._abort_requested = False

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
            if self._queue.full():
                raise AgentRuntimeQueueFull("Agent event queue is full")
            if self._event_journal is not None:
                try:
                    self._event_journal.accepted(event)
                except Exception as exc:
                    self._state = "failed"
                    self._queue.put_nowait(_STOP)
                    raise AgentRuntimeJournalError(
                        "Agent event could not be durably accepted"
                    ) from exc
            try:
                self._queue.put_nowait(envelope)
            except Full as exc:
                raise AgentRuntimeQueueFull("Agent event queue is full") from exc
            self._observe_telemetry("event_accepted", event, self._queue.qsize())
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

    def abort(self) -> None:
        """Stop without draining or invoking completion hooks for active work."""

        with self._state_lock:
            if self._state in {"stopped", "aborted"}:
                return
            self._abort_requested = True
            self._state = "aborted"
            worker = self._worker
        while True:
            try:
                pending = self._queue.get_nowait()
            except Empty:
                break
            try:
                if pending is not _STOP:
                    envelope = cast(_Envelope[Any], pending)
                    if not envelope.future.done():
                        envelope.future.set_exception(
                            AgentRuntimeStopped("Agent runtime was abruptly stopped")
                        )
            finally:
                self._queue.task_done()
        self._queue.put(_STOP)
        if worker is not None:
            worker.join()

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
                envelope = cast(_Envelope[Any], item)
                self._sequence += 1
                event = replace(envelope.event, processing_sequence=self._sequence)
                can_deliver = envelope.future.set_running_or_notify_cancel()
                if self._event_journal is not None:
                    try:
                        self._event_journal.started(event)
                    except Exception:
                        self._fail_stop(
                            envelope,
                            AgentRuntimeJournalError(
                                "Agent event start could not be journaled"
                            ),
                            can_deliver=can_deliver,
                        )
                        return
                self._record(event, "started")
                self._observe_telemetry("event_started", event, self._queue.qsize())
                event_token = _current_event.set(event)
                rollback_token = _rollback_callbacks.set(())
                try:
                    value = envelope.handler()
                except Exception as exc:
                    try:
                        snapshot_hash = (
                            None
                            if self._failure_hook is None
                            else self._failure_hook(event, exc)
                        )
                        if self._event_journal is not None:
                            self._event_journal.failed(
                                event, type(exc).__name__, snapshot_hash
                            )
                    except Exception:
                        self._fail_stop(
                            envelope,
                            AgentRuntimeJournalError(
                                "Agent event failure could not be committed"
                            ),
                            can_deliver=can_deliver,
                        )
                        return
                    self._record(event, "failed", exception_type=type(exc).__name__)
                    self._observe_telemetry(
                        "event_finished",
                        event,
                        self._queue.qsize(),
                        status="failure",
                    )
                    if can_deliver:
                        envelope.future.set_exception(exc)
                else:
                    if self._abort_requested:
                        if can_deliver and not envelope.future.done():
                            envelope.future.set_exception(
                                AgentRuntimeStopped(
                                    "Agent runtime was abruptly stopped before commit"
                                )
                            )
                        continue
                    try:
                        snapshot_hash = (
                            None
                            if self._completion_hook is None
                            else self._completion_hook(event)
                        )
                        if self._event_journal is not None:
                            if snapshot_hash is None:
                                raise RuntimeError(
                                    "durable journal requires a committed snapshot hash"
                                )
                            self._event_journal.completed(event, snapshot_hash)
                    except Exception as exc:
                        for callback in reversed(_rollback_callbacks.get()):
                            try:
                                callback()
                            except Exception:
                                pass
                        self._record(event, "failed", exception_type=type(exc).__name__)
                        if self._event_journal is None:
                            if can_deliver:
                                envelope.future.set_exception(exc)
                        else:
                            self._fail_stop(
                                envelope,
                                AgentRuntimeJournalError(
                                    "Agent event completion could not be committed"
                                ),
                                can_deliver=can_deliver,
                            )
                            return
                    else:
                        self._record(event, "completed")
                        self._observe_telemetry(
                            "event_finished",
                            event,
                            self._queue.qsize(),
                            status="success",
                        )
                        if can_deliver:
                            envelope.future.set_result(
                                AgentEventOutcome(event=event, value=value)
                            )
                finally:
                    _rollback_callbacks.reset(rollback_token)
                    _current_event.reset(event_token)
            finally:
                self._queue.task_done()

    def _fail_stop(
        self,
        envelope: _Envelope[Any],
        error: AgentRuntimeJournalError,
        *,
        can_deliver: bool,
    ) -> None:
        with self._state_lock:
            self._state = "failed"
        self._observe_telemetry(
            "event_finished",
            envelope.event,
            self._queue.qsize(),
            status="failure",
        )
        if can_deliver and not envelope.future.done():
            envelope.future.set_exception(error)
        while True:
            try:
                pending = self._queue.get_nowait()
            except Empty:
                break
            try:
                if pending is not _STOP:
                    pending_envelope = cast(_Envelope[Any], pending)
                    if pending_envelope.future.done():
                        continue
                    pending_envelope.future.set_exception(
                        AgentRuntimeJournalError(
                            "Agent runtime stopped after journal failure"
                        )
                    )
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

    def _observe_telemetry(
        self,
        method: str,
        event: AgentEvent,
        queue_depth: int,
        *,
        status: str | None = None,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            callback = getattr(self._telemetry, method)
            if status is None:
                callback(event, queue_depth)
            else:
                callback(event, status, queue_depth)
        except Exception:
            # Telemetry persistence and export are never authoritative runtime work.
            return


def current_agent_event() -> AgentEvent | None:
    return _current_event.get()
