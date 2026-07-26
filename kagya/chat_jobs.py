"""Durable public chat jobs submitted to the authoritative AgentRuntime."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import hmac
import json
import os
from pathlib import Path
from threading import Condition, Event, RLock, Thread, Timer
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from kagya.operation_status import (
    OperationCancelCode,
    OperationErrorCode,
    OperationState,
    OperationStatus,
    operation_now,
)
from kagya.chat_projection import ChatProjectionPublisher
from kagya.runtime import (
    AgentEventOutcome,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeJournalError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    CancellationToken,
    OperationCanceled,
)


class ChatJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    status: OperationStatus
    enqueue_sequence: int = Field(ge=1)
    client_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    sealed_request: str
    pending_result: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    last_stream_sequence: int = Field(default=0, ge=0)
    requested_cancel_code: OperationCancelCode | None = None
    cancel_requested_at: datetime | None = None
    timeout_deadline: datetime | None = None
    terminal_projection_start_sequence: int | None = Field(default=None, ge=1)
    terminal_projection_event_count: int = Field(default=0, ge=0)
    terminal_projection_state: Literal[
        "none", "pending", "published", "projection_failed"
    ] = "none"


@dataclass(frozen=True)
class ChatStreamEvent:
    event_id: int
    event: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ChatRecoveryDisposition:
    state: str
    cancel_code: OperationCancelCode | None = None


Executor = Callable[[dict[str, Any]], dict[str, Any]]
CompletionObserver = Callable[[dict[str, Any]], None]
ResultReconstructor = Callable[[str], dict[str, Any] | None]
ReplayCompleteCallback = Callable[[], None]


class ReplayState(StrEnum):
    INACTIVE = "inactive"
    STARTING = "starting"
    REPLAYING = "replaying"
    COMPLETE = "complete"
    FAILED = "failed"


class ChatJobRegistry:
    """Persistent metadata and bounded public replay around one AgentRuntime."""

    def __init__(
        self,
        path: Path,
        runtime: AgentRuntime,
        executor: Executor,
        *,
        replay_limit: int = 256,
        timeout_seconds: float = 300.0,
        completion_observer: CompletionObserver | None = None,
        recovery_dispositions: Mapping[
            str, str | ChatRecoveryDisposition
        ] | None = None,
        required_event_ids: set[str] | None = None,
        result_reconstructor: ResultReconstructor | None = None,
    ) -> None:
        self.path = path
        self.runtime = runtime
        self.executor = executor
        self.replay_limit = replay_limit
        self.timeout_seconds = timeout_seconds
        self.completion_observer = completion_observer
        self._has_recovery_evidence = recovery_dispositions is not None
        self.recovery_dispositions = {
            event_id: (
                value
                if isinstance(value, ChatRecoveryDisposition)
                else ChatRecoveryDisposition(value)
            )
            for event_id, value in (recovery_dispositions or {}).items()
        }
        self.required_event_ids = required_event_ids or set()
        self.result_reconstructor = result_reconstructor
        self._lock = RLock()
        self._changed = Condition(self._lock)
        self._records: dict[str, ChatJobRecord] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._timers: dict[str, Timer] = {}
        self._events: dict[str, deque[ChatStreamEvent]] = {}
        self._event_sequences: dict[str, int] = {}
        self._errors: dict[str, BaseException] = {}
        self._enqueue_sequence = 0
        self._replay_state = ReplayState.INACTIVE
        self._queued_for_replay: list[str] = []
        self._replay_stop = Event()
        self._replay_thread: Thread | None = None
        self._replay_complete_callback: ReplayCompleteCallback | None = None
        self._projection = ChatProjectionPublisher(self)
        self._key = self._load_key()
        self._load()
        self._missing_required_event_ids = self.required_event_ids - {
            record.status.event_id for record in self._records.values()
        }
        self._recovery_ambiguous = False
        with self._lock:
            self._recover_without_submission()

    def activate(
        self, completion_callback: ReplayCompleteCallback | None = None
    ) -> None:
        """Start queued replay without blocking API process startup."""

        invoke_callback = False
        self._projection.start()
        with self._lock:
            if completion_callback is not None:
                self._replay_complete_callback = completion_callback
            if self._replay_state == ReplayState.COMPLETE:
                invoke_callback = completion_callback is not None
            elif self._replay_state != ReplayState.INACTIVE:
                return
            elif not self._queued_for_replay:
                self._replay_state = ReplayState.COMPLETE
                invoke_callback = completion_callback is not None
            else:
                self._replay_state = ReplayState.STARTING
                self._replay_thread = Thread(
                    target=self._run_replay_feeder,
                    name="kagya-chat-replay",
                    daemon=True,
                )
                self._replay_thread.start()
        if invoke_callback:
            callback = completion_callback or self._replay_complete_callback
            if callback is not None:
                callback()

    def enqueue(
        self,
        request: dict[str, Any],
        *,
        client_id: str,
        idempotency_key: str,
        correlation_id: str,
    ) -> tuple[ChatJobRecord, bool]:
        self._projection.start()
        client_id = _safe_identity(client_id)
        idempotency_key = _safe_identity(idempotency_key)
        with self._lock:
            duplicate = next(
                (
                    record
                    for record in self._records.values()
                    if record.client_id == client_id
                    and record.idempotency_key == idempotency_key
                ),
                None,
            )
            if duplicate is not None:
                return duplicate.model_copy(deep=True), False
            operation_id = str(uuid4())
            event_id = str(uuid4())
            now = operation_now()
            self._enqueue_sequence += 1
            queue_position = self._queued_count() + 1
            status = OperationStatus(
                operation_id=operation_id,
                event_id=event_id,
                status=OperationState.QUEUED,
                status_sequence=1,
                queue_position=queue_position,
                submitted_at=now,
                updated_at=now,
            )
            record = ChatJobRecord(
                status=status,
                enqueue_sequence=self._enqueue_sequence,
                client_id=client_id,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                sealed_request=self._seal(request),
            )
            self._records[operation_id] = record
            self._persist()
            self._emit(operation_id, "status", status.model_dump(mode="json"))
            try:
                self._submit(record)
            except (AgentRuntimeQueueFull, AgentRuntimeStopped):
                del self._records[operation_id]
                self._persist()
                raise
            except Exception:
                # Keep the durable spool when journal acceptance may have succeeded.
                raise
            return record.model_copy(deep=True), True

    def get(self, operation_id: str) -> ChatJobRecord | None:
        with self._lock:
            record = self._records.get(operation_id)
            return None if record is None else record.model_copy(deep=True)

    def error(self, operation_id: str) -> BaseException | None:
        with self._lock:
            return self._errors.get(operation_id)

    def projection_pending(self, operation_id: str) -> bool:
        with self._lock:
            record = self._records.get(operation_id)
            return bool(
                record is not None
                and record.terminal_projection_state
                in {"pending", "projection_failed"}
            )

    @property
    def is_ready(self) -> bool:
        return (
            self._replay_state == ReplayState.COMPLETE
            and not self._missing_required_event_ids
            and not self._recovery_ambiguous
        )

    def shutdown(self) -> None:
        self._replay_stop.set()
        with self._changed:
            self._changed.notify_all()
        replay_thread = self._replay_thread
        if replay_thread is not None:
            replay_thread.join(timeout=5.0)
            if replay_thread.is_alive():
                raise RuntimeError("chat replay feeder did not stop")
        with self._lock:
            cancellable = [
                operation_id
                for operation_id, record in self._records.items()
                if record.status.status
                in {OperationState.QUEUED, OperationState.RUNNING}
            ]
        for operation_id in cancellable:
            self.cancel(operation_id, OperationCancelCode.SHUTDOWN)

    def close(self) -> None:
        self._projection.shutdown()

    def cancel(self, operation_id: str, code: OperationCancelCode) -> str:
        with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                return "not_found"
            if record.status.status == OperationState.COMPLETED:
                return "already_completed"
            if record.status.status in {
                OperationState.CANCELED,
                OperationState.FAILED,
            }:
                return record.status.status.value
            if record.status.status == OperationState.FINALIZING:
                return "already_finalizing"
            token = self._tokens.get(operation_id)
            previous = record.model_copy(deep=True)
            previous_event_sequence = self._event_sequences.get(operation_id, 0)
            if record.status.status == OperationState.QUEUED:
                try:
                    record.requested_cancel_code = code
                    record.cancel_requested_at = operation_now()
                    self._transition(
                        operation_id, OperationState.CANCELED, cancel_code=code
                    )
                except Exception:
                    self._records[operation_id] = previous
                    self._event_sequences[operation_id] = previous_event_sequence
                    raise
                if token is not None:
                    token.cancel(code.value)
                self._refresh_queue_positions()
                self._emit(operation_id, "canceled", {"code": code.value})
                return "canceled"
            try:
                record.requested_cancel_code = code
                record.cancel_requested_at = operation_now()
                self._transition(
                    operation_id,
                    record.status.status,
                    cancel_requested=True,
                )
            except Exception:
                self._records[operation_id] = previous
                self._event_sequences[operation_id] = previous_event_sequence
                raise
            if token is not None:
                token.cancel(code.value)
            return "cancel_requested"

    def events_after(
        self, operation_id: str, last_event_id: int
    ) -> list[ChatStreamEvent]:
        with self._lock:
            return [
                event
                for event in self._events.get(operation_id, ())
                if event.event_id > last_event_id
            ]

    def wait_for_events(
        self, operation_id: str, last_event_id: int, timeout: float
    ) -> list[ChatStreamEvent]:
        with self._changed:
            events = self.events_after(operation_id, last_event_id)
            if events:
                return events
            self._changed.wait(timeout)
            return self.events_after(operation_id, last_event_id)

    def _recover_without_submission(self) -> None:
        for record in sorted(
            self._records.values(), key=lambda item: item.enqueue_sequence
        ):
            operation_id = record.status.operation_id
            self._events[operation_id] = deque(maxlen=self.replay_limit)
            self._event_sequences[operation_id] = record.last_stream_sequence
            recovery = self.recovery_dispositions.get(record.status.event_id)
            disposition = None if recovery is None else recovery.state
            reconcile = record.status.status in {
                OperationState.RUNNING,
                OperationState.FINALIZING,
            }
            if disposition == "committed":
                reconcile = (
                    reconcile or record.status.status != OperationState.COMPLETED
                )
                reconcile = reconcile or record.result is None
            elif disposition == "canceled":
                reconcile = reconcile or record.status.status != OperationState.CANCELED
            elif disposition == "failed":
                reconcile = reconcile or record.status.status != OperationState.FAILED
            elif disposition == "uncommitted":
                expected = (
                    OperationState.CANCELED
                    if record.status.cancel_requested
                    else OperationState.FAILED
                )
                reconcile = reconcile or record.status.status != expected
            elif disposition == "ambiguous":
                reconcile = True

            if record.status.status == OperationState.QUEUED and disposition in {
                None,
                "queued",
            }:
                if self._has_recovery_evidence and disposition is None:
                    raise ValueError(
                        "queued chat job has no matching journal recovery evidence"
                    )
                self._open(record.sealed_request)
                self._queued_for_replay.append(operation_id)
            elif reconcile:
                if disposition == "committed":
                    result = record.pending_result
                    if result is None and self.result_reconstructor is not None:
                        try:
                            result = self.result_reconstructor(record.status.event_id)
                        except Exception:
                            self._recovery_ambiguous = True
                            continue
                    record.result = result
                    record.pending_result = None
                    if result is None:
                        self._transition(
                            operation_id,
                            OperationState.COMPLETED,
                            error_code=OperationErrorCode.COMMITTED_RESULT_UNAVAILABLE,
                        )
                        self._emit(
                            operation_id,
                            "error",
                            {
                                "code": OperationErrorCode.COMMITTED_RESULT_UNAVAILABLE.value
                            },
                        )
                    else:
                        self._complete(operation_id, result)
                elif disposition in {"uncommitted", "failed", "canceled"}:
                    record.pending_result = None
                    record.result = None
                    canceled = disposition == "canceled" or (
                        disposition == "uncommitted" and record.status.cancel_requested
                    )
                    if canceled:
                        journal_cancel_code = (
                            None if recovery is None else recovery.cancel_code
                        )
                        registry_cancel_code = (
                            record.requested_cancel_code or record.status.cancel_code
                        )
                        if (
                            journal_cancel_code is not None
                            and registry_cancel_code is not None
                            and journal_cancel_code != registry_cancel_code
                        ):
                            self._recovery_ambiguous = True
                            continue
                        code = (
                            journal_cancel_code
                            or registry_cancel_code
                            or OperationCancelCode.SHUTDOWN
                        )
                        self._transition(
                            operation_id, OperationState.CANCELED, cancel_code=code
                        )
                        self._emit(operation_id, "canceled", {"code": code.value})
                    else:
                        self._transition(
                            operation_id,
                            OperationState.FAILED,
                            error_code=OperationErrorCode.INTERRUPTED,
                        )
                        self._emit(
                            operation_id,
                            "error",
                            {"code": OperationErrorCode.INTERRUPTED.value},
                        )
                elif self._has_recovery_evidence:
                    self._recovery_ambiguous = True
                else:
                    record.pending_result = None
                    self._transition(
                        operation_id,
                        OperationState.FAILED,
                        error_code=OperationErrorCode.INTERRUPTED,
                    )
                    self._emit(
                        operation_id,
                        "error",
                        {"code": OperationErrorCode.INTERRUPTED.value},
                    )
            else:
                if record.terminal_projection_start_sequence is not None:
                    continue
                self._emit(
                    operation_id,
                    "status",
                    record.status.model_dump(mode="json"),
                )
                if record.result is not None:
                    self._emit(operation_id, "final", record.result)
                elif record.status.status == OperationState.FAILED:
                    error_value = (
                        record.status.error_code.value
                        if record.status.error_code
                        else OperationErrorCode.INTERNAL_ERROR.value
                    )
                    self._emit(
                        operation_id,
                        "error",
                        {"code": error_value},
                    )
                elif record.status.status == OperationState.CANCELED:
                    cancel_value = (
                        record.status.cancel_code.value
                        if record.status.cancel_code
                        else OperationCancelCode.CLIENT_REQUEST.value
                    )
                    self._emit(
                        operation_id,
                        "canceled",
                        {"code": cancel_value},
                    )
        self._refresh_queue_positions()

    def _run_replay_feeder(self) -> None:
        try:
            with self._lock:
                self._replay_state = ReplayState.REPLAYING
            while not self._replay_stop.is_set():
                with self._changed:
                    if not self._queued_for_replay:
                        self._replay_state = ReplayState.COMPLETE
                        callback = self._replay_complete_callback
                        break
                    operation_id = self._queued_for_replay[0]
                    record = self._records[operation_id]
                    try:
                        self._submit(record)
                    except AgentRuntimeQueueFull:
                        self._changed.wait()
                        continue
                    self._queued_for_replay.pop(0)
            else:
                return
        except Exception:
            with self._lock:
                self._replay_state = ReplayState.FAILED
                self._recovery_ambiguous = True
            return
        if callback is not None:
            callback()

    def _submit(self, record: ChatJobRecord) -> None:
        operation_id = record.status.operation_id
        token = CancellationToken()
        self._tokens[operation_id] = token
        timer: Timer | None = None
        if self.timeout_seconds > 0:
            if record.timeout_deadline is None:
                record.timeout_deadline = operation_now() + timedelta(
                    seconds=self.timeout_seconds
                )
                self._persist()
            remaining = max(
                0.0, (record.timeout_deadline - operation_now()).total_seconds()
            )
            if remaining == 0:
                if record.requested_cancel_code is None:
                    record.requested_cancel_code = OperationCancelCode.TIMEOUT
                    record.cancel_requested_at = operation_now()
                    self._transition(
                        operation_id,
                        OperationState.QUEUED,
                        cancel_requested=True,
                        queue_position=record.status.queue_position,
                    )
                token.cancel(record.requested_cancel_code.value)
            else:
                timer = Timer(
                    remaining,
                    lambda: self.cancel(operation_id, OperationCancelCode.TIMEOUT),
                )
                timer.daemon = True
                self._timers[operation_id] = timer

        def run() -> dict[str, Any]:
            with self._lock:
                current = self._records[operation_id]
                if current.status.status == OperationState.CANCELED:
                    raise OperationCanceled(
                        current.status.cancel_code or "client_request"
                    )
                self._transition(operation_id, OperationState.RUNNING)
                self._refresh_queue_positions()
                self._changed.notify_all()
            token.raise_if_canceled()
            result = self.executor(self._open(record.sealed_request))
            with self._lock:
                if self._records[operation_id].status.status == OperationState.RUNNING:
                    self._enter_finalizing(operation_id, token)
                self._records[operation_id].pending_result = result
                self._persist()
            return result

        try:
            future = self.runtime.submit(
                AgentEventType.CHAT,
                source="api.chat.job",
                handler=run,
                payload={"operation_id": operation_id},
                correlation_id=record.correlation_id,
                event_id=record.status.event_id,
                cancellation_token=token,
                finalization_boundary=lambda: self._enter_finalizing(
                    operation_id, token
                ),
            )
        except Exception:
            self._tokens.pop(operation_id, None)
            self._timers.pop(operation_id, None)
            raise

        def finish(completed: Future[AgentEventOutcome[dict[str, Any]]]) -> None:
            self._finish(operation_id, completed)

        future.add_done_callback(finish)
        if timer is not None:
            with self._lock:
                if self._timers.get(operation_id) is timer:
                    timer.start()

    def _finish(
        self,
        operation_id: str,
        future: Future[AgentEventOutcome[dict[str, Any]]],
    ) -> None:
        try:
            outcome = future.result()
        except OperationCanceled as exc:
            self._errors[operation_id] = exc
            try:
                cancel_code = OperationCancelCode(exc.code)
            except ValueError:
                with self._lock:
                    self._records[operation_id].pending_result = None
                    self._transition(
                        operation_id,
                        OperationState.FAILED,
                        error_code=OperationErrorCode.INTERNAL_ERROR,
                    )
                    self._emit(
                        operation_id,
                        "error",
                        {"code": OperationErrorCode.INTERNAL_ERROR.value},
                    )
                return
            with self._lock:
                if self._records[operation_id].status.status != OperationState.CANCELED:
                    self._records[operation_id].pending_result = None
                    self._transition(
                        operation_id,
                        OperationState.CANCELED,
                        cancel_code=cancel_code,
                    )
                    self._emit(operation_id, "canceled", {"code": cancel_code.value})
        except AgentRuntimeJournalError as exc:
            self._errors[operation_id] = exc
            with self._lock:
                record = self._records[operation_id]
                if record.status.status == OperationState.FINALIZING:
                    self._transition(
                        operation_id,
                        OperationState.FINALIZING,
                        error_code=OperationErrorCode.COMMIT_INDETERMINATE,
                    )
        except Exception as exc:
            self._errors[operation_id] = exc
            error_code = (
                OperationErrorCode.PROVIDER_ERROR
                if "provider" in type(exc).__name__.lower()
                else OperationErrorCode.INTERNAL_ERROR
            )
            with self._lock:
                if self._records[operation_id].status.status != OperationState.CANCELED:
                    self._records[operation_id].pending_result = None
                    self._transition(
                        operation_id, OperationState.FAILED, error_code=error_code
                    )
                    self._emit(operation_id, "error", {"code": error_code.value})
        else:
            with self._lock:
                record = self._records[operation_id]
                if record.status.status == OperationState.CANCELED:
                    return
                record.result = outcome.value
                record.pending_result = None
                self._complete(operation_id, outcome.value)
        finally:
            with self._lock:
                self._tokens.pop(operation_id, None)
                timer = self._timers.pop(operation_id, None)
                if timer is not None:
                    timer.cancel()
                self._refresh_queue_positions()

    def _complete(self, operation_id: str, result: dict[str, Any]) -> None:
        if self.completion_observer is not None:
            try:
                self.completion_observer(result)
            except Exception:
                pass
        self._transition(operation_id, OperationState.COMPLETED)
        chunks = _public_chunks(str(result.get("response", "")))
        record = self._records[operation_id]
        start = self._event_sequences.get(operation_id, 0) + 1
        count = len(chunks) + 1
        record.terminal_projection_start_sequence = start
        record.terminal_projection_event_count = count
        record.terminal_projection_state = "pending"
        record.last_stream_sequence = start + count - 1
        self._event_sequences[operation_id] = record.last_stream_sequence
        self._persist()
        self._projection.publish_terminal(operation_id)

    def pending_projection_ids(self) -> list[str]:
        with self._lock:
            return [
                record.status.operation_id
                for record in self._records.values()
                if record.terminal_projection_start_sequence is not None
                and record.terminal_projection_state
                in {"pending", "projection_failed"}
            ]

    def projection_ids_for_recovery(self) -> list[str]:
        with self._lock:
            return [
                record.status.operation_id
                for record in self._records.values()
                if record.terminal_projection_start_sequence is not None
            ]

    def publish_terminal_projection(self, operation_id: str) -> None:
        with self._changed:
            record = self._records[operation_id]
            start = record.terminal_projection_start_sequence
            if start is None or record.result is None:
                return
            chunks = _public_chunks(str(record.result.get("response", "")))
            if record.terminal_projection_event_count != len(chunks) + 1:
                raise ValueError("terminal projection metadata does not match result")
            events = self._events.setdefault(
                operation_id, deque(maxlen=self.replay_limit)
            )
            existing = {event.event_id for event in events}
            bundle = [
                ChatStreamEvent(start + index, "token", {"text": chunk})
                for index, chunk in enumerate(chunks)
            ]
            bundle.append(
                ChatStreamEvent(start + len(chunks), "final", record.result)
            )
            for event in bundle:
                if event.event_id not in existing:
                    events.append(event)
            if record.terminal_projection_state != "published":
                record.terminal_projection_state = "published"
                self._persist()
            self._changed.notify_all()

    def mark_projection_failed(self, operation_id: str) -> None:
        with self._lock:
            record = self._records.get(operation_id)
            if record is None or record.terminal_projection_state == "published":
                return
            record.terminal_projection_state = "projection_failed"
            self._persist()

    def _enter_finalizing(self, operation_id: str, token: CancellationToken) -> None:
        with self._lock:
            record = self._records[operation_id]
            if record.status.status == OperationState.FINALIZING:
                return
            if record.status.status != OperationState.RUNNING:
                token.raise_if_canceled()
                raise RuntimeError(
                    "chat job cannot enter finalizing from its current state"
                )
            if record.status.cancel_requested or token.is_canceled:
                token.raise_if_canceled()
                raise OperationCanceled(
                    (
                        record.requested_cancel_code or OperationCancelCode.SHUTDOWN
                    ).value
                )
            self._transition(operation_id, OperationState.FINALIZING)

    def _transition(
        self,
        operation_id: str,
        state: OperationState,
        *,
        error_code: OperationErrorCode | None = None,
        cancel_code: OperationCancelCode | None = None,
        cancel_requested: bool | None = None,
        queue_position: int | None = None,
    ) -> None:
        record = self._records[operation_id]
        previous = record.status
        now = operation_now()
        status = OperationStatus(
            operation_id=previous.operation_id,
            event_id=previous.event_id,
            status=state,
            status_sequence=previous.status_sequence + 1,
            queue_position=queue_position if state == OperationState.QUEUED else None,
            submitted_at=previous.submitted_at,
            started_at=(
                now
                if state == OperationState.RUNNING and previous.started_at is None
                else previous.started_at
            ),
            finalizing_at=(
                now if state == OperationState.FINALIZING else previous.finalizing_at
            ),
            completed_at=(
                now
                if state
                in {
                    OperationState.COMPLETED,
                    OperationState.FAILED,
                    OperationState.CANCELED,
                }
                else None
            ),
            updated_at=now,
            error_code=error_code,
            cancel_code=cancel_code,
            cancel_requested=(
                previous.cancel_requested
                if cancel_requested is None
                else cancel_requested
            ),
            result_available=(
                state == OperationState.COMPLETED
                and error_code != OperationErrorCode.COMMITTED_RESULT_UNAVAILABLE
            ),
        )
        record.status = status
        self._emit(operation_id, "status", status.model_dump(mode="json"))

    def _refresh_queue_positions(self) -> None:
        queued = sorted(
            (
                record
                for record in self._records.values()
                if record.status.status == OperationState.QUEUED
            ),
            key=lambda item: item.enqueue_sequence,
        )
        for position, record in enumerate(queued, 1):
            if record.status.queue_position != position:
                self._transition(
                    record.status.operation_id,
                    OperationState.QUEUED,
                    queue_position=position,
                )

    def _queued_count(self) -> int:
        return sum(
            record.status.status == OperationState.QUEUED
            for record in self._records.values()
        )

    def _emit(self, operation_id: str, event: str, data: dict[str, Any]) -> None:
        with self._changed:
            events = self._events.setdefault(
                operation_id, deque(maxlen=self.replay_limit)
            )
            sequence = self._event_sequences.get(operation_id, 0) + 1
            self._event_sequences[operation_id] = sequence
            self._records[operation_id].last_stream_sequence = sequence
            self._persist()
            events.append(ChatStreamEvent(sequence, event, data))
            self._changed.notify_all()

    def _load(self) -> None:
        if not self.path.exists():
            return
        values = json.loads(self.path.read_text(encoding="utf-8"))
        for value in values:
            if value.get("schema_version", 1) in {1, 2}:
                value["schema_version"] = 3
                value.setdefault("requested_cancel_code", None)
                value.setdefault("cancel_requested_at", None)
                value.setdefault("timeout_deadline", None)
                value.setdefault("terminal_projection_start_sequence", None)
                value.setdefault("terminal_projection_event_count", 0)
                value.setdefault("terminal_projection_state", "none")
        records = [ChatJobRecord.model_validate(value) for value in values]
        self._records = {record.status.operation_id: record for record in records}
        self._enqueue_sequence = max(
            (record.enqueue_sequence for record in records), default=0
        )

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        payload = json.dumps(
            [
                record.model_dump(mode="json")
                for record in sorted(
                    self._records.values(), key=lambda item: item.enqueue_sequence
                )
            ],
            separators=(",", ":"),
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_key(self) -> bytes:
        key_path = self.path.with_suffix(self.path.suffix + ".key")
        if key_path.exists():
            return key_path.read_bytes()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(32)
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(key)
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(key_path.parent)
        return key

    def _seal(self, value: dict[str, Any]) -> str:
        plaintext = json.dumps(value, separators=(",", ":")).encode()
        nonce = os.urandom(16)
        ciphertext = _xor_stream(plaintext, self._key, nonce)
        signature = hmac.digest(self._key, nonce + ciphertext, "sha256")
        return (nonce + signature + ciphertext).hex()

    def _open(self, value: str) -> dict[str, Any]:
        sealed = bytes.fromhex(value)
        nonce, signature, ciphertext = sealed[:16], sealed[16:48], sealed[48:]
        if not hmac.compare_digest(
            signature, hmac.digest(self._key, nonce + ciphertext, "sha256")
        ):
            raise ValueError("chat request spool authentication failed")
        opened = json.loads(_xor_stream(ciphertext, self._key, nonce))
        if not isinstance(opened, dict):
            raise ValueError("chat request spool is invalid")
        return opened


def _xor_stream(value: bytes, key: bytes, nonce: bytes) -> bytes:
    output = bytearray()
    for counter in range((len(value) + 31) // 32):
        output.extend(hmac.digest(key, nonce + counter.to_bytes(8, "big"), "sha256"))
    return bytes(left ^ right for left, right in zip(value, output, strict=False))


def _safe_identity(value: str) -> str:
    stripped = value.strip()
    if not stripped or len(stripped) > 128:
        raise ValueError("idempotency identity must contain 1 to 128 characters")
    return stripped


def _public_chunks(value: str, size: int = 24) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
