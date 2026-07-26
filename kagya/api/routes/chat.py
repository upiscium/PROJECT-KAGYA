"""Synchronous and durable streaming chat routes."""

from collections.abc import Iterator
import json
from threading import RLock
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from uuid import uuid4

from kagya.api.observability import RuntimeEventLog
from kagya.api.dependencies import get_agent_runtime
from kagya.api.schemas.chat import (
    ChatCancelResponse,
    ChatJobAccepted,
    ChatJobResult,
    ChatJobStatus,
    ChatRequest,
    ChatResponse,
    EmotionSchema,
    ModelSchema,
)
from kagya.chat_jobs import ChatJobRegistry, ChatStreamEvent
from kagya.operation_status import OperationCancelCode, OperationState
from kagya.runtime import JournalLifecycle
from kagya.runtime import (
    AgentRuntime,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    ChatResult,
)
from kagya.identity import OriginActor


router = APIRouter(prefix="/api", tags=["chat"])
_registry_lock = RLock()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http_request: Request,
    _runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatResponse:
    registry = get_chat_job_registry(http_request)
    context_id = request.context_id or f"ctx-{uuid4()}"
    payload = request.model_dump(mode="json")
    payload["_context_id"] = context_id
    payload["_create_context"] = request.context_id is None
    try:
        job, _ = registry.enqueue(
            payload,
            client_id=request.client_session_id or f"sync:{uuid4()}",
            idempotency_key=str(uuid4()),
            correlation_id=context_id,
        )
        final = _wait_for_terminal(registry, job.status.operation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Context not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AgentRuntimeQueueFull as exc:
        raise HTTPException(
            status_code=429, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    except AgentRuntimeStopped as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if final.result is None:
        error = registry.error(job.status.operation_id)
        if isinstance(error, KeyError):
            raise HTTPException(status_code=404, detail="Context not found")
        if isinstance(error, ValueError):
            raise HTTPException(status_code=409, detail=str(error))
        if isinstance(error, RuntimeError) and str(error) != "Operation canceled":
            raise HTTPException(status_code=500, detail=str(error))
        if final.status.status == OperationState.CANCELED:
            raise HTTPException(status_code=409, detail="Chat was canceled")
        raise HTTPException(status_code=500, detail="Chat operation failed")
    return ChatResponse.model_validate(final.result)


@router.post(
    "/chat/jobs", response_model=ChatJobAccepted, status_code=status.HTTP_202_ACCEPTED
)
def create_chat_job(
    request: ChatRequest,
    http_request: Request,
    idempotency_key: str = Header(
        alias="Idempotency-Key", min_length=1, max_length=128
    ),
    client_id: str | None = Header(default=None, alias="X-KAGYA-Client-ID"),
    _runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatJobAccepted:
    registry = get_chat_job_registry(http_request)
    context_id = request.context_id or f"ctx-{uuid4()}"
    payload = request.model_dump(mode="json")
    payload["_context_id"] = context_id
    payload["_create_context"] = request.context_id is None
    try:
        record, created = registry.enqueue(
            payload,
            client_id=client_id
            or request.client_session_id
            or _request_client(http_request),
            idempotency_key=idempotency_key,
            correlation_id=context_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AgentRuntimeQueueFull as exc:
        raise HTTPException(
            status_code=429, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    except AgentRuntimeStopped as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    operation_id = record.status.operation_id
    return ChatJobAccepted(
        operation=record.status,
        status_url=f"/api/chat/jobs/{operation_id}",
        result_url=f"/api/chat/jobs/{operation_id}/result",
        events_url=f"/api/chat/jobs/{operation_id}/events",
        duplicate=not created,
    )


@router.get("/chat/jobs/{operation_id}", response_model=ChatJobStatus)
def get_chat_job(
    operation_id: str,
    request: Request,
    _runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatJobStatus:
    record = _required_job(get_chat_job_registry(request), operation_id)
    return ChatJobStatus(operation=record.status)


@router.get("/chat/jobs/{operation_id}/result", response_model=ChatJobResult)
def get_chat_result(
    operation_id: str,
    request: Request,
    _runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatJobResult:
    record = _required_job(get_chat_job_registry(request), operation_id)
    if record.status.status != OperationState.COMPLETED or record.result is None:
        raise HTTPException(status_code=409, detail="Chat result is not available")
    return ChatJobResult(
        operation=record.status, result=ChatResponse.model_validate(record.result)
    )


@router.delete("/chat/jobs/{operation_id}", response_model=ChatCancelResponse)
def cancel_chat_job(
    operation_id: str,
    request: Request,
    _runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ChatCancelResponse:
    registry = get_chat_job_registry(request)
    disposition = registry.cancel(operation_id, OperationCancelCode.CLIENT_REQUEST)
    if disposition == "not_found":
        raise HTTPException(status_code=404, detail="Chat job not found")
    record = _required_job(registry, operation_id)
    if disposition == "already_finalizing":
        raise HTTPException(status_code=409, detail="already_finalizing")
    return ChatCancelResponse(disposition=disposition, operation=record.status)


@router.get("/chat/jobs/{operation_id}/events")
def stream_chat_job(
    operation_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    _runtime: AgentRuntime = Depends(get_agent_runtime),
) -> StreamingResponse:
    registry = get_chat_job_registry(request)
    _required_job(registry, operation_id)
    try:
        cursor = 0 if last_event_id is None else max(0, int(last_event_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Last-Event-ID must be an integer"
        ) from exc

    def generate() -> Iterator[str]:
        nonlocal cursor
        while True:
            events = registry.wait_for_events(operation_id, cursor, 15.0)
            if not events:
                yield ": heartbeat\n\n"
            for event in events:
                cursor = event.event_id
                yield _sse(event)
            record = registry.get(operation_id)
            if record is None or (
                record.status.status
                in {
                    OperationState.COMPLETED,
                    OperationState.FAILED,
                    OperationState.CANCELED,
                }
                and not registry.events_after(operation_id, cursor)
            ):
                return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_chat_job_registry(app: Any) -> ChatJobRegistry:
    settings = app.state.settings.api
    registry_path = settings.chat_job_registry_path
    if registry_path == type(settings).model_fields["chat_job_registry_path"].default:
        registry_path = app.state.settings.agent_state.path.parent / "chat_jobs.json"

    def execute(payload: dict[str, Any]) -> dict[str, Any]:
        context_id = str(payload.pop("_context_id"))
        create_context = bool(payload.pop("_create_context"))
        request = ChatRequest.model_validate(payload)
        result = app.state.main_loop.chat(
            request.text,
            debug=False,
            attachments=attachment_metadata(request),
            context_id=context_id,
            source_channel="api.chat",
            source_session_id=request.client_session_id,
            interlocutor_key=request.interlocutor_key,
            create_context=create_context,
            origin_actor=OriginActor.USER,
        )
        record_appraisal(app.state.runtime_event_log, result, debug=False)
        return chat_response_from_result(result).model_dump(mode="json")

    def observe_completed(result: dict[str, Any]) -> None:
        model = result.get("model")
        if isinstance(model, dict) and model.get("fallback_used") is True:
            app.state.runtime_event_log.record(
                category="model",
                event_type="fallback_used",
                message="Chat response used the fallback model",
                metadata={"model_id": model.get("model_id"), "debug": False},
            )

    journal = getattr(app.state, "event_journal", None)
    journal_records = [] if journal is None else journal.verify()
    committed_event_ids = (
        set()
        if journal is None
        else {
            record.event_id
            for record in journal_records
            if record.lifecycle == JournalLifecycle.COMPLETED
            or (
                record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and record.failure_category == "committed_before_crash"
            )
        }
    )
    latest_chat_records = {
        record.event_id: record
        for record in journal_records
        if record.source == "api.chat.job"
    }
    required_event_ids = {
        event_id
        for event_id, record in latest_chat_records.items()
        if record.lifecycle == JournalLifecycle.ACCEPTED
        or (
            record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
            and record.failure_category == "accepted_not_started"
        )
    }
    return ChatJobRegistry(
        registry_path,
        app.state.agent_runtime,
        execute,
        replay_limit=settings.chat_stream_replay_limit,
        timeout_seconds=settings.chat_timeout_seconds,
        completion_observer=observe_completed,
        committed_event_ids=committed_event_ids,
        required_event_ids=required_event_ids,
    )


def get_chat_job_registry(request: Request) -> ChatJobRegistry:
    registry = getattr(request.app.state, "chat_job_registry", None)
    if registry is not None:
        return registry
    with _registry_lock:
        registry = getattr(request.app.state, "chat_job_registry", None)
        if registry is None:
            registry = create_chat_job_registry(request.app)
            request.app.state.chat_job_registry = registry
    return registry


def _required_job(registry: ChatJobRegistry, operation_id: str):
    record = registry.get(operation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Chat job not found")
    return record


def _wait_for_terminal(registry: ChatJobRegistry, operation_id: str):
    cursor = 0
    while True:
        record = _required_job(registry, operation_id)
        if record.status.status in {
            OperationState.COMPLETED,
            OperationState.FAILED,
            OperationState.CANCELED,
        }:
            return record
        events = registry.wait_for_events(operation_id, cursor, 1.0)
        if events:
            cursor = events[-1].event_id


def _request_client(request: Request) -> str:
    return "unknown" if request.client is None else request.client.host


def _sse(event: ChatStreamEvent) -> str:
    payload = json.dumps(event.data, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.event}\ndata: {payload}\n\n"


def chat_response_from_result(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        context_id=result.context_id,
        episode_id=result.episode_id,
        experience_id=result.experience_id,
        response=result.response,
        emotion=EmotionSchema(
            valence=result.valence,
            arousal=result.arousal,
            optimal_loss=result.optimal_loss,
        ),
        model=ModelSchema(
            model_id=result.model_id,
            adapter_id=result.adapter_id,
            adapter_hash=result.adapter_hash,
            activation_sequence=result.activation_sequence,
            fallback_used=result.fallback_used,
        ),
    )


def record_appraisal(
    event_log: RuntimeEventLog, result: ChatResult, *, debug: bool
) -> None:
    event_log.record(
        category="appraisal",
        event_type="evaluated"
        if result.loss_measurement.valid
        else "measurement_invalid",
        message="Cognitive appraisal completed",
        metadata={
            "model_key": result.loss_measurement.model_key,
            "measurement_valid": result.loss_measurement.valid,
            "invalid_reason": result.loss_measurement.invalid_reason,
            "novelty": result.appraisal.novelty,
            "goal_progress": result.appraisal.goal_progress,
            "threat": result.appraisal.threat,
            "controllability": result.appraisal.controllability,
            "certainty": result.appraisal.certainty,
            "social_relevance": result.appraisal.social_relevance,
            "effort_cost": result.appraisal.effort_cost,
            "reasons": list(result.emotion_update.reasons),
            "debug": debug,
        },
    )


def attachment_metadata(request: ChatRequest) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in attachment.model_dump(
                include={"type", "name", "url", "content_type"}
            ).items()
            if value is not None
        }
        for attachment in request.attachments
    ]
