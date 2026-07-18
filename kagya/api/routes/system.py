"""Operator-safe system metadata routes."""

from importlib.metadata import PackageNotFoundError, version
import os
import resource
import subprocess

from fastapi import APIRouter, Depends, Query, Request, Response

from kagya.api.dependencies import (
    get_api_settings,
    get_operational_telemetry,
    get_runtime_event_log,
    require_admin,
)
from kagya.api.observability import OperationalTelemetry, RuntimeEvent, RuntimeEventLog
from kagya.api.schemas.system import (
    BuildInfoSchema,
    JournalRecordListResponse,
    JournalRecordSchema,
    RuntimeEventListResponse,
    RuntimeEventSchema,
    RuntimeInfoSchema,
    SystemInfoResponse,
)
from kagya.config import Settings
from kagya.runtime import EventJournal, JournalRecord
from kagya.tools import ToolAuditEvent, ToolAuditLog


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/info", response_model=SystemInfoResponse)
def system_info(settings: Settings = Depends(get_api_settings)) -> SystemInfoResponse:
    """Return public-safe deployment metadata for operators."""

    return SystemInfoResponse(
        project=settings.project.name,
        status="ok",
        build=BuildInfoSchema(
            version=_package_version(),
            commit=_build_commit(),
        ),
        runtime=RuntimeInfoSchema(
            environment=settings.project.environment,
            provider=settings.model.provider,
            primary_model_id=settings.model.primary_id,
            fallback_configured=bool(settings.model.fallback_id),
            transformers_4bit=settings.model.load_in_4bit,
            qlora_dry_run=settings.qlora.dry_run,
            admin_token_configured=bool(os.getenv(settings.api.admin_token_env)),
        ),
    )


@router.get(
    "/events",
    response_model=RuntimeEventListResponse,
    dependencies=[Depends(require_admin)],
)
def runtime_events(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    settings: Settings = Depends(get_api_settings),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> RuntimeEventListResponse:
    """Return recent operator-visible runtime and lifecycle events."""

    events = [event_schema(event) for event in event_log.recent(limit)]
    events.extend(_tool_audit_event_schemas(request, settings, limit))
    return RuntimeEventListResponse(events=events[-limit:])


@router.get(
    "/journal",
    response_model=JournalRecordListResponse,
    dependencies=[Depends(require_admin)],
)
def event_journal(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
) -> JournalRecordListResponse:
    """Return durable operator-safe event lifecycle records."""

    journal: EventJournal = request.app.state.event_journal
    return JournalRecordListResponse(
        records=[journal_record_schema(record) for record in journal.recent(limit)]
    )


@router.get("/metrics", dependencies=[Depends(require_admin)])
def operational_metrics(
    telemetry: OperationalTelemetry = Depends(get_operational_telemetry),
) -> Response:
    """Export bounded operational metrics in Prometheus text format."""

    _observe_resources(telemetry)
    return Response(
        content=telemetry.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/telemetry", dependencies=[Depends(require_admin)])
def otlp_telemetry(
    telemetry: OperationalTelemetry = Depends(get_operational_telemetry),
) -> dict:
    """Export dependency-free OpenTelemetry Protocol JSON-compatible data."""

    _observe_resources(telemetry)
    return telemetry.otlp_json()


@router.get("/traces", dependencies=[Depends(require_admin)])
def operational_traces(
    limit: int = Query(default=100, ge=1, le=1000),
    event_id: str | None = Query(default=None, max_length=128),
    telemetry: OperationalTelemetry = Depends(get_operational_telemetry),
) -> dict[str, object]:
    """Find safe causal spans, optionally by authoritative event ID."""

    return {
        "traces": [
            record.__dict__
            for record in telemetry.recent_traces(limit, event_id=event_id)
        ]
    }


def _package_version() -> str:
    try:
        return version("project-kagya")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _build_commit() -> str | None:
    for env_name in ("KAGYA_BUILD_COMMIT", "GIT_COMMIT"):
        value = os.getenv(env_name)
        if value:
            return value[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def event_schema(event: RuntimeEvent) -> RuntimeEventSchema:
    return RuntimeEventSchema(
        id=event.id,
        timestamp=event.timestamp,
        category=event.category,
        event_type=event.event_type,
        message=event.message,
        metadata=event.metadata,
    )


def journal_record_schema(record: JournalRecord) -> JournalRecordSchema:
    return JournalRecordSchema(
        record_id=record.record_id,
        timestamp=record.timestamp.isoformat(),
        lifecycle=record.lifecycle.value,
        event_id=record.event_id,
        event_type=record.event_type,
        source=record.source,
        processing_sequence=record.processing_sequence,
        snapshot_sequence=record.snapshot_sequence,
        causation_id=record.causation_id,
        correlation_id=record.correlation_id,
        state_hash_before=record.state_hash_before,
        state_hash_after=record.state_hash_after,
        snapshot_hash=record.snapshot_hash,
        failure_category=record.failure_category,
        actor_id=record.actor_id,
        actor_role=record.actor_role,
        target=record.target,
        reauthenticated=record.reauthenticated,
        previous_record_hash=record.previous_record_hash,
        record_hash=record.record_hash,
    )


def _tool_audit_event_schemas(
    request: Request, settings: Settings, limit: int
) -> list[RuntimeEventSchema]:
    audit_events = ToolAuditLog(settings.tools.audit_path).recent(limit)
    if not audit_events:
        audit_events = _in_memory_tool_audit_events(request)
    return [
        _tool_audit_event_schema(index, audit_event)
        for index, audit_event in enumerate(audit_events[-limit:], start=1)
    ]


def _in_memory_tool_audit_events(request: Request) -> list[ToolAuditEvent]:
    executor = getattr(request.app.state, "tool_executor", None)
    return list(getattr(executor, "audit_log", []))


def _tool_audit_event_schema(
    index: int, audit_event: ToolAuditEvent
) -> RuntimeEventSchema:
    return RuntimeEventSchema(
        id=-index,
        timestamp=audit_event.timestamp,
        category="tool",
        event_type="executed" if audit_event.executed else "blocked",
        message="Tool execution audit event",
        metadata={
            "tool_name": audit_event.tool_name,
            "status": None if audit_event.status is None else audit_event.status.value,
            "tool_type": None
            if audit_event.tool_type is None
            else audit_event.tool_type.value,
            "reason": audit_event.reason,
        },
    )


def _observe_resources(telemetry: OperationalTelemetry) -> None:
    try:
        # Linux reports KiB; this project deploys on Linux/NixOS.
        telemetry.gauge(
            "kagya_process_resident_memory_bytes",
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        )
    except (OSError, ValueError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            telemetry.gauge(
                "kagya_accelerator_memory_bytes",
                float(torch.cuda.memory_allocated()),
                kind="allocated",
            )
            telemetry.gauge(
                "kagya_accelerator_memory_bytes",
                float(torch.cuda.memory_reserved()),
                kind="reserved",
            )
    except (ImportError, RuntimeError):
        pass
