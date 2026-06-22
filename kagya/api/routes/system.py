"""Operator-safe system metadata routes."""

from importlib.metadata import PackageNotFoundError, version
import os
import subprocess

from fastapi import APIRouter, Depends, Query, Request

from kagya.api.dependencies import (
    get_api_settings,
    get_runtime_event_log,
    require_admin,
)
from kagya.api.observability import RuntimeEvent, RuntimeEventLog
from kagya.api.schemas.system import (
    BuildInfoSchema,
    RuntimeEventListResponse,
    RuntimeEventSchema,
    RuntimeInfoSchema,
    SystemInfoResponse,
)
from kagya.config import Settings


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
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> RuntimeEventListResponse:
    """Return recent operator-visible runtime and lifecycle events."""

    events = [event_schema(event) for event in event_log.recent(limit)]
    events.extend(_tool_audit_event_schemas(request))
    return RuntimeEventListResponse(events=events[-limit:])


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


def _tool_audit_event_schemas(request: Request) -> list[RuntimeEventSchema]:
    executor = getattr(request.app.state, "tool_executor", None)
    audit_log = getattr(executor, "audit_log", [])
    events: list[RuntimeEventSchema] = []
    for index, audit_event in enumerate(audit_log, start=1):
        events.append(
            RuntimeEventSchema(
                id=-index,
                timestamp="",
                category="tool",
                event_type="executed" if audit_event.executed else "blocked",
                message="Tool execution audit event",
                metadata={
                    "tool_name": audit_event.tool_name,
                    "status": None
                    if audit_event.status is None
                    else audit_event.status.value,
                    "tool_type": None
                    if audit_event.tool_type is None
                    else audit_event.tool_type.value,
                },
            )
        )
    return events
