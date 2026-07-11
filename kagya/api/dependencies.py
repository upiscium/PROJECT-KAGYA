"""FastAPI dependency wiring."""

from hmac import compare_digest
import os

from fastapi import Header, HTTPException, Request, status

from kagya.api.observability import RuntimeEventLog
from kagya.config import Settings, get_settings
from kagya.learning import (
    AdapterEntry,
    AdapterRegistry,
    AdapterStatus,
    SleepCycleManager,
)
from kagya.memory import DualMemorySystem
from kagya.models import ModelProvider, load_model_provider
from kagya.runtime import KagyaMainLoop
from kagya.tools import ToolAuditLog, ToolExecutor, ToolRegistry


def get_api_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def get_runtime_event_log(request: Request) -> RuntimeEventLog:
    event_log = getattr(request.app.state, "runtime_event_log", None)
    if event_log is None:
        event_log = RuntimeEventLog()
        request.app.state.runtime_event_log = event_log
    return event_log


def require_admin(
    request: Request,
    x_kagya_admin_token: str | None = Header(default=None, alias="X-KAGYA-Admin-Token"),
) -> None:
    settings = get_api_settings(request)
    expected = os.getenv(settings.api.admin_token_env)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Admin token env var {settings.api.admin_token_env} is not configured",
        )
    if x_kagya_admin_token is None or not compare_digest(x_kagya_admin_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )


def get_model_provider(request: Request) -> ModelProvider:
    provider = getattr(request.app.state, "model_provider", None)
    if provider is None:
        provider = load_model_provider(get_api_settings(request))
        request.app.state.model_provider = provider
    return provider


def sync_main_loop_to_active_adapter(request: Request) -> KagyaMainLoop:
    active_adapter = _get_active_adapter(request)
    provider = _get_runtime_model_provider(request, active_adapter)
    main_loop = KagyaMainLoop(
        get_api_settings(request),
        provider,
        get_memory_system(request),
        adapter_id=None if active_adapter is None else active_adapter.adapter_id,
    )
    request.app.state.main_loop = main_loop
    return main_loop


def get_memory_system(request: Request) -> DualMemorySystem:
    memory = getattr(request.app.state, "memory_system", None)
    if memory is None:
        memory = DualMemorySystem(get_api_settings(request))
        request.app.state.memory_system = memory
    return memory


def get_adapter_registry(request: Request) -> AdapterRegistry:
    registry = getattr(request.app.state, "adapter_registry", None)
    if registry is None:
        registry = AdapterRegistry(get_api_settings(request))
        request.app.state.adapter_registry = registry
    return registry


def get_tool_registry(request: Request) -> ToolRegistry:
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        registry = ToolRegistry(get_api_settings(request).tools.path)
        request.app.state.tool_registry = registry
    return registry


def get_tool_executor(request: Request) -> ToolExecutor:
    executor = getattr(request.app.state, "tool_executor", None)
    if executor is None:
        settings = get_api_settings(request)
        executor = ToolExecutor(
            get_tool_registry(request),
            audit_log_store=ToolAuditLog(settings.tools.audit_path),
        )
        request.app.state.tool_executor = executor
    return executor


def get_main_loop(request: Request) -> KagyaMainLoop:
    main_loop = getattr(request.app.state, "main_loop", None)
    active_adapter = _get_active_adapter(request)
    active_adapter_id = None if active_adapter is None else active_adapter.adapter_id
    if main_loop is None or main_loop.adapter_id != active_adapter_id:
        main_loop = sync_main_loop_to_active_adapter(request)
    return main_loop


def get_sleep_cycle_manager(request: Request) -> SleepCycleManager:
    return SleepCycleManager(
        get_api_settings(request),
        get_memory_system(request),
        get_model_provider(request),
        get_adapter_registry(request),
    )


def _get_active_adapter(request: Request) -> AdapterEntry | None:
    return next(
        (
            entry
            for entry in get_adapter_registry(request).list()
            if entry.status == AdapterStatus.ACTIVE
        ),
        None,
    )


def _get_runtime_model_provider(
    request: Request, active_adapter: AdapterEntry | None
) -> ModelProvider:
    settings = get_api_settings(request)
    provider_adapter_id = getattr(request.app.state, "model_provider_adapter_id", None)
    if settings.model.provider.lower() == "transformers" and active_adapter is not None:
        if (
            getattr(request.app.state, "model_provider", None) is None
            or provider_adapter_id != active_adapter.adapter_id
        ):
            request.app.state.model_provider = load_model_provider(
                settings, adapter_path=active_adapter.path
            )
            request.app.state.model_provider_adapter_id = active_adapter.adapter_id
        return request.app.state.model_provider

    provider = get_model_provider(request)
    request.app.state.model_provider_adapter_id = None
    return provider
