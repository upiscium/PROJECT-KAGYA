"""FastAPI dependency wiring."""

from hmac import compare_digest
import os

from fastapi import Header, HTTPException, Request, status

from kagya.config import Settings
from kagya.learning import AdapterRegistry, SleepCycleManager
from kagya.memory import DualMemorySystem
from kagya.models import ModelProvider
from kagya.runtime import AgentRuntime, KagyaMainLoop


def get_api_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise RuntimeError("API settings are not initialized")
    return settings


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
        raise RuntimeError("model provider is not initialized")
    return provider


def get_memory_system(request: Request) -> DualMemorySystem:
    memory = getattr(request.app.state, "memory_system", None)
    if memory is None:
        raise RuntimeError("memory system is not initialized")
    return memory


def get_adapter_registry(request: Request) -> AdapterRegistry:
    registry = getattr(request.app.state, "adapter_registry", None)
    if registry is None:
        raise RuntimeError("adapter registry is not initialized")
    return registry


def get_main_loop(request: Request) -> KagyaMainLoop:
    main_loop = getattr(request.app.state, "main_loop", None)
    if main_loop is None:
        raise RuntimeError("main loop is not initialized")
    return main_loop


def get_sleep_cycle_manager(request: Request) -> SleepCycleManager:
    manager = getattr(request.app.state, "sleep_cycle_manager", None)
    if manager is None:
        raise RuntimeError("sleep cycle manager is not initialized")
    return manager


def get_agent_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise RuntimeError("agent runtime is not initialized")
    return runtime
