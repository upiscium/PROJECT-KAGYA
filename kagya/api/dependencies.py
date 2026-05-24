"""FastAPI dependency wiring."""

from fastapi import Request

from kagya.config import Settings, get_settings
from kagya.learning import AdapterRegistry, SleepCycleManager
from kagya.memory import DualMemorySystem
from kagya.models import ModelProvider, load_model_provider
from kagya.runtime import KagyaMainLoop


def get_api_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def get_model_provider(request: Request) -> ModelProvider:
    provider = getattr(request.app.state, "model_provider", None)
    if provider is None:
        provider = load_model_provider(get_api_settings(request))
        request.app.state.model_provider = provider
    return provider


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


def get_main_loop(request: Request) -> KagyaMainLoop:
    main_loop = getattr(request.app.state, "main_loop", None)
    if main_loop is None:
        main_loop = KagyaMainLoop(
            get_api_settings(request),
            get_model_provider(request),
            get_memory_system(request),
        )
        request.app.state.main_loop = main_loop
    return main_loop


def get_sleep_cycle_manager(request: Request) -> SleepCycleManager:
    return SleepCycleManager(
        get_api_settings(request),
        get_memory_system(request),
        get_model_provider(request),
        get_adapter_registry(request),
    )
