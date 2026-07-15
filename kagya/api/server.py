"""FastAPI startup foundation for PROJECT-KAGYA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kagya.api.routes import (
    adapters,
    chat,
    contexts,
    debug,
    evaluations,
    memory,
    sleep,
    state,
    system,
    values,
)
from kagya.api.observability import RuntimeEventLog
from kagya.config import Settings, get_settings
from kagya.learning import AdapterRegistry, AdapterStatus
from kagya.memory import DualMemorySystem
from kagya.models import load_model_provider
from kagya.runtime import AgentRuntime, AgentStateStore, EmotionTimer, KagyaMainLoop


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = settings or get_settings()
    app = FastAPI(title=app_settings.project.name, lifespan=_lifespan(app_settings))
    app.state.settings = app_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "project": app_settings.project.name}

    app.include_router(chat.router)
    app.include_router(debug.router)
    app.include_router(memory.router)
    app.include_router(sleep.router)
    app.include_router(adapters.router)
    app.include_router(evaluations.router)
    app.include_router(system.router)
    app.include_router(state.router)
    app.include_router(contexts.router)
    app.include_router(values.router)

    return app


def _lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _preload_runtime(app, settings)
        try:
            yield
        finally:
            timer = getattr(app.state, "emotion_timer", None)
            if timer is not None:
                timer.stop()
                app.state.emotion_timer = None
            app.state.agent_runtime.shutdown()
            app.state.agent_runtime = None

    return lifespan


def _preload_runtime(app: FastAPI, settings: Settings) -> None:
    if getattr(app.state, "runtime_event_log", None) is None:
        app.state.runtime_event_log = RuntimeEventLog()
    if getattr(app.state, "agent_state_store", None) is None:
        app.state.agent_state_store = AgentStateStore(
            settings.agent_state.path, app.state.runtime_event_log
        )
    snapshot = app.state.agent_state_store.load(
        settings.emotion.baseline_surprisal
    )
    if getattr(app.state, "memory_system", None) is None:
        app.state.memory_system = DualMemorySystem(settings)
    embedding_model = getattr(app.state.memory_system.embedding_function, "_get_model", None)
    if callable(embedding_model):
        embedding_model()
    if getattr(app.state, "adapter_registry", None) is None:
        app.state.adapter_registry = AdapterRegistry(settings)
    active_adapter = next(
        (
            entry
            for entry in app.state.adapter_registry.list()
            if entry.status == AdapterStatus.ACTIVE
        ),
        None,
    )
    if getattr(app.state, "model_provider", None) is None:
        adapter_path = (
            active_adapter.path
            if settings.model.provider.lower() == "transformers"
            and active_adapter is not None
            else None
        )
        provider = (
            load_model_provider(settings, adapter_path=adapter_path)
            if adapter_path is not None
            else load_model_provider(settings)
        )
        app.state.model_provider = provider
        app.state.model_provider_adapter_id = (
            None if active_adapter is None else active_adapter.adapter_id
        )
    else:
        provider = app.state.model_provider
    if settings.model.provider.lower() == "transformers":
        get_processor = getattr(provider, "get_processor", None)
        get_model = getattr(provider, "get_model", None)
        if callable(get_processor):
            get_processor()
        if callable(get_model):
            get_model()
    if getattr(app.state, "main_loop", None) is None:
        app.state.main_loop = KagyaMainLoop(
            settings,
            provider,
            app.state.memory_system,
            adapter_id=None if active_adapter is None else active_adapter.adapter_id,
        )
    app.state.agent_state_store.restore_into(app.state.main_loop, snapshot)
    if getattr(app.state, "agent_runtime", None) is None:
        app.state.agent_runtime = AgentRuntime(
            queue_capacity=settings.api.agent_queue_capacity,
            event_recorder=app.state.runtime_event_log,
            initial_sequence=snapshot.last_processed_event_sequence,
            completion_hook=lambda event: app.state.agent_state_store.save(
                app.state.agent_state_store.capture(
                    app.state.main_loop,
                    _event_sequence(event.processing_sequence),
                )
            ),
            failure_hook=lambda event, exc: app.state.agent_state_store.save_failed_sequence(
                _event_sequence(event.processing_sequence)
            ),
        )
        app.state.agent_runtime.start()
    if settings.appraisal.timer_enabled and getattr(app.state, "emotion_timer", None) is None:
        app.state.emotion_timer = EmotionTimer(
            app.state.agent_runtime,
            lambda elapsed: app.state.main_loop.advance_time(elapsed),
            interval_seconds=settings.appraisal.timer_interval_seconds,
        )
        app.state.emotion_timer.start()


def _event_sequence(sequence: int | None) -> int:
    if sequence is None:
        raise RuntimeError("Agent event has no processing sequence")
    return sequence


app = create_app()


def main() -> None:
    """Run the development API server."""

    settings = get_settings()
    runtime_app = create_app(settings)
    uvicorn.run(
        runtime_app,
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
