"""FastAPI startup foundation for PROJECT-KAGYA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kagya.api.routes import adapters, chat, debug, memory, sleep
from kagya.config import Settings, get_settings
from kagya.learning import AdapterRegistry, SleepCycleManager
from kagya.memory import DualMemorySystem
from kagya.models import load_model_provider
from kagya.runtime import AgentRuntime, KagyaMainLoop


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_provider = getattr(
            app.state, "model_provider", None
        ) or load_model_provider(app_settings)
        app.state.memory_system = getattr(
            app.state, "memory_system", None
        ) or DualMemorySystem(app_settings)
        app.state.adapter_registry = getattr(
            app.state, "adapter_registry", None
        ) or AdapterRegistry(app_settings)
        app.state.main_loop = getattr(app.state, "main_loop", None) or KagyaMainLoop(
            app_settings, app.state.model_provider, app.state.memory_system
        )
        app.state.sleep_cycle_manager = getattr(
            app.state, "sleep_cycle_manager", None
        ) or SleepCycleManager(
            app_settings,
            app.state.memory_system,
            app.state.model_provider,
            app.state.adapter_registry,
        )
        app.state.agent_runtime = getattr(
            app.state, "agent_runtime", None
        ) or AgentRuntime(app_settings.runtime.queue_capacity)
        app.state.agent_runtime.start()
        try:
            yield
        finally:
            app.state.agent_runtime.shutdown()

    app = FastAPI(title=app_settings.project.name, lifespan=lifespan)
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

    return app


app = create_app()


def main() -> None:
    """Run the development API server."""

    settings = get_settings()
    uvicorn.run(
        "kagya.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
