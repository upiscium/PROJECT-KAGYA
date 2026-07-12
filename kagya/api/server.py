"""FastAPI startup foundation for PROJECT-KAGYA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kagya.api.routes import adapters, chat, debug, evaluations, memory, sleep, system
from kagya.config import Settings, get_settings
from kagya.memory import DualMemorySystem
from kagya.models import load_model_provider


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

    return app


def _lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _preload_runtime(app, settings)
        yield

    return lifespan


def _preload_runtime(app: FastAPI, settings: Settings) -> None:
    if getattr(app.state, "memory_system", None) is None:
        app.state.memory_system = DualMemorySystem(settings)
    if (
        settings.model.provider.lower() == "transformers"
        and getattr(app.state, "model_provider", None) is None
    ):
        provider = load_model_provider(settings)
        get_processor = getattr(provider, "get_processor", None)
        get_model = getattr(provider, "get_model", None)
        if callable(get_processor):
            get_processor()
        if callable(get_model):
            get_model()
        app.state.model_provider = provider


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
