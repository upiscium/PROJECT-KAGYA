"""FastAPI startup foundation for PROJECT-KAGYA."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kagya.api.routes import adapters, chat, debug, evaluations, memory, sleep
from kagya.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = settings or get_settings()
    app = FastAPI(title=app_settings.project.name)
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
