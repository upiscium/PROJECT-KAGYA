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
from kagya.runtime import (
    AgentEvent,
    AgentRuntime,
    AgentStateLoadError,
    AgentStateSnapshot,
    AgentStateStore,
    EventJournal,
    EventJournalLease,
    KagyaMainLoop,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        existing_journal = getattr(app.state, "event_journal", None)
        journal_lease: EventJournalLease | None = None
        try:
            if existing_journal is None:
                journal_lease = EventJournalLease(app_settings.event_journal.path)
                app.state.event_journal = EventJournal(
                    app_settings.event_journal.path,
                    app_settings.event_journal.max_bytes,
                    app_settings.event_journal.retained_files,
                    lease=journal_lease,
                )
            else:
                if not existing_journal.has_exclusive_authority:
                    raise RuntimeError(
                        "Injected EventJournal has no exclusive authority"
                    )
                app.state.event_journal = existing_journal

            app.state.agent_state_store = getattr(
                app.state, "agent_state_store", None
            ) or AgentStateStore(
                app_settings.agent_state.path,
                app_settings.emotion.baseline_surprisal,
            )
            journal_has_history = app.state.event_journal.has_history
            if (
                journal_has_history
                and not app.state.agent_state_store.snapshot_exists()
            ):
                raise AgentStateLoadError(
                    "AgentState snapshot is required by EventJournal history"
                )
            snapshot = app.state.agent_state_store.load()
            snapshot_hash = app.state.agent_state_store.snapshot_hash(snapshot)
            if journal_has_history:
                recovery = app.state.event_journal.verify_and_reconcile(
                    snapshot.last_processed_event_sequence,
                    snapshot_hash,
                )
                app.state.agent_state_store.ensure_published(snapshot)
            else:
                app.state.agent_state_store.ensure_published(snapshot)
                recovery = app.state.event_journal.verify_and_reconcile(
                    snapshot.last_processed_event_sequence,
                    snapshot_hash,
                )

            app.state.model_provider = getattr(
                app.state, "model_provider", None
            ) or load_model_provider(app_settings)
            app.state.memory_system = getattr(
                app.state, "memory_system", None
            ) or DualMemorySystem(app_settings)
            app.state.adapter_registry = getattr(
                app.state, "adapter_registry", None
            ) or AdapterRegistry(app_settings)
            app.state.main_loop = getattr(
                app.state, "main_loop", None
            ) or KagyaMainLoop(
                app_settings, app.state.model_provider, app.state.memory_system
            )
            app.state.agent_state_store.restore_into(app.state.main_loop, snapshot)
            app.state.sleep_cycle_manager = getattr(
                app.state, "sleep_cycle_manager", None
            ) or SleepCycleManager(
                app_settings,
                app.state.memory_system,
                app.state.model_provider,
                app.state.adapter_registry,
            )
        except BaseException:
            if journal_lease is not None:
                journal_lease.close()
            journal = getattr(app.state, "event_journal", None)
            if journal is not None:
                journal.close()
            raise
        committed_snapshot: AgentStateSnapshot = snapshot
        committed_snapshot_hash = snapshot_hash

        def admission_checkpoint(event: AgentEvent) -> None:
            app.state.event_journal.append_accepted(event)

        def started_checkpoint(event: AgentEvent) -> None:
            app.state.event_journal.append_started(event)

        def completion_checkpoint(event: AgentEvent) -> None:
            nonlocal committed_snapshot, committed_snapshot_hash
            sequence = event.processing_sequence
            assert sequence is not None
            candidate = app.state.agent_state_store.capture(
                app.state.main_loop, sequence
            )
            candidate_hash = app.state.agent_state_store.snapshot_hash(candidate)
            app.state.event_journal.append_prepared(
                event,
                committed_snapshot_hash,
                candidate_hash,
            )
            app.state.agent_state_store.save(candidate)
            app.state.event_journal.append_completed(
                event,
                sequence,
                candidate_hash,
            )
            committed_snapshot = candidate
            committed_snapshot_hash = candidate_hash

        def failure_checkpoint(event: AgentEvent) -> None:
            app.state.agent_state_store.restore_into(
                app.state.main_loop, committed_snapshot
            )
            app.state.event_journal.append_failed(
                event,
                committed_snapshot.last_processed_event_sequence,
                committed_snapshot_hash,
            )

        try:
            if getattr(app.state, "agent_runtime", None) is None:
                app.state.agent_runtime = AgentRuntime(
                    app_settings.runtime.queue_capacity,
                    initial_sequence=recovery.processing_high_water,
                    admission_checkpoint=admission_checkpoint,
                    started_checkpoint=started_checkpoint,
                    completion_checkpoint=completion_checkpoint,
                    failure_checkpoint=failure_checkpoint,
                )
            else:
                app.state.agent_runtime.configure_durability(
                    initial_sequence=recovery.processing_high_water,
                    admission_checkpoint=admission_checkpoint,
                    started_checkpoint=started_checkpoint,
                    completion_checkpoint=completion_checkpoint,
                    failure_checkpoint=failure_checkpoint,
                )
        except BaseException:
            app.state.event_journal.close()
            raise
        try:
            app.state.agent_runtime.start()
        except BaseException:
            app.state.event_journal.close()
            raise
        try:
            yield
        finally:
            app.state.agent_runtime.shutdown()
            app.state.event_journal.close()

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
