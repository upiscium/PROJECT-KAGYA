"""FastAPI startup foundation for PROJECT-KAGYA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kagya.api.routes import (
    adapters,
    beliefs,
    chat,
    contexts,
    debug,
    decisions,
    evaluations,
    experiences,
    goals,
    memory,
    sleep,
    self_model,
    state,
    system,
    training,
    values,
)
from kagya.api.observability import RuntimeEventLog
from kagya.config import NodeRole, Settings, get_settings, validate_deployment_hostname
from kagya.learning import AdapterRegistry, AdapterStatus
from kagya.memory import DualMemorySystem
from kagya.models import load_model_provider
from kagya.runtime import (
    AgentRuntime,
    AgentEvent,
    AgentStateStore,
    EmotionTimer,
    EventJournal,
    KagyaMainLoop,
    RemoteTrainingDispatcher,
    TrainingWorkerRuntime,
    hash_snapshot,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = settings or get_settings()
    app = FastAPI(title=app_settings.project.name, lifespan=_lifespan(app_settings))
    app.state.settings = app_settings
    app.state.node_role = app_settings.deployment.node.role
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "project": app_settings.project.name,
            "role": app_settings.deployment.node.role.value,
        }

    role = app_settings.deployment.node.role
    if role in {NodeRole.ALL, NodeRole.INFERENCE}:
        app.include_router(chat.router)
        app.include_router(debug.router)
        app.include_router(memory.router)
        app.include_router(sleep.router)
        app.include_router(adapters.router)
        app.include_router(evaluations.router)
        app.include_router(system.router)
        app.include_router(training.router)
        app.include_router(state.router)
        app.include_router(contexts.router)
        app.include_router(values.router)
        app.include_router(goals.router)
        app.include_router(goals.commitment_router)
        app.include_router(decisions.router)
        app.include_router(self_model.router)
        app.include_router(experiences.router)
        app.include_router(beliefs.router)

    return app


def _lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        validate_deployment_hostname(settings)
        if settings.deployment.node.role == NodeRole.TRAINING_WORKER:
            _preload_worker_runtime(app, settings)
        else:
            _preload_subject_runtime(app, settings)
        try:
            yield
        finally:
            coordinator = getattr(app.state, "sleep_coordinator", None)
            if coordinator is not None:
                coordinator.shutdown()
                app.state.sleep_coordinator = None
            timer = getattr(app.state, "emotion_timer", None)
            if timer is not None:
                timer.stop()
                app.state.emotion_timer = None
            runtime = getattr(app.state, "agent_runtime", None)
            if runtime is not None:
                runtime.shutdown()
                app.state.agent_runtime = None

    return lifespan


def _preload_subject_runtime(app: FastAPI, settings: Settings) -> None:
    if getattr(app.state, "runtime_event_log", None) is None:
        app.state.runtime_event_log = RuntimeEventLog()
    if getattr(app.state, "agent_state_store", None) is None:
        app.state.agent_state_store = AgentStateStore(
            settings.agent_state.path, app.state.runtime_event_log
        )
    snapshot = app.state.agent_state_store.load(
        settings.emotion.baseline_surprisal
    )
    if getattr(app.state, "event_journal", None) is None:
        app.state.event_journal = EventJournal(
            settings.agent_journal.path,
            max_bytes=settings.agent_journal.max_bytes,
            retained_files=settings.agent_journal.retained_files,
        )
    app.state.event_journal.reconcile(snapshot)
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
            adapter_hash=None if active_adapter is None else active_adapter.adapter_hash,
            activation_sequence=(
                None if active_adapter is None else active_adapter.activation_sequence
            ),
        )
    app.state.agent_state_store.restore_into(app.state.main_loop, snapshot)
    if getattr(app.state, "agent_runtime", None) is None:
        app.state.agent_runtime = AgentRuntime(
            queue_capacity=settings.api.agent_queue_capacity,
            event_recorder=app.state.runtime_event_log,
            event_journal=app.state.event_journal,
            initial_sequence=snapshot.last_processed_event_sequence,
            completion_hook=lambda event: _commit_subject_event(app, event),
            failure_hook=lambda event, exc: _fail_subject_event(app, event),
        )
        app.state.agent_runtime.start()
    if settings.appraisal.timer_enabled and getattr(app.state, "emotion_timer", None) is None:
        app.state.emotion_timer = EmotionTimer(
            app.state.agent_runtime,
            lambda elapsed: app.state.main_loop.advance_time(elapsed),
            interval_seconds=settings.appraisal.timer_interval_seconds,
        )
        app.state.emotion_timer.start()
    if settings.deployment.node.role == NodeRole.INFERENCE:
        remote = settings.deployment.training.remote_worker
        if remote is None:
            raise RuntimeError("Inference role requires remote worker settings")
        app.state.training_dispatcher = RemoteTrainingDispatcher.from_settings(remote)


def _preload_worker_runtime(app: FastAPI, settings: Settings) -> None:
    worker = settings.deployment.training.worker
    if worker is None:
        raise RuntimeError("Training worker role requires worker settings")
    app.state.worker_runtime = TrainingWorkerRuntime.from_settings(
        settings.deployment.node.id, worker
    )


def _event_sequence(sequence: int | None) -> int:
    if sequence is None:
        raise RuntimeError("Agent event has no processing sequence")
    return sequence


def _commit_subject_event(app: FastAPI, event: AgentEvent) -> str:
    store = app.state.agent_state_store
    previous = store.last_snapshot
    if previous is None:
        raise RuntimeError("Agent state store has no previous snapshot")
    candidate = store.capture(
        app.state.main_loop, _event_sequence(event.processing_sequence)
    )
    before_hash = hash_snapshot(previous)
    after_hash = hash_snapshot(candidate)
    app.state.event_journal.prepared(
        event,
        state_hash_before=before_hash,
        state_hash_after=after_hash,
    )
    saved = store.save(candidate)
    return hash_snapshot(saved)


def _fail_subject_event(app: FastAPI, event: AgentEvent) -> str:
    store = app.state.agent_state_store
    previous = store.last_snapshot
    if previous is None:
        raise RuntimeError("Agent state store has no previous snapshot")
    store.restore_into(app.state.main_loop, previous)
    saved = store.save_failed_sequence(_event_sequence(event.processing_sequence))
    if saved is None:
        raise RuntimeError("Agent failure snapshot was not saved")
    return hash_snapshot(saved)


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
