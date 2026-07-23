"""FastAPI startup foundation for PROJECT-KAGYA."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import time

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kagya.actions import ActionExecutionLayer
from kagya.api.routes import (
    actions,
    attributions,
    adapters,
    attention,
    beliefs,
    chat,
    contexts,
    counterfactuals,
    debug,
    decisions,
    evaluations,
    experiences,
    feedback,
    goals,
    memory,
    metacognition,
    plans,
    motivation,
    narrative_self,
    relationships,
    sleep,
    self_model,
    state,
    system,
    training,
    values,
    autonomy,
    outbox,
)
from kagya.api.observability import OperationalTelemetry, RuntimeEventLog
from kagya.config import NodeRole, Settings, get_settings, validate_deployment_hostname
from kagya.decision import DecisionStatus
from kagya.learning import AdapterRegistry, AdapterStatus
from kagya.memory import DualMemorySystem
from kagya.models import load_model_provider
from kagya.motivation import GoalStatus
from kagya.outbox import Outbox
from kagya.runtime import (
    AgentRuntime,
    AgentEvent,
    AgentStateStore,
    AgentStateSnapshot,
    EmotionTimer,
    EventJournal,
    ExternalTransactionCoordinator,
    KagyaMainLoop,
    RemoteTrainingDispatcher,
    TrainingWorkerRuntime,
    StateWAL,
    StateWalIntegrityError,
    hash_snapshot,
    AutonomyLoop,
    SchedulerBudget,
    SubjectScheduler,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = Settings.model_validate((settings or get_settings()).model_dump())
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

    @app.exception_handler(RequestValidationError)
    async def validation_error_without_input(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {key: value for key, value in error.items() if key not in {"input", "ctx"}}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "project": app_settings.project.name,
            "role": app_settings.deployment.node.role.value,
        }

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        """Report only whether the API process can serve requests."""

        return {"status": "alive"}

    @app.get("/health/ready")
    def readiness(response: Response) -> dict[str, object]:
        """Report whether role-specific authoritative runtime dependencies are ready."""

        if app_settings.deployment.node.role == NodeRole.TRAINING_WORKER:
            ready = getattr(app.state, "worker_runtime", None) is not None
            checks = {"worker_runtime": ready}
        else:
            runtime = getattr(app.state, "agent_runtime", None)
            journal = getattr(app.state, "event_journal", None)
            state_wal = getattr(app.state, "state_wal", None)
            journal_ready = False
            if journal is not None:
                try:
                    journal.verify()
                    journal_ready = True
                except Exception:
                    journal_ready = False
            runtime_ready = bool(
                runtime is not None and runtime.is_alive and runtime.is_accepting
            )
            wal_ready = False
            if state_wal is not None:
                try:
                    state_wal.verify()
                    wal_ready = True
                except Exception:
                    wal_ready = False
            checks = {
                "agent_runtime": runtime_ready,
                "journal": journal_ready,
                "state_wal": wal_ready,
            }
            ready = all(checks.values())
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "ready" if ready else "not_ready", "checks": checks}

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
        app.include_router(plans.router)
        app.include_router(decisions.router)
        app.include_router(actions.router)
        app.include_router(attributions.router)
        app.include_router(counterfactuals.router)
        app.include_router(self_model.router)
        app.include_router(experiences.router)
        app.include_router(beliefs.router)
        app.include_router(motivation.router)
        app.include_router(narrative_self.router)
        app.include_router(attention.router)
        app.include_router(autonomy.router)
        app.include_router(feedback.router)
        app.include_router(metacognition.router)
        app.include_router(relationships.router)
        app.include_router(outbox.router)

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
            autonomy_loop = getattr(app.state, "autonomy_loop", None)
            if autonomy_loop is not None:
                autonomy_loop.shutdown()
                app.state.autonomy_loop = None
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
    if getattr(app.state, "operational_telemetry", None) is None:
        observability = settings.observability
        app.state.operational_telemetry = OperationalTelemetry(
            observability.metrics_path,
            observability.traces_path,
            max_series=observability.max_series,
            max_traces=observability.max_traces,
            enabled=observability.enabled,
        )
    if getattr(app.state, "agent_state_store", None) is None:
        app.state.agent_state_store = AgentStateStore(
            settings.agent_state.path, app.state.runtime_event_log
        )
    snapshot = app.state.agent_state_store.load(settings.emotion.baseline_surprisal)
    if getattr(app.state, "event_journal", None) is None:
        app.state.event_journal = EventJournal(
            settings.agent_journal.path,
            max_bytes=settings.agent_journal.max_bytes,
            retained_files=settings.agent_journal.retained_files,
            telemetry=app.state.operational_telemetry,
        )
    if getattr(app.state, "state_wal", None) is None:
        app.state.state_wal = StateWAL(settings.agent_state_wal.path)
    snapshot = _reconcile_state_wal(app, snapshot)
    app.state.event_journal.reconcile(snapshot)
    if getattr(app.state, "memory_system", None) is None:
        app.state.memory_system = DualMemorySystem(settings)
    if getattr(app.state, "external_transaction_coordinator", None) is None:
        app.state.external_transaction_coordinator = ExternalTransactionCoordinator(
            [app.state.memory_system]
        )
    app.state.external_reconciliation = (
        app.state.external_transaction_coordinator.reconcile(
            app.state.event_journal.verify()
        )
    )
    embedding_model = getattr(
        app.state.memory_system.embedding_function, "_get_model", None
    )
    if callable(embedding_model):
        embedding_model()
    if getattr(app.state, "adapter_registry", None) is None:
        app.state.adapter_registry = AdapterRegistry(settings)
    from kagya.learning import BehavioralArtifactStore

    app.state.behavioral_artifact_reconciliation = BehavioralArtifactStore(
        settings.adapter_registry.eval_result_dir
    ).reconcile(app.state.adapter_registry)
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
            adapter_hash=None
            if active_adapter is None
            else active_adapter.adapter_hash,
            activation_sequence=(
                None if active_adapter is None else active_adapter.activation_sequence
            ),
            telemetry=app.state.operational_telemetry,
        )
    app.state.agent_state_store.restore_into(app.state.main_loop, snapshot)
    outbox_settings = settings.outbox
    app.state.outbox = Outbox(
        app.state.main_loop,
        quiet_hours_start=outbox_settings.quiet_hours_start,
        quiet_hours_end=outbox_settings.quiet_hours_end,
        max_deliveries_per_hour=outbox_settings.max_deliveries_per_hour,
        event_recorder=app.state.runtime_event_log,
    )
    app.state.main_loop.outbox = app.state.outbox
    action_settings = settings.actions
    app.state.action_execution = ActionExecutionLayer(
        app.state.main_loop,
        document_root=action_settings.document_root,
        calendar_path=action_settings.calendar_path,
    )
    app.state.main_loop.action_execution = app.state.action_execution
    if getattr(app.state, "agent_runtime", None) is None:
        app.state.agent_runtime = AgentRuntime(
            queue_capacity=settings.api.agent_queue_capacity,
            event_recorder=app.state.runtime_event_log,
            event_journal=app.state.event_journal,
            initial_sequence=snapshot.last_processed_event_sequence,
            completion_hook=lambda event: _commit_subject_event(app, event),
            failure_hook=lambda event, exc: _fail_subject_event(app, event),
            telemetry=app.state.operational_telemetry,
        )
        app.state.agent_runtime.start()
    if settings.autonomy.enabled and getattr(app.state, "autonomy_loop", None) is None:
        autonomy_settings = settings.autonomy
        app.state.subject_scheduler = SubjectScheduler(
            app.state.agent_runtime,
            app.state.main_loop,
            budget=SchedulerBudget(
                max_events=autonomy_settings.max_events_per_cycle,
                max_inferences=autonomy_settings.max_inferences_per_cycle,
                max_wall_seconds=autonomy_settings.max_wall_seconds_per_cycle,
            ),
            reevaluation_interval_seconds=(
                autonomy_settings.reevaluation_interval_seconds
            ),
            telemetry=app.state.operational_telemetry,
        )
        app.state.autonomy_loop = AutonomyLoop(
            app.state.subject_scheduler,
            poll_interval_seconds=autonomy_settings.poll_interval_seconds,
        )
        app.state.autonomy_loop.start()
    if (
        settings.appraisal.timer_enabled
        and getattr(app.state, "emotion_timer", None) is None
    ):
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
    started = time.perf_counter()
    status = "failure"
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
    app.state.state_wal.append_transition(event, previous, candidate)
    try:
        saved = store.save(candidate)
        app.state.external_transaction_coordinator.finalize_event(
            event.event_id, _event_sequence(event.processing_sequence)
        )
        try:
            _observe_subject_state(app)
        except Exception:
            pass
        status = "success"
        return hash_snapshot(saved)
    finally:
        try:
            app.state.operational_telemetry.storage_observation(
                "snapshot", "save", status, time.perf_counter() - started
            )
        except Exception:
            pass


def _fail_subject_event(app: FastAPI, event: AgentEvent) -> str:
    store = app.state.agent_state_store
    app.state.external_transaction_coordinator.orphan_event(
        event.event_id, "internal_mutation_failed"
    )
    previous = store.last_snapshot
    if previous is None:
        raise RuntimeError("Agent state store has no previous snapshot")
    store.restore_into(app.state.main_loop, previous)
    candidate = previous.model_copy(
        update={
            "saved_at": datetime.now(UTC),
            "last_processed_event_sequence": _event_sequence(event.processing_sequence),
        }
    )
    app.state.event_journal.prepared(
        event,
        state_hash_before=hash_snapshot(previous),
        state_hash_after=hash_snapshot(candidate),
    )
    app.state.state_wal.append_transition(event, previous, candidate)
    saved = store.save(candidate)
    app.state.external_transaction_coordinator.compensate_event(
        event.event_id, "internal_mutation_failed"
    )
    return hash_snapshot(saved)


def _reconcile_state_wal(
    app: FastAPI, snapshot: AgentStateSnapshot
) -> AgentStateSnapshot:
    wal = app.state.state_wal
    wal.bootstrap(snapshot)
    latest = wal.reconstruct()
    snapshot_hash = hash_snapshot(snapshot)
    if latest.sequence == snapshot.last_processed_event_sequence:
        if latest.snapshot_hash != snapshot_hash:
            raise StateWalIntegrityError(
                "snapshot and private state WAL hashes disagree"
            )
        return snapshot
    if latest.sequence == snapshot.last_processed_event_sequence + 1:
        return app.state.agent_state_store.save(latest.snapshot)
    raise StateWalIntegrityError("snapshot and private state WAL sequences diverge")


def _observe_subject_state(app: FastAPI) -> None:
    loop = app.state.main_loop
    telemetry = app.state.operational_telemetry
    telemetry.gauge(
        "kagya_active_goals",
        float(len(loop.goal_manager.list_goals(status=GoalStatus.ACTIVE))),
    )
    telemetry.gauge(
        "kagya_unresolved_decisions",
        float(
            len(
                loop.decision_store.list_records(status=DecisionStatus.AWAITING_OUTCOME)
            )
        ),
    )
    telemetry.gauge("kagya_working_memory_items", float(len(loop.working_memory.items)))


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
