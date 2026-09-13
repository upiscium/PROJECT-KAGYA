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
    AgentRuntimeStatus,
    AgentStateStore,
    CompatibleAgentStateSnapshot,
    EventJournal,
    EventJournalError,
    EventJournalLease,
    InternalCommitEvidence,
    KagyaMainLoop,
    StateRecoveryCoordinator,
    StateRecoveryError,
    StateRecoveryResult,
    StateWAL,
    TransactionCoordinator,
    WorkingMemory,
)
from kagya.runtime.startup_reconciliation import StartupReconciliationCoordinator


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
            app.state.state_wal = getattr(app.state, "state_wal", None) or StateWAL(
                app_settings.state_wal.directory
            )
            app.state.state_recovery = StateRecoveryCoordinator(
                app.state.agent_state_store,
                app.state.event_journal,
                app.state.state_wal,
            )
            journal_schema = app.state.event_journal.inspect().schema_version
            app.state.memory_system = getattr(
                app.state, "memory_system", None
            ) or DualMemorySystem(app_settings)
            app.state.startup_reconciliation = StartupReconciliationCoordinator(
                app.state.event_journal,
                app.state.state_recovery,
                app.state.memory_system,
            )
            app.state.startup_reconciliation.resume_prepared_gate_clear()
            participants_consistent = True
            degraded_reason = None
            if journal_schema == 3:
                participants_consistent, degraded_reason = (
                    app.state.startup_reconciliation.reconcile_open_transactions()
                )
            recovery: StateRecoveryResult | None = None
            if participants_consistent:
                recovery = app.state.state_recovery.prepare_startup()
                journal_schema = app.state.event_journal.inspect().schema_version
                if journal_schema == 2:
                    app.state.event_journal.append_v3_migration_checkpoint()
                    journal_schema = app.state.event_journal.inspect().schema_version
                if journal_schema != 3:
                    raise StateRecoveryError("Runtime requires EventJournal schema 3")
                if recovery.external_reconciliation_required:
                    reconciliation = (
                        app.state.startup_reconciliation.reconcile_recovery_gate(
                            recovery
                        )
                    )
                    recovery = reconciliation.recovery
                    participants_consistent = reconciliation.participants_consistent
                    degraded_reason = reconciliation.degraded_reason
                else:
                    app.state.startup_reconciliation.ensure_adoption_baseline(
                        recovery
                    )
                startup_state = recovery
            else:
                if journal_schema != 3:
                    raise StateRecoveryError("Degraded startup requires schema 3")
                startup_state = app.state.state_recovery.inspect_degraded_startup()
            app.state.external_reconciliation_required = (
                (recovery.external_reconciliation_required if recovery else False)
                or not participants_consistent
            )
            app.state.reconciliation_degraded_reason = degraded_reason
            retention_status = app.state.event_journal.admission_status()
            app.state.startup_retention_admission_available = retention_status.available
            app.state.startup_retention_reason = retention_status.reason
            snapshot = startup_state.snapshot
            snapshot_hash = startup_state.snapshot_hash

            app.state.model_provider = getattr(
                app.state, "model_provider", None
            ) or load_model_provider(app_settings)
            app.state.adapter_registry = getattr(
                app.state, "adapter_registry", None
            ) or AdapterRegistry(app_settings)
            injected_working_memory = getattr(app.state, "working_memory", None)
            app.state.working_memory = (
                injected_working_memory
                if injected_working_memory is not None
                else WorkingMemory(
                    item_capacity=app_settings.working_memory.item_capacity,
                    projection_max_bytes=app_settings.working_memory.projection_max_bytes,
                )
            )
            app.state.main_loop = getattr(
                app.state, "main_loop", None
            ) or KagyaMainLoop(
                app_settings,
                app.state.model_provider,
                app.state.memory_system,
                working_memory=app.state.working_memory,
            )
            app.state.working_memory = app.state.main_loop.working_memory
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
        committed_snapshot: CompatibleAgentStateSnapshot = snapshot
        committed_snapshot_hash = snapshot_hash
        app.state.transaction_coordinator = TransactionCoordinator(
            app.state.event_journal,
            app.state.state_recovery.verify_internal_commit,
        )

        def admission_checkpoint(event: AgentEvent) -> None:
            del event

        def pre_admission_guard(event: AgentEvent) -> bool:
            return app.state.event_journal.append_accepted_if_admission_available(
                event
            )

        def started_checkpoint(event: AgentEvent) -> None:
            app.state.event_journal.append_started(event)

        def preparation_checkpoint(event: AgentEvent, value: object) -> object:
            return app.state.transaction_coordinator.prepare_result(event, value)

        def internal_commit_checkpoint(event: AgentEvent) -> InternalCommitEvidence:
            nonlocal committed_snapshot, committed_snapshot_hash
            sequence = event.processing_sequence
            assert sequence is not None
            candidate = app.state.agent_state_store.capture(
                app.state.main_loop, sequence
            )
            candidate_hash = app.state.agent_state_store.snapshot_hash(candidate)
            evidence = app.state.state_recovery.commit_internal_candidate(
                event, committed_snapshot, candidate
            )
            committed_snapshot = candidate
            committed_snapshot_hash = candidate_hash
            return evidence

        def finalization_checkpoint(event: AgentEvent, evidence: object) -> None:
            if not isinstance(evidence, InternalCommitEvidence):
                raise StateRecoveryError("Internal commit evidence is unavailable")
            app.state.transaction_coordinator.finalize_event(event, evidence)

        def terminal_completion_checkpoint(event: AgentEvent, evidence: object) -> None:
            if not isinstance(evidence, InternalCommitEvidence):
                raise StateRecoveryError("Internal commit evidence is unavailable")
            app.state.state_recovery.complete_committed_event(event, evidence)

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
                    initial_sequence=startup_state.processing_high_water,
                    pre_admission_guard=pre_admission_guard,
                    admission_checkpoint=admission_checkpoint,
                    started_checkpoint=started_checkpoint,
                    preparation_checkpoint=preparation_checkpoint,
                    internal_commit_checkpoint=internal_commit_checkpoint,
                    finalization_checkpoint=finalization_checkpoint,
                    terminal_completion_checkpoint=terminal_completion_checkpoint,
                    failure_checkpoint=failure_checkpoint,
                )
            else:
                app.state.agent_runtime.configure_durability(
                    initial_sequence=startup_state.processing_high_water,
                    pre_admission_guard=pre_admission_guard,
                    admission_checkpoint=admission_checkpoint,
                    started_checkpoint=started_checkpoint,
                    preparation_checkpoint=preparation_checkpoint,
                    internal_commit_checkpoint=internal_commit_checkpoint,
                    finalization_checkpoint=finalization_checkpoint,
                    terminal_completion_checkpoint=terminal_completion_checkpoint,
                    failure_checkpoint=failure_checkpoint,
                )
        except BaseException:
            app.state.event_journal.close()
            raise
        if (
            not app.state.external_reconciliation_required
            and app.state.startup_retention_admission_available
        ):
            try:
                assert recovery is not None
                app.state.agent_runtime.start()
                app.state.state_recovery.publish_boot_anchor(recovery)
            except BaseException:
                app.state.agent_runtime.shutdown()
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
        if app.state.external_reconciliation_required:
            return {
                "status": "degraded",
                "project": app_settings.project.name,
                "reason": app.state.reconciliation_degraded_reason
                or "external_reconciliation_required",
            }
        try:
            retention_available = app.state.event_journal.admission_status().available
        except EventJournalError:
            retention_available = False
        if (
            not retention_available
            or not app.state.startup_retention_admission_available
        ):
            return {
                "status": "degraded",
                "project": app_settings.project.name,
                "reason": "event_journal_retention_exhausted",
            }
        if app.state.agent_runtime.status is not AgentRuntimeStatus.ACCEPTING:
            return {
                "status": "degraded",
                "project": app_settings.project.name,
                "reason": "agent_runtime_unavailable",
            }
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
