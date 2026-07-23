"""Isolated deterministic construction of the real subject runtime graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from threading import Lock
from typing import Any, cast, TypeVar

from pydantic import JsonValue

from kagya.actions import ActionExecutionLayer
from kagya.config import Settings
from kagya.external_transaction import ExternalTransactionCoordinator
from kagya.learning.behavioral_evaluation import (
    BehavioralTrace,
    ActionAttempt,
    PublicBehaviorClass,
    StateTransition,
    TransitionKind,
)
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.outbox import Outbox
from kagya.runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventType,
    AgentRuntime,
    AgentStateStore,
    AutonomyLoop,
    EventJournal,
    JournalIntegrityError,
    KagyaMainLoop,
    SchedulerBudget,
    StateWAL,
    SubjectScheduler,
    hash_snapshot,
)
from kagya.runtime.agent_state import AgentStateSnapshot
from kagya.tools import ToolAuditLog, ToolExecutor, ToolRegistry


T = TypeVar("T")


class InjectedRuntimeFailure(RuntimeError):
    """Deterministic crash at a named persistence boundary."""


@dataclass
class FailureInjector:
    checkpoints: set[str] = field(default_factory=set)
    reached: list[str] = field(default_factory=list)

    def arm(self, *checkpoints: str) -> None:
        self.checkpoints.update(checkpoints)

    def clear(self) -> None:
        self.checkpoints.clear()

    def __call__(self, checkpoint: str, _reference: str | None = None) -> None:
        self.reached.append(checkpoint)
        if checkpoint in self.checkpoints:
            self.checkpoints.remove(checkpoint)
            raise InjectedRuntimeFailure(checkpoint)


class ControlledClock:
    def __init__(self, current: datetime) -> None:
        if current.tzinfo is None:
            raise ValueError("Controlled clock requires a timezone")
        self._current = current
        self._monotonic = 0.0
        self._lock = Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._current

    def monotonic(self) -> float:
        with self._lock:
            return self._monotonic

    def advance(self, seconds: float) -> datetime:
        if seconds < 0:
            raise ValueError("Controlled time cannot move backwards")
        with self._lock:
            self._current += timedelta(seconds=seconds)
            self._monotonic += seconds
            return self._current


class DeterministicRuntimeProvider(DummyProvider):
    """Distinct deterministic provider instance for each evaluated subject."""

    def __init__(self, subject_id: str, responses: tuple[str, ...] = ()) -> None:
        self.subject_id = subject_id
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return self.response_text


@dataclass
class ControlledToolEnvironment:
    outcomes: dict[str, object] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def invoke(self, tool_name: str, arguments: dict[str, object]) -> object:
        self.calls.append((tool_name, dict(arguments)))
        outcome = self.outcomes.get(tool_name)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ControlledActionExecutionLayer(ActionExecutionLayer):
    def __init__(
        self,
        *args: Any,
        environment: ControlledToolEnvironment,
        injector: FailureInjector,
        **kwargs: Any,
    ) -> None:
        self._controlled_environment = environment
        self._controlled_injector = injector
        super().__init__(*args, **kwargs)

    def _invoke(self, intent: Any, arguments: dict[str, JsonValue]) -> JsonValue:
        if intent.tool_name not in self._controlled_environment.outcomes:
            return super()._invoke(intent, arguments)
        self._controlled_injector("external_write")
        return cast(
            JsonValue,
            _json_value(
                self._controlled_environment.invoke(
                    intent.tool_name, cast(dict[str, object], arguments)
                )
            ),
        )


class _HarnessJournal(EventJournal):
    def __init__(self, *args: Any, injector: FailureInjector, **kwargs: Any) -> None:
        self._injector = injector
        super().__init__(*args, **kwargs)

    def accepted(self, event: Any) -> Any:
        value = super().accepted(event)
        self._injector("journal_accepted")
        return value

    def started(self, event: Any) -> Any:
        value = super().started(event)
        self._injector("journal_started")
        return value

    def completed(self, event: Any, snapshot_hash: str) -> Any:
        self._injector("before_journal_completed")
        return super().completed(event, snapshot_hash)


@dataclass
class RuntimeGraph:
    memory_system: DualMemorySystem
    state_store: AgentStateStore
    state_wal: StateWAL
    event_journal: EventJournal
    external_transactions: ExternalTransactionCoordinator
    main_loop: KagyaMainLoop
    runtime: AgentRuntime
    scheduler: SubjectScheduler
    autonomy_loop: AutonomyLoop
    action_execution: ActionExecutionLayer
    outbox: Outbox
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor


class AuthoritativeTransitionCollector:
    """Derive observed transitions solely from runtime persistence and state."""

    def __init__(self, harness: "SubjectRuntimeHarness") -> None:
        self.harness = harness
        self.before = harness.capture_authoritative_state()
        self._journal_record_ids = {
            record.record_id for record in harness.journal_records()
        }
        self._wal_record_ids = {record.record_id for record in harness.wal_records()}

    def capture(
        self, behavior: PublicBehaviorClass, payload: dict[str, object] | None = None
    ) -> BehavioralTrace:
        after = self.harness.capture_authoritative_state()
        records = self.harness.journal_records()
        wal_records = self.harness.wal_records()
        latest = records[-1] if records else None
        event_id = None if latest is None else latest.event_id
        sequence = None if latest is None else latest.processing_sequence
        evidence = self.harness.evidence_references()
        transitions = tuple(
            [
                *_diff_transitions(
                    self.before,
                    after,
                    evidence=evidence,
                    event_id=event_id,
                    event_sequence=sequence,
                ),
                *(
                    StateTransition(
                        path=("runtime_events", record.event_type),
                        kind=TransitionKind.APPEND,
                        after=record.processing_sequence,
                        evidence_refs=(f"journal:{record.record_hash}",),
                        event_id=record.event_id,
                        event_sequence=record.processing_sequence,
                    )
                    for record in records
                    if record.record_id not in self._journal_record_ids
                    and record.lifecycle.value == "completed"
                ),
                *(
                    StateTransition(
                        path=("durability", "wal_append"),
                        kind=TransitionKind.APPEND,
                        after=record.processing_sequence,
                        evidence_refs=(f"wal:{record.record_hash}",),
                        event_id=record.event_id,
                        event_sequence=record.processing_sequence,
                    )
                    for record in wal_records
                    if record.record_id not in self._wal_record_ids
                ),
            ]
        )
        attempts = self.harness.action_attempts()
        side_effects = self.harness.side_effect_keys()
        return BehavioralTrace(
            final_authoritative_state=after,
            transitions=transitions,
            public_behavior=behavior,
            public_payload=cast(
                dict[str, JsonValue],
                _json_value(payload or {"behavior_class": behavior.value}),
            ),
            side_effect_keys=side_effects,
            action_attempts=attempts,
        )


class SubjectRuntimeHarness:
    """Own an isolated, restartable real runtime without FastAPI globals."""

    def __init__(
        self,
        root: Path,
        settings: Settings,
        *,
        subject_id: str,
        clock: ControlledClock | None = None,
        provider: DeterministicRuntimeProvider | None = None,
        tool_environment: ControlledToolEnvironment | None = None,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        self.root = root.resolve()
        self.settings_template = settings
        self.subject_id = subject_id
        self.clock = clock or ControlledClock(datetime(2026, 7, 23, 14, tzinfo=UTC))
        self.provider = provider or DeterministicRuntimeProvider(subject_id)
        self.tool_environment = tool_environment or ControlledToolEnvironment()
        self.failure_injector = failure_injector or FailureInjector()
        self.graph: RuntimeGraph | None = None
        self.readiness = False
        self.startup_error: str | None = None

    def create(self) -> "SubjectRuntimeHarness":
        self.root.mkdir(parents=True, exist_ok=True)
        settings = _isolated_settings(self.settings_template, self.root)
        snapshot_was_corrupt = _snapshot_is_corrupt(settings.agent_state.path)
        store = AgentStateStore(
            settings.agent_state.path, failure_injector=self.failure_injector
        )
        snapshot = store.load(settings.emotion.baseline_surprisal)
        if not settings.agent_state.path.exists():
            snapshot = store.save(
                snapshot.model_copy(update={"saved_at": self.clock.now()})
            )
        wal = StateWAL(settings.agent_state_wal.path, self.failure_injector)
        wal_records = wal.verify()
        if (
            snapshot_was_corrupt
            and wal_records
            and hash_snapshot(snapshot) != wal_records[-1].state_hash_after
        ):
            snapshot = wal.reconstruct().snapshot
            store.save(snapshot)
        try:
            journal = _HarnessJournal(
                settings.agent_journal.path,
                max_bytes=settings.agent_journal.max_bytes,
                retained_files=settings.agent_journal.retained_files,
                injector=self.failure_injector,
            )
            journal.reconcile(snapshot)
        except JournalIntegrityError as exc:
            self.startup_error = str(exc)
            self.readiness = False
            return self
        if wal_records and hash_snapshot(snapshot) != wal_records[-1].state_hash_after:
            if (
                wal_records[-1].processing_sequence
                > snapshot.last_processed_event_sequence
            ):
                wal.truncate_after(snapshot.last_processed_event_sequence)
            else:
                raise RuntimeError(
                    "Snapshot and WAL hashes disagree at the same sequence"
                )
        wal.bootstrap(snapshot)
        memory = DualMemorySystem(settings, DeterministicEmbeddingFunction())
        memory.set_external_failure_injector(self.failure_injector)
        external = ExternalTransactionCoordinator([memory])
        external.reconcile(cast(Any, journal.verify()))
        loop = KagyaMainLoop(settings, self.provider, memory)
        store.restore_into(loop, snapshot)
        outbox = Outbox(
            loop,
            quiet_hours_start=settings.outbox.quiet_hours_start,
            quiet_hours_end=settings.outbox.quiet_hours_end,
            max_deliveries_per_hour=settings.outbox.max_deliveries_per_hour,
            clock=self.clock.now,
        )
        loop.outbox = outbox
        actions = _ControlledActionExecutionLayer(
            loop,
            document_root=settings.actions.document_root,
            calendar_path=settings.actions.calendar_path,
            clock=self.clock.now,
            monotonic=self.clock.monotonic,
            environment=self.tool_environment,
            injector=self.failure_injector,
        )
        loop.action_execution = actions

        def commit(event: AgentEvent) -> str:
            previous = store.last_snapshot
            if previous is None or event.processing_sequence is None:
                raise RuntimeError("Runtime commit lacks snapshot continuity")
            candidate = store.capture(loop, event.processing_sequence).model_copy(
                update={"saved_at": self.clock.now()}
            )
            journal.prepared(
                cast(Any, event),
                state_hash_before=hash_snapshot(previous),
                state_hash_after=hash_snapshot(candidate),
            )
            self.failure_injector("external_prepare")
            wal.append_transition(cast(Any, event), previous, candidate)
            saved = store.save(candidate)
            self.failure_injector("before_external_finalize")
            external.finalize_event(event.event_id, event.processing_sequence)
            self.failure_injector("after_external_finalize")
            return hash_snapshot(saved)

        def fail(event: AgentEvent, exception: Exception) -> str:
            external.orphan_event(event.event_id, type(exception).__name__)
            previous = store.last_snapshot
            if previous is None or event.processing_sequence is None:
                raise RuntimeError("Runtime failure lacks snapshot continuity")
            store.restore_into(loop, previous)
            external.compensate_event(event.event_id, type(exception).__name__)
            return hash_snapshot(previous)

        runtime = AgentRuntime(
            queue_capacity=settings.api.agent_queue_capacity,
            initial_sequence=snapshot.last_processed_event_sequence,
            completion_hook=commit,
            failure_hook=fail,
            event_journal=cast(Any, journal),
        )
        scheduler = SubjectScheduler(
            runtime,
            loop,
            budget=SchedulerBudget(
                max_events=settings.autonomy.max_events_per_cycle,
                max_inferences=settings.autonomy.max_inferences_per_cycle,
                max_wall_seconds=settings.autonomy.max_wall_seconds_per_cycle,
            ),
            reevaluation_interval_seconds=settings.autonomy.reevaluation_interval_seconds,
            clock=self.clock.now,
        )
        autonomy = AutonomyLoop(
            scheduler, poll_interval_seconds=settings.autonomy.poll_interval_seconds
        )
        tools = ToolRegistry(settings.tools.path)
        tool_executor = ToolExecutor(
            tools, audit_log_store=ToolAuditLog(settings.tools.audit_path)
        )
        self.graph = RuntimeGraph(
            memory,
            store,
            wal,
            journal,
            external,
            loop,
            runtime,
            scheduler,
            autonomy,
            actions,
            outbox,
            tools,
            tool_executor,
        )
        self.readiness = True
        self.startup_error = None
        return self

    def start(self) -> "SubjectRuntimeHarness":
        if not self.readiness or self.graph is None:
            raise RuntimeError(self.startup_error or "Runtime harness is not ready")
        self.graph.runtime.start()
        return self

    def execute(
        self,
        event_type: AgentEventType,
        handler: Callable[[KagyaMainLoop], T],
        *,
        source: str = "behavioral.runtime",
        payload: dict[str, object] | None = None,
    ) -> AgentEventOutcome[T]:
        graph = self._ready_graph()
        return graph.runtime.execute(
            event_type,
            source=source,
            handler=lambda: handler(graph.main_loop),
            payload=payload,
        )

    def seed(self, seed: dict[str, object]) -> None:
        # Seed data is validated through the persisted snapshot schema and an
        # ordinary runtime event; expectations are never accepted here.
        json.dumps(seed, allow_nan=False)

        def apply(loop: KagyaMainLoop) -> None:
            loop.persistent_state.extensions["runtime_fixture_seed"] = dict(seed)

        self.execute(AgentEventType.STATE_RESTORE, apply, payload={"seed": "validated"})

    def crash(self) -> None:
        """Abruptly discard graph ownership without a graceful runtime drain."""
        self.graph = None
        self.readiness = False

    def restart(self) -> "SubjectRuntimeHarness":
        old_graph = self.graph
        if old_graph is not None:
            old_graph.autonomy_loop.shutdown()
            old_graph.runtime.shutdown()
        self.graph = None
        self.readiness = False
        # A new provider and every graph object are reconstructed from files.
        self.provider = DeterministicRuntimeProvider(self.subject_id)
        self.create()
        if self.readiness:
            self.start()
        if old_graph is not None and self.graph is old_graph:
            raise AssertionError("Runtime restart reused its object graph")
        return self

    def capture_trace(
        self,
        collector: AuthoritativeTransitionCollector,
        behavior: PublicBehaviorClass,
        payload: dict[str, object] | None = None,
    ) -> BehavioralTrace:
        return collector.capture(behavior, payload)

    def shutdown(self) -> None:
        if self.graph is not None:
            self.graph.autonomy_loop.shutdown()
            self.graph.runtime.shutdown()
        self.readiness = False

    def capture_authoritative_state(self) -> dict[str, Any]:
        graph = self._ready_graph()
        snapshot = graph.state_store.capture(
            graph.main_loop,
            graph.state_store.last_snapshot.last_processed_event_sequence
            if graph.state_store.last_snapshot is not None
            else 0,
        )
        state = snapshot.model_dump(mode="json")
        state["domains"] = {
            "experiences": _model_values(graph.main_loop.experience_store),
            "motivations": _model_values(graph.main_loop.motivation_dynamics),
            "goals": _model_values(graph.main_loop.goal_manager),
            "plans": _model_values(graph.main_loop.plan_store),
            "decisions": _model_values(graph.main_loop.decision_store),
            "actions": {
                "intents": _json_value(graph.action_execution.list_intents()),
                "receipts": _json_value(graph.action_execution.list_receipts()),
                "observations": _json_value(graph.action_execution.list_observations()),
                "verifications": _json_value(
                    graph.action_execution.list_verifications()
                ),
            },
            "agency": _model_values(graph.main_loop.agency_attribution_store),
            "counterfactuals": _model_values(graph.main_loop.counterfactual_store),
            "outbox": _model_values(graph.outbox),
        }
        return state

    def journal_records(self) -> list[Any]:
        return self._ready_graph().event_journal.verify()

    def wal_records(self) -> list[Any]:
        return self._ready_graph().state_wal.verify()

    def evidence_references(self) -> tuple[str, ...]:
        graph = self._ready_graph()
        refs = [
            *(f"journal:{item.record_hash}" for item in graph.event_journal.verify()),
            *(f"wal:{item.record_hash}" for item in graph.state_wal.verify()),
        ]
        for intent in graph.action_execution.list_intents():
            refs.append(f"action-intent:{intent.intent_id}@{intent.revision}")
        for receipt in graph.action_execution.list_receipts():
            refs.append(f"action-receipt:{receipt.receipt_id}")
        for observation in graph.action_execution.list_observations():
            refs.append(f"action-observation:{observation.observation_id}")
        for verification in graph.action_execution.list_verifications():
            refs.append(f"action-verification:{verification.verification_id}")
        return tuple(refs)

    def side_effect_keys(self) -> tuple[str, ...]:
        graph = self._ready_graph()
        return tuple(
            item.idempotency_key
            for item in graph.action_execution.list_intents()
            if item.receipt_id is not None
        )

    def action_attempts(self) -> tuple[ActionAttempt, ...]:
        graph = self._ready_graph()
        return tuple(
            ActionAttempt(
                tool_name=item.tool_name,
                risk_class=item.risk_class.value,
                arguments_valid=True,
                policy_allowed=item.policy.allowed,
                approval_required=item.policy.approval_required,
                approved=not item.policy.approval_required
                or any(
                    approval.intent_id == item.intent_id
                    and approval.status == "approved"
                    for approval in graph.action_execution.list_approvals()
                ),
                executed=item.attempts > 0,
            )
            for item in graph.action_execution.list_intents()
        )

    def _ready_graph(self) -> RuntimeGraph:
        if not self.readiness or self.graph is None:
            raise RuntimeError(self.startup_error or "Runtime harness is not ready")
        return self.graph


def _isolated_settings(settings: Settings, root: Path) -> Settings:
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": root / "memory",
                    "embedding_model_id": "deterministic",
                }
            ),
            "agent_state": settings.agent_state.model_copy(
                update={"path": root / "agent_state.json"}
            ),
            "agent_journal": settings.agent_journal.model_copy(
                update={"path": root / "event_journal.jsonl"}
            ),
            "agent_state_wal": settings.agent_state_wal.model_copy(
                update={"path": root / "private" / "state_wal.jsonl"}
            ),
            "tools": settings.tools.model_copy(
                update={
                    "path": root / "tools.json",
                    "audit_path": root / "tool_audit.jsonl",
                }
            ),
            "actions": settings.actions.model_copy(
                update={
                    "document_root": root / "documents",
                    "calendar_path": root / "calendar.json",
                }
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": root / "adapters.json",
                    "eval_result_dir": root / "results",
                }
            ),
            "autonomy": settings.autonomy.model_copy(update={"enabled": False}),
        }
    )


def _snapshot_is_corrupt(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        AgentStateSnapshot.model_validate(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return True
    return False


def _model_values(value: Any) -> Any:
    for method_name in (
        "list",
        "list_records",
        "list_goals",
        "list_plans",
        "list_current",
        "list_intents",
        "list_messages",
    ):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _json_value(method())
            except TypeError:
                continue
    state = getattr(value, "_state", None)
    if callable(state):
        return _json_value(state())
    return []


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _diff_transitions(
    before: Any,
    after: Any,
    *,
    path: tuple[str, ...] = (),
    evidence: tuple[str, ...],
    event_id: str | None,
    event_sequence: int | None,
) -> list[StateTransition]:
    if isinstance(before, dict) and isinstance(after, dict):
        result: list[StateTransition] = []
        for key in sorted(set(before) | set(after)):
            child = (*path, str(key))
            if key not in before:
                result.append(
                    StateTransition(
                        path=child,
                        kind=TransitionKind.CREATE,
                        after=after[key],
                        evidence_refs=evidence,
                        event_id=event_id,
                        event_sequence=event_sequence,
                    )
                )
            elif key not in after:
                result.append(
                    StateTransition(
                        path=child,
                        kind=TransitionKind.REMOVE,
                        before=before[key],
                        evidence_refs=evidence,
                        event_id=event_id,
                        event_sequence=event_sequence,
                    )
                )
            else:
                result.extend(
                    _diff_transitions(
                        before[key],
                        after[key],
                        path=child,
                        evidence=evidence,
                        event_id=event_id,
                        event_sequence=event_sequence,
                    )
                )
        return result
    if before == after:
        return []
    kind = (
        TransitionKind.APPEND
        if isinstance(before, list)
        and isinstance(after, list)
        and after[: len(before)] == before
        else TransitionKind.UPDATE
    )
    revision_before = (
        before.get("revision")
        if isinstance(before, dict) and isinstance(before.get("revision"), int)
        else None
    )
    revision_after = (
        after.get("revision")
        if isinstance(after, dict) and isinstance(after.get("revision"), int)
        else None
    )
    return [
        StateTransition(
            path=path or ("root",),
            kind=kind,
            before=before,
            after=after,
            evidence_refs=evidence,
            event_id=event_id,
            event_sequence=event_sequence,
            revision_before=revision_before,
            revision_after=revision_after,
        )
    ]
