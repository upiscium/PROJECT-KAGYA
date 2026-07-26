"""Isolated deterministic construction of the real subject runtime graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
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
    RuntimeBehaviorClassifier,
    RuntimeBehaviorObservation,
    StateTransition,
    TransitionKind,
)
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider, ModelProvider
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
from kagya.structured_response import PublicBehaviorClass


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

    def queue_responses(self, *responses: str) -> None:
        self._responses.extend(responses)


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
        self,
        visible_response: str | None = None,
        *,
        declared_behavior: PublicBehaviorClass | None = None,
        parse_valid: bool | None = None,
        parse_status: str | None = None,
        runtime_state_behavior: PublicBehaviorClass | None = None,
        duplicate_retry: bool = False,
        duplicate_tool_calls: int = 0,
        duplicate_receipts: int = 0,
        payload: dict[str, object] | None = None,
    ) -> BehavioralTrace:
        after = self.harness.capture_authoritative_state()
        records = self.harness.journal_records()
        wal_records = self.harness.wal_records()
        before_intents = _list_at(self.before, ("domains", "actions", "intents"))
        after_intents = _list_at(after, ("domains", "actions", "intents"))
        before_receipts = _list_at(self.before, ("domains", "actions", "receipts"))
        after_receipts = _list_at(after, ("domains", "actions", "receipts"))
        transitions = tuple(
            [
                *_domain_record_transitions(self.before, after),
                *_diff_transitions(
                    {
                        key: value
                        for key, value in self.before.items()
                        if key != "domains"
                    },
                    {key: value for key, value in after.items() if key != "domains"},
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
        behavior = RuntimeBehaviorClassifier().classify(
            RuntimeBehaviorObservation(
                visible_response=visible_response,
                declared_behavior=declared_behavior,
                parse_valid=parse_valid,
                runtime_state_behavior=runtime_state_behavior,
                before_authoritative_state=self.before,
                after_authoritative_state=after,
                new_action_intents=max(0, len(after_intents) - len(before_intents)),
                new_external_effects=max(0, len(self.harness.tool_environment.calls)),
                duplicate_retry=duplicate_retry,
                tool_call_count=(
                    duplicate_tool_calls
                    if duplicate_retry
                    else len(self.harness.tool_environment.calls)
                ),
                receipt_count=(
                    duplicate_receipts
                    if duplicate_retry
                    else max(0, len(after_receipts) - len(before_receipts))
                ),
            )
        )
        return BehavioralTrace(
            final_authoritative_state=after,
            transitions=transitions,
            public_behavior=behavior,
            declared_public_behavior=declared_behavior,
            behavior_parse_valid=parse_valid,
            behavior_parse_status=parse_status,
            public_payload=cast(
                dict[str, JsonValue],
                _json_value(
                    payload
                    or {
                        "visible_response": visible_response or "",
                        "classified_behavior": behavior.value,
                    }
                ),
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
        provider: ModelProvider | None = None,
        tool_environment: ControlledToolEnvironment | None = None,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        self.root = root.resolve()
        self.settings_template = settings
        self.subject_id = subject_id
        self.clock = clock or ControlledClock(datetime(2026, 7, 24, tzinfo=UTC))
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
        memory.set_external_boundary_injector(self.failure_injector)
        external = ExternalTransactionCoordinator([memory])
        external.reconcile(cast(Any, journal.verify()))
        loop = KagyaMainLoop(
            settings,
            self.provider,
            memory,
            adapter_id=getattr(self.provider, "runtime_adapter_id", None),
            adapter_hash=getattr(self.provider, "runtime_adapter_hash", None),
            activation_sequence=getattr(
                self.provider, "runtime_activation_sequence", None
            ),
            clock=self.clock.now,
        )
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
            self.failure_injector("before_finalize")
            self.failure_injector("before_external_finalize")
            external.finalize_event(event.event_id, event.processing_sequence)
            self.failure_injector("after_finalize")
            self.failure_injector("after_external_finalize")
            return hash_snapshot(saved)

        def fail(event: AgentEvent, exception: Exception) -> str:
            previous = store.last_snapshot
            if previous is None or event.processing_sequence is None:
                raise RuntimeError("Runtime failure lacks snapshot continuity")
            if previous.last_processed_event_sequence == event.processing_sequence:
                return hash_snapshot(previous)
            external.orphan_event(event.event_id, type(exception).__name__)
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

    def abrupt_stop(self) -> None:
        """Terminate threads without draining or committing active runtime work."""

        graph = self.graph
        if graph is not None:
            graph.runtime.abort()
            graph.autonomy_loop.shutdown()
        self.graph = None
        self.readiness = False

    def crash(self) -> None:
        self.abrupt_stop()

    def restart(self) -> "SubjectRuntimeHarness":
        old_graph = self.graph
        if old_graph is not None:
            old_graph.autonomy_loop.shutdown()
            old_graph.runtime.shutdown()
        self.graph = None
        self.readiness = False
        # Every graph object is reconstructed from files. The explicitly supplied
        # provider remains the evaluated subject across the restart boundary.
        self.create()
        if self.readiness:
            self.start()
        if old_graph is not None and self.graph is old_graph:
            raise AssertionError("Runtime restart reused its object graph")
        return self

    def capture_trace(
        self,
        collector: AuthoritativeTransitionCollector,
        visible_response: str | None = None,
        *,
        declared_behavior: PublicBehaviorClass | None = None,
        parse_valid: bool | None = None,
        parse_status: str | None = None,
        runtime_state_behavior: PublicBehaviorClass | None = None,
        duplicate_retry: bool = False,
        duplicate_tool_calls: int = 0,
        duplicate_receipts: int = 0,
        payload: dict[str, object] | None = None,
    ) -> BehavioralTrace:
        return collector.capture(
            visible_response,
            declared_behavior=declared_behavior,
            parse_valid=parse_valid,
            parse_status=parse_status,
            runtime_state_behavior=runtime_state_behavior,
            duplicate_retry=duplicate_retry,
            duplicate_tool_calls=duplicate_tool_calls,
            duplicate_receipts=duplicate_receipts,
            payload=payload,
        )

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
        identity = state.get("identity")
        if isinstance(identity, dict):
            extensions = identity.get("extensions")
            if isinstance(extensions, dict) and "identity_boundary" in extensions:
                extensions["identity_boundary"] = (
                    graph.main_loop.identity_boundary_store.public_json()
                )
        state["domains"] = {
            "attention": _json_value(graph.main_loop.attention_system.to_json()),
            "experiences": _model_values(graph.main_loop.experience_store),
            "motivations": _model_values(graph.main_loop.motivation_dynamics),
            "goals": _model_values(graph.main_loop.goal_manager),
            "commitments": _json_value(
                graph.main_loop.commitment_store.list_commitments()
            ),
            "values": _json_value(graph.main_loop.value_system.list_values()),
            "beliefs": _model_values(graph.main_loop.belief_store),
            "plans": _model_values(graph.main_loop.plan_store),
            "decisions": _without_argument_bodies(
                _model_values(graph.main_loop.decision_store)
            ),
            "actions": {
                "intents": [
                    _action_intent_evidence(item)
                    for item in graph.action_execution.list_intents()
                ],
                "validation_records": _json_value(
                    graph.action_execution.list_validation_records()
                ),
                "receipts": _json_value(graph.action_execution.list_receipts()),
                "observations": _json_value(graph.action_execution.list_observations()),
                "verifications": _json_value(
                    graph.action_execution.list_verifications()
                ),
            },
            "agency": _model_values(graph.main_loop.agency_attribution_store),
            "counterfactuals": _model_values(graph.main_loop.counterfactual_store),
            "outbox": _model_values(graph.outbox),
            "relationships": _model_values(graph.main_loop.relationship_store),
            "narrative": _model_values(graph.main_loop.narrative_self),
            "metacognition": _model_values(graph.main_loop.metacognition),
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
        for validation in graph.action_execution.list_validation_records():
            refs.append(f"action-validation:{validation.validation_id}")
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
        validations = {
            item.validation_id: item
            for item in graph.action_execution.list_validation_records()
        }
        accepted = tuple(
            ActionAttempt(
                tool_name=item.tool_name,
                risk_class=item.risk_class.value,
                arguments_valid=validations[item.validation_record_id].arguments_valid,
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
            if item.validation_record_id in validations
        )
        rejected = tuple(
            ActionAttempt(
                tool_name=item.tool_name,
                risk_class="read_only"
                if item.risk_class is None
                else item.risk_class.value,
                arguments_valid=item.arguments_valid,
                policy_allowed=False,
                approval_required=False,
                approved=False,
                executed=False,
            )
            for item in validations.values()
            if item.intent_id is None
        )
        return (*accepted, *rejected)

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


def _action_intent_evidence(intent: Any) -> JsonValue:
    value = intent.model_dump(mode="json")
    value.pop("arguments", None)
    preview = value.get("preview")
    if isinstance(preview, dict):
        preview.pop("arguments", None)
    return cast(JsonValue, value)


def _without_argument_bodies(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_argument_bodies(item)
            for key, item in value.items()
            if key != "arguments"
        }
    if isinstance(value, list):
        return [_without_argument_bodies(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_argument_bodies(item) for item in value)
    return value


def _model_values(value: Any) -> Any:
    for method_name in (
        "list",
        "list_records",
        "list_values",
        "list_goals",
        "list_commitments",
        "list_plans",
        "list_current",
        "list_relationships",
        "list_assessments",
        "list_intents",
        "list_messages",
        "to_json",
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
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
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
                    )
                )
            elif key not in after:
                result.append(
                    StateTransition(
                        path=child,
                        kind=TransitionKind.REMOVE,
                        before=before[key],
                    )
                )
            else:
                result.extend(
                    _diff_transitions(
                        before[key],
                        after[key],
                        path=child,
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
    return [
        StateTransition(
            path=path or ("root",),
            kind=kind,
            before=before,
            after=after,
        )
    ]


def _domain_record_transitions(
    before_state: dict[str, Any], after_state: dict[str, Any]
) -> list[StateTransition]:
    before_domains = before_state.get("domains", {})
    after_domains = after_state.get("domains", {})
    if not isinstance(before_domains, dict) or not isinstance(after_domains, dict):
        return []
    transitions: list[StateTransition] = []
    for domain in sorted(set(before_domains) | set(after_domains)):
        transitions.extend(
            _record_container_transitions(
                before_domains.get(domain),
                after_domains.get(domain),
                ("domains", str(domain)),
            )
        )
    return transitions


def _record_container_transitions(
    before: Any, after: Any, path: tuple[str, ...]
) -> list[StateTransition]:
    if isinstance(before, dict) and isinstance(after, dict):
        result: list[StateTransition] = []
        for key in sorted(set(before) | set(after)):
            result.extend(
                _record_container_transitions(
                    before.get(key), after.get(key), (*path, str(key))
                )
            )
        return result
    if isinstance(before, list) and isinstance(after, list):
        before_records = {
            identifier: item
            for item in before
            if isinstance(item, dict)
            and (identifier := _record_identifier(item)) is not None
        }
        after_records = {
            identifier: item
            for item in after
            if isinstance(item, dict)
            and (identifier := _record_identifier(item)) is not None
        }
        if before_records or after_records:
            result = []
            for identifier in sorted(set(before_records) | set(after_records)):
                previous = before_records.get(identifier)
                current = after_records.get(identifier)
                if previous == current:
                    continue
                record = current if current is not None else previous
                if record is None:
                    continue
                result.append(
                    StateTransition(
                        path=(*path, identifier),
                        kind=(
                            TransitionKind.CREATE
                            if previous is None
                            else TransitionKind.REMOVE
                            if current is None
                            else TransitionKind.UPDATE
                        ),
                        before=previous,
                        after=current,
                        evidence_refs=_record_evidence(record),
                        event_id=_record_event_id(record),
                        event_sequence=_record_event_sequence(record),
                        revision_before=_record_revision(previous),
                        revision_after=_record_revision(current),
                    )
                )
                result.extend(
                    _nested_revision_transitions(previous, current, (*path, identifier))
                )
            return result
    return _diff_transitions(before, after, path=path)


def _nested_revision_transitions(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    path: tuple[str, ...],
) -> list[StateTransition]:
    if after is None:
        return []
    result: list[StateTransition] = []
    previous = before or {}
    for key, value in after.items():
        old_value = previous.get(key)
        if key in {"revisions", "transitions", "deliberations"} and isinstance(
            value, list
        ):
            result.extend(
                _record_container_transitions(old_value or [], value, (*path, key))
            )
    return result


def _record_identifier(record: dict[str, Any]) -> str | None:
    for key in (
        "experience_id",
        "motivation_id",
        "goal_id",
        "commitment_id",
        "plan_id",
        "decision_id",
        "intent_id",
        "receipt_id",
        "observation_id",
        "verification_id",
        "attribution_id",
        "simulation_id",
        "message_id",
        "value_id",
        "belief_id",
        "relationship_id",
        "assessment_id",
        "revision_id",
        "transition_id",
        "deliberation_id",
        "candidate_id",
        "history_id",
        "link_id",
        "episode_id",
        "chapter_id",
        "claim_id",
        "hypothesis_id",
    ):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    revision = record.get("revision")
    return f"revision-{revision}" if isinstance(revision, int) else None


def _record_revision(record: dict[str, Any] | None) -> int | None:
    if record is None:
        return None
    value = record.get("revision", record.get("to_revision"))
    return value if isinstance(value, int) else None


def _record_evidence(record: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (
            key.endswith("_ref")
            or key.endswith("_refs")
            or key
            in {
                "event_id",
                "source_event_id",
                "decision_id",
                "intent_id",
                "receipt_id",
                "observation_id",
                "verification_id",
            }
        ):
            values.append(value)

    visit(record)
    return tuple(dict.fromkeys(values))


def _record_event_id(record: dict[str, Any]) -> str | None:
    candidates: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in {
            "event_id",
            "source_event_id",
            "triggering_event_id",
            "proposal_event_id",
        }:
            candidates.append(value)

    visit(record)
    return candidates[-1] if candidates else None


def _record_event_sequence(record: dict[str, Any]) -> int | None:
    values: list[int] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, int) and key.endswith("event_sequence"):
            values.append(value)

    visit(record)
    return max(values) if values else None


def _list_at(value: dict[str, Any], path: tuple[str, ...]) -> list[Any]:
    current: Any = value
    for item in path:
        if not isinstance(current, dict):
            return []
        current = current.get(item)
    return current if isinstance(current, list) else []
