from dataclasses import dataclass
import time
from enum import StrEnum
from typing import Any, Callable, Generic, TypeVar

from kagya.body import EmotionState, EmotionUpdate
from kagya.cognition import (
    AppraisalResult,
    AppraisalSignals,
    LossMeasurement,
)
from kagya.identity import (
    OriginActor,
    OriginInputKind,
    new_identity_origin,
)
from kagya.experience import (
    build_chat_experience,
)
from kagya.memory import MemoryContext, MemoryLifecycleStatus
from kagya.memory import ValidationStatus
from kagya.memory.quality import assess_generation_health
from kagya.motivation import (
    ACCEPTED_COMMITMENT_STATUSES,
    GoalStatus,
)
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemoryItem,
    WorkingMemoryKind,
    WorkingMemoryView,
    working_memory_item,
)
from kagya.runtime.context import ContextFrame, InterlocutorModel
from kagya.structured_response import PublicBehaviorClass, StructuredResponseStatus
from kagya.runtime.coordinators._shared import (
    RuntimeDomainMixin,
    fallback_used as provider_fallback_used,
    generate_fallback,
)


@dataclass(frozen=True)
class ChatResult:
    episode_id: str
    experience_id: str
    response: str
    hidden_thought: str
    behavior_class: PublicBehaviorClass
    response_parse_valid: bool
    response_status: StructuredResponseStatus
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None
    adapter_hash: str | None
    activation_sequence: int | None
    fallback_used: bool
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView
    context_id: str
    loss_measurement: LossMeasurement
    appraisal: AppraisalResult
    emotion_update: EmotionUpdate


T = TypeVar("T")


class ChatStage(StrEnum):
    PREPARE_CONTEXT = "prepare_context"
    APPRAISE_AND_GENERATE = "appraise_and_generate"
    PREPARE_EXTERNAL = "prepare_external"
    INTEGRATE_DOMAINS = "integrate_domains"
    COMMIT_AUTHORITATIVE = "commit_authoritative"


class ChatRollbackScope(StrEnum):
    LOCAL_DOMAIN = "local_domain"
    EXTERNAL_SAGA = "external_saga"


@dataclass(frozen=True)
class ChatTransactionTrace:
    stages: tuple[ChatStage, ...]
    rollback_scopes: tuple[ChatRollbackScope, ...]


@dataclass(frozen=True)
class ChatTransactionCallbacks(Generic[T]):
    prepare_context: Callable[[], None]
    appraise_and_generate: Callable[[], None]
    prepare_external: Callable[[], None]
    integrate_domains: Callable[[], None]
    commit_authoritative: Callable[[], T]
    rollback_local_domain: Callable[[], None]
    rollback_external_saga: Callable[[], None]


class ChatOrchestrationCoordinator(RuntimeDomainMixin, Generic[T]):
    """Runs the local transaction; AgentRuntime owns external saga completion."""

    default_context_id: str | None
    STAGES = tuple(ChatStage)
    ROLLBACK_SCOPES = (
        ChatRollbackScope.LOCAL_DOMAIN,
        ChatRollbackScope.EXTERNAL_SAGA,
    )

    def __init__(self, execute_local_transaction: Callable[..., T]) -> None:
        self._execute_local_transaction = execute_local_transaction

    @classmethod
    def transaction_contract(cls) -> ChatTransactionTrace:
        return ChatTransactionTrace(cls.STAGES, cls.ROLLBACK_SCOPES)

    @staticmethod
    def run_transaction(callbacks: ChatTransactionCallbacks[T]) -> T:
        try:
            callbacks.prepare_context()
            callbacks.appraise_and_generate()
            callbacks.prepare_external()
            callbacks.integrate_domains()
            return callbacks.commit_authoritative()
        except Exception:
            for rollback in (
                callbacks.rollback_local_domain,
                callbacks.rollback_external_saga,
            ):
                try:
                    rollback()
                except Exception:
                    pass
            raise

    def chat(self, *args: object, **kwargs: object) -> T:
        if hasattr(self, "_execute_local_transaction"):
            return self._execute_local_transaction(*args, **kwargs)
        return self.chat_coordinator.chat(*args, **kwargs)

    def _chat_transaction(
        self,
        user_input: str,
        debug: bool = False,
        *,
        attachments: list[dict[str, Any]] | None = None,
        context_id: str | None = None,
        source_channel: str = "runtime.chat",
        source_session_id: str | None = None,
        interlocutor_key: str | None = None,
        create_context: bool = False,
        origin_actor: OriginActor = OriginActor.USER,
    ) -> ChatResult:
        previous_context_frames = self.context_registry.frames
        previous_interlocutors = self.context_registry.interlocutors
        previous_default_context_id = self.default_context_id
        previous_calibration = self.surprisal_calculator.export_history()
        previous_experience_state = self.experience_store.to_json()
        previous_relationship_state = self.relationship_store.to_json()
        previous_motivation_state = self.motivation_dynamics.to_json()
        previous_narrative_state = self.narrative_self.to_json()
        current_context = self._resolve_context(
            context_id,
            source_channel=source_channel,
            source_session_id=source_session_id,
            interlocutor_key=interlocutor_key,
            create_context=create_context,
        )
        previous_working_memory = self.working_memory.items
        previous_emotion_state = self.emotion_engine.state
        try:
            working_memory_count = len(self.working_memory.items)
            self.working_memory.advance()
            evicted = max(0, working_memory_count - len(self.working_memory.items))
            if evicted:
                self._metric(
                    "counter",
                    "kagya_working_memory_evictions_total",
                    float(evicted),
                    reason="decay",
                )
            self._sync_belief_working_memory(current_context.context_id)

            def compatibility(source_id: str | None) -> tuple[float, str]:
                return self.context_registry.compatibility(source_id, current_context)

            context_view = self.working_memory.select(
                resolver=self._resolve_working_memory,
                context_compatibility=compatibility,
                attention_refs=self.attention_system.focused_working_memory_refs(),
            )
            context_text = context_view.context_text()
            model_key = (
                f"{self.settings.model.provider}:{self.settings.model.primary_id}:"
                f"{self.adapter_id or 'base'}"
            )
            loss_measurement = self.surprisal_calculator.measure(
                context_text, user_input, model_key=model_key
            )
        except Exception:
            self.working_memory.restore(previous_working_memory)
            self.context_registry.restore(
                previous_context_frames, previous_interlocutors
            )
            self.default_context_id = previous_default_context_id
            self.surprisal_calculator.restore_history(previous_calibration)
            raise
        try:
            relationship_influence = self.relationship_store.influence(
                current_context.participant_ids
            )
            appraisal = self.appraiser.assess(
                loss_measurement,
                AppraisalSignals(
                    controllability=self.settings.appraisal.default_controllability
                    if loss_measurement.valid
                    else 0.5,
                    certainty=min(
                        self.settings.appraisal.default_certainty
                        if loss_measurement.valid
                        else 0.2,
                        relationship_influence.certainty,
                    )
                    if relationship_influence.relationship_refs
                    else (
                        self.settings.appraisal.default_certainty
                        if loss_measurement.valid
                        else 0.2
                    ),
                    threat=relationship_influence.threat,
                    social_relevance=max(
                        0.5 if current_context.participant_ids else 0.0,
                        relationship_influence.social_relevance,
                    ),
                    effort_cost=min(
                        1.0,
                        len(user_input.encode("utf-8"))
                        / self.settings.working_memory.token_capacity,
                    ),
                ),
            )
            emotion_update = self.emotion_engine.update_from_appraisal(appraisal)
            emotion_state = emotion_update.state
            retrieval_started = time.perf_counter()
            try:
                memory_context = self.memory_system.retrieve_context(
                    user_input,
                    current_context_id=current_context.context_id,
                    context_compatibility=compatibility,
                )
            finally:
                self._metric(
                    "observe",
                    "kagya_memory_retrieval_duration_seconds",
                    time.perf_counter() - retrieval_started,
                )
            for tier, records in (
                ("episodic", memory_context.db1_results),
                ("semantic", memory_context.db2_results),
            ):
                for record in records:
                    self._metric(
                        "observe",
                        "kagya_memory_relevance",
                        record.semantic_relevance,
                        tier=tier,
                    )
            event = current_agent_event()
            self._admit_runtime_context(
                memory_context, emotion_state, event, current_context
            )
            working_memory_view = self.working_memory.select(
                resolver=self._resolve_working_memory,
                context_compatibility=compatibility,
                attention_refs=self.attention_system.focused_working_memory_refs(),
            )
            prompt = self.prompt_builder.build(
                user_input,
                emotion_state,
                working_memory_view,
                current_context=current_context,
                attachments=attachments or [],
                subject_summary=self._public_subject_summary(
                    current_context.context_id, current_context.participant_ids
                ),
            )
            generation_started = time.perf_counter()
            try:
                raw_response = self.agent.generate(
                    prompt, attachments=attachments or []
                )
            except Exception as exc:
                if provider_fallback_used(self.provider):
                    raise RuntimeError("Fallback model generation failed") from exc
                raw_response = generate_fallback(self.provider, prompt)
            processed_response = self.postprocessor.process(raw_response)
            model_id = str(
                getattr(self.provider, "last_model_id", self.settings.model.primary_id)
            )
            fallback_used = bool(getattr(self.provider, "last_fallback_used", False))
            generation_duration = max(time.perf_counter() - generation_started, 1e-9)
            metric_labels = {
                "provider": self.settings.model.provider.lower(),
                "fallback": str(fallback_used).lower(),
            }
            self._metric(
                "observe",
                "kagya_generation_duration_seconds",
                generation_duration,
                **metric_labels,
            )
            approximate_tokens = max(
                1.0, len(processed_response.visible_response.encode("utf-8")) / 4.0
            )
            self._metric(
                "observe",
                "kagya_generation_tokens_per_second",
                approximate_tokens / generation_duration,
                **metric_labels,
            )
            if fallback_used:
                self._metric(
                    "counter",
                    "kagya_provider_fallback_total",
                    1.0,
                    provider=self.settings.model.provider.lower(),
                )
            generation_health = assess_generation_health(
                processed_response.visible_response,
                loss=loss_measurement.raw_loss
                if loss_measurement.raw_loss is not None
                else float("nan"),
                fallback_used=fallback_used,
            )
            if not generation_health.healthy:
                self._metric(
                    "counter",
                    "kagya_memory_quarantine_total",
                    1.0,
                    reason="health_check",
                )
            episode_id = self.memory_system.save_episodic(
                user_input,
                processed_response.visible_response,
                loss=loss_measurement.raw_loss
                if loss_measurement.raw_loss is not None
                else 0.0,
                emotion_valence=emotion_state.valence,
                emotion_arousal=emotion_state.arousal,
                generation_health=generation_health,
                source_event_id=None if event is None else event.event_id,
                source="runtime.chat" if event is None else event.source,
                processing_sequence=None
                if event is None
                else event.processing_sequence,
                causation_id=None if event is None else event.causation_id,
                correlation_id=None if event is None else event.correlation_id,
                context_id=current_context.context_id,
                source_channel=current_context.source_channel,
                source_session_id=current_context.source_session_id,
                provider=self.settings.model.provider,
                model_id=model_id,
                model_revision=str(getattr(self.provider, "model_revision", "unknown")),
                adapter_id=None if fallback_used else self.adapter_id,
                validation_status=ValidationStatus.UNVERIFIED,
                stage_external=event is not None,
            )
            experience_input = build_chat_experience(
                source_event_id=None if event is None else event.event_id,
                source_event_sequence=None
                if event is None
                else event.processing_sequence,
                episode_id=episode_id,
                identity_origin=new_identity_origin(
                    origin_actor,
                    OriginInputKind.OBSERVATION,
                    source_ref=f"context:{current_context.context_id}",
                    event_id=None if event is None else event.event_id,
                    event_sequence=None if event is None else event.processing_sequence,
                ),
                context_id=current_context.context_id,
                interlocutor_ids=current_context.participant_ids,
                appraisal=appraisal,
                valence=emotion_state.valence,
                arousal=emotion_state.arousal,
                prediction_error=(
                    loss_measurement.mean_token_loss if loss_measurement.valid else None
                ),
                value_revision_refs={
                    value.value_id: value.revision
                    for value in self.value_system.list_values()
                },
                active_goal_refs=tuple(
                    goal.goal_id
                    for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
                ),
                self_model_revision=self.self_model.state.revision,
            )
            integration = self.experience_coordinator.integrate(
                experience_input,
                active_commitment_refs=tuple(
                    f"commitment:{item.commitment_id}"
                    for item in self.commitment_store.list_commitments()
                    if item.status in ACCEPTED_COMMITMENT_STATUSES
                ),
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
            experience = integration.experience
            if integration.narrative_episode is not None:
                self._sync_self_references()
                self._persist_self_model_state()
                experience = self.link_experience_result(
                    experience.experience_id,
                    kind="self_model",
                    reference=f"self-model@{self.self_model.state.revision}",
                    evidence_refs=(f"experience:{experience.experience_id}",),
                )
            self.memory_system.link_experience(
                episode_id,
                experience_id=experience.experience_id,
                subjective_salience=experience.subjective_salience,
                autobiographical_importance=experience.autobiographical_importance,
            )
            self.context_registry.record_shared_history(
                current_context.participant_ids, f"episode:{episode_id}"
            )
            self._persist_appraisal_state()
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"episode:{episode_id}",
                    kind=WorkingMemoryKind.CONVERSATION,
                    reference=f"episode:{episode_id}",
                    source_event_id=None if event is None else event.event_id,
                    source_event_sequence=None
                    if event is None
                    else event.processing_sequence,
                    context_id=current_context.context_id,
                    source="runtime.chat" if event is None else event.source,
                    source_channel=current_context.source_channel,
                    source_session_id=current_context.source_session_id,
                    activation=0.5 + 0.5 * experience.subjective_salience,
                    salience=experience.subjective_salience,
                    retention_reason=RetentionReason.RECENT_CONTEXT,
                )
            )
            self._metric(
                "gauge",
                "kagya_working_memory_items",
                float(len(self.working_memory.items)),
            )
            self._metric(
                "gauge",
                "kagya_attention_focus_items",
                float(len(working_memory_view.selected)),
            )
            self.refresh_attention(compete=True)
        except Exception:
            self.emotion_engine.state = previous_emotion_state
            self.working_memory.restore(previous_working_memory)
            self.context_registry.restore(
                previous_context_frames, previous_interlocutors
            )
            self.default_context_id = previous_default_context_id
            self.surprisal_calculator.restore_history(previous_calibration)
            self._persist_appraisal_state()
            self.experience_store.restore(previous_experience_state)
            self._persist_experience_state()
            self.relationship_store.restore(previous_relationship_state)
            self.motivation_dynamics.restore(previous_motivation_state)
            self._persist_motivation_state()
            self.narrative_self.restore(previous_narrative_state)
            self._persist_narrative_self_state()
            self._sync_self_references()
            self._persist_self_model_state()
            raise
        return ChatResult(
            episode_id=episode_id,
            experience_id=experience.experience_id,
            response=processed_response.visible_response,
            hidden_thought=processed_response.hidden_thought if debug else "",
            behavior_class=processed_response.behavior_class,
            response_parse_valid=processed_response.parse_valid,
            response_status=processed_response.status,
            loss=loss_measurement.raw_loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            model_id=model_id,
            adapter_id=None if fallback_used else self.adapter_id,
            adapter_hash=None if fallback_used else self.adapter_hash,
            activation_sequence=(None if fallback_used else self.activation_sequence),
            fallback_used=fallback_used,
            prompt=prompt,
            memory_context=memory_context,
            working_memory_view=working_memory_view,
            context_id=current_context.context_id,
            loss_measurement=loss_measurement,
            appraisal=appraisal,
            emotion_update=emotion_update,
        )

    def _resolve_context(
        self,
        context_id: str | None,
        *,
        source_channel: str,
        source_session_id: str | None,
        interlocutor_key: str | None,
        create_context: bool,
    ) -> ContextFrame:
        identifier = context_id or self.default_context_id
        participants = () if interlocutor_key is None else (interlocutor_key,)
        if identifier is None or (
            create_context and self.context_registry.get(identifier) is None
        ):
            frame = self.context_registry.create(
                context_id=identifier,
                source_channel=source_channel,
                source_session_id=source_session_id,
                participant_ids=participants,
            )
            self.default_context_id = frame.context_id
        else:
            existing_frame = self.context_registry.get(identifier)
            if existing_frame is None:
                raise KeyError(identifier)
            frame = self.context_registry.resume(identifier)
        if (
            interlocutor_key is not None
            and self.context_registry.get_interlocutor(interlocutor_key) is None
        ):
            self.context_registry.register_interlocutor(
                InterlocutorModel(identity_key=interlocutor_key)
            )
        if interlocutor_key is not None:
            self.relationship_store.ensure_interlocutor(interlocutor_key)
        return frame

    def _admit_runtime_context(
        self,
        memory_context: MemoryContext,
        emotion_state: EmotionState,
        event: Any,
        current_context: ContextFrame,
    ) -> None:
        event_id = None if event is None else event.event_id
        event_sequence = None if event is None else event.processing_sequence
        context_id = current_context.context_id
        for record in memory_context.db1_results:
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"episode:{record.id}",
                    kind=WorkingMemoryKind.EPISODIC,
                    reference=f"episode:{record.id}",
                    source_event_id=record.source_event_id,
                    source_event_sequence=record.processing_sequence,
                    context_id=record.context_id,
                    source=record.source,
                    source_channel=record.source_channel,
                    source_session_id=record.source_session_id,
                    activation=0.7,
                    salience=0.6,
                )
            )
        for semantic_record in memory_context.db2_results:
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"semantic:{semantic_record.id}",
                    kind=WorkingMemoryKind.SEMANTIC,
                    reference=f"semantic:{semantic_record.id}",
                    source_event_id=None,
                    source_event_sequence=None,
                    context_id=semantic_record.context_id,
                    source=semantic_record.source,
                    source_channel=semantic_record.source_channel,
                    source_session_id=semantic_record.source_session_id,
                    activation=0.65,
                    salience=0.65,
                )
            )
        self.working_memory.admit(
            working_memory_item(
                item_id="emotion:current",
                kind=WorkingMemoryKind.EMOTION,
                content=(
                    f"Current valence={emotion_state.valence:.3f}, "
                    f"arousal={emotion_state.arousal:.3f}"
                ),
                source_event_id=event_id,
                source_event_sequence=event_sequence,
                context_id=context_id,
                source="runtime.emotion",
                source_channel=current_context.source_channel,
                source_session_id=current_context.source_session_id,
                activation=1.0,
                salience=max(0.5, emotion_state.arousal),
                retention_reason=RetentionReason.CURRENT_EMOTION,
            )
        )
        self._admit_structured_state(
            self.goal_manager.goals_json(),
            kind=WorkingMemoryKind.GOAL,
            reason=RetentionReason.ONGOING_GOAL,
            required_status=GoalStatus.ACTIVE.value,
        )
        self._admit_structured_state(
            self.commitment_store.to_json(),
            kind=WorkingMemoryKind.COMMITMENT,
            reason=RetentionReason.ACTIVE_COMMITMENT,
            required_status={status.value for status in ACCEPTED_COMMITMENT_STATUSES},
        )
        unresolved = self.persistent_state.working_memory_metadata.get(
            "unresolved_items", []
        )
        if isinstance(unresolved, list):
            self._admit_structured_state(
                [entry for entry in unresolved if isinstance(entry, dict)],
                kind=WorkingMemoryKind.UNRESOLVED,
                reason=RetentionReason.UNRESOLVED,
            )

    def _admit_structured_state(
        self,
        entries: list[dict[str, Any]],
        *,
        kind: WorkingMemoryKind,
        reason: RetentionReason,
        required_status: str | set[str] | None = None,
    ) -> None:
        for entry in entries:
            allowed_statuses = (
                {required_status}
                if isinstance(required_status, str)
                else required_status
            )
            if (
                allowed_statuses is not None
                and entry.get("status") not in allowed_statuses
            ):
                continue
            entry_id = entry.get("id", entry.get("goal_id", entry.get("commitment_id")))
            content = entry.get("content", entry.get("description", entry.get("text")))
            if not isinstance(entry_id, str) or not isinstance(content, str):
                continue
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"{kind.value}:{entry_id}",
                    kind=kind,
                    content=content,
                    activation=0.9,
                    salience=0.9,
                    retention_reason=reason,
                )
            )

    def _resolve_working_memory(self, item: WorkingMemoryItem) -> str | None:
        if item.reference is None:
            return item.content
        if item.reference.startswith("episode:"):
            episode = self.memory_system.get_episodic(
                item.reference.removeprefix("episode:")
            )
            if (
                episode is None
                or episode.archived
                or episode.lifecycle_status != MemoryLifecycleStatus.ACTIVE
            ):
                return None
            return (
                "Past recorded interaction (not a current fact): "
                f"User: {episode.user_input} | Assistant: {episode.response}"
            )
        if item.reference.startswith("semantic:"):
            semantic = self.memory_system.get_semantic(
                item.reference.removeprefix("semantic:")
            )
            if semantic is None or not self.memory_system.semantic_is_retrievable(
                semantic
            ):
                return None
            return f"Stored semantic record (not an adopted belief): {semantic.text}"
        return None
