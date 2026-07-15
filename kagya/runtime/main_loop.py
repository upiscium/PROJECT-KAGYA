"""Integrated runtime main loop."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionUpdate
from kagya.cognition import (
    ActionScore,
    AppraisalResult,
    AppraisalSignals,
    CognitiveAppraiser,
    LossMeasurement,
    SurprisalCalculator,
    ValueConflictDefinition,
    ValueEvidence,
    ValueState,
    ValueSystem,
    ValueUpdateKind,
    ValueUpdateRecord,
)
from kagya.config import Settings
from kagya.memory import DualMemorySystem, MemoryContext, MemoryLifecycleStatus
from kagya.memory import ValidationStatus
from kagya.memory.quality import assess_generation_health
from kagya.models import ModelProvider
from kagya.persona import ConsciousAgent, PromptBuilder, ResponsePostprocessor
from kagya.runtime.session_state import SessionState
from kagya.runtime.agent_state import PersistentAgentState
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryKind,
    WorkingMemoryView,
    working_memory_item,
)
from kagya.runtime.context import ContextFrame, ContextRegistry, InterlocutorModel


@dataclass(frozen=True)
class ChatResult:
    episode_id: str
    response: str
    hidden_thought: str
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None
    fallback_used: bool
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView
    context_id: str
    loss_measurement: LossMeasurement
    appraisal: AppraisalResult
    emotion_update: EmotionUpdate


class KagyaMainLoop:
    """Connect prediction error, emotion, memory, generation, and storage."""

    def __init__(
        self,
        settings: Settings,
        provider: ModelProvider,
        memory_system: DualMemorySystem,
        *,
        session_state: SessionState | None = None,
        emotion_engine: EmotionEngineAllostasis | None = None,
        prompt_builder: PromptBuilder | None = None,
        agent: ConsciousAgent | None = None,
        postprocessor: ResponsePostprocessor | None = None,
        adapter_id: str | None = None,
        persistent_state: PersistentAgentState | None = None,
        working_memory: WorkingMemory | None = None,
        context_registry: ContextRegistry | None = None,
        appraiser: CognitiveAppraiser | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
        self.session_state = session_state or SessionState()
        self.surprisal_calculator = SurprisalCalculator(
            provider,
            initial_baseline=settings.emotion.baseline_surprisal,
            initial_scale=settings.appraisal.initial_loss_scale,
            minimum_scale=settings.appraisal.minimum_loss_scale,
        )
        self.emotion_engine = emotion_engine or EmotionEngineAllostasis(
            EmotionState(optimal_loss=settings.emotion.baseline_surprisal),
            adaptation_rate=settings.emotion.decay_rate,
            response_rate=settings.emotion.appraisal_response_rate,
            resting_valence=settings.emotion.resting_valence,
            resting_arousal=settings.emotion.resting_arousal,
            valence_recovery_rate=settings.emotion.valence_recovery_rate,
            arousal_recovery_rate=settings.emotion.arousal_recovery_rate,
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.agent = agent or ConsciousAgent(provider)
        self.postprocessor = postprocessor or ResponsePostprocessor()
        self.adapter_id = adapter_id
        self.persistent_state = persistent_state or PersistentAgentState()
        self.working_memory = working_memory or WorkingMemory(
            item_capacity=settings.working_memory.item_capacity,
            token_capacity=settings.working_memory.token_capacity,
        )
        self.context_registry = context_registry or ContextRegistry()
        self.appraiser = appraiser or CognitiveAppraiser()
        self.value_system = ValueSystem(
            seeds=[
                ValueState(
                    value_id=seed.value_id,
                    name=seed.name,
                    weight=seed.weight,
                    confidence=seed.confidence,
                    stability=seed.stability,
                    source=seed.source,
                    origin=seed.origin,
                    last_updated_at=datetime.now(UTC).isoformat(),
                    allowed_update_rate=seed.allowed_update_rate,
                )
                for seed in settings.values.seeds
            ],
            conflicts=[
                ValueConflictDefinition(
                    left_value_id=conflict.left_value_id,
                    right_value_id=conflict.right_value_id,
                    name=conflict.name,
                )
                for conflict in settings.values.conflicts
            ],
            max_update_per_event=settings.values.max_update_per_event,
            max_total_update_per_event=settings.values.max_total_update_per_event,
        )
        self.default_context_id: str | None = None
        self.restore_appraisal_state()
        self.restore_value_state()

    def chat(
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
    ) -> ChatResult:
        previous_context_frames = self.context_registry.frames
        previous_interlocutors = self.context_registry.interlocutors
        previous_default_context_id = self.default_context_id
        previous_calibration = self.surprisal_calculator.export_history()
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
            self.working_memory.advance()

            def compatibility(source_id: str | None) -> tuple[float, str]:
                return self.context_registry.compatibility(source_id, current_context)

            context_view = self.working_memory.select(
                resolver=self._resolve_working_memory,
                context_compatibility=compatibility,
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
            appraisal = self.appraiser.assess(
                loss_measurement,
                AppraisalSignals(
                    controllability=self.settings.appraisal.default_controllability
                    if loss_measurement.valid
                    else 0.5,
                    certainty=self.settings.appraisal.default_certainty
                    if loss_measurement.valid
                    else 0.2,
                    social_relevance=0.5
                    if current_context.participant_ids
                    else 0.0,
                    effort_cost=min(
                        1.0,
                        len(user_input.encode("utf-8"))
                        / self.settings.working_memory.token_capacity,
                    ),
                ),
            )
            emotion_update = self.emotion_engine.update_from_appraisal(appraisal)
            emotion_state = emotion_update.state
            memory_context = self.memory_system.retrieve_context(
                user_input,
                current_context_id=current_context.context_id,
                context_compatibility=compatibility,
            )
            event = current_agent_event()
            self._admit_runtime_context(
                memory_context, emotion_state, event, current_context
            )
            working_memory_view = self.working_memory.select(
                resolver=self._resolve_working_memory,
                context_compatibility=compatibility,
            )
            prompt = self.prompt_builder.build(
                user_input,
                emotion_state,
                working_memory_view,
                current_context=current_context,
                attachments=attachments or [],
            )
            try:
                raw_response = self.agent.generate(
                    prompt, attachments=attachments or []
                )
            except Exception as exc:
                if _fallback_used(self.provider):
                    raise RuntimeError("Fallback model generation failed") from exc
                raw_response = _generate_fallback(self.provider, prompt)
            processed_response = self.postprocessor.process(raw_response)
            if not processed_response.visible_response.strip():
                if _fallback_used(self.provider):
                    raise RuntimeError(
                        "Fallback model produced an empty visible response"
                    )
                raw_response = _generate_fallback(self.provider, prompt)
                processed_response = self.postprocessor.process(raw_response)
                if not processed_response.visible_response.strip():
                    raise RuntimeError(
                        "Fallback model produced an empty visible response"
                    )
            model_id = str(
                getattr(self.provider, "last_model_id", self.settings.model.primary_id)
            )
            fallback_used = bool(
                getattr(self.provider, "last_fallback_used", False)
            )
            generation_health = assess_generation_health(
                processed_response.visible_response,
                loss=loss_measurement.raw_loss
                if loss_measurement.raw_loss is not None
                else float("nan"),
                fallback_used=fallback_used,
            )
            episode_id = self.memory_system.save_episodic(
                user_input,
                processed_response.visible_response,
                hidden_thought=processed_response.hidden_thought,
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
                model_revision=str(
                    getattr(self.provider, "model_revision", "unknown")
                ),
                adapter_id=None if fallback_used else self.adapter_id,
                validation_status=ValidationStatus.UNVERIFIED,
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
                    activation=1.0,
                    salience=max(0.5, emotion_state.arousal),
                    retention_reason=RetentionReason.RECENT_CONTEXT,
                )
            )
        except Exception:
            self.emotion_engine.state = previous_emotion_state
            self.working_memory.restore(previous_working_memory)
            self.context_registry.restore(
                previous_context_frames, previous_interlocutors
            )
            self.default_context_id = previous_default_context_id
            self.surprisal_calculator.restore_history(previous_calibration)
            self._persist_appraisal_state()
            raise
        return ChatResult(
            episode_id=episode_id,
            response=processed_response.visible_response,
            hidden_thought=processed_response.hidden_thought
            if debug
            else processed_response.hidden_thought,
            loss=loss_measurement.raw_loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            model_id=model_id,
            adapter_id=None if fallback_used else self.adapter_id,
            fallback_used=fallback_used,
            prompt=prompt,
            memory_context=memory_context,
            working_memory_view=working_memory_view,
            context_id=current_context.context_id,
            loss_measurement=loss_measurement,
            appraisal=appraisal,
            emotion_update=emotion_update,
        )

    def advance_time(self, elapsed_seconds: float) -> EmotionUpdate:
        update = self.emotion_engine.advance_time(elapsed_seconds)
        self.working_memory.admit(
            working_memory_item(
                item_id="emotion:current",
                kind=WorkingMemoryKind.EMOTION,
                content=(
                    f"Current valence={update.state.valence:.3f}, "
                    f"arousal={update.state.arousal:.3f}"
                ),
                activation=1.0,
                salience=max(0.5, update.state.arousal),
                retention_reason=RetentionReason.CURRENT_EMOTION,
                source="runtime.emotion_tick",
            )
        )
        return update

    def restore_appraisal_state(self) -> None:
        self.surprisal_calculator.restore_history(
            self.persistent_state.extensions.get("appraisal_calibration")
        )

    def _persist_appraisal_state(self) -> None:
        self.persistent_state.extensions["appraisal_calibration"] = (
            self.surprisal_calculator.export_history()
        )

    def restore_value_state(self) -> None:
        self.value_system.restore(self.persistent_state.values)
        self._persist_value_state()

    def _persist_value_state(self) -> None:
        self.persistent_state.values = self.value_system.to_json()

    def apply_value_impacts(
        self,
        appraisal: AppraisalResult,
        impacts: dict[str, float],
        *,
        kind: ValueUpdateKind,
        memory_ids: tuple[str, ...] = (),
        source: str = "runtime",
        proposal_id: str | None = None,
    ) -> list[ValueUpdateRecord]:
        event = current_agent_event()
        evidence = ValueEvidence(
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            memory_ids=memory_ids,
            source=source,
        )
        proposals = self.value_system.proposals_from_appraisal(
            appraisal,
            impacts,
            kind=kind,
            evidence=evidence,
            proposal_id=proposal_id,
        )
        records = self.value_system.apply(proposals)
        self._persist_value_state()
        return records

    def evaluate_value_options(
        self, options: dict[str, dict[str, float]]
    ) -> list[ActionScore]:
        return self.value_system.evaluate(options)

    def freeze_value(self, value_id: str, *, frozen: bool) -> ValueState:
        state = self.value_system.freeze(value_id, frozen=frozen)
        self._persist_value_state()
        return state

    def rollback_value(self, value_id: str, *, target_revision: int) -> ValueState:
        state = self.value_system.rollback(value_id, target_revision=target_revision)
        self._persist_value_state()
        return state

    def reset_values(self, value_ids: tuple[str, ...] | None = None) -> list[ValueState]:
        states = self.value_system.reset(value_ids)
        self._persist_value_state()
        return states

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
            frame = self.context_registry.get(identifier)
            if frame is None:
                raise KeyError(identifier)
            frame = self.context_registry.resume(identifier)
        if (
            interlocutor_key is not None
            and self.context_registry.get_interlocutor(interlocutor_key) is None
        ):
            self.context_registry.register_interlocutor(
                InterlocutorModel(identity_key=interlocutor_key)
            )
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
        for record in memory_context.db2_results:
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"semantic:{record.id}",
                    kind=WorkingMemoryKind.SEMANTIC,
                    reference=f"semantic:{record.id}",
                    source_event_id=None,
                    source_event_sequence=None,
                    context_id=record.context_id,
                    source=record.source,
                    source_channel=record.source_channel,
                    source_session_id=record.source_session_id,
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
            self.persistent_state.active_goals,
            kind=WorkingMemoryKind.GOAL,
            reason=RetentionReason.ONGOING_GOAL,
        )
        self._admit_structured_state(
            self.persistent_state.commitments,
            kind=WorkingMemoryKind.COMMITMENT,
            reason=RetentionReason.ACTIVE_COMMITMENT,
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
    ) -> None:
        for entry in entries:
            entry_id = entry.get("id")
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
            record = self.memory_system.get_episodic(item.reference.removeprefix("episode:"))
            if (
                record is None
                or record.archived
                or record.lifecycle_status != MemoryLifecycleStatus.ACTIVE
            ):
                return None
            return f"User: {record.user_input} | Assistant: {record.response}"
        if item.reference.startswith("semantic:"):
            record = self.memory_system.get_semantic(item.reference.removeprefix("semantic:"))
            if (
                record is None
                or record.archived
                or record.metadata.get("publication_status", "published")
                != "published"
            ):
                return None
            return record.text
        return None


def _generate_fallback(provider: ModelProvider, prompt: str) -> str:
    generate_fallback = getattr(provider, "generate_fallback", None)
    if not callable(generate_fallback):
        raise RuntimeError("Model generation failed and no fallback model is available")
    return str(generate_fallback(prompt))


def _fallback_used(provider: ModelProvider) -> bool:
    return bool(getattr(provider, "last_fallback_used", False))
