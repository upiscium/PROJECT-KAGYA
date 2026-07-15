"""Integrated runtime main loop."""

from dataclasses import dataclass
from typing import Any

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.cognition import SurprisalCalculator
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


@dataclass(frozen=True)
class ChatResult:
    episode_id: str
    response: str
    hidden_thought: str
    loss: float
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None
    fallback_used: bool
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView


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
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
        self.session_state = session_state or SessionState()
        self.surprisal_calculator = SurprisalCalculator(provider)
        self.emotion_engine = emotion_engine or EmotionEngineAllostasis(
            EmotionState(optimal_loss=settings.emotion.baseline_surprisal),
            adaptation_rate=settings.emotion.decay_rate,
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

    def chat(
        self,
        user_input: str,
        debug: bool = False,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        previous_working_memory = self.working_memory.items
        self.working_memory.advance()
        context_view = self.working_memory.select(resolver=self._resolve_working_memory)
        context_text = context_view.context_text()
        loss = self.surprisal_calculator.calculate(context_text, user_input)
        previous_emotion_state = self.emotion_engine.state
        try:
            emotion_state = self.emotion_engine.update(loss)
            memory_context = self.memory_system.retrieve_context(user_input)
            event = current_agent_event()
            self._admit_runtime_context(memory_context, emotion_state, event)
            working_memory_view = self.working_memory.select(
                resolver=self._resolve_working_memory
            )
            prompt = self.prompt_builder.build(
                user_input,
                emotion_state,
                working_memory_view,
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
                loss=loss,
                fallback_used=fallback_used,
            )
            episode_id = self.memory_system.save_episodic(
                user_input,
                processed_response.visible_response,
                hidden_thought=processed_response.hidden_thought,
                loss=loss,
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
                provider=self.settings.model.provider,
                model_id=model_id,
                model_revision=str(
                    getattr(self.provider, "model_revision", "unknown")
                ),
                adapter_id=None if fallback_used else self.adapter_id,
                validation_status=ValidationStatus.UNVERIFIED,
            )
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"episode:{episode_id}",
                    kind=WorkingMemoryKind.CONVERSATION,
                    reference=f"episode:{episode_id}",
                    source_event_id=None if event is None else event.event_id,
                    source_event_sequence=None
                    if event is None
                    else event.processing_sequence,
                    context_id=None if event is None else event.correlation_id,
                    activation=1.0,
                    salience=max(0.5, emotion_state.arousal),
                    retention_reason=RetentionReason.RECENT_CONTEXT,
                )
            )
        except Exception:
            self.emotion_engine.state = previous_emotion_state
            self.working_memory.restore(previous_working_memory)
            raise
        return ChatResult(
            episode_id=episode_id,
            response=processed_response.visible_response,
            hidden_thought=processed_response.hidden_thought
            if debug
            else processed_response.hidden_thought,
            loss=loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            model_id=model_id,
            adapter_id=None if fallback_used else self.adapter_id,
            fallback_used=fallback_used,
            prompt=prompt,
            memory_context=memory_context,
            working_memory_view=working_memory_view,
        )

    def _admit_runtime_context(
        self,
        memory_context: MemoryContext,
        emotion_state: EmotionState,
        event: Any,
    ) -> None:
        event_id = None if event is None else event.event_id
        event_sequence = None if event is None else event.processing_sequence
        context_id = None if event is None else event.correlation_id
        for record in memory_context.db1_results:
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"episode:{record.id}",
                    kind=WorkingMemoryKind.EPISODIC,
                    reference=f"episode:{record.id}",
                    source_event_id=record.source_event_id,
                    source_event_sequence=record.processing_sequence,
                    context_id=record.context_id,
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
                    source_event_id=event_id,
                    source_event_sequence=event_sequence,
                    context_id=context_id,
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
