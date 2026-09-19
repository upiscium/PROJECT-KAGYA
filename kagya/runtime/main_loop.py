"""Integrated runtime main loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.cognition import SurprisalCalculator
from kagya.config import Settings
from kagya.memory import DualMemorySystem, MemoryContext, MemoryRecordType
from kagya.models import ModelProvider
from kagya.persona import ConsciousAgent, PromptBuilder, ResponsePostprocessor
from kagya.runtime.session_participant import (
    SessionTurnOperation,
    SessionTurnParticipant,
)
from kagya.runtime.session_state import SessionState
from kagya.runtime.context import ContextRegistry
from kagya.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionBoundValue,
    TransactionParticipant,
)
from kagya.runtime.working_memory import (
    WorkingMemory,
    WorkingMemorySourceKind,
    WorkingMemoryView,
)

if TYPE_CHECKING:
    from kagya.memory.episodic_participant import MemoryEpisodicParticipant


@dataclass(frozen=True)
class ChatResult:
    """Visible/structured result safe to pass to ordinary callers."""

    episode_id: str
    response: str
    loss: float
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None


@dataclass(frozen=True)
class DebugChatTrace:
    """Request-scoped diagnostic data that must never become durable authority."""

    hidden_thought: str
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView


@dataclass(frozen=True)
class _ComputedChat:
    response: str
    loss: float
    valence: float
    arousal: float
    optimal_loss: float
    trace: DebugChatTrace | None
    memory_participant: MemoryEpisodicParticipant
    session_participant: SessionTurnParticipant


class KagyaMainLoop:
    """Connect prediction error, emotion, memory, generation, and storage."""

    def __init__(
        self,
        settings: Settings,
        provider: ModelProvider,
        memory_system: DualMemorySystem,
        *,
        session_state: SessionState | None = None,
        working_memory: WorkingMemory | None = None,
        emotion_engine: EmotionEngineAllostasis | None = None,
        prompt_builder: PromptBuilder | None = None,
        agent: ConsciousAgent | None = None,
        postprocessor: ResponsePostprocessor | None = None,
        adapter_id: str | None = None,
        context_registry: ContextRegistry | None = None,
    ) -> None:
        from kagya.memory.working_memory_resolver import MemoryWorkingMemoryResolver

        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
        self.working_memory_resolver = MemoryWorkingMemoryResolver(memory_system)
        self.session_state = session_state or SessionState()
        self.working_memory = (
            working_memory
            if working_memory is not None
            else WorkingMemory(
                item_capacity=settings.working_memory.item_capacity,
                projection_max_bytes=settings.working_memory.projection_max_bytes,
            )
        )
        self.surprisal_calculator = SurprisalCalculator(provider)
        self.emotion_engine = emotion_engine or EmotionEngineAllostasis(
            EmotionState(optimal_loss=settings.emotion.baseline_surprisal),
            adaptation_rate=settings.emotion.decay_rate,
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.agent = agent or ConsciousAgent(provider)
        self.postprocessor = postprocessor or ResponsePostprocessor()
        self.adapter_id = adapter_id
        self.context_registry = (
            context_registry if context_registry is not None else ContextRegistry()
        )

    def chat(self, user_input: str) -> CoordinatedResult[ChatResult]:
        """Compute an ordinary turn and return its process-local mutation plan."""

        computed = self._run_chat(user_input, capture_debug=False)
        return CoordinatedResult(
            TransactionBoundValue(
                lambda transaction_id: self._chat_result(computed, transaction_id)
            ),
            self._participants(computed),
        )

    def chat_debug(
        self, user_input: str
    ) -> CoordinatedResult[tuple[ChatResult, DebugChatTrace]]:
        """Compute a turn with ephemeral diagnostics and the same mutation plan."""

        computed = self._run_chat(user_input, capture_debug=True)
        if computed.trace is None:  # pragma: no cover - internal invariant
            raise RuntimeError("Debug trace was not captured")
        trace = computed.trace
        return CoordinatedResult(
            TransactionBoundValue(
                lambda transaction_id: (
                    self._chat_result(computed, transaction_id),
                    trace,
                )
            ),
            self._participants(computed),
        )

    def _run_chat(
        self, user_input: str, *, capture_debug: bool
    ) -> _ComputedChat:
        from kagya.memory.episodic_participant import (
            EpisodicWrite,
            MemoryEpisodicParticipant,
        )

        current_context = self.context_registry.current_context
        provenance = (
            (
                current_context.context_id,
                current_context.source_channel,
                current_context.source_session_id,
            )
            if current_context is not None
            else (None, None, None)
        )
        context_text = self.session_state.context_text()
        loss = self.surprisal_calculator.calculate(context_text, user_input)
        emotion_state = self.emotion_engine.update(loss)
        memory_context = self.memory_system.retrieve_context(user_input)
        self.working_memory.advance()
        candidates = [
            (WorkingMemorySourceKind.EPISODIC, record.id, rank)
            for rank, record in enumerate(memory_context.db1_results)
        ] + [
            (WorkingMemorySourceKind.SEMANTIC, record.id, rank)
            for rank, record in enumerate(memory_context.db2_results)
        ]
        # Admit larger rank numbers first (lower retrieval priority). Equal-rank
        # references use ascending source kind and source ID as the total tie-break.
        for source_kind, source_id, rank in sorted(
            candidates,
            key=lambda candidate: (-candidate[2], candidate[0].value, candidate[1]),
        ):
            self.working_memory.admit(
                source_kind,
                source_id,
                activation=1.0,
                salience=1.0 / (rank + 1),
            )
        working_memory_view = self.working_memory.select(self.working_memory_resolver)
        prompt = self.prompt_builder.build(
            user_input, emotion_state, working_memory_view
        )
        raw_response = self.agent.generate(prompt)
        processed_response = self.postprocessor.process(raw_response)
        memory_participant = MemoryEpisodicParticipant(
            self.memory_system,
            EpisodicWrite(
                user_input=user_input,
                response=processed_response.visible_response,
                loss=loss,
                emotion_valence=emotion_state.valence,
                emotion_arousal=emotion_state.arousal,
                record_type=MemoryRecordType.EPISODIC_LOG,
                created_at=datetime.now(UTC).isoformat(),
                context_id=provenance[0],
                source_channel=provenance[1],
                source_session_id=provenance[2],
                schema_version=2,
            ),
        )
        session_participant = SessionTurnParticipant(
            self.session_state,
            SessionTurnOperation(
                user_input=user_input,
                response=processed_response.visible_response,
            ),
        )
        trace = None
        if capture_debug:
            trace = DebugChatTrace(
                hidden_thought=processed_response.hidden_thought,
                prompt=prompt,
                memory_context=memory_context,
                working_memory_view=working_memory_view,
            )
        return _ComputedChat(
            response=processed_response.visible_response,
            loss=loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            trace=trace,
            memory_participant=memory_participant,
            session_participant=session_participant,
        )

    def _chat_result(
        self, computed: _ComputedChat, transaction_id: str
    ) -> ChatResult:
        return ChatResult(
            episode_id=computed.memory_participant.episode_id(transaction_id),
            response=computed.response,
            loss=computed.loss,
            valence=computed.valence,
            arousal=computed.arousal,
            optimal_loss=computed.optimal_loss,
            model_id=self.settings.model.primary_id,
            adapter_id=self.adapter_id,
        )

    @staticmethod
    def _participants(computed: _ComputedChat) -> tuple[TransactionParticipant, ...]:
        return (computed.memory_participant, computed.session_participant)
