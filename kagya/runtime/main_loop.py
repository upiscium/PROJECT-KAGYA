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
from kagya.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionBoundValue,
    TransactionParticipant,
)
from kagya.runtime.working_memory import WorkingMemory

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
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
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

        context_text = self.session_state.context_text()
        loss = self.surprisal_calculator.calculate(context_text, user_input)
        emotion_state = self.emotion_engine.update(loss)
        memory_context = self.memory_system.retrieve_context(user_input)
        prompt = self.prompt_builder.build(user_input, emotion_state, memory_context)
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
