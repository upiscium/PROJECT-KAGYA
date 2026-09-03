"""Integrated runtime main loop."""

from dataclasses import dataclass

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.cognition import SurprisalCalculator
from kagya.config import Settings
from kagya.memory import DualMemorySystem, MemoryContext
from kagya.models import ModelProvider
from kagya.persona import ConsciousAgent, PromptBuilder, ResponsePostprocessor
from kagya.runtime.session_state import SessionState


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

    def chat(self, user_input: str) -> ChatResult:
        """Run an ordinary chat turn without returning private diagnostic data."""

        result, _trace = self._run_chat(user_input, capture_debug=False)
        return result

    def chat_debug(self, user_input: str) -> tuple[ChatResult, DebugChatTrace]:
        """Run a turn and return explicitly ephemeral admin/debug diagnostics."""

        result, trace = self._run_chat(user_input, capture_debug=True)
        if trace is None:  # pragma: no cover - internal invariant
            raise RuntimeError("Debug trace was not captured")
        return result, trace

    def _run_chat(
        self, user_input: str, *, capture_debug: bool
    ) -> tuple[ChatResult, DebugChatTrace | None]:
        context_text = self.session_state.context_text()
        loss = self.surprisal_calculator.calculate(context_text, user_input)
        emotion_state = self.emotion_engine.update(loss)
        memory_context = self.memory_system.retrieve_context(user_input)
        prompt = self.prompt_builder.build(user_input, emotion_state, memory_context)
        raw_response = self.agent.generate(prompt)
        processed_response = self.postprocessor.process(raw_response)
        episode_id = self.memory_system.save_episodic(
            user_input,
            processed_response.visible_response,
            loss=loss,
            emotion_valence=emotion_state.valence,
            emotion_arousal=emotion_state.arousal,
        )
        self.session_state.add_turn(user_input, processed_response.visible_response)
        result = ChatResult(
            episode_id=episode_id,
            response=processed_response.visible_response,
            loss=loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            model_id=self.settings.model.primary_id,
            adapter_id=self.adapter_id,
        )
        trace = None
        if capture_debug:
            trace = DebugChatTrace(
                hidden_thought=processed_response.hidden_thought,
                prompt=prompt,
                memory_context=memory_context,
            )
        return result, trace
