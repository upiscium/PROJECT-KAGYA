"""Integrated runtime main loop."""

from dataclasses import dataclass
from typing import Any

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.cognition import SurprisalCalculator
from kagya.config import Settings
from kagya.memory import DualMemorySystem, MemoryContext
from kagya.models import ModelProvider
from kagya.persona import ConsciousAgent, PromptBuilder, ResponsePostprocessor
from kagya.runtime.session_state import SessionState


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

    def chat(
        self,
        user_input: str,
        debug: bool = False,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        context_text = self.session_state.context_text()
        loss = self.surprisal_calculator.calculate(context_text, user_input)
        emotion_state = self.emotion_engine.update(loss)
        memory_context = self.memory_system.retrieve_context(user_input)
        prompt = self.prompt_builder.build(
            user_input, emotion_state, memory_context, attachments=attachments or []
        )
        try:
            raw_response = self.agent.generate(prompt, attachments=attachments or [])
        except Exception as exc:
            if _fallback_used(self.provider):
                raise RuntimeError("Fallback model generation failed") from exc
            raw_response = _generate_fallback(self.provider, prompt)
        processed_response = self.postprocessor.process(raw_response)
        if not processed_response.visible_response.strip():
            if _fallback_used(self.provider):
                raise RuntimeError("Fallback model produced an empty visible response")
            raw_response = _generate_fallback(self.provider, prompt)
            processed_response = self.postprocessor.process(raw_response)
            if not processed_response.visible_response.strip():
                raise RuntimeError("Fallback model produced an empty visible response")
        model_id = str(
            getattr(self.provider, "last_model_id", self.settings.model.primary_id)
        )
        fallback_used = bool(getattr(self.provider, "last_fallback_used", False))
        episode_id = self.memory_system.save_episodic(
            user_input,
            processed_response.visible_response,
            hidden_thought=processed_response.hidden_thought,
            loss=loss,
            emotion_valence=emotion_state.valence,
            emotion_arousal=emotion_state.arousal,
        )
        self.session_state.add_turn(user_input, processed_response.visible_response)
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
        )


def _generate_fallback(provider: ModelProvider, prompt: str) -> str:
    generate_fallback = getattr(provider, "generate_fallback", None)
    if not callable(generate_fallback):
        raise RuntimeError("Model generation failed and no fallback model is available")
    return str(generate_fallback(prompt))


def _fallback_used(provider: ModelProvider) -> bool:
    return bool(getattr(provider, "last_fallback_used", False))
