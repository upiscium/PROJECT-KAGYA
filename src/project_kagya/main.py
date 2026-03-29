from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .conscious_agent import ConsciousAgent
from .dual_memory_system import DualMemorySystem
from .emotion_engine import EmotionEngineAllostasis
from .settings import (
    AppSettings,
    LoggingSettings,
    SleepSettings,
    resolve_log_path,
    resolve_model_adapter_path,
    resolve_sleep_output_path,
)
from .sleep_consolidation import SleepCycleManager
from .surprisal_calculator import compute_surprisal_loss


@dataclass
class RuntimeComponents:
    surprisal_model: object
    surprisal_tokenizer: object
    memory: DualMemorySystem = field(default_factory=DualMemorySystem)
    emotion_engine: EmotionEngineAllostasis = field(
        default_factory=EmotionEngineAllostasis
    )
    agent: ConsciousAgent = field(default_factory=ConsciousAgent)
    initial_valence: float = 0.0
    initial_arousal: float = 0.0


class SubjectiveAIAgent:
    def __init__(self, components: RuntimeComponents) -> None:
        self.components = components
        self.valence = components.initial_valence
        self.arousal = components.initial_arousal

    def handle_turn(self, user_input: str) -> str:
        context = self.components.memory.retrieve_context(user_input)
        loss = compute_surprisal_loss(
            self.components.surprisal_model,
            self.components.surprisal_tokenizer,
            context,
            user_input,
        )
        state = self.components.emotion_engine.update(loss, self.valence, self.arousal)
        self.valence = state.valence
        self.arousal = state.arousal
        response = self.components.agent.generate(
            user_input, self.valence, self.arousal, context
        )
        self.components.memory.save_episodic(
            user_input, response, self.valence, self.arousal
        )
        return response


def load_base_model(
    model_name: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required to load a base model") from exc

    model_kwargs: dict[str, Any] = {}
    if load_in_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "device_map": "auto",
            }
        )

    model: Any = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if adapter_path:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "peft is required when adapter_path is provided"
            ) from exc

        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def load_runtime_from_disk(
    model_name: str,
    adapter_path: str | None = None,
    memory: DualMemorySystem | None = None,
    emotion_engine: EmotionEngineAllostasis | None = None,
    agent: ConsciousAgent | None = None,
    load_in_4bit: bool = True,
) -> SubjectiveAIAgent:
    model, tokenizer = load_base_model(
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )
    return build_runtime(
        surprisal_model=model,
        surprisal_tokenizer=tokenizer,
        memory=memory,
        emotion_engine=emotion_engine,
        agent=agent,
    )


def load_runtime_from_settings(settings: AppSettings) -> SubjectiveAIAgent:
    memory = DualMemorySystem(top_k=settings.memory.top_k)
    emotion_engine = EmotionEngineAllostasis(
        optimal_loss=settings.emotion.optimal_loss,
        adaptation_rate=settings.emotion.adaptation_rate,
    )
    agent = ConsciousAgent()

    if settings.runtime.backend == "dummy":
        surprisal_model, surprisal_tokenizer = _build_dummy_runtime_components()
        return build_runtime(
            surprisal_model=surprisal_model,
            surprisal_tokenizer=surprisal_tokenizer,
            memory=memory,
            emotion_engine=emotion_engine,
            agent=agent,
            initial_valence=settings.runtime.initial_valence,
            initial_arousal=settings.runtime.initial_arousal,
        )

    adapter_path = resolve_model_adapter_path(settings)
    runtime = load_runtime_from_disk(
        model_name=settings.model.model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
        memory=memory,
        emotion_engine=emotion_engine,
        agent=agent,
        load_in_4bit=settings.model.load_in_4bit,
    )
    runtime.valence = settings.runtime.initial_valence
    runtime.arousal = settings.runtime.initial_arousal
    return runtime


def build_runtime(
    surprisal_model: object,
    surprisal_tokenizer: object,
    memory: DualMemorySystem | None = None,
    emotion_engine: EmotionEngineAllostasis | None = None,
    agent: ConsciousAgent | None = None,
    initial_valence: float = 0.0,
    initial_arousal: float = 0.0,
) -> SubjectiveAIAgent:
    components = RuntimeComponents(
        surprisal_model=surprisal_model,
        surprisal_tokenizer=surprisal_tokenizer,
        memory=memory or DualMemorySystem(),
        emotion_engine=emotion_engine or EmotionEngineAllostasis(),
        agent=agent or ConsciousAgent(),
        initial_valence=initial_valence,
        initial_arousal=initial_arousal,
    )
    return SubjectiveAIAgent(components)


def ensure_adapter_path(adapter_path: str | Path | None) -> Path | None:
    if adapter_path is None:
        return None
    path = Path(adapter_path)
    return path if path.exists() else None


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


class _DummyModel:
    def __call__(
        self, input_ids: list[int], labels: list[int], attention_mask: list[int]
    ) -> object:
        loss = sum(token for token in input_ids if token) / max(1, len(input_ids))
        return type("Output", (), {"loss": loss})()


def _build_dummy_runtime_components() -> tuple[object, object]:
    return _DummyModel(), _DummyTokenizer()


def build_sleep_manager(settings: SleepSettings) -> SleepCycleManager:
    return SleepCycleManager(settings=settings)


def get_logging_settings(settings: AppSettings) -> LoggingSettings:
    return settings.logging


def get_runtime_log_path(settings: AppSettings) -> Path:
    return resolve_log_path(settings)


def get_sleep_output_path(settings: AppSettings) -> Path:
    return resolve_sleep_output_path(settings)


def build_runtime_from_settings_file(settings: AppSettings) -> SubjectiveAIAgent:
    runtime = load_runtime_from_settings(settings)
    return runtime
