from project_kagya.conscious_agent import ConsciousAgent
from project_kagya.dual_memory_system import DualMemorySystem
from project_kagya.emotion_engine import EmotionEngineAllostasis
from project_kagya.main import build_runtime, load_runtime_from_settings
from project_kagya.settings import (
    AppSettings,
    LoggingSettings,
    MemorySettings,
    EmotionSettings,
    RuntimeSettings,
)


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(word) for word in text.split()]


class DummyModel:
    def __call__(self, **kwargs: object) -> object:
        return type("Output", (), {"loss": 1.0})()


def test_runtime_turn_executes_flow() -> None:
    memory = DualMemorySystem()
    memory.save_episodic("hello", "world", 0.1, 0.1)
    runtime = build_runtime(
        surprisal_model=DummyModel(),
        surprisal_tokenizer=DummyTokenizer(),
        memory=memory,
        emotion_engine=EmotionEngineAllostasis(),
        agent=ConsciousAgent(lambda prompt: "<think>planned</think> response"),
    )

    response = runtime.handle_turn("hello again")

    assert response.startswith("<think>")
    assert len(memory.hippocampus.records) >= 2


def test_runtime_can_be_built_from_settings() -> None:
    settings = AppSettings(
        runtime=RuntimeSettings(
            backend="dummy",
            input_text="hello there",
            initial_valence=0.3,
            initial_arousal=0.4,
        ),
        memory=MemorySettings(top_k=1),
        emotion=EmotionSettings(optimal_loss=2.0, adaptation_rate=0.2),
        logging=LoggingSettings(level="DEBUG", file_path="test.log"),
    )

    runtime = load_runtime_from_settings(settings)

    assert runtime.valence == 0.3
    assert runtime.arousal == 0.4
