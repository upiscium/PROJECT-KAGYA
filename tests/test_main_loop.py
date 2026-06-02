from pathlib import Path

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import KagyaMainLoop


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class ThinkingDummyProvider(DummyProvider):
    response_text = "<think>internal runtime thought</think>Visible runtime answer."

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


def test_dummy_provider_drives_user_input_to_response_end_to_end(tmp_path: Path) -> None:
    provider = ThinkingDummyProvider()
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    loop = KagyaMainLoop(_settings_for_tmp_memory(tmp_path), provider, memory)

    result = loop.chat("hello", debug=True)

    assert result.response == "Visible runtime answer."
    assert result.hidden_thought == "internal runtime thought"
    assert result.loss == DummyProvider.loss_value
    assert result.episode_id.startswith("episode-")
    assert result.model_id == _settings_for_tmp_memory(tmp_path).model.primary_id
    assert result.adapter_id is None


def test_db1_receives_saved_episode_with_hidden_thought(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result = loop.chat("remember this", debug=True)
    stored = memory.db1.get(ids=[result.episode_id], include=["metadatas"])

    assert stored["ids"] == [result.episode_id]
    assert stored["metadatas"][0]["user_input"] == "remember this"
    assert stored["metadatas"][0]["response"] == "Visible runtime answer."
    assert stored["metadatas"][0]["hidden_thought"] == "internal runtime thought"


def test_visible_response_does_not_contain_think_tags(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
    ).chat("hello", debug=False)

    assert "<think>" not in result.response
    assert "</think>" not in result.response


def test_emotion_state_changes_after_loss_calculation(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))
    before = loop.emotion_engine.state

    result = loop.chat("emotion update", debug=True)

    assert result.arousal != before.arousal
    assert result.optimal_loss != before.optimal_loss


def test_prompt_includes_emotion_and_retrieved_memory(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    memory = DualMemorySystem(settings)
    memory.save_episodic("old episode", "old answer")
    memory.save_semantic("stable semantic memory")
    loop = KagyaMainLoop(settings, provider, memory)

    result = loop.chat("old semantic query", debug=True)

    assert "valence:" in result.prompt
    assert "arousal:" in result.prompt
    assert "optimal_loss:" in result.prompt
    assert "old episode" in result.prompt
    assert "stable semantic memory" in result.prompt
    assert "hidden_thought" in result.prompt
    assert "<think>" not in result.prompt
    assert "Assistant response:" not in result.prompt
    assert result.prompt.endswith("<start_of_turn>model")
    assert provider.prompts == [result.prompt]


def test_prompt_uses_gemma_style_visible_answer_contract(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
    ).chat("answer naturally", debug=True)

    assert result.prompt.startswith("<start_of_turn>user")
    assert "Answer the user directly and naturally as the assistant." in result.prompt
    assert "Do not reveal hidden_thought" in result.prompt
    assert "Do not include XML/HTML tags" in result.prompt
    assert "<end_of_turn>\n<start_of_turn>model" in result.prompt


def _settings_for_tmp_memory(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_runtime_test",
                    "db2_collection": "cortex_runtime_test",
                }
            )
        }
    )
