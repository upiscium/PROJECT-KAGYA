from pathlib import Path

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import KagyaMainLoop


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"


class ThinkingDummyProvider(DummyProvider):
    response_text = f"<think>{PRIVATE_SENTINEL}</think>Visible runtime answer."

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


def test_dummy_provider_drives_user_input_to_public_response_end_to_end(
    tmp_path: Path,
) -> None:
    provider = ThinkingDummyProvider()
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    loop = KagyaMainLoop(settings, provider, memory)

    result = loop.chat("hello")

    assert result.response == "Visible runtime answer."
    assert result.loss == DummyProvider.loss_value
    assert result.episode_id.startswith("episode-")
    assert result.model_id == settings.model.primary_id
    assert result.adapter_id is None
    assert not hasattr(result, "hidden_thought")
    assert not hasattr(result, "prompt")
    assert not hasattr(result, "memory_context")


def test_debug_trace_exposes_private_thought_only_ephemerally(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
    )

    result, trace = loop.chat_debug("inspect this turn")

    assert result.response == "Visible runtime answer."
    assert trace.hidden_thought == PRIVATE_SENTINEL
    assert PRIVATE_SENTINEL not in str(result)
    assert "Assistant:" in trace.prompt


def test_db1_never_persists_extracted_private_thought(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result, trace = loop.chat_debug("remember this")
    stored = memory.db1.get(
        ids=[result.episode_id], include=["documents", "metadatas"]
    )

    assert trace.hidden_thought == PRIVATE_SENTINEL
    assert stored["ids"] == [result.episode_id]
    assert stored["metadatas"][0]["user_input"] == "remember this"
    assert stored["metadatas"][0]["response"] == "Visible runtime answer."
    assert "hidden_thought" not in stored["metadatas"][0]
    assert PRIVATE_SENTINEL not in str(stored)


def test_visible_response_does_not_contain_think_tags_or_private_sentinel(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
    ).chat("hello")

    assert "<think>" not in result.response
    assert "</think>" not in result.response
    assert PRIVATE_SENTINEL not in result.response


def test_emotion_state_changes_after_loss_calculation(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))
    before = loop.emotion_engine.state

    result = loop.chat("emotion update")

    assert result.arousal != before.arousal
    assert result.optimal_loss != before.optimal_loss


def test_prompt_includes_emotion_and_retrieved_memory(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    memory = DualMemorySystem(settings)
    memory.save_episodic("old episode", "old answer")
    memory.save_semantic("stable semantic memory")
    loop = KagyaMainLoop(settings, provider, memory)

    _result, trace = loop.chat_debug("old semantic query")

    assert "valence:" in trace.prompt
    assert "arousal:" in trace.prompt
    assert "optimal_loss:" in trace.prompt
    assert "old episode" in trace.prompt
    assert "stable semantic memory" in trace.prompt
    assert "hidden_thought" not in trace.prompt
    assert "<think>" not in trace.prompt
    assert "Assistant response:" not in trace.prompt
    assert trace.prompt.endswith("Assistant:")
    assert provider.prompts == [trace.prompt]


def test_prompt_uses_plain_visible_answer_contract(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    _result, trace = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
    ).chat_debug("answer naturally")

    assert trace.prompt.startswith("Context: PROJECT-KAGYA")
    assert "private local AI assistant" in trace.prompt
    assert "Private runtime data below is for tone and context only" in trace.prompt
    assert "User: answer naturally\nAssistant:" in trace.prompt


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
