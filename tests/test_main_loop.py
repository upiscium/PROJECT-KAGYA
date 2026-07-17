from pathlib import Path
import math

from kagya.config import Settings, load_settings
from kagya.experience import ExperienceAppraisal
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
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


class ThinkOnlyPrimaryProvider(ThinkingDummyProvider):
    response_text = "<think>internal only</think>"

    def __init__(self, fallback_response: str = "Fallback visible answer.") -> None:
        super().__init__()
        self.fallback_response = fallback_response
        self.fallback_calls = 0
        self.last_model_id = "primary-model"
        self.last_fallback_used = False

    def generate_fallback(self, prompt: str) -> str:
        self.fallback_calls += 1
        self.last_model_id = "fallback-model"
        self.last_fallback_used = True
        return self.fallback_response


class InvalidLossThinkingProvider(ThinkingDummyProvider):
    def calculate_loss(self, context_text: str, target_text: str) -> float:
        return math.nan


def test_dummy_provider_drives_user_input_to_response_end_to_end(
    tmp_path: Path,
) -> None:
    provider = ThinkingDummyProvider()
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    loop = KagyaMainLoop(_settings_for_tmp_memory(tmp_path), provider, memory)

    result = loop.chat("hello", debug=True)

    assert result.response == "Visible runtime answer."
    assert result.hidden_thought == "internal runtime thought"
    assert result.loss == DummyProvider.loss_value
    assert result.episode_id.startswith("episode-")
    assert result.experience_id.startswith("experience-")
    experience = loop.experience_store.get(result.experience_id)
    assert experience.result_refs["memory"] == (f"episode:{result.episode_id}",)
    episode_item = next(
        item
        for item in loop.working_memory.items
        if item.item_id == f"episode:{result.episode_id}"
    )
    assert episode_item.salience == experience.subjective_salience
    assert loop.persistent_state.extensions["experiences"]["records"][0][
        "experience_id"
    ] == result.experience_id
    assert result.model_id == _settings_for_tmp_memory(tmp_path).model.primary_id
    assert result.adapter_id is None


def test_db1_receives_saved_episode_with_hidden_thought(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
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
        _memory(settings),
    ).chat("hello", debug=False)

    assert "<think>" not in result.response
    assert "</think>" not in result.response


def test_truncated_think_only_primary_response_uses_fallback(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkOnlyPrimaryProvider()
    provider.response_text = "<think>private truncated reasoning"

    result = KagyaMainLoop(settings, provider, _memory(settings)).chat(
        "hello", debug=False
    )

    assert result.response == "Fallback visible answer."
    assert "private truncated reasoning" not in result.response
    assert result.fallback_used is True


def test_emotion_state_changes_after_loss_calculation(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    before = loop.emotion_engine.state

    result = loop.chat("emotion update", debug=True)

    assert result.arousal != before.arousal
    assert result.valence != before.valence


def test_experience_reassessment_updates_linked_memory_salience(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)
    result = loop.chat("reassess this", debug=True)

    revised = loop.reassess_experience(
        result.experience_id,
        appraisal=ExperienceAppraisal(
            valence=-0.5,
            arousal=0.9,
            novelty=1.0,
            novelty_valid=True,
            goal_progress=-0.5,
            threat=0.8,
            controllability=0.2,
            certainty=0.9,
            social_relevance=0.8,
            effort_cost=0.3,
            reason_codes=("later_evidence",),
        ),
        reason_code="later_evidence",
        evidence_refs=("memory:evidence",),
    )
    episode = memory.get_episodic(result.episode_id)

    assert episode is not None
    assert revised.revision == 1
    assert episode.subjective_salience == revised.subjective_salience
    assert episode.autobiographical_importance == revised.autobiographical_importance


def test_invalid_loss_does_not_abort_chat_or_become_zero_novelty(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)

    result = KagyaMainLoop(
        settings, InvalidLossThinkingProvider(), memory
    ).chat("hello", debug=True)

    assert result.response == "Visible runtime answer."
    assert result.loss is None
    assert result.loss_measurement.valid is False
    assert result.appraisal.novelty is None
    assert "novelty_omitted" in result.emotion_update.reasons
    stored = memory.get_episodic(result.episode_id)
    assert stored is not None
    assert stored.generation_health.non_finite_score is True


def test_prompt_includes_emotion_and_retrieved_memory(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    memory = _memory(settings)
    memory.save_episodic("old episode", "old answer")
    memory.save_semantic("stable semantic memory")
    loop = KagyaMainLoop(settings, provider, memory)

    result = loop.chat("old semantic query", debug=True)

    assert "valence:" in result.prompt
    assert "arousal:" in result.prompt
    assert "optimal_loss:" in result.prompt
    assert "old episode" in result.prompt
    assert "stable semantic memory" in result.prompt
    assert "Past recorded interaction (not a current fact)" in result.prompt
    assert "Stored semantic record (not an adopted belief)" in result.prompt
    assert "hidden_thought" not in result.prompt
    assert "<think>" not in result.prompt
    assert "Assistant response:" not in result.prompt
    assert result.prompt.endswith("Assistant:")
    assert provider.prompts == [result.prompt]


def test_previous_exchange_reaches_prompt_through_bounded_working_memory(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    loop = KagyaMainLoop(settings, provider, _memory(settings))

    first = loop.chat("first topic", debug=True)
    second = loop.chat("continue", debug=True)

    assert "Working memory:" in second.prompt
    assert "first topic" in second.prompt
    assert first.response in second.prompt
    assert loop.session_state.turns == []
    assert len(loop.working_memory.items) <= settings.working_memory.item_capacity


def test_cross_context_memory_is_marked_with_its_origin(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    loop.chat(
        "shared project detail",
        context_id="ctx-first",
        create_context=True,
        interlocutor_key="person-a",
    )

    result = loop.chat(
        "shared project",
        debug=True,
        context_id="ctx-second",
        create_context=True,
        interlocutor_key="person-b",
    )

    assert "Current context:\n- id: ctx-second" in result.prompt
    assert "source_context=ctx-first" in result.prompt
    assert any(
        decision.cross_context
        for decision in result.working_memory_view.decisions
        if decision.item_id.startswith("episode:")
    )


def test_prompt_includes_safe_attachment_metadata(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        _memory(settings),
    ).chat(
        "describe this",
        debug=True,
        attachments=[
            {
                "type": "image",
                "name": "image.png",
                "url": "file:///tmp/image.png",
                "content_type": "image/png",
                "ignored": "not shown",
            }
        ],
    )

    assert "Attachments:" in result.prompt
    assert "type=image" in result.prompt
    assert "name=image.png" in result.prompt
    assert "source=file" in result.prompt
    assert "file:///tmp/image.png" not in result.prompt
    assert "content_type=image/png" in result.prompt
    assert "ignored" not in result.prompt


def test_empty_visible_primary_response_uses_fallback_and_clears_adapter(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkOnlyPrimaryProvider()
    result = KagyaMainLoop(
        settings,
        provider,
        _memory(settings),
        adapter_id="adapter-primary",
    ).chat("hello", debug=True)

    assert result.response == "Fallback visible answer."
    assert result.model_id == "fallback-model"
    assert result.fallback_used is True
    assert result.adapter_id is None
    assert provider.fallback_calls == 1


def test_empty_visible_fallback_response_raises_runtime_error(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkOnlyPrimaryProvider(fallback_response="<think>still hidden</think>")
    loop = KagyaMainLoop(
        settings,
        provider,
        _memory(settings),
    )
    working_memory_before = loop.working_memory.items

    try:
        loop.chat("hello", debug=True)
    except RuntimeError as exc:
        assert "empty visible response" in str(exc)
        assert provider.fallback_calls == 1
        assert loop.working_memory.items == working_memory_before
        assert loop.surprisal_calculator.history == {}
    else:
        raise AssertionError("empty fallback output should fail")


def test_prompt_uses_plain_visible_answer_contract(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        _memory(settings),
    ).chat("answer naturally", debug=True)

    assert result.prompt.startswith("Context: PROJECT-KAGYA")
    assert "private local AI assistant" in result.prompt
    assert "Private runtime data below is for tone and context only" in result.prompt
    assert "Answer only the latest user message" in result.prompt
    assert "answer in natural Japanese" in result.prompt
    assert "User: answer naturally\nAssistant:" in result.prompt


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


def _memory(settings: Settings) -> DualMemorySystem:
    return DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )
