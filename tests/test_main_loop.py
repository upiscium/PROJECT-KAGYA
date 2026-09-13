from pathlib import Path

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import (
    CoordinatedResult,
    KagyaMainLoop,
    TransactionBoundValue,
    WorkingMemory,
    WorkingMemorySourceKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"
TRANSACTION_ID = "d16db71e-0d94-5c0a-b827-375a90ab6404"


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

    plan = loop.chat("hello")
    result = _materialize(plan)

    assert result.response == "Visible runtime answer."
    assert result.loss == DummyProvider.loss_value
    assert result.episode_id.startswith("episode-")
    assert result.model_id == settings.model.primary_id
    assert result.adapter_id is None
    assert not hasattr(result, "hidden_thought")
    assert not hasattr(result, "prompt")
    assert not hasattr(result, "memory_context")


def test_main_loop_passively_owns_configured_or_injected_working_memory(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    configured = KagyaMainLoop(settings, provider, DualMemorySystem(settings))
    injected = WorkingMemory(item_capacity=1, projection_max_bytes=7)
    explicit = KagyaMainLoop(
        settings, provider, DualMemorySystem(settings), working_memory=injected
    )

    assert (
        configured.working_memory.item_capacity
        == settings.working_memory.item_capacity
    )
    assert (
        configured.working_memory.projection_max_bytes
        == settings.working_memory.projection_max_bytes
    )
    assert explicit.working_memory is injected


def test_ordinary_and_debug_chat_leave_working_memory_untouched(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    working_memory = WorkingMemory(item_capacity=2, projection_max_bytes=20)
    working_memory.admit(
        WorkingMemorySourceKind.EPISODIC,
        "episode-passive",
        0.8,
        0.8,
    )
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
        working_memory=working_memory,
    )
    before = (working_memory.revision, working_memory.items)

    ordinary = _materialize(loop.chat("ordinary"))
    debug_result, trace = _materialize(loop.chat_debug("debug"))

    assert ordinary.response == debug_result.response == "Visible runtime answer."
    assert "User: debug\nAssistant:" in trace.prompt
    assert (working_memory.revision, working_memory.items) == before


def test_debug_trace_exposes_private_thought_only_ephemerally(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
    )

    result, trace = _materialize(loop.chat_debug("inspect this turn"))

    assert result.response == "Visible runtime answer."
    assert trace.hidden_thought == PRIVATE_SENTINEL
    assert PRIVATE_SENTINEL not in str(result)
    assert "Assistant:" in trace.prompt


def test_computation_does_not_write_memory_or_session(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    plan = loop.chat_debug("remember this")
    result, trace = _materialize(plan)
    stored = memory.db1.get(include=["documents", "metadatas"])

    assert trace.hidden_thought == PRIVATE_SENTINEL
    assert result.episode_id.startswith("episode-")
    assert stored["ids"] == []
    assert loop.session_state.turns == []
    assert not (settings.memory.persist_directory / ".r07-episodic-pending").exists()
    assert PRIVATE_SENTINEL not in str(stored)


def test_visible_response_does_not_contain_think_tags_or_private_sentinel(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = _materialize(
        KagyaMainLoop(
            settings,
            ThinkingDummyProvider(),
            DualMemorySystem(settings),
        ).chat("hello")
    )

    assert "<think>" not in result.response
    assert "</think>" not in result.response
    assert PRIVATE_SENTINEL not in result.response


def test_emotion_state_changes_after_loss_calculation(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))
    before = loop.emotion_engine.state

    result = _materialize(loop.chat("emotion update"))

    assert result.arousal != before.arousal
    assert result.optimal_loss != before.optimal_loss


def test_prompt_includes_emotion_and_retrieved_memory(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    memory = DualMemorySystem(settings)
    memory.save_episodic("old episode", "old answer")
    memory.save_semantic("stable semantic memory")
    loop = KagyaMainLoop(settings, provider, memory)

    _result, trace = _materialize(loop.chat_debug("old semantic query"))

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
    _result, trace = _materialize(
        KagyaMainLoop(
            settings,
            ThinkingDummyProvider(),
            DualMemorySystem(settings),
        ).chat_debug("answer naturally")
    )

    assert trace.prompt.startswith("Context: PROJECT-KAGYA")
    assert "private local AI assistant" in trace.prompt
    assert "Private runtime data below is for tone and context only" in trace.prompt
    assert "User: answer naturally\nAssistant:" in trace.prompt


def test_chat_plan_has_fixed_memory_then_session_participants(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))

    plan = loop.chat("plan")

    assert [participant.participant_id for participant in plan.participants] == [
        "memory.episodic",
        "session.turn",
    ]
    assert loop.session_state.turns == []


def _materialize(plan: CoordinatedResult[object]):
    assert isinstance(plan.value, TransactionBoundValue)
    return plan.value.materialize(TRANSACTION_ID)


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
