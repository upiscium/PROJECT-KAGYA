from dataclasses import fields
import math
from pathlib import Path

import pytest

from kagya.cognition import LossCalibration, LossInvalidReason, model_key
from kagya.config import Settings, load_settings
from kagya.memory import (
    DualMemorySystem,
    EpisodicMemoryRecord,
    MemoryContext,
    SemanticMemoryFormatError,
    SemanticMemoryReadError,
    SemanticMemoryRecord,
)
from kagya.runtime.context import ContextType
from kagya.models import DummyProvider
from kagya.persona import PromptBuilder
from kagya.runtime import (
    ChatContextSelectors,
    CoordinatedResult,
    ContextRegistry,
    KagyaMainLoop,
    TransactionBinding,
    TransactionBoundValue,
    TransactionKind,
    WorkingMemory,
    WorkingMemoryDecisionReason,
    WorkingMemoryRetentionReason,
    WorkingMemorySourceKind,
)
from kagya.runtime.main_loop import DebugChatTrace


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


class NonFiniteLossProvider(ThinkingDummyProvider):
    loss_value = math.nan


class InfiniteLossProvider(ThinkingDummyProvider):
    loss_value = math.inf


class RaisingLossProvider(ThinkingDummyProvider):
    def calculate_loss(self, context_text: str, target_text: str) -> float:
        raise ValueError(PRIVATE_SENTINEL)


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


def test_main_loop_accepts_context_registry_without_creating_or_selecting_context(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    registry = ContextRegistry()
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
        context_registry=registry,
    )

    assert loop.context_registry is registry
    assert registry.state.revision == 0
    assert registry.current_context_id is None


def test_chat_computation_uses_live_context_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    registry = ContextRegistry()
    memory = DualMemorySystem(settings)
    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db2_results=[SemanticMemoryRecord("live-source", "live body")]
        ),
    )
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        memory,
        context_registry=registry,
    )
    before = registry.state
    before_working_memory = (loop.working_memory.revision, loop.working_memory.items)
    before_emotion = loop.emotion_engine.state

    plan = loop.chat("live turn")

    assert registry.state != before
    assert registry.current_context_id == "conversation.default"
    assert loop.working_memory.revision > before_working_memory[0]
    assert loop.working_memory.items != before_working_memory[1]
    assert loop.emotion_engine.state != before_emotion
    assert plan.participants[0].operation.context_id == "conversation.default"

    debug_loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        memory,
        context_registry=ContextRegistry(),
    )
    debug_before = debug_loop.emotion_engine.state
    debug_loop.chat_debug("live debug turn")
    assert debug_loop.context_registry.current_context_id == "conversation.default"
    assert debug_loop.working_memory.revision > 0
    assert debug_loop.working_memory.items
    assert debug_loop.emotion_engine.state != debug_before


def test_ordinary_and_debug_chat_use_working_memory_without_prompt_mutation(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    working_memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    memory = DualMemorySystem(settings)
    passive_id = memory.save_episodic("passive", "memory")
    working_memory.admit(
        WorkingMemorySourceKind.EPISODIC,
        passive_id,
        0.8,
        0.8,
    )
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        memory,
        working_memory=working_memory,
    )
    ordinary = _materialize(loop.chat("ordinary"))
    debug_result, trace = _materialize(loop.chat_debug("debug"))

    assert ordinary.response == debug_result.response == "Visible runtime answer."
    assert "User: debug\nAssistant:" in trace.prompt
    assert trace.working_memory_view.revision == working_memory.revision
    assert trace.working_memory_view.selected[0].source_id == passive_id


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


def test_main_loop_resolves_committed_body_and_passes_view_to_prompt_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    semantic_id = memory.save_semantic("committed body")
    captured: list[object] = []

    class CapturingPromptBuilder:
        def build(self, user_input, emotion_state, working_memory_view):
            captured.append(working_memory_view)
            return PromptBuilder().build(user_input, emotion_state, working_memory_view)

    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db2_results=[SemanticMemoryRecord(semantic_id, "stale retrieval body")]
        ),
    )
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        memory,
        prompt_builder=CapturingPromptBuilder(),
    )

    _result, trace = _materialize(loop.chat_debug("query"))

    assert len(captured) == 1
    assert captured[0] is trace.working_memory_view
    assert [selection.rendered_content for selection in trace.working_memory_view.selected] == [
        "committed body"
    ]
    assert "committed body" in trace.prompt
    assert "stale retrieval body" not in trace.prompt


def test_retrieval_failure_does_not_age_working_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    working_memory = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    working_memory.admit(WorkingMemorySourceKind.SEMANTIC, "existing", 1.0, 1.0)
    memory = DualMemorySystem(settings)

    def fail(_query: str) -> MemoryContext:
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(memory, "retrieve_context", fail)
    loop = KagyaMainLoop(
        settings, ThinkingDummyProvider(), memory, working_memory=working_memory
    )
    before = (working_memory.revision, working_memory.items)

    with pytest.raises(RuntimeError, match="retrieval failed"):
        loop.chat("query")

    assert (working_memory.revision, working_memory.items) == before


def test_retrieval_candidates_are_admitted_in_exact_cross_kind_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    context = MemoryContext(
        db1_results=[
            EpisodicMemoryRecord("episode-rank-0", "ignored", "ignored"),
            EpisodicMemoryRecord("episode-rank-1", "ignored", "ignored"),
        ],
        db2_results=[
            SemanticMemoryRecord("semantic-rank-0", "ignored"),
            SemanticMemoryRecord("semantic-rank-1", "ignored"),
        ],
    )
    monkeypatch.setattr(memory, "retrieve_context", lambda _query: context)
    working = WorkingMemory(item_capacity=4, projection_max_bytes=100)
    loop = KagyaMainLoop(
        settings, ThinkingDummyProvider(), memory, working_memory=working
    )

    loop.chat("rank candidates")

    by_source = {item.source_id: item for item in working.items}
    assert working.revision == 4
    assert len(working.items) == working.item_capacity == 4
    assert {
        source_id: (
            item.activation,
            item.salience,
            item.created_revision,
            item.last_activated_revision,
        )
        for source_id, item in by_source.items()
    } == {
        "episode-rank-1": (1.0, 0.5, 1, 1),
        "semantic-rank-1": (1.0, 0.5, 2, 2),
        "episode-rank-0": (1.0, 1.0, 3, 3),
        "semantic-rank-0": (1.0, 1.0, 4, 4),
    }


def test_prior_working_memory_decays_and_retrieved_reference_reactivates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    retrieved_id = memory.save_semantic("retrieved exact body")
    working = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    working.admit(WorkingMemorySourceKind.SEMANTIC, retrieved_id, 0.5, 0.4)
    working.admit(WorkingMemorySourceKind.SEMANTIC, "semantic-not-retrieved", 0.5, 0.4)
    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db2_results=[SemanticMemoryRecord(retrieved_id, "ignored stale body")]
        ),
    )
    loop = KagyaMainLoop(
        settings, ThinkingDummyProvider(), memory, working_memory=working
    )

    loop.chat("reactivate")

    by_source = {item.source_id: item for item in working.items}
    assert working.revision == 4
    assert by_source[retrieved_id].activation == 1.0
    assert by_source[retrieved_id].salience == 1.0
    assert by_source[retrieved_id].retention_reason is (
        WorkingMemoryRetentionReason.REACTIVATED
    )
    assert by_source["semantic-not-retrieved"].activation == pytest.approx(0.35)
    assert by_source["semantic-not-retrieved"].retention_reason is (
        WorkingMemoryRetentionReason.RECENT
    )


def test_oversized_exact_source_is_excluded_and_smaller_source_is_prompted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    large_id = memory.save_semantic("X" * 20)
    small_id = memory.save_semantic("fits")
    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db2_results=[
                SemanticMemoryRecord(large_id, "untrusted large"),
                SemanticMemoryRecord(small_id, "untrusted small"),
            ]
        ),
    )
    working = WorkingMemory(item_capacity=2, projection_max_bytes=5)
    loop = KagyaMainLoop(
        settings, ThinkingDummyProvider(), memory, working_memory=working
    )

    _result, trace = _materialize(loop.chat_debug("budget query"))

    assert [decision.reason for decision in trace.working_memory_view.decisions] == [
        WorkingMemoryDecisionReason.PROJECTION_BUDGET,
        WorkingMemoryDecisionReason.SELECTED,
    ]
    assert "X" * 20 not in trace.prompt
    assert "fits" in trace.prompt
    assert trace.working_memory_view.projected_bytes == 4


def test_missing_and_archived_references_remain_but_do_not_enter_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    archived_id = memory.save_episodic("archived user", "archived answer")
    memory._archive_episodic(archived_id)
    missing_id = "semantic-disappeared"
    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db1_results=[
                EpisodicMemoryRecord(
                    archived_id, "untrusted archived", "untrusted archived"
                )
            ],
            db2_results=[SemanticMemoryRecord(missing_id, "untrusted missing")],
        ),
    )
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result, trace = _materialize(loop.chat_debug("status query"))

    assert result.response == "Visible runtime answer."
    assert {item.source_id for item in loop.working_memory.items} == {
        archived_id,
        missing_id,
    }
    assert {decision.source_id: decision.reason for decision in trace.working_memory_view.decisions} == {
        archived_id: WorkingMemoryDecisionReason.SOURCE_ARCHIVED,
        missing_id: WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
    }
    assert "archived user" not in trace.prompt
    assert "untrusted archived" not in trace.prompt
    assert "untrusted missing" not in trace.prompt


@pytest.mark.parametrize(
    ("error_type", "expected_reason"),
    [
        (SemanticMemoryReadError, WorkingMemoryDecisionReason.SOURCE_UNAVAILABLE),
        (SemanticMemoryFormatError, WorkingMemoryDecisionReason.SOURCE_MALFORMED),
    ],
)
def test_exact_source_failure_is_bounded_and_chat_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected_reason: WorkingMemoryDecisionReason,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    source_id = memory.save_semantic("source body must stay absent")
    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db2_results=[SemanticMemoryRecord(source_id, "untrusted retrieval body")]
        ),
    )

    def fail_exact(_source_id: str) -> None:
        raise error_type("PRIVATE RAW SOURCE DETAIL")

    monkeypatch.setattr(memory, "get_committed_semantic", fail_exact)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result, trace = _materialize(loop.chat_debug("failure query"))

    assert result.response == "Visible runtime answer."
    assert trace.working_memory_view.decisions[0].reason is expected_reason
    assert trace.working_memory_view.selected == ()
    assert source_id in {item.source_id for item in loop.working_memory.items}
    assert "source body must stay absent" not in trace.prompt
    assert "untrusted retrieval body" not in trace.prompt
    assert "PRIVATE RAW SOURCE DETAIL" not in trace.prompt
    assert "PRIVATE RAW SOURCE DETAIL" not in repr(result)


def test_malformed_exact_source_is_not_prompted_or_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    source_id = memory.save_semantic("authoritative document")
    stored = memory.db2.get(ids=[source_id], include=["metadatas"])
    metadata = dict(stored["metadatas"][0])
    metadata["text"] = "conflicting metadata body"
    memory.db2.update(ids=[source_id], metadatas=[metadata])
    before = memory.db2.get(
        ids=[source_id], include=["documents", "metadatas"]
    )
    monkeypatch.setattr(
        memory,
        "retrieve_context",
        lambda _query: MemoryContext(
            db2_results=[SemanticMemoryRecord(source_id, "untrusted retrieval body")]
        ),
    )
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result, trace = _materialize(loop.chat_debug("malformed query"))

    assert result.response == "Visible runtime answer."
    assert trace.working_memory_view.decisions[0].reason is (
        WorkingMemoryDecisionReason.SOURCE_MALFORMED
    )
    assert trace.working_memory_view.selected == ()
    assert "authoritative document" not in trace.prompt
    assert "conflicting metadata body" not in trace.prompt
    assert "untrusted retrieval body" not in trace.prompt
    assert memory.db2.get(
        ids=[source_id], include=["documents", "metadatas"]
    ) == before


def test_select_and_prompt_build_are_pure_after_explicit_chat_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    semantic_id = memory.save_semantic("pure source")
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    loop = KagyaMainLoop(
        settings, ThinkingDummyProvider(), memory, working_memory=working
    )
    original_select = working.select_contextual
    selection_states: list[tuple[object, object]] = []

    def checked_select(resolver, context_registry, current_context_id):
        before = (working.revision, working.items)
        view = original_select(resolver, context_registry, current_context_id)
        selection_states.append((before, (working.revision, working.items)))
        return view

    monkeypatch.setattr(working, "select_contextual", checked_select)

    class PurePromptBuilder:
        def build(self, user_input, emotion_state, working_memory_view):
            before = (working.revision, working.items)
            prompt = PromptBuilder().build(
                user_input, emotion_state, working_memory_view
            )
            assert (working.revision, working.items) == before
            return prompt

    loop.prompt_builder = PurePromptBuilder()  # type: ignore[assignment]

    loop.chat("pure source")

    assert selection_states == [(selection_states[0][0], selection_states[0][0])]
    assert {item.source_id for item in working.items} == {semantic_id}


def test_current_future_episode_is_absent_from_its_own_working_memory_view(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(
        settings, ThinkingDummyProvider(), DualMemorySystem(settings)
    )

    result, trace = _materialize(loop.chat_debug("current turn only"))

    assert result.episode_id not in {
        selection.source_id for selection in trace.working_memory_view.selected
    }
    assert result.episode_id not in {item.source_id for item in loop.working_memory.items}
    assert result.response not in trace.prompt


def test_ordinary_and_debug_apply_equivalent_working_memory_semantics(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    ordinary_working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    debug_working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    for working in (ordinary_working, debug_working):
        working.admit(
            WorkingMemorySourceKind.SEMANTIC,
            "semantic-shared-missing",
            0.6,
            0.4,
        )
    ordinary_loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
        working_memory=ordinary_working,
    )
    debug_settings = _settings_for_tmp_memory(tmp_path / "debug")
    debug_loop = KagyaMainLoop(
        debug_settings,
        ThinkingDummyProvider(),
        DualMemorySystem(debug_settings),
        working_memory=debug_working,
    )

    ordinary_plan = ordinary_loop.chat("same input")
    debug_plan = debug_loop.chat_debug("same input")
    ordinary = _materialize(ordinary_plan)
    debug_result, trace = _materialize(debug_plan)

    assert (
        ordinary.response,
        ordinary.loss,
        ordinary.valence,
        ordinary.arousal,
        ordinary.optimal_loss,
        ordinary.model_id,
        ordinary.adapter_id,
    ) == (
        debug_result.response,
        debug_result.loss,
        debug_result.valence,
        debug_result.arousal,
        debug_result.optimal_loss,
        debug_result.model_id,
        debug_result.adapter_id,
    )
    assert ordinary_working.items == debug_working.items
    assert ordinary_working.revision == debug_working.revision
    assert [participant.participant_id for participant in ordinary_plan.participants] == [
        participant.participant_id for participant in debug_plan.participants
    ] == ["memory.episodic", "session.turn"]
    assert trace.working_memory_view.revision == debug_working.revision
    assert tuple(field.name for field in fields(DebugChatTrace)) == (
        "hidden_thought",
        "prompt",
        "memory_context",
        "working_memory_view",
        "diagnostics",
    )


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


def test_main_loop_default_loss_calibration_uses_configured_model_keys(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))

    expected = tuple(
        sorted(
            {
                model_key(settings.model.provider, settings.model.primary_id),
                model_key(settings.model.provider, settings.model.fallback_id),
            }
        )
    )
    assert loop.loss_calibration.approved_keys == expected
    assert len(loop.loss_calibration.approved_keys) == len(set(expected))
    assert (
        loop.loss_calibration._initial_baseline
        == settings.emotion.baseline_surprisal
    )
    assert loop.loss_calibration._initial_scale == settings.appraisal.initial_loss_scale
    assert loop.loss_calibration._minimum_scale == settings.appraisal.minimum_loss_scale


def test_main_loop_accepts_only_exactly_matching_injected_calibration(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    approved = tuple(
        sorted(
            {
                model_key(settings.model.provider, settings.model.primary_id),
                model_key(settings.model.provider, settings.model.fallback_id),
            }
        )
    )
    calibration = LossCalibration(
        approved, initial_baseline=1.0, initial_scale=1.0, minimum_scale=0.01
    )
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
        loss_calibration=calibration,
    )
    assert loop.loss_calibration is calibration

    mismatched = LossCalibration(
        (model_key(settings.model.provider, "other-model"),),
        initial_baseline=1.0,
        initial_scale=1.0,
        minimum_scale=0.01,
    )
    with pytest.raises(ValueError, match="approved keys"):
        KagyaMainLoop(
            settings,
            ThinkingDummyProvider(),
            DualMemorySystem(settings),
            loss_calibration=mismatched,
        )


def test_main_loop_passes_u2_emotion_policy_to_default_engine(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))

    assert loop.emotion_engine.adaptation_rate == settings.emotion.decay_rate
    assert (
        loop.emotion_engine.appraisal_response_rate
        == settings.emotion.appraisal_response_rate
    )
    assert loop.emotion_engine.resting_valence == settings.emotion.resting_valence
    assert loop.emotion_engine.resting_arousal == settings.emotion.resting_arousal
    assert (
        loop.emotion_engine.valence_recovery_rate
        == settings.emotion.valence_recovery_rate
    )
    assert (
        loop.emotion_engine.arousal_recovery_rate
        == settings.emotion.arousal_recovery_rate
    )


def test_main_loop_ordinary_chat_uses_structured_appraisal_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))
    calls: list[tuple[str, object]] = []

    def fail_legacy(*_args: object, **_kwargs: object) -> None:
        pytest.fail("ordinary chat used the legacy raw-loss emotion path")

    original_advance_to = loop.emotion_engine.advance_to
    original_measure = loop.surprisal_calculator.measure
    original_appraise = loop.appraiser.appraise
    original_update = loop.emotion_engine.update_from_appraisal

    def advance_to():
        calls.append(("advance_to", None))
        return original_advance_to()

    def measure(context: str, target: str, *, model_key: str, calibration):
        calls.append(("measure", (context, target, model_key)))
        return original_measure(
            context, target, model_key=model_key, calibration=calibration
        )

    def appraise(_appraiser, measurement, signals):
        calls.append(("appraise", signals))
        return original_appraise(measurement, signals)

    def update(appraisal, *, primary_loss: float | None = None):
        calls.append(("update_from_appraisal", primary_loss))
        return original_update(appraisal, primary_loss=primary_loss)

    monkeypatch.setattr(loop.surprisal_calculator, "calculate", fail_legacy)
    monkeypatch.setattr(loop.emotion_engine, "update", fail_legacy)
    monkeypatch.setattr(loop.emotion_engine, "advance_to", advance_to)
    monkeypatch.setattr(loop.surprisal_calculator, "measure", measure)
    monkeypatch.setattr(type(loop.appraiser), "appraise", appraise)
    monkeypatch.setattr(
        loop.emotion_engine, "update_from_appraisal", update
    )
    plan = loop.chat("structured path")

    assert [name for name, _value in calls] == [
        "advance_to",
        "measure",
        "appraise",
        "update_from_appraisal",
    ]
    assert calls[1][1][2] == loop.primary_model_key
    signals = calls[2][1]
    assert all(getattr(signals, name) is None for name in (
        "goal_progress",
        "threat",
        "controllability",
        "certainty",
        "social_relevance",
        "effort_cost",
    ))
    assert calls[3] == ("update_from_appraisal", DummyProvider.loss_value)
    assert len(loop.loss_calibration.export()) == 1
    assert plan.participants[0].operation.schema_version == 3


@pytest.mark.parametrize(
    ("provider_type", "invalid_reason"),
    [
        (NonFiniteLossProvider, LossInvalidReason.NON_FINITE_LOSS),
        (InfiniteLossProvider, LossInvalidReason.NON_FINITE_LOSS),
        (RaisingLossProvider, LossInvalidReason.PROVIDER_ERROR),
    ],
)
def test_invalid_loss_generates_without_calibration_or_optimal_loss_mutation(
    tmp_path: Path,
    provider_type: type[ThinkingDummyProvider],
    invalid_reason: LossInvalidReason,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    loop = KagyaMainLoop(settings, provider_type(), memory)
    before_state = loop.emotion_engine.state
    before_calibration = loop.loss_calibration.export()

    plan = loop.chat_debug("invalid loss")
    result, trace = _materialize(plan)

    assert result.response == "Visible runtime answer."
    assert result.loss is None
    assert loop.loss_calibration.export() == before_calibration
    assert loop.emotion_engine.state == before_state
    assert trace.diagnostics.measurement.valid is False
    assert trace.diagnostics.measurement.invalid_reason is invalid_reason
    assert trace.diagnostics.measurement.raw_loss is None
    assert trace.diagnostics.measurement.calibrated_novelty is None
    assert trace.diagnostics.appraisal.novelty is None
    assert trace.diagnostics.appraisal.novelty_valid is False
    assert all(
        getattr(trace.diagnostics.appraisal, name) is None
        for name in (
            "goal_progress",
            "threat",
            "controllability",
            "certainty",
            "social_relevance",
            "effort_cost",
        )
    )
    assert PRIVATE_SENTINEL not in repr(trace.diagnostics)
    assert plan.participants[0].operation.loss is None
    assert plan.participants[0].operation.schema_version == 3


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


def test_chat_uses_default_context_provenance(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), DualMemorySystem(settings))

    plan = loop.chat("no context")

    assert plan.participants[0].operation.context_id == "conversation.default"
    assert plan.participants[0].operation.source_channel == "chat"
    assert plan.participants[0].operation.source_session_id is None
    assert plan.participants[0].operation.schema_version == 3


def test_chat_captures_current_context_once_and_freezes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    registry = ContextRegistry()
    frame = registry.create(
        "context-a", ContextType.CONVERSATION, "chat", "session-a"
    )
    registry.set_current(frame.context_id)
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
        context_registry=registry,
    )
    reads = 0
    current = type(registry).current_context.fget

    def counted(registry):
        nonlocal reads
        reads += 1
        return current(registry)

    monkeypatch.setattr(type(registry), "current_context", property(counted))
    plan = loop.chat(
        "capture", selectors=ChatContextSelectors(context_id="context-a")
    )

    assert reads == 0
    operation = plan.participants[0].operation
    assert (operation.context_id, operation.source_channel, operation.source_session_id) == (
        "context-a", "chat", "session-a"
    )


def test_chat_context_switch_after_compute_does_not_change_frozen_provenance(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    registry = ContextRegistry()
    first = registry.create("context-a", ContextType.CONVERSATION, "chat")
    second = registry.create("context-b", ContextType.CONVERSATION, "web")
    registry.set_current(first.context_id)
    loop = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        DualMemorySystem(settings),
        context_registry=registry,
    )

    plan = loop.chat(
        "freeze", selectors=ChatContextSelectors(context_id="context-a")
    )
    registry.set_current(second.context_id)
    participant = plan.participants[0]
    binding = TransactionBinding(
        transaction_id="f51090e6-25a3-5d8e-b701-cbbdb8e88dca",
        event_id="5cefdcd0-88a3-5850-b6cc-72cab6f9989e",
        processing_sequence=1,
        participant_id=participant.participant_id,
        operation_digest=participant.operation_digest,
        transaction_kind=TransactionKind.EVENT_MUTATION,
    )
    participant.prepare(binding)
    participant.finalize(binding)
    committed = loop.memory_system.get_committed_episodic(
        participant.episode_id(binding.transaction_id)
    )

    assert committed is not None
    assert committed.record.context_id == "context-a"
    assert committed.record.source_channel == "chat"


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
