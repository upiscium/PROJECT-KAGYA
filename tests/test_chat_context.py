"""U4 request-scoped Context selection and contextual projection tests."""

from datetime import UTC, datetime
import hashlib
from pathlib import Path

import pytest

from kagya.body import EmotionState
from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.memory.working_memory_resolver import MemoryWorkingMemoryResolver
from kagya.persona import ContextPromptView, PromptBuilder
from kagya.runtime import (
    ChatContextSelectors,
    ContextConflict,
    ContextNotFound,
    ContextRegistry,
    ContextRelation,
    ContextStateInvalid,
    ContextType,
    WorkingMemory,
    WorkingMemoryDecisionReason,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
    resolve_chat_context,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Clock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return NOW


def test_default_context_is_created_reused_and_participant_binding_is_bounded() -> None:
    clock = Clock()
    registry = ContextRegistry(clock=clock)

    first = resolve_chat_context(
        registry, ChatContextSelectors(interlocutor_key="interlocutor-a")
    )
    before = registry.state
    second = resolve_chat_context(
        registry, ChatContextSelectors(interlocutor_key="interlocutor-a")
    )

    assert first.context_id == second.context_id == "conversation.default"
    assert first.source_channel == second.source_channel == "chat"
    assert first.source_session_id is second.source_session_id is None
    assert second.participant_refs == ("interlocutor-a",)
    assert registry.current_context_id == "conversation.default"
    assert registry.state == before
    assert clock.calls == 1

    with pytest.raises(ContextConflict, match="interlocutor"):
        resolve_chat_context(
            registry, ChatContextSelectors(interlocutor_key="interlocutor-b")
        )
    assert registry.state == before


def test_session_context_id_is_deterministic_and_never_falls_back() -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    session_id = "browser:session-7"
    expected = (
        "conversation.session."
        + hashlib.sha256(
            b"PROJECT-KAGYA:R09:CHAT-SESSION:V1\0" + session_id.encode("ascii")
        ).hexdigest()
    )

    first = resolve_chat_context(
        registry,
        ChatContextSelectors(
            client_session_id=session_id, interlocutor_key="interlocutor-a"
        ),
    )
    before = registry.state
    second = resolve_chat_context(
        registry, ChatContextSelectors(client_session_id=session_id)
    )

    assert first.context_id == second.context_id == expected
    assert first.source_session_id == session_id
    assert second.participant_refs == ("interlocutor-a",)
    assert registry.state == before

    with pytest.raises(ContextConflict, match="session"):
        resolve_chat_context(
            registry,
            ChatContextSelectors(
                context_id=expected, client_session_id="other-session"
            ),
        )


def test_explicit_context_has_precedence_and_rejects_unknown_or_inactive() -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    frame = registry.create(
        "explicit-context",
        ContextType.CONVERSATION,
        "web",
        source_session_id="web-session",
    )

    selected = resolve_chat_context(
        registry,
        ChatContextSelectors(
            context_id=frame.context_id,
            client_session_id="web-session",
            interlocutor_key="interlocutor-a",
        ),
    )
    assert selected.source_channel == "web"
    assert selected.participant_refs == ("interlocutor-a",)
    assert registry.current_context_id == frame.context_id

    with pytest.raises(ContextNotFound, match="not found"):
        resolve_chat_context(
            registry, ChatContextSelectors(context_id="missing-context")
        )

    registry.suspend(frame.context_id)
    with pytest.raises(ContextStateInvalid, match="active"):
        resolve_chat_context(
            registry, ChatContextSelectors(context_id=frame.context_id)
        )


def test_duplicate_or_inactive_session_mappings_fail_without_mutation() -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    registry.create(
        "session-a",
        ContextType.CONVERSATION,
        "chat",
        source_session_id="client-session",
    )
    registry.create(
        "session-b",
        ContextType.CONVERSATION,
        "chat",
        source_session_id="client-session",
    )
    before = registry.state

    with pytest.raises(ContextConflict, match="multiple"):
        resolve_chat_context(
            registry, ChatContextSelectors(client_session_id="client-session")
        )
    assert registry.state == before

    registry = ContextRegistry(clock=lambda: NOW)
    frame = registry.create(
        "inactive-session",
        ContextType.CONVERSATION,
        "chat",
        source_session_id="client-session",
    )
    registry.suspend(frame.context_id)
    before = registry.state
    with pytest.raises(ContextStateInvalid, match="active"):
        resolve_chat_context(
            registry, ChatContextSelectors(client_session_id="client-session")
        )
    assert registry.state == before


def test_registry_session_lookup_and_participant_mutation_are_pure_or_idempotent() -> None:
    clock = Clock()
    registry = ContextRegistry(clock=clock)
    frame = registry.create(
        "lookup-context",
        ContextType.CONVERSATION,
        "chat",
        source_session_id="client-session",
    )
    before_lookup = registry.state

    assert registry.get(frame.context_id) is frame
    assert registry.find_by_source_session("chat", "client-session") == (frame,)
    assert registry.state == before_lookup
    assert clock.calls == 1

    updated = registry.add_participant_ref(frame.context_id, "ref-a")
    assert updated.participant_refs == ("ref-a",)
    assert registry.state.revision == before_lookup.revision + 1
    calls_after_update = clock.calls
    assert registry.add_participant_ref(frame.context_id, "ref-a") is updated
    assert registry.state.revision == before_lookup.revision + 1
    assert clock.calls == calls_after_update


def test_contextual_working_memory_uses_compatibility_without_mutating_authority() -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    current = registry.create(
        "current-context", ContextType.CONVERSATION, "chat", participant_refs=()
    )
    related = registry.create(
        "related-context", ContextType.CONVERSATION, "chat", participant_refs=()
    )
    registry.relate(current.context_id, related.context_id)

    memory = WorkingMemory(item_capacity=4, projection_max_bytes=100)
    memory.admit(WorkingMemorySourceKind.SEMANTIC, "same-source", 0.8, 0.8)
    memory.admit(WorkingMemorySourceKind.SEMANTIC, "related-source", 0.8, 0.8)
    memory.admit(WorkingMemorySourceKind.SEMANTIC, "legacy-source", 0.8, 0.8)
    memory.admit(WorkingMemorySourceKind.SEMANTIC, "unknown-source", 0.8, 0.8)
    before = (memory.revision, memory.items, registry.state)
    source_contexts = {
        "same-source": current.context_id,
        "related-source": related.context_id,
        "legacy-source": None,
        "unknown-source": "missing-context",
    }

    def resolver(item):
        return WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            f"body:{item.source_id}",
            source_contexts[item.source_id],
        )

    view = memory.select_contextual(resolver, registry, current.context_id)

    assert [selection.source_id for selection in view.selected] == [
        "same-source",
        "related-source",
        "legacy-source",
        "unknown-source",
    ]
    evidence = {selection.source_id: selection for selection in view.selected}
    assert evidence["same-source"].context_relation is ContextRelation.SAME_CONTEXT
    assert evidence["same-source"].context_compatibility == 1.0
    assert evidence["related-source"].context_relation is ContextRelation.RELATED
    assert evidence["related-source"].effective_score == pytest.approx(
        evidence["related-source"].score * 0.75
    )
    assert evidence["legacy-source"].context_relation is ContextRelation.LEGACY_UNKNOWN
    assert evidence["unknown-source"].context_relation is ContextRelation.UNKNOWN_CONTEXT
    assert (memory.revision, memory.items, registry.state) == before
    assert all(
        decision.reason is WorkingMemoryDecisionReason.SELECTED
        for decision in view.decisions
    )


def test_contextual_compatibility_can_reverse_r08_order_and_budget_packing() -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    current = registry.create("rank-current", ContextType.CONVERSATION, "chat")
    unrelated = registry.create("rank-unrelated", ContextType.CONVERSATION, "chat")
    source_contexts = {
        "item-a": unrelated.context_id,
        "item-b": current.context_id,
    }

    def resolver(item):
        return WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            f"body-{item.source_id[-1]}",
            source_contexts[item.source_id],
        )

    def populate(memory: WorkingMemory) -> None:
        memory.admit(WorkingMemorySourceKind.SEMANTIC, "item-a", 1.0, 1.0)
        memory.admit(WorkingMemorySourceKind.SEMANTIC, "item-b", 0.8, 0.75)

    ordinary = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    contextual = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    tight = WorkingMemory(item_capacity=2, projection_max_bytes=6)
    for memory in (ordinary, contextual, tight):
        populate(memory)

    ordinary_view = ordinary.select(resolver)
    contextual_view = contextual.select_contextual(
        resolver, registry, current.context_id
    )
    tight_view = tight.select_contextual(resolver, registry, current.context_id)

    ordinary_items = {item.source_id: item for item in ordinary.items}
    assert ordinary.score(ordinary_items["item-a"]) > ordinary.score(
        ordinary_items["item-b"]
    )
    ordinary_order = [selection.source_id for selection in ordinary_view.selected]
    contextual_order = [selection.source_id for selection in contextual_view.selected]
    assert ordinary_order == ["item-a", "item-b"]
    assert contextual_order == ["item-b", "item-a"]
    contextual_evidence = {
        decision.source_id: decision for decision in contextual_view.decisions
    }
    assert contextual_evidence["item-a"].effective_score == pytest.approx(0.2)
    assert contextual_evidence["item-b"].effective_score == pytest.approx(0.78 * 1.0)
    assert contextual_evidence["item-a"].effective_score < (
        contextual_evidence["item-b"].effective_score or 0.0
    )
    assert [selection.source_id for selection in tight_view.selected] == ["item-b"]
    assert [decision.source_id for decision in tight_view.decisions] == [
        "item-b",
        "item-a",
    ]
    assert tight_view.decisions[1].reason is WorkingMemoryDecisionReason.PROJECTION_BUDGET


@pytest.mark.parametrize(
    ("relation", "expected_score"),
    [
        (ContextRelation.SAME_CONTEXT, 1.0),
        (ContextRelation.PARENT_CHILD, 0.8),
        (ContextRelation.RELATED, 0.75),
        (ContextRelation.SHARED_INTERLOCUTOR, 0.65),
        (ContextRelation.LEGACY_UNKNOWN, 0.45),
        (ContextRelation.UNKNOWN_CONTEXT, 0.35),
        (ContextRelation.UNRELATED, 0.2),
    ],
)
def test_all_compatibility_classes_cross_the_contextual_projection_path(
    relation: ContextRelation, expected_score: float
) -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    parent = registry.create("relation-parent", ContextType.CONVERSATION, "chat")
    current = registry.create(
        "relation-current",
        ContextType.CONVERSATION,
        "chat",
        parent_context_id=parent.context_id,
        participant_refs=("shared-ref",),
    )
    related = registry.create("relation-related", ContextType.CONVERSATION, "chat")
    registry.relate(current.context_id, related.context_id)
    shared = registry.create(
        "relation-shared",
        ContextType.CONVERSATION,
        "chat",
        participant_refs=("shared-ref",),
    )
    unrelated = registry.create("relation-unrelated", ContextType.CONVERSATION, "chat")
    source_contexts = {
        ContextRelation.SAME_CONTEXT: current.context_id,
        ContextRelation.PARENT_CHILD: parent.context_id,
        ContextRelation.RELATED: related.context_id,
        ContextRelation.SHARED_INTERLOCUTOR: shared.context_id,
        ContextRelation.LEGACY_UNKNOWN: None,
        ContextRelation.UNKNOWN_CONTEXT: "relation-unknown",
        ContextRelation.UNRELATED: unrelated.context_id,
    }
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    source_id = f"relation-{relation.value}"
    memory.admit(WorkingMemorySourceKind.SEMANTIC, source_id, 1.0, 1.0)
    view = memory.select_contextual(
        lambda _item: WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            "projection body",
            source_contexts[relation],
        ),
        registry,
        current.context_id,
    )
    selection = view.selected[0]
    prompt = PromptBuilder().build("hello", EmotionState(), view)

    assert selection.context_relation is relation
    assert selection.context_compatibility == pytest.approx(expected_score)
    assert f"context_relation={relation.value}" in prompt
    assert f"compatibility={expected_score:.2f}" in prompt


def test_real_semantic_provenance_reaches_contextual_selection(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(settings)
    registry = ContextRegistry(clock=lambda: NOW)
    current = registry.create("semantic-current", ContextType.CONVERSATION, "chat")
    episode_id = "episode-semantic-context"
    memory.publish_coordinated_episodic(
        episode_id,
        "source input",
        "source response",
        loss=0.1,
        emotion_valence=0.2,
        emotion_arousal=0.3,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at=NOW.isoformat(),
        coordination_schema=2,
        context_id=current.context_id,
        source_channel="chat",
    )
    semantic_id = memory.save_semantic(
        "semantic body", source_episode_ids=[episode_id]
    )
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    working.admit(WorkingMemorySourceKind.SEMANTIC, semantic_id, 1.0, 1.0)
    resolver = MemoryWorkingMemoryResolver(memory)

    resolution = resolver.resolve(working.items[0])
    view = working.select_contextual(resolver, registry, current.context_id)

    assert resolution.source_context_id == current.context_id
    assert view.selected[0].source_context_id == current.context_id
    assert view.selected[0].context_relation is ContextRelation.SAME_CONTEXT
    assert view.selected[0].context_compatibility == 1.0


def test_context_prompt_view_is_bounded_and_memory_labels_are_ephemeral() -> None:
    registry = ContextRegistry(clock=lambda: NOW)
    frame = registry.create(
        "prompt-context",
        ContextType.CONVERSATION,
        "chat",
        source_session_id="client-session",
        participant_refs=("ref-a",),
    )
    prompt_view = ContextPromptView.from_frame(frame)
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    memory.admit(WorkingMemorySourceKind.EPISODIC, "episode-source", 1.0, 1.0)
    view = memory.select_contextual(
        lambda item: WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            "bounded body",
            frame.context_id,
        ),
        registry,
        frame.context_id,
    )
    prompt = PromptBuilder().build("hello", EmotionState(), view, prompt_view)

    assert "context_id: prompt-context" in prompt
    assert "source_session_id: client-session" in prompt
    assert "participant_refs: ref-a" in prompt
    assert "context_relation=same_context" in prompt
    assert "compatibility=1.00" in prompt
    assert "episode-source" not in prompt
    assert "bounded body" in prompt


def _settings_for_tmp_memory(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_context_chat_test",
                    "db2_collection": "cortex_context_chat_test",
                }
            )
        }
    )
