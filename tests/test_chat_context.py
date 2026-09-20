"""U4 request-scoped Context selection and contextual projection tests."""

from datetime import UTC, datetime
import hashlib

import pytest

from kagya.body import EmotionState
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
