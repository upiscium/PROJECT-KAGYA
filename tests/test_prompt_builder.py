"""U4 PromptBuilder contract tests."""

from kagya.body import EmotionState
from kagya.identity import (
    ValueAdmissionStatus,
    ValuePromptEntry,
    ValuePromptView,
    ValueScope,
)
from kagya.persona import PromptBuilder
from kagya.runtime import (
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
)


def view(
    *selections: WorkingMemorySelection,
    decisions: tuple[WorkingMemoryDecision, ...] = (),
) -> WorkingMemoryView:
    return WorkingMemoryView(
        selected=selections,
        decisions=decisions,
        projected_bytes=0,
        item_capacity=4,
        projection_max_bytes=1024,
        revision=1,
    )


def selection(kind: WorkingMemorySourceKind, content: str) -> WorkingMemorySelection:
    return WorkingMemorySelection(
        item_id=f"item-{content}",
        source_kind=kind,
        source_id=f"source-{content}",
        rendered_content=content,
        score=0.5,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )


def test_build_uses_selected_rendered_content_only() -> None:
    prompt = PromptBuilder().build(
        "hello",
        EmotionState(),
        view(
            selection(WorkingMemorySourceKind.SEMANTIC, "semantic content"),
            selection(WorkingMemorySourceKind.EPISODIC, "episodic content"),
        ),
    )

    assert "semantic content" in prompt
    assert "episodic content" in prompt
    assert "source-semantic content" not in prompt
    assert "item-semantic content" not in prompt


def test_build_groups_selected_items_and_renders_empty_sections() -> None:
    prompt = PromptBuilder().build(
        "hello",
        EmotionState(),
        view(
            selection(WorkingMemorySourceKind.SEMANTIC, "semantic one"),
            selection(WorkingMemorySourceKind.SEMANTIC, "semantic two"),
        ),
    )

    assert prompt.index("Episodic memories:") < prompt.index("- none")
    assert prompt.index("semantic one") < prompt.index("semantic two")
    assert prompt.count("- none") == 1


def test_build_does_not_render_decision_content() -> None:
    decision = WorkingMemoryDecision(
        item_id="unresolved-item",
        source_kind=WorkingMemorySourceKind.EPISODIC,
        source_id="unresolved-source",
        selected=False,
        score=0.9,
        reason=WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
    )

    prompt = PromptBuilder().build("hello", EmotionState(), view(decisions=(decision,)))

    assert "unresolved-item" not in prompt
    assert "unresolved-source" not in prompt
    assert "unresolved_reference" not in prompt


def test_build_is_purely_repeatable() -> None:
    memory_view = view(
        selection(WorkingMemorySourceKind.SEMANTIC, "semantic"),
        selection(WorkingMemorySourceKind.EPISODIC, "episodic"),
    )

    first = PromptBuilder().build("hello", EmotionState(), memory_view)
    second = PromptBuilder().build("hello", EmotionState(), memory_view)

    assert first == second


def test_build_renders_bounded_active_value_projection_deterministically() -> None:
    value_view = ValuePromptView(
        (
            ValuePromptEntry(
                value_id="care",
                name="care",
                concept="Protect wellbeing.",
                polarity=1,
                strength=0.8,
                confidence=0.9,
                authority_class=ValueAdmissionStatus.SYSTEM_AUTHORIZED,
                scope=ValueScope.SUBJECT,
            ),
        ),
        context_id="conversation-default",
    )

    prompt = PromptBuilder().build(
        "hello",
        EmotionState(),
        view(),
        value_view=value_view,
    )

    assert "Active Values:" in prompt
    assert (
        "- value_id=care; authority=system_authorized; scope=subject; "
        "polarity=+1; strength=0.800000; confidence=0.900000; "
        "name=care; concept=Protect wellbeing."
    ) in prompt
    for forbidden in (
        "source_ref",
        "event_id",
        "event_sequence",
        "origin_id",
        "seed_contract_digest",
        "evidence_refs",
        "revision",
    ):
        assert forbidden not in prompt


def test_build_renders_explicit_empty_value_projection_and_omits_none() -> None:
    empty_prompt = PromptBuilder().build(
        "hello", EmotionState(), view(), value_view=ValuePromptView(())
    )
    omitted_prompt = PromptBuilder().build("hello", EmotionState(), view())

    assert "Active Values:\n- none" in empty_prompt
    assert "Active Values:" not in omitted_prompt
