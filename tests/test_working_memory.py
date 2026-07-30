from datetime import UTC, datetime, timedelta

import pytest

from kagya.runtime import (
    RetentionReason,
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryKind,
    working_memory_item,
)


def test_item_requires_content_or_reference() -> None:
    with pytest.raises(ValueError, match="requires content or reference"):
        working_memory_item(item_id="empty", kind=WorkingMemoryKind.CONVERSATION)


def test_item_capacity_evicts_lowest_ranked_item() -> None:
    memory = WorkingMemory(item_capacity=2, token_capacity=100)
    memory.admit(_item("low", activation=0.1, salience=0.1))
    memory.admit(_item("high", activation=0.9, salience=0.9))
    memory.admit(_item("middle", activation=0.5, salience=0.5))

    assert {item.item_id for item in memory.items} == {"high", "middle"}


def test_low_attention_candidate_is_not_admitted() -> None:
    memory = WorkingMemory(item_capacity=2, token_capacity=100)

    memory.admit(_item("noise", activation=0.1, salience=0.1))

    assert memory.items == ()


def test_token_capacity_limits_selected_view_with_reason() -> None:
    memory = WorkingMemory(
        item_capacity=3,
        token_capacity=5,
        token_counter=len,
    )
    memory.admit(_item("first", content="12345", activation=1.0))
    memory.admit(_item("second", content="67890", activation=0.5))

    view = memory.select()

    assert [selection.item.item_id for selection in view.selected] == ["first"]
    rejected = next(decision for decision in view.decisions if decision.item_id == "second")
    assert rejected.selected is False
    assert "token_capacity" in rejected.reasons


def test_token_count_does_not_mutate_items_or_resolve_references() -> None:
    memory = WorkingMemory(item_capacity=3, token_capacity=100, token_counter=len)
    memory.admit(_item("inline", content="12345"))
    memory.admit(_item("reference", reference="episode:private"))
    before = memory.items

    assert memory.token_count == 5
    assert memory.items == before


def test_goal_retention_outweighs_newer_low_salience_context() -> None:
    memory = WorkingMemory(item_capacity=1, token_capacity=100)
    memory.admit(
        _item(
            "goal",
            activation=0.3,
            salience=0.3,
            reason=RetentionReason.ONGOING_GOAL,
            kind=WorkingMemoryKind.GOAL,
        )
    )
    memory.admit(_item("recent", activation=0.5, salience=0.2))

    assert [item.item_id for item in memory.items] == ["goal"]


def test_duplicate_reference_reactivates_existing_item() -> None:
    memory = WorkingMemory(item_capacity=2, token_capacity=100)
    first = _item("episode:one", reference="episode:one", activation=0.2)
    memory.admit(first)
    memory.admit(_item("different-id", reference="episode:one", activation=0.4))

    assert len(memory.items) == 1
    restored = memory.items[0]
    assert restored.item_id == "episode:one"
    assert restored.activation == pytest.approx(0.6)
    assert restored.retention_reason == RetentionReason.REACTIVATED


def test_readmitting_active_goal_preserves_protected_retention() -> None:
    memory = WorkingMemory(item_capacity=2, token_capacity=100)
    goal = _item(
        "goal",
        activation=0.3,
        reason=RetentionReason.ONGOING_GOAL,
        kind=WorkingMemoryKind.GOAL,
    )
    memory.admit(goal)

    memory.admit(goal)

    assert memory.items[0].retention_reason == RetentionReason.ONGOING_GOAL


def test_decay_forgets_unprotected_but_retains_commitment() -> None:
    memory = WorkingMemory(item_capacity=2, token_capacity=100)
    memory.admit(_item("ordinary", activation=0.1))
    memory.admit(
        _item(
            "commitment",
            activation=0.1,
            reason=RetentionReason.ACTIVE_COMMITMENT,
            kind=WorkingMemoryKind.COMMITMENT,
        )
    )

    memory.advance(decay=0.1, forget_below=0.1)

    assert {item.item_id for item in memory.items} == {"commitment"}


def test_release_retention_allows_later_forgetting() -> None:
    memory = WorkingMemory(item_capacity=1, token_capacity=100)
    memory.admit(
        _item(
            "goal",
            activation=0.1,
            reason=RetentionReason.ONGOING_GOAL,
            kind=WorkingMemoryKind.GOAL,
        )
    )
    memory.release_retention("goal")
    memory.advance(decay=0.1, forget_below=0.1)

    assert memory.items == ()


def test_restore_trims_deterministically_to_current_capacity() -> None:
    memory = WorkingMemory(item_capacity=1, token_capacity=100)

    memory.restore([_item("low", activation=0.1), _item("high", activation=0.9)])

    assert [item.item_id for item in memory.items] == ["high"]


def test_context_compatibility_prioritizes_local_but_keeps_cross_context() -> None:
    memory = WorkingMemory(item_capacity=3, token_capacity=100)
    memory.admit(
        WorkingMemoryItem(
            **{
                **_item("same", activation=0.5).__dict__,
                "context_id": "ctx-current",
            }
        )
    )
    memory.admit(
        WorkingMemoryItem(
            **{
                **_item("other", activation=0.5).__dict__,
                "context_id": "ctx-other",
            }
        )
    )

    view = memory.select(
        context_compatibility=lambda context_id: (
            (1.0, "same_context")
            if context_id == "ctx-current"
            else (0.2, "unrelated")
        )
    )

    assert [selection.item.item_id for selection in view.selected] == ["same", "other"]
    assert view.selected[0].cross_context is False
    assert view.selected[1].cross_context is True
    assert view.selected[1].context_relation == "unrelated"


def _item(
    item_id: str,
    *,
    content: str = "content",
    reference: str | None = None,
    activation: float = 0.5,
    salience: float = 0.5,
    reason: RetentionReason = RetentionReason.RECENT_CONTEXT,
    kind: WorkingMemoryKind = WorkingMemoryKind.CONVERSATION,
) -> WorkingMemoryItem:
    created = datetime.now(UTC) - timedelta(seconds=1)
    return WorkingMemoryItem(
        item_id=item_id,
        kind=kind,
        content=None if reference else content,
        reference=reference,
        activation=activation,
        salience=salience,
        created_at=created,
        last_accessed_at=created,
        retention_reason=reason,
    )
