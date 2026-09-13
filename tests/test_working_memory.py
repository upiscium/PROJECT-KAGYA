"""R08 U1 bounded Working Memory state-machine contract tests."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from threading import Thread

import pytest
import yaml

from kagya.config import load_settings
from kagya.runtime import (
    WorkingMemory,
    WorkingMemoryAdmissionReason,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySourceKind,
    working_memory_item_id,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def admit(
    memory: WorkingMemory,
    source_id: str,
    *,
    activation: float = 0.5,
    salience: float = 0.5,
    kind: WorkingMemorySourceKind = WorkingMemorySourceKind.EPISODIC,
) -> WorkingMemoryItem:
    return memory.admit(kind, source_id, activation, salience).item


def test_hard_item_bound_evicts_deterministic_lowest_rank() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    low = admit(memory, "episode-low", activation=0.1, salience=0.1)
    high = admit(memory, "episode-high", activation=0.9, salience=0.9)
    middle = admit(memory, "episode-middle", activation=0.5, salience=0.5)

    assert len(memory.items) == 2
    assert {item.item_id for item in memory.items} == {
        high.item_id,
        middle.item_id,
    }
    assert low.item_id not in {item.item_id for item in memory.items}


def test_reactivated_and_high_salience_items_cannot_overflow_capacity() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    protected = admit(memory, "episode-protected", activation=1.0, salience=1.0)
    protected = admit(memory, "episode-protected", activation=1.0, salience=1.0)

    for index in range(10):
        admit(memory, f"episode-{index}", activation=1.0, salience=1.0)
        assert len(memory.items) == 1

    assert protected.retention_reason is WorkingMemoryRetentionReason.REACTIVATED
    assert memory.items[0].item_id == protected.item_id


def test_admission_reports_when_new_candidate_evicts_itself() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    retained = admit(memory, "episode-retained", activation=1.0, salience=1.0)

    rejected = memory.admit(
        WorkingMemorySourceKind.EPISODIC, "episode-too-weak", 0.0, 0.0
    )

    assert not rejected.retained
    assert rejected.reason is WorkingMemoryAdmissionReason.CAPACITY_EVICTED
    assert rejected.evicted_item_id == rejected.item.item_id
    assert memory.items == (retained,)


def test_duplicate_reference_reactivates_stable_identity() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    first = admit(memory, "episode-duplicate", activation=0.2, salience=0.7)
    second = admit(memory, "episode-duplicate", activation=0.4, salience=0.3)

    assert len(memory.items) == 1
    assert second.item_id == first.item_id
    assert second.created_revision == first.created_revision == 1
    assert second.last_activated_revision == memory.revision == 2
    assert second.activation == pytest.approx(0.6)
    assert second.salience == pytest.approx(0.7)
    assert second.retention_reason is WorkingMemoryRetentionReason.REACTIVATED


def test_equal_score_eviction_tie_is_total_and_repeatable() -> None:
    def outcome() -> tuple[str, ...]:
        memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
        admit(memory, "episode-a", activation=0.5, salience=0.5)
        admit(memory, "episode-b", activation=0.5, salience=0.5)
        admit(memory, "episode-c", activation=0.5, salience=0.5)
        return tuple(item.source_id for item in memory.items)

    assert outcome() == outcome()
    assert set(outcome()) == {"episode-b", "episode-c"}


def test_explicit_decay_is_deterministic_and_forgets_weak_items() -> None:
    memory = WorkingMemory(item_capacity=3, projection_max_bytes=100)
    retained = admit(memory, "episode-retained", activation=0.5, salience=0.4)
    removed = admit(memory, "episode-removed", activation=0.2, salience=0.9)

    result = memory.advance(decay=0.15, forget_below=0.1)

    assert memory.revision == 3
    assert {item.item_id for item in result} == {retained.item_id}
    assert result[0].activation == pytest.approx(0.35)
    assert removed.item_id not in {item.item_id for item in memory.items}
    before = (memory.revision, memory.items)
    memory.advance(decay=0.0, forget_below=0.0)
    assert (memory.revision, memory.items) == before


def test_forget_is_idempotent_and_advances_only_on_removal() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    item = admit(memory, "episode-forget")

    assert memory.forget(item.item_id)
    assert memory.revision == 2
    assert not memory.forget(item.item_id)
    assert memory.revision == 2
    assert memory.items == ()


def test_select_is_exactly_pure_and_repeatable() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    admit(memory, "episode-pure", activation=0.8, salience=0.6)
    before = (memory.revision, memory.items)

    first = memory.select(lambda item: f"resolved:{item.source_id}")
    second = memory.select(lambda item: f"resolved:{item.source_id}")

    assert first == second
    assert (memory.revision, memory.items) == before
    with pytest.raises(FrozenInstanceError):
        first.projected_bytes = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.decisions[0].selected = False  # type: ignore[misc]


def test_capacity_revision_and_budget_are_read_only() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)

    with pytest.raises(AttributeError):
        memory.item_capacity = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        memory.projection_max_bytes = 200  # type: ignore[misc]
    with pytest.raises(AttributeError):
        memory.revision = 9  # type: ignore[misc]


def test_resolver_cannot_reenter_authoritative_mutation() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    admit(memory, "episode-reentrant")
    before = (memory.revision, memory.items)

    def reenter(_item: WorkingMemoryItem) -> str:
        admit(memory, "episode-forbidden")
        return "not selected"

    view = memory.select(reenter)

    assert view.decisions[0].reason is WorkingMemoryDecisionReason.RESOLVER_FAILURE
    assert (memory.revision, memory.items) == before


def test_cross_thread_mutation_is_rejected_while_resolver_runs() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    admit(memory, "episode-cross-thread")
    failures: list[str] = []
    worker: Thread | None = None

    def resolver(_item: WorkingMemoryItem) -> str:
        nonlocal worker

        def mutate() -> None:
            try:
                admit(memory, "episode-forbidden-thread")
            except RuntimeError as error:
                failures.append(str(error))

        worker = Thread(target=mutate)
        worker.start()
        worker.join(timeout=1)
        return "resolved"

    before = (memory.revision, memory.items)
    view = memory.select(resolver)
    assert worker is not None
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert failures == ["Working Memory cannot mutate during selection"]
    assert view.selected[0].rendered_content == "resolved"
    assert (memory.revision, memory.items) == before


def test_projection_never_exceeds_utf8_byte_budget() -> None:
    memory = WorkingMemory(item_capacity=3, projection_max_bytes=6)
    admit(memory, "episode-high", activation=1.0, salience=1.0)
    admit(memory, "episode-low", activation=0.5, salience=0.5)

    view = memory.select(
        lambda item: "éé" if item.source_id == "episode-high" else "ab"
    )

    assert view.projected_bytes == 6
    assert view.projected_bytes <= view.projection_max_bytes
    assert [selection.rendered_content for selection in view.selected] == ["éé", "ab"]


def test_oversized_source_is_rejected_without_truncation() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=4)
    admit(memory, "episode-oversized", activation=1.0, salience=1.0)

    view = memory.select(lambda _item: "12345")

    assert view.selected == ()
    assert view.projected_bytes == 0
    assert view.decisions[0].reason is WorkingMemoryDecisionReason.PROJECTION_BUDGET


def test_lower_ranked_small_source_packs_after_large_rejection() -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=5)
    admit(memory, "episode-large", activation=1.0, salience=1.0)
    admit(memory, "episode-small", activation=0.2, salience=0.2)

    view = memory.select(
        lambda item: "too-large" if item.source_id == "episode-large" else "fits"
    )

    assert [decision.reason for decision in view.decisions] == [
        WorkingMemoryDecisionReason.PROJECTION_BUDGET,
        WorkingMemoryDecisionReason.SELECTED,
    ]
    assert [selection.source_id for selection in view.selected] == ["episode-small"]
    assert view.projected_bytes == 4


def test_unresolved_reference_remains_authoritative_without_mutation() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=10)
    item = admit(memory, "semantic-missing", kind=WorkingMemorySourceKind.SEMANTIC)
    before = (memory.revision, memory.items)

    view = memory.select(lambda _item: None)

    assert view.selected == ()
    assert view.decisions[0].item_id == item.item_id
    assert view.decisions[0].reason is (
        WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
    )
    assert (memory.revision, memory.items) == before


def test_resolver_exception_is_bounded_and_private_free() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=10)
    admit(memory, "episode-resolver-failure")
    before = (memory.revision, memory.items)

    def fail(_item: WorkingMemoryItem) -> str:
        raise RuntimeError("PRIVATE-RESOLVER-DETAIL")

    view = memory.select(fail)

    assert view.decisions[0].reason is WorkingMemoryDecisionReason.RESOLVER_FAILURE
    assert view.decisions[0].selected is False
    assert view.selected == ()
    assert "PRIVATE-RESOLVER-DETAIL" not in repr(view)
    assert (memory.revision, memory.items) == before


@pytest.mark.parametrize(
    "source_id",
    ["", "line\nbreak", "tab\tvalue", "../episode", "a/b", "a\\b", "x" * 129],
)
def test_invalid_source_identifiers_are_rejected(source_id: str) -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=10)

    with pytest.raises(ValueError):
        admit(memory, source_id)

    assert memory.revision == 0
    assert memory.items == ()


def test_source_identifiers_support_current_memory_id_shapes() -> None:
    memory = WorkingMemory(item_capacity=3, projection_max_bytes=10)
    values = (
        "09c5b821-8c65-5af8-8f64-97067df3ccf5",
        "episode-09c5b821-8c65-4af8-8f64-97067df3ccf5",
        "semantic-existing_42",
    )

    for value in values:
        admit(memory, value)

    assert {item.source_id for item in memory.items} == set(values)
    with pytest.raises(TypeError):
        memory.admit(WorkingMemorySourceKind.EPISODIC, {"id": "nested"}, 0.5, 0.5)  # type: ignore[arg-type]


def test_item_schema_has_no_content_or_later_domain_authority() -> None:
    assert tuple(field.name for field in fields(WorkingMemoryItem)) == (
        "item_id",
        "source_kind",
        "source_id",
        "activation",
        "salience",
        "retention_reason",
        "created_revision",
        "last_activated_revision",
    )
    vocabulary = {
        *(kind.value for kind in WorkingMemorySourceKind),
        *(reason.value for reason in WorkingMemoryRetentionReason),
    }
    assert vocabulary == {"episodic", "semantic", "recent", "reactivated"}
    assert vocabulary.isdisjoint(
        {"context", "goal", "commitment", "belief", "self_model", "attention"}
    )


def test_identity_and_selection_are_tokenizer_and_provider_independent() -> None:
    item_id = working_memory_item_id(
        WorkingMemorySourceKind.EPISODIC, "episode-independent"
    )
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=3)
    item = admit(memory, "episode-independent")
    view = memory.select(lambda _item: "abc")

    assert item.item_id == item_id
    assert item_id == (
        "wm-1124c114ab7bce9a9e865907f3d1247b5f88adc25407c4cabdec7a804eff67ce"
    )
    assert view.projected_bytes == 3
    assert not hasattr(memory, "tokenizer")
    assert not hasattr(memory, "provider")


def test_working_memory_config_defaults_and_legacy_compatibility(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    assert settings.working_memory.item_capacity == 32
    assert settings.working_memory.projection_max_bytes == 2048

    legacy = yaml.safe_load(CONFIG_PATH.read_text())
    del legacy["working_memory"]
    legacy_path = tmp_path / "legacy-config.yaml"
    legacy_path.write_text(yaml.safe_dump(legacy))

    loaded = load_settings(legacy_path)
    assert loaded.working_memory.item_capacity == 32
    assert loaded.working_memory.projection_max_bytes == 2048
