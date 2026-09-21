"""R08 U1 bounded Working Memory state-machine contract tests."""

from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from threading import Thread

import pytest
import yaml

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.cognition import LossCalibration
from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
    SemanticMemoryFormatError,
    SemanticMemoryReadError,
)
from kagya.memory.working_memory_resolver import MemoryWorkingMemoryResolver
from kagya.models import DummyProvider
from kagya.persona import PromptBuilder
from kagya.runtime import (
    AgentStateStore,
    KagyaMainLoop,
    StateWAL,
    WorkingMemory,
    WorkingMemoryAdmissionReason,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
    working_memory_item_id,
)
from kagya.runtime.context import ContextRegistry
from kagya.runtime.event_journal import EventJournal


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
MODEL_KEY = "model." + "0" * 64


def admit(
    memory: WorkingMemory,
    source_id: str,
    *,
    activation: float = 0.5,
    salience: float = 0.5,
    kind: WorkingMemorySourceKind = WorkingMemorySourceKind.EPISODIC,
) -> WorkingMemoryItem:
    return memory.admit(kind, source_id, activation, salience).item


def capture_loop(memory: WorkingMemory) -> SimpleNamespace:
    return SimpleNamespace(
        emotion_engine=EmotionEngineAllostasis(EmotionState()),
        working_memory=memory,
        context_registry=ContextRegistry(clock=lambda: NOW),
        loss_calibration=LossCalibration(
            (MODEL_KEY,),
            initial_baseline=1.0,
            initial_scale=1.0,
            minimum_scale=0.1,
        ),
    )


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


def test_memory_resolver_imports_cleanly_before_runtime_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from kagya.memory.working_memory_resolver import "
            "MemoryWorkingMemoryResolver",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_repeated_selection_and_prompt_builds_preserve_canonical_evidence(
    tmp_path: Path,
) -> None:
    memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    admit(memory, "episode-repeat", activation=0.8, salience=0.6)
    before = (memory.revision, memory.items)
    loop = capture_loop(memory)
    store = AgentStateStore(tmp_path / "agent-state.json", 1.0, clock=lambda: NOW)

    views = [
        memory.select(lambda item: f"resolved:{item.source_id}") for _ in range(3)
    ]
    prompts = [
        PromptBuilder().build("hello", loop.emotion_engine.state, view)
        for view in views
    ]
    snapshots = [store.capture(loop, sequence=7) for _ in range(3)]

    assert views[0] == views[1] == views[2]
    assert prompts[0] == prompts[1] == prompts[2]
    assert (memory.revision, memory.items) == before
    assert all(
        store.canonical_bytes(snapshot) == store.canonical_bytes(snapshots[0])
        for snapshot in snapshots
    )
    assert all(
        store.snapshot_hash(snapshot) == store.snapshot_hash(snapshots[0])
        for snapshot in snapshots
    )


def test_projection_budget_is_ephemeral_and_never_changes_canonical_state() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=3)
    item = admit(memory, "episode-budget", activation=1.0, salience=1.0)
    before = (memory.revision, memory.items)

    view = memory.select(lambda _item: "too large")

    assert view.selected == ()
    assert view.projected_bytes == 0
    assert view.decisions[0].reason is WorkingMemoryDecisionReason.PROJECTION_BUDGET
    assert (memory.revision, memory.items) == before
    assert memory.items == (item,)


def test_restore_exact_preserves_revision_and_membership_without_replay() -> None:
    source = WorkingMemory(item_capacity=2, projection_max_bytes=10)
    item = admit(source, "episode-exact", activation=0.7, salience=0.8)
    source.advance(decay=0.0, forget_below=0.0)
    target = WorkingMemory(item_capacity=2, projection_max_bytes=10)

    target.restore_exact(source.revision, source.items)

    assert target.revision == source.revision == 1
    assert target.items == (item,)


def test_restore_exact_rejects_capacity_decrease_but_accepts_larger_capacity() -> None:
    source = WorkingMemory(item_capacity=2, projection_max_bytes=10)
    admit(source, "episode-one")
    admit(source, "episode-two")
    smaller = WorkingMemory(item_capacity=1, projection_max_bytes=10)
    larger = WorkingMemory(item_capacity=3, projection_max_bytes=10)

    with pytest.raises(ValueError):
        smaller.restore_exact(source.revision, source.items)
    larger.restore_exact(source.revision, source.items)

    assert larger.revision == source.revision == 2
    assert larger.items == source.items


@pytest.mark.parametrize(
    "revision,item_change",
    [
        (1, {"item_id": "wm-invalid"}),
        (1, {"source_kind": "invalid"}),
        (1, {"source_id": "../invalid"}),
        (1, {"activation": float("nan")}),
        (1, {"activation": 1}),
        (1, {"salience": 2.0}),
        (1, {"retention_reason": "invalid"}),
        (1, {"created_revision": 2}),
        (1, {"last_activated_revision": 2}),
        (-1, {}),
        (True, {}),
    ],
)
def test_restore_exact_rejects_malformed_canonical_state_without_mutation(
    revision: int, item_change: dict[str, object]
) -> None:
    source = WorkingMemory(item_capacity=1, projection_max_bytes=10)
    item = admit(source, "episode-valid")
    target = WorkingMemory(item_capacity=2, projection_max_bytes=10)
    retained = admit(target, "episode-retained")
    before = (target.revision, target.items)
    malformed = replace(item, **item_change)

    with pytest.raises((TypeError, ValueError)):
        target.restore_exact(revision, (malformed,))

    assert (target.revision, target.items) == before == (1, (retained,))


def test_restore_exact_rejects_duplicate_identity_and_source_reference() -> None:
    source = WorkingMemory(item_capacity=1, projection_max_bytes=10)
    item = admit(source, "episode-duplicate-restore")
    target = WorkingMemory(item_capacity=2, projection_max_bytes=10)

    with pytest.raises(ValueError, match="duplicated"):
        target.restore_exact(source.revision, (item, item))

    assert target.revision == 0
    assert target.items == ()


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


@pytest.mark.parametrize(
    "status",
    [
        WorkingMemoryResolutionStatus.MISSING,
        WorkingMemoryResolutionStatus.ARCHIVED,
        WorkingMemoryResolutionStatus.UNAVAILABLE,
        WorkingMemoryResolutionStatus.MALFORMED,
    ],
)
def test_nonresolved_resolution_cannot_expose_content(
    status: WorkingMemoryResolutionStatus,
) -> None:
    with pytest.raises(ValueError):
        WorkingMemoryResolution(status, "private body")
    with pytest.raises(ValueError):
        WorkingMemoryResolution(status, source_context_id="context-private")
    assert WorkingMemoryResolution(status).rendered_content is None


@pytest.mark.parametrize(
    "status",
    [
        WorkingMemoryResolutionStatus.RESOLVED,
        WorkingMemoryResolutionStatus.MISSING,
        WorkingMemoryResolutionStatus.ARCHIVED,
        WorkingMemoryResolutionStatus.UNAVAILABLE,
        WorkingMemoryResolutionStatus.MALFORMED,
    ],
)
def test_resolution_outcomes_preserve_wm_wal_and_journal_evidence(
    tmp_path: Path, status: WorkingMemoryResolutionStatus
) -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    admit(memory, "episode-evidence")
    before = (memory.revision, memory.items)
    loop = capture_loop(memory)
    store = AgentStateStore(tmp_path / "agent-state.json", 1.0, clock=lambda: NOW)
    snapshot = store.capture(loop, sequence=0)
    snapshot_hash = store.snapshot_hash(snapshot)
    wal = StateWAL(tmp_path / "wal")
    wal.bootstrap(snapshot, 0)
    journal = EventJournal(
        tmp_path / "journal.jsonl", 100_000, 4, clock=lambda: NOW
    )
    journal.verify_and_reconcile(0, snapshot_hash)
    wal_before = tuple(
        (path.relative_to(tmp_path / "wal"), path.read_bytes())
        for path in sorted((tmp_path / "wal").rglob("*"))
        if path.is_file()
    )
    journal_before = journal.path.read_bytes()

    result = WorkingMemoryResolution(
        status, "body" if status is WorkingMemoryResolutionStatus.RESOLVED else None
    )
    memory.select(lambda _item: result)

    after = store.capture(loop, sequence=0)
    assert (memory.revision, memory.items) == before
    assert store.canonical_bytes(after) == store.canonical_bytes(snapshot)
    assert store.snapshot_hash(after) == snapshot_hash
    assert tuple(
        (path.relative_to(tmp_path / "wal"), path.read_bytes())
        for path in sorted((tmp_path / "wal").rglob("*"))
        if path.is_file()
    ) == wal_before
    assert journal.path.read_bytes() == journal_before


def test_typed_resolution_selects_and_maps_source_statuses() -> None:
    memory = WorkingMemory(item_capacity=5, projection_max_bytes=100)
    for index, status in enumerate(
        (
            WorkingMemoryResolutionStatus.RESOLVED,
            WorkingMemoryResolutionStatus.MISSING,
            WorkingMemoryResolutionStatus.ARCHIVED,
            WorkingMemoryResolutionStatus.UNAVAILABLE,
            WorkingMemoryResolutionStatus.MALFORMED,
        )
    ):
        admit(memory, f"episode-typed-{index}")

    results = {
        f"episode-typed-{index}": WorkingMemoryResolution(
            status,
            "authoritative body"
            if status is WorkingMemoryResolutionStatus.RESOLVED
            else None,
        )
        for index, status in enumerate(
            (
                WorkingMemoryResolutionStatus.RESOLVED,
                WorkingMemoryResolutionStatus.MISSING,
                WorkingMemoryResolutionStatus.ARCHIVED,
                WorkingMemoryResolutionStatus.UNAVAILABLE,
                WorkingMemoryResolutionStatus.MALFORMED,
            )
        )
    }
    view = memory.select(lambda item: results[item.source_id])

    assert [selection.rendered_content for selection in view.selected] == [
        "authoritative body"
    ]
    reasons = {decision.source_id: decision.reason for decision in view.decisions}
    assert reasons == {
        f"episode-typed-{index}": expected
        for index, expected in enumerate(
            (
                WorkingMemoryDecisionReason.SELECTED,
                WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
                WorkingMemoryDecisionReason.SOURCE_ARCHIVED,
                WorkingMemoryDecisionReason.SOURCE_UNAVAILABLE,
                WorkingMemoryDecisionReason.SOURCE_MALFORMED,
            )
        )
    }


def test_invalid_typed_result_is_resolver_failure() -> None:
    memory = WorkingMemory(item_capacity=1, projection_max_bytes=20)
    admit(memory, "episode-invalid-typed")

    view = memory.select(
        lambda _item: SimpleNamespace(status="resolved", rendered_content=3)
    )

    assert view.decisions[0].reason is WorkingMemoryDecisionReason.RESOLVER_FAILURE


def test_memory_resolver_dispatches_committed_reads_and_maps_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kagya.memory.dual_memory_system as dual_memory_system

    class EpisodicReadError(Exception):
        pass

    class EpisodicFormatError(Exception):
        pass

    class SemanticReadError(Exception):
        pass

    class SemanticFormatError(Exception):
        pass

    monkeypatch.setattr(
        dual_memory_system, "EpisodicMemoryReadError", EpisodicReadError
    )
    monkeypatch.setattr(
        dual_memory_system, "EpisodicMemoryFormatError", EpisodicFormatError
    )
    monkeypatch.setattr(
        dual_memory_system, "SemanticMemoryReadError", SemanticReadError, raising=False
    )
    monkeypatch.setattr(
        dual_memory_system,
        "SemanticMemoryFormatError",
        SemanticFormatError,
        raising=False,
    )

    episodic = WorkingMemoryItem(
        working_memory_item_id(WorkingMemorySourceKind.EPISODIC, "episode-real"),
        WorkingMemorySourceKind.EPISODIC,
        "episode-real",
        0.5,
        0.5,
        WorkingMemoryRetentionReason.RECENT,
        0,
        0,
    )
    semantic = replace(
        episodic,
        source_kind=WorkingMemorySourceKind.SEMANTIC,
        source_id="semantic-real",
        item_id=working_memory_item_id(
            WorkingMemorySourceKind.SEMANTIC, "semantic-real"
        ),
    )

    class Memory:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def get_committed_episodic(self, source_id: str) -> object:
            self.calls.append(("episodic", source_id))
            return SimpleNamespace(
                document="episode body", record=SimpleNamespace(archived=False)
            )

        def get_committed_semantic(self, source_id: str) -> object:
            self.calls.append(("semantic", source_id))
            return SimpleNamespace(
                document="semantic body", record=SimpleNamespace(context_id=None)
            )

    memory = Memory()
    resolver = MemoryWorkingMemoryResolver(memory)

    assert resolver(episodic) == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.RESOLVED, "episode body"
    )
    assert resolver(semantic) == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.RESOLVED, "semantic body"
    )
    assert memory.calls == [
        ("episodic", "episode-real"),
        ("semantic", "semantic-real"),
    ]


def test_resolver_passes_episodic_and_semantic_context_ephemerally(
    tmp_path: Path,
) -> None:
    source = _dual_memory(tmp_path)
    source.publish_coordinated_episodic(
        "episode-context",
        "context input",
        "context response",
        loss=0.1,
        emotion_valence=0.2,
        emotion_arousal=0.3,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at=NOW.isoformat(),
        coordination_schema=2,
        context_id="context-a",
        source_channel="chat",
    )
    semantic_id = source.save_semantic(
        "semantic context body", source_episode_ids=["episode-context"]
    )
    working = WorkingMemory(item_capacity=2, projection_max_bytes=1000)
    episodic_item = admit(working, "episode-context")
    semantic_item = admit(
        working, semantic_id, kind=WorkingMemorySourceKind.SEMANTIC
    )
    resolver = MemoryWorkingMemoryResolver(source)

    episodic_resolution = resolver.resolve(episodic_item)
    semantic_resolution = resolver.resolve(semantic_item)
    view = working.select(resolver)

    assert episodic_resolution.source_context_id == "context-a"
    assert semantic_resolution.source_context_id == "context-a"
    assert {
        selection.source_id: selection.source_context_id
        for selection in view.selected
    } == {
        "episode-context": "context-a",
        semantic_id: "context-a",
    }
    assert all(
        not hasattr(item, "source_context_id") for item in working.items
    )
    prompt = PromptBuilder().build("hello", EmotionState(), view)
    assert "context-a" not in prompt


def test_semantic_resolver_does_not_reinfer_context_from_source_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _dual_memory(tmp_path)
    source.publish_coordinated_episodic(
        "episode-context",
        "context input",
        "context response",
        loss=0.1,
        emotion_valence=0.2,
        emotion_arousal=0.3,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at=NOW.isoformat(),
        coordination_schema=2,
        context_id="context-a",
        source_channel="chat",
    )
    semantic_id = source.save_semantic(
        "semantic context body", source_episode_ids=["episode-context"]
    )
    item = WorkingMemoryItem(
        working_memory_item_id(WorkingMemorySourceKind.SEMANTIC, semantic_id),
        WorkingMemorySourceKind.SEMANTIC,
        semantic_id,
        0.5,
        0.5,
        WorkingMemoryRetentionReason.RECENT,
        0,
        0,
    )

    def no_source_lookup(_source_id: str) -> object:
        raise AssertionError("Semantic resolution must use its committed record")

    monkeypatch.setattr(source, "get_committed_episodic", no_source_lookup)
    resolution = MemoryWorkingMemoryResolver(source).resolve(item)

    assert resolution.source_context_id == "context-a"


def test_context_resolution_metadata_does_not_change_selection_or_prompt() -> None:
    context_memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    plain_memory = WorkingMemory(item_capacity=2, projection_max_bytes=100)
    for memory in (context_memory, plain_memory):
        admit(memory, "episode-context-one", activation=0.8, salience=0.6)
        admit(memory, "episode-context-two", activation=0.4, salience=0.7)

    context_view = context_memory.select(
        lambda item: WorkingMemoryResolution(
            WorkingMemoryResolutionStatus.RESOLVED,
            f"body:{item.source_id}",
            "context-a",
        )
    )
    plain_view = plain_memory.select(lambda item: f"body:{item.source_id}")
    context_prompt = PromptBuilder().build("hello", EmotionState(), context_view)
    plain_prompt = PromptBuilder().build("hello", EmotionState(), plain_view)

    assert [item.source_id for item in context_view.selected] == [
        item.source_id for item in plain_view.selected
    ]
    assert [item.score for item in context_view.selected] == [
        item.score for item in plain_view.selected
    ]
    assert [item.reason for item in context_view.selected] == [
        item.reason for item in plain_view.selected
    ]
    assert context_view.projected_bytes == plain_view.projected_bytes
    assert context_prompt == plain_prompt
    assert all(
        item.source_context_id == "context-a" for item in context_view.selected
    )


def test_real_episodic_resolution_is_exact_pure_and_archived_is_ineligible(
    tmp_path: Path,
) -> None:
    source = _dual_memory(tmp_path)
    episode_id = source.save_episodic("remember me", "I remember")
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = admit(working, episode_id)
    resolver = MemoryWorkingMemoryResolver(source)
    before_working = (working.revision, working.items)
    before_db1 = source.db1.get(include=["documents", "metadatas"])

    resolution = resolver.resolve(item)
    view = working.select(resolver)

    assert resolution == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.RESOLVED,
        "User: remember me\nAssistant: I remember",
    )
    assert view.selected[0].rendered_content == resolution.rendered_content
    assert (working.revision, working.items) == before_working
    assert source.db1.get(include=["documents", "metadatas"]) == before_db1

    source._archive_episodic(episode_id)
    archived_db1 = source.db1.get(include=["documents", "metadatas"])
    archived = resolver.resolve(item)
    archived_view = working.select(resolver)

    assert archived == WorkingMemoryResolution(WorkingMemoryResolutionStatus.ARCHIVED)
    assert archived.rendered_content is None
    assert archived_view.decisions[0].reason is (
        WorkingMemoryDecisionReason.SOURCE_ARCHIVED
    )
    assert (working.revision, working.items) == before_working
    assert source.db1.get(include=["documents", "metadatas"]) == archived_db1


@pytest.mark.parametrize(
    ("kind", "status", "decision"),
    [
        (
            WorkingMemorySourceKind.EPISODIC,
            WorkingMemoryResolutionStatus.MISSING,
            WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
        ),
        (
            WorkingMemorySourceKind.SEMANTIC,
            WorkingMemoryResolutionStatus.MISSING,
            WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
        ),
    ],
)
def test_real_missing_source_remains_authoritative(
    tmp_path: Path,
    kind: WorkingMemorySourceKind,
    status: WorkingMemoryResolutionStatus,
    decision: WorkingMemoryDecisionReason,
) -> None:
    source = _dual_memory(tmp_path)
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = admit(working, f"{kind.value}-missing", kind=kind)
    before = (working.revision, working.items)
    resolver = MemoryWorkingMemoryResolver(source)

    resolution = resolver.resolve(item)
    view = working.select(resolver)

    assert resolution == WorkingMemoryResolution(status)
    assert resolution.rendered_content is None
    assert view.decisions[0].reason is decision
    assert (working.revision, working.items) == before


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        (WorkingMemorySourceKind.EPISODIC, EpisodicMemoryReadError),
        (WorkingMemorySourceKind.SEMANTIC, SemanticMemoryReadError),
    ],
)
def test_known_backend_unavailability_is_not_resolver_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: WorkingMemorySourceKind,
    error_type: type[Exception],
) -> None:
    source = _dual_memory(tmp_path)
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = admit(working, f"{kind.value}-unavailable", kind=kind)
    method_name = f"get_committed_{kind.value}"

    def unavailable(_source_id: str) -> None:
        raise error_type("PRIVATE backend exception")

    monkeypatch.setattr(source, method_name, unavailable)
    before = (working.revision, working.items)

    resolution = MemoryWorkingMemoryResolver(source).resolve(item)
    view = working.select(MemoryWorkingMemoryResolver(source))

    assert resolution == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.UNAVAILABLE
    )
    assert view.decisions[0].reason is (WorkingMemoryDecisionReason.SOURCE_UNAVAILABLE)
    assert "PRIVATE" not in repr(resolution)
    assert "PRIVATE" not in repr(view)
    assert (working.revision, working.items) == before


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        (WorkingMemorySourceKind.EPISODIC, EpisodicMemoryFormatError),
        (WorkingMemorySourceKind.SEMANTIC, SemanticMemoryFormatError),
    ],
)
def test_malformed_source_is_distinct_and_body_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: WorkingMemorySourceKind,
    error_type: type[Exception],
) -> None:
    source = _dual_memory(tmp_path)
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = admit(working, f"{kind.value}-malformed", kind=kind)

    def malformed(_source_id: str) -> None:
        raise error_type("PRIVATE malformed source body")

    monkeypatch.setattr(source, f"get_committed_{kind.value}", malformed)
    before = (working.revision, working.items)

    resolution = MemoryWorkingMemoryResolver(source).resolve(item)
    view = working.select(MemoryWorkingMemoryResolver(source))

    assert resolution == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.MALFORMED
    )
    assert resolution.rendered_content is None
    assert view.decisions[0].reason is WorkingMemoryDecisionReason.SOURCE_MALFORMED
    assert "PRIVATE" not in repr(resolution)
    assert "PRIVATE" not in repr(view)
    assert (working.revision, working.items) == before


@pytest.mark.parametrize(
    "kind",
    [WorkingMemorySourceKind.EPISODIC, WorkingMemorySourceKind.SEMANTIC],
)
def test_real_conflicting_source_resolves_malformed_without_repair(
    tmp_path: Path, kind: WorkingMemorySourceKind
) -> None:
    source = _dual_memory(tmp_path)
    if kind is WorkingMemorySourceKind.EPISODIC:
        source_id = source.save_episodic("visible", "answer")
        source.db1.update(ids=[source_id], documents=["conflicting document"])
        collection = source.db1
    else:
        source_id = source.save_semantic("visible semantic")
        stored = source.db2.get(ids=[source_id], include=["metadatas"])
        metadata = dict(stored["metadatas"][0])
        metadata["text"] = "conflicting metadata"
        source.db2.update(ids=[source_id], metadatas=[metadata])
        collection = source.db2
    before_source = collection.get(ids=[source_id], include=["documents", "metadatas"])
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    item = admit(working, source_id, kind=kind)
    before_working = (working.revision, working.items)
    resolver = MemoryWorkingMemoryResolver(source)

    resolution = resolver.resolve(item)
    view = working.select(resolver)

    assert resolution == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.MALFORMED
    )
    assert resolution.rendered_content is None
    assert view.decisions[0].reason is WorkingMemoryDecisionReason.SOURCE_MALFORMED
    assert (working.revision, working.items) == before_working
    assert (
        collection.get(ids=[source_id], include=["documents", "metadatas"])
        == before_source
    )


def test_real_semantic_resolution_and_utf8_budget_use_authoritative_documents(
    tmp_path: Path,
) -> None:
    source = _dual_memory(tmp_path)
    large_id = source.save_semantic("é" * 20)
    small_id = source.save_semantic("fits")
    working = WorkingMemory(item_capacity=2, projection_max_bytes=5)
    admit(
        working,
        large_id,
        activation=1.0,
        salience=1.0,
        kind=WorkingMemorySourceKind.SEMANTIC,
    )
    admit(
        working,
        small_id,
        activation=0.2,
        salience=0.2,
        kind=WorkingMemorySourceKind.SEMANTIC,
    )
    resolver = MemoryWorkingMemoryResolver(source)
    small_item = next(item for item in working.items if item.source_id == small_id)

    resolution = resolver.resolve(small_item)
    view = working.select(resolver)

    assert resolution == WorkingMemoryResolution(
        WorkingMemoryResolutionStatus.RESOLVED, "fits"
    )
    assert [decision.reason for decision in view.decisions] == [
        WorkingMemoryDecisionReason.PROJECTION_BUDGET,
        WorkingMemoryDecisionReason.SELECTED,
    ]
    assert [selection.rendered_content for selection in view.selected] == ["fits"]
    assert view.projected_bytes == len("fits".encode("utf-8"))
    assert view.projected_bytes <= view.projection_max_bytes


def test_resolved_body_never_enters_agent_state_or_wal(tmp_path: Path) -> None:
    sentinel = "U3-EPHEMERAL-BODY-SENTINEL"
    settings = _settings_for_tmp_memory(tmp_path)
    source = DualMemorySystem(settings)
    semantic_id = source.save_semantic(sentinel)
    working = WorkingMemory(item_capacity=1, projection_max_bytes=100)
    admit(working, semantic_id, kind=WorkingMemorySourceKind.SEMANTIC)
    loop = KagyaMainLoop(settings, DummyProvider(), source, working_memory=working)
    state_store = AgentStateStore(
        tmp_path / "agent-state.json", settings.emotion.baseline_surprisal
    )

    view = working.select(MemoryWorkingMemoryResolver(source))
    snapshot = state_store.capture(loop, sequence=0)
    canonical = state_store.canonical_bytes(snapshot)
    wal = StateWAL(tmp_path / "wal")
    wal.bootstrap(snapshot, 0)
    durable_wal = b"".join(
        path.read_bytes() for path in (tmp_path / "wal").rglob("*") if path.is_file()
    )

    assert view.selected[0].rendered_content == sentinel
    assert sentinel not in repr(working.items)
    assert sentinel.encode() not in canonical
    assert sentinel.encode() not in durable_wal


def _settings_for_tmp_memory(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_working_memory_test",
                    "db2_collection": "cortex_working_memory_test",
                }
            )
        }
    )


def _dual_memory(tmp_path: Path) -> DualMemorySystem:
    return DualMemorySystem(_settings_for_tmp_memory(tmp_path))
