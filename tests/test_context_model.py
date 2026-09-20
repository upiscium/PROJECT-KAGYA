"""R09 U1 strict Context state-machine contract tests."""

import ast
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta, timezone
import inspect
import math
from pathlib import Path

import pytest

from kagya.identifiers import validate_identifier
import kagya.runtime.context as context_module
from kagya.runtime import (
    MAX_CONTEXTS,
    MAX_EVIDENCE_REFERENCES,
    MAX_INTERLOCUTOR_BINDINGS,
    MAX_PARTICIPANTS_PER_CONTEXT,
    MAX_RELATIONS_PER_CONTEXT,
    ContextCapacityExceeded,
    ContextCompatibility,
    ContextConflict,
    ContextError,
    ContextFrame,
    ContextNotFound,
    ContextRegistry,
    ContextRegistryState,
    ContextRelation,
    ContextStateInvalid,
    ContextStatus,
    ContextType,
    InterlocutorBinding,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


class DeterministicClock:
    def __init__(self, *values: datetime) -> None:
        self.values = list(values or (NOW,))
        self.calls = 0

    def __call__(self) -> datetime:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def create_context(
    registry: ContextRegistry,
    context_id: str,
    *,
    participants: tuple[str, ...] = (),
    parent: str | None = None,
) -> ContextFrame:
    return registry.create(
        context_id,
        ContextType.CONVERSATION,
        "chat",
        source_session_id=f"session:{context_id}",
        participant_refs=participants,
        parent_context_id=parent,
    )


def nontrivial_registry(clock: DeterministicClock | None = None) -> ContextRegistry:
    registry = ContextRegistry(clock=clock or DeterministicClock())
    create_context(registry, "context-a", participants=("ref-b", "ref-a"))
    create_context(registry, "context-b", parent="context-a")
    create_context(registry, "context-c")
    registry.relate("context-a", "context-c")
    registry.upsert_interlocutor_binding(
        "ref-b", "person-x", 0.6, ("episode-b", "episode-a")
    )
    registry.set_current("context-a")
    registry.suspend("context-b")
    return registry


def test_create_uses_one_injected_clock_read_and_exact_first_revision() -> None:
    source_time = datetime(2026, 1, 1, 9, tzinfo=timezone(timedelta(hours=9)))
    clock = DeterministicClock(source_time)
    registry = ContextRegistry(clock=clock)

    frame = registry.create(
        "context-1",
        ContextType.CONVERSATION,
        "web:chat",
        "session-1",
        ("ref-z", "ref-a", "ref-z"),
    )

    assert frame.status is ContextStatus.ACTIVE
    assert registry.state.revision == 1
    assert frame.created_revision == frame.last_modified_revision == 1
    assert frame.started_at == frame.last_active_at == source_time.astimezone(UTC)
    assert frame.participant_refs == ("ref-a", "ref-z")
    assert registry.current_context_id is None
    assert clock.calls == 1


def test_false_valued_clock_is_still_the_clock_authority() -> None:
    class FalseClock(DeterministicClock):
        def __bool__(self) -> bool:
            return False

    clock = FalseClock(NOW)
    registry = ContextRegistry(clock=clock)

    frame = create_context(registry, "context-a")

    assert frame.started_at == NOW
    assert clock.calls == 1


def test_invalid_or_failing_clock_is_bounded_and_does_not_mutate() -> None:
    naive_clock = DeterministicClock(NOW.replace(tzinfo=None))
    registry = ContextRegistry(clock=naive_clock)
    with pytest.raises(ContextStateInvalid, match="timestamp"):
        create_context(registry, "context-a")
    assert registry.state.revision == 0
    assert naive_clock.calls == 1

    def failing_clock() -> datetime:
        raise RuntimeError("PRIVATE-SENTINEL-R09")

    registry = ContextRegistry(clock=failing_clock)
    with pytest.raises(ContextStateInvalid, match="clock failed") as captured:
        create_context(registry, "context-a")
    assert "PRIVATE-SENTINEL-R09" not in str(captured.value)
    assert registry.state.revision == 0

    def failing_domain_clock() -> datetime:
        raise ContextStateInvalid("PRIVATE-SENTINEL-R09")

    registry = ContextRegistry(clock=failing_domain_clock)
    with pytest.raises(ContextStateInvalid, match="clock failed") as captured:
        create_context(registry, "context-a")
    assert "PRIVATE-SENTINEL-R09" not in str(captured.value)


def test_duplicate_create_is_bounded_and_atomic() -> None:
    clock = DeterministicClock()
    registry = ContextRegistry(clock=clock)
    create_context(registry, "context-a")
    before = registry.state

    with pytest.raises(ContextConflict, match="already exists"):
        create_context(registry, "context-a")

    assert registry.state == before
    assert clock.calls == 1


def test_lifecycle_is_finite_terminal_and_idempotent() -> None:
    times = tuple(NOW + timedelta(minutes=index) for index in range(4))
    clock = DeterministicClock(*times)
    registry = ContextRegistry(clock=clock)
    created = create_context(registry, "context-a")

    suspended = registry.suspend("context-a")
    assert suspended.status is ContextStatus.SUSPENDED
    assert suspended.last_modified_revision == 2
    assert suspended.last_active_at == created.last_active_at
    assert registry.suspend("context-a") is suspended
    assert registry.state.revision == 2

    resumed = registry.resume("context-a")
    assert resumed.status is ContextStatus.ACTIVE
    assert resumed.last_modified_revision == 3
    assert resumed.last_active_at == times[2]
    assert registry.resume("context-a") is resumed

    closed = registry.close("context-a")
    assert closed.status is ContextStatus.CLOSED
    assert closed.last_modified_revision == 4
    assert registry.close("context-a") is closed
    with pytest.raises(ContextStateInvalid, match="transition"):
        registry.resume("context-a")
    with pytest.raises(ContextStateInvalid, match="transition"):
        registry.suspend("context-a")
    assert registry.state.revision == 4
    assert clock.calls == 4


def test_resume_rejects_regressing_clock_without_mutation() -> None:
    clock = DeterministicClock(NOW, NOW + timedelta(minutes=1), NOW - timedelta(1))
    registry = ContextRegistry(clock=clock)
    create_context(registry, "context-a")
    registry.suspend("context-a")
    before = registry.state

    with pytest.raises(ContextStateInvalid, match="regressed"):
        registry.resume("context-a")

    assert registry.state == before
    assert clock.calls == 3


def test_reentrant_iterable_and_clock_mutations_are_rejected_atomically() -> None:
    registry = ContextRegistry(clock=DeterministicClock())

    class ReentrantParticipants:
        def __iter__(self):  # type: ignore[no-untyped-def]
            create_context(registry, "nested")
            yield "ref-a"

    with pytest.raises(ContextStateInvalid, match="invalid references"):
        registry.create(
            "outer",
            ContextType.CONVERSATION,
            "chat",
            participant_refs=ReentrantParticipants(),
        )
    assert registry.state.revision == 0

    first_clock = DeterministicClock(NOW)
    registry = ContextRegistry(clock=first_clock)
    create_context(registry, "context-a")
    registry.suspend("context-a")

    def reentrant_clock() -> datetime:
        registry.close("context-a")
        return NOW + timedelta(minutes=1)

    registry._clock = reentrant_clock
    before = registry.state
    with pytest.raises(ContextStateInvalid, match="clock failed"):
        registry.resume("context-a")
    assert registry.state == before


def test_suspended_context_can_close_without_resume() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "context-a")
    registry.suspend("context-a")

    assert registry.close("context-a").status is ContextStatus.CLOSED
    assert registry.state.revision == 3


def test_current_context_requires_active_and_clears_atomically() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "context-a")
    create_context(registry, "context-b")

    registry.set_current("context-a")
    assert registry.current_context_id == "context-a"
    assert registry.state.revision == 3
    registry.set_current("context-a")
    assert registry.state.revision == 3

    registry.suspend("context-a")
    assert registry.current_context_id is None
    assert registry.state.revision == 4
    with pytest.raises(ContextStateInvalid, match="must be active"):
        registry.set_current("context-a")
    registry.resume("context-a")
    assert registry.current_context_id is None
    registry.set_current("context-a")
    registry.close("context-a")
    assert registry.current_context_id is None
    assert registry.state.revision == 7

    with pytest.raises(ContextNotFound, match="not found"):
        registry.set_current("unknown-context")
    registry.set_current("context-b")
    registry.set_current(None)
    assert registry.current_context_id is None


def test_current_context_returns_exact_active_immutable_frame_without_mutation() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    frame = create_context(registry, "context-a")
    registry.set_current("context-a")
    before = registry.state

    current = registry.current_context

    assert current is frame
    assert current is registry.state.frames[0]
    assert current.status is ContextStatus.ACTIVE
    assert registry.state == before


def test_current_context_returns_none_when_no_context_is_current() -> None:
    clock = DeterministicClock()
    registry = ContextRegistry(clock=clock)
    before = registry.state

    assert registry.current_context is None
    assert registry.state == before
    assert clock.calls == 0


def test_shared_identifier_validator_and_context_validation_have_grammar_parity() -> None:
    valid = ("a", "A0._:-", "x" * 100)
    invalid = (
        "",
        "0" + "x" * 128,
        "a..b",
        "a/b",
        "a\\b",
        "a b",
        "é",
        "a\n",
    )
    for value in valid:
        assert validate_identifier(value) == value
        assert (
            create_context(ContextRegistry(clock=DeterministicClock()), value).context_id
            == value
        )
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            validate_identifier(value)
        with pytest.raises(ContextStateInvalid, match="identifier"):
            create_context(ContextRegistry(clock=DeterministicClock()), value)

    for value in (True, 1, None):
        if type(value) is str:
            continue
        with pytest.raises((TypeError, ValueError)):
            validate_identifier(value)
        with pytest.raises(ContextStateInvalid, match="identifier"):
            create_context(
                ContextRegistry(clock=DeterministicClock()), value  # type: ignore[arg-type]
            )


def test_parent_must_exist_and_is_not_an_implicit_relation() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "parent")
    child = create_context(registry, "child", parent="parent")

    assert child.parent_context_id == "parent"
    assert child.related_context_ids == ()
    assert registry.state.frames[1].related_context_ids == ()

    before = registry.state
    with pytest.raises(ContextStateInvalid, match="parent"):
        create_context(registry, "orphan", parent="missing")
    with pytest.raises(ContextStateInvalid, match="parent"):
        registry.create(
            "self-parent",
            ContextType.CONVERSATION,
            "chat",
            parent_context_id="self-parent",
        )
    assert registry.state == before


def test_explicit_relation_is_symmetric_sorted_idempotent_and_one_revision() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    for context_id in ("context-c", "context-a", "context-b"):
        create_context(registry, context_id)

    registry.relate("context-c", "context-b")
    first_relation_revision = registry.state.revision
    registry.relate("context-c", "context-a")
    state = registry.state
    frames = {frame.context_id: frame for frame in state.frames}

    assert first_relation_revision == 4
    assert state.revision == 5
    assert frames["context-c"].related_context_ids == ("context-a", "context-b")
    assert frames["context-a"].related_context_ids == ("context-c",)
    assert frames["context-b"].related_context_ids == ("context-c",)
    assert frames["context-c"].last_modified_revision == 5
    assert frames["context-a"].last_modified_revision == 5

    registry.relate("context-a", "context-c")
    assert registry.state == state
    with pytest.raises(ContextStateInvalid, match="itself"):
        registry.relate("context-a", "context-a")
    with pytest.raises(ContextNotFound, match="not found"):
        registry.relate("context-a", "missing")
    assert registry.state == state


def test_compatibility_has_exact_precedence_relations_and_scores() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "current", participants=("ref-shared",))
    create_context(registry, "child", parent="current")
    create_context(registry, "related", participants=("ref-shared",))
    create_context(registry, "shared", participants=("ref-shared",))
    create_context(registry, "unrelated", participants=("ref-other",))
    registry.relate("current", "related")

    expected = (
        ("current", ContextRelation.SAME_CONTEXT, 1.0),
        ("child", ContextRelation.PARENT_CHILD, 0.8),
        ("related", ContextRelation.RELATED, 0.75),
        ("shared", ContextRelation.SHARED_INTERLOCUTOR, 0.65),
        (None, ContextRelation.LEGACY_UNKNOWN, 0.45),
        ("unknown", ContextRelation.UNKNOWN_CONTEXT, 0.35),
        ("unrelated", ContextRelation.UNRELATED, 0.2),
    )
    for source_id, relation, score in expected:
        assert registry.compatibility(source_id, "current") == ContextCompatibility(
            score, relation, source_id, "current"
        )


def test_compatibility_requires_existing_active_current_context() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "suspended")
    registry.suspend("suspended")

    with pytest.raises(ContextNotFound, match="not found"):
        registry.compatibility(None, "missing")
    with pytest.raises(ContextStateInvalid, match="must be active"):
        registry.compatibility(None, "suspended")


def test_compatibility_is_exactly_pure_and_does_not_equate_identity_hypotheses() -> (
    None
):
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "context-a", participants=("ref-a",))
    create_context(registry, "context-b", participants=("ref-b",))
    registry.upsert_interlocutor_binding("ref-a", "person-x", 0.9, ("evidence-a",))
    registry.upsert_interlocutor_binding("ref-b", "person-x", 0.9, ("evidence-b",))
    before = registry.state

    results = [registry.compatibility("context-a", "context-b") for _ in range(3)]

    assert all(result.relation is ContextRelation.UNRELATED for result in results)
    assert registry.state == before
    assert registry.state.revision == before.revision


def test_minimal_interlocutor_uncertainty_preserves_distinct_references() -> None:
    registry = ContextRegistry()
    first = registry.upsert_interlocutor_binding(
        "ref-A", "person-X", 0.6, ("episode-A",)
    )
    second = registry.upsert_interlocutor_binding(
        "ref-B", "person-X", 0.4, ("episode-B",)
    )

    assert first.reference_key != second.reference_key
    assert first.identity_key == second.identity_key == "person-X"
    assert registry.state.interlocutor_bindings == (first, second)
    assert registry.state.revision == 2


def test_binding_upsert_is_explicit_canonical_and_idempotent() -> None:
    registry = ContextRegistry()
    created = registry.upsert_interlocutor_binding(
        "ref-A", None, 0, ("episode-b", "episode-a", "episode-b")
    )
    assert created.confidence == 0.0
    assert created.evidence_references == ("episode-a", "episode-b")
    assert created.created_revision == created.last_modified_revision == 1
    assert (
        registry.upsert_interlocutor_binding(
            "ref-A", None, 0.0, ("episode-a", "episode-b")
        )
        is created
    )
    assert registry.state.revision == 1

    updated = registry.upsert_interlocutor_binding(
        "ref-A", "person-X", 0.5, ("episode-c",)
    )
    assert updated.created_revision == 1
    assert updated.last_modified_revision == 2
    assert registry.state.revision == 2


def test_context_participant_may_have_no_binding() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    frame = create_context(registry, "context-a", participants=("unknown-ref",))

    assert frame.participant_refs == ("unknown-ref",)
    assert registry.state.interlocutor_bindings == ()


@pytest.mark.parametrize(
    "confidence", [math.nan, math.inf, -math.inf, -0.01, 1.01, True]
)
def test_confidence_is_strict_finite_unit_interval(confidence: object) -> None:
    registry = ContextRegistry()
    before = registry.state

    with pytest.raises(ContextStateInvalid, match="confidence"):
        registry.upsert_interlocutor_binding(  # type: ignore[arg-type]
            "ref-A", "person-X", confidence
        )

    assert registry.state == before


def test_extreme_confidence_and_failing_reference_iterable_are_bounded() -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    with pytest.raises(ContextStateInvalid, match="confidence"):
        registry.upsert_interlocutor_binding("ref-A", confidence=10**10_000)

    class FailingReferences:
        def __iter__(self):
            raise RuntimeError("PRIVATE-SENTINEL-R09")

    with pytest.raises(ContextStateInvalid, match="references") as captured:
        registry.create(
            "context-a",
            ContextType.CONVERSATION,
            "chat",
            participant_refs=FailingReferences(),  # type: ignore[arg-type]
        )
    assert "PRIVATE-SENTINEL-R09" not in str(captured.value)
    assert registry.state.revision == 0


@pytest.mark.parametrize(
    "bad_id",
    ["", " space", "line\nbreak", "tab\tvalue", "../context", "a/b", "a\\b", "x" * 129],
)
def test_all_opaque_identifiers_use_bounded_non_path_ascii_tokens(bad_id: str) -> None:
    registry = ContextRegistry(clock=DeterministicClock())
    before = registry.state

    with pytest.raises(ContextStateInvalid, match="identifier"):
        registry.create(bad_id, ContextType.CONVERSATION, "chat")
    with pytest.raises(ContextStateInvalid):
        registry.create("valid", ContextType.CONVERSATION, bad_id)
    with pytest.raises(ContextStateInvalid):
        registry.upsert_interlocutor_binding("valid-ref", bad_id, 0.5)
    with pytest.raises(ContextStateInvalid):
        registry.upsert_interlocutor_binding("valid-ref", "person", 0.5, (bad_id,))
    assert registry.state == before


def test_finite_capacity_exhaustion_is_atomic_without_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        MAX_CONTEXTS,
        MAX_PARTICIPANTS_PER_CONTEXT,
        MAX_RELATIONS_PER_CONTEXT,
        MAX_INTERLOCUTOR_BINDINGS,
        MAX_EVIDENCE_REFERENCES,
    ) == (1024, 32, 64, 1024, 32)

    monkeypatch.setattr(context_module, "MAX_CONTEXTS", 2)
    registry = ContextRegistry(clock=DeterministicClock())
    create_context(registry, "context-a")
    create_context(registry, "context-b")
    before = registry.state
    with pytest.raises(ContextCapacityExceeded, match="context capacity"):
        create_context(registry, "context-c")
    assert registry.state == before

    monkeypatch.setattr(context_module, "MAX_INTERLOCUTOR_BINDINGS", 1)
    registry.upsert_interlocutor_binding("ref-a", confidence=0.5)
    before = registry.state
    with pytest.raises(ContextCapacityExceeded, match="binding capacity"):
        registry.upsert_interlocutor_binding("ref-b", confidence=0.5)
    assert registry.state == before


def test_participant_evidence_and_relation_capacities_fail_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_module, "MAX_PARTICIPANTS_PER_CONTEXT", 2)
    registry = ContextRegistry(clock=DeterministicClock())
    with pytest.raises(ContextCapacityExceeded, match="reference capacity"):
        create_context(registry, "context-a", participants=("ref-a", "ref-b", "ref-c"))
    assert registry.state.revision == 0

    monkeypatch.setattr(context_module, "MAX_EVIDENCE_REFERENCES", 2)
    with pytest.raises(ContextCapacityExceeded, match="reference capacity"):
        registry.upsert_interlocutor_binding(
            "ref-a", confidence=0.5, evidence_references=("ev-a", "ev-b", "ev-c")
        )
    assert registry.state.revision == 0

    monkeypatch.setattr(context_module, "MAX_RELATIONS_PER_CONTEXT", 1)
    for context_id in ("context-a", "context-b", "context-c"):
        create_context(registry, context_id)
    registry.relate("context-a", "context-b")
    before = registry.state
    with pytest.raises(ContextCapacityExceeded, match="relation capacity"):
        registry.relate("context-a", "context-c")
    assert registry.state == before


def test_state_projection_has_canonical_order_independent_of_insert_order() -> None:
    first = ContextRegistry(clock=DeterministicClock())
    second = ContextRegistry(clock=DeterministicClock())
    for registry, ids in ((first, ("c", "a", "b")), (second, ("b", "c", "a"))):
        for context_id in ids:
            create_context(registry, context_id, participants=("ref-z", "ref-a"))
        for reference in reversed(ids):
            registry.upsert_interlocutor_binding(
                f"ref-{reference}", confidence=0.5, evidence_references=("ev-z", "ev-a")
            )

    for state in (first.state, second.state):
        assert tuple(frame.context_id for frame in state.frames) == ("a", "b", "c")
        assert tuple(
            binding.reference_key for binding in state.interlocutor_bindings
        ) == ("ref-a", "ref-b", "ref-c")
        assert all(
            frame.participant_refs == ("ref-a", "ref-z") for frame in state.frames
        )
        assert all(
            binding.evidence_references == ("ev-a", "ev-z")
            for binding in state.interlocutor_bindings
        )


def test_exact_restore_round_trips_nontrivial_state_without_clock_or_replay() -> None:
    source = nontrivial_registry()
    snapshot = source.state
    clock = DeterministicClock()
    target = ContextRegistry(clock=clock)

    target.restore_exact(snapshot)

    assert target.state == snapshot
    assert target.state.revision == snapshot.revision
    assert target.current_context_id == "context-a"
    assert clock.calls == 0


def malformed_states() -> tuple[ContextRegistryState, ...]:
    valid = nontrivial_registry().state
    frames = {frame.context_id: frame for frame in valid.frames}
    bindings = valid.interlocutor_bindings
    naive = NOW.replace(tzinfo=None)
    return (
        replace(
            valid,
            frames=(
                replace(frames["context-a"], parent_context_id="missing"),
                frames["context-b"],
                frames["context-c"],
            ),
        ),
        replace(
            valid,
            frames=(
                replace(frames["context-a"], related_context_ids=()),
                frames["context-b"],
                frames["context-c"],
            ),
        ),
        replace(
            valid,
            current_context_id="context-b",
        ),
        replace(
            valid,
            frames=(
                replace(frames["context-a"], status=ContextStatus.CLOSED),
                frames["context-b"],
                frames["context-c"],
            ),
        ),
        replace(valid, frames=(frames["context-a"], frames["context-a"])),
        replace(
            valid,
            interlocutor_bindings=(bindings[0], bindings[0]),
        ),
        replace(
            valid,
            frames=(
                replace(
                    frames["context-a"],
                    created_revision=valid.revision + 1,
                    last_modified_revision=valid.revision,
                ),
                frames["context-b"],
                frames["context-c"],
            ),
        ),
        replace(
            valid,
            frames=(
                replace(frames["context-a"], started_at=naive),
                frames["context-b"],
                frames["context-c"],
            ),
        ),
        replace(
            valid,
            frames=(
                replace(frames["context-a"], context_id="bad/id"),
                frames["context-b"],
                frames["context-c"],
            ),
        ),
    )


@pytest.mark.parametrize("malformed", malformed_states())
def test_malformed_restore_fails_closed_without_partial_mutation(
    malformed: ContextRegistryState,
) -> None:
    clock = DeterministicClock()
    target = ContextRegistry(clock=clock)
    create_context(target, "existing")
    before = target.state
    calls_before = clock.calls

    with pytest.raises(ContextStateInvalid):
        target.restore_exact(malformed)

    assert target.state == before
    assert clock.calls == calls_before


def test_restore_rejects_boolean_revision_bad_time_order_and_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = nontrivial_registry().state
    target = ContextRegistry()
    frame = valid.frames[0]

    bad_states = (
        replace(valid, revision=True),
        replace(
            valid,
            frames=(
                replace(frame, last_active_at=frame.started_at - timedelta(seconds=1)),
                *valid.frames[1:],
            ),
        ),
    )
    for state in bad_states:
        with pytest.raises(ContextStateInvalid):
            target.restore_exact(state)
        assert target.state.revision == 0

    monkeypatch.setattr(context_module, "MAX_CONTEXTS", 2)
    with pytest.raises(ContextCapacityExceeded):
        target.restore_exact(valid)
    assert target.state.revision == 0


def test_restore_rejects_zero_object_revisions_and_parent_cycles() -> None:
    valid = nontrivial_registry().state
    target = ContextRegistry()
    frames = {frame.context_id: frame for frame in valid.frames}
    zero_revision = replace(
        valid,
        frames=(
            replace(frames["context-a"], created_revision=0, last_modified_revision=0),
            frames["context-b"],
            frames["context-c"],
        ),
    )
    parent_cycle = replace(
        valid,
        frames=(
            replace(frames["context-a"], parent_context_id="context-b"),
            frames["context-b"],
            frames["context-c"],
        ),
    )
    duplicate_creation_revision = replace(
        valid,
        interlocutor_bindings=(
            replace(
                valid.interlocutor_bindings[0],
                created_revision=frames["context-a"].created_revision,
            ),
        ),
    )

    for malformed in (zero_revision, parent_cycle, duplicate_creation_revision):
        with pytest.raises(ContextStateInvalid):
            target.restore_exact(malformed)
        assert target.state.revision == 0


def test_restore_rejects_noncanonical_order_and_references() -> None:
    valid = nontrivial_registry().state
    target = ContextRegistry()

    with pytest.raises(ContextStateInvalid, match="ordering"):
        target.restore_exact(replace(valid, frames=tuple(reversed(valid.frames))))
    with pytest.raises(ContextStateInvalid, match="noncanonical references"):
        target.restore_exact(
            replace(
                valid,
                frames=(
                    replace(valid.frames[0], participant_refs=("ref-b", "ref-a")),
                    *valid.frames[1:],
                ),
            )
        )

    class TupleSubclass(tuple[object, ...]):
        pass

    with pytest.raises(ContextStateInvalid, match="registry state"):
        target.restore_exact(
            replace(valid, frames=TupleSubclass(valid.frames))  # type: ignore[arg-type]
        )
    with pytest.raises(ContextStateInvalid, match="frame references"):
        target.restore_exact(
            replace(
                valid,
                frames=(
                    replace(
                        valid.frames[0],
                        participant_refs=TupleSubclass(
                            valid.frames[0].participant_refs
                        ),  # type: ignore[arg-type]
                    ),
                    *valid.frames[1:],
                ),
            )
        )
    assert target.state.revision == 0


def test_restore_rejects_equivalent_non_utc_timestamps_without_mutating() -> None:
    valid = nontrivial_registry().state
    offset = timezone(timedelta(hours=9))
    frame = valid.frames[0]
    non_utc_frame = replace(
        frame,
        started_at=frame.started_at.astimezone(offset),
        last_active_at=frame.last_active_at.astimezone(offset),
    )
    malformed = replace(valid, frames=(non_utc_frame, *valid.frames[1:]))
    target = ContextRegistry(clock=DeterministicClock())
    create_context(target, "existing")
    before = target.state

    with pytest.raises(ContextStateInvalid, match="timestamp"):
        target.restore_exact(malformed)

    assert target.state == before


def test_context_models_are_immutable_and_have_exact_fields() -> None:
    assert issubclass(ContextNotFound, ContextError)
    assert issubclass(ContextConflict, ContextError)
    assert issubclass(ContextCapacityExceeded, ContextError)
    assert issubclass(ContextStateInvalid, ContextError)
    assert tuple(field.name for field in fields(ContextFrame)) == (
        "context_id",
        "context_type",
        "source_channel",
        "source_session_id",
        "participant_refs",
        "parent_context_id",
        "related_context_ids",
        "status",
        "created_revision",
        "last_modified_revision",
        "started_at",
        "last_active_at",
    )
    assert tuple(field.name for field in fields(ContextRegistryState)) == (
        "revision",
        "current_context_id",
        "frames",
        "interlocutor_bindings",
    )
    assert tuple(field.name for field in fields(InterlocutorBinding)) == (
        "reference_key",
        "identity_key",
        "confidence",
        "evidence_references",
        "created_revision",
        "last_modified_revision",
    )
    assert tuple(field.name for field in fields(ContextCompatibility)) == (
        "score",
        "relation",
        "source_context_id",
        "current_context_id",
    )
    frame = create_context(ContextRegistry(clock=DeterministicClock()), "immutable")
    with pytest.raises(FrozenInstanceError):
        frame.status = ContextStatus.CLOSED  # type: ignore[misc]


def test_context_domain_has_no_later_authority_or_raw_local_text_fields() -> None:
    field_names = {
        field.name
        for model in (ContextFrame, ContextRegistryState, InterlocutorBinding)
        for field in fields(model)
    }
    forbidden_fields = {
        "active_topic",
        "active_task",
        "message",
        "turn",
        "turns",
        "transcript",
        "content",
        "prompt",
        "raw_prompt",
        "hidden_thought",
        "private_reasoning",
        "attachment",
        "attachments",
        "relationship_metadata",
        "preferences",
        "trust",
        "closeness",
        "metadata",
        "extensions",
    }
    assert field_names.isdisjoint(forbidden_fields)

    tree = ast.parse(Path(inspect.getfile(context_module)).read_text())
    imported_modules = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported_modules <= {
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "enum",
        "kagya",
        "math",
        "re",
        "threading",
        "typing",
    }


def test_create_requires_explicit_id_and_has_no_hidden_randomness() -> None:
    signature = inspect.signature(ContextRegistry.create)
    assert signature.parameters["context_id"].default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        ContextRegistry().create(  # type: ignore[call-arg]
            context_type=ContextType.CONVERSATION, source_channel="chat"
        )


def test_errors_never_include_arbitrary_offending_payload() -> None:
    sentinel = "PRIVATE-SENTINEL-R09/invalid"
    with pytest.raises(ContextStateInvalid) as captured:
        ContextRegistry().create(sentinel, ContextType.CONVERSATION, "chat")

    assert sentinel not in str(captured.value)

    class DomainErrorIterable:
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise ContextStateInvalid(sentinel)

    with pytest.raises(ContextStateInvalid, match="invalid references") as captured:
        ContextRegistry().create(
            "context-a",
            ContextType.CONVERSATION,
            "chat",
            participant_refs=DomainErrorIterable(),
        )
    assert sentinel not in str(captured.value)
