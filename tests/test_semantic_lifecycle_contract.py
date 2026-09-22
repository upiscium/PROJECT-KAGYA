"""Focused tests for the pure R12 Semantic lifecycle boundary."""

from datetime import UTC, datetime

import pytest

from kagya.memory.semantic_lifecycle import (
    SEMANTIC_MAX_EVENT_SEQUENCE,
    SEMANTIC_MAX_SOURCE_EDGES,
    SemanticLifecycle,
    SemanticProvenanceClass,
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticRevisionReason,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    classify_provenance,
    normalize_semantic_content,
    semantic_content_digest,
    validate_revision_digest,
    provenance_for_edges,
)


CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def edge(
    source_id: str,
    context_id: str | None,
    status: SemanticSourceStatus = SemanticSourceStatus.AVAILABLE,
    kind: SemanticSourceKind = SemanticSourceKind.EPISODIC,
    source_revision: int | None = None,
) -> SemanticSourceEdge:
    return SemanticSourceEdge(
        source_kind=kind,
        source_id=source_id,
        source_revision=source_revision,
        captured_context_id=context_id,
        source_status=status,
    )


def test_content_normalization_and_literal_golden_digest() -> None:
    assert normalize_semantic_content(" \r\n fact \r") == "fact"
    assert semantic_content_digest(" fact ") == (
        "6f1741f2f777d35a552efba91260b8788d69bfa12534eceb0c377d529d0d2afc"
    )
    with pytest.raises(ValueError):
        normalize_semantic_content("\x00fact")
    with pytest.raises(ValueError):
        normalize_semantic_content("x" * 16_385)
    with pytest.raises(ValueError):
        normalize_semantic_content(" \r\n")


def test_source_edges_are_canonical_and_conflicting_duplicates_fail() -> None:
    first = edge("episode:a", "context:x")
    second = edge("episode:b", "context:x")
    revision = SemanticRevision(
        "semantic:1",
        0,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
        source_edges=(second, first, first),
    )
    assert revision.source_edges == (first, second)
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            0,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            source_edges=(first, edge("episode:a", "context:y")),
        )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            0,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            source_edges=tuple(
                edge(f"episode:{index}", "context:x")
                for index in range(SEMANTIC_MAX_SOURCE_EDGES + 1)
            ),
        )


def test_source_kind_requires_the_matching_source_revision_shape() -> None:
    with pytest.raises(ValueError):
        edge("semantic:source", "context:x", kind=SemanticSourceKind.SEMANTIC)
    with pytest.raises(ValueError):
        edge(
            "episode:source",
            "context:x",
            kind=SemanticSourceKind.EPISODIC,
            source_revision=1,
        )
    semantic = edge(
        "semantic:source",
        "context:x",
        kind=SemanticSourceKind.SEMANTIC,
        source_revision=1,
    )
    assert semantic.source_revision == 1


@pytest.mark.parametrize(
    ("edges", "expected", "source_count"),
    [
        ((), SemanticProvenanceClass.UNKNOWN, 0),
        ((edge("episode:a", "context:x"),), SemanticProvenanceClass.SINGLE_CONTEXT, 1),
        (
            (edge("episode:a", "context:x"), edge("episode:b", "context:x")),
            SemanticProvenanceClass.SINGLE_CONTEXT,
            2,
        ),
        (
            (edge("episode:a", "context:x"), edge("episode:b", "context:y")),
            SemanticProvenanceClass.MULTI_CONTEXT,
            2,
        ),
        (
            (edge("episode:a", "context:x"), edge("episode:b", None)),
            SemanticProvenanceClass.INCOMPLETE,
            2,
        ),
        (
            (edge("episode:a", None, SemanticSourceStatus.UNKNOWN),),
            SemanticProvenanceClass.UNKNOWN,
            1,
        ),
        (
            (edge("episode:a", None, SemanticSourceStatus.MISSING),),
            SemanticProvenanceClass.INCOMPLETE,
            1,
        ),
    ],
)
def test_provenance_classification_preserves_source_count(
    edges: tuple[SemanticSourceEdge, ...],
    expected: SemanticProvenanceClass,
    source_count: int,
) -> None:
    assert classify_provenance(edges) is expected
    revision = SemanticRevision(
        "semantic:1",
        0,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
        source_edges=edges,
    )
    assert revision.provenance.source_count == source_count
    assert revision.provenance_class is expected


def test_provenance_digest_binds_the_canonical_source_graph() -> None:
    first = edge("episode:a", "context:x")
    second = edge("episode:b", "context:x")
    base = provenance_for_edges((first,))
    canonical = provenance_for_edges((first, second))
    reordered_duplicate = provenance_for_edges((second, first, first))
    assert canonical.digest == reordered_duplicate.digest
    assert canonical.source_edges == (first, second)

    assert base.digest != provenance_for_edges(
        (edge("episode:other", "context:x"),)
    ).digest
    assert base.digest != provenance_for_edges(
        (edge("episode:a", "context:y"),)
    ).digest

    semantic_revision_one = edge(
        "source:semantic",
        "context:x",
        kind=SemanticSourceKind.SEMANTIC,
        source_revision=1,
    )
    semantic_revision_two = edge(
        "source:semantic",
        "context:x",
        kind=SemanticSourceKind.SEMANTIC,
        source_revision=2,
    )
    assert provenance_for_edges((semantic_revision_one,)).digest != provenance_for_edges(
        (semantic_revision_two,)
    ).digest

    missing = edge(
        "episode:status",
        None,
        status=SemanticSourceStatus.MISSING,
    )
    retracted = edge(
        "episode:status",
        None,
        status=SemanticSourceStatus.RETRACTED,
    )
    assert provenance_for_edges((missing,)).digest != provenance_for_edges(
        (retracted,)
    ).digest

    episodic = edge("source:kind", "context:x")
    semantic = edge(
        "source:kind",
        "context:x",
        kind=SemanticSourceKind.SEMANTIC,
        source_revision=1,
    )
    assert provenance_for_edges((episodic,)).digest != provenance_for_edges(
        (semantic,)
    ).digest


def test_revision_digest_is_order_independent_and_immutable() -> None:
    first = edge("episode:a", "context:x")
    second = edge("episode:b", "context:y")
    genesis = SemanticRevision(
        "semantic:1",
        0,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
    )
    left = SemanticRevision(
        "semantic:1",
        1,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
        source_edges=(first, second),
        lifecycle=SemanticLifecycle.ACTIVE,
        operation=SemanticRevisionOperation.CORRECT,
        reason=SemanticRevisionReason.CORRECTION,
        previous_revision_digest=genesis.revision_digest,
    )
    right = SemanticRevision(
        "semantic:1",
        1,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
        source_edges=(second, first),
        lifecycle=SemanticLifecycle.ACTIVE,
        operation=SemanticRevisionOperation.CORRECT,
        reason=SemanticRevisionReason.CORRECTION,
        previous_revision_digest=genesis.revision_digest,
    )
    assert left.revision_digest == right.revision_digest
    assert validate_revision_digest(left) == left.revision_digest
    with pytest.raises(AttributeError):
        left.semantic_content = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        SemanticRevision("semantic:1", 0, "fact", "0" * 64, CREATED_AT)
    with pytest.raises(ValueError):
        SemanticRevision("semantic:1", 2**31, "fact", semantic_content_digest("fact"), CREATED_AT)


def test_revision_chain_event_binding_and_compatibility_are_strict() -> None:
    genesis = SemanticRevision(
        "semantic:1",
        0,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
    )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            0,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            previous_revision_digest="0" * 64,
        )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            1,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            operation=SemanticRevisionOperation.CORRECT,
            reason=SemanticRevisionReason.CORRECTION,
        )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            1,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            operation=SemanticRevisionOperation.CORRECT,
            reason=SemanticRevisionReason.CORRECTION,
            previous_revision_digest=genesis.revision_digest,
            event_id="event:1",
        )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            1,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            operation=SemanticRevisionOperation.CORRECT,
            reason=SemanticRevisionReason.CORRECTION,
            previous_revision_digest=genesis.revision_digest,
            event_id="event:1",
            event_sequence=0,
        )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            1,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            operation=SemanticRevisionOperation.CORRECT,
            reason=SemanticRevisionReason.CORRECTION,
            previous_revision_digest=genesis.revision_digest,
            event_id="event:1",
            event_sequence=SEMANTIC_MAX_EVENT_SEQUENCE + 1,
        )
    with pytest.raises(ValueError):
        SemanticRevision(
            "semantic:1",
            1,
            "fact",
            semantic_content_digest("fact"),
            CREATED_AT,
            lifecycle=SemanticLifecycle.RETRACTED,
            operation=SemanticRevisionOperation.CORRECT,
            reason=SemanticRevisionReason.CORRECTION,
            previous_revision_digest=genesis.revision_digest,
        )
    corrected = SemanticRevision(
        "semantic:1",
        1,
        "fact",
        semantic_content_digest("fact"),
        CREATED_AT,
        operation=SemanticRevisionOperation.CORRECT,
        reason=SemanticRevisionReason.CORRECTION,
        previous_revision_digest=genesis.revision_digest,
        event_id="event:1",
        event_sequence=1,
    )
    assert corrected.event_id == "event:1"
