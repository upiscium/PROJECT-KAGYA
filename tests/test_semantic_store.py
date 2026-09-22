"""Durability tests for the Memory-owned R12 Semantic store."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kagya.memory.semantic_lifecycle import (
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticRevisionReason,
    semantic_content_digest,
)
from kagya.memory.semantic_store import (
    SemanticStore,
    SemanticStoreCorrupt,
    semantic_revision_from_dict,
    semantic_revision_to_dict,
)


CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _event_id(sequence: int) -> str:
    return f"11111111-1111-4111-8111-{sequence:012d}"


def _revision(
    semantic_id: str,
    revision: int,
    previous_digest: str | None = None,
) -> SemanticRevision:
    content = f"semantic revision {revision}"
    return SemanticRevision(
        semantic_id=semantic_id,
        revision=revision,
        semantic_content=content,
        content_digest=semantic_content_digest(content),
        created_at=CREATED_AT + timedelta(seconds=revision),
        operation=(
            SemanticRevisionOperation.CREATE
            if revision == 0
            else SemanticRevisionOperation.CORRECT
        ),
        reason=(
            SemanticRevisionReason.CREATION
            if revision == 0
            else SemanticRevisionReason.CORRECTION
        ),
        previous_revision_digest=previous_digest,
        event_id=_event_id(revision + 1),
        event_sequence=revision + 1,
    )


def test_revision_round_trip_and_exact_current_plus_32_retention(tmp_path: Path) -> None:
    store = SemanticStore(tmp_path / "semantic")
    semantic_id = "semantic:retention"
    current = _revision(semantic_id, 0)
    store.publish_create(current, "a" * 64)
    for revision_number in range(1, 35):
        revision = _revision(semantic_id, revision_number, current.revision_digest)
        store.publish_revision(
            revision,
            "b" * 64,
            expected_revision=current.revision,
            expected_digest=current.revision_digest,
        )
        current = revision

    loaded = store.load_current(semantic_id)
    assert loaded is not None
    assert loaded.revision == current
    files = sorted(
        (path.name for path in (store.records_root / semantic_id).iterdir() if path.suffix == ".json"),
        key=lambda name: int(name[:-5]),
    )
    assert files == [f"{revision}.json" for revision in range(2, 35)]
    assert loaded.anchor_revision == 1
    assert loaded.anchor_revision_digest == _revision(semantic_id, 1, _revision(semantic_id, 0).revision_digest).revision_digest


def test_revision_serialization_rejects_digest_tampering() -> None:
    revision = _revision("semantic:roundtrip", 0)
    payload = semantic_revision_to_dict(revision)
    assert semantic_revision_from_dict(payload) == revision
    payload["revision_digest"] = "0" * 64
    with pytest.raises(SemanticStoreCorrupt):
        semantic_revision_from_dict(payload)


def test_store_rejects_unknown_revision_artifact(tmp_path: Path) -> None:
    store = SemanticStore(tmp_path / "semantic")
    revision = _revision("semantic:unknown", 0)
    store.publish_create(revision, "a" * 64)
    unknown = store.records_root / revision.semantic_id / "unexpected.bin"
    unknown.write_bytes(b"not a revision")
    unknown.chmod(0o600)

    with pytest.raises(SemanticStoreCorrupt):
        store.load_current(revision.semantic_id)
