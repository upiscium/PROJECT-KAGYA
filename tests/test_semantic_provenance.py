"""U3 Semantic source-Context provenance and DB2 compatibility tests."""

import json
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
    SemanticMemoryFormatError,
)
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "semantic_u3_db1",
                    "db2_collection": "semantic_u3_db2",
                }
            )
        }
    )


def _publish_source(
    memory: DualMemorySystem, episode_id: str, context_id: str
) -> None:
    memory.publish_coordinated_episodic(
        episode_id,
        f"source input {episode_id}",
        "source response",
        loss=0.1,
        emotion_valence=0.2,
        emotion_arousal=0.3,
        record_type=MemoryRecordType.EPISODIC_LOG,
        created_at="2026-01-01T00:00:00+00:00",
        coordination_schema=2,
        context_id=context_id,
        source_channel="chat",
    )


def _metadata(memory: DualMemorySystem, semantic_id: str) -> dict[str, object]:
    stored = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])
    return dict(stored["metadatas"][0] or {})


def test_singleton_and_all_same_sources_round_trip_context_and_source_ids(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-x", "context-x")
    _publish_source(memory, "episode-x2", "context-x")

    singleton = ["episode-x"]
    singleton_id = memory.save_legacy_semantic(
        "singleton context fact", source_episode_ids=singleton
    )
    all_same = ["episode-x", "episode-x2"]
    all_same_id = memory.save_legacy_semantic(
        "all same context fact", source_episode_ids=all_same
    )

    assert singleton == ["episode-x"]
    assert memory.get_committed_semantic(singleton_id).record.context_id == "context-x"  # type: ignore[union-attr]
    assert memory.get_committed_semantic(singleton_id).record.source_episode_ids == singleton  # type: ignore[union-attr]
    assert memory.get_committed_semantic(all_same_id).record.context_id == "context-x"  # type: ignore[union-attr]
    assert memory.get_committed_semantic(all_same_id).record.source_episode_ids == all_same  # type: ignore[union-attr]
    assert _metadata(memory, singleton_id)["context_id"] == "context-x"
    assert _metadata(memory, all_same_id)["context_id"] == "context-x"


def test_literal_none_context_id_round_trips_as_known_provenance(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-none-string", "None")

    semantic_id = memory.save_legacy_semantic(
        "literal None context fact", source_episode_ids=["episode-none-string"]
    )

    committed = memory.get_committed_semantic(semantic_id)
    assert committed is not None
    assert committed.record.context_id == "None"
    assert _metadata(memory, semantic_id)["context_id"] == "None"


def test_mixed_unknown_missing_and_empty_sources_store_none_without_dropping_refs(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-x", "context-x")
    _publish_source(memory, "episode-y", "context-y")
    unknown_id = memory.save_episodic("legacy source", "legacy response")

    source_sets = (
        ["episode-x", "episode-y"],
        ["episode-x", unknown_id],
        ["episode-x", "episode-missing"],
        [unknown_id],
        [],
    )
    for index, source_ids in enumerate(source_sets):
        semantic_id = memory.save_legacy_semantic(
            f"none context fact {index}", source_episode_ids=source_ids
        )
        committed = memory.get_committed_semantic(semantic_id)

        assert committed is not None
        assert committed.record.context_id is None
        assert committed.record.source_episode_ids == source_ids
        assert "context_id" not in _metadata(memory, semantic_id)


def test_source_order_and_duplicate_lookup_do_not_change_context_or_saved_refs(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-x", "context-x")
    _publish_source(memory, "episode-x2", "context-x")
    _publish_source(memory, "episode-y", "context-y")

    duplicate_refs = ["episode-x", "episode-x", "episode-x2"]
    duplicate_id = memory.save_legacy_semantic(
        "duplicate source fact", source_episode_ids=duplicate_refs
    )
    reverse_mixed_refs = ["episode-y", "episode-x"]
    reverse_mixed_id = memory.save_legacy_semantic(
        "reverse mixed fact", source_episode_ids=reverse_mixed_refs
    )

    duplicate = memory.get_committed_semantic(duplicate_id)
    reverse_mixed = memory.get_committed_semantic(reverse_mixed_id)
    assert duplicate is not None
    assert reverse_mixed is not None
    assert duplicate.record.context_id == "context-x"
    assert duplicate.record.source_episode_ids == duplicate_refs
    assert reverse_mixed.record.context_id is None
    assert reverse_mixed.record.source_episode_ids == reverse_mixed_refs


def test_source_ids_are_copied_and_invalid_source_arguments_publish_nothing(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-x", "context-x")
    refs = ["episode-x"]
    semantic_id = memory.save_legacy_semantic("copied refs", source_episode_ids=refs)
    refs.append("episode-missing")

    committed = memory.get_committed_semantic(semantic_id)
    assert committed is not None
    assert committed.record.source_episode_ids == ["episode-x"]

    before = memory.db2.get(include=["documents", "metadatas"])
    with pytest.raises(TypeError):
        memory.save_legacy_semantic("tuple refs", source_episode_ids=("episode-x",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        memory.save_legacy_semantic("empty ref", source_episode_ids=[""])
    with pytest.raises(TypeError):
        memory.save_legacy_semantic("typed ref", source_episode_ids=[None])  # type: ignore[list-item]
    assert memory.db2.get(include=["documents", "metadatas"]) == before


def test_context_metadata_is_evidence_derived_not_extra_authority(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-x", "context-x")

    semantic_id = memory.save_legacy_semantic(
        "evidence wins",
        source_episode_ids=["episode-x"],
        metadata={"context_id": "context-y", "safe": "kept"},
    )

    stored = _metadata(memory, semantic_id)
    assert stored["context_id"] == "context-x"
    assert json.loads(stored["extra"]) == {
        "context_id": "context-y",
        "safe": "kept",
    }
    committed = memory.get_committed_semantic(semantic_id)
    assert committed is not None
    assert committed.record.context_id == "context-x"


def test_unavailable_source_is_bounded_and_does_not_publish_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-x", "context-x")
    before = memory.db2.get(include=["documents", "metadatas"])
    original_get = memory.db1.get

    def unavailable(**kwargs: object) -> object:
        if kwargs.get("ids") == ["episode-unavailable"]:
            raise RuntimeError("PRIVATE DB1 backend detail")
        return original_get(**kwargs)

    monkeypatch.setattr(memory.db1, "get", unavailable)

    with pytest.raises(EpisodicMemoryReadError) as error:
        memory.save_legacy_semantic(
            "unavailable source fact",
            source_episode_ids=["episode-x", "episode-unavailable"],
        )

    assert "PRIVATE" not in str(error.value)
    assert error.value.__cause__ is None
    assert memory.db2.get(include=["documents", "metadatas"]) == before


def test_source_failure_is_not_hidden_after_unknown_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    unknown_id = memory.save_episodic("unknown source", "legacy response")
    calls: list[str] = []
    original = memory.get_committed_episodic

    def read(source_id: str) -> object:
        calls.append(source_id)
        if source_id == "episode-unavailable":
            raise EpisodicMemoryReadError("Committed episodic Memory is unavailable")
        return original(source_id)

    monkeypatch.setattr(memory, "get_committed_episodic", read)

    with pytest.raises(EpisodicMemoryReadError):
        memory.save_legacy_semantic(
            "failure after unknown",
            source_episode_ids=[unknown_id, "episode-unavailable"],
        )

    assert calls == [unknown_id, "episode-unavailable"]


def test_malformed_source_is_bounded_and_does_not_publish_semantic(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-malformed", "context-x")
    metadata = dict(
        memory.db1.get(ids=["episode-malformed"], include=["metadatas"])["metadatas"][0]
    )
    metadata["context_id"] = "bad/id"
    memory.db1.update(ids=["episode-malformed"], metadatas=[metadata])
    before = memory.db2.get(include=["documents", "metadatas"])

    with pytest.raises(EpisodicMemoryFormatError) as error:
        memory.save_legacy_semantic(
            "malformed source fact", source_episode_ids=["episode-malformed"]
        )

    assert str(error.value) == "Committed episodic Memory is invalid"
    assert memory.db2.get(include=["documents", "metadatas"]) == before


def test_consolidation_derives_context_and_read_does_not_recompute_after_source_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-consolidate", "context-x")
    source_metadata = dict(
        memory.db1.get(ids=["episode-consolidate"], include=["metadatas"])[
            "metadatas"
        ][0]
    )
    source_metadata["emotion_arousal"] = 1.0
    memory.db1.update(ids=["episode-consolidate"], metadatas=[source_metadata])

    semantic_ids = memory.consolidate_to_legacy_semantic(DummyProvider())

    assert len(semantic_ids) == 1
    semantic = memory.get_committed_semantic(semantic_ids[0])
    source = memory.get_committed_episodic("episode-consolidate")
    assert semantic is not None
    assert source is not None
    assert semantic.record.context_id == "context-x"
    assert semantic.record.source_episode_ids == ["episode-consolidate"]
    assert source.record.archived is True

    memory.db1.delete(ids=["episode-consolidate"])

    def no_source_recompute(_source_id: str) -> object:
        raise AssertionError("Semantic read must not consult DB1")

    monkeypatch.setattr(memory, "get_committed_episodic", no_source_recompute)
    retained = memory.get_committed_semantic(semantic_ids[0])
    assert retained is not None
    assert retained.record.context_id == "context-x"


def test_consolidation_does_not_archive_when_semantic_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-failure", "context-x")
    source_metadata = dict(
        memory.db1.get(ids=["episode-failure"], include=["metadatas"])["metadatas"][0]
    )
    source_metadata["emotion_arousal"] = 1.0
    memory.db1.update(ids=["episode-failure"], metadatas=[source_metadata])

    def fail_add(**_: object) -> None:
        raise RuntimeError("DB2 publication failed")

    monkeypatch.setattr(memory.db2, "add", fail_add)
    with pytest.raises(RuntimeError, match="DB2 publication failed"):
        memory.consolidate_to_legacy_semantic(DummyProvider())

    source = memory.get_committed_episodic("episode-failure")
    assert source is not None
    assert source.record.archived is False


def test_legacy_db2_without_context_is_read_without_backfill_or_rewrite(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    semantic_id = memory.save_legacy_semantic("legacy semantic")
    before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])

    reopened = DualMemorySystem(settings)
    committed = reopened.get_committed_semantic(semantic_id)

    assert committed is not None
    assert committed.record.context_id is None
    assert reopened.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before


@pytest.mark.parametrize("invalid_context_id", ["", "bad/id", "x\n", "x" * 129])
def test_present_invalid_context_id_fails_exact_read_without_repair(
    tmp_path: Path, invalid_context_id: str
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    semantic_id = memory.save_legacy_semantic("invalid semantic")
    metadata = _metadata(memory, semantic_id)
    metadata["context_id"] = invalid_context_id
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])

    with pytest.raises(SemanticMemoryFormatError):
        memory.get_committed_semantic(semantic_id)

    assert memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before


def test_startup_scrub_rejects_invalid_context_before_rewriting_record(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    semantic_id = memory.save_legacy_semantic("startup invalid semantic")
    metadata = _metadata(memory, semantic_id)
    metadata["context_id"] = "bad/id"
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])

    with pytest.raises(SemanticMemoryFormatError):
        DualMemorySystem(settings)

    assert memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before


def test_startup_scrub_rejects_conflicting_semantic_metadata_without_repair(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    semantic_id = memory.save_legacy_semantic("startup conflicting semantic")
    metadata = _metadata(memory, semantic_id)
    metadata["text"] = "forged semantic text"
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])

    with pytest.raises(SemanticMemoryFormatError):
        DualMemorySystem(settings)

    assert memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before


def test_query_projection_preserves_known_context_and_missing_key_is_none(
    tmp_path: Path,
) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    _publish_source(memory, "episode-query", "context-query")
    known_id = memory.save_legacy_semantic(
        "query context sentinel", source_episode_ids=["episode-query"]
    )
    legacy_id = memory.save_legacy_semantic("query legacy sentinel")

    context = memory.retrieve_context("query context sentinel")
    records = {record.id: record for record in context.db2_results}

    assert records[known_id].context_id == "context-query"
    assert records[legacy_id].context_id is None


def test_query_rejects_present_invalid_context_without_repair(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    semantic_id = memory.save_legacy_semantic("query invalid context sentinel")
    metadata = _metadata(memory, semantic_id)
    metadata["context_id"] = "bad/id"
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])

    with pytest.raises(SemanticMemoryFormatError):
        memory.retrieve_context("query invalid context sentinel")

    assert memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before


def test_empty_source_does_not_consult_db1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = DualMemorySystem(_settings(tmp_path))

    def unexpected_source_read(_source_id: str) -> object:
        raise AssertionError("source-less Semantic save must not read DB1")

    monkeypatch.setattr(memory, "get_committed_episodic", unexpected_source_read)
    semantic_id = memory.save_legacy_semantic("source-less semantic")

    committed = memory.get_committed_semantic(semantic_id)
    assert committed is not None
    assert committed.record.context_id is None
