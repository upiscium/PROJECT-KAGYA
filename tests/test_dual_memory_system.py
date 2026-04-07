from __future__ import annotations

from project_kagya.dual_memory_system import (
    DualMemorySystem,
    InMemoryMemoryCollection,
    SemanticRecord,
)


def test_save_episodic_clamps_out_of_range_values() -> None:
    system = DualMemorySystem()

    record = system.save_episodic("hello", "world", 2.0, -1.0)

    assert record.valence == 1.0
    assert record.arousal == 0.0
    assert len(system.hippocampus.all()) == 1


def test_retrieve_context_prioritizes_recent_memory_and_handles_empty_sections() -> (
    None
):
    system = DualMemorySystem()

    output = system.retrieve_context("missing query")

    assert "Recent Memory (DB1)" in output
    assert "Semantic Memory (DB2)" in output
    assert "- なし" in output


def test_retrieve_context_formats_db1_and_db2_entries() -> None:
    system = DualMemorySystem()
    system.save_episodic("user prefers concise answers", "ok", 0.8, 0.2)
    system.cortex.add(
        SemanticRecord(
            id="semantic-1",
            fact="user prefers Japanese replies",
            source_ids=["source-1"],
            confidence=0.9,
            status="confirmed",
            timestamp="2026-04-07T00:00:00+00:00",
        )
    )

    output = system.retrieve_context("prefers")

    assert "Recent Memory (DB1)" in output
    assert "Semantic Memory (DB2)" in output
    assert "user prefers concise answers" in output
    assert "user prefers Japanese replies" in output


def test_consolidate_to_semantic_only_migrates_confirmed_items() -> None:
    system = DualMemorySystem()
    source = system.save_episodic("user likes tea", "noted", 0.3, 0.4)

    def pipeline(_: object) -> list[dict[str, object]]:
        return [
            {
                "id": "semantic-1",
                "fact": "user likes tea",
                "source_ids": [source.id],
                "confidence": 0.95,
                "status": "confirmed",
                "timestamp": "2026-04-07T00:00:00+00:00",
            },
            {
                "id": "semantic-2",
                "fact": "user may like coffee",
                "source_ids": [source.id],
                "confidence": 0.4,
                "status": "tentative",
                "timestamp": "2026-04-07T00:00:00+00:00",
            },
            {
                "id": "semantic-3",
                "fact": "user dislikes tea",
                "source_ids": [source.id],
                "confidence": 0.2,
                "status": "conflicted",
                "timestamp": "2026-04-07T00:00:00+00:00",
            },
        ]

    result = system.consolidate_to_semantic(pipeline)

    assert result.migrated == 1
    assert result.pending == 1
    assert result.failed == 1
    assert len(system.cortex.all()) == 1
    assert len(system.hippocampus.all()) == 0


def test_consolidate_to_semantic_keeps_episodic_records_on_failure() -> None:
    system = DualMemorySystem()
    system.save_episodic("user likes coffee", "noted", 0.2, 0.1)

    def pipeline(_: object) -> None:
        return None

    result = system.consolidate_to_semantic(pipeline)

    assert result.migrated == 0
    assert result.pending == 0
    assert result.failed == 0
    assert len(system.hippocampus.all()) == 1
