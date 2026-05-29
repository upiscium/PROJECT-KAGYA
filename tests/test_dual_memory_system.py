from pathlib import Path

from kagya.config import Settings, load_settings
from kagya.memory import DualMemorySystem, MemoryRecordType
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_saving_episodic_record_returns_episode_id(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))

    episode_id = memory.save_episodic("hello", "world")

    assert episode_id.startswith("episode-")


def test_saved_episodic_records_can_be_retrieved_from_db1(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic(
        "I like lunar gardens",
        "Remembering lunar gardens.",
        hidden_thought="garden affinity",
        loss=0.25,
        emotion_valence=0.7,
        emotion_arousal=0.8,
    )

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db1_results] == [episode_id]
    assert context.db1_results[0].record_type == MemoryRecordType.EPISODIC_LOG
    assert context.db1_results[0].hidden_thought == "garden affinity"


def test_semantic_records_can_be_retrieved_from_db2(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("The user likes lunar gardens.")

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db2_results] == [semantic_id]
    assert context.db2_results[0].record_type == MemoryRecordType.SEMANTIC_MEMORY


def test_consolidation_archives_db1_records_instead_of_deleting(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("fact", "response", emotion_arousal=1.0)

    semantic_ids = memory.consolidate_to_semantic(DummyProvider())

    assert len(semantic_ids) == 1
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    assert stored["ids"] == [episode_id]
    assert stored["metadatas"][0]["archived"] is True
    assert memory.retrieve_context("fact").db1_results == []


def test_retrieval_respects_configured_db1_and_db2_top_k(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings_for_tmp_memory(tmp_path, db1_top_k=2, db2_top_k=1))
    for index in range(3):
        memory.save_episodic(f"shared topic episode {index}", "response")
        memory.save_semantic(f"shared topic semantic {index}")

    context = memory.retrieve_context("shared topic")

    assert len(context.db1_results) == 2
    assert len(context.db2_results) == 1


def _settings_for_tmp_memory(
    tmp_path: Path,
    *,
    db1_top_k: int = 5,
    db2_top_k: int = 5,
) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_test",
                    "db2_collection": "cortex_test",
                    "db1_top_k": db1_top_k,
                    "db2_top_k": db2_top_k,
                }
            )
        }
    )
