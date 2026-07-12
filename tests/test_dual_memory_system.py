from pathlib import Path

from kagya.config import Settings, load_settings
from kagya.memory import (
    DeterministicEmbeddingFunction,
    DualMemorySystem,
    MemoryRecordType,
    SentenceTransformerEmbeddingFunction,
    create_embedding_function,
)
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_saving_episodic_record_returns_episode_id(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))

    episode_id = memory.save_episodic("hello", "world")

    assert episode_id.startswith("episode-")


def test_saved_episodic_records_can_be_retrieved_from_db1(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
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
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("The user likes lunar gardens.")

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db2_results] == [semantic_id]
    assert context.db2_results[0].record_type == MemoryRecordType.SEMANTIC_MEMORY


def test_consolidation_archives_db1_records_instead_of_deleting(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("fact", "response", emotion_arousal=1.0)

    semantic_ids = memory.consolidate_to_semantic(DummyProvider())

    assert len(semantic_ids) == 1
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    assert stored["ids"] == [episode_id]
    assert stored["metadatas"][0]["archived"] is True
    assert memory.retrieve_context("fact").db1_results == []


def test_operator_can_archive_episodic_record_without_deleting(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("archive me", "kept")

    archived = memory.archive_episodic(episode_id)

    assert archived is not None
    assert archived.archived is True
    assert memory.db1.get(ids=[episode_id])["ids"] == [episode_id]
    assert memory.retrieve_context("archive me").db1_results == []


def test_operator_can_archive_semantic_record_without_deleting(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("archive semantic")

    archived = memory.archive_semantic(semantic_id)

    assert archived is not None
    assert archived.archived is True
    assert memory.db2.get(ids=[semantic_id])["ids"] == [semantic_id]
    assert memory.retrieve_context("archive semantic").db2_results == []


def test_operator_can_tag_episodic_and_semantic_records(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("tag episode", "response")
    semantic_id = memory.save_semantic("tag semantic")

    episode = memory.update_episodic_metadata(
        episode_id,
        tags=["review", " review ", "keep"],
        operator_metadata={"owner": "operator"},
    )
    semantic = memory.update_semantic_metadata(semantic_id, tags=["fact"])

    assert episode is not None
    assert episode.tags == ["review", "keep"]
    assert episode.operator_metadata == {"owner": "operator"}
    assert semantic is not None
    assert semantic.tags == ["fact"]


def test_retrieval_respects_configured_db1_and_db2_top_k(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path, db1_top_k=2, db2_top_k=1))
    for index in range(3):
        memory.save_episodic(f"shared topic episode {index}", "response")
        memory.save_semantic(f"shared topic semantic {index}")

    context = memory.retrieve_context("shared topic")

    assert len(context.db1_results) == 2
    assert len(context.db2_results) == 1


def test_default_embedding_function_uses_configured_model_id(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    embedding = create_embedding_function(settings)

    assert isinstance(embedding, SentenceTransformerEmbeddingFunction)
    assert embedding.model_id == settings.memory.embedding_model_id


def test_non_legacy_embedding_uses_versioned_collection_names(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)

    class NonLegacyEmbedding(DeterministicEmbeddingFunction):
        @staticmethod
        def name() -> str:
            return "sentence-transformers:test-model"

        @staticmethod
        def is_legacy() -> bool:
            return False

    memory = DualMemorySystem(settings, embedding_function=NonLegacyEmbedding())

    assert memory.db1.name.startswith(
        "hippocampus_test-sentence-transformers-test-model-"
    )
    assert memory.db2.name.startswith("cortex_test-sentence-transformers-test-model-")


def test_sentence_transformer_embedding_function_encodes_with_configured_model() -> None:
    loaded: dict[str, object] = {}

    class FakeModel:
        def encode(self, texts: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
            loaded["texts"] = texts
            loaded["normalize_embeddings"] = normalize_embeddings
            return [[1.0, 0.0] for _ in texts]

    embedding = SentenceTransformerEmbeddingFunction(
        "sentence-transformers/test-model",
        model_loader=lambda model_id: loaded.setdefault("model_id", model_id) and FakeModel(),
    )

    vectors = embedding(["hello", "world"])

    assert loaded["model_id"] == "sentence-transformers/test-model"
    assert loaded["texts"] == ["hello", "world"]
    assert loaded["normalize_embeddings"] is True
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]


def _memory(settings: Settings) -> DualMemorySystem:
    return DualMemorySystem(settings, embedding_function=DeterministicEmbeddingFunction())


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
