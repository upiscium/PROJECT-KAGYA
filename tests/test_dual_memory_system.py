from pathlib import Path

from kagya.config import Settings, load_settings
from kagya.memory import (
    DeterministicEmbeddingFunction,
    DualMemorySystem,
    GenerationHealth,
    MemoryLifecycleStatus,
    MemoryRecordType,
    SentenceTransformerEmbeddingFunction,
    create_embedding_function,
    ValidationStatus,
)
from kagya.models import DummyProvider
from kagya.memory.memory_evaluator import MemoryEvaluator


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


def test_experience_salience_affects_retrieval_order(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    low = memory.save_episodic(
        "shared retrieval topic", "low", source_event_id="event-low"
    )
    high = memory.save_episodic(
        "shared retrieval topic", "high", source_event_id="event-high"
    )
    memory.link_experience(
        low,
        experience_id="experience-low",
        subjective_salience=0.1,
        autobiographical_importance=0.1,
    )
    memory.link_experience(
        high,
        experience_id="experience-high",
        subjective_salience=0.9,
        autobiographical_importance=0.8,
    )

    context = memory.retrieve_context("shared retrieval topic")

    assert [record.id for record in context.db1_results[:2]] == [high, low]
    assert context.db1_results[0].experience_id == "experience-high"


def test_experience_salience_can_select_low_arousal_episode_for_consolidation(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = DualMemorySystem(
        settings,
        embedding_function=DeterministicEmbeddingFunction(),
        evaluator=MemoryEvaluator(
            min_arousal=0.9,
            min_subjective_salience=0.7,
        ),
    )
    episode_id = memory.save_episodic(
        "subjectively important", "response", emotion_arousal=0.1
    )
    memory.link_experience(
        episode_id,
        experience_id="experience-important",
        subjective_salience=0.8,
        autobiographical_importance=0.9,
    )

    semantic_ids = memory.consolidate_to_semantic(DummyProvider())

    assert len(semantic_ids) == 1


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


def test_quarantined_generation_is_persisted_but_not_retrieved(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic(
        "hello",
        "broken broken broken",
        generation_health=GenerationHealth(
            healthy=False, reasons=["repetitive"], repetitive=True
        ),
        source_event_id="event-1",
        processing_sequence=3,
        provider="dummy",
        model_id="dummy-model",
    )

    record = memory.get_episodic(episode_id)
    assert record is not None
    assert record.lifecycle_status == MemoryLifecycleStatus.QUARANTINED
    assert record.source_event_id == "event-1"
    assert record.processing_sequence == 3
    assert memory.retrieve_context("hello").db1_results == []

    reviewed = memory.review_episodic(
        episode_id,
        validation_status=ValidationStatus.VERIFIED,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )
    assert reviewed is not None
    assert [item.id for item in memory.retrieve_context("hello").db1_results] == [episode_id]


def test_same_event_and_content_is_deduplicated(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))

    first = memory.save_episodic("same", "answer", source_event_id="event-1")
    second = memory.save_episodic("same", "answer", source_event_id="event-1")
    distinct = memory.save_episodic("same", "answer", source_event_id="event-2")

    assert second == first
    assert distinct != first


def test_retrieval_keeps_semantic_and_context_scores_separate(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    same_id = memory.save_episodic(
        "shared topic same",
        "answer",
        source_event_id="same-event",
        context_id="ctx-current",
        source_channel="api.chat",
        source_session_id="session-1",
    )
    other_id = memory.save_episodic(
        "shared topic other",
        "answer",
        source_event_id="other-event",
        context_id="ctx-other",
    )

    context = memory.retrieve_context(
        "shared topic",
        current_context_id="ctx-current",
    )

    records = {record.id: record for record in context.db1_results}
    assert records[same_id].context_compatibility == 1.0
    assert records[same_id].context_relation == "same_context"
    assert records[same_id].cross_context is False
    assert records[same_id].source_channel == "api.chat"
    assert records[same_id].source_session_id == "session-1"
    assert records[other_id].context_compatibility == 0.2
    assert records[other_id].cross_context is True
    assert records[same_id].semantic_relevance >= 0.0


def test_legacy_repetitive_episode_is_quarantined_on_read(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = "legacy-repetitive"
    response = "\n".join(["あなたはこんにちは"] * 4)
    memory.db1.add(
        ids=[episode_id],
        documents=[f"User: hello\nAssistant: {response}"],
        metadatas=[
            {
                "user_input": "hello",
                "response": response,
                "loss": 0.5,
                "record_type": "episodic_log",
                "archived": False,
                "extra": "{}",
            }
        ],
    )

    record = memory.get_episodic(episode_id)

    assert record is not None
    assert record.lifecycle_status == MemoryLifecycleStatus.QUARANTINED
    assert memory.retrieve_context("hello").db1_results == []


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
