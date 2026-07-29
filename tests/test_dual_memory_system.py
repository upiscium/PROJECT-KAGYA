from pathlib import Path
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import chromadb
import numpy as np
import pytest

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
from kagya.memory.transactional_chroma import ChromaTransactionStore
from kagya.external_transaction import (
    ExternalTransactionCoordinator,
    ExternalTransactionStatus,
)


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
        loss=0.25,
        emotion_valence=0.7,
        emotion_arousal=0.8,
    )

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db1_results] == [episode_id]
    assert context.db1_results[0].record_type == MemoryRecordType.EPISODIC_LOG
    assert not hasattr(context.db1_results[0], "hidden_thought")


def test_restart_scrubs_legacy_hidden_thought_from_chroma(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    episode_id = memory.save_episodic("legacy input", "visible output")
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    metadata = dict(stored["metadatas"][0])
    metadata["hidden_thought"] = "legacy private thought"
    memory.db1.update(
        ids=[episode_id],
        metadatas=[metadata],
        documents=[
            "User: legacy input\nAssistant: visible output\nThought: legacy private thought"
        ],
    )

    restarted = _memory(settings)
    scrubbed = restarted.db1.get(ids=[episode_id], include=["documents", "metadatas"])

    assert "hidden_thought" not in scrubbed["metadatas"][0]
    assert "legacy private thought" not in scrubbed["documents"][0]


def test_semantic_records_can_be_retrieved_from_db2(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("The user likes lunar gardens.")

    context = memory.retrieve_context("lunar gardens")

    assert [record.id for record in context.db2_results] == [semantic_id]
    assert context.db2_results[0].record_type == MemoryRecordType.SEMANTIC_MEMORY


def test_semantic_dedup_merges_provenance_without_growing_db2(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    first_source = memory.save_episodic("first", "source")
    second_source = memory.save_episodic("second", "source")

    first = memory.save_semantic(
        " The user likes lunar gardens. ", source_episode_ids=[first_source]
    )
    duplicate = memory.save_semantic(
        "the user likes LUNAR gardens.",
        source_episode_ids=[second_source],
        source_feedback_ids=["feedback-1"],
    )

    assert duplicate == first
    assert memory.db2.count() == 1
    record = memory.get_semantic(first)
    assert record is not None
    assert record.source_episode_ids == [first_source, second_source]
    assert record.source_feedback_ids == ["feedback-1"]
    assert record.version == 2
    assert [event["operation"] for event in record.audit_log] == [
        "create",
        "deduplicate",
    ]


def test_inactive_semantic_records_never_enter_retrieval(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    expired = memory.save_semantic(
        "expired semantic",
        expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    forgotten = memory.save_semantic("forgotten semantic")
    memory.forget_semantic(forgotten, idempotency_key="forget-once")

    assert memory.retrieve_context("semantic").db2_results == []
    expired_record = memory.get_semantic(expired)
    assert expired_record is not None
    assert expired_record.lifecycle_status.value == "expired"
    assert expired_record.audit_log[-1]["operation"] == "expire"


def test_semantic_policy_updates_are_idempotent_and_control_retrieval(
    tmp_path: Path,
) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("policy controlled semantic")
    arguments = {
        "idempotency_key": "policy-1",
        "confidence": 0.4,
        "validity": "invalid",
        "valid_from": None,
        "valid_until": None,
        "expires_at": None,
        "decay_rate": 0.1,
    }

    first = memory.update_semantic_policy(semantic_id, **arguments)
    replay = memory.update_semantic_policy(semantic_id, **arguments)

    assert first is not None and replay is not None
    assert first.version == replay.version
    assert first.confidence == 0.4
    assert first.validity == "invalid"
    assert memory.retrieve_context("policy controlled semantic").db2_results == []


def test_source_rejection_and_restore_reevaluate_derived_semantic(
    tmp_path: Path,
) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("source fact", "response")
    semantic_id = memory.save_semantic(
        "derived source fact", source_episode_ids=[episode_id]
    )

    memory.review_episodic(
        episode_id,
        validation_status=ValidationStatus.REJECTED,
        lifecycle_status=MemoryLifecycleStatus.REJECTED,
    )

    rejected = memory.get_semantic(semantic_id)
    assert rejected is not None
    assert rejected.lifecycle_status.value == "source_rejected"
    assert memory.retrieve_context("derived source fact").db2_results == []

    memory.review_episodic(
        episode_id,
        validation_status=ValidationStatus.VERIFIED,
        lifecycle_status=MemoryLifecycleStatus.ACTIVE,
    )
    restored = memory.get_semantic(semantic_id)
    assert restored is not None
    assert restored.lifecycle_status.value == "active"


def test_cold_source_archive_does_not_reject_derived_semantic(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("archived source", "response")
    semantic_id = memory.save_semantic(
        "derived from cold source", source_episode_ids=[episode_id]
    )

    memory.archive_episodic(episode_id)

    semantic = memory.get_semantic(semantic_id)
    assert semantic is not None
    assert semantic.lifecycle_status.value == "active"
    assert [item.id for item in memory.retrieve_context("cold source").db2_results] == [
        semantic_id
    ]


def test_withdrawn_feedback_source_is_reevaluated(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic(
        "feedback derived semantic", source_feedback_ids=["feedback-1"]
    )

    memory.reevaluate_semantics_for_feedback("feedback-1", rejected=True)

    rejected = memory.get_semantic(semantic_id)
    assert rejected is not None
    assert rejected.lifecycle_status.value == "source_rejected"
    assert memory.retrieve_context("feedback derived semantic").db2_results == []

    memory.reevaluate_semantics_for_feedback("feedback-1", rejected=False)
    restored = memory.get_semantic(semantic_id)
    assert restored is not None
    assert restored.lifecycle_status.value == "active"


def test_semantic_lineage_is_bidirectional_idempotent_and_audited(
    tmp_path: Path,
) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    original = memory.save_semantic("old fact")
    correction = memory.save_semantic("corrected fact")

    first = memory.propose_semantic_relationship(
        correction,
        target_id=original,
        relationship="correction",
        idempotency_key="correction-1",
    )
    replay = memory.propose_semantic_relationship(
        correction,
        target_id=original,
        relationship="correction",
        idempotency_key="correction-1",
    )

    old = memory.get_semantic(original)
    assert old is not None
    assert old.lifecycle_status.value == "corrected"
    assert old.corrected_by_id == correction
    assert first.supersedes_id == original
    assert replay.version == first.version
    assert {item.id for item in memory.semantic_graph(correction)} == {
        original,
        correction,
    }
    assert original not in {
        item.id for item in memory.retrieve_context("old fact").db2_results
    }

    with pytest.raises(ValueError, match="Idempotency key"):
        memory.propose_semantic_relationship(
            correction,
            target_id=original,
            relationship="merge",
            idempotency_key="correction-1",
        )


def test_semantic_archive_restore_and_physical_delete_are_distinct(
    tmp_path: Path,
) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    semantic_id = memory.save_semantic("cold archive fact")

    archived = memory.archive_semantic(semantic_id, idempotency_key="archive-1")
    restored = memory.restore_semantic(semantic_id, idempotency_key="restore-1")

    assert archived is not None and archived.archived is True
    assert restored is not None and restored.archived is False
    assert [
        item.id for item in memory.retrieve_context("cold archive fact").db2_results
    ] == [semantic_id]
    assert memory.delete_semantic(semantic_id, idempotency_key="delete-1") is True
    assert memory.get_semantic(semantic_id) is None


def test_physical_delete_removes_dangling_lineage_references(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    original = memory.save_semantic("delete lineage original")
    replacement = memory.save_semantic("delete lineage replacement")
    memory.propose_semantic_relationship(
        replacement,
        target_id=original,
        relationship="correction",
        idempotency_key="link-before-delete",
    )

    memory.delete_semantic(original, idempotency_key="delete-original")

    remaining = memory.get_semantic(replacement)
    assert remaining is not None
    assert remaining.supersedes_id is None
    assert remaining.audit_log[-1]["operation"] == "lineage_target_deleted"


def test_legacy_semantic_records_are_backfilled_on_startup(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    memory.db2.add(
        ids=["legacy-semantic"],
        documents=["legacy fact"],
        metadatas=[
            {
                "text": "legacy fact",
                "source_episode_ids": "[]",
                "record_type": "semantic_memory",
                "archived": False,
                "extra": "{}",
            }
        ],
    )

    migrated = _memory(settings).get_semantic("legacy-semantic")

    assert migrated is not None
    assert migrated.schema_version == 2
    assert migrated.content_hash
    assert migrated.lifecycle_status.value == "active"
    assert migrated.audit_log[0]["operation"] == "backfill"


def test_consolidation_archives_db1_records_instead_of_deleting(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic("fact", "response", emotion_arousal=1.0)

    semantic_ids = memory.consolidate_to_semantic(DummyProvider())

    assert len(semantic_ids) == 1
    stored = memory.db1.get(ids=[episode_id], include=["metadatas"])
    assert stored["ids"] == [episode_id]
    assert stored["metadatas"][0]["archived"] is True
    assert memory.retrieve_context("fact").db1_results == []
    semantic = memory.get_semantic(semantic_ids[0])
    assert semantic is not None
    assert semantic.lifecycle_status.value == "active"


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
    assert [item.id for item in memory.retrieve_context("hello").db1_results] == [
        episode_id
    ]


def test_same_event_and_content_is_deduplicated(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))

    first = memory.save_episodic("same", "answer", source_event_id="event-1")
    second = memory.save_episodic("same", "answer", source_event_id="event-1")
    distinct = memory.save_episodic("same", "answer", source_event_id="event-2")

    assert second == first
    assert distinct != first


def test_staged_episode_is_hidden_until_idempotent_finalize(tmp_path: Path) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic(
        "staged private input",
        "staged response",
        source_event_id="event-staged",
        processing_sequence=7,
        source="api.chat",
        stage_external=True,
    )

    assert memory.retrieve_context("staged private input").db1_results == []
    pending = memory.get_episodic(episode_id)
    assert pending is None

    assert memory.finalize_external_event("event-staged", 7) == 1
    assert memory.finalize_external_event("event-staged", 7) == 0

    committed = memory.get_episodic(episode_id)
    assert committed is not None
    assert committed.external_transaction_status == ExternalTransactionStatus.COMMITTED
    assert committed.external_transaction_revision == 2
    assert [
        item.id for item in memory.retrieve_context("staged private input").db1_results
    ] == [episode_id]


def test_orphan_compensation_quarantines_and_privacy_metadata_is_rejected(
    tmp_path: Path,
) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic(
        "orphan input",
        "orphan response",
        source_event_id="event-orphan",
        processing_sequence=8,
        stage_external=True,
    )

    assert memory.orphan_external_event("event-orphan", "mutation_failed") == 1
    orphaned = memory.get_episodic(episode_id)
    assert orphaned is None
    assert memory.compensate_external_event("event-orphan", "mutation_failed") == 1
    assert memory.compensate_external_event("event-orphan", "mutation_failed") == 0

    compensated = memory.get_episodic(episode_id)
    assert compensated is None
    assert memory.retrieve_context("orphan input").db1_results == []

    with pytest.raises(ValueError, match="private field"):
        memory.save_episodic("private", "response", metadata={"api_token": "secret"})


def test_reconciliation_preserves_failure_intent_across_snapshot_crash_window(
    tmp_path: Path,
) -> None:
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    episode_id = memory.save_episodic(
        "crash window",
        "response",
        source_event_id="event-crash-window",
        processing_sequence=9,
        stage_external=True,
    )
    memory.orphan_external_event("event-crash-window", "handler_failed")

    class RecoveredOutcome:
        event_id = "event-crash-window"
        lifecycle = "recovery_classified"
        processing_sequence = 9
        failure_category = "committed_before_crash"

    report = ExternalTransactionCoordinator([memory]).reconcile([RecoveredOutcome()])

    record = memory.get_episodic(episode_id)
    assert record is None
    assert report.compensated == 1
    assert memory.retrieve_context("crash window").db1_results == []


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


def test_sentence_transformer_embedding_function_encodes_with_configured_model() -> (
    None
):
    loaded: dict[str, object] = {}

    class FakeModel:
        def encode(
            self, texts: list[str], *, normalize_embeddings: bool
        ) -> list[list[float]]:
            loaded["texts"] = texts
            loaded["normalize_embeddings"] = normalize_embeddings
            return [[1.0, 0.0] for _ in texts]

    embedding = SentenceTransformerEmbeddingFunction(
        "sentence-transformers/test-model",
        model_loader=lambda model_id: (
            loaded.setdefault("model_id", model_id) and FakeModel()
        ),
    )

    vectors = embedding(["hello", "world"])

    assert loaded["model_id"] == "sentence-transformers/test-model"
    assert loaded["texts"] == ["hello", "world"]
    assert loaded["normalize_embeddings"] is True
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]


def test_sentence_transformer_embedding_revision_is_loaded_and_identified() -> None:
    loaded: dict[str, str | None] = {}

    class FakeModel:
        def encode(
            self, texts: list[str], *, normalize_embeddings: bool
        ) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    def load(model_id: str, revision: str | None) -> FakeModel:
        loaded["model_id"] = model_id
        loaded["revision"] = revision
        return FakeModel()

    embedding = SentenceTransformerEmbeddingFunction(
        "sentence-transformers/test-model", "a" * 40, model_loader=load
    )
    embedding(["hello"])

    assert loaded == {
        "model_id": "sentence-transformers/test-model",
        "revision": "a" * 40,
    }
    assert embedding.name().endswith("@" + "a" * 40)


def test_http_query_embedding_is_float32_batch_without_live_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_embeddings: list[np.ndarray] = []

    class FakeCollection:
        def __init__(self, name: str, embedding: object) -> None:
            self.name = name
            self._embedding = embedding

        def get(self, **_kwargs: object) -> dict[str, list[object]]:
            return {"ids": [], "documents": [], "metadatas": []}

        def query(
            self, *, query_texts: list[str], **_kwargs: object
        ) -> dict[str, object]:
            vector = self._embedding.embed_query(query_texts)  # type: ignore[attr-defined]
            query_embeddings.append(vector)
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        def update(self, **_kwargs: object) -> None:
            pass

    class FakeHttpClient:
        def __init__(self, **_kwargs: object) -> None:
            self.collections: dict[str, FakeCollection] = {}

        def get_or_create_collection(
            self, name: str, embedding_function: object, **_kwargs: object
        ) -> FakeCollection:
            return self.collections.setdefault(
                name, FakeCollection(name, embedding_function)
            )

    monkeypatch.setattr(chromadb, "HttpClient", FakeHttpClient)
    settings = _settings_for_tmp_memory(tmp_path)
    settings = settings.model_copy(
        update={"memory": settings.memory.model_copy(update={"backend": "http"})}
    )
    memory = DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )

    memory.retrieve_context("realistic HTTP query")

    assert len(query_embeddings) == 2
    assert all(vector.shape == (1, 16) for vector in query_embeddings)
    assert all(vector.dtype == np.float32 for vector in query_embeddings)


def test_transactional_chroma_hides_update_delete_until_commit_marker(
    tmp_path: Path,
) -> None:
    holder: list[SimpleNamespace | None] = [None]
    client = chromadb.PersistentClient(path=str(tmp_path / "transactional-chroma"))
    embedding = DeterministicEmbeddingFunction()
    raw_db1 = client.get_or_create_collection("tx_db1", embedding_function=embedding)
    raw_db2 = client.get_or_create_collection("tx_db2", embedding_function=embedding)
    store = ChromaTransactionStore(
        client,
        collection_name="tx_commands",
        node_id="node-a",
        fencing_token=lambda: 17,
        embedding_function=embedding,
        event_provider=lambda: holder[0],
    )
    db1 = store.wrap(raw_db1, "db1")
    db2 = store.wrap(raw_db2, "db2")
    db1.add(ids=["episode"], documents=["old"], metadatas=[{"state": "old"}])
    db2.add(ids=["semantic"], documents=["fact"], metadatas=[{"state": "old"}])

    holder[0] = SimpleNamespace(event_id="event-1", processing_sequence=4)
    db1.update(ids=["episode"], metadatas=[{"state": "new"}])
    db2.delete(ids=["semantic"])
    assert db1.get(ids=["episode"])["metadatas"][0]["state"] == "new"
    holder[0] = None

    assert db1.get(ids=["episode"])["metadatas"][0]["state"] == "old"
    assert db2.get(ids=["semantic"])["ids"] == ["semantic"]
    assert [record.status for record in store.records()] == [
        ExternalTransactionStatus.PENDING,
        ExternalTransactionStatus.PENDING,
    ]

    assert store.finalize_event("event-1", 4) == 2
    assert store.finalize_event("event-1", 4) == 0
    assert db1.get(ids=["episode"])["metadatas"][0]["state"] == "new"
    assert db2.get(ids=["semantic"])["ids"] == []
    assert store.materialize_event("event-1", 4) == 2
    assert raw_db1.get()["ids"] == ["episode"]
    assert raw_db2.get()["ids"] == []
    assert store._collection.count() == 2


def test_transactional_chroma_compensation_never_applies_pending_add(
    tmp_path: Path,
) -> None:
    holder: list[SimpleNamespace | None] = [
        SimpleNamespace(event_id="stale-event", processing_sequence=5)
    ]
    client = chromadb.PersistentClient(path=str(tmp_path / "compensated-chroma"))
    embedding = DeterministicEmbeddingFunction()
    raw = client.get_or_create_collection("comp_db1", embedding_function=embedding)
    store = ChromaTransactionStore(
        client,
        collection_name="comp_commands",
        node_id="stale-node",
        fencing_token=lambda: 11,
        embedding_function=embedding,
        event_provider=lambda: holder[0],
    )
    db1 = store.wrap(raw, "db1")
    db1.add(ids=["pending"], documents=["invisible"], metadatas=[{"safe": True}])
    holder[0] = None

    assert db1.get(ids=["pending"])["ids"] == []
    assert store.compensate_event("stale-event", "cas_failed") == 1
    assert store.compensate_event("stale-event", "cas_failed") == 0
    assert db1.get(ids=["pending"])["ids"] == []
    assert raw.get()["ids"] == []
    holder[0] = SimpleNamespace(event_id="stale-event", processing_sequence=5)
    with pytest.raises(RuntimeError, match="already sealed"):
        db1.add(ids=["late"], documents=["late"])


def test_transaction_reconciliation_requires_completed_journal_evidence(
    tmp_path: Path,
) -> None:
    holder: list[SimpleNamespace | None] = [None]
    client = chromadb.PersistentClient(path=str(tmp_path / "reconciled-chroma"))
    embedding = DeterministicEmbeddingFunction()
    raw = client.get_or_create_collection("reconcile_db1", embedding_function=embedding)
    store = ChromaTransactionStore(
        client,
        collection_name="reconcile_commands",
        node_id="node-a",
        fencing_token=lambda: 21,
        embedding_function=embedding,
        event_provider=lambda: holder[0],
    )
    db1 = store.wrap(raw, "db1")
    holder[0] = SimpleNamespace(event_id="completed-event", processing_sequence=8)
    db1.add(ids=["committed"], documents=["yes"], metadatas=[{"safe": True}])
    holder[0] = SimpleNamespace(event_id="no-evidence-event", processing_sequence=9)
    db1.add(ids=["discarded"], documents=["no"], metadatas=[{"safe": True}])
    holder[0] = None

    class StoreAdapter:
        def list_external_transactions(self):
            return store.records()

        def finalize_external_event(self, event_id: str, sequence: int) -> int:
            changed = store.finalize_event(event_id, sequence)
            return changed + store.materialize_event(event_id, sequence)

        def orphan_external_event(self, event_id: str, reason: str) -> int:
            return store.compensate_event(event_id, reason)

        def compensate_external_event(self, event_id: str, reason: str) -> int:
            return store.compensate_event(event_id, reason)

    outcome = SimpleNamespace(
        event_id="completed-event",
        lifecycle="completed",
        processing_sequence=8,
        failure_category=None,
    )
    report = ExternalTransactionCoordinator([StoreAdapter()]).reconcile([outcome])

    assert report.finalized == 1
    assert report.compensated == 1
    assert report.retryable == 0
    assert db1.get(ids=["committed"])["ids"] == ["committed"]
    assert db1.get(ids=["discarded"])["ids"] == []


@pytest.mark.parametrize(
    "phase",
    ["before_canonical", "after_canonical", "after_continuity", "after_cleanup"],
)
def test_transaction_materialization_recovers_every_crash_phase(
    tmp_path: Path, phase: str
) -> None:
    holder: list[SimpleNamespace | None] = [None]
    client = chromadb.PersistentClient(path=str(tmp_path / phase))
    embedding = DeterministicEmbeddingFunction()
    raw = client.get_or_create_collection("crash_db1", embedding_function=embedding)
    store = ChromaTransactionStore(
        client,
        collection_name="crash_commands",
        node_id="node-a",
        fencing_token=lambda: 31,
        embedding_function=embedding,
        event_provider=lambda: holder[0],
    )
    db1 = store.wrap(raw, "db1")
    db1.add(ids=["record"], documents=["old"], metadatas=[{"version": 0}])
    holder[0] = SimpleNamespace(event_id="crash-event", processing_sequence=1)
    db1.update(ids=["record"], metadatas=[{"version": 1}])
    holder[0] = None
    store.finalize_event("crash-event", 1)
    failed = False

    def inject(current: str, _event_id: str) -> None:
        nonlocal failed
        if current == phase and not failed:
            failed = True
            raise RuntimeError("crash")

    store.set_materialization_failure_injector(inject)
    with pytest.raises(RuntimeError, match="crash"):
        store.materialize_event("crash-event", 1)

    restarted = ChromaTransactionStore(
        client,
        collection_name="crash_commands",
        node_id="node-b",
        fencing_token=lambda: 32,
        embedding_function=embedding,
        event_provider=lambda: None,
    )
    restarted_db1 = restarted.wrap(raw, "db1")
    restarted.recover_materialization()
    result = restarted_db1.get(ids=["record"])
    assert result["metadatas"][0]["version"] == 1
    assert raw.get()["ids"] == ["record"]
    assert restarted._collection.count() == 2


def test_transaction_materialization_bounds_command_and_version_growth(
    tmp_path: Path,
) -> None:
    holder: list[SimpleNamespace | None] = [None]
    client = chromadb.PersistentClient(path=str(tmp_path / "bounded"))
    embedding = DeterministicEmbeddingFunction()
    raw = client.get_or_create_collection("bounded_db1", embedding_function=embedding)
    store = ChromaTransactionStore(
        client,
        collection_name="bounded_commands",
        node_id="node-a",
        fencing_token=lambda: 41,
        embedding_function=embedding,
        event_provider=lambda: holder[0],
    )
    db1 = store.wrap(raw, "db1")
    db1.add(ids=["record"], documents=["value"], metadatas=[{"version": 0}])
    for sequence in range(1, 71):
        event_id = f"event-{sequence}"
        holder[0] = SimpleNamespace(event_id=event_id, processing_sequence=sequence)
        db1.update(ids=["record"], metadatas=[{"version": sequence}])
        holder[0] = None
        store.finalize_event(event_id, sequence)
        store.materialize_event(event_id, sequence)

    assert raw.get()["ids"] == ["record"]
    assert db1.get(ids=["record"])["metadatas"][0]["version"] == 70
    assert store._collection.count() == 65


def test_shared_memory_continuity_accepts_pre_and_post_finalize_heads(
    tmp_path: Path,
) -> None:
    holder: list[SimpleNamespace | None] = [None]
    client = chromadb.PersistentClient(path=str(tmp_path / "continuity"))
    embedding = DeterministicEmbeddingFunction()
    raw = client.get_or_create_collection(
        "continuity_db1", embedding_function=embedding
    )
    store = ChromaTransactionStore(
        client,
        collection_name="continuity_commands",
        node_id="node-a",
        fencing_token=lambda: 51,
        embedding_function=embedding,
        event_provider=lambda: holder[0],
    )
    db1 = store.wrap(raw, "db1")
    memory = object.__new__(DualMemorySystem)
    memory._transaction_store = store
    before_hash, before_head = memory.shared_memory_state()
    holder[0] = SimpleNamespace(event_id="event", processing_sequence=1)
    db1.add(ids=["record"], documents=["value"], metadatas=[{"safe": True}])
    holder[0] = None
    store.finalize_event("event", 1)
    store.materialize_event("event", 1)
    after_hash, after_head = memory.shared_memory_state()

    memory.validate_shared_memory_state(before_hash, before_head, 1)
    memory.validate_shared_memory_state(after_hash, after_head, 1)
    with pytest.raises(RuntimeError, match="continuity"):
        memory.validate_shared_memory_state("f" * 64, before_head, 1)


def test_foreign_shared_memory_requires_operator_bootstrap_hash(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path / "foreign"))
    embedding = DeterministicEmbeddingFunction()
    raw = client.get_or_create_collection("foreign_db1", embedding_function=embedding)
    store = ChromaTransactionStore(
        client,
        collection_name="foreign_commands",
        node_id="node-a",
        fencing_token=lambda: 61,
        embedding_function=embedding,
        event_provider=lambda: None,
    )
    store.wrap(raw, "db1")
    raw.add(ids=["foreign"], documents=["existing"], metadatas=[{"safe": True}])
    memory = object.__new__(DualMemorySystem)
    memory._transaction_store = store

    with pytest.raises(RuntimeError, match="bootstrap_memory_hash"):
        memory.verify_bootstrap_memory(None)
    expected, _ = memory.shared_memory_state()
    memory.verify_bootstrap_memory(expected)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        memory.verify_bootstrap_memory("f" * 64)


def _memory(settings: Settings) -> DualMemorySystem:
    return DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )


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
