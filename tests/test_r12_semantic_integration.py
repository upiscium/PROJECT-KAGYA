"""R12 integration boundaries between lifecycle authority, DB2, and Sleep."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry, SleepCycleManager
from kagya.memory import DualMemorySystem
from kagya.memory.dual_memory_system import SemanticMemoryFormatError
from kagya.memory.semantic_store import SemanticStore
from kagya.models import DummyProvider
from kagya.runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    CoordinatedResult,
    TransactionBinding,
    TransactionCoordinator,
    TransactionKind,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
PRIVATE_SENTINEL = "R12-PRIVATE-HIDDEN-THOUGHT"


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "r12_integration_db1",
                    "db2_collection": "r12_integration_db2",
                }
            ),
            "sleep": settings.sleep.model_copy(
                update={"dream_dataset_path": tmp_path / "dreams" / "dataset.jsonl"}
            ),
            "qlora": settings.qlora.model_copy(
                update={"output_dir": tmp_path / "adapters", "dry_run": True}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval-results",
                    "eval_sets": [],
                }
            ),
        }
    )


def _event() -> AgentEvent:
    return AgentEvent(
        event_id="44444444-4444-4444-8444-444444444444",
        event_type=AgentEventType.SLEEP,
        source=AgentEventSource.API_SLEEP_RUN,
        requested_at=datetime(2026, 1, 1, tzinfo=UTC),
        processing_sequence=1,
    )


def test_new_projection_read_verifies_authority_without_repair(tmp_path: Path) -> None:
    memory = DualMemorySystem(_settings(tmp_path))
    event = _event()
    manager = SleepCycleManager(
        memory.settings,
        memory,
        DummyProvider(),
        AdapterRegistry(memory.settings),
    )

    memory.save_episodic("sleep input", "sleep output", emotion_arousal=0.9)

    class Runtime:
        def current_event(self) -> AgentEvent:
            return event

    manager.bind_runtime(Runtime())  # type: ignore[arg-type]
    coordinated = manager.run()
    assert isinstance(coordinated, CoordinatedResult)
    participant = coordinated.participants[0]
    transaction_id = TransactionCoordinator.derive_transaction_id(
        event, TransactionKind.EVENT_MUTATION
    )
    binding = TransactionBinding(
        transaction_id,
        event.event_id,
        event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    participant.prepare(binding)
    participant.finalize(binding)
    semantic_id = coordinated.value.materialize(transaction_id).semantic_memory_ids[0]
    stored_before = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])
    metadata = dict(stored_before["metadatas"][0])
    metadata["semantic_content_digest"] = "0" * 64
    memory.db2.update(ids=[semantic_id], metadatas=[metadata])
    before_read = memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"])

    with pytest.raises(SemanticMemoryFormatError):
        memory.get_committed_semantic(semantic_id)

    assert memory.db2.get(ids=[semantic_id], include=["documents", "metadatas"]) == before_read


def test_sleep_persists_visible_semantic_only_and_never_reruns_model(
    tmp_path: Path,
) -> None:
    class PrivateProvider(DummyProvider):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return f"<think>{PRIVATE_SENTINEL}</think>Visible semantic result."

    settings = _settings(tmp_path)
    memory = DualMemorySystem(settings)
    provider = PrivateProvider()
    memory.save_episodic("sleep input", "sleep output", emotion_arousal=0.9)
    manager = SleepCycleManager(
        settings, memory, provider, AdapterRegistry(settings)
    )
    event = _event()

    class Runtime:
        def current_event(self) -> AgentEvent:
            return event

    manager.bind_runtime(Runtime())  # type: ignore[arg-type]
    coordinated = manager.run()
    assert isinstance(coordinated, CoordinatedResult)
    participant = coordinated.participants[0]
    transaction_id = TransactionCoordinator.derive_transaction_id(
        event, TransactionKind.EVENT_MUTATION
    )
    binding = TransactionBinding(
        transaction_id,
        event.event_id,
        event.processing_sequence or 0,
        participant.participant_id,
        participant.operation_digest,
        TransactionKind.EVENT_MUTATION,
    )
    participant.prepare(binding)
    participant.finalize(binding)
    result = coordinated.value.materialize(transaction_id)
    assert len(provider.calls) == 1
    assert result.semantic_memory_ids

    semantic_store = SemanticStore.from_memory_root(settings.memory.persist_directory)
    current = semantic_store.load_current(result.semantic_memory_ids[0])
    assert current is not None
    assert current.revision.semantic_content == "Visible semantic result."
    receipt = semantic_store.receipt_path(transaction_id).read_text(encoding="ascii")
    db2 = str(memory.db2.get(ids=result.semantic_memory_ids, include=["documents", "metadatas"]))
    assert PRIVATE_SENTINEL not in receipt
    assert PRIVATE_SENTINEL not in db2
    assert PRIVATE_SENTINEL not in current.revision.semantic_content
    assert "Extract one concise semantic memory" not in receipt
