from datetime import UTC, datetime
from pathlib import Path
import json
import os

from kagya.body import EmotionState
from kagya.config import load_settings
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import (
    AgentStateSnapshot,
    AgentStateStore,
    KagyaMainLoop,
    InterlocutorModel,
    WorkingMemoryKind,
    working_memory_item,
)
from kagya.runtime.agent_state import EmotionStateSnapshot, default_agent_state_snapshot


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_snapshot_round_trip_restores_internal_state(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    loop.emotion_engine.state = EmotionState(0.2, 0.4, 0.8)
    loop.persistent_state.working_memory_metadata = {"focus": "goal-1"}
    loop.persistent_state.active_goals = [{"id": "goal-1"}]
    loop.persistent_state.commitments = [{"id": "promise-1"}]
    loop.persistent_state.values = {"care": 0.9}
    loop.persistent_state.self_model = {"certainty": 0.5}
    loop.working_memory.admit(
        working_memory_item(
            item_id="episode:one",
            kind=WorkingMemoryKind.CONVERSATION,
            reference="episode:one",
            source_event_id="event-1",
            source_event_sequence=4,
        )
    )
    frame = loop.context_registry.create(
        context_id="ctx-one",
        source_channel="api.chat",
        participant_ids=("person-1",),
    )
    loop.context_registry.suspend(frame.context_id)
    loop.context_registry.register_interlocutor(
        InterlocutorModel(identity_key="person-1")
    )
    loop.surprisal_calculator.measure("", "target", model_key="dummy:model")
    loop._persist_appraisal_state()
    store = AgentStateStore(tmp_path / "agent_state.json")

    saved = store.save(store.capture(loop, 12))
    restored_loop = _loop(tmp_path / "restored")
    loaded = AgentStateStore(store.path).load(1.0)
    store.restore_into(restored_loop, loaded)

    assert loaded == saved
    assert restored_loop.emotion_engine.state == EmotionState(0.2, 0.4, 0.8)
    assert restored_loop.persistent_state.active_goals == [{"id": "goal-1"}]
    assert restored_loop.persistent_state.commitments == [{"id": "promise-1"}]
    assert restored_loop.persistent_state.values == {"care": 0.9}
    assert restored_loop.persistent_state.self_model == {"certainty": 0.5}
    assert restored_loop.working_memory.items[0].reference == "episode:one"
    assert loaded.working_memory.items[0].content is None
    assert restored_loop.context_registry.get("ctx-one") is not None
    assert restored_loop.context_registry.get("ctx-one").status.value == "suspended"
    assert restored_loop.context_registry.interlocutors[0].identity_key == "person-1"
    assert restored_loop.surprisal_calculator.history["dummy:model"].count == 1


def test_corrupt_snapshot_uses_safe_defaults_without_reading_temp(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text("private corrupt content", encoding="utf-8")
    (tmp_path / ".agent_state.json.crash.tmp").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    snapshot = AgentStateStore(path).load(0.7)

    assert snapshot.last_processed_event_sequence == 0
    assert snapshot.emotion_state.optimal_loss == 0.7
    assert path.read_text(encoding="utf-8") == "private corrupt content"


def test_v0_snapshot_migrates_to_current_schema(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "last_event_sequence": 7,
                "emotion": {"valence": 0.1, "arousal": 0.2, "optimal_loss": 0.9},
            }
        ),
        encoding="utf-8",
    )

    snapshot = AgentStateStore(path).load(1.0)

    assert snapshot.schema_version == 3
    assert snapshot.last_processed_event_sequence == 7
    assert snapshot.emotion_state.valence == 0.1


def test_v1_snapshot_migrates_with_empty_working_memory_items(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "saved_at": datetime.now(UTC).isoformat(),
                "last_processed_event_sequence": 5,
                "emotion_state": {
                    "valence": 0.1,
                    "arousal": 0.2,
                    "optimal_loss": 0.9,
                },
                "working_memory": {"metadata": {}, "extensions": {}},
                "motivation": {"active_goals": [], "commitments": [], "extensions": {}},
                "identity": {"values": {}, "self_model": {}, "extensions": {}},
                "extensions": {},
            }
        ),
        encoding="utf-8",
    )

    snapshot = AgentStateStore(path).load(1.0)

    assert snapshot.schema_version == 3
    assert snapshot.last_processed_event_sequence == 5
    assert snapshot.working_memory.items == []


def test_future_snapshot_version_is_rejected_with_fallback(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text(json.dumps({"schema_version": 4}), encoding="utf-8")

    snapshot = AgentStateStore(path).load(0.6)

    assert snapshot.schema_version == 3
    assert snapshot.emotion_state.optimal_loss == 0.6


def test_v2_snapshot_migrates_with_empty_context_state(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    payload = default_agent_state_snapshot(1.0).model_dump(mode="json")
    payload["schema_version"] = 2
    payload.pop("context_state")
    path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = AgentStateStore(path).load(1.0)

    assert snapshot.schema_version == 3
    assert snapshot.context_state.frames == []
    assert snapshot.context_state.interlocutors == []


def test_snapshot_contains_no_session_or_generation_private_data(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    loop.session_state.add_turn("private user input", "<think>private thought</think>")
    store = AgentStateStore(tmp_path / "agent_state.json")

    store.save(store.capture(loop, 1))
    serialized = store.path.read_text(encoding="utf-8")

    assert "private user input" not in serialized
    assert "private thought" not in serialized
    assert "hidden_thought" not in serialized
    assert "prompt" not in serialized


def test_explicit_snapshot_model_validates_sequence() -> None:
    snapshot = AgentStateSnapshot(
        saved_at=datetime.now(UTC),
        last_processed_event_sequence=1,
        emotion_state=EmotionStateSnapshot(
            valence=0.0, arousal=0.0, optimal_loss=1.0
        ),
    )

    assert snapshot.last_processed_event_sequence == 1


def test_failed_atomic_replace_keeps_previous_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "agent_state.json"
    store = AgentStateStore(path)
    original = store.save(store.capture(_loop(tmp_path), 1))
    original_bytes = path.read_bytes()
    replacement = original.model_copy(
        update={"last_processed_event_sequence": 2}
    )
    monkeypatch.setattr(
        os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("interrupted")),
    )

    try:
        store.save(replacement)
    except OSError as exc:
        assert "interrupted" in str(exc)
    else:
        raise AssertionError("atomic replacement failure should be reported")

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".agent_state.json.*.tmp")) == []


def _loop(tmp_path: Path) -> KagyaMainLoop:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={"persist_directory": tmp_path / "chroma"}
            )
        }
    )
    return KagyaMainLoop(
        settings,
        DummyProvider(),
        DualMemorySystem(
            settings, embedding_function=DeterministicEmbeddingFunction()
        ),
    )
