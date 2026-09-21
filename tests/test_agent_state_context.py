"""R09 U2 Context continuity and AgentState v3 contract tests."""

from datetime import UTC, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionTemporalState
from kagya.cognition import LossCalibration
from kagya.runtime import (
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshot,
    AgentStateSnapshotV3,
    AgentStateSnapshotV2,
    AppraisalStateSnapshot,
    CalibrationEntrySnapshot,
    AgentStateStore,
    ContextRegistry,
    ContextStateSnapshot,
    ContextType,
    EmotionStateSnapshot,
    EventJournal,
    StateWAL,
    StateRecoveryCoordinator,
    AgentStateSnapshotV1,
    WorkingMemory,
    WorkingMemoryItemSnapshot,
    WorkingMemorySnapshot,
)


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
MODEL_KEY = "model." + "0" * 64


class LoopStub:
    def __init__(self, *, clock=lambda: NOW) -> None:
        self.emotion_engine = EmotionEngineAllostasis(
            EmotionState(valence=0.1, arousal=0.2, optimal_loss=1.0)
        )
        self.working_memory = WorkingMemory(item_capacity=32, projection_max_bytes=2048)
        self.context_registry = ContextRegistry(clock=clock)
        self.loss_calibration = LossCalibration(
            (MODEL_KEY,),
            initial_baseline=1.0,
            initial_scale=1.0,
            minimum_scale=0.1,
        )


def make_store(path: Path) -> AgentStateStore:
    return AgentStateStore(path, baseline_surprisal=1.0, clock=lambda: NOW)


def make_v2_snapshot() -> AgentStateSnapshotV2:
    return AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=4,
        emotion_state=EmotionStateSnapshot(
            valence=0.2, arousal=0.3, optimal_loss=1.2
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )


def make_context_loop() -> LoopStub:
    loop = LoopStub()
    registry = loop.context_registry
    registry.create(
        "context-a",
        ContextType.CONVERSATION,
        "chat",
        source_session_id="session-a",
        participant_refs=("ref-b", "ref-a"),
    )
    registry.create(
        "context-b",
        ContextType.CONVERSATION,
        "chat",
        parent_context_id="context-a",
    )
    registry.relate("context-a", "context-b")
    registry.upsert_interlocutor_binding(
        "ref-a", "person-a", 0.75, ("evidence-b", "evidence-a")
    )
    registry.set_current("context-a")
    loop.loss_calibration.sample(MODEL_KEY, 0.4)
    loop.loss_calibration.sample(MODEL_KEY, 1.6)
    loop.emotion_engine.temporal_state = EmotionTemporalState(NOW)
    return loop


def as_v3(snapshot: AgentStateSnapshot) -> AgentStateSnapshotV3:
    return AgentStateSnapshotV3(
        saved_at=snapshot.saved_at,
        last_processed_event_sequence=snapshot.last_processed_event_sequence,
        emotion_state=snapshot.emotion_state,
        working_memory=snapshot.working_memory,
        context_state=snapshot.context_state,
    )


def test_v2_canonical_bytes_and_hash_remain_exact() -> None:
    snapshot = make_v2_snapshot()
    fixture = (
        b'{"emotion_state":{"arousal":0.3,"optimal_loss":1.2,"valence":0.2},'
        b'"last_processed_event_sequence":4,"saved_at":"2026-01-02T03:04:05Z",'
        b'"schema_version":2,"working_memory":{"items":[],"revision":0}}'
    )

    assert AgentStateStore._canonical_bytes(snapshot) == fixture
    assert hashlib.sha256(fixture).hexdigest() == (
        "ee9b71ed758520d00a62fb928e505047e915c585f604298bbee58a3912bcad60"
    )


def test_v2_nonempty_working_memory_canonical_bytes_and_hash_remain_exact() -> None:
    item = {
        "activation": 0.7,
        "created_revision": 2,
        "item_id": "wm-6e5ff5f0ffb5e82bc7174a955ecf100d5801157f6eff3367ad7d11b62925de0e",
        "last_activated_revision": 5,
        "retention_reason": "reactivated",
        "salience": 0.8,
        "source_id": "episode-wm",
        "source_kind": "episodic",
    }
    snapshot = AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=9,
        emotion_state=EmotionStateSnapshot(
            valence=-0.4, arousal=0.6, optimal_loss=0.8
        ),
        working_memory=WorkingMemorySnapshot(
            revision=5,
            items=(
                WorkingMemoryItemSnapshot(**item),
            ),
        ),
    )

    fixture = (
        b'{"emotion_state":{"arousal":0.6,"optimal_loss":0.8,"valence":-0.4},'
        b'"last_processed_event_sequence":9,"saved_at":"2026-01-02T03:04:05Z",'
        b'"schema_version":2,"working_memory":{"items":[{"activation":0.7,'
        b'"created_revision":2,"item_id":"wm-6e5ff5f0ffb5e82bc7174a955ecf100d5801157f6eff3367ad7d11b62925de0e",'
        b'"last_activated_revision":5,"retention_reason":"reactivated","salience":0.8,'
        b'"source_id":"episode-wm","source_kind":"episodic"}],"revision":5}}'
    )

    assert AgentStateStore._canonical_bytes(snapshot) == fixture
    assert hashlib.sha256(fixture).hexdigest() == (
        "237b970ca3326b84cef9f12ea34efb9f26fbd7e1cc0b801f1aa271311f0340af"
    )


def test_v3_canonical_bytes_and_hash_remain_exact() -> None:
    snapshot = AgentStateSnapshotV3(
        saved_at=NOW,
        last_processed_event_sequence=4,
        emotion_state=EmotionStateSnapshot(
            valence=0.2, arousal=0.3, optimal_loss=1.2
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
        context_state=ContextStateSnapshot(
            revision=0,
            current_context_id=None,
            frames=(),
            interlocutor_bindings=(),
        ),
    )
    fixture = (
        b'{"context_state":{"current_context_id":null,"frames":[],'
        b'"interlocutor_bindings":[],"revision":0},'
        b'"emotion_state":{"arousal":0.3,"optimal_loss":1.2,"valence":0.2},'
        b'"last_processed_event_sequence":4,"saved_at":"2026-01-02T03:04:05Z",'
        b'"schema_version":3,"working_memory":{"items":[],"revision":0}}'
    )

    assert AgentStateStore._canonical_bytes(snapshot) == fixture
    assert hashlib.sha256(fixture).hexdigest() == (
        "08a14b6263c2a690fa4105589aaae2da68d0387bd59fa42eeaeef4b0b3d79700"
    )


def test_capture_without_context_authority_is_a_bounded_capture_failure(
    tmp_path: Path,
) -> None:
    class MissingContextAuthority:
        def __init__(self) -> None:
            self.emotion_engine = EmotionEngineAllostasis(
                EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0)
            )
            self.working_memory = WorkingMemory(
                item_capacity=32, projection_max_bytes=2048
            )

    store = make_store(tmp_path / "agent_state.json")

    with pytest.raises(AgentStateSaveError) as error:
        store.capture(MissingContextAuthority(), sequence=1)

    assert error.value.stage is AgentStateSaveStage.CAPTURE
    assert error.value.published is False
    assert not (tmp_path / "agent_state.json").exists()


@pytest.mark.parametrize("missing", ["emotion_engine", "working_memory", "loss_calibration"])
def test_capture_requires_all_runtime_authorities(
    tmp_path: Path, missing: str
) -> None:
    loop = make_context_loop()
    delattr(loop, missing)
    store = make_store(tmp_path / "agent_state.json")

    with pytest.raises(AgentStateSaveError) as error:
        store.capture(loop, sequence=1)

    assert error.value.stage is AgentStateSaveStage.CAPTURE
    assert error.value.published is False


def test_valid_noncanonical_v2_is_not_rewritten_and_restores_empty_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    snapshot = make_v2_snapshot()
    noncanonical = json.dumps(
        snapshot.model_dump(mode="json"), indent=2, sort_keys=False
    ).encode()
    path.write_bytes(noncanonical)

    target = make_context_loop()
    loaded = store.load()
    store.restore_into(target, loaded)
    store.ensure_published(loaded)

    assert isinstance(loaded, AgentStateSnapshotV2)
    assert target.context_registry.state.revision == 0
    assert target.context_registry.state.frames == ()
    assert path.read_bytes() == noncanonical


def test_startup_keeps_noncanonical_v2_bytes_until_a_v4_capture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    snapshot = make_v2_snapshot().model_copy(
        update={"last_processed_event_sequence": 0}
    )
    noncanonical = json.dumps(
        snapshot.model_dump(mode="json"), indent=2, sort_keys=False
    ).encode()
    path.write_bytes(noncanonical)
    journal = EventJournal(
        tmp_path / "events.jsonl", 100_000, 4, clock=lambda: NOW
    )
    recovery = StateRecoveryCoordinator(store, journal, StateWAL(tmp_path / "wal"))

    result = recovery.prepare_startup()

    assert isinstance(result.snapshot, AgentStateSnapshotV2)
    assert path.read_bytes() == noncanonical


def test_startup_keeps_noncanonical_v3_bytes_until_a_v4_capture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    snapshot = as_v3(store.capture(make_context_loop(), sequence=0))
    noncanonical = json.dumps(
        snapshot.model_dump(mode="json"), indent=2, sort_keys=False
    ).encode()
    path.write_bytes(noncanonical)
    journal = EventJournal(
        tmp_path / "events.jsonl", 100_000, 4, clock=lambda: NOW
    )
    recovery = StateRecoveryCoordinator(store, journal, StateWAL(tmp_path / "wal"))

    result = recovery.prepare_startup()

    assert isinstance(result.snapshot, AgentStateSnapshotV3)
    assert path.read_bytes() == noncanonical


def test_v4_appraisal_snapshot_rejects_invalid_statistics_order_and_timestamp() -> None:
    valid = CalibrationEntrySnapshot(
        model_key=MODEL_KEY, count=2, mean=1.0, m2=0.5
    )
    with pytest.raises(ValueError):
        CalibrationEntrySnapshot(model_key=MODEL_KEY, count=1, mean=1.0, m2=0.5)
    with pytest.raises(ValueError):
        AppraisalStateSnapshot(
            calibration_entries=(valid, valid), last_emotion_update_at=None
        )
    with pytest.raises(ValueError):
        AppraisalStateSnapshot(
            calibration_entries=(valid,),
            last_emotion_update_at=datetime(
                2026, 1, 2, 12, 0, tzinfo=timezone(timedelta(hours=9))
            ),
        )


def test_v4_private_appraisal_field_is_rejected_without_rewriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    raw = store.capture(make_context_loop(), sequence=8).model_dump(mode="json")
    raw["appraisal_state"]["private_reasoning"] = "PRIVATE-APPRAISAL-SENTINEL"
    original = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(original)

    with pytest.raises(AgentStateLoadError):
        store.load()

    assert path.read_bytes() == original


def test_v4_capture_restore_round_trips_exact_context_without_clock_reads(
    tmp_path: Path,
) -> None:
    source = make_context_loop()
    store = make_store(tmp_path / "agent_state.json")
    snapshot = store.capture(source, sequence=8)
    store.save(snapshot)

    clock_calls = 0

    def fail_clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        raise AssertionError("restore must not read the Context clock")

    target = LoopStub(clock=fail_clock)
    loaded = store.load()
    store.restore_into(target, loaded)

    assert isinstance(loaded, AgentStateSnapshot)
    assert loaded.schema_version == 4
    assert loaded.context_state == snapshot.context_state
    assert target.context_registry.state == source.context_registry.state
    assert clock_calls == 0


def test_v3_rejects_non_utc_context_timestamps_without_rewriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    raw = as_v3(store.capture(make_context_loop(), sequence=8)).model_dump(
        mode="json"
    )
    raw["context_state"]["frames"][0]["started_at"] = (
        "2026-01-02T12:04:05+09:00"
    )
    original = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(original)

    with pytest.raises(AgentStateLoadError):
        store.load()

    assert path.read_bytes() == original


def test_v3_rejects_non_utc_saved_at_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    raw = as_v3(store.capture(make_context_loop(), sequence=8)).model_dump(
        mode="json"
    )
    raw["saved_at"] = "2026-01-02T12:04:05+09:00"
    original = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(original)

    with pytest.raises(AgentStateLoadError):
        store.load()

    assert path.read_bytes() == original


def test_invalid_v3_context_fails_before_mutating_target(tmp_path: Path) -> None:
    store = make_store(tmp_path / "agent_state.json")
    snapshot = as_v3(store.capture(make_context_loop(), sequence=8))
    raw = snapshot.model_dump(mode="python")
    raw["context_state"]["current_context_id"] = "missing-context"
    target = make_context_loop()
    before_context = target.context_registry.state
    before_emotion = target.emotion_engine.state
    before_working_memory = (
        target.working_memory.revision,
        target.working_memory.items,
    )

    class InvalidSnapshot:
        def model_dump(self, *, mode: str) -> object:
            del mode
            return raw

    with pytest.raises(AgentStateLoadError):
        store.restore_into(target, cast(AgentStateSnapshotV3, InvalidSnapshot()))

    assert target.context_registry.state == before_context
    assert target.emotion_engine.state == before_emotion
    assert (
        target.working_memory.revision,
        target.working_memory.items,
    ) == before_working_memory


def test_state_wal_reconstructs_v4_context_without_new_record_schema(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "agent_state.json")
    snapshot = store.capture(make_context_loop(), sequence=8)
    wal = StateWAL(tmp_path / "wal")

    wal.bootstrap(snapshot, 8)
    inspection = wal.inspect()
    reconstructed = wal.reconstruct(
        sequence=8,
        snapshot_hash=store.snapshot_hash(snapshot),
    )

    assert inspection.records[0].schema_version == 1
    assert reconstructed == snapshot
    assert isinstance(reconstructed, AgentStateSnapshot)


def test_state_wal_reconstructs_mixed_v1_v2_v3_v4_history(tmp_path: Path) -> None:
    store = make_store(tmp_path / "agent_state.json")
    v1 = AgentStateSnapshotV1(
        saved_at=NOW,
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.0, arousal=0.0, optimal_loss=1.0
        ),
    )
    v2 = make_v2_snapshot().model_copy(
        update={"last_processed_event_sequence": 1}
    )
    captured = store.capture(make_context_loop(), sequence=2)
    v3 = as_v3(captured)
    wal = StateWAL(tmp_path / "wal")
    wal.bootstrap(v1, 0)
    wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=1,
        prior_snapshot=v1,
        candidate_snapshot=v2,
    )
    wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=2,
        prior_snapshot=v2,
        candidate_snapshot=v3,
    )
    wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=3,
        prior_snapshot=v3,
        candidate_snapshot=captured.model_copy(
            update={"last_processed_event_sequence": 3}
        ),
    )

    reconstructed = wal.reconstruct(
        sequence=3,
        snapshot_hash=store.snapshot_hash(
            captured.model_copy(update={"last_processed_event_sequence": 3})
        ),
    )

    assert reconstructed.schema_version == 4
    assert isinstance(reconstructed, AgentStateSnapshot)
