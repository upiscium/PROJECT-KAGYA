from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import cast

import pytest
from pydantic import ValidationError

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionTemporalState
from kagya.cognition.surprisal_calculator import LossCalibration
from kagya.config import Settings, load_settings
from kagya.identity import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    ValueMutationEvidence,
    ValueMutationReason,
    ValueSeedDeclaration,
    ValueScope,
    ValueSelfAdmission,
    ValueState,
    ValueSystem,
)
from kagya.runtime import (
    AgentStateConfigurationDrift,
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshotV1,
    AgentStateSnapshotV2,
    AgentStateSnapshotV3,
    AgentStateSnapshotV4,
    AgentStateSnapshotV5,
    AppraisalStateSnapshot,
    AgentStateStore,
    CalibrationEntrySnapshot,
    ContextRegistry,
    ContextStateSnapshot,
    ContextType,
    EmotionStateSnapshot,
    UnsupportedAgentStateVersion,
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryItemSnapshot,
    WorkingMemorySnapshot,
    WorkingMemoryRetentionReason,
    WorkingMemorySourceKind,
    working_memory_item_id,
    ValueSystemStateSnapshot,
)
import kagya.runtime.agent_state as agent_state_module


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
MODEL_KEY = "model." + "0" * 64
UNAPPROVED_MODEL_KEY = "model." + "1" * 64
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class LoopStub:
    def __init__(
        self,
        emotion: EmotionState,
        *,
        item_capacity: int = 32,
        projection_max_bytes: int = 2048,
    ) -> None:
        self.emotion_engine = EmotionEngineAllostasis(emotion)
        self.working_memory = WorkingMemory(
            item_capacity=item_capacity,
            projection_max_bytes=projection_max_bytes,
        )
        self.context_registry = ContextRegistry(clock=lambda: NOW)
        self.loss_calibration = LossCalibration(
            (MODEL_KEY,),
            initial_baseline=1.0,
            initial_scale=1.0,
            minimum_scale=0.1,
        )
        self.value_system = ValueSystem()


def value_seed(value_id: str = "value-1", name: str = "care") -> ValueSeedDeclaration:
    return ValueSeedDeclaration(
        value_id=value_id,
        name=name,
        concept="Protect the wellbeing of the subject.",
        scope=ValueScope.SUBJECT,
        context_ids=(),
        polarity=1,
        initial_strength=0.8,
        confidence=0.9,
        stability=0.0,
        protectedness=0.0,
        negotiability=1.0,
        allowed_update_rate=0.1,
    )


class ValueLoopStub(LoopStub):
    def __init__(self, value_system: ValueSystem) -> None:
        super().__init__(EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0))
        self.value_system = value_system


def value_system_with_history(seed: ValueSeedDeclaration) -> ValueSystem:
    system = ValueSystem.from_seed_declarations((seed,))
    for index in range(33):
        event_id = f"value-event-{index}"
        origin = IdentityOrigin(
            OriginActor.SELF,
            OriginInputKind.INTERNAL_STATE,
            ValueAdmissionStatus.SELF_ENDORSED,
            event_id=event_id,
            event_sequence=index,
        )
        system.apply_update(
            ValueSelfAdmission(
                target_value_id=seed.value_id,
                subject_origin=origin,
                evidence_refs=(f"value-evidence-{index}",),
                requested_delta=1.0,
                confidence=1.0,
                reason=ValueMutationReason.ADMITTED_UPDATE,
            ),
            ValueMutationEvidence(
                event_id=event_id,
                event_sequence=index,
                recorded_at=NOW,
            ),
        )
    return system


def assert_bounded_exception(error: Exception, sentinel: str) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert sentinel not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def make_snapshot(sequence: int = 4) -> AgentStateSnapshotV2:
    return AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=0.2,
            arousal=0.3,
            optimal_loss=1.2,
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )


def make_v1_snapshot(sequence: int = 4) -> AgentStateSnapshotV1:
    return AgentStateSnapshotV1(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=0.2, arousal=0.3, optimal_loss=1.2
        ),
    )


def make_v3_snapshot(sequence: int = 4) -> AgentStateSnapshotV3:
    return AgentStateSnapshotV3(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
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


def make_wm_snapshot() -> AgentStateSnapshotV2:
    item = WorkingMemoryItemSnapshot(
        item_id=working_memory_item_id(
            WorkingMemorySourceKind.EPISODIC, "episode-wm"
        ),
        source_kind="episodic",
        source_id="episode-wm",
        activation=0.7,
        salience=0.8,
        retention_reason="reactivated",
        created_revision=2,
        last_activated_revision=5,
    )
    return AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=9,
        emotion_state=EmotionStateSnapshot(
            valence=-0.4, arousal=0.6, optimal_loss=0.8
        ),
        working_memory=WorkingMemorySnapshot(revision=5, items=(item,)),
    )


def make_store(
    path: Path,
    *,
    hook=None,
) -> AgentStateStore:
    return AgentStateStore(
        path,
        baseline_surprisal=1.0,
        clock=lambda: NOW,
        save_stage_hook=hook,
    )


def test_minimal_capture_save_load_restore_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "agent_state.json"
    store = make_store(path)
    original = LoopStub(EmotionState(valence=-0.4, arousal=0.6, optimal_loss=0.8))

    store.save(store.capture(original, sequence=7))
    restored = LoopStub(EmotionState(valence=0.0, arousal=0.0, optimal_loss=3.0))
    new_store = make_store(path)
    snapshot = new_store.load()
    new_store.restore_into(restored, snapshot)

    assert snapshot.schema_version == 5
    assert snapshot.last_processed_event_sequence == 7
    assert restored.emotion_engine.state == original.emotion_engine.state
    assert restored.working_memory.revision == 0
    assert restored.working_memory.items == ()
    assert restored.context_registry.state.revision == 0


def test_current_v4_round_trip_preserves_exact_nonempty_working_memory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    original = LoopStub(EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0))
    original.working_memory.restore_exact(
        5,
        (
            WorkingMemoryItem(
                item_id=working_memory_item_id(
                    WorkingMemorySourceKind.EPISODIC, "episode-wm"
                ),
                source_kind=WorkingMemorySourceKind.EPISODIC,
                source_id="episode-wm",
                activation=0.7,
                salience=0.8,
                retention_reason=WorkingMemoryRetentionReason.REACTIVATED,
                created_revision=2, last_activated_revision=5,
            ),
        ),
    )
    snapshot = store.capture(original, sequence=9)
    store.save(snapshot)
    restored = LoopStub(
        EmotionState(valence=1.0, arousal=1.0, optimal_loss=2.0)
    )
    loaded = make_store(path).load()
    make_store(path).restore_into(restored, loaded)

    assert loaded.schema_version == 5
    assert loaded.working_memory == snapshot.working_memory
    assert restored.working_memory.revision == 5
    assert restored.working_memory.items == original.working_memory.items


def test_v4_round_trip_preserves_calibration_and_temporal_state(
    tmp_path: Path,
) -> None:
    saved_at = datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
    store = AgentStateStore(
        tmp_path / "agent_state.json", baseline_surprisal=1.0, clock=lambda: saved_at
    )
    source = LoopStub(EmotionState(valence=0.2, arousal=0.4, optimal_loss=0.8))
    source.loss_calibration.sample(MODEL_KEY, 0.4)
    source.loss_calibration.sample(MODEL_KEY, 1.6)
    source.emotion_engine.temporal_state = EmotionTemporalState(NOW)

    snapshot = store.capture(source, sequence=12)
    store.save(snapshot)
    target = LoopStub(EmotionState(valence=-0.8, arousal=0.9, optimal_loss=2.0))
    target.loss_calibration.sample(MODEL_KEY, 0.2)
    target.emotion_engine.temporal_state = EmotionTemporalState(
        datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    loaded = store.load()
    store.restore_into(target, loaded)

    assert loaded.saved_at == saved_at
    assert loaded.appraisal_state.last_emotion_update_at == NOW
    assert loaded.appraisal_state.calibration_entries[0].model_key == MODEL_KEY
    assert loaded.appraisal_state.calibration_entries[0].count == 2
    assert target.loss_calibration.export() == source.loss_calibration.export()
    assert target.emotion_engine.temporal_state == EmotionTemporalState(NOW)


def test_v5_round_trip_preserves_complete_value_authority_without_replay(
    tmp_path: Path,
) -> None:
    seed = value_seed()
    source_system = value_system_with_history(seed)
    store = AgentStateStore(
        tmp_path / "agent_state.json",
        baseline_surprisal=1.0,
        value_seeds=(seed,),
        clock=lambda: NOW,
    )
    source = ValueLoopStub(source_system)

    snapshot = store.capture(source, sequence=33)
    assert isinstance(snapshot, AgentStateSnapshotV5)
    assert len(snapshot.value_state.values) == 1
    assert len(snapshot.value_state.histories[0].records) == 32
    assert len(snapshot.value_state.evidence_ledgers[0].evidence_refs) == 33
    store.save(snapshot)

    target_system = ValueSystem.from_seed_declarations((seed,))
    target = ValueLoopStub(target_system)
    loaded = store.load()
    store.restore_into(target, loaded)

    assert target.value_system is not source_system
    assert target.value_system.snapshot() == source_system.snapshot()
    assert target.value_system.history(seed.value_id).history_anchor_revision == 1


def test_v5_seed_drift_fails_closed_without_rewriting_snapshot(tmp_path: Path) -> None:
    seed = value_seed()
    path = tmp_path / "agent_state.json"
    store = AgentStateStore(path, 1.0, value_seeds=(seed,), clock=lambda: NOW)
    store.save(
        store.capture(ValueLoopStub(ValueSystem.from_seed_declarations((seed,))), 1)
    )
    original = path.read_bytes()
    changed = replace(seed, name="different-care")

    drifted = AgentStateStore(path, 1.0, value_seeds=(changed,), clock=lambda: NOW)
    with pytest.raises(AgentStateConfigurationDrift):
        drifted.load()
    assert path.read_bytes() == original


def test_v5_removed_and_new_config_seeds_do_not_rewrite_persisted_values(
    tmp_path: Path,
) -> None:
    seed = value_seed()
    path = tmp_path / "agent_state.json"
    original_store = AgentStateStore(path, 1.0, value_seeds=(seed,), clock=lambda: NOW)
    original_store.save(
        original_store.capture(
            ValueLoopStub(ValueSystem.from_seed_declarations((seed,))), 1
        )
    )
    added = value_seed("value-2", "honesty")
    changed_config = AgentStateStore(
        path, 1.0, value_seeds=(seed, added), clock=lambda: NOW
    )

    loaded = changed_config.load()
    assert tuple(value.value_id for value in loaded.value_state.values) == (seed.value_id,)

    removed_config = AgentStateStore(path, 1.0, value_seeds=(), clock=lambda: NOW)
    retained = removed_config.load()
    assert tuple(value.value_id for value in retained.value_state.values) == (seed.value_id,)


def test_v5_config_seed_id_collision_with_self_value_fails_closed(tmp_path: Path) -> None:
    seed = value_seed()
    event_id = "self-admission"
    origin = IdentityOrigin(
        OriginActor.SELF,
        OriginInputKind.INTERNAL_STATE,
        ValueAdmissionStatus.SELF_ENDORSED,
        event_id=event_id,
        event_sequence=0,
    )
    candidate = ValueState(
        value_id=seed.value_id,
        revision=0,
        name=seed.name,
        concept=seed.concept,
        scope=seed.scope,
        context_ids=seed.context_ids,
        polarity=seed.polarity,
        strength=seed.initial_strength,
        confidence=seed.confidence,
        stability=seed.stability,
        protectedness=seed.protectedness,
        negotiability=seed.negotiability,
        allowed_update_rate=seed.allowed_update_rate,
        frozen=False,
        origin=origin,
        evidence_refs=("self-admission-evidence",),
    )
    self_system = ValueSystem()
    self_system.admit_self_value(
        candidate,
        ValueSelfAdmission(
            target_value_id=seed.value_id,
            subject_origin=origin,
            evidence_refs=("self-admission-evidence",),
            requested_delta=0.0,
            confidence=1.0,
            reason=ValueMutationReason.SELF_ADMISSION,
        ),
        ValueMutationEvidence(event_id=event_id, event_sequence=0, recorded_at=NOW),
    )
    path = tmp_path / "agent_state.json"
    unconfigured = AgentStateStore(path, 1.0, clock=lambda: NOW)
    unconfigured.save(unconfigured.capture(ValueLoopStub(self_system), 1))

    configured = AgentStateStore(path, 1.0, value_seeds=(seed,), clock=lambda: NOW)
    with pytest.raises(AgentStateConfigurationDrift):
        configured.load()


@pytest.mark.parametrize(
    "legacy",
    [make_v1_snapshot(), make_snapshot(), make_v3_snapshot()],
)
def test_legacy_snapshot_restore_clears_appraisal_state(
    tmp_path: Path, legacy
) -> None:
    target = LoopStub(EmotionState(valence=0.8, arousal=0.7, optimal_loss=2.0))
    target.loss_calibration.sample(MODEL_KEY, 0.4)
    target.loss_calibration.sample(MODEL_KEY, 1.6)
    target.emotion_engine.temporal_state = EmotionTemporalState(NOW)

    make_store(tmp_path / "agent_state.json").restore_into(target, legacy)

    assert target.loss_calibration.export() == ()
    assert target.emotion_engine.temporal_state == EmotionTemporalState()


def test_restore_failure_rolls_back_all_five_authorities(tmp_path: Path) -> None:
    source = LoopStub(EmotionState(valence=0.2, arousal=0.4, optimal_loss=0.8))
    source.working_memory.admit(
        WorkingMemorySourceKind.EPISODIC, "source-item", 0.6, 0.7
    )
    source.context_registry.create(
        "source-context", ContextType.CONVERSATION, "chat"
    )
    source.loss_calibration.sample(MODEL_KEY, 0.4)
    source.loss_calibration.sample(MODEL_KEY, 1.6)
    source.emotion_engine.temporal_state = EmotionTemporalState(NOW)
    snapshot = make_store(tmp_path / "agent_state.json").capture(source, sequence=3)

    class FailingTemporalEmotionEngine(EmotionEngineAllostasis):
        def __init__(self, state: EmotionState) -> None:
            self._fail_temporal = False
            self._temporal_state = EmotionTemporalState()
            super().__init__(state)

        @property
        def temporal_state(self) -> EmotionTemporalState:
            return self._temporal_state

        @temporal_state.setter
        def temporal_state(self, value: EmotionTemporalState) -> None:
            if self._fail_temporal:
                raise RuntimeError("injected temporal restore failure")
            self._temporal_state = value

    target = LoopStub(EmotionState(valence=-0.8, arousal=0.9, optimal_loss=2.0))
    target.working_memory.admit(
        WorkingMemorySourceKind.EPISODIC, "target-item", 0.3, 0.4
    )
    target.context_registry.create(
        "target-context", ContextType.CONVERSATION, "chat"
    )
    target.loss_calibration.sample(MODEL_KEY, 0.2)
    before_emotion = target.emotion_engine.state
    before_working_memory = (
        target.working_memory.revision,
        target.working_memory.items,
    )
    before_context = target.context_registry.state
    before_calibration = target.loss_calibration.export()
    before_temporal = EmotionTemporalState(datetime(2026, 1, 1, tzinfo=timezone.utc))
    failing_engine = FailingTemporalEmotionEngine(before_emotion)
    target.emotion_engine = failing_engine
    failing_engine.temporal_state = before_temporal
    failing_engine._fail_temporal = True

    with pytest.raises(AgentStateLoadError):
        make_store(tmp_path / "agent_state.json").restore_into(target, snapshot)

    assert target.emotion_engine.state == before_emotion
    assert (
        target.working_memory.revision,
        target.working_memory.items,
    ) == before_working_memory
    assert target.context_registry.state == before_context
    assert target.loss_calibration.export() == before_calibration
    assert target.emotion_engine.temporal_state == before_temporal


def test_v5_restore_failure_rolls_back_value_authority_with_other_five(
    tmp_path: Path,
) -> None:
    seed = value_seed()
    store = AgentStateStore(
        tmp_path / "agent_state.json",
        1.0,
        value_seeds=(seed,),
        clock=lambda: NOW,
    )
    source = ValueLoopStub(value_system_with_history(seed))
    snapshot = store.capture(source, sequence=33)

    class FailingTemporalEmotionEngine(EmotionEngineAllostasis):
        def __init__(self, state: EmotionState) -> None:
            self._fail_temporal = False
            self._temporal_state = EmotionTemporalState()
            super().__init__(state)

        @property
        def temporal_state(self) -> EmotionTemporalState:
            return self._temporal_state

        @temporal_state.setter
        def temporal_state(self, value: EmotionTemporalState) -> None:
            if self._fail_temporal:
                raise RuntimeError("injected temporal restore failure")
            self._temporal_state = value

    target = ValueLoopStub(ValueSystem.from_seed_declarations((seed,)))
    failing_engine = FailingTemporalEmotionEngine(target.emotion_engine.state)
    target.emotion_engine = failing_engine
    before_value_system = target.value_system
    failing_engine._fail_temporal = True

    with pytest.raises(AgentStateLoadError):
        store.restore_into(target, snapshot)

    assert target.value_system is before_value_system
    assert target.value_system.snapshot() != source.value_system.snapshot()


def test_persisted_unapproved_key_restore_fails_and_rolls_back_all_authorities(
    tmp_path: Path,
) -> None:
    source = LoopStub(EmotionState(valence=0.2, arousal=0.4, optimal_loss=0.8))
    source.working_memory.admit(
        WorkingMemorySourceKind.EPISODIC, "source-persisted-item", 0.6, 0.7
    )
    source.context_registry.create(
        "source-persisted-context", ContextType.CONVERSATION, "chat"
    )
    source.emotion_engine.temporal_state = EmotionTemporalState(NOW)
    store = make_store(tmp_path / "agent_state.json")
    snapshot = store.capture(source, sequence=3).model_copy(
        update={
            "appraisal_state": AppraisalStateSnapshot(
                calibration_entries=(
                    CalibrationEntrySnapshot(
                        model_key=UNAPPROVED_MODEL_KEY,
                        count=2,
                        mean=1.5,
                        m2=0.5,
                    ),
                ),
                last_emotion_update_at=NOW,
            )
        }
    )
    store.save(snapshot)
    persisted_bytes = store.path.read_bytes()
    persisted = store.load()

    target = LoopStub(EmotionState(valence=-0.8, arousal=0.9, optimal_loss=2.0))
    target.working_memory.admit(
        WorkingMemorySourceKind.EPISODIC, "target-persisted-item", 0.3, 0.4
    )
    target.context_registry.create(
        "target-persisted-context", ContextType.CONVERSATION, "chat"
    )
    target.loss_calibration.sample(MODEL_KEY, 0.2)
    target.emotion_engine.temporal_state = EmotionTemporalState(
        datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert target.loss_calibration.approved_keys == (MODEL_KEY,)
    assert UNAPPROVED_MODEL_KEY not in target.loss_calibration.approved_keys
    before_emotion = target.emotion_engine.state
    before_working_memory = (
        target.working_memory.revision,
        target.working_memory.items,
    )
    before_context = target.context_registry.state
    before_calibration = target.loss_calibration.export()
    before_temporal = target.emotion_engine.temporal_state

    with pytest.raises(AgentStateLoadError):
        store.restore_into(target, persisted)

    assert target.emotion_engine.state == before_emotion
    assert (
        target.working_memory.revision,
        target.working_memory.items,
    ) == before_working_memory
    assert target.context_registry.state == before_context
    assert target.loss_calibration.export() == before_calibration
    assert target.emotion_engine.temporal_state == before_temporal
    assert store.path.read_bytes() == persisted_bytes


def test_capacity_decrease_restore_fails_without_trimming_or_rewriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    source = LoopStub(
        EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0),
        item_capacity=2,
    )
    source.working_memory.admit(
        WorkingMemorySourceKind.EPISODIC, "episode-one", 0.5, 0.5
    )
    source.working_memory.admit(
        WorkingMemorySourceKind.EPISODIC, "episode-two", 0.5, 0.5
    )
    snapshot = store.capture(source, sequence=2)
    store.save(snapshot)
    canonical = path.read_bytes()
    target = LoopStub(
        EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0),
        item_capacity=1,
    )

    with pytest.raises(AgentStateLoadError, match="restore failed"):
        store.restore_into(target, store.load())

    assert target.working_memory.items == ()
    assert target.working_memory.revision == 0
    assert path.read_bytes() == canonical


def test_projection_budget_change_does_not_change_agent_state_hash(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "agent_state.json")
    emotion = EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0)
    smaller = LoopStub(emotion, projection_max_bytes=8)
    larger = LoopStub(emotion, projection_max_bytes=4096)
    for loop in (smaller, larger):
        loop.working_memory.admit(
            WorkingMemorySourceKind.EPISODIC, "episode-shared", 0.5, 0.5
        )

    smaller_snapshot = store.capture(smaller, sequence=1)
    larger_snapshot = store.capture(larger, sequence=1)

    assert smaller_snapshot == larger_snapshot
    assert store.snapshot_hash(smaller_snapshot) == store.snapshot_hash(
        larger_snapshot
    )


def test_restart_with_projection_config_change_preserves_state_but_repacks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    source = LoopStub(
        EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0),
        item_capacity=2,
        projection_max_bytes=100,
    )
    for source_id in ("episode-pack-a", "episode-pack-b"):
        source.working_memory.admit(
            WorkingMemorySourceKind.EPISODIC, source_id, 0.5, 0.5
        )
    snapshot = store.capture(source, sequence=3)
    store.save(snapshot)

    target = LoopStub(
        EmotionState(valence=1.0, arousal=1.0, optimal_loss=2.0),
        item_capacity=2,
        projection_max_bytes=4,
    )
    loaded = make_store(path).load()
    make_store(path).restore_into(target, loaded)
    wide = source.working_memory.select(lambda _item: "long")
    narrow = target.working_memory.select(lambda _item: "long")
    recaptured = make_store(path).capture(target, sequence=3)

    assert target.working_memory.items == source.working_memory.items
    assert target.working_memory.revision == source.working_memory.revision
    assert recaptured == snapshot
    assert make_store(path).snapshot_hash(
        recaptured
    ) == make_store(path).snapshot_hash(snapshot)
    assert len(wide.selected) == 2
    assert len(narrow.selected) == 1
    assert narrow.projected_bytes <= 4


def test_legacy_v1_fixture_bytes_and_hash_remain_unchanged(tmp_path: Path) -> None:
    store = make_store(tmp_path / "agent_state.json")
    snapshot = make_v1_snapshot()
    fixture = (
        b'{"emotion_state":{"arousal":0.3,"optimal_loss":1.2,"valence":0.2},'
        b'"last_processed_event_sequence":4,"saved_at":"2026-01-02T03:04:05Z",'
        b'"schema_version":1}'
    )

    assert store.canonical_bytes(snapshot) == fixture
    assert hashlib.sha256(fixture).hexdigest() == (
        "d117f3f036ac792121b30ae02bb17b2653806608e2abb9d0a4dd7de0a26d7391"
    )
    assert store.snapshot_hash(snapshot) == (
        "d117f3f036ac792121b30ae02bb17b2653806608e2abb9d0a4dd7de0a26d7391"
    )


def test_legacy_v1_restore_preserves_emotion_sequence_and_file(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    legacy = make_v1_snapshot(sequence=17)
    original = store.canonical_bytes(legacy)
    path.write_bytes(original)
    loop = LoopStub(EmotionState(valence=0.9, arousal=0.9, optimal_loss=2.0))
    loop.working_memory.admit(WorkingMemorySourceKind.EPISODIC, "episode-old", 0.5, 0.5)

    loaded = store.load()
    store.restore_into(loop, loaded)

    assert isinstance(loaded, AgentStateSnapshotV1)
    assert loaded.last_processed_event_sequence == 17
    assert loop.emotion_engine.state == EmotionState(
        valence=0.2, arousal=0.3, optimal_loss=1.2
    )
    assert loop.working_memory.revision == 0
    assert loop.working_memory.items == ()
    assert path.read_bytes() == original


def test_ensure_published_preserves_valid_noncanonical_v1_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    legacy = make_v1_snapshot(sequence=17)
    noncanonical = json.dumps(
        legacy.model_dump(mode="json"), indent=2, sort_keys=False
    ).encode()
    path.write_bytes(noncanonical)

    loaded = store.load()
    store.ensure_published(loaded)

    assert loaded == legacy
    assert path.read_bytes() == noncanonical


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.pop("working_memory"),
        lambda raw: raw["working_memory"].pop("items"),
        lambda raw: raw["working_memory"]["items"].append(
            raw["working_memory"]["items"][0].copy()
        ),
        lambda raw: raw["working_memory"]["items"].__setitem__(
            0, {**raw["working_memory"]["items"][0], "source_id": "episode-other"}
        ),
        lambda raw: raw["working_memory"]["items"][0].__setitem__(
            "source_kind", "EPISODIC"
        ),
        lambda raw: raw["working_memory"]["items"][0].__setitem__(
            "retention_reason", "RECENT"
        ),
        lambda raw: raw["working_memory"].__setitem__("revision", -1),
        lambda raw: raw["working_memory"]["items"][0].__setitem__(
            "created_revision", 6
        ),
        lambda raw: raw["working_memory"]["items"][0].__setitem__(
            "activation", float("inf")
        ),
        lambda raw: raw["working_memory"].__setitem__(
            "private_reasoning", PRIVATE_SENTINEL
        ),
        lambda raw: raw["working_memory"].__setitem__("unknown", PRIVATE_SENTINEL),
    ],
)
def test_malformed_working_memory_is_rejected_without_rewriting(
    tmp_path: Path, mutate
) -> None:
    path = tmp_path / "agent_state.json"
    raw = make_wm_snapshot().model_dump(mode="json")
    mutate(raw)
    original = json.dumps(raw, allow_nan=True).encode()
    path.write_bytes(original)

    with pytest.raises(AgentStateLoadError):
        make_store(path).load()
    assert path.read_bytes() == original


def test_missing_snapshot_returns_safe_configured_default(tmp_path: Path) -> None:
    snapshot = AgentStateStore(
        tmp_path / "missing.json",
        baseline_surprisal=2.5,
        clock=lambda: NOW,
    ).load()

    assert snapshot.last_processed_event_sequence == 0
    assert snapshot.saved_at == NOW
    assert snapshot.emotion_state == EmotionStateSnapshot(
        valence=0.0,
        arousal=0.0,
        optimal_loss=2.5,
    )
    assert isinstance(snapshot, AgentStateSnapshotV5)
    assert snapshot.schema_version == 5
    assert snapshot.value_state == ValueSystemStateSnapshot(
        values=(), conflicts=(), histories=(), evidence_ledgers=()
    )
    assert snapshot.working_memory == WorkingMemorySnapshot(revision=0, items=())
    assert snapshot.context_state == ContextStateSnapshot(
        revision=0,
        current_context_id=None,
        frames=(),
        interlocutor_bindings=(),
    )


def test_agent_state_config_is_explicit_and_pre_r04_config_uses_default() -> None:
    settings = load_settings(CONFIG_PATH)
    assert settings.agent_state.path == Path(".kagya/agent_state.json")

    pre_r04 = settings.model_dump(mode="python")
    pre_r04.pop("agent_state")
    compatible = Settings.model_validate(pre_r04)
    assert compatible.agent_state.path == Path(".kagya/agent_state.json")


def test_v0_migrates_strictly_to_v4(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "last_event_sequence": 7,
                "emotion": {
                    "valence": 0.1,
                    "arousal": 0.2,
                    "optimal_loss": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )

    migrated = make_store(path).load()

    assert migrated == AgentStateSnapshotV5(
        saved_at=NOW,
        last_processed_event_sequence=7,
        emotion_state=EmotionStateSnapshot(
            valence=0.1,
            arousal=0.2,
            optimal_loss=0.9,
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
        context_state=ContextStateSnapshot(
            revision=0,
            current_context_id=None,
            frames=(),
            interlocutor_bindings=(),
        ),
        appraisal_state=AppraisalStateSnapshot(
            calibration_entries=(), last_emotion_update_at=None
        ),
        value_state=ValueSystemStateSnapshot(
            values=(), conflicts=(), histories=(), evidence_ledgers=()
        ),
    )


def test_v0_migration_rejects_unexpected_fields(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "last_event_sequence": 7,
                "emotion": {
                    "valence": 0.1,
                    "arousal": 0.2,
                    "optimal_loss": 0.9,
                    "unexpected": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentStateLoadError):
        make_store(path).load()


def test_v0_validation_detail_is_absent_from_full_exception(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "last_event_sequence": 7,
                "emotion": {
                    "valence": PRIVATE_SENTINEL,
                    "arousal": 0.2,
                    "optimal_loss": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentStateLoadError) as error:
        make_store(path).load()

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


def test_future_version_is_distinct_and_never_defaults(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    original = b'{"schema_version":999}'
    path.write_bytes(original)

    with pytest.raises(UnsupportedAgentStateVersion):
        make_store(path).load()

    assert path.read_bytes() == original


def test_snapshot_symlink_is_rejected_instead_of_followed(tmp_path: Path) -> None:
    target = tmp_path / "attacker-controlled.json"
    target.write_text(
        json.dumps(make_snapshot().model_dump(mode="json")),
        encoding="utf-8",
    )
    path = tmp_path / "agent_state.json"
    path.symlink_to(target)

    with pytest.raises(AgentStateLoadError):
        make_store(path).load()

    assert path.is_symlink()


def test_snapshot_inspection_error_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent_state.json"

    def fail_lstat(_path: Path):
        raise PermissionError(PRIVATE_SENTINEL)

    monkeypatch.setattr(Path, "lstat", fail_lstat)
    with pytest.raises(AgentStateLoadError) as error:
        make_store(path).load()

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


@pytest.mark.parametrize("content", [b"{broken", b"[]", b'{"schema_version":NaN}'])
def test_existing_corrupt_snapshot_never_defaults_or_changes(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "agent_state.json"
    path.write_bytes(content)

    with pytest.raises(AgentStateLoadError):
        make_store(path).load()

    assert path.read_bytes() == content


@pytest.mark.parametrize(
    "extra",
    [
        {"unexpected": 1},
        {"emotion_state": {"unexpected": 1}},
    ],
)
def test_current_schema_rejects_unknown_root_and_nested_fields(
    tmp_path: Path, extra: dict
) -> None:
    path = tmp_path / "agent_state.json"
    raw = make_snapshot().model_dump(mode="json")
    if "emotion_state" in extra:
        raw["emotion_state"].update(extra["emotion_state"])
    else:
        raw.update(extra)
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AgentStateLoadError):
        make_store(path).load()


def test_invalid_schema_value_is_absent_from_full_exception(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    raw = make_snapshot().model_dump(mode="json")
    raw["emotion_state"]["valence"] = PRIVATE_SENTINEL
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AgentStateLoadError) as error:
        make_store(path).load()

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


def test_unknown_field_value_is_absent_from_full_exception(tmp_path: Path) -> None:
    path = tmp_path / "agent_state.json"
    raw = make_snapshot().model_dump(mode="json")
    raw["unexpected"] = PRIVATE_SENTINEL
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(AgentStateLoadError) as error:
        make_store(path).load()

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


@pytest.mark.parametrize(
    "private_key",
    [
        "content",
        "user_input",
        "response",
        "turn",
        "turns",
        "transcript",
        "prompt",
        "raw_prompt",
        "hidden_thought",
        "private_reasoning",
        "attachment",
        "attachments",
        "event_payload",
        "debug_trace",
    ],
)
def test_normalized_private_aliases_fail_closed(
    tmp_path: Path, private_key: str
) -> None:
    path = tmp_path / "agent_state.json"
    raw = make_snapshot().model_dump(mode="json")
    raw["unknown_container"] = [{private_key: PRIVATE_SENTINEL}]
    original = json.dumps(raw).encode()
    path.write_bytes(original)

    with pytest.raises(AgentStateLoadError) as error:
        make_store(path).load()

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)
    assert path.read_bytes() == original


def test_canonical_snapshot_contains_no_private_or_independent_store_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    make_store(path).save(make_snapshot())

    raw = path.read_text(encoding="utf-8")
    assert raw == (
        '{"emotion_state":{"arousal":0.3,"optimal_loss":1.2,"valence":0.2},'
        '"last_processed_event_sequence":4,"saved_at":"2026-01-02T03:04:05Z",'
        '"schema_version":2,"working_memory":{"items":[],"revision":0}}'
    )
    assert PRIVATE_SENTINEL not in raw
    for forbidden in (
        "prompt",
        "hidden_thought",
        "turns",
        "episodic",
        "semantic",
        "adapter",
        "evaluation",
    ):
        assert forbidden not in raw.casefold()
    assert path.stat().st_mode & 0o777 == 0o600


def test_agent_state_has_no_working_memory_participant_or_journal_authority() -> None:
    assert set(AgentStateSnapshotV4.model_fields) == {
        "saved_at",
        "last_processed_event_sequence",
        "emotion_state",
        "working_memory",
        "context_state",
        "appraisal_state",
        "schema_version",
    }
    assert set(WorkingMemorySnapshot.model_fields) == {"revision", "items"}
    assert not {
        "participant",
        "store",
        "journal",
        "journaling",
    }.intersection(AgentStateSnapshotV4.model_fields)


@pytest.mark.parametrize(
    "stage",
    [
        AgentStateSaveStage.TEMP_WRITE,
        AgentStateSaveStage.TEMP_FSYNC,
        AgentStateSaveStage.ATOMIC_REPLACE,
    ],
)
def test_prepublication_failure_preserves_previous_snapshot_and_cleans_temp(
    tmp_path: Path, stage: AgentStateSaveStage
) -> None:
    path = tmp_path / "agent_state.json"
    old_bytes = b"last-known-good"
    path.write_bytes(old_bytes)

    def fail_at(current: AgentStateSaveStage) -> None:
        if current is stage:
            raise OSError(PRIVATE_SENTINEL)

    with pytest.raises(AgentStateSaveError) as error:
        make_store(path, hook=fail_at).save(make_snapshot())

    assert error.value.stage is stage
    assert error.value.published is False
    assert_bounded_exception(error.value, PRIVATE_SENTINEL)
    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".agent_state.json.*.tmp")) == []


def test_save_validation_detail_is_absent_from_full_exception(tmp_path: Path) -> None:
    raw = make_snapshot().model_dump(mode="json")
    raw["emotion_state"]["valence"] = PRIVATE_SENTINEL

    with pytest.raises(AgentStateSaveError) as error:
        make_store(tmp_path / "agent_state.json").save(cast(AgentStateSnapshotV4, raw))

    assert error.value.stage is AgentStateSaveStage.TEMP_WRITE
    assert error.value.published is False
    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


def test_parent_fsync_failure_reports_published_without_rollback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    path.write_bytes(b"last-known-good")

    def fail_parent(current: AgentStateSaveStage) -> None:
        if current is AgentStateSaveStage.PARENT_FSYNC:
            raise OSError("injected failure")

    with pytest.raises(AgentStateSaveError) as error:
        make_store(path, hook=fail_parent).save(make_snapshot(sequence=9))

    assert error.value.stage is AgentStateSaveStage.PARENT_FSYNC
    assert error.value.published is True
    assert path.read_bytes() != b"last-known-good"
    assert list(tmp_path.glob(".agent_state.json.*.tmp")) == []


def test_save_fsyncs_file_then_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent_state.json"
    stages: list[AgentStateSaveStage] = []
    fsynced_descriptors: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        fsynced_descriptors.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(agent_state_module.os, "fsync", record_fsync)
    make_store(path, hook=stages.append).save(make_snapshot())

    assert stages == [
        AgentStateSaveStage.TEMP_WRITE,
        AgentStateSaveStage.TEMP_FSYNC,
        AgentStateSaveStage.ATOMIC_REPLACE,
        AgentStateSaveStage.PARENT_FSYNC,
    ]
    assert len(fsynced_descriptors) == 2


def test_model_constraints_are_strict_finite_and_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        EmotionStateSnapshot(valence=float("nan"), arousal=0.0, optimal_loss=1.0)
    with pytest.raises(ValidationError):
        EmotionStateSnapshot(valence=0.0, arousal=float("inf"), optimal_loss=1.0)
    with pytest.raises(ValidationError):
        AgentStateSnapshotV2(
            saved_at=datetime(2026, 1, 1),
            last_processed_event_sequence=0,
            emotion_state=EmotionStateSnapshot(
                valence=0.0, arousal=0.0, optimal_loss=1.0
            ),
            working_memory=WorkingMemorySnapshot(revision=0, items=()),
        )
    with pytest.raises(ValidationError):
        AgentStateSnapshotV2(
            saved_at=NOW,
            last_processed_event_sequence=True,
            emotion_state=EmotionStateSnapshot(
                valence=0.0, arousal=0.0, optimal_loss=1.0
            ),
            working_memory=WorkingMemorySnapshot(revision=0, items=()),
        )


def test_capture_operational_failure_is_a_bounded_save_error(tmp_path: Path) -> None:
    def fail_clock() -> datetime:
        raise OSError(PRIVATE_SENTINEL)

    store = AgentStateStore(
        tmp_path / "agent_state.json",
        baseline_surprisal=1.0,
        clock=fail_clock,
    )
    loop = LoopStub(EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0))

    with pytest.raises(AgentStateSaveError) as error:
        store.capture(loop, sequence=1)

    assert error.value.stage is AgentStateSaveStage.CAPTURE
    assert error.value.published is False
    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


def test_restore_failure_is_absent_from_full_exception(tmp_path: Path) -> None:
    class BrokenSnapshot:
        def model_dump(self, *, mode: str) -> object:
            raise OSError(PRIVATE_SENTINEL)

    store = make_store(tmp_path / "agent_state.json")
    loop = LoopStub(EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0))

    with pytest.raises(AgentStateLoadError) as error:
        store.restore_into(loop, cast(AgentStateSnapshotV4, BrokenSnapshot()))

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


def test_restore_validation_detail_is_absent_from_full_exception(
    tmp_path: Path,
) -> None:
    class InvalidSnapshot:
        def model_dump(self, *, mode: str) -> object:
            raw = make_snapshot().model_dump(mode=mode)
            raw["emotion_state"]["valence"] = PRIVATE_SENTINEL
            return raw

    store = make_store(tmp_path / "agent_state.json")
    loop = LoopStub(EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0))

    with pytest.raises(AgentStateLoadError) as error:
        store.restore_into(loop, cast(AgentStateSnapshotV4, InvalidSnapshot()))

    assert_bounded_exception(error.value, PRIVATE_SENTINEL)


def test_snapshot_hash_uses_exact_canonical_bytes_including_saved_at(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "agent_state.json")
    snapshot = make_snapshot()

    assert (
        store.snapshot_hash(snapshot)
        == hashlib.sha256(store.canonical_bytes(snapshot)).hexdigest()
    )
    later = snapshot.model_copy(
        update={"saved_at": datetime(2026, 1, 2, tzinfo=timezone.utc)}
    )
    assert store.snapshot_hash(later) != store.snapshot_hash(snapshot)


def test_ensure_published_stabilizes_bootstrap_and_v0_snapshot(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing" / "agent_state.json"
    missing_store = make_store(missing_path)
    bootstrap = missing_store.load()
    missing_store.ensure_published(bootstrap)
    assert missing_path.read_bytes() == missing_store.canonical_bytes(bootstrap)

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "last_event_sequence": 7,
                "emotion": {
                    "valence": 0.1,
                    "arousal": 0.2,
                    "optimal_loss": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    legacy_store = make_store(legacy_path)
    migrated = legacy_store.load()
    legacy_store.ensure_published(migrated)
    assert legacy_path.read_bytes() == legacy_store.canonical_bytes(migrated)
    assert json.loads(legacy_path.read_text(encoding="utf-8"))["schema_version"] == 5


def test_ensure_published_does_not_rewrite_identical_canonical_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_state.json"
    store = make_store(path)
    store.save(make_snapshot())
    original_stat = path.stat()

    def fail_if_saved(_stage: AgentStateSaveStage) -> None:
        raise AssertionError("identical snapshot must not be rewritten")

    checking_store = make_store(path, hook=fail_if_saved)
    checking_store.ensure_published(checking_store.load())

    assert path.stat().st_ino == original_stat.st_ino
