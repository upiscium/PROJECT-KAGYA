from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.config import Settings, load_settings
from kagya.runtime import (
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshot,
    AgentStateStore,
    EmotionStateSnapshot,
    UnsupportedAgentStateVersion,
)
import kagya.runtime.agent_state as agent_state_module


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class LoopStub:
    def __init__(self, emotion: EmotionState) -> None:
        self.emotion_engine = EmotionEngineAllostasis(emotion)


def make_snapshot(sequence: int = 4) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(
            valence=0.2,
            arousal=0.3,
            optimal_loss=1.2,
        ),
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

    assert snapshot.schema_version == 1
    assert snapshot.last_processed_event_sequence == 7
    assert restored.emotion_engine.state == original.emotion_engine.state


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


def test_agent_state_config_is_explicit_and_pre_r04_config_uses_default() -> None:
    settings = load_settings(CONFIG_PATH)
    assert settings.agent_state.path == Path(".kagya/agent_state.json")

    pre_r04 = settings.model_dump(mode="python")
    pre_r04.pop("agent_state")
    compatible = Settings.model_validate(pre_r04)
    assert compatible.agent_state.path == Path(".kagya/agent_state.json")


def test_v0_migrates_strictly_to_v1(tmp_path: Path) -> None:
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

    assert migrated == AgentStateSnapshot(
        saved_at=NOW,
        last_processed_event_sequence=7,
        emotion_state=EmotionStateSnapshot(
            valence=0.1,
            arousal=0.2,
            optimal_loss=0.9,
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
        raise PermissionError("private filesystem detail")

    monkeypatch.setattr(Path, "lstat", fail_lstat)
    with pytest.raises(AgentStateLoadError) as error:
        make_store(path).load()

    assert "private filesystem detail" not in str(error.value)


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


@pytest.mark.parametrize(
    "private_key",
    ["hiddenThought", "Hidden-Thought", "private_reasoning", "eventPayload", "turns"],
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

    assert PRIVATE_SENTINEL not in str(error.value)
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
        '"schema_version":1}'
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
            raise OSError("injected failure")

    with pytest.raises(AgentStateSaveError) as error:
        make_store(path, hook=fail_at).save(make_snapshot())

    assert error.value.stage is stage
    assert error.value.published is False
    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".agent_state.json.*.tmp")) == []


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
        AgentStateSnapshot(
            saved_at=datetime(2026, 1, 1),
            last_processed_event_sequence=0,
            emotion_state=EmotionStateSnapshot(
                valence=0.0, arousal=0.0, optimal_loss=1.0
            ),
        )
    with pytest.raises(ValidationError):
        AgentStateSnapshot(
            saved_at=NOW,
            last_processed_event_sequence=True,
            emotion_state=EmotionStateSnapshot(
                valence=0.0, arousal=0.0, optimal_loss=1.0
            ),
        )


def test_capture_operational_failure_is_a_bounded_save_error(tmp_path: Path) -> None:
    def fail_clock() -> datetime:
        raise OSError("private clock detail")

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
    assert "private clock detail" not in str(error.value)
