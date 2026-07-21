from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from kagya.runtime import (
    AgentEvent,
    AgentEventType,
    StateWAL,
    StateWalIntegrityError,
    hash_snapshot,
)
from kagya.runtime.agent_state import default_agent_state_snapshot


def test_wal_reconstructs_validated_transition_and_dry_run_is_pure(
    tmp_path: Path,
) -> None:
    baseline = default_agent_state_snapshot(1.0)
    after = baseline.model_copy(
        update={
            "last_processed_event_sequence": 1,
            "emotion_state": baseline.emotion_state.model_copy(
                update={"valence": 0.25}
            ),
        }
    )
    wal = StateWAL(tmp_path / "private" / "state.jsonl")
    wal.bootstrap(baseline)
    wal.append_transition(_event(1), baseline, after)

    reconstructed = wal.reconstruct(1)
    dry_run = wal.dry_run(after, 0)

    assert reconstructed.snapshot == after
    assert reconstructed.snapshot_hash == hash_snapshot(after)
    assert dry_run.current_hash == hash_snapshot(after)
    assert dry_run.target_hash == hash_snapshot(baseline)
    assert any(change.path == "/emotion_state/valence" for change in dry_run.changes)
    assert wal.reconstruct().snapshot == after
    assert (wal.path.stat().st_mode & 0o777) == 0o600


def test_wal_rejects_corrupt_unsupported_and_private_records(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    baseline = default_agent_state_snapshot(1.0)
    wal = StateWAL(path)
    wal.bootstrap(baseline)

    private = baseline.model_copy(
        update={
            "last_processed_event_sequence": 1,
            "extensions": {"raw_prompt": "must not persist"},
        }
    )
    with pytest.raises(StateWalIntegrityError, match="private field"):
        wal.append_transition(_event(1), baseline, private)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(StateWalIntegrityError, match="invalid"):
        StateWAL(path)


def test_wal_rejects_hash_tampering_and_sequence_gaps(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    baseline = default_agent_state_snapshot(1.0)
    wal = StateWAL(path)
    wal.bootstrap(baseline)

    with pytest.raises(StateWalIntegrityError, match="contiguous"):
        wal.append_transition(
            _event(2),
            baseline,
            baseline.model_copy(update={"last_processed_event_sequence": 2}),
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"] = "tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(StateWalIntegrityError, match="hash mismatch"):
        StateWAL(path)


def _event(sequence: int) -> AgentEvent:
    now = datetime.now(UTC)
    return replace(
        AgentEvent(
            event_id=f"event-{sequence}",
            event_type=AgentEventType.EMOTION_TICK,
            source="test",
            observed_at=now,
            requested_at=now,
        ),
        processing_sequence=sequence,
    )
