from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Barrier, Thread
from typing import Any

import pytest

from kagya.external_transaction import (
    ExternalTransactionCoordinator,
    ExternalTransactionRecord,
    ExternalTransactionStatus,
)
from kagya.runtime.agent_runtime import AgentEvent, AgentEventType
from kagya.runtime.agent_state import (
    AgentStateSnapshot,
    AgentStateStore,
    default_agent_state_snapshot,
)
from kagya.runtime.event_journal import EventJournal, hash_snapshot
from kagya.runtime.operator_restore import (
    OperatorRestoreService,
    RestoreCommitRequest,
    RestoreContractError,
    RestoreErrorCode,
    logical_state_digest,
)
from kagya.runtime.state_wal import StateWAL


@dataclass
class Event:
    event_id: str
    processing_sequence: int
    event_type: str = "state_update"
    source: str = "tests.operator_restore"
    causation_id: str | None = None
    correlation_id: str | None = None
    payload: dict[str, Any] | None = None


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _snapshot(
    base: AgentStateSnapshot, sequence: int, valence: float
) -> AgentStateSnapshot:
    return base.model_copy(
        update={
            "saved_at": datetime(2026, 1, 1, 0, 0, sequence, tzinfo=UTC),
            "last_processed_event_sequence": sequence,
            "emotion_state": base.emotion_state.model_copy(update={"valence": valence}),
        }
    )


def _service(
    tmp_path: Any, *, clock: Clock | None = None, current: int = 2
) -> tuple[OperatorRestoreService, AgentStateStore, StateWAL, EventJournal]:
    base = default_agent_state_snapshot(0.2)
    snapshots = [
        _snapshot(base, 0, 0.0),
        _snapshot(base, 1, 0.2),
        _snapshot(base, 2, 0.4),
    ]
    store = AgentStateStore(tmp_path / "state.json")
    wal = StateWAL(tmp_path / "state-wal.jsonl")
    journal = EventJournal(tmp_path / "journal.jsonl")
    store.save(snapshots[current])
    wal.bootstrap(snapshots[0])
    journal.reconcile(snapshots[0])
    for sequence in range(1, len(snapshots)):
        event = Event(f"event-{sequence}", sequence)
        journal.started(event)
        wal.append_transition(event, snapshots[sequence - 1], snapshots[sequence])
        journal.completed(event, hash_snapshot(snapshots[sequence]))
    store.last_snapshot = snapshots[current]
    service = OperatorRestoreService(
        store, wal, journal, ExternalTransactionCoordinator([]), clock=clock or Clock()
    )
    return service, store, wal, journal


def _request(preview: Any, **updates: Any) -> RestoreCommitRequest:
    values = {
        "target_sequence": preview.target_sequence,
        "expected_target_hash": preview.target_snapshot_hash,
        "expected_semantic_revision": preview.semantic_revision,
        "expected_current_logical_digest": preview.current_logical_digest,
        "expected_preview_digest": preview.preview_digest,
        "expected_external_effect_digest": preview.external_effects.effect_digest,
        "confirmation_phrase": preview.confirmation_phrase,
    }
    values.update(updates)
    return RestoreCommitRequest(**values)


def test_public_models_use_expected_semantic_revision_and_forbid_bindings() -> None:
    preview_fields = RestoreCommitRequest.model_fields
    assert "expected_semantic_revision" in preview_fields
    assert "expected_current_logical_revision" not in preview_fields
    with pytest.raises(Exception):
        RestoreCommitRequest.model_validate(
            {"target_sequence": 1, "capability": "private"}
        )


def test_verified_targets_and_summary_are_public_and_private_data_free(
    tmp_path: Any,
) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    summary = service.summary()
    assert {target.target_sequence for target in summary.targets} == {0, 1, 2}
    assert all(target.eligible for target in summary.targets)
    preview = service.preview(1)
    dumped = preview.model_dump(mode="json")
    assert preview.restoreable
    assert "before" not in dumped and "after" not in dumped
    assert "hidden_thought" not in str(dumped).lower()
    assert preview.semantic_revision == summary.semantic_revision
    assert preview.external_effects.external_side_effects_replayed is False


def test_read_only_operations_do_not_make_preview_stale(tmp_path: Any) -> None:
    service, store, wal, journal = _service(tmp_path)
    preview = service.preview(1)
    current = store.last_snapshot
    assert current is not None
    read_snapshot = current.model_copy(
        update={
            "saved_at": current.saved_at + timedelta(seconds=1),
            "last_processed_event_sequence": 3,
        }
    )
    event = Event("read-3", 3, event_type="memory_read")
    journal.started(event)
    wal.append_transition(event, current, read_snapshot)
    journal.completed(event, hash_snapshot(read_snapshot))
    store.last_snapshot = read_snapshot
    assert service.summary().semantic_revision == preview.semantic_revision
    assert service.reserve(preview, "operator") == preview


def test_state_mutation_makes_reserved_preview_stale(tmp_path: Any) -> None:
    service, store, _wal, _journal = _service(tmp_path)
    preview = service.preview(1)
    changed = store.last_snapshot.model_copy(
        update={
            "emotion_state": store.last_snapshot.emotion_state.model_copy(
                update={"valence": 0.9}
            )
        }
    )
    store.last_snapshot = changed
    with pytest.raises(RestoreContractError) as exc:
        service.preflight(_request(preview), "operator")
    assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value


def test_reverted_logical_state_is_stale_when_wal_revision_changed(
    tmp_path: Any,
) -> None:
    service, store, wal, journal = _service(tmp_path)
    preview = service.preview(1)
    original = store.last_snapshot
    changed = original.model_copy(
        update={
            "last_processed_event_sequence": 3,
            "emotion_state": original.emotion_state.model_copy(update={"valence": 0.8}),
        }
    )
    reverted = original.model_copy(update={"last_processed_event_sequence": 4})
    for sequence, snapshot in ((3, changed), (4, reverted)):
        event = Event(f"revert-{sequence}", sequence)
        journal.started(event)
        wal.append_transition(event, original if sequence == 3 else changed, snapshot)
        journal.completed(event, hash_snapshot(snapshot))
    store.last_snapshot = reverted
    assert logical_state_digest(reverted) == preview.current_logical_digest
    with pytest.raises(RestoreContractError) as exc:
        service.preflight(_request(preview), "operator")
    assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value


def test_wrong_bindings_are_rejected_without_disclosure(tmp_path: Any) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    preview = service.preview(1)
    for field, value in (
        ("target_sequence", 2),
        ("expected_target_hash", "a" * 64),
        ("expected_semantic_revision", preview.semantic_revision + 1),
        ("expected_preview_digest", "b" * 64),
    ):
        with pytest.raises(RestoreContractError) as exc:
            service.preflight(_request(preview, **{field: value}), "operator")
        assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value
        assert str(exc.value) == "restore request rejected"


def test_target_retention_and_unverified_wal_targets_are_distinguished(
    tmp_path: Any,
) -> None:
    service, _store, wal, journal = _service(tmp_path)
    with pytest.raises(RestoreContractError) as exc:
        service.preview(99)
    assert exc.value.code == RestoreErrorCode.TARGET_NOT_RETAINED.value
    # A retained WAL record without terminal journal evidence is not verified.
    before = wal.reconstruct(2).snapshot
    unverified = before.model_copy(update={"last_processed_event_sequence": 3})
    wal.append_transition(Event("unverified", 3), before, unverified)
    with pytest.raises(RestoreContractError) as exc:
        service.preview(3)
    assert exc.value.code == RestoreErrorCode.TARGET_UNVERIFIED.value
    assert journal.verify()


def test_preview_expiry_and_duplicate_reservation_are_atomic(tmp_path: Any) -> None:
    clock = Clock()
    service, _store, _wal, _journal = _service(tmp_path, clock=clock)
    preview = service.preview(1)
    service.reserve(preview, "operator")
    with pytest.raises(RestoreContractError) as exc:
        service.reserve(preview, "operator")
    assert exc.value.code == RestoreErrorCode.OPERATION_IN_PROGRESS.value
    service.release(preview.preview_digest)
    clock.now += timedelta(seconds=301)
    expired = service.preview(1)
    clock.now += timedelta(seconds=301)
    with pytest.raises(RestoreContractError) as exc:
        service.reserve(expired, "operator")
    assert exc.value.code == RestoreErrorCode.PREVIEW_EXPIRED.value


def test_concurrent_reservation_allows_exactly_one_winner(tmp_path: Any) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    preview = service.preview(1)
    barrier = Barrier(2)
    outcomes: list[str] = []

    def reserve() -> None:
        barrier.wait()
        try:
            service.reserve(preview, "operator")
            outcomes.append("won")
        except RestoreContractError as exc:
            outcomes.append(exc.code)

    threads = [Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("won") == 1
    assert outcomes.count(RestoreErrorCode.OPERATION_IN_PROGRESS.value) == 1


def test_unknown_extension_is_not_restoreable(tmp_path: Any) -> None:
    service, store, _wal, _journal = _service(tmp_path)
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"future_extension": {"secret_id": "x"}}}
    )
    preview = service.preview(1)
    assert not preview.restoreable
    assert RestoreErrorCode.UNSUPPORTED_DOMAIN.value in preview.reason_codes
    with pytest.raises(RestoreContractError) as exc:
        service.reserve(preview, "operator")
    assert exc.value.code == RestoreErrorCode.UNSUPPORTED_DOMAIN.value


def test_restore_rejects_changed_state_that_could_rearm_external_work(
    tmp_path: Any,
) -> None:
    service, store, _wal, _journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    store.last_snapshot = current.model_copy(
        update={
            "extensions": {
                **current.extensions,
                "action_execution": {
                    "schema_version": 4,
                    "intents": [{"intent_id": "action-1"}],
                },
            }
        }
    )
    preview = service.preview(1)
    assert not preview.restoreable
    assert RestoreErrorCode.UNSUPPORTED_DOMAIN.value in preview.reason_codes


def test_unsequenced_pending_external_transaction_fails_closed(tmp_path: Any) -> None:
    class Store:
        def list_external_transactions(self) -> list[ExternalTransactionRecord]:
            return [
                ExternalTransactionRecord(
                    transaction_id="transaction-1",
                    revision=1,
                    artifact_type="episodic_chroma",
                    artifact_id="memory-1",
                    status=ExternalTransactionStatus.PENDING,
                    source="test",
                    created_at="2026-01-01T00:00:00Z",
                    updated_at="2026-01-01T00:00:00Z",
                    audit=[],
                )
            ]

        def finalize_external_event(
            self, event_id: str, processing_sequence: int
        ) -> int:
            return 0

        def orphan_external_event(self, event_id: str, reason: str) -> int:
            return 0

        def compensate_external_event(self, event_id: str, reason: str) -> int:
            return 0

    service, _store, _wal, _journal = _service(tmp_path)
    service.external = ExternalTransactionCoordinator([Store()])
    preview = service.preview(1)
    assert not preview.restoreable
    assert preview.external_effects.consistency_status == "inconsistent"
    assert preview.external_effects.pending_count == 1
    assert preview.external_effects.artifacts == []


def test_adapter_or_dataset_head_change_makes_preview_stale(tmp_path: Any) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    heads = {"adapter": "adapter-1", "dataset": "dataset-1"}
    service.external_heads = lambda: dict(heads)
    preview = service.preview(1)
    heads["adapter"] = "adapter-2"
    with pytest.raises(RestoreContractError) as exc:
        service.preflight(_request(preview), "operator")
    assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value


def test_restore_reads_require_current_authority(tmp_path: Any) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    service.authority = lambda _actor: False
    for operation in (service.summary, lambda: service.preview(1)):
        with pytest.raises(RestoreContractError) as exc:
            operation()
        assert exc.value.code == RestoreErrorCode.NOT_AUTHORITATIVE.value


def test_accepted_restore_without_sequence_is_recovered_as_failed(
    tmp_path: Any,
) -> None:
    service, store, wal, journal = _service(tmp_path)
    operation_id = "00000000-0000-0000-0000-000000000001"
    target = wal.reconstruct(1)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    journal.accepted(
        AgentEvent(
            event_id=f"operator-restore-{operation_id}",
            event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.operator_restore.commit",
            observed_at=now,
            requested_at=now,
            payload={
                "journal_target": f"restore:1:{target.snapshot_hash}",
            },
            correlation_id="a" * 64,
        )
    )
    current = store.last_snapshot
    assert current is not None
    journal.reconcile(current)
    operation = service.summary().latest_operation
    assert operation is not None
    assert operation.operation_id == operation_id
    assert operation.processing_sequence is None
    assert operation.state == "failed"
    assert operation.error_code == "restore_commit_failed"


def test_operation_projection_preserves_public_operation_shape(tmp_path: Any) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    preview = service.preview(1)
    operation = preview.model_dump(mode="json")
    operation = {
        "operation_id": preview.operation_id,
        "target_sequence": 1,
        "target_snapshot_hash": preview.target_snapshot_hash,
        "preview_digest": preview.preview_digest,
        "requested_at": preview.created_at,
        "started_at": None,
        "completed_at": None,
        "event_id": "restore-event",
        "processing_sequence": 3,
        "state": "finalizing",
        "error_code": None,
    }
    current = service.state_store.last_snapshot
    service.state_store.last_snapshot = current.model_copy(
        update={"extensions": {"operator_restore": [operation]}}
    )
    projected = service.summary().latest_operation
    assert projected is not None
    assert projected.state == "commit_indeterminate"
    assert "before" not in projected.model_dump()
