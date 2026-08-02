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
    RestoreOperation,
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
    tmp_path: Any,
    *,
    clock: Clock | None = None,
    current: int = 2,
    token_factory: Any = None,
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
        store,
        wal,
        journal,
        ExternalTransactionCoordinator([]),
        clock=clock or Clock(),
        token_factory=token_factory,
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


def test_reserved_capability_is_pinned_and_release_preserves_consumed_marker(
    tmp_path: Any,
) -> None:
    clock = Clock()
    service, _store, _wal, _journal = _service(tmp_path, clock=clock)
    preview = service.preview(1)
    service.reserve(preview, "operator")
    clock.now += timedelta(seconds=301)
    with service._lock:
        service._evict_capabilities_locked()
    assert preview.preview_digest in service._capabilities

    service._consumed.add(preview.preview_digest)
    service.release(preview.preview_digest)
    assert preview.preview_digest not in service._capabilities
    assert preview.preview_digest in service._consumed


def test_apply_claim_is_atomic_before_authoritative_work(tmp_path: Any) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    preview = service.preview(1)
    request = _request(preview)
    service.preflight(request, "operator")

    with pytest.raises(RestoreContractError) as first:
        service.apply_commit(object(), request, "operator")
    assert first.value.code == RestoreErrorCode.NOT_AUTHORITATIVE.value
    with pytest.raises(RestoreContractError) as second:
        service.apply_commit(object(), request, "operator")
    assert second.value.code == RestoreErrorCode.OPERATION_IN_PROGRESS.value


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
        "event_id": f"operator-restore-{preview.operation_id}",
        "processing_sequence": 2,
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


def test_rotated_journal_restart_uses_only_retained_suffix_for_semantic_hashes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = default_agent_state_snapshot(0.2)
    store = AgentStateStore(tmp_path / "state.json")
    wal = StateWAL(tmp_path / "state-wal.jsonl")
    journal = EventJournal(tmp_path / "journal.jsonl", max_bytes=900, retained_files=4)
    store.save(base)
    wal.bootstrap(base)
    journal.reconcile(base)
    previous = base
    for sequence in range(1, 24):
        current = _snapshot(base, sequence, sequence / 100)
        event = Event(f"rotation-{sequence}", sequence)
        journal.started(event)
        wal.append_transition(event, previous, current)
        journal.completed(event, hash_snapshot(current))
        previous = current
    store.last_snapshot = previous
    store.save(previous)

    records = EventJournal(journal.path, max_bytes=900, retained_files=4).verify()
    floor = max(wal.verify()[0].processing_sequence, records[0].snapshot_sequence or 0)
    assert floor > 0
    assert not journal.path.with_name("journal.jsonl.5").exists()
    assert len(journal.durable_files()) <= 5

    import kagya.runtime.operator_restore as restore_module

    hashed_sequences: list[int] = []
    original_digest = restore_module.logical_state_digest

    def instrumented_digest(snapshot: AgentStateSnapshot) -> str:
        hashed_sequences.append(snapshot.last_processed_event_sequence)
        return original_digest(snapshot)

    monkeypatch.setattr(restore_module, "logical_state_digest", instrumented_digest)
    restarted_store = AgentStateStore(tmp_path / "state.json")
    restarted_store.load(0.2)
    restarted = OperatorRestoreService(
        restarted_store,
        StateWAL(tmp_path / "state-wal.jsonl"),
        EventJournal(tmp_path / "journal.jsonl", max_bytes=900, retained_files=4),
        ExternalTransactionCoordinator([]),
    )
    summary = restarted.summary()
    initial_older_hashes = [
        sequence for sequence in hashed_sequences if sequence < floor
    ]
    assert initial_older_hashes and set(initial_older_hashes) == {floor - 1}
    hashed_sequences.clear()
    assert summary.retained_min_sequence == floor
    assert summary.retained_min_sequence > 0
    assert summary.retained_max_sequence == 23
    assert [target.target_sequence for target in summary.targets] == [23, 22, 21, 20]
    assert [
        target.target_sequence for target in restarted.summary(limit=1).targets
    ] == [23]
    assert {target.target_sequence for target in summary.targets} <= set(
        range(floor, 24)
    )
    with pytest.raises(RestoreContractError) as exc:
        restarted.preview(floor - 1)
    assert exc.value.code == RestoreErrorCode.TARGET_NOT_RETAINED
    assert restarted.preview(floor).target_sequence == floor
    assert all(sequence >= floor for sequence in hashed_sequences)
    bounded = OperatorRestoreService(
        restarted_store,
        StateWAL(tmp_path / "state-wal.jsonl"),
        EventJournal(tmp_path / "journal.jsonl", max_bytes=900, retained_files=4),
        ExternalTransactionCoordinator([]),
        max_targets=3,
    )
    assert [target.target_sequence for target in bounded.summary().targets] == [
        23,
        22,
        21,
    ]
    assert restarted.preview(20).target_sequence == 20
    persisted_failed = {
        "operation_id": "00000000-0000-0000-0000-000000000077",
        "target_sequence": 0,
        "target_snapshot_hash": wal.reconstruct(0).snapshot_hash,
        "preview_digest": "d" * 64,
        "requested_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:01Z",
        "completed_at": "2026-01-01T00:00:02Z",
        "event_id": "operator-restore-00000000-0000-0000-0000-000000000077",
        "processing_sequence": floor - 1,
        "state": "failed",
        "error_code": "restore_commit_failed",
        "external_side_effects_replayed": False,
    }
    restarted_store.last_snapshot = previous.model_copy(
        update={"extensions": {"operator_restore": [persisted_failed]}}
    )
    retained_operation = restarted.summary().latest_operation
    assert retained_operation is not None
    assert retained_operation.state == "failed"
    assert retained_operation.error_code == "restore_commit_failed"
    persisted_completed = {
        **persisted_failed,
        "operation_id": "00000000-0000-0000-0000-000000000078",
        "event_id": "operator-restore-00000000-0000-0000-0000-000000000078",
        "state": "completed",
        "error_code": None,
    }
    restarted_store.last_snapshot = previous.model_copy(
        update={"extensions": {"operator_restore": [persisted_completed]}}
    )
    retained_operation = restarted.summary().latest_operation
    assert retained_operation is not None
    assert retained_operation.state == "completed"


def test_read_only_rotation_keeps_retained_preview_semantically_valid(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = default_agent_state_snapshot(0.2)
    store = AgentStateStore(tmp_path / "state.json")
    wal = StateWAL(tmp_path / "state-wal.jsonl")
    journal = EventJournal(tmp_path / "journal.jsonl", max_bytes=2500, retained_files=3)
    wal.bootstrap(base)
    previous = base
    for sequence in range(1, 19):
        current = _snapshot(base, sequence, sequence / 100)
        event = Event(f"mutation-{sequence}", sequence)
        journal.started(event)
        wal.append_transition(event, previous, current)
        journal.completed(event, hash_snapshot(current))
        previous = current
    store.save(previous)
    service = OperatorRestoreService(
        store, wal, journal, ExternalTransactionCoordinator([])
    )
    preview = service.preview(18)
    initial_floor = service.summary().retained_min_sequence
    import kagya.runtime.operator_restore as restore_module

    hashed_sequences: list[int] = []
    original_digest = restore_module.logical_state_digest

    def instrumented_digest(snapshot: AgentStateSnapshot) -> str:
        hashed_sequences.append(snapshot.last_processed_event_sequence)
        return original_digest(snapshot)

    monkeypatch.setattr(restore_module, "logical_state_digest", instrumented_digest)
    advanced_floor = initial_floor
    for sequence in range(19, 29):
        read_snapshot = previous.model_copy(
            update={
                "saved_at": previous.saved_at + timedelta(seconds=sequence),
                "last_processed_event_sequence": sequence,
            }
        )
        event = Event(f"read-{sequence}", sequence, event_type="memory_read")
        journal.started(event)
        wal.append_transition(event, previous, read_snapshot)
        journal.completed(event, hash_snapshot(read_snapshot))
        previous = read_snapshot
        store.last_snapshot = read_snapshot
        advanced_floor = service.summary().retained_min_sequence
        if initial_floor < advanced_floor <= preview.target_sequence:
            break
    assert initial_floor < advanced_floor <= preview.target_sequence
    assert service.preflight(_request(preview), "operator") == preview
    assert all(sequence >= initial_floor for sequence in hashed_sequences)


def test_rotation_checkpoint_preserves_failed_sequence_as_non_target(
    tmp_path: Any,
) -> None:
    base = default_agent_state_snapshot(0.2)
    store = AgentStateStore(tmp_path / "state.json")
    wal = StateWAL(tmp_path / "state-wal.jsonl")
    journal = EventJournal(
        tmp_path / "journal.jsonl", max_bytes=100_000, retained_files=4
    )
    wal.bootstrap(base)
    first = _snapshot(base, 1, 0.1)
    event1 = Event("event-1", 1)
    journal.started(event1)
    wal.append_transition(event1, base, first)
    journal.completed(event1, hash_snapshot(first))
    failed = first.model_copy(
        update={
            "saved_at": first.saved_at + timedelta(seconds=1),
            "last_processed_event_sequence": 2,
        }
    )
    event2 = Event("event-2", 2)
    journal.started(event2)
    wal.append_transition(event2, first, failed)
    journal.failed(event2, "injected_failure", hash_snapshot(failed))
    journal.max_bytes = 1
    third = _snapshot(base, 3, 0.3)
    event3 = Event("event-3", 3)
    journal.started(event3)
    wal.append_transition(event3, failed, third)
    journal.completed(event3, hash_snapshot(third))
    store.save(third)
    summary = OperatorRestoreService(
        store, wal, journal, ExternalTransactionCoordinator([])
    ).summary()
    assert 2 not in {target.target_sequence for target in summary.targets}
    assert 3 in {target.target_sequence for target in summary.targets}


@pytest.mark.parametrize("corruption", ["gap", "hash"])
def test_rotated_suffix_gap_or_hash_mismatch_fails_closed(
    tmp_path: Any, corruption: str
) -> None:
    service, _store, _wal, journal = _service(tmp_path)
    # Put the corruption after the bootstrap floor, while retaining the normal
    # service's evidence objects so the public contract reports a safe code.
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    if corruption == "gap":
        lines.pop(2)
    else:
        import json

        record = json.loads(lines[2])
        record["source"] = "tampered-after-floor"
        lines[2] = json.dumps(record)
    journal.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("gap", RestoreErrorCode.TARGET_UNVERIFIED),
        ("hash", RestoreErrorCode.CHECKPOINT_MISMATCH),
        ("event", RestoreErrorCode.CHECKPOINT_MISMATCH),
    ],
)
def test_verified_journal_evidence_gap_or_mismatch_after_floor_fails_closed(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    expected: RestoreErrorCode,
) -> None:
    service, _store, _wal, journal = _service(tmp_path)
    records = journal.verify()
    completed = next(
        item
        for item in records
        if item.lifecycle.value == "completed" and item.snapshot_sequence == 2
    )
    if corruption == "gap":
        retained = [item for item in records if item is not completed]
    elif corruption == "hash":
        retained = [
            item.model_copy(update={"snapshot_hash": "f" * 64})
            if item is completed
            else item
            for item in records
        ]
    else:
        retained = [
            item.model_copy(update={"event_id": "event-mismatch-2"})
            if item is completed
            else item
            for item in records
        ]
    monkeypatch.setattr(journal, "verify", lambda: retained)
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == expected.value


def test_legacy_restore_event_survives_restart_without_governing_latest_operation(
    tmp_path: Any,
) -> None:
    service, store, wal, journal = _service(tmp_path)
    target = wal.reconstruct(1)
    legacy = Event(
        "legacy-state-point-in-time-restore",
        3,
        event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
        source="legacy.operator.restore",
    )
    current = store.last_snapshot
    assert current is not None
    legacy_snapshot = current.model_copy(update={"last_processed_event_sequence": 3})
    journal.started(legacy)
    wal.append_transition(legacy, current, legacy_snapshot)
    journal.completed(legacy, hash_snapshot(legacy_snapshot))
    store.save(legacy_snapshot)
    restarted_store = AgentStateStore(tmp_path / "state.json")
    restarted_store.load(0.2)
    restarted = OperatorRestoreService(
        restarted_store,
        StateWAL(tmp_path / "state-wal.jsonl"),
        EventJournal(tmp_path / "journal.jsonl"),
        ExternalTransactionCoordinator([]),
    )
    assert restarted.summary().latest_operation is None
    assert restarted.preview(1).restoreable

    governed_id = "00000000-0000-0000-0000-000000000042"
    governed = Event(
        f"operator-restore-{governed_id}",
        4,
        event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
        source="api.state.operator_restore.commit",
        correlation_id="a" * 64,
        payload={"journal_target": f"restore:1:{target.snapshot_hash}"},
    )
    next_snapshot = legacy_snapshot.model_copy(
        update={"last_processed_event_sequence": 4}
    )
    journal.started(governed)
    wal.append_transition(governed, legacy_snapshot, next_snapshot)
    journal.completed(governed, hash_snapshot(next_snapshot))
    latest = restarted.summary().latest_operation
    assert latest is not None
    assert latest.operation_id == governed_id


def test_restore_projections_omit_private_refs_and_opaque_external_artifacts(
    tmp_path: Any,
) -> None:
    service, store, _wal, journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    private_ids = {
        "goal_id": "PRIVATE_SENTINEL",
        "decision_id": "hidden_thought",
        "intent_id": "credential_token",
        "message_id": "api_secret",
        "memory_id": "password",
        "belief_id": "raw_prompt",
        "event_id": "event-prompt-1",
        "commitment_id": "commitment-AliceSmith-1",
    }
    changed = current.model_copy(
        update={
            "extensions": {
                "experiences": [private_ids],
            },
        }
    )
    store.last_snapshot = changed
    preview = service.preview(1)
    serialized = preview.model_dump_json()
    assert all(value not in serialized for value in private_ids.values())
    assert any(domain.truncated for domain in preview.domains)
    assert all(
        all(value not in ref.model_dump_json() for value in private_ids.values())
        for domain in preview.domains
        for ref in domain.refs
    )

    class Store:
        def list_external_transactions(self) -> list[ExternalTransactionRecord]:
            return [
                ExternalTransactionRecord(
                    transaction_id="credential_token",
                    revision=1,
                    artifact_type="episodic_chroma",
                    artifact_id="PRIVATE_SENTINEL",
                    status=ExternalTransactionStatus.COMMITTED,
                    event_id="hidden_thought",
                    processing_sequence=2,
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

    service.external = ExternalTransactionCoordinator([Store()])
    external_preview = service.preview(1)
    external_text = external_preview.model_dump_json()
    assert "PRIVATE_SENTINEL" not in external_text
    assert "credential_token" not in external_text
    assert "hidden_thought" not in external_text
    artifact_handle = external_preview.external_effects.artifacts[0].refs[0]
    assert len(artifact_handle) == 64
    assert (
        artifact_handle != __import__("hashlib").sha256(b"PRIVATE_SENTINEL").hexdigest()
    )

    governed = Event(
        "operator-restore-00000000-0000-0000-0000-000000000099",
        3,
        event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
        source="api.state.operator_restore.commit",
        correlation_id="b" * 64,
        payload={"journal_target": f"restore:1:{hash_snapshot(current)}"},
    )
    journal.started(governed)
    record = journal.verify()[-1]
    assert record.target == f"restore:1:{hash_snapshot(current)}"
    assert record.correlation_id == "b" * 64
    journal_metadata = f"{record.target} {record.correlation_id}"
    assert all(value not in journal_metadata for value in private_ids.values())


def test_target_summary_cap_does_not_hide_retained_preview_target(
    tmp_path: Any,
) -> None:
    service, _store, _wal, _journal = _service(tmp_path)
    # The UI list is intentionally smaller than the retained evidence set.
    bounded = OperatorRestoreService(
        service.state_store,
        service.wal,
        service.journal,
        service.external,
        max_targets=1,
    )
    assert [item.target_sequence for item in bounded.summary().targets] == [2]
    assert bounded.preview(1).target_sequence == 1


def test_summary_uses_first_verified_target_after_non_target_transition(
    tmp_path: Any,
) -> None:
    service, store, wal, journal = _service(tmp_path)
    prior = store.last_snapshot
    assert prior is not None
    failed = _snapshot(prior, 3, 0.6)
    failed_event = Event("failed-3", 3)
    journal.started(failed_event)
    wal.append_transition(failed_event, prior, failed)
    journal.failed(failed_event, "injected_failure", hash_snapshot(failed))
    verified = _snapshot(failed, 4, 0.6)
    verified_event = Event("event-4", 4)
    journal.started(verified_event)
    wal.append_transition(verified_event, failed, verified)
    journal.completed(verified_event, hash_snapshot(verified))
    latest = _snapshot(verified, 5, 0.8)
    latest_event = Event("event-5", 5)
    journal.started(latest_event)
    wal.append_transition(latest_event, verified, latest)
    journal.completed(latest_event, hash_snapshot(latest))
    store.last_snapshot = latest

    targets = {item.target_sequence for item in service.summary().targets}
    assert 3 not in targets
    assert {4, 5} <= targets


def test_latest_operation_uses_authoritative_order_across_sources(
    tmp_path: Any,
) -> None:
    service, store, _wal, _journal = _service(tmp_path)
    target_hash = service.preview(1).target_snapshot_hash

    def operation(
        identifier: str, sequence: int | None, requested: str
    ) -> RestoreOperation:
        return RestoreOperation(
            operation_id=identifier,
            target_sequence=1,
            target_snapshot_hash=target_hash,
            preview_digest="b" * 64,
            requested_at=requested,
            started_at=requested,
            completed_at=requested,
            event_id=f"operator-restore-{identifier}",
            processing_sequence=sequence,
            state="completed",
            error_code=None,
        )

    in_memory = operation(
        "00000000-0000-0000-0000-000000000010", 1, "2026-01-01T00:00:04+00:00"
    )
    journal = operation(
        "00000000-0000-0000-0000-000000000011", 2, "2026-01-01T00:00:01+00:00"
    )
    persisted = operation(
        "00000000-0000-0000-0000-000000000012", 2, "2026-01-01T00:00:01+00:00"
    )
    service._operations = [in_memory]
    service._journal_operations = lambda _records: [journal]
    current = store.last_snapshot
    assert current is not None
    target_hash = service.preview(1).target_snapshot_hash
    store.last_snapshot = current.model_copy(
        update={"extensions": {"operator_restore": [persisted.model_dump(mode="json")]}}
    )
    latest = service.summary().latest_operation
    assert latest is not None
    assert latest.operation_id == persisted.operation_id


def test_newer_unsequenced_journal_operation_wins_with_clock_rollback(
    tmp_path: Any,
) -> None:
    service, store, _wal, journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    target_hash = service.preview(1).target_snapshot_hash
    older_id = "00000000-0000-0000-0000-000000000012"
    older = RestoreOperation(
        operation_id=older_id,
        target_sequence=1,
        target_snapshot_hash=target_hash,
        preview_digest="b" * 64,
        requested_at="2027-01-01T00:00:00+00:00",
        started_at="2027-01-01T00:00:00+00:00",
        completed_at="2027-01-01T00:00:00+00:00",
        event_id=f"operator-restore-{older_id}",
        processing_sequence=2,
        state="completed",
        error_code=None,
    )
    store.last_snapshot = current.model_copy(
        update={"extensions": {"operator_restore": [older.model_dump(mode="json")]}}
    )
    newer_id = "00000000-0000-0000-0000-000000000013"
    journal.accepted(
        Event(
            f"operator-restore-{newer_id}",
            None,  # type: ignore[arg-type]
            event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
            source="api.state.operator_restore.commit",
            correlation_id="c" * 64,
            payload={"journal_target": f"restore:1:{target_hash}"},
        )
    )
    latest = service.summary().latest_operation
    assert latest is not None
    assert latest.operation_id == newer_id
    assert latest.state == "commit_indeterminate"


def test_operation_chronology_uses_acceptance_not_later_lifecycle(
    tmp_path: Any,
) -> None:
    service, store, _wal, journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    target_hash = service.preview(1).target_snapshot_hash

    def restore_event(identifier: str) -> Event:
        return Event(
            f"operator-restore-{identifier}",
            None,  # type: ignore[arg-type]
            event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
            source="api.state.operator_restore.commit",
            correlation_id="d" * 64,
            payload={"journal_target": f"restore:1:{target_hash}"},
        )

    older = restore_event("00000000-0000-0000-0000-000000000014")
    newer = restore_event("00000000-0000-0000-0000-000000000015")
    journal.accepted(older)
    journal.accepted(newer)
    older.processing_sequence = 3
    journal.started(older)

    latest = service.summary().latest_operation
    assert latest is not None
    assert latest.operation_id == "00000000-0000-0000-0000-000000000015"


def test_external_preview_tokens_are_opaque_per_preview_and_cross_preview_tokens_fail(
    tmp_path: Any,
) -> None:
    service, _store, _wal, _journal = _service(tmp_path)

    class Store:
        def list_external_transactions(self) -> list[ExternalTransactionRecord]:
            return [
                ExternalTransactionRecord(
                    transaction_id="transaction-opaque",
                    revision=1,
                    artifact_type="episodic_chroma",
                    artifact_id="memory-opaque-1",
                    status=ExternalTransactionStatus.COMMITTED,
                    event_id="event-opaque-1",
                    processing_sequence=2,
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

    service.external = ExternalTransactionCoordinator([Store()])
    first = service.preview(1)
    second = service.preview(1)
    assert first.external_effects.effect_digest != second.external_effects.effect_digest
    assert first.external_effects.artifacts != second.external_effects.artifacts
    with pytest.raises(RestoreContractError) as exc:
        service.preflight(
            _request(
                first,
                expected_external_effect_digest=second.external_effects.effect_digest,
            ),
            "operator",
        )
    assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value


def _restore_action_state(*, active: bool) -> dict[str, Any]:
    from kagya.actions.execution import (
        ActionBudget,
        ActionIntent,
        ActionPreview,
        ActionProvenance,
        ActionState,
        ActionValidationRecord,
        ExecutionReceipt,
        IntentStatus,
        Observation,
        OutcomeVerification,
        PolicyEvaluation,
        ReceiptStatus,
        RiskClass,
        _digest,
        _TOOLS,
        _validation_schema_revision,
    )

    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    budget = ActionBudget()
    arguments = {"namespace": "project", "key": "name"}
    digest = _digest(arguments)
    status = IntentStatus.RETRY_PENDING if active else IntentStatus.SUCCEEDED
    receipt_status = ReceiptStatus.FAILED if active else ReceiptStatus.SUCCEEDED
    intent = ActionIntent(
        intent_id="action-1",
        revision=1,
        idempotency_key="action-1",
        tool_name="restricted_metadata_read",
        arguments=arguments,
        risk_class=RiskClass.READ_ONLY,
        status=status,
        dry_run=False,
        policy=PolicyEvaluation(
            evaluation_id="policy-1",
            tool_name="restricted_metadata_read",
            risk_class=RiskClass.READ_ONLY,
            allowed=True,
            approval_required=False,
            reasons=("allowlisted_tool",),
            argument_digest=digest,
            evaluated_at=timestamp,
        ),
        preview=ActionPreview(
            tool_name="restricted_metadata_read",
            risk_class=RiskClass.READ_ONLY,
            arguments=arguments,
            effect=_TOOLS["restricted_metadata_read"].effect,
            bounded_by=budget,
            compensation_available=False,
        ),
        provenance=ActionProvenance(
            decision_id="decision-1", candidate_id="candidate-1"
        ),
        budget=budget,
        attempts=1,
        cost_units_used=0,
        created_at=timestamp,
        updated_at=timestamp,
        deadline_at=timestamp + timedelta(minutes=1),
        receipt_id="receipt-1",
        validation_record_id="validation-1",
    )
    validation = ActionValidationRecord(
        validation_id="validation-1",
        idempotency_key="action-1",
        request_digest="b" * 64,
        decision_id="decision-1",
        intent_id="action-1",
        tool_name="restricted_metadata_read",
        risk_class=RiskClass.READ_ONLY,
        arguments_valid=True,
        validation_schema_revision=_validation_schema_revision(
            "restricted_metadata_read", _TOOLS["restricted_metadata_read"]
        ),
        validation_error_codes=(),
        validated_event_id="event-1",
        validated_event_sequence=1,
        canonical_arguments_digest=digest,
        validated_at=timestamp,
    )
    receipt = ExecutionReceipt(
        receipt_id="receipt-1",
        intent_id="action-1",
        idempotency_key="action-1",
        attempt=1,
        status=receipt_status,
        started_at=timestamp,
        finished_at=timestamp,
        duration_ms=0,
        observation_id="observation-1",
        verification_id="verification-1",
        decision_id="decision-1",
    )
    observation = Observation(
        observation_id="observation-1",
        intent_id="action-1",
        receipt_id="receipt-1",
        observed_at=timestamp,
        data={"namespace": "project", "key": "name", "value": "PROJECT-KAGYA"},
        result_digest="c" * 64,
        valid=True,
    )
    verification = OutcomeVerification(
        verification_id="verification-1",
        intent_id="action-1",
        observation_id="observation-1",
        success=not active,
        reason="verified" if not active else "retryable_failure",
        verified_at=timestamp,
    )
    return ActionState(
        intents=(intent,),
        validation_records=(validation,),
        receipts=(receipt,),
        observations=(observation,),
        verifications=(verification,),
    ).model_dump(mode="json")


def _protected_state(domain: str, *, active: bool) -> dict[str, Any]:
    if domain == "action_execution":
        return _restore_action_state(active=active)
    if domain == "proactive_outbox":
        from kagya.outbox import DeliveryStatus, OutboxMessage, OutboxState

        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        message = OutboxMessage(
            message_id="message-1",
            kind="action_result",
            title="Result",
            body="Done",
            not_before=timestamp,
            deduplication_key="message-1",
            created_at=timestamp,
            updated_at=timestamp,
            delivery_status=(
                DeliveryStatus.PENDING if active else DeliveryStatus.DELIVERED
            ),
            delivered_at=None if active else timestamp,
        )
        return OutboxState(messages=(message,)).model_dump(mode="json")
    from kagya.runtime.autonomy import ScheduleStatus, WakeUpSchedule

    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    schedule = WakeUpSchedule(
        schedule_id="schedule-1",
        kind="operator",
        wake_at=timestamp,
        status=ScheduleStatus.PENDING if active else ScheduleStatus.COMPLETED,
        created_at=timestamp,
        completed_at=None if active else timestamp,
    )
    return {"schema_version": 1, "schedules": [schedule.model_dump(mode="json")]}


@pytest.mark.parametrize(
    "domain", ["action_execution", "proactive_outbox", "subject_scheduler"]
)
def test_terminal_protected_histories_restore_but_pending_or_invalid_fail_closed(
    tmp_path: Any, domain: str
) -> None:
    service, store, _wal, _journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    value = _protected_state(domain, active=False)
    store.last_snapshot = current.model_copy(update={"extensions": {domain: value}})
    assert service.preview(1).restoreable

    pending = _protected_state(domain, active=True)
    store.last_snapshot = current.model_copy(update={"extensions": {domain: pending}})
    assert not service.preview(1).restoreable

    malformed = dict(value)
    malformed.pop("schema_version")
    store.last_snapshot = current.model_copy(update={"extensions": {domain: malformed}})
    assert not service.preview(1).restoreable

    if domain == "action_execution":
        from kagya.actions.execution import ActionState, ApprovalRecord

        invalid_reference = ActionState(
            approvals=(
                ApprovalRecord(
                    approval_id="approval-1",
                    intent_id="action-missing",
                    status="rejected",
                    requested_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            )
        ).model_dump(mode="json")
        store.last_snapshot = current.model_copy(
            update={"extensions": {domain: invalid_reference}}
        )
        assert not service.preview(1).restoreable


def test_pending_approval_blocks_terminal_action_restore(tmp_path: Any) -> None:
    from kagya.actions.execution import (
        ActionBudget,
        ActionIntent,
        ActionPreview,
        ActionProvenance,
        ActionState,
        ActionValidationRecord,
        ApprovalRecord,
        IntentStatus,
        NotificationArguments,
        PolicyEvaluation,
        RiskClass,
        _digest,
        _TOOLS,
        _validation_schema_revision,
        validate_action_state_semantics,
    )

    service, store, _wal, _journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    arguments = NotificationArguments(
        title="Reminder",
        body="Review completed work",
        public_preview={
            "kind": "local_notification_enqueue",
            "title": "Reminder",
            "body": "Review completed work",
        },
    ).model_dump(mode="json")
    digest = _digest(arguments)
    budget = ActionBudget()
    intent = ActionIntent(
        intent_id="action-approval-1",
        revision=1,
        idempotency_key="action-approval-1",
        tool_name="local_notification_enqueue",
        arguments=arguments,
        risk_class=RiskClass.REVERSIBLE_WRITE,
        status=IntentStatus.AWAITING_APPROVAL,
        dry_run=False,
        policy=PolicyEvaluation(
            evaluation_id="policy-approval-1",
            tool_name="local_notification_enqueue",
            risk_class=RiskClass.REVERSIBLE_WRITE,
            allowed=True,
            approval_required=True,
            reasons=("human_approval_required",),
            argument_digest=digest,
            evaluated_at=timestamp,
        ),
        preview=ActionPreview(
            tool_name="local_notification_enqueue",
            risk_class=RiskClass.REVERSIBLE_WRITE,
            arguments=arguments,
            effect=_TOOLS["local_notification_enqueue"].effect,
            bounded_by=budget,
            compensation_available=True,
        ),
        provenance=ActionProvenance(
            decision_id="decision-approval-1", candidate_id="candidate-approval-1"
        ),
        budget=budget,
        attempts=0,
        cost_units_used=0,
        created_at=timestamp,
        updated_at=timestamp,
        deadline_at=timestamp + timedelta(minutes=1),
        approval_id="approval-1",
        validation_record_id="validation-approval-1",
    )
    state = ActionState(
        intents=(intent,),
        validation_records=(
            ActionValidationRecord(
                validation_id="validation-approval-1",
                idempotency_key="action-approval-1",
                request_digest="d" * 64,
                decision_id="decision-approval-1",
                intent_id="action-approval-1",
                tool_name="local_notification_enqueue",
                risk_class=RiskClass.REVERSIBLE_WRITE,
                arguments_valid=True,
                validation_schema_revision=_validation_schema_revision(
                    "local_notification_enqueue",
                    _TOOLS["local_notification_enqueue"],
                ),
                validated_event_id="event-approval-1",
                validated_event_sequence=1,
                canonical_arguments_digest=digest,
                validated_at=timestamp,
            ),
        ),
        approvals=(
            ApprovalRecord(
                approval_id="approval-1",
                intent_id="action-approval-1",
                status="pending",
                requested_at=timestamp,
            ),
        ),
    )
    validate_action_state_semantics(state)
    store.last_snapshot = current.model_copy(
        update={"extensions": {"action_execution": state.model_dump(mode="json")}}
    )
    assert not service.preview(1).restoreable


def test_rebound_terminal_action_receipt_fails_closed(tmp_path: Any) -> None:
    from kagya.actions.execution import ActionState

    service, store, _wal, _journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    valid = ActionState.model_validate(_restore_action_state(active=False))
    state = valid.model_copy(
        update={
            "receipts": (
                valid.receipts[0].model_copy(update={"idempotency_key": "rebound-key"}),
            )
        }
    )
    store.last_snapshot = current.model_copy(
        update={"extensions": {"action_execution": state.model_dump(mode="json")}}
    )
    assert not service.preview(1).restoreable


def _operation_for_test(
    operation_id: str,
    target_hash: str,
    *,
    state: str = "finalizing",
    sequence: int | None = 2,
) -> RestoreOperation:
    terminal = state in {"completed", "failed"}
    return RestoreOperation(
        operation_id=operation_id,
        target_sequence=1,
        target_snapshot_hash=target_hash,
        preview_digest="b" * 64,
        requested_at="2026-01-01T00:00:00+00:00",
        started_at="2026-01-01T00:00:01+00:00",
        completed_at="2026-01-01T00:00:02+00:00" if terminal else None,
        event_id=f"operator-restore-{operation_id}",
        processing_sequence=sequence,
        state=state,  # type: ignore[arg-type]
        error_code="restore_commit_failed" if state == "failed" else None,
    )


@pytest.mark.parametrize(
    "field",
    [
        "target_sequence",
        "target_snapshot_hash",
        "preview_digest",
        "processing_sequence",
    ],
)
def test_persisted_and_journal_same_uuid_mismatch_is_rejected(
    tmp_path: Any, field: str
) -> None:
    service, store, wal, _journal = _service(tmp_path)
    operation_id = "00000000-0000-0000-0000-000000000201"
    target_hash = wal.reconstruct(1).snapshot_hash
    journal_operation = _operation_for_test(operation_id, target_hash, sequence=3)
    persisted = journal_operation.model_dump(mode="json")
    persisted[field] = {
        "target_sequence": 2,
        "target_snapshot_hash": "c" * 64,
        "preview_digest": "d" * 64,
        "processing_sequence": 4,
    }[field]
    if field == "target_sequence":
        persisted["target_snapshot_hash"] = wal.reconstruct(2).snapshot_hash
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"operator_restore": [persisted]}}
    )
    service._journal_operations = lambda _records: [journal_operation]
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value


def test_persisted_and_journal_same_uuid_event_identity_mismatch_is_rejected(
    tmp_path: Any,
) -> None:
    service, store, wal, _journal = _service(tmp_path)
    operation_id = "00000000-0000-0000-0000-000000000202"
    operation = _operation_for_test(operation_id, wal.reconstruct(1).snapshot_hash)
    persisted = operation.model_dump(mode="json")
    persisted["event_id"] = "operator-restore-00000000-0000-0000-0000-000000000203"
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"operator_restore": [persisted]}}
    )
    service._journal_operations = lambda _records: [operation]
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value


def test_duplicate_persisted_operation_uuid_is_rejected(tmp_path: Any) -> None:
    service, store, wal, _journal = _service(tmp_path)
    operation = _operation_for_test(
        "00000000-0000-0000-0000-000000000204", wal.reconstruct(1).snapshot_hash
    )
    raw = operation.model_dump(mode="json")
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"operator_restore": [raw, raw]}}
    )
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value


def test_invalid_persisted_operation_timestamp_order_is_rejected(tmp_path: Any) -> None:
    service, store, wal, _journal = _service(tmp_path)
    raw = _operation_for_test(
        "00000000-0000-0000-0000-000000000205", wal.reconstruct(1).snapshot_hash
    ).model_dump(mode="json")
    raw["completed_at"] = "2025-12-31T23:59:59+00:00"
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"operator_restore": [raw]}}
    )
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value


@pytest.mark.parametrize(
    "persisted_state,journal_state", [("completed", "failed"), ("failed", "completed")]
)
def test_persisted_terminal_state_conflicting_with_journal_is_rejected(
    tmp_path: Any, persisted_state: str, journal_state: str
) -> None:
    service, store, wal, _journal = _service(tmp_path)
    operation_id = "00000000-0000-0000-0000-000000000206"
    target_hash = wal.reconstruct(1).snapshot_hash
    persisted = _operation_for_test(operation_id, target_hash, state=persisted_state)
    journal = _operation_for_test(operation_id, target_hash, state="finalizing")
    journal = journal.model_copy(update={"state": journal_state})
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"operator_restore": [persisted.model_dump(mode="json")]}}
    )
    service._journal_operations = lambda _records: [journal]
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value


def test_finalizing_persisted_operation_with_completed_journal_projects_completed(
    tmp_path: Any,
) -> None:
    service, store, wal, journal = _service(tmp_path)
    operation_id = "00000000-0000-0000-0000-000000000207"
    target_hash = wal.reconstruct(1).snapshot_hash
    operation = _operation_for_test(operation_id, target_hash, sequence=3)
    event = Event(
        operation.event_id,
        3,
        event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
        source="api.state.operator_restore.commit",
        correlation_id=operation.preview_digest,
        payload={"journal_target": f"restore:1:{target_hash}"},
    )
    journal.accepted(
        Event(
            event.event_id,
            None,  # type: ignore[arg-type]
            event_type=event.event_type,
            source=event.source,
            correlation_id=event.correlation_id,
            payload=event.payload,
        )
    )
    journal.started(event)
    current = store.last_snapshot
    assert current is not None
    post = _snapshot(current, 3, 0.6)
    wal.append_transition(event, current, post)
    journal.completed(event, hash_snapshot(post))
    store.last_snapshot = post.model_copy(
        update={"extensions": {"operator_restore": [operation.model_dump(mode="json")]}}
    )
    projected = service.summary().latest_operation
    assert projected is not None
    assert projected.state == "completed"


def test_corrupt_target_snapshot_operation_history_is_rejected_before_apply(
    tmp_path: Any,
) -> None:
    service, store, wal, _journal = _service(tmp_path)
    operation = _operation_for_test("00000000-0000-0000-0000-000000000208", "e" * 64)
    store.last_snapshot = store.last_snapshot.model_copy(
        update={"extensions": {"operator_restore": [operation.model_dump(mode="json")]}}
    )
    before = hash_snapshot(store.last_snapshot)
    with pytest.raises(RestoreContractError) as exc:
        service.summary()
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value
    assert hash_snapshot(store.last_snapshot) == before


def test_apply_rejects_corrupted_reconstructed_target_history(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kagya.runtime.agent_runtime import _current_event

    service, _store, wal, _journal = _service(tmp_path)
    preview = service.preview(1)
    request = _request(preview)
    service.preflight(request, "operator")
    reconstruction = wal.reconstruct(1)
    corrupt_operation = _operation_for_test(
        "00000000-0000-0000-0000-000000000209", "e" * 64, sequence=1
    )
    corrupt_snapshot = reconstruction.snapshot.model_copy(
        update={
            "extensions": {
                **reconstruction.snapshot.extensions,
                "operator_restore": [corrupt_operation.model_dump(mode="json")],
            }
        }
    )
    monkeypatch.setattr(
        wal,
        "reconstruct",
        lambda _sequence: reconstruction.model_copy(
            update={"snapshot": corrupt_snapshot}
        ),
    )
    token = _current_event.set(
        AgentEvent(
            event_id=f"operator-restore-{preview.operation_id}",
            event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.operator_restore.commit",
            processing_sequence=3,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            requested_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    try:
        with pytest.raises(RestoreContractError) as exc:
            service.apply_commit(object(), request, "operator")
    finally:
        _current_event.reset(token)
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value
    assert request.expected_preview_digest not in service._consumed


@pytest.mark.parametrize(
    ("journal_terminal", "persisted_state"),
    [("completed", "failed"), ("failed", "completed")],
)
def test_apply_rejects_persisted_history_conflicting_with_journal_lifecycle(
    tmp_path: Any, journal_terminal: str, persisted_state: str
) -> None:
    from kagya.runtime.agent_runtime import _current_event

    service, store, wal, journal = _service(tmp_path)
    current = store.last_snapshot
    assert current is not None
    target_hash = wal.reconstruct(1).snapshot_hash
    historical_id = "00000000-0000-0000-0000-000000000210"
    accepted_event = Event(
        f"operator-restore-{historical_id}",
        None,  # type: ignore[arg-type]
        event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE.value,
        source="api.state.operator_restore.commit",
        correlation_id="e" * 64,
        payload={"journal_target": f"restore:1:{target_hash}"},
    )
    journal.accepted(accepted_event)
    historical_event = Event(
        accepted_event.event_id,
        3,
        event_type=accepted_event.event_type,
        source=accepted_event.source,
        correlation_id=accepted_event.correlation_id,
        payload=accepted_event.payload,
    )
    journal.started(historical_event)
    post = _snapshot(current, 3, 0.6)
    wal.append_transition(historical_event, current, post)
    if journal_terminal == "completed":
        terminal_record = journal.completed(historical_event, hash_snapshot(post))
    else:
        terminal_record = journal.failed(
            historical_event, "injected_failure", hash_snapshot(post)
        )
    journal_operation = service._journal_operations(journal.verify())[-1]
    persisted = RestoreOperation.model_validate(
        {
            **journal_operation.model_dump(mode="json"),
            "state": persisted_state,
            "completed_at": terminal_record.timestamp.isoformat(),
            "error_code": (
                "restore_commit_failed" if persisted_state == "failed" else None
            ),
        }
    )
    store.last_snapshot = post.model_copy(
        update={
            "extensions": {
                **post.extensions,
                "operator_restore": [persisted.model_dump(mode="json")],
            }
        }
    )
    preview = service.preview(1)
    request = _request(preview)
    service.preflight(request, "operator")
    token = _current_event.set(
        AgentEvent(
            event_id=f"operator-restore-{preview.operation_id}",
            event_type=AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.operator_restore.commit",
            processing_sequence=4,
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            requested_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    try:
        with pytest.raises(RestoreContractError) as exc:
            service.apply_commit(object(), request, "operator")
    finally:
        _current_event.reset(token)
    assert exc.value.code == RestoreErrorCode.JOURNAL_INTEGRITY_INVALID.value
    assert request.expected_preview_digest not in service._consumed


class _OpaqueRestoreStore:
    def list_external_transactions(self) -> list[ExternalTransactionRecord]:
        return [
            ExternalTransactionRecord(
                transaction_id="opaque-transaction",
                revision=1,
                artifact_type="episodic_chroma",
                artifact_id="opaque-memory",
                status=ExternalTransactionStatus.COMMITTED,
                event_id="opaque-event",
                processing_sequence=2,
                source="test",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                audit=[],
            )
        ]

    def finalize_external_event(self, event_id: str, processing_sequence: int) -> int:
        return 0

    def orphan_external_event(self, event_id: str, reason: str) -> int:
        return 0

    def compensate_external_event(self, event_id: str, reason: str) -> int:
        return 0


def test_external_capability_uses_injected_256_bit_token_and_is_opaque(
    tmp_path: Any,
) -> None:
    tokens = iter([bytes.fromhex("a" * 64), bytes.fromhex("b" * 64)])
    seen: list[bytes] = []

    def factory() -> bytes:
        token = next(tokens)
        seen.append(token)
        return token

    service, _store, _wal, _journal = _service(tmp_path, token_factory=factory)
    service.external = ExternalTransactionCoordinator([_OpaqueRestoreStore()])
    first = service.preview(1)
    second = service.preview(1)
    assert seen and all(len(value) == 32 for value in seen)
    assert (
        first.external_effects.artifacts[0].refs
        != second.external_effects.artifacts[0].refs
    )
    assert "opaque-memory" not in first.model_dump_json()
    assert (
        __import__("hashlib").sha256(b"opaque-memory").hexdigest()
        not in first.model_dump_json()
    )


@pytest.mark.parametrize("bad_token", [b"a" * 31, b"a" * 33, "a" * 64, 123])
def test_external_capability_rejects_malformed_factory_output(
    tmp_path: Any, bad_token: Any
) -> None:
    service, _store, _wal, _journal = _service(
        tmp_path, token_factory=lambda: bad_token
    )
    service.external = ExternalTransactionCoordinator([_OpaqueRestoreStore()])
    with pytest.raises(RestoreContractError) as exc:
        service.preview(1)
    assert exc.value.code == RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT.value


def test_external_capability_stales_on_state_mutation_and_rejects_forged_token(
    tmp_path: Any,
) -> None:
    tokens = iter([bytes.fromhex("b" * 64), bytes.fromhex("c" * 64)])
    service, store, _wal, _journal = _service(
        tmp_path, token_factory=lambda: next(tokens)
    )
    service.external = ExternalTransactionCoordinator([_OpaqueRestoreStore()])
    first = service.preview(1)
    second = service.preview(1)
    assert first.external_effects.effect_digest != second.external_effects.effect_digest
    store.last_snapshot = store.last_snapshot.model_copy(
        update={
            "emotion_state": store.last_snapshot.emotion_state.model_copy(
                update={"valence": 0.8}
            )
        }
    )
    with pytest.raises(RestoreContractError) as exc:
        service.preflight(_request(first), "operator")
    assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value
    forged = _request(second, expected_external_effect_digest="d" * 64)
    with pytest.raises(RestoreContractError) as exc:
        service.preflight(forged, "operator")
    assert exc.value.code == RestoreErrorCode.PREVIEW_STALE.value
