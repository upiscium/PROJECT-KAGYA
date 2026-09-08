from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from kagya.runtime import (
    AbortOutcome,
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    CoordinatedResult,
    EventJournal,
    EventLifecycle,
    InternalCommitEvidence,
    ParticipantCapability,
    ParticipantOutcome,
    TransactionCoordinator,
    TransactionFinalizationError,
    TransactionParticipant,
    TransactionPreparationError,
    TransactionKind,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
HASH_0 = "a" * 64
HASH_1 = "b" * 64
PRIVATE_SENTINEL = "PRIVATE-PARTICIPANT-PAYLOAD-R07"
BASIC_CAPABILITIES = (
    ParticipantCapability.IDEMPOTENT_FINALIZE,
    ParticipantCapability.PREPARE,
)
ABORT_CAPABILITIES = (
    ParticipantCapability.ABORT,
    *BASIC_CAPABILITIES,
)


class FakeParticipant:
    def __init__(
        self,
        participant_id: str,
        *,
        capabilities: tuple[ParticipantCapability, ...] = BASIC_CAPABILITIES,
        fail_prepare: bool = False,
        fail_abort: bool = False,
        fail_finalize: bool = False,
        order: list[str] | None = None,
    ) -> None:
        self.participant_id = participant_id
        self.operation_digest = HASH_1
        self.capabilities = capabilities
        self.fail_prepare = fail_prepare
        self.fail_abort = fail_abort
        self.fail_finalize = fail_finalize
        self.order = order
        self.calls: list[str] = []

    def prepare(self, _binding: object) -> None:
        self.calls.append("prepare")
        if self.order is not None:
            self.order.append(f"prepare-{self.participant_id}")
        if self.fail_prepare:
            raise OSError(PRIVATE_SENTINEL)

    def abort(self, _binding: object) -> AbortOutcome:
        self.calls.append("abort")
        if self.order is not None:
            self.order.append(f"abort-{self.participant_id}")
        if self.fail_abort:
            raise OSError(PRIVATE_SENTINEL)
        return AbortOutcome.ABORTED

    def finalize(self, _binding: object) -> ParticipantOutcome:
        self.calls.append("finalize")
        if self.order is not None:
            self.order.append(f"finalize-{self.participant_id}")
        if self.fail_finalize:
            raise OSError(PRIVATE_SENTINEL)
        return ParticipantOutcome.FINALIZED


def event(name: str = "event", sequence: int = 1) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, name)),
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        NOW,
        sequence,
    )


def journal(path: Path) -> EventJournal:
    value = EventJournal(path, 100_000, 4, clock=lambda: NOW)
    generation = str(uuid5(NAMESPACE_URL, "generation"))
    value.append_v2_bootstrap_checkpoint(
        0,
        HASH_0,
        generation,
        str(uuid5(NAMESPACE_URL, "wal-record")),
        HASH_1,
    )
    value.append_v3_migration_checkpoint()
    return value


def start(value: EventJournal, item: AgentEvent) -> None:
    value.append_accepted(item)
    value.append_started(item)


def commit_internal(value: EventJournal, item: AgentEvent) -> InternalCommitEvidence:
    generation = value.records[0].wal_generation_id
    assert generation is not None
    value.append_prepared(item, HASH_0, HASH_1, generation)
    return InternalCommitEvidence(
        event_id=item.event_id,
        processing_sequence=item.processing_sequence or 0,
        snapshot_sequence=item.processing_sequence or 0,
        snapshot_hash=HASH_1,
        wal_generation_id=generation,
        wal_record_id=str(uuid5(NAMESPACE_URL, f"wal-{item.event_id}")),
        wal_record_hash=HASH_1,
    )


def assert_bounded(error: Exception) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert PRIVATE_SENTINEL not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def coordinator(value: EventJournal) -> TransactionCoordinator:
    return TransactionCoordinator(value, lambda _event, _evidence: None)


def test_no_transaction_is_a_true_no_op(tmp_path: Path) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    transaction_coordinator = coordinator(value)

    public = transaction_coordinator.prepare_result(item, {"answer": "ok"})
    transaction_coordinator.finalize_event(item, commit_internal(value, item))

    assert public == {"answer": "ok"}
    assert not any(record.transaction_id for record in value.records)


def test_prepare_and_finalize_are_sorted_and_publish_only_public_value(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    order: list[str] = []
    first = FakeParticipant("z.participant", order=order)
    second = FakeParticipant("a.participant", order=order)
    transaction_coordinator = coordinator(value)

    public = transaction_coordinator.prepare_result(
        item, CoordinatedResult("public", (first, second))
    )
    transaction_coordinator.finalize_event(item, commit_internal(value, item))

    assert public == "public"
    assert order == [
        "prepare-a.participant",
        "prepare-z.participant",
        "finalize-a.participant",
        "finalize-z.participant",
    ]
    transaction = value.inspect().completed_transactions[0]
    assert tuple(x.participant_id for x in transaction.required_participants) == (
        "a.participant",
        "z.participant",
    )
    assert PRIVATE_SENTINEL not in (tmp_path / "events.jsonl").read_text()


def test_invalid_complete_plan_fails_before_durable_declaration(tmp_path: Path) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    participant = FakeParticipant("participant")
    transaction_coordinator = coordinator(value)

    with pytest.raises(TransactionPreparationError) as error:
        transaction_coordinator.prepare_result(
            item, CoordinatedResult("public", (participant, participant))
        )

    assert_bounded(error.value)
    assert participant.calls == []
    assert not any(record.transaction_id for record in value.records)


def test_missing_required_operation_fails_before_durable_declaration(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)

    class DeclarationOnly:
        participant_id = "participant"
        operation_digest = HASH_1
        capabilities = BASIC_CAPABILITIES

    with pytest.raises(TransactionPreparationError):
        coordinator(value).prepare_result(
            item,
            CoordinatedResult(
                "public", (cast(TransactionParticipant, DeclarationOnly()),)
            ),
        )

    assert not any(record.transaction_id for record in value.records)


def test_declared_abort_must_be_callable_before_durable_declaration(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)

    class InvalidAbortParticipant:
        participant_id = "participant"
        operation_digest = HASH_1
        capabilities = ABORT_CAPABILITIES
        abort = None

        def __init__(self) -> None:
            self.calls: list[str] = []

        def prepare(self, _binding: object) -> None:
            self.calls.append("prepare")

        def finalize(self, _binding: object) -> ParticipantOutcome:
            self.calls.append("finalize")
            return ParticipantOutcome.FINALIZED

    participant = InvalidAbortParticipant()
    with pytest.raises(TransactionPreparationError):
        coordinator(value).prepare_result(
            item,
            CoordinatedResult("public", (cast(TransactionParticipant, participant),)),
        )

    assert participant.calls == []
    assert not any(record.transaction_id for record in value.records)


def test_prepare_failure_aborts_all_abort_capable_participants(tmp_path: Path) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    first = FakeParticipant("a", capabilities=ABORT_CAPABILITIES)
    second = FakeParticipant("b", capabilities=ABORT_CAPABILITIES, fail_prepare=True)
    transaction_coordinator = coordinator(value)

    with pytest.raises(TransactionPreparationError) as error:
        transaction_coordinator.prepare_result(
            item, CoordinatedResult("public", (second, first))
        )

    assert_bounded(error.value)
    assert first.calls == ["prepare", "abort"]
    assert second.calls == ["prepare", "abort"]
    transaction = value.inspect().aborted_transactions[0]
    assert transaction.terminal_lifecycle is EventLifecycle.TRANSACTION_ABORTED
    assert {participant for participant, _ in transaction.abort_outcomes} == {"a", "b"}


def test_prepare_failure_without_abort_capability_terminalizes_without_cleanup(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event("zero-abort")
    start(value, item)
    participant = FakeParticipant("participant", fail_prepare=True)

    with pytest.raises(TransactionPreparationError):
        coordinator(value).prepare_result(
            item, CoordinatedResult("public", (participant,))
        )

    assert participant.calls == ["prepare"]
    transaction = value.inspect().aborted_transactions[0]
    assert transaction.terminal_lifecycle is EventLifecycle.TRANSACTION_ABORTED
    assert transaction.abort_outcomes == ()
    assert not any(
        record.lifecycle is EventLifecycle.PARTICIPANT_ABORTED
        for record in value.records
    )
    assert not any(
        record.lifecycle is EventLifecycle.PREPARED for record in value.records
    )
    assert "finalize" not in participant.calls


def test_mixed_abort_capabilities_require_cleanup_only_for_declared_abort(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event("mixed-abort")
    start(value, item)
    abortable = FakeParticipant("a", capabilities=ABORT_CAPABILITIES)
    non_abortable = FakeParticipant("b", fail_prepare=True)

    with pytest.raises(TransactionPreparationError):
        coordinator(value).prepare_result(
            item, CoordinatedResult("public", (non_abortable, abortable))
        )

    assert abortable.calls == ["prepare", "abort"]
    assert non_abortable.calls == ["prepare"]
    transaction = value.inspect().aborted_transactions[0]
    assert tuple(participant for participant, _ in transaction.abort_outcomes) == (
        "a",
    )


def test_partial_abort_failure_records_exact_unresolved_set(tmp_path: Path) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    first = FakeParticipant("a", capabilities=ABORT_CAPABILITIES, fail_abort=True)
    second = FakeParticipant("b", capabilities=ABORT_CAPABILITIES, fail_prepare=True)
    transaction_coordinator = coordinator(value)

    with pytest.raises(TransactionPreparationError) as error:
        transaction_coordinator.prepare_result(
            item, CoordinatedResult("public", (first, second))
        )

    assert_bounded(error.value)
    transaction = value.inspect().open_transactions[0]
    assert transaction.abort_reason is not None
    assert transaction.unresolved_participants == ("a",)


def test_internal_commit_failure_never_aborts_or_finalizes(tmp_path: Path) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    participant = FakeParticipant("a", capabilities=ABORT_CAPABILITIES)
    transaction_coordinator = coordinator(value)

    transaction_coordinator.prepare_result(
        item, CoordinatedResult("public", (participant,))
    )

    assert participant.calls == ["prepare"]
    transaction = value.inspect().open_transactions[0]
    assert transaction.abort_outcomes == ()
    assert transaction.participant_outcomes == ()


def test_unverified_internal_commit_cannot_call_external_finalize(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    participant = FakeParticipant("a")

    def reject(_event: AgentEvent, _evidence: InternalCommitEvidence) -> None:
        raise OSError(PRIVATE_SENTINEL)

    transaction_coordinator = TransactionCoordinator(value, reject)
    transaction_coordinator.prepare_result(
        item, CoordinatedResult("public", (participant,))
    )

    with pytest.raises(TransactionFinalizationError) as error:
        transaction_coordinator.finalize_event(item, commit_internal(value, item))

    assert_bounded(error.value)
    assert participant.calls == ["prepare"]
    assert value.inspect().open_transactions[0].participant_outcomes == ()


def test_partial_finalize_failure_preserves_success_and_exact_unresolved_set(
    tmp_path: Path,
) -> None:
    value = journal(tmp_path / "events.jsonl")
    item = event()
    start(value, item)
    first = FakeParticipant("a")
    second = FakeParticipant("b", fail_finalize=True)
    transaction_coordinator = coordinator(value)
    transaction_coordinator.prepare_result(
        item, CoordinatedResult("public", (second, first))
    )

    with pytest.raises(TransactionFinalizationError) as error:
        transaction_coordinator.finalize_event(item, commit_internal(value, item))

    assert_bounded(error.value)
    assert first.calls == ["prepare", "finalize"]
    assert second.calls == ["prepare", "finalize"]
    transaction = value.inspect().open_transactions[0]
    assert transaction.participant_outcomes == (("a", ParticipantOutcome.FINALIZED),)
    assert transaction.unresolved_participants == ("b",)
    assert transaction.reconciliation_reason is not None


def test_transaction_identity_is_stable_across_coordinator_instances(
    tmp_path: Path,
) -> None:
    item = event("stable-identity", 7)
    first = coordinator(journal(tmp_path / "first.jsonl"))
    second = coordinator(journal(tmp_path / "second.jsonl"))

    assert first.derive_transaction_id(item, TransactionKind.EVENT_MUTATION) == (
        second.derive_transaction_id(item, TransactionKind.EVENT_MUTATION)
    )


def test_transaction_identity_binds_event_sequence_and_kind(tmp_path: Path) -> None:
    value = coordinator(journal(tmp_path / "events.jsonl"))
    original = value.derive_transaction_id(
        event("identity", 1), TransactionKind.EVENT_MUTATION
    )

    assert original != value.derive_transaction_id(
        event("other-identity", 1), TransactionKind.EVENT_MUTATION
    )
    assert original != value.derive_transaction_id(
        event("identity", 2), TransactionKind.EVENT_MUTATION
    )
    assert original != value.derive_transaction_id(
        event("identity", 1), TransactionKind.MAINTENANCE_MUTATION
    )


def test_transaction_identity_ignores_participant_order_and_private_payload(
    tmp_path: Path,
) -> None:
    item = event("order-independent")
    first_journal = journal(tmp_path / "first.jsonl")
    second_journal = journal(tmp_path / "second.jsonl")
    start(first_journal, item)
    start(second_journal, item)
    a = FakeParticipant("a")
    b = FakeParticipant("b")
    a.private_payload = PRIVATE_SENTINEL
    b.private_payload = PRIVATE_SENTINEL

    coordinator(first_journal).prepare_result(
        item, CoordinatedResult("first", (a, b))
    )
    coordinator(second_journal).prepare_result(
        item, CoordinatedResult("second", (b, a))
    )

    first_id = first_journal.inspect().open_transactions[0].transaction_id
    second_id = second_journal.inspect().open_transactions[0].transaction_id
    assert first_id == second_id
    assert first_id == "675fd3a1-4f5b-5e0c-8924-a35dd8b6234b"
    assert str(UUID(first_id)) == first_id
    assert PRIVATE_SENTINEL not in first_id
