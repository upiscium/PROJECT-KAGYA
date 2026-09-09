"""Startup-only reconciliation of external transaction participants.

The coordinator consumes Journal, snapshot, and WAL evidence but does not own any
of those authorities.  It is deliberately separate from live event execution so
startup can never replay handlers or recover request payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid5

from kagya.memory.dual_memory_system import DualMemorySystem
from kagya.memory.episodic_participant import (
    MEMORY_EPISODIC_PARTICIPANT_ID,
    MemoryEpisodicParticipant,
)
from kagya.runtime.agent_runtime import AgentEvent
from kagya.runtime.event_journal import (
    EventJournal,
    EventJournalIntegrityError,
    EventJournalParticipantBaseline,
    EventJournalRecord,
    EventJournalStartupReconciliation,
    EventJournalTransaction,
    EventLifecycle,
    EventRecoveryCategory,
    ParticipantBaseline,
    ParticipantCapability,
    ParticipantDomain,
    ParticipantOutcome,
    ParticipantRequirement,
    StartupParticipantOutcome,
    startup_participant_aggregate_digest,
)
from kagya.runtime.session_participant import (
    SESSION_TURN_PARTICIPANT_ID,
    inspect_reset_session_operation,
)
from kagya.runtime.state_recovery import (
    InternalCommitClassification,
    StateRecoveryCoordinator,
    StateRecoveryError,
    StateRecoveryResult,
)
from kagya.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    UnsupportedParticipantReconciliationError,
)


_BASELINE_NAMESPACE = UUID("f0ee9f9a-c256-5ec8-8f02-c245f9e8db04")
_RECONCILIATION_NAMESPACE = UUID("0b941247-478c-5e32-9142-d82b165a330f")


class StartupReconciliationError(StateRecoveryError, EventJournalIntegrityError):
    """Startup evidence is ambiguous and runtime admission must fail closed."""


@dataclass(frozen=True, slots=True)
class StartupReconciliationResult:
    recovery: StateRecoveryResult
    participants_consistent: bool
    degraded_reason: str | None = None


class StartupReconciliationCoordinator:
    """Resolve open transactions and a true-rollback external recovery gate."""

    participant_registry = (
        ParticipantBaseline(
            participant_id=MEMORY_EPISODIC_PARTICIPANT_ID,
            domain=ParticipantDomain.DURABLE_DOMAIN,
        ),
        ParticipantBaseline(
            participant_id=SESSION_TURN_PARTICIPANT_ID,
            domain=ParticipantDomain.EPHEMERAL_PROCESS,
        ),
    )

    def __init__(
        self,
        journal: EventJournal,
        state_recovery: StateRecoveryCoordinator,
        memory: DualMemorySystem,
    ) -> None:
        self.journal = journal
        self.state_recovery = state_recovery
        self.memory = memory

    def reconcile_open_transactions(self) -> tuple[bool, str | None]:
        """Resolve Path A without replaying an event handler or model call."""

        transactions = sorted(
            self.journal.inspect().open_transactions,
            key=lambda item: item.transaction_id,
        )
        for transaction in transactions:
            proof = self.state_recovery.inspect_transaction_commit(transaction)
            if proof.classification is InternalCommitClassification.AMBIGUOUS:
                raise StartupReconciliationError(
                    "Open transaction internal commit evidence is ambiguous"
                )
            try:
                if proof.classification is InternalCommitClassification.PRE_INTERNAL:
                    self._abort_transaction(transaction)
                else:
                    self._roll_forward_transaction(transaction)
            except (
                ParticipantDivergedError,
                ParticipantUnavailableError,
                UnsupportedParticipantReconciliationError,
            ):
                return False, "external_participant_reconciliation_required"
        return True, None

    def resume_prepared_gate_clear(self) -> bool:
        """Finish a crash window after CLEAR_PREPARED, including a prior WAL CAS."""

        inspection = self.journal.inspect()
        clear = inspection.open_gate_clear
        if clear is None:
            return False
        baseline = next(
            (
                item
                for item in inspection.baselines
                if item.baseline_id == clear.baseline_id
            ),
            None,
        )
        reconciliation = next(
            (
                item
                for item in inspection.completed_startup_reconciliations
                if item.reconciliation_id == clear.reconciliation_id
            ),
            None,
        )
        if baseline is None or reconciliation is None:
            raise StartupReconciliationError("Prepared gate clear proof is incomplete")
        snapshot = self.state_recovery.state_store.load()
        snapshot_hash = self.state_recovery.state_store.snapshot_hash(snapshot)
        wal = self.state_recovery.wal.inspect()
        manifest = wal.active_manifest
        if manifest is None:
            raise StartupReconciliationError("Prepared gate clear WAL is unavailable")
        resumed = StateRecoveryResult(
            snapshot,
            snapshot_hash,
            clear.processing_high_water,
            manifest,
            manifest.external_reconciliation_required,
        )
        self.state_recovery.clear_recovery_gate(
            resumed, reconciliation, baseline, clear
        )
        self.journal.append_cleared(
            clear.baseline_id,
            clear.reconciliation_id,
            clear.recovery_id,
            clear.snapshot_sequence,
            clear.snapshot_hash,
            clear.processing_high_water,
            clear.wal_generation_id,
            clear.wal_record_id,
            clear.wal_record_hash,
            clear.journal_lineage_id,
        )
        return True

    def reconcile_recovery_gate(
        self, recovery: StateRecoveryResult
    ) -> StartupReconciliationResult:
        """Resolve Path B and clear only a proof-bound true-rollback gate."""

        if not recovery.external_reconciliation_required:
            return StartupReconciliationResult(recovery, True)
        inspection = self.journal.inspect()
        recovery_record = self._true_rollback_record(inspection.records)
        if recovery_record is None:
            raise StartupReconciliationError(
                "External gate has no true rollback completion proof"
            )
        baseline = self._baseline(recovery, recovery_record)
        transactions = tuple(
            sorted(
                (
                    *inspection.completed_transactions,
                    *inspection.reconciled_transactions,
                ),
                key=lambda item: item.transaction_id,
            )
        )
        requirements = self._aggregate_requirements(transactions)
        reconciliation = self._startup_reconciliation(
            recovery_record, baseline, requirements
        )
        known = {item[0] for item in reconciliation.participant_outcomes}
        try:
            for requirement in requirements:
                if requirement.participant_id in known:
                    continue
                outcome = self._reconcile_aggregate_participant(
                    requirement.participant_id, transactions
                )
                self.journal.append_startup_participant_reconciled(
                    reconciliation.reconciliation_id,
                    reconciliation.recovery_id,
                    reconciliation.snapshot_sequence,
                    reconciliation.snapshot_hash,
                    reconciliation.recovery_processing_high_water,
                    reconciliation.wal_generation_id,
                    reconciliation.wal_record_id,
                    reconciliation.wal_record_hash,
                    reconciliation.journal_lineage_id,
                    requirement.participant_id,
                    requirement.operation_digest,
                    outcome,
                )
            reconciliation = self._complete_startup(reconciliation)
        except (
            ParticipantDivergedError,
            ParticipantUnavailableError,
            UnsupportedParticipantReconciliationError,
        ):
            return StartupReconciliationResult(
                recovery, False, "external_participant_reconciliation_required"
            )

        clear = self.journal.inspect().open_gate_clear
        if clear is None:
            self.journal.append_clear_prepared(
                baseline.baseline_id,
                reconciliation.reconciliation_id,
                reconciliation.recovery_id,
                reconciliation.snapshot_sequence,
                reconciliation.snapshot_hash,
                reconciliation.recovery_processing_high_water,
                reconciliation.wal_generation_id,
                reconciliation.wal_record_id,
                reconciliation.wal_record_hash,
                reconciliation.journal_lineage_id,
            )
            clear = self.journal.inspect().open_gate_clear
        if clear is None:
            raise StartupReconciliationError("Gate clear preparation is unavailable")
        cleared = self.state_recovery.clear_recovery_gate(
            recovery, reconciliation, baseline, clear
        )
        self.journal.append_cleared(
            baseline.baseline_id,
            reconciliation.reconciliation_id,
            reconciliation.recovery_id,
            reconciliation.snapshot_sequence,
            reconciliation.snapshot_hash,
            reconciliation.recovery_processing_high_water,
            reconciliation.wal_generation_id,
            reconciliation.wal_record_id,
            reconciliation.wal_record_hash,
            reconciliation.journal_lineage_id,
        )
        return StartupReconciliationResult(cleared, True)

    def _abort_transaction(self, transaction: EventJournalTransaction) -> None:
        event = self._event(transaction)
        known = {item[0] for item in transaction.abort_outcomes}
        for requirement in transaction.required_participants:
            if (
                requirement.participant_id in known
                or ParticipantCapability.ABORT not in requirement.capabilities
            ):
                continue
            outcome = MemoryEpisodicParticipant.abort_pending(
                self.memory, self._binding(transaction, requirement)
            )
            self.journal.append_participant_aborted(
                event,
                transaction.transaction_id,
                requirement.participant_id,
                requirement.operation_digest,
                outcome,
            )
        current = self._transaction(transaction.transaction_id)
        abort_required = {
            item.participant_id
            for item in current.required_participants
            if ParticipantCapability.ABORT in item.capabilities
        }
        if {item[0] for item in current.abort_outcomes} != abort_required:
            raise ParticipantUnavailableError("Participant abort is incomplete")
        self.journal.append_transaction_aborted(event, transaction.transaction_id)

    def _roll_forward_transaction(
        self, transaction: EventJournalTransaction
    ) -> None:
        event = self._event(transaction)
        known = {item[0] for item in transaction.participant_outcomes}
        for requirement in transaction.required_participants:
            if requirement.participant_id in known:
                continue
            if requirement.participant_id == MEMORY_EPISODIC_PARTICIPANT_ID:
                outcome = self._memory_participant(
                    transaction, requirement
                ).finalize(self._binding(transaction, requirement))
            elif requirement.participant_id == SESSION_TURN_PARTICIPANT_ID:
                inspect_reset_session_operation(
                    transaction.transaction_id,
                    requirement.participant_id,
                    requirement.operation_digest,
                )
                outcome = ParticipantOutcome.ALREADY_CONSISTENT
            else:
                raise UnsupportedParticipantReconciliationError(
                    "Participant resolver is not registered"
                )
            self.journal.append_participant_finalized(
                event,
                transaction.transaction_id,
                requirement.participant_id,
                requirement.operation_digest,
                outcome,
            )
        if transaction.reconciliation_reason is None:
            self.journal.append_transaction_completed(event, transaction.transaction_id)
        else:
            self.journal.append_transaction_reconciled(event, transaction.transaction_id)

    def _baseline(
        self, recovery: StateRecoveryResult, record: EventJournalRecord
    ) -> EventJournalParticipantBaseline:
        identity = ":".join(
            (
                record.recovery_id or "",
                str(recovery.snapshot.last_processed_event_sequence),
                recovery.snapshot_hash,
            )
        )
        baseline_id = str(uuid5(_BASELINE_NAMESPACE, identity))
        inspection = self.journal.inspect()
        existing = next(
            (item for item in inspection.baselines if item.baseline_id == baseline_id),
            None,
        )
        if existing is not None:
            return existing
        if (
            record.wal_generation_id is None
            or record.wal_record_id is None
            or record.wal_record_hash is None
            or inspection.journal_lineage_id is None
        ):
            raise StartupReconciliationError("Recovery baseline evidence is incomplete")
        self.journal.append_participant_baseline(
            baseline_id,
            recovery.snapshot.last_processed_event_sequence,
            recovery.snapshot_hash,
            recovery.processing_high_water,
            record.wal_generation_id,
            record.wal_record_id,
            record.wal_record_hash,
            inspection.journal_lineage_id,
            self.participant_registry,
        )
        return self.journal.inspect().baselines[-1]

    def _startup_reconciliation(
        self,
        recovery: EventJournalRecord,
        baseline: EventJournalParticipantBaseline,
        requirements: tuple[ParticipantRequirement, ...],
    ) -> EventJournalStartupReconciliation:
        reconciliation_id = str(
            uuid5(_RECONCILIATION_NAMESPACE, baseline.baseline_id)
        )
        inspection = self.journal.inspect()
        existing = next(
            (
                item
                for item in (
                    *inspection.open_startup_reconciliations,
                    *inspection.completed_startup_reconciliations,
                )
                if item.reconciliation_id == reconciliation_id
            ),
            None,
        )
        if existing is not None:
            if existing.required_participants != requirements:
                raise StartupReconciliationError(
                    "Startup reconciliation aggregate changed"
                )
            return existing
        assert recovery.recovery_id is not None
        self.journal.append_startup_reconciliation_prepared(
            reconciliation_id,
            recovery.recovery_id,
            baseline.snapshot_sequence,
            baseline.snapshot_hash,
            baseline.processing_high_water,
            baseline.wal_generation_id,
            baseline.wal_record_id,
            baseline.wal_record_hash,
            baseline.journal_lineage_id,
            requirements,
        )
        return self.journal.inspect().open_startup_reconciliations[-1]

    def _complete_startup(
        self, reconciliation: EventJournalStartupReconciliation
    ) -> EventJournalStartupReconciliation:
        if reconciliation.completed:
            return reconciliation
        current = next(
            item
            for item in self.journal.inspect().open_startup_reconciliations
            if item.reconciliation_id == reconciliation.reconciliation_id
        )
        self.journal.append_startup_reconciliation_completed(
            current.reconciliation_id,
            current.recovery_id,
            current.snapshot_sequence,
            current.snapshot_hash,
            current.recovery_processing_high_water,
            current.wal_generation_id,
            current.wal_record_id,
            current.wal_record_hash,
            current.journal_lineage_id,
        )
        return next(
            item
            for item in self.journal.inspect().completed_startup_reconciliations
            if item.reconciliation_id == current.reconciliation_id
        )

    def _reconcile_aggregate_participant(
        self,
        participant_id: str,
        transactions: tuple[EventJournalTransaction, ...],
    ) -> StartupParticipantOutcome:
        rolled_forward = False
        for transaction in transactions:
            requirement = next(
                (
                    item
                    for item in transaction.required_participants
                    if item.participant_id == participant_id
                ),
                None,
            )
            if requirement is None:
                continue
            if participant_id == SESSION_TURN_PARTICIPANT_ID:
                inspect_reset_session_operation(
                    transaction.transaction_id,
                    participant_id,
                    requirement.operation_digest,
                )
                continue
            if participant_id != MEMORY_EPISODIC_PARTICIPANT_ID:
                raise UnsupportedParticipantReconciliationError(
                    "Participant resolver is not registered"
                )
            participant = self._memory_participant(transaction, requirement)
            binding = self._binding(transaction, requirement)
            try:
                participant.inspect_reconciliation(binding)
            except ParticipantUnavailableError:
                participant.reconcile(binding)
                rolled_forward = True
        return (
            StartupParticipantOutcome.ROLLED_FORWARD
            if rolled_forward
            else StartupParticipantOutcome.VERIFIED_CONSISTENT
        )

    @staticmethod
    def _aggregate_requirements(
        transactions: tuple[EventJournalTransaction, ...],
    ) -> tuple[ParticipantRequirement, ...]:
        result: list[ParticipantRequirement] = []
        for participant_id in (
            MEMORY_EPISODIC_PARTICIPANT_ID,
            SESSION_TURN_PARTICIPANT_ID,
        ):
            digest = startup_participant_aggregate_digest(
                transactions, participant_id
            )
            result.append(
                ParticipantRequirement(
                    participant_id=participant_id,
                    operation_digest=digest,
                    capabilities=(
                        ParticipantCapability.IDEMPOTENT_FINALIZE,
                        ParticipantCapability.INSPECT_RECONCILE,
                        ParticipantCapability.PREPARE,
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _binding(
        transaction: EventJournalTransaction, requirement: ParticipantRequirement
    ) -> TransactionBinding:
        return TransactionBinding(
            transaction_id=transaction.transaction_id,
            event_id=transaction.event_id,
            processing_sequence=transaction.processing_sequence,
            participant_id=requirement.participant_id,
            operation_digest=requirement.operation_digest,
            transaction_kind=transaction.kind,
        )

    def _memory_participant(
        self,
        transaction: EventJournalTransaction,
        requirement: ParticipantRequirement,
    ) -> MemoryEpisodicParticipant:
        if requirement.participant_id != MEMORY_EPISODIC_PARTICIPANT_ID:
            raise UnsupportedParticipantReconciliationError(
                "Abort participant resolver is not registered"
            )
        return MemoryEpisodicParticipant.from_pending(
            self.memory,
            transaction.transaction_id,
            requirement.participant_id,
            requirement.operation_digest,
        )

    @staticmethod
    def _event(transaction: EventJournalTransaction) -> AgentEvent:
        return AgentEvent(
            event_id=transaction.event_id,
            event_type=transaction.event_type,
            source=transaction.source,
            requested_at=datetime.now(timezone.utc),
            processing_sequence=transaction.processing_sequence,
        )

    def _transaction(self, transaction_id: str) -> EventJournalTransaction:
        return next(
            item
            for item in self.journal.inspect().open_transactions
            if item.transaction_id == transaction_id
        )

    @staticmethod
    def _true_rollback_record(
        records: tuple[EventJournalRecord, ...],
    ) -> EventJournalRecord | None:
        matches = [
            record
            for record in records
            if record.lifecycle is EventLifecycle.RECOVERY_COMPLETED
            and record.recovery_category is EventRecoveryCategory.TRUE_ROLLBACK
            and record.external_reconciliation_required is True
        ]
        return matches[-1] if matches else None
