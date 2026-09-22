"""Journal-aware retirement of Memory-owned Semantic operation receipts."""

from __future__ import annotations

from kagya.memory.semantic_participant import (
    MEMORY_SEMANTIC_PARTICIPANT_ID,
    MemorySemanticParticipant,
    SemanticCreateIntent,
)
from kagya.memory.semantic_store import (
    SemanticStore,
    SemanticStoreConflict,
    SemanticStoreError,
)
from kagya.runtime.event_journal import EventJournal
from kagya.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantUnavailableError,
    UnsupportedParticipantReconciliationError,
)


class SemanticReceiptRetentionCoordinator:
    """Retire receipts only after Journal participant-finalized evidence."""

    def __init__(self, journal: EventJournal, store: SemanticStore) -> None:
        self.journal = journal
        self.store = store

    def _verified_checkpoint_horizon(self, inspection: object) -> int | None:
        checkpoint = self.store.load_checkpoint()
        if checkpoint is None:
            return None
        lineage = getattr(inspection, "journal_lineage_id", None)
        processing_high_water = getattr(inspection, "processing_high_water", None)
        records = getattr(inspection, "records", ())
        if (
            checkpoint.journal_lineage_id != lineage
            or type(processing_high_water) is not int
            or checkpoint.processing_high_water > processing_high_water
            or not any(
                record.record_id == checkpoint.journal_tail_record_id
                and record.record_hash == checkpoint.journal_tail_record_hash
                for record in records
            )
        ):
            return None
        return checkpoint.processing_high_water

    def checkpoint_covers(self, transaction: object) -> bool:
        """Return whether a clean Semantic checkpoint covers this transaction."""

        try:
            inspection = self.journal.inspect()
            horizon = self._verified_checkpoint_horizon(inspection)
        except SemanticStoreError:
            return False
        sequence = getattr(transaction, "processing_sequence", None)
        return horizon is not None and type(sequence) is int and sequence <= horizon

    def advance_checkpoint(self) -> None:
        """Persist a cross-authority clean boundary after terminal completion."""

        inspection = self.journal.inspect()
        if (
            inspection.open_events
            or inspection.open_transactions
            or inspection.open_recoveries
            or inspection.external_reconciliation_required
            or inspection.journal_lineage_id is None
            or inspection.tail_record_id is None
            or inspection.tail_record_hash is None
        ):
            return
        self.store.write_checkpoint(
            processing_high_water=inspection.processing_high_water,
            journal_lineage_id=inspection.journal_lineage_id,
            journal_tail_record_id=inspection.tail_record_id,
            journal_tail_record_hash=inspection.tail_record_hash,
        )

    def _retired_transaction_proofs(self) -> dict[str, str]:
        inspection = self.journal.inspect()
        transactions = (
            *inspection.completed_transactions,
            *inspection.reconciled_transactions,
        )
        latest_baseline = inspection.baselines[-1] if inspection.baselines else None
        checkpoint_horizon = self._verified_checkpoint_horizon(inspection)
        baseline_horizon = (
            None
            if latest_baseline is None
            else latest_baseline.processing_high_water
        )
        horizons = tuple(
            horizon
            for horizon in (baseline_horizon, checkpoint_horizon)
            if horizon is not None
        )
        safe_horizon = max(horizons) if horizons else None
        proofs: dict[str, str] = {}
        for transaction in transactions:
            if not any(
                participant_id == MEMORY_SEMANTIC_PARTICIPANT_ID
                for participant_id, _outcome in transaction.participant_outcomes
            ):
                continue
            requirement = next(
                (
                    item
                    for item in transaction.required_participants
                    if item.participant_id == MEMORY_SEMANTIC_PARTICIPANT_ID
                ),
                None,
            )
            if requirement is None:
                continue
            if safe_horizon is not None and transaction.processing_sequence <= safe_horizon:
                reconstructible = True
            else:
                try:
                    operation = MemorySemanticParticipant.operation_from_authority(
                        self.store,
                        transaction.transaction_id,
                        transaction.event_id,
                        transaction.processing_sequence,
                        requirement.operation_digest,
                    )
                except (
                    ParticipantDivergedError,
                    ParticipantUnavailableError,
                    UnsupportedParticipantReconciliationError,
                ):
                    reconstructible = False
                else:
                    # Revision authority can later be compacted out of the
                    # retained Semantic window.  U3 create batches have
                    # deterministic revision-zero artifacts; retain revision
                    # receipts until a baseline proves them unreachable.
                    reconstructible = all(
                        isinstance(entry.mutation, SemanticCreateIntent)
                        for entry in operation.entries
                    )
                if not reconstructible:
                    continue
            proofs[transaction.transaction_id] = requirement.operation_digest
        return proofs

    def before_prepare(self) -> None:
        """Retry safe retirement before admitting another coordinated mutation."""

        proofs = self._retired_transaction_proofs()
        for transaction_id, operation_digest in proofs.items():
            receipt = self.store.load_receipt(transaction_id)
            if receipt is not None and receipt.get("operation_digest") != operation_digest:
                raise SemanticStoreConflict("Semantic receipt proof conflicts")
        self.store.prune_receipts(proofs)

    def after_participant_finalized(self) -> None:
        """Best-effort cleanup after the Journal outcome is durable."""

        try:
            self.before_prepare()
        except SemanticStoreError:
            # The participant outcome is already durable.  Leaving a receipt
            # behind is safe; startup and the next admission retry cleanup.
            return

    def after_terminal_completion(self) -> None:
        """Advance the clean checkpoint, then retire newly unreachable receipts."""

        try:
            self.advance_checkpoint()
            self.before_prepare()
        except SemanticStoreError:
            # Terminal Journal/state evidence is already durable.  Keep
            # receipts until a later startup or admission retry can prove and
            # retire them.
            return


__all__ = ["SemanticReceiptRetentionCoordinator"]
