"""Journal-aware retirement of Memory-owned Semantic operation receipts."""

from __future__ import annotations

from kagya.memory.semantic_participant import MEMORY_SEMANTIC_PARTICIPANT_ID
from kagya.memory.semantic_store import (
    SemanticStore,
    SemanticStoreConflict,
    SemanticStoreError,
)
from kagya.runtime.event_journal import EventJournal


class SemanticReceiptRetentionCoordinator:
    """Retire receipts only after Journal participant-finalized evidence."""

    def __init__(self, journal: EventJournal, store: SemanticStore) -> None:
        self.journal = journal
        self.store = store

    def _retired_transaction_proofs(self) -> dict[str, str]:
        inspection = self.journal.inspect()
        transactions = (
            *inspection.completed_transactions,
            *inspection.reconciled_transactions,
        )
        latest_baseline = inspection.baselines[-1] if inspection.baselines else None
        proofs: dict[str, str] = {}
        for transaction in transactions:
            if (
                latest_baseline is None
                or transaction.processing_sequence > latest_baseline.processing_high_water
            ):
                # A true rollback may still revisit every transaction after the
                # latest clean participant baseline.  Those receipts remain
                # the Memory-owned reconstruction evidence until a later
                # baseline epoch makes them unreachable.
                continue
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


__all__ = ["SemanticReceiptRetentionCoordinator"]
