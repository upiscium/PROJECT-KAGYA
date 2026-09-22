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

    def _retired_transaction_proofs(self) -> dict[str, str]:
        inspection = self.journal.inspect()
        transactions = (
            *inspection.completed_transactions,
            *inspection.reconciled_transactions,
        )
        latest_baseline = inspection.baselines[-1] if inspection.baselines else None
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
            if (
                latest_baseline is not None
                and transaction.processing_sequence <= latest_baseline.processing_high_water
            ):
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


__all__ = ["SemanticReceiptRetentionCoordinator"]
