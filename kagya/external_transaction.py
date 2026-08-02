"""Staged saga coordination for non-authoritative external artifact stores."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ExternalTransactionStatus(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"
    ORPHANED = "orphaned"
    COMPENSATED = "compensated"


class ExternalTransactionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    revision: int = Field(ge=1)
    status: ExternalTransactionStatus
    timestamp: str
    reason: str


class ExternalTransactionRecord(BaseModel):
    """Privacy-safe lifecycle record; artifact payloads are deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    revision: int = Field(ge=1)
    transaction_id: str
    artifact_type: str
    artifact_id: str
    status: ExternalTransactionStatus
    event_id: str | None = None
    processing_sequence: int | None = Field(default=None, ge=0)
    source: str
    causation_id: str | None = None
    correlation_id: str | None = None
    created_at: str
    updated_at: str
    audit: list[ExternalTransactionAudit]


class ExternalRestoreEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    artifact_type: str
    artifact_id: str
    event_id: str
    processing_sequence: int = Field(ge=0)
    status: ExternalTransactionStatus
    restore_action: str = "retained_not_replayed"


class ExternalRestoreDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_sequence: int = Field(ge=0)
    effects: list[ExternalRestoreEffect]
    external_side_effects_replayed: bool = False


class ExternalReconciliationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finalized: int = 0
    compensated: int = 0
    retryable: int = 0


class ExternalArtifactStore(Protocol):
    """Contract implemented by Chroma now and adapter/dataset stores later."""

    def list_external_transactions(self) -> list[ExternalTransactionRecord]: ...

    def finalize_external_event(
        self, event_id: str, processing_sequence: int
    ) -> int: ...

    def orphan_external_event(self, event_id: str, reason: str) -> int: ...

    def compensate_external_event(self, event_id: str, reason: str) -> int: ...


class JournalOutcome(Protocol):
    event_id: str
    lifecycle: Any
    processing_sequence: int | None
    failure_category: str | None


class ExternalTransactionCoordinator:
    def __init__(self, stores: list[ExternalArtifactStore]) -> None:
        self._stores = list(stores)
        self._last_reconciliation = ExternalReconciliationReport()

    @property
    def last_reconciliation(self) -> ExternalReconciliationReport:
        return self._last_reconciliation.model_copy()

    def records(self) -> list[ExternalTransactionRecord]:
        records = [
            record
            for store in self._stores
            for record in store.list_external_transactions()
        ]
        return sorted(
            records,
            key=lambda record: (
                record.processing_sequence
                if record.processing_sequence is not None
                else -1,
                record.transaction_id,
            ),
        )

    def finalize_event(
        self, event_id: str, processing_sequence: int, *, attempts: int = 3
    ) -> bool:
        attempts = max(1, attempts)
        for _ in range(attempts):
            try:
                for store in self._stores:
                    store.finalize_external_event(event_id, processing_sequence)
            except Exception:
                continue
            return True
        return False

    def orphan_event(self, event_id: str, reason: str) -> int:
        return sum(
            store.orphan_external_event(event_id, reason) for store in self._stores
        )

    def compensate_event(self, event_id: str, reason: str) -> int:
        changed = 0
        for store in self._stores:
            orphaned = store.orphan_external_event(event_id, reason)
            compensated = store.compensate_external_event(event_id, reason)
            changed += max(orphaned, compensated)
        return changed

    def reconcile(
        self, journal_records: list[JournalOutcome]
    ) -> ExternalReconciliationReport:
        latest: dict[str, JournalOutcome] = {}
        for record in journal_records:
            if str(record.lifecycle) not in {"accepted", "checkpoint", "audit"}:
                latest[record.event_id] = record

        finalized = compensated = retryable = 0
        pending_event_ids = {
            record.event_id
            for record in self.records()
            if record.status
            in {
                ExternalTransactionStatus.PENDING,
                ExternalTransactionStatus.ORPHANED,
            }
            and record.event_id is not None
        }
        for event_id in sorted(pending_event_ids):
            outcome = latest.get(event_id)
            if outcome is None:
                compensated += self.compensate_event(
                    event_id, "journal_commit_evidence_missing"
                )
                continue
            lifecycle = str(outcome.lifecycle)
            committed = lifecycle == "completed" or (
                lifecycle == "recovery_classified"
                and outcome.failure_category == "committed_before_crash"
            )
            event_statuses = {
                record.status
                for record in self.records()
                if record.event_id == event_id
            }
            if ExternalTransactionStatus.ORPHANED in event_statuses:
                compensated += self.compensate_event(
                    event_id, "failure_intent_recovered"
                )
                continue
            if committed and outcome.processing_sequence is not None:
                if self.finalize_event(event_id, outcome.processing_sequence):
                    finalized += 1
                else:
                    retryable += 1
                continue
            failed = lifecycle == "failed" or (
                lifecycle == "recovery_classified"
                and outcome.failure_category != "committed_before_crash"
            )
            if failed:
                compensated += self.compensate_event(
                    event_id, outcome.failure_category or lifecycle
                )
            else:
                retryable += 1
        report = ExternalReconciliationReport(
            finalized=finalized,
            compensated=compensated,
            retryable=retryable,
        )
        self._last_reconciliation = report
        return report

    def restore_diff(self, target_sequence: int) -> ExternalRestoreDiff:
        _records, diff, _reconciliation = self.restore_view(target_sequence)
        return diff

    def restore_view(
        self, target_sequence: int
    ) -> tuple[
        list[ExternalTransactionRecord],
        ExternalRestoreDiff,
        ExternalReconciliationReport,
    ]:
        """Capture one immutable record set for a restore consistency decision."""

        records = self.records()
        effects = [
            ExternalRestoreEffect(
                transaction_id=record.transaction_id,
                artifact_type=record.artifact_type,
                artifact_id=record.artifact_id,
                event_id=record.event_id,
                processing_sequence=record.processing_sequence,
                status=record.status,
            )
            for record in records
            if record.event_id is not None
            and record.processing_sequence is not None
            and record.processing_sequence > target_sequence
            and record.status == ExternalTransactionStatus.COMMITTED
        ]
        return (
            records,
            ExternalRestoreDiff(target_sequence=target_sequence, effects=effects),
            self.last_reconciliation,
        )
