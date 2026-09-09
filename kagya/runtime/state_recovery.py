"""Cross-authority startup recovery without transferring authority ownership."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import stat
from threading import RLock
from uuid import UUID, uuid4

from kagya.runtime.agent_runtime import AgentEvent
from kagya.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSnapshot,
    AgentStateStore,
)
from kagya.runtime.event_journal import (
    EventJournalGateClear,
    EventFailureCategory,
    EventJournal,
    EventJournalInspection,
    EventJournalParticipantBaseline,
    EventJournalStartupReconciliation,
    EventJournalTransaction,
    EventLifecycle,
    EventRecoveryCategory,
    startup_participant_aggregate_digest,
)
from kagya.runtime.state_wal import (
    BaselineRecord,
    Manifest,
    RecoveryReason,
    StateWAL,
    StateWALError,
    StateWALInspection,
    TransitionRecord,
)


class StateRecoveryError(Exception):
    """Cross-authority evidence cannot select one safe internal state."""


class InternalCommitClassification(str, Enum):
    PRE_INTERNAL = "PRE_INTERNAL"
    INTERNALLY_COMMITTED = "INTERNALLY_COMMITTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class StateRecoveryResult:
    snapshot: AgentStateSnapshot
    snapshot_hash: str
    processing_high_water: int
    manifest: Manifest
    external_reconciliation_required: bool
    exact_current_reconstructed: bool = False
    true_rollback_performed: bool = False


@dataclass(frozen=True, slots=True)
class InternalCommitEvidence:
    event_id: str
    processing_sequence: int
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str


@dataclass(frozen=True, slots=True)
class InternalCommitProof:
    """Read-only, bounded evidence for startup transaction classification."""

    classification: InternalCommitClassification
    transaction_id: str
    event_id: str
    processing_sequence: int
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str | None = None
    wal_record_id: str | None = None
    wal_record_hash: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryGateClearProof:
    """The immutable Journal proof consumed by the gate-clear operation."""

    startup_reconciliation: EventJournalStartupReconciliation
    baseline: EventJournalParticipantBaseline
    clear_prepared: EventJournalGateClear


class StateRecoveryCoordinator:
    """Verify and coordinate Journal, WAL, and snapshot boundaries."""

    def __init__(
        self,
        state_store: AgentStateStore,
        journal: EventJournal,
        wal: StateWAL,
    ) -> None:
        self.state_store = state_store
        self.journal = journal
        self.wal = wal
        self._lock = RLock()

    def classify_transaction_commit(
        self, transaction: EventJournalTransaction
    ) -> InternalCommitProof:
        """Classify internal publication without appending or mutating state."""
        base = InternalCommitProof(
            classification=InternalCommitClassification.AMBIGUOUS,
            transaction_id=transaction.transaction_id,
            event_id=transaction.event_id,
            processing_sequence=transaction.processing_sequence,
            snapshot_sequence=transaction.processing_sequence,
            snapshot_hash="0" * 64,
        )
        try:
            journal = self.journal.inspect()
            known_transactions = (
                *journal.open_transactions,
                *journal.completed_transactions,
                *journal.reconciled_transactions,
                *journal.reconciliation_required_transactions,
                *journal.abort_required_transactions,
                *journal.aborted_transactions,
            )
            if not any(item == transaction for item in known_transactions):
                return base
            prepared = [
                record for record in journal.records
                if record.lifecycle is EventLifecycle.PREPARED
                and record.event_id == transaction.event_id
                and record.processing_sequence == transaction.processing_sequence
            ]
            if len(prepared) > 1:
                return base
            canonical = self.state_store.load()
            canonical_hash = self.state_store.snapshot_hash(canonical)
            wal = self.wal.inspect()
            event_transitions = [
                record
                for record in wal.records
                if isinstance(record, TransitionRecord)
                and (
                    str(record.event_id) == transaction.event_id
                    or record.processing_sequence == transaction.processing_sequence
                )
            ]
            if not prepared:
                if (
                    not event_transitions
                    and journal.snapshot_sequence
                    == canonical.last_processed_event_sequence
                    and journal.snapshot_hash == canonical_hash
                    and wal.latest_snapshot_sequence
                    == canonical.last_processed_event_sequence
                    and wal.latest_snapshot_hash == canonical_hash
                    and canonical.last_processed_event_sequence
                    < transaction.processing_sequence
                ):
                    return InternalCommitProof(
                        InternalCommitClassification.PRE_INTERNAL,
                        transaction.transaction_id,
                        transaction.event_id,
                        transaction.processing_sequence,
                        canonical.last_processed_event_sequence,
                        canonical_hash,
                        str(wal.active_manifest.active_generation_id)
                        if wal.active_manifest is not None
                        else None,
                    )
                return base
            preparation = prepared[0]
            if (
                preparation.event_type is not transaction.event_type
                or preparation.source is not transaction.source
                or preparation.wal_generation_id is None
                or preparation.state_hash_after is None
            ):
                return base
            manifest = wal.active_manifest
            if manifest is None or str(manifest.active_generation_id) != preparation.wal_generation_id:
                return base
            transitions = [
                (record, record_hash) for record, record_hash in
                zip(wal.records, wal.record_hashes)
                if isinstance(record, TransitionRecord)
                and str(record.event_id) == transaction.event_id
                and record.processing_sequence == transaction.processing_sequence
                and record.candidate_snapshot_hash == preparation.state_hash_after
                and record.prior_snapshot_hash == preparation.state_hash_before
                and str(record.generation_id) == preparation.wal_generation_id
                and record.event_type == transaction.event_type.value
                and record.event_source == transaction.source.value
            ]
            if len(event_transitions) != len(transitions):
                return base
            if len(transitions) > 1:
                return base
            if transitions:
                transition, record_hash = transitions[0]
                if wal.records[-1] != transition:
                    return base
                committed = (
                    canonical.last_processed_event_sequence == transition.candidate_snapshot_sequence
                    and canonical_hash == transition.candidate_snapshot_hash
                )
                if not committed:
                    return base
                return InternalCommitProof(
                    InternalCommitClassification.INTERNALLY_COMMITTED,
                    transaction.transaction_id, transaction.event_id,
                    transaction.processing_sequence,
                    transition.candidate_snapshot_sequence,
                    transition.candidate_snapshot_hash,
                    preparation.wal_generation_id, str(transition.record_id), record_hash,
                )
            if canonical_hash == preparation.state_hash_after:
                return base
            if canonical_hash != preparation.state_hash_before:
                return base
            if (
                wal.latest_snapshot_sequence
                != canonical.last_processed_event_sequence
                or wal.latest_snapshot_hash != canonical_hash
            ):
                return base
            return InternalCommitProof(
                InternalCommitClassification.PRE_INTERNAL,
                transaction.transaction_id, transaction.event_id,
                transaction.processing_sequence,
                canonical.last_processed_event_sequence, canonical_hash,
                preparation.wal_generation_id,
            )
        except Exception:
            return base

    # Names used by startup coordinators are intentionally aliases, not new
    # authority surfaces.
    inspect_transaction_commit = classify_transaction_commit

    def clear_recovery_gate(
        self,
        result: StateRecoveryResult,
        startup_reconciliation: EventJournalStartupReconciliation,
        baseline: EventJournalParticipantBaseline,
        clear_prepared: EventJournalGateClear,
    ) -> StateRecoveryResult:
        """Clear external reconciliation only after re-verifying all proof."""
        with self._lock:
            journal = self.journal.inspect()
            current = next(
                (item for item in journal.completed_startup_reconciliations
                 if item.reconciliation_id == startup_reconciliation.reconciliation_id), None
            )
            current_baseline = next(
                (item for item in journal.baselines if item.baseline_id == baseline.baseline_id), None
            )
            current_clear = journal.open_gate_clear or journal.terminal_gate_clear
            if current != startup_reconciliation or current_baseline != baseline or current_clear != clear_prepared:
                raise StateRecoveryError("recovery gate proof is stale")
            if (
                startup_reconciliation.snapshot_sequence != baseline.snapshot_sequence
                or startup_reconciliation.snapshot_hash != baseline.snapshot_hash
                or startup_reconciliation.recovery_processing_high_water
                != baseline.processing_high_water
                or startup_reconciliation.wal_generation_id != baseline.wal_generation_id
                or startup_reconciliation.wal_record_id != baseline.wal_record_id
                or startup_reconciliation.wal_record_hash != baseline.wal_record_hash
                or startup_reconciliation.journal_lineage_id
                != baseline.journal_lineage_id
            ):
                raise StateRecoveryError("startup reconciliation binding changed")
            if not startup_reconciliation.completed or not startup_reconciliation.required_participants:
                raise StateRecoveryError("startup reconciliation is incomplete")
            required = {item.participant_id for item in startup_reconciliation.required_participants}
            outcomes = {item[0] for item in startup_reconciliation.participant_outcomes}
            if outcomes != required or len(outcomes) != len(startup_reconciliation.participant_outcomes):
                raise StateRecoveryError("startup participant evidence is incomplete")
            if not baseline.participant_registry:
                raise StateRecoveryError("participant baseline is incomplete")
            baseline_participants = {
                item.participant_id for item in baseline.participant_registry
            }
            if required != baseline_participants:
                raise StateRecoveryError("startup participant registry changed")
            covered_transactions = tuple(
                sorted(
                    (
                        *journal.completed_transactions,
                        *journal.reconciled_transactions,
                    ),
                    key=lambda item: item.transaction_id,
                )
            )
            if any(
                item.processing_sequence
                > startup_reconciliation.recovery_processing_high_water
                for item in covered_transactions
            ):
                raise StateRecoveryError("startup transaction coverage is stale")
            expected_digests = {
                participant_id: startup_participant_aggregate_digest(
                    covered_transactions, participant_id
                )
                for participant_id in baseline_participants
            }
            if {
                item.participant_id: item.operation_digest
                for item in startup_reconciliation.required_participants
            } != expected_digests:
                raise StateRecoveryError("startup transaction coverage changed")
            completed = [
                item for item in journal.records
                if item.lifecycle is EventLifecycle.RECOVERY_COMPLETED
                and item.recovery_id == startup_reconciliation.recovery_id
            ]
            if (
                len(completed) != 1
                or completed[0].recovery_category
                is not EventRecoveryCategory.TRUE_ROLLBACK
                or completed[0].external_reconciliation_required is not True
            ):
                raise StateRecoveryError("true rollback completion proof is required")
            proof = completed[0]
            expected = (baseline.snapshot_sequence, baseline.snapshot_hash,
                        baseline.processing_high_water, baseline.wal_generation_id,
                        baseline.wal_record_id, baseline.wal_record_hash,
                        baseline.journal_lineage_id)
            if (proof.snapshot_sequence, proof.snapshot_hash,
                proof.recovery_processing_high_water, proof.wal_generation_id,
                proof.wal_record_id, proof.wal_record_hash,
                journal.journal_lineage_id) != expected:
                raise StateRecoveryError("recovery proof bindings changed")
            clear_binding = (
                clear_prepared.snapshot_sequence,
                clear_prepared.snapshot_hash,
                clear_prepared.processing_high_water,
                clear_prepared.wal_generation_id,
                clear_prepared.wal_record_id,
                clear_prepared.wal_record_hash,
                clear_prepared.journal_lineage_id,
            )
            if clear_binding != expected or clear_prepared.baseline_id != baseline.baseline_id:
                raise StateRecoveryError("clear-prepared proof bindings changed")
            snapshot = self.state_store.load()
            if (snapshot.last_processed_event_sequence != baseline.snapshot_sequence
                    or self.state_store.snapshot_hash(snapshot) != baseline.snapshot_hash
                    or result.processing_high_water != baseline.processing_high_water
                    or result.manifest.active_generation_id != UUID(baseline.wal_generation_id)):
                raise StateRecoveryError("canonical recovery baseline changed")
            inspection = self.wal.inspect()
            manifest = inspection.active_manifest
            if (manifest is None or manifest != result.manifest
                    or inspection.latest_snapshot_sequence != baseline.snapshot_sequence
                    or inspection.latest_snapshot_hash != baseline.snapshot_hash):
                raise StateRecoveryError("WAL recovery baseline changed")
            cleared = self.wal.clear_external_reconciliation_gate(manifest)
            return StateRecoveryResult(snapshot, baseline.snapshot_hash,
                baseline.processing_high_water, cleared, False,
                result.exact_current_reconstructed, result.true_rollback_performed)

    clear_external_reconciliation_gate = clear_recovery_gate
    clear_prepared_recovery_gate = clear_recovery_gate
    clear_external_reconciliation = clear_recovery_gate

    def prepare_startup(self) -> StateRecoveryResult:
        """Inspect every authority before applying migration or recovery writes."""

        with self._lock:
            return self._prepare_startup()

    def _prepare_startup(self) -> StateRecoveryResult:

        journal_inspection = self.journal.inspect()
        snapshot, snapshot_error = self._inspect_snapshot()
        try:
            wal_inspection = self.wal.inspect_optional()
            wal_error: StateWALError | None = None
        except StateWALError as error:
            wal_inspection = None
            wal_error = error
        try:
            anchor = self.wal.inspect_boot_anchor_optional()
        except StateWALError:
            raise StateRecoveryError("StateWAL boot anchor is invalid") from None

        if wal_inspection is None:
            if journal_inspection.open_recoveries:
                if len(journal_inspection.open_recoveries) != 1:
                    raise StateRecoveryError("Journal recovery lifecycle is ambiguous")
                pending = journal_inspection.open_recoveries[0]
                if (
                    snapshot is not None
                    and pending.snapshot_sequence
                    == snapshot.last_processed_event_sequence
                    and pending.snapshot_hash
                    == self.state_store.snapshot_hash(snapshot)
                ):
                    return self._finish_event_reconciliation(
                        self._resume_invalid_current_recovery(
                            journal_inspection, snapshot, snapshot
                        )
                    )
                bound_resume = self._resume_journal_bound_open_recovery(
                    journal_inspection
                )
                if bound_resume is not None:
                    return bound_resume
            if not journal_inspection.open_recoveries and snapshot is None:
                bound_current = self._recover_journal_bound_current(journal_inspection)
                if bound_current is not None:
                    return bound_current
            if snapshot is not None and (journal_inspection.schema_version or 0) >= 2:
                return self._repair_v2_invalid_current(snapshot)
            if (
                snapshot is not None
                and anchor is None
                and journal_inspection.schema_version in {None, 1}
            ):
                return self._replace_unanchored_provisional(
                    journal_inspection, snapshot
                )
            if anchor is None:
                raise StateRecoveryError("StateWAL is invalid and unanchored") from None
            return self._recover_from_anchor(
                journal_inspection,
                snapshot,
                anchor_expected=True,
                wal_error=wal_error,
            )
        if not wal_inspection.exists:
            if journal_inspection.open_recoveries:
                if len(journal_inspection.open_recoveries) != 1:
                    raise StateRecoveryError("Journal recovery lifecycle is ambiguous")
                pending = journal_inspection.open_recoveries[0]
                if (
                    snapshot is not None
                    and pending.snapshot_sequence
                    == snapshot.last_processed_event_sequence
                    and pending.snapshot_hash
                    == self.state_store.snapshot_hash(snapshot)
                ):
                    return self._finish_event_reconciliation(
                        self._resume_invalid_current_recovery(
                            journal_inspection, snapshot, snapshot
                        )
                    )
                bound_resume = self._resume_journal_bound_open_recovery(
                    journal_inspection
                )
                if bound_resume is not None:
                    return bound_resume
                if anchor is not None:
                    return self._recover_from_anchor(
                        journal_inspection,
                        snapshot,
                        anchor_expected=True,
                        wal_error=None,
                    )
                raise StateRecoveryError("Prepared recovery target is unavailable")
            if snapshot is None:
                bound_current = self._recover_journal_bound_current(journal_inspection)
                if bound_current is not None:
                    return bound_current
            if (journal_inspection.schema_version or 0) >= 2:
                if snapshot is not None:
                    return self._repair_v2_invalid_current(snapshot)
                if anchor is not None:
                    return self._recover_from_anchor(
                        journal_inspection,
                        snapshot,
                        anchor_expected=True,
                        wal_error=None,
                    )
                raise StateRecoveryError("StateWAL is missing and unanchored")
            return self._bootstrap_r06(journal_inspection, snapshot, snapshot_error)

        if journal_inspection.schema_version in {None, 1}:
            reconstructed = False
            baseline = (
                wal_inspection.records[0]
                if len(wal_inspection.records) == 1
                and isinstance(wal_inspection.records[0], BaselineRecord)
                else None
            )
            if journal_inspection.schema_version is None and (
                baseline is None
                or baseline.reason is not RecoveryReason.BOOTSTRAP
                or baseline.journal_processing_high_water
                != baseline.baseline_snapshot_sequence
                or wal_inspection.active_manifest is None
                or wal_inspection.active_manifest.external_reconciliation_required
            ):
                raise StateRecoveryError(
                    "partial bootstrap WAL baseline is inconsistent"
                ) from None
            if snapshot is None:
                try:
                    snapshot = self.wal.reconstruct(
                        sequence=(
                            journal_inspection.snapshot_sequence
                            if journal_inspection.schema_version == 1
                            else baseline.baseline_snapshot_sequence
                            if baseline is not None
                            else None
                        ),
                        snapshot_hash=(
                            journal_inspection.snapshot_hash
                            if journal_inspection.schema_version == 1
                            else baseline.baseline_snapshot_hash
                            if baseline is not None
                            else None
                        ),
                    )
                except StateWALError:
                    raise StateRecoveryError(
                        "R05 migration state cannot be reconstructed"
                    ) from None
                reconstructed = True
            return self._finish_v1_migration(
                journal_inspection,
                wal_inspection,
                snapshot,
                reconstructed=reconstructed,
            )

        self._validate_cross_authority(journal_inspection, wal_inspection)
        if journal_inspection.open_recoveries:
            result = self._resume_recovery(journal_inspection, wal_inspection, snapshot)
            return self._finish_event_reconciliation(result)

        if snapshot is not None:
            classified = self.journal.inspect(
                snapshot.last_processed_event_sequence,
                self.state_store.snapshot_hash(snapshot),
            )
            self._validate_cross_authority(classified, wal_inspection)
            recovery = self.journal.apply_planned_reconciliation(
                snapshot.last_processed_event_sequence,
                self.state_store.snapshot_hash(snapshot),
            )
            reconciled = self.journal.inspect()
            if (
                recovery.snapshot_sequence != snapshot.last_processed_event_sequence
                or recovery.snapshot_hash != self.state_store.snapshot_hash(snapshot)
            ):
                raise StateRecoveryError(
                    "Journal reconciliation selected another snapshot"
                )
            if self._wal_latest_matches(wal_inspection, snapshot):
                if (
                    reconciled.wal_snapshot_sequence
                    != snapshot.last_processed_event_sequence
                    or reconciled.wal_snapshot_hash
                    != self.state_store.snapshot_hash(snapshot)
                ):
                    wal_record, wal_record_hash = self._record_for_snapshot(
                        wal_inspection,
                        snapshot,
                        self.state_store.snapshot_hash(snapshot),
                    )
                    assert wal_inspection.active_manifest is not None
                    self.journal.append_v2_current_checkpoint(
                        snapshot.last_processed_event_sequence,
                        self.state_store.snapshot_hash(snapshot),
                        str(wal_inspection.active_manifest.active_generation_id),
                        str(wal_record.record_id),
                        wal_record_hash,
                    )
                return self._result(snapshot, recovery.processing_high_water)
            return self._start_recovery(
                reconciled,
                wal_inspection,
                snapshot,
                EventRecoveryCategory.UNCOMMITTED_TAIL,
                RecoveryReason.UNCOMMITTED_TAIL,
                external=False,
            )

        target_sequence = journal_inspection.snapshot_sequence
        target_hash = journal_inspection.snapshot_hash
        try:
            target = self.wal.reconstruct(
                sequence=target_sequence, snapshot_hash=target_hash
            )
        except StateWALError:
            return self._recover_from_anchor(
                journal_inspection,
                snapshot,
                anchor_expected=anchor is not None,
                wal_error=None,
            )
        category = (
            EventRecoveryCategory.EXACT_CURRENT
            if self._wal_latest_matches(wal_inspection, target)
            else EventRecoveryCategory.UNCOMMITTED_TAIL
        )
        reason = (
            RecoveryReason.EXACT_CURRENT_REPAIR
            if category is EventRecoveryCategory.EXACT_CURRENT
            else RecoveryReason.UNCOMMITTED_TAIL
        )
        result = self._start_recovery(
            journal_inspection,
            wal_inspection,
            target,
            category,
            reason,
            external=journal_inspection.external_reconciliation_required,
        )
        return self._finish_event_reconciliation(result)

    def _recover_journal_bound_current(
        self,
        journal: EventJournalInspection,
    ) -> StateRecoveryResult | None:
        target = self._inspect_journal_bound_current(journal)
        if target is None:
            return None
        result = self._start_recovery(
            journal,
            None,
            target,
            EventRecoveryCategory.EXACT_CURRENT,
            RecoveryReason.EXACT_CURRENT_REPAIR,
            external=journal.external_reconciliation_required,
            invalid_current=True,
        )
        return self._finish_event_reconciliation(result)

    def _resume_journal_bound_open_recovery(
        self,
        journal: EventJournalInspection,
    ) -> StateRecoveryResult | None:
        if len(journal.open_recoveries) != 1:
            return None
        pending = journal.open_recoveries[0]
        if pending.category not in {
            EventRecoveryCategory.EXACT_CURRENT,
            EventRecoveryCategory.UNCOMMITTED_TAIL,
        }:
            return None
        target = self._inspect_journal_bound_current(journal)
        if (
            target is None
            or target.last_processed_event_sequence != pending.snapshot_sequence
            or self.state_store.snapshot_hash(target) != pending.snapshot_hash
        ):
            return None
        result = self._resume_invalid_current_recovery(journal, target, None)
        return self._finish_event_reconciliation(result)

    def _inspect_journal_bound_current(
        self,
        journal: EventJournalInspection,
    ) -> AgentStateSnapshot | None:
        if (
            (journal.schema_version or 0) < 2
            or journal.wal_generation_id is None
            or journal.wal_record_id is None
            or journal.wal_record_hash is None
            or journal.wal_snapshot_sequence != journal.snapshot_sequence
            or journal.wal_snapshot_hash != journal.snapshot_hash
        ):
            return None
        try:
            target = self.wal.inspect_bound_prefix(
                generation_id=UUID(journal.wal_generation_id),
                record_id=UUID(journal.wal_record_id),
                record_hash=journal.wal_record_hash,
                snapshot_sequence=journal.snapshot_sequence,
                snapshot_hash=journal.snapshot_hash,
            )
        except (StateWALError, ValueError):
            return None
        return target

    def _repair_v2_invalid_current(
        self,
        snapshot: AgentStateSnapshot,
    ) -> StateRecoveryResult:
        snapshot_hash = self.state_store.snapshot_hash(snapshot)
        self.journal.inspect(snapshot.last_processed_event_sequence, snapshot_hash)
        recovery = self.journal.apply_planned_reconciliation(
            snapshot.last_processed_event_sequence, snapshot_hash
        )
        reconciled = self.journal.inspect()
        if (
            recovery.snapshot_sequence != snapshot.last_processed_event_sequence
            or recovery.snapshot_hash != snapshot_hash
            or reconciled.snapshot_sequence != snapshot.last_processed_event_sequence
            or reconciled.snapshot_hash != snapshot_hash
        ):
            raise StateRecoveryError("Journal reconciliation selected another snapshot")
        return self._start_recovery(
            reconciled,
            None,
            snapshot,
            EventRecoveryCategory.EXACT_CURRENT,
            RecoveryReason.EXACT_CURRENT_REPAIR,
            external=reconciled.external_reconciliation_required,
            invalid_current=True,
        )

    def _replace_unanchored_provisional(
        self,
        journal: EventJournalInspection,
        snapshot: AgentStateSnapshot,
    ) -> StateRecoveryResult:
        """Finish bootstrap/migration when no v2 evidence anchors provisional WAL."""

        snapshot_hash = self.state_store.snapshot_hash(snapshot)
        if journal.schema_version is None:
            if journal.records or snapshot.last_processed_event_sequence != 0:
                raise StateRecoveryError(
                    "provisional bootstrap state is not authoritative"
                )
            high_water = 0
        else:
            self.journal.inspect(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
            )
            recovery = self.journal.apply_planned_reconciliation(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
            )
            if (
                recovery.snapshot_sequence != snapshot.last_processed_event_sequence
                or recovery.snapshot_hash != snapshot_hash
            ):
                raise StateRecoveryError("R05 reconciliation selected another snapshot")
            high_water = recovery.processing_high_water
            journal = self.journal.inspect()
        manifest = self.wal.replace_unanchored_provisional(snapshot, high_water)
        inspection = self.wal.inspect()
        if (
            inspection.baseline_record_id is None
            or inspection.baseline_record_hash is None
        ):
            raise StateRecoveryError("replacement WAL baseline is incomplete")
        self.state_store.ensure_published(snapshot)
        if journal.schema_version == 1:
            self.journal.append_v2_migration_checkpoint(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
                str(manifest.active_generation_id),
                str(inspection.baseline_record_id),
                inspection.baseline_record_hash,
            )
        else:
            self.journal.append_v2_bootstrap_checkpoint(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
                str(manifest.active_generation_id),
                str(inspection.baseline_record_id),
                inspection.baseline_record_hash,
                processing_high_water=high_water,
            )
        return self._result(snapshot, high_water)

    def commit_internal_candidate(
        self,
        event: AgentEvent,
        prior_snapshot: AgentStateSnapshot,
        candidate_snapshot: AgentStateSnapshot,
    ) -> InternalCommitEvidence:
        """Publish internal state without claiming overall event completion."""

        with self._lock:
            return self._commit_internal_candidate(
                event, prior_snapshot, candidate_snapshot
            )

    def _commit_internal_candidate(
        self,
        event: AgentEvent,
        prior_snapshot: AgentStateSnapshot,
        candidate_snapshot: AgentStateSnapshot,
    ) -> InternalCommitEvidence:

        sequence = event.processing_sequence
        if sequence is None:
            raise StateRecoveryError("State transition event has no sequence")
        inspection = self.wal.inspect()
        manifest = inspection.active_manifest
        if manifest is None or manifest.external_reconciliation_required:
            raise StateRecoveryError("Authoritative mutation is recovery-gated")
        before_hash = self.state_store.snapshot_hash(prior_snapshot)
        after_hash = self.state_store.snapshot_hash(candidate_snapshot)
        generation_id = str(manifest.active_generation_id)
        self.journal.append_prepared(
            event,
            before_hash,
            after_hash,
            wal_generation_id=generation_id,
        )
        transition = self.wal.append_transition(
            event_id=UUID(event.event_id),
            event_type=event.event_type.value,
            event_source=event.source.value,
            processing_sequence=sequence,
            prior_snapshot=prior_snapshot,
            candidate_snapshot=candidate_snapshot,
        )
        self.state_store.save(candidate_snapshot)
        return InternalCommitEvidence(
            event_id=event.event_id,
            processing_sequence=sequence,
            snapshot_sequence=sequence,
            snapshot_hash=after_hash,
            wal_generation_id=generation_id,
            wal_record_id=str(transition.record_id),
            wal_record_hash=transition.record_hash,
        )

    def complete_committed_event(
        self,
        event: AgentEvent,
        evidence: InternalCommitEvidence,
    ) -> None:
        """Verify internal commit evidence before appending overall completion."""

        with self._lock:
            self._complete_committed_event(event, evidence)

    def verify_internal_commit(
        self,
        event: AgentEvent,
        evidence: InternalCommitEvidence,
    ) -> None:
        """Verify that canonical state, Journal, and WAL still bind this commit."""

        with self._lock:
            self._verify_internal_commit(event, evidence)

    def _complete_committed_event(
        self,
        event: AgentEvent,
        evidence: InternalCommitEvidence,
    ) -> None:
        self._verify_internal_commit(event, evidence)
        self.journal.append_completed(
            event,
            evidence.snapshot_sequence,
            evidence.snapshot_hash,
            wal_generation_id=evidence.wal_generation_id,
            wal_record_id=evidence.wal_record_id,
            wal_record_hash=evidence.wal_record_hash,
        )

    def _verify_internal_commit(
        self,
        event: AgentEvent,
        evidence: InternalCommitEvidence,
    ) -> None:
        sequence = event.processing_sequence
        if (
            sequence is None
            or evidence.event_id != event.event_id
            or evidence.processing_sequence != sequence
            or evidence.snapshot_sequence != sequence
        ):
            raise StateRecoveryError("Internal commit event identity is invalid")

        canonical = self.state_store.load()
        if (
            canonical.last_processed_event_sequence != evidence.snapshot_sequence
            or self.state_store.snapshot_hash(canonical) != evidence.snapshot_hash
        ):
            raise StateRecoveryError("Canonical internal commit evidence is stale")

        journal = self.journal.inspect()
        prepared = next(
            (
                record
                for record in reversed(journal.records)
                if record.lifecycle is EventLifecycle.PREPARED
                and record.event_id == evidence.event_id
            ),
            None,
        )
        if (
            prepared is None
            or prepared.event_type is not event.event_type
            or prepared.source is not event.source
            or prepared.processing_sequence != evidence.processing_sequence
            or prepared.state_hash_after != evidence.snapshot_hash
            or prepared.wal_generation_id != evidence.wal_generation_id
        ):
            raise StateRecoveryError("Journal internal commit evidence is stale")

        wal = self.wal.inspect()
        manifest = wal.active_manifest
        transition = next(
            (
                record
                for record in reversed(wal.records)
                if isinstance(record, TransitionRecord)
                and str(record.record_id) == evidence.wal_record_id
            ),
            None,
        )
        if (
            manifest is None
            or manifest.external_reconciliation_required
            or str(manifest.active_generation_id) != evidence.wal_generation_id
            or wal.latest_snapshot_sequence != evidence.snapshot_sequence
            or wal.latest_snapshot_hash != evidence.snapshot_hash
            or transition is None
            or not wal.records
            or wal.records[-1] != transition
            or transition.record_hash != evidence.wal_record_hash
            or str(transition.event_id) != event.event_id
            or transition.event_type != event.event_type.value
            or transition.event_source != event.source.value
            or transition.processing_sequence != evidence.processing_sequence
            or transition.candidate_snapshot_sequence != evidence.snapshot_sequence
            or transition.candidate_snapshot_hash != evidence.snapshot_hash
        ):
            raise StateRecoveryError("StateWAL internal commit evidence is stale")

    def publish_boot_anchor(self, result: StateRecoveryResult) -> None:
        """Mark bootability only after the runtime graph has started."""

        with self._lock:
            self._publish_boot_anchor(result)

    def _publish_boot_anchor(self, result: StateRecoveryResult) -> None:

        if result.external_reconciliation_required:
            return
        inspection = self.wal.inspect()
        journal = self.journal.inspect()
        manifest = inspection.active_manifest
        if manifest is None or not inspection.records or not journal.records:
            raise StateRecoveryError("Boot anchor authority is incomplete")
        if journal.journal_lineage_id is None:
            raise StateRecoveryError("Journal lineage authority is incomplete")
        if (
            manifest != result.manifest
            or journal.snapshot_sequence
            != result.snapshot.last_processed_event_sequence
            or journal.snapshot_hash != result.snapshot_hash
            or journal.processing_high_water != result.processing_high_water
        ):
            raise StateRecoveryError("Boot anchor result is stale")
        record, record_hash = self._record_for_snapshot(
            inspection, result.snapshot, result.snapshot_hash
        )
        self.wal.publish_boot_anchor(
            snapshot_sequence=result.snapshot.last_processed_event_sequence,
            snapshot_hash=result.snapshot_hash,
            generation_id=manifest.active_generation_id,
            anchored_record_id=record.record_id,
            anchored_record_hash=record_hash,
            journal_processing_high_water=result.processing_high_water,
            journal_tail_record_id=UUID(journal.tail_record_id or ""),
            journal_tail_record_hash=journal.tail_record_hash,
            journal_lineage_id=UUID(journal.journal_lineage_id),
        )

    def _bootstrap_r06(
        self,
        journal: EventJournalInspection,
        snapshot: AgentStateSnapshot | None,
        snapshot_error: AgentStateLoadError | None,
    ) -> StateRecoveryResult:
        if snapshot is None:
            if snapshot_error is not None:
                raise snapshot_error
            if journal.records:
                raise StateRecoveryError(
                    "StateWAL cannot bootstrap inconsistent current state"
                ) from None
            snapshot = self.state_store.load()
        snapshot_hash = self.state_store.snapshot_hash(snapshot)
        if journal.records:
            recovery = self.journal.apply_planned_reconciliation(
                snapshot.last_processed_event_sequence, snapshot_hash
            )
        else:
            self.state_store.ensure_published(snapshot)
            recovery = None
        high_water = (
            recovery.processing_high_water
            if recovery is not None
            else snapshot.last_processed_event_sequence
        )
        manifest = self.wal.bootstrap(snapshot, high_water)
        inspection = self.wal.inspect()
        assert inspection.baseline_record_id is not None
        assert inspection.baseline_record_hash is not None
        self.state_store.ensure_published(snapshot)
        if journal.records:
            self.journal.append_v2_migration_checkpoint(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
                str(manifest.active_generation_id),
                str(inspection.baseline_record_id),
                inspection.baseline_record_hash,
            )
        else:
            self.journal.append_v2_bootstrap_checkpoint(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
                str(manifest.active_generation_id),
                str(inspection.baseline_record_id),
                inspection.baseline_record_hash,
                processing_high_water=high_water,
            )
        return self._result(snapshot, high_water)

    def _finish_v1_migration(
        self,
        journal: EventJournalInspection,
        wal: StateWALInspection,
        snapshot: AgentStateSnapshot,
        *,
        reconstructed: bool = False,
    ) -> StateRecoveryResult:
        if not self._wal_latest_matches(wal, snapshot) or len(wal.records) != 1:
            raise StateRecoveryError("R05 migration WAL baseline is inconsistent")
        snapshot_hash = self.state_store.snapshot_hash(snapshot)
        if journal.schema_version == 1:
            recovery = self.journal.apply_planned_reconciliation(
                snapshot.last_processed_event_sequence, snapshot_hash
            )
            high_water = recovery.processing_high_water
            post = self.journal.inspect()
        else:
            baseline = wal.records[0]
            assert isinstance(baseline, BaselineRecord)
            high_water = baseline.journal_processing_high_water
            post = journal
        manifest = wal.active_manifest
        if (
            manifest is None
            or wal.baseline_record_id is None
            or wal.baseline_record_hash is None
        ):
            raise StateRecoveryError("R05 migration WAL baseline is incomplete")
        self.state_store.ensure_published(snapshot)
        if post.schema_version == 1:
            self.journal.append_v2_migration_checkpoint(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
                str(manifest.active_generation_id),
                str(wal.baseline_record_id),
                wal.baseline_record_hash,
            )
        elif post.schema_version is None:
            self.journal.append_v2_bootstrap_checkpoint(
                snapshot.last_processed_event_sequence,
                snapshot_hash,
                str(manifest.active_generation_id),
                str(wal.baseline_record_id),
                wal.baseline_record_hash,
                processing_high_water=high_water,
            )
        return self._result(snapshot, high_water, exact=reconstructed)

    def _start_recovery(
        self,
        journal: EventJournalInspection,
        wal: StateWALInspection | None,
        target: AgentStateSnapshot,
        category: EventRecoveryCategory,
        reason: RecoveryReason,
        *,
        external: bool,
        generation_id: UUID | None = None,
        invalid_current: bool = False,
    ) -> StateRecoveryResult:
        target_hash = self.state_store.snapshot_hash(target)
        recovery_id = str(uuid4())
        current_manifest = wal.active_manifest if wal is not None else None
        latest_matches = wal is not None and self._wal_latest_matches(wal, target)
        target_generation = (
            current_manifest.active_generation_id
            if latest_matches and current_manifest is not None
            else generation_id or uuid4()
        )
        self.journal.append_recovery_prepared(
            recovery_id,
            target.last_processed_event_sequence,
            target_hash,
            str(target_generation),
            category,
            processing_high_water=journal.processing_high_water,
        )
        if not latest_matches:
            predecessor_id = (
                current_manifest.active_generation_id
                if current_manifest is not None
                else None
            )
            predecessor_hash = (
                wal.record_hashes[-1] if wal is not None and wal.record_hashes else None
            )
            if invalid_current:
                self.wal.rebaseline_prepared_current(
                    target,
                    journal.processing_high_water,
                    recovery_id=UUID(recovery_id),
                    generation_id=target_generation,
                    external_reconciliation_required=external,
                )
            else:
                self.wal.resume_prepared_generation(
                    target,
                    journal.processing_high_water,
                    recovery_id=UUID(recovery_id),
                    reason=reason,
                    generation_id=target_generation,
                    predecessor_generation_id=predecessor_id,
                    predecessor_generation_hash=predecessor_hash,
                    external_reconciliation_required=external,
                )
        post_wal = self.wal.inspect()
        wal_record, wal_record_hash = self._record_for_snapshot(
            post_wal, target, target_hash
        )
        self.state_store.save(target)
        self.journal.append_recovery_completed(
            recovery_id,
            target.last_processed_event_sequence,
            target_hash,
            str(target_generation),
            category,
            external,
            processing_high_water=journal.processing_high_water,
            wal_record_id=str(wal_record.record_id),
            wal_record_hash=wal_record_hash,
        )
        return self._result(
            target,
            journal.processing_high_water,
            exact=category is EventRecoveryCategory.EXACT_CURRENT,
            rollback=category is EventRecoveryCategory.TRUE_ROLLBACK,
        )

    def _resume_recovery(
        self,
        journal: EventJournalInspection,
        wal: StateWALInspection,
        snapshot: AgentStateSnapshot | None,
    ) -> StateRecoveryResult:
        if len(journal.open_recoveries) != 1:
            raise StateRecoveryError("Journal recovery lifecycle is ambiguous")
        pending = journal.open_recoveries[0]
        try:
            target = self.wal.reconstruct(
                sequence=pending.snapshot_sequence,
                snapshot_hash=pending.snapshot_hash,
            )
        except StateWALError:
            _anchor, target = self.wal.inspect_anchored_prefix()
            if (
                target.last_processed_event_sequence != pending.snapshot_sequence
                or self.state_store.snapshot_hash(target) != pending.snapshot_hash
            ):
                raise StateRecoveryError("Recovery target is unavailable") from None
        target_generation = UUID(pending.wal_generation_id)
        active = wal.active_manifest
        if active is None or active.active_generation_id != target_generation:
            self.wal.resume_prepared_generation(
                target,
                pending.processing_high_water,
                recovery_id=UUID(pending.recovery_id),
                reason=self._reason_for_category(pending.category),
                generation_id=target_generation,
                predecessor_generation_id=(
                    active.active_generation_id if active is not None else None
                ),
                predecessor_generation_hash=(
                    wal.record_hashes[-1] if wal.record_hashes else None
                ),
                external_reconciliation_required=(
                    pending.category is EventRecoveryCategory.TRUE_ROLLBACK
                    or journal.external_reconciliation_required
                ),
            )
        if snapshot != target:
            self.state_store.save(target)
        external = (
            pending.category is EventRecoveryCategory.TRUE_ROLLBACK
            or journal.external_reconciliation_required
        )
        post_wal = self.wal.inspect()
        wal_record, wal_record_hash = self._record_for_snapshot(
            post_wal, target, pending.snapshot_hash
        )
        self.journal.append_recovery_completed(
            pending.recovery_id,
            pending.snapshot_sequence,
            pending.snapshot_hash,
            pending.wal_generation_id,
            pending.category,
            external,
            processing_high_water=pending.processing_high_water,
            wal_record_id=str(wal_record.record_id),
            wal_record_hash=wal_record_hash,
        )
        return self._result(
            target,
            pending.processing_high_water,
            exact=pending.category is EventRecoveryCategory.EXACT_CURRENT,
            rollback=pending.category is EventRecoveryCategory.TRUE_ROLLBACK,
        )

    def _resume_invalid_current_recovery(
        self,
        journal: EventJournalInspection,
        target: AgentStateSnapshot,
        current_snapshot: AgentStateSnapshot | None,
    ) -> StateRecoveryResult:
        if len(journal.open_recoveries) != 1:
            raise StateRecoveryError("Journal recovery lifecycle is ambiguous")
        pending = journal.open_recoveries[0]
        recovery_id = UUID(pending.recovery_id)
        generation_id = UUID(pending.wal_generation_id)
        external = (
            pending.category is EventRecoveryCategory.TRUE_ROLLBACK
            or journal.external_reconciliation_required
        )
        if external:
            self.wal.resume_prepared_generation(
                target,
                pending.processing_high_water,
                recovery_id=recovery_id,
                reason=RecoveryReason.TRUE_ROLLBACK,
                generation_id=generation_id,
                predecessor_generation_id=None,
                predecessor_generation_hash=None,
                external_reconciliation_required=True,
            )
        else:
            self.wal.rebaseline_prepared_current(
                target,
                pending.processing_high_water,
                recovery_id=recovery_id,
                generation_id=generation_id,
                external_reconciliation_required=external,
            )
        if current_snapshot != target:
            self.state_store.save(target)
        inspection = self.wal.inspect()
        wal_record, wal_record_hash = self._record_for_snapshot(
            inspection, target, pending.snapshot_hash
        )
        self.journal.append_recovery_completed(
            pending.recovery_id,
            pending.snapshot_sequence,
            pending.snapshot_hash,
            pending.wal_generation_id,
            pending.category,
            external,
            processing_high_water=pending.processing_high_water,
            wal_record_id=str(wal_record.record_id),
            wal_record_hash=wal_record_hash,
        )
        return self._result(
            target,
            pending.processing_high_water,
            exact=pending.category is EventRecoveryCategory.EXACT_CURRENT,
            rollback=pending.category is EventRecoveryCategory.TRUE_ROLLBACK,
        )

    def _recover_from_anchor(
        self,
        journal: EventJournalInspection,
        snapshot: AgentStateSnapshot | None,
        *,
        anchor_expected: bool,
        wal_error: StateWALError | None,
    ) -> StateRecoveryResult:
        del wal_error
        if not anchor_expected:
            raise StateRecoveryError("No verified recovery point exists") from None
        try:
            anchor, target = self.wal.inspect_anchored_prefix()
        except StateWALError:
            raise StateRecoveryError("Bootable recovery point is invalid") from None
        matching_journal_record = any(
            record.record_id == str(anchor.journal_tail_record_id)
            and record.record_hash == anchor.journal_tail_record_hash
            for record in journal.records
        )
        matching_journal_lineage = (
            str(anchor.journal_lineage_id) == journal.journal_lineage_id
        )
        if (
            not matching_journal_record and not matching_journal_lineage
        ) or anchor.journal_processing_high_water > journal.processing_high_water:
            raise StateRecoveryError("Boot anchor is not bound to this Journal")
        if journal.open_recoveries:
            if len(journal.open_recoveries) != 1:
                raise StateRecoveryError("Journal recovery lifecycle is ambiguous")
            pending = journal.open_recoveries[0]
            target_hash = self.state_store.snapshot_hash(target)
            if (
                pending.snapshot_sequence != target.last_processed_event_sequence
                or pending.snapshot_hash != target_hash
            ):
                raise StateRecoveryError("Open recovery does not match boot anchor")
            return self._finish_event_reconciliation(
                self._resume_invalid_current_recovery(
                    journal,
                    target,
                    snapshot,
                )
            )
        target_hash = self.state_store.snapshot_hash(target)
        exact_current = (
            target.last_processed_event_sequence == journal.snapshot_sequence
            and target_hash == journal.snapshot_hash
        )
        if journal.open_events:
            self.journal.inspect(target.last_processed_event_sequence, target_hash)
        result = self._start_recovery(
            journal,
            None,
            target,
            (
                EventRecoveryCategory.EXACT_CURRENT
                if exact_current
                else EventRecoveryCategory.TRUE_ROLLBACK
            ),
            (
                RecoveryReason.EXACT_CURRENT_REPAIR
                if exact_current
                else RecoveryReason.TRUE_ROLLBACK
            ),
            external=(
                journal.external_reconciliation_required if exact_current else True
            ),
            invalid_current=exact_current,
        )
        return self._finish_event_reconciliation(result)

    def _finish_event_reconciliation(
        self, result: StateRecoveryResult
    ) -> StateRecoveryResult:
        recovery = self.journal.apply_planned_reconciliation(
            result.snapshot.last_processed_event_sequence,
            result.snapshot_hash,
        )
        if recovery.processing_high_water != result.processing_high_water:
            raise StateRecoveryError("Recovery changed processing high-water")
        return result

    def _result(
        self,
        snapshot: AgentStateSnapshot,
        high_water: int,
        *,
        exact: bool = False,
        rollback: bool = False,
    ) -> StateRecoveryResult:
        inspection = self.wal.inspect()
        manifest = inspection.active_manifest
        if manifest is None:
            raise StateRecoveryError("StateWAL manifest is absent")
        return StateRecoveryResult(
            snapshot=snapshot,
            snapshot_hash=self.state_store.snapshot_hash(snapshot),
            processing_high_water=high_water,
            manifest=manifest,
            external_reconciliation_required=(
                manifest.external_reconciliation_required
            ),
            exact_current_reconstructed=exact,
            true_rollback_performed=rollback,
        )

    def _inspect_snapshot(
        self,
    ) -> tuple[AgentStateSnapshot | None, AgentStateLoadError | None]:
        if not self.state_store.snapshot_exists():
            return None, None
        try:
            status = Path(self.state_store.path).lstat()
            if not stat.S_ISREG(status.st_mode):
                raise AgentStateLoadError("AgentState snapshot target is unsafe")
            return self.state_store.load(), None
        except AgentStateLoadError as error:
            return None, error

    def _validate_cross_authority(
        self,
        journal: EventJournalInspection,
        wal: StateWALInspection,
    ) -> None:
        manifest = wal.active_manifest
        if manifest is None or not wal.records:
            raise StateRecoveryError("StateWAL evidence is incomplete")
        expected_generation = self._journal_active_generation(journal)
        if expected_generation is not None and expected_generation != str(
            manifest.active_generation_id
        ):
            if not any(
                pending.wal_generation_id == str(manifest.active_generation_id)
                for pending in journal.open_recoveries
            ):
                raise StateRecoveryError("Journal and WAL generations diverge")
        if len(journal.open_recoveries) == 1:
            pending = journal.open_recoveries[0]
            baseline = wal.records[0]
            if (
                pending.wal_generation_id == str(manifest.active_generation_id)
                and len(wal.records) == 1
                and isinstance(baseline, BaselineRecord)
                and baseline.baseline_snapshot_sequence == pending.snapshot_sequence
                and baseline.baseline_snapshot_hash == pending.snapshot_hash
                and baseline.journal_processing_high_water
                == pending.processing_high_water
            ):
                return
        anchor: tuple[int, int] | None = None
        for journal_index, evidence in enumerate(journal.records):
            if (
                evidence.schema_version < 2
                or evidence.lifecycle
                not in {
                    EventLifecycle.CHECKPOINT,
                    EventLifecycle.COMPLETED,
                    EventLifecycle.RECOVERY_COMPLETED,
                }
                or evidence.wal_generation_id != str(manifest.active_generation_id)
                or evidence.wal_record_id is None
                or evidence.wal_record_hash is None
                or evidence.snapshot_sequence is None
                or evidence.snapshot_hash is None
            ):
                continue
            for wal_index, (record, record_hash) in enumerate(
                zip(wal.records, wal.record_hashes)
            ):
                state = (
                    record.baseline_snapshot
                    if isinstance(record, BaselineRecord)
                    else record.candidate_snapshot
                )
                if (
                    evidence.wal_record_id == str(record.record_id)
                    and evidence.wal_record_hash == record_hash
                    and evidence.snapshot_sequence
                    == state.last_processed_event_sequence
                    and evidence.snapshot_hash == self.state_store.snapshot_hash(state)
                ):
                    anchor = (journal_index, wal_index)
                    break
            if anchor is not None:
                break
        if anchor is None:
            raise StateRecoveryError("Retained Journal lacks current WAL anchor")
        journal_index, wal_index = anchor
        retained = journal.records[journal_index + 1 :]
        noncommitting = {
            record.processing_sequence
            for record in retained
            if record.processing_sequence is not None
            and (
                record.lifecycle is EventLifecycle.FAILED
                or (
                    record.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED
                    and record.failure_category
                    is EventFailureCategory.UNCOMMITTED_AFTER_CRASH
                )
            )
        }
        anchor_evidence = journal.records[journal_index]
        prior_processing = (
            anchor_evidence.recovery_processing_high_water
            if anchor_evidence.lifecycle is EventLifecycle.RECOVERY_COMPLETED
            else anchor_evidence.processing_sequence
        )
        if prior_processing is None:
            raise StateRecoveryError("Retained Journal WAL anchor is incomplete")
        for index, record in enumerate(
            wal.records[wal_index + 1 :], start=wal_index + 1
        ):
            if not isinstance(record, TransitionRecord):
                raise StateRecoveryError("StateWAL transition is invalid")
            if any(
                sequence not in noncommitting
                for sequence in range(prior_processing + 1, record.processing_sequence)
            ):
                raise StateRecoveryError("WAL transition gap lacks Journal evidence")
            prepared = next(
                (
                    item
                    for item in retained
                    if item.lifecycle is EventLifecycle.PREPARED
                    and item.event_id == str(record.event_id)
                    and item.processing_sequence == record.processing_sequence
                    and item.state_hash_before == record.prior_snapshot_hash
                    and item.state_hash_after == record.candidate_snapshot_hash
                    and item.wal_generation_id == str(record.generation_id)
                ),
                None,
            )
            if prepared is None:
                raise StateRecoveryError(
                    "StateWAL transition lacks Journal preparation"
                )
            completed = next(
                (
                    item
                    for item in retained
                    if item.lifecycle is EventLifecycle.COMPLETED
                    and item.event_id == str(record.event_id)
                    and item.wal_record_id == str(record.record_id)
                    and item.wal_record_hash == record.record_hash
                ),
                None,
            )
            classified_committed = any(
                item.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED
                and item.event_id == str(record.event_id)
                and item.failure_category is EventFailureCategory.COMMITTED_BEFORE_CRASH
                for item in retained
            )
            if (
                index < len(wal.records) - 1
                and completed is None
                and not classified_committed
            ):
                raise StateRecoveryError("Uncommitted WAL transition is not the tail")
            prior_processing = record.processing_sequence

    @staticmethod
    def _journal_active_generation(
        journal: EventJournalInspection,
    ) -> str | None:
        for record in reversed(journal.records):
            if (
                record.schema_version >= 2
                and record.wal_generation_id is not None
                and record.lifecycle
                in {
                    EventLifecycle.CHECKPOINT,
                    EventLifecycle.COMPLETED,
                    EventLifecycle.RECOVERY_COMPLETED,
                }
            ):
                return record.wal_generation_id
        return None

    def _wal_latest_matches(
        self, wal: StateWALInspection, snapshot: AgentStateSnapshot
    ) -> bool:
        return (
            wal.latest_snapshot_sequence == snapshot.last_processed_event_sequence
            and wal.latest_snapshot_hash == self.state_store.snapshot_hash(snapshot)
        )

    def _record_for_snapshot(
        self,
        wal: StateWALInspection,
        snapshot: AgentStateSnapshot,
        snapshot_hash: str,
    ) -> tuple[BaselineRecord | TransitionRecord, str]:
        for record, record_hash in reversed(tuple(zip(wal.records, wal.record_hashes))):
            state = (
                record.baseline_snapshot
                if isinstance(record, BaselineRecord)
                else record.candidate_snapshot
            )
            if (
                state.last_processed_event_sequence
                == snapshot.last_processed_event_sequence
                and self.state_store.snapshot_hash(state) == snapshot_hash
            ):
                return record, record_hash
        raise StateRecoveryError("Bootable snapshot is not retained")

    @staticmethod
    def _reason_for_category(category: EventRecoveryCategory) -> RecoveryReason:
        if category is EventRecoveryCategory.TRUE_ROLLBACK:
            return RecoveryReason.TRUE_ROLLBACK
        if category is EventRecoveryCategory.UNCOMMITTED_TAIL:
            return RecoveryReason.UNCOMMITTED_TAIL
        return RecoveryReason.EXACT_CURRENT_REPAIR
