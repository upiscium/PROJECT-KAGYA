"""Cross-authority startup recovery without transferring authority ownership."""

from __future__ import annotations

from dataclasses import dataclass
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
    EventFailureCategory,
    EventJournal,
    EventJournalInspection,
    EventLifecycle,
    EventRecoveryCategory,
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


@dataclass(frozen=True, slots=True)
class StateRecoveryResult:
    snapshot: AgentStateSnapshot
    snapshot_hash: str
    processing_high_water: int
    manifest: Manifest
    external_reconciliation_required: bool
    exact_current_reconstructed: bool = False
    true_rollback_performed: bool = False


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
            if anchor is None:
                raise StateRecoveryError("StateWAL is invalid and unanchored") from None
            return self._recover_from_anchor(
                journal_inspection,
                snapshot,
                anchor_expected=True,
                wal_error=wal_error,
            )
        if not wal_inspection.exists:
            return self._bootstrap_r06(journal_inspection, snapshot, snapshot_error)

        if journal_inspection.schema_version in {None, 1}:
            if snapshot is None:
                raise StateRecoveryError(
                    "R05 migration requires a valid canonical snapshot"
                ) from None
            return self._finish_v1_migration(
                journal_inspection, wal_inspection, snapshot
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
            external=False,
        )
        return self._finish_event_reconciliation(result)

    def commit_candidate(
        self,
        event: AgentEvent,
        prior_snapshot: AgentStateSnapshot,
        candidate_snapshot: AgentStateSnapshot,
    ) -> TransitionRecord:
        """Execute prepared -> WAL -> snapshot -> completed in exact order."""

        with self._lock:
            return self._commit_candidate(event, prior_snapshot, candidate_snapshot)

    def _commit_candidate(
        self,
        event: AgentEvent,
        prior_snapshot: AgentStateSnapshot,
        candidate_snapshot: AgentStateSnapshot,
    ) -> TransitionRecord:

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
        self.journal.append_completed(
            event,
            sequence,
            after_hash,
            wal_generation_id=generation_id,
            wal_record_id=str(transition.record_id),
            wal_record_hash=transition.record_hash,
        )
        return transition

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
        self.state_store.ensure_published(snapshot)
        return self._result(snapshot, high_water)

    def _finish_v1_migration(
        self,
        journal: EventJournalInspection,
        wal: StateWALInspection,
        snapshot: AgentStateSnapshot,
    ) -> StateRecoveryResult:
        if not self._wal_latest_matches(wal, snapshot) or len(wal.records) != 1:
            raise StateRecoveryError("R05 migration WAL baseline is inconsistent")
        snapshot_hash = self.state_store.snapshot_hash(snapshot)
        recovery = self.journal.apply_planned_reconciliation(
            snapshot.last_processed_event_sequence, snapshot_hash
        )
        post = self.journal.inspect()
        manifest = wal.active_manifest
        if (
            manifest is None
            or wal.baseline_record_id is None
            or wal.baseline_record_hash is None
        ):
            raise StateRecoveryError("R05 migration WAL baseline is incomplete")
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
                processing_high_water=recovery.processing_high_water,
            )
        self.state_store.ensure_published(snapshot)
        return self._result(snapshot, recovery.processing_high_water)

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
        self.state_store.save(target)
        self.journal.append_recovery_completed(
            recovery_id,
            target.last_processed_event_sequence,
            target_hash,
            str(target_generation),
            category,
            external,
            processing_high_water=journal.processing_high_water,
        )
        return self._result(
            target,
            journal.processing_high_water,
            exact=category is EventRecoveryCategory.EXACT_CURRENT,
            rollback=external,
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
                ),
            )
        if snapshot != target:
            self.state_store.save(target)
        external = pending.category is EventRecoveryCategory.TRUE_ROLLBACK
        self.journal.append_recovery_completed(
            pending.recovery_id,
            pending.snapshot_sequence,
            pending.snapshot_hash,
            pending.wal_generation_id,
            pending.category,
            external,
            processing_high_water=pending.processing_high_water,
        )
        return self._result(
            target,
            pending.processing_high_water,
            exact=pending.category is EventRecoveryCategory.EXACT_CURRENT,
            rollback=external,
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
                pending.category is not EventRecoveryCategory.TRUE_ROLLBACK
                or pending.snapshot_sequence != target.last_processed_event_sequence
                or pending.snapshot_hash != target_hash
            ):
                raise StateRecoveryError("Open recovery does not match boot anchor")
            target_generation = UUID(pending.wal_generation_id)
            self.wal.resume_prepared_generation(
                target,
                pending.processing_high_water,
                recovery_id=UUID(pending.recovery_id),
                reason=RecoveryReason.TRUE_ROLLBACK,
                generation_id=target_generation,
                predecessor_generation_id=None,
                predecessor_generation_hash=None,
                external_reconciliation_required=True,
            )
            if snapshot != target:
                self.state_store.save(target)
            self.journal.append_recovery_completed(
                pending.recovery_id,
                pending.snapshot_sequence,
                pending.snapshot_hash,
                pending.wal_generation_id,
                pending.category,
                True,
                processing_high_water=pending.processing_high_water,
            )
            return self._result(
                target,
                pending.processing_high_water,
                rollback=True,
            )
        return self._start_recovery(
            journal,
            None,
            target,
            EventRecoveryCategory.TRUE_ROLLBACK,
            RecoveryReason.TRUE_ROLLBACK,
            external=True,
        )

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
        noncommitting = {
            record.processing_sequence
            for record in journal.records
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
        baseline = wal.records[0]
        if not isinstance(baseline, BaselineRecord):
            raise StateRecoveryError("StateWAL baseline is invalid")
        prior_processing = baseline.journal_processing_high_water
        for index, record in enumerate(wal.records[1:], start=1):
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
                    for item in journal.records
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
                    for item in journal.records
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
                for item in journal.records
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
                record.schema_version == 2
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
