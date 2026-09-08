"""Process-local orchestration for capability-declared transaction participants."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable
from uuid import UUID, uuid5

from kagya.runtime.agent_runtime import AgentEvent
from kagya.runtime.event_journal import (
    AbortOutcome,
    EventJournal,
    EventJournalTransaction,
    ParticipantCapability,
    ParticipantOutcome,
    ParticipantRequirement,
    ReconciliationReason,
    StartupParticipantOutcome,
    TransactionKind,
)
from kagya.runtime.state_recovery import InternalCommitEvidence


T = TypeVar("T")
_TRANSACTION_ID_NAMESPACE = UUID("b27ca4e8-d34a-5e77-bfef-35f221571c4e")


@dataclass(frozen=True, slots=True)
class TransactionBinding:
    transaction_id: str
    event_id: str
    processing_sequence: int
    participant_id: str
    operation_digest: str
    transaction_kind: TransactionKind


@runtime_checkable
class TransactionParticipant(Protocol):
    @property
    def participant_id(self) -> str: ...

    @property
    def operation_digest(self) -> str: ...

    @property
    def capabilities(self) -> tuple[ParticipantCapability, ...]: ...

    def prepare(self, binding: TransactionBinding) -> None: ...

    def finalize(self, binding: TransactionBinding) -> ParticipantOutcome: ...


@runtime_checkable
class AbortableTransactionParticipant(TransactionParticipant, Protocol):
    def abort(self, binding: TransactionBinding) -> AbortOutcome: ...


@runtime_checkable
class ReconcilableTransactionParticipant(TransactionParticipant, Protocol):
    def inspect_reconciliation(
        self, binding: TransactionBinding
    ) -> StartupParticipantOutcome: ...

    def reconcile(self, binding: TransactionBinding) -> StartupParticipantOutcome: ...


@dataclass(frozen=True, slots=True)
class CoordinatedResult(Generic[T]):
    value: T
    participants: tuple[TransactionParticipant, ...]
    transaction_kind: TransactionKind = TransactionKind.EVENT_MUTATION


class TransactionCoordinatorError(Exception):
    """A bounded participant protocol failure."""


class TransactionPreparationError(TransactionCoordinatorError):
    """Participant preparation did not reach safe completion."""


class TransactionFinalizationError(TransactionCoordinatorError):
    """Participant finalization requires later reconciliation."""


class ParticipantDivergedError(Exception):
    """A participant reports a typed revision conflict."""


class ParticipantUnavailableError(Exception):
    """A participant cannot currently complete its operation."""


class UnsupportedParticipantReconciliationError(Exception):
    """A participant cannot provide the required reconciliation capability."""


@dataclass(frozen=True, slots=True)
class _LiveTransaction:
    event_id: str
    processing_sequence: int
    transaction_id: str
    kind: TransactionKind
    participants: tuple[TransactionParticipant, ...]
    requirements: tuple[ParticipantRequirement, ...]
    bindings: tuple[TransactionBinding, ...]


class TransactionCoordinator:
    """Coordinate live participants while EventJournal remains durable authority."""

    def __init__(
        self,
        journal: EventJournal,
        internal_commit_verifier: Callable[[AgentEvent, InternalCommitEvidence], None],
    ) -> None:
        self.journal = journal
        self._internal_commit_verifier = internal_commit_verifier
        self._lock = RLock()
        self._live: dict[str, _LiveTransaction] = {}

    def prepare_result(self, event: AgentEvent, raw_value: object) -> object:
        """Durably declare and prepare a coordinated handler result."""

        if not isinstance(raw_value, CoordinatedResult):
            return raw_value
        with self._lock:
            return self._prepare_result(event, raw_value)

    @staticmethod
    def derive_transaction_id(
        event: AgentEvent, transaction_kind: TransactionKind
    ) -> str:
        """Derive UUIDv5 from the exact JSON array [event UUID, sequence, kind]."""

        sequence = event.processing_sequence
        identity_invalid = False
        try:
            parsed_event_id = UUID(event.event_id)
            identity_invalid = str(parsed_event_id) != event.event_id
        except (AttributeError, TypeError, ValueError):
            identity_invalid = True
        if (
            identity_invalid
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= 0
            or not isinstance(transaction_kind, TransactionKind)
        ):
            raise TransactionPreparationError("Transaction identity is invalid")
        canonical = json.dumps(
            [event.event_id, sequence, transaction_kind.value],
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        return str(uuid5(_TRANSACTION_ID_NAMESPACE, canonical))

    def _prepare_result(
        self, event: AgentEvent, result: CoordinatedResult[object]
    ) -> object:
        sequence = event.processing_sequence
        if (
            sequence is None
            or event.event_id in self._live
            or not isinstance(result.transaction_kind, TransactionKind)
            or not isinstance(result.participants, tuple)
        ):
            raise TransactionPreparationError("Transaction preparation is invalid")

        participants, requirements = self._validate_plan(result.participants)
        transaction_id = self.derive_transaction_id(event, result.transaction_kind)
        bindings = tuple(
            TransactionBinding(
                transaction_id=transaction_id,
                event_id=event.event_id,
                processing_sequence=sequence,
                participant_id=requirement.participant_id,
                operation_digest=requirement.operation_digest,
                transaction_kind=result.transaction_kind,
            )
            for requirement in requirements
        )
        live = _LiveTransaction(
            event_id=event.event_id,
            processing_sequence=sequence,
            transaction_id=transaction_id,
            kind=result.transaction_kind,
            participants=participants,
            requirements=requirements,
            bindings=bindings,
        )
        self.journal.append_transaction_prepared(
            event, transaction_id, result.transaction_kind, requirements
        )
        self._live[event.event_id] = live

        preparation_failed = False
        for participant, binding in zip(participants, bindings, strict=True):
            try:
                participant.prepare(binding)
            except Exception:
                preparation_failed = True
                break
        if not preparation_failed:
            return result.value

        self._abort_after_preparation_failure(event, live)
        raise TransactionPreparationError("Transaction preparation failed")

    @staticmethod
    def _validate_plan(
        planned: tuple[TransactionParticipant, ...],
    ) -> tuple[tuple[TransactionParticipant, ...], tuple[ParticipantRequirement, ...]]:
        if not planned:
            raise TransactionPreparationError("Transaction participant plan is empty")
        validated: list[tuple[TransactionParticipant, ParticipantRequirement]] = []
        validation_failed = False
        try:
            for participant in planned:
                if not callable(getattr(participant, "prepare", None)) or not callable(
                    getattr(participant, "finalize", None)
                ):
                    raise TypeError
                requirement = ParticipantRequirement(
                    participant_id=participant.participant_id,
                    operation_digest=participant.operation_digest,
                    capabilities=participant.capabilities,
                )
                capabilities = set(requirement.capabilities)
                if (
                    ParticipantCapability.ABORT in capabilities
                    and not callable(getattr(participant, "abort", None))
                ) or (
                    ParticipantCapability.INSPECT_RECONCILE in capabilities
                    and (
                        not callable(
                            getattr(participant, "inspect_reconciliation", None)
                        )
                        or not callable(getattr(participant, "reconcile", None))
                    )
                ):
                    raise TypeError
                validated.append((participant, requirement))
        except Exception:
            validation_failed = True
        if validation_failed:
            raise TransactionPreparationError("Transaction participant plan is invalid")
        validated.sort(key=lambda item: item[1].participant_id)
        ids = tuple(requirement.participant_id for _, requirement in validated)
        if len(set(ids)) != len(ids):
            raise TransactionPreparationError(
                "Transaction participant plan is duplicated"
            )
        return (
            tuple(participant for participant, _ in validated),
            tuple(requirement for _, requirement in validated),
        )

    def _abort_after_preparation_failure(
        self, event: AgentEvent, live: _LiveTransaction
    ) -> None:
        abort_required = {
            requirement.participant_id
            for requirement in live.requirements
            if ParticipantCapability.ABORT in requirement.capabilities
        }
        reason = ReconciliationReason.PARTICIPANT_UNAVAILABLE
        for participant, requirement, binding in zip(
            live.participants, live.requirements, live.bindings, strict=True
        ):
            if requirement.participant_id not in abort_required:
                continue
            try:
                outcome = cast(AbortableTransactionParticipant, participant).abort(
                    binding
                )
                if not isinstance(outcome, AbortOutcome):
                    raise TypeError
                self.journal.append_participant_aborted(
                    event,
                    live.transaction_id,
                    requirement.participant_id,
                    requirement.operation_digest,
                    outcome,
                )
            except Exception as error:
                reason = self._reason_for(error)

        try:
            transaction = self._transaction(live.transaction_id)
            confirmed = {participant for participant, _ in transaction.abort_outcomes}
            unresolved = tuple(sorted(abort_required - confirmed))
            if unresolved:
                self.journal.append_transaction_abort_required(
                    event, live.transaction_id, reason, unresolved
                )
            else:
                self.journal.append_transaction_aborted(event, live.transaction_id)
                del self._live[event.event_id]
        except Exception:
            pass

    def finalize_event(
        self, event: AgentEvent, evidence: InternalCommitEvidence
    ) -> None:
        """Finalize a live transaction only after matching U2 internal proof."""

        with self._lock:
            self._finalize_event(event, evidence)

    def _finalize_event(
        self, event: AgentEvent, evidence: InternalCommitEvidence
    ) -> None:
        live = self._live.get(event.event_id)
        if live is None:
            if any(
                transaction.event_id == event.event_id
                and transaction.terminal_lifecycle is None
                for transaction in self.journal.inspect().open_transactions
            ):
                raise TransactionFinalizationError(
                    "Live transaction handles are unavailable"
                )
            return
        if not isinstance(evidence, InternalCommitEvidence):
            raise TransactionFinalizationError("Internal commit evidence is invalid")
        if (
            event.processing_sequence != live.processing_sequence
            or evidence.event_id != live.event_id
            or evidence.processing_sequence != live.processing_sequence
        ):
            raise TransactionFinalizationError("Internal commit evidence is invalid")
        verification_failed = False
        try:
            self._internal_commit_verifier(event, evidence)
        except Exception:
            verification_failed = True
        if verification_failed:
            raise TransactionFinalizationError("Internal commit evidence is invalid")

        transaction = self._transaction(live.transaction_id)
        known = {participant for participant, _ in transaction.participant_outcomes}
        failure: Exception | None = None
        for participant, requirement, binding in zip(
            live.participants, live.requirements, live.bindings, strict=True
        ):
            if requirement.participant_id in known:
                continue
            try:
                outcome = participant.finalize(binding)
                if not isinstance(outcome, ParticipantOutcome):
                    raise TypeError
                self.journal.append_participant_finalized(
                    event,
                    live.transaction_id,
                    requirement.participant_id,
                    requirement.operation_digest,
                    outcome,
                )
                known.add(requirement.participant_id)
            except Exception as error:
                failure = error
                break

        if failure is not None:
            self._record_finalization_required(event, live, failure)
            raise TransactionFinalizationError("Transaction finalization failed")
        self.journal.append_transaction_completed(event, live.transaction_id)
        del self._live[event.event_id]

    def _record_finalization_required(
        self, event: AgentEvent, live: _LiveTransaction, error: Exception
    ) -> None:
        try:
            transaction = self._transaction(live.transaction_id)
            known = {participant for participant, _ in transaction.participant_outcomes}
            unresolved = tuple(
                requirement.participant_id
                for requirement in live.requirements
                if requirement.participant_id not in known
            )
            if unresolved and transaction.reconciliation_reason is None:
                self.journal.append_transaction_reconciliation_required(
                    event,
                    live.transaction_id,
                    self._reason_for(error),
                    unresolved,
                )
        except Exception:
            pass

    def _transaction(self, transaction_id: str) -> EventJournalTransaction:
        inspection = self.journal.inspect()
        for transaction in (
            *inspection.open_transactions,
            *inspection.completed_transactions,
            *inspection.reconciled_transactions,
            *inspection.aborted_transactions,
        ):
            if transaction.transaction_id == transaction_id:
                return transaction
        raise TransactionCoordinatorError("Transaction evidence is unavailable")

    @staticmethod
    def _reason_for(error: Exception) -> ReconciliationReason:
        if isinstance(error, ParticipantDivergedError):
            return ReconciliationReason.PARTICIPANT_DIVERGED
        if isinstance(error, UnsupportedParticipantReconciliationError):
            return ReconciliationReason.UNSUPPORTED_RECONCILIATION
        return ReconciliationReason.PARTICIPANT_UNAVAILABLE
