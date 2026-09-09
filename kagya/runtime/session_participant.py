"""Ephemeral compatibility participant for process-local SessionState turns."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from uuid import UUID

from kagya.runtime.event_journal import (
    ParticipantCapability,
    ParticipantOutcome,
    StartupParticipantOutcome,
)
from kagya.runtime.session_state import SessionState
from kagya.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    TransactionBinding,
    validate_transaction_binding,
)


SESSION_TURN_PARTICIPANT_ID = "session.turn"
_SESSION_HASH_DOMAIN = b"PROJECT-KAGYA:R07:SESSION-TURN:V1\x00"


@dataclass(frozen=True, slots=True)
class SessionTurnOperation:
    user_input: str
    response: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_input, str) or not isinstance(self.response, str):
            raise ValueError("Session turn is invalid")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "user_input": self.user_input,
            "response": self.response,
        }


def session_turn_operation_digest(operation: SessionTurnOperation) -> str:
    canonical = json.dumps(
        operation.canonical_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_SESSION_HASH_DOMAIN + canonical).hexdigest()


def inspect_reset_session_operation(
    transaction_id: str, participant_id: str, operation_digest: str
) -> StartupParticipantOutcome:
    """Classify a fresh ephemeral epoch without reconstructing an old turn."""

    try:
        parsed = UUID(transaction_id)
    except (TypeError, ValueError):
        raise ParticipantDivergedError("Session identity is invalid") from None
    if (
        str(parsed) != transaction_id
        or participant_id != SESSION_TURN_PARTICIPANT_ID
        or re.fullmatch(r"[0-9a-f]{64}", operation_digest) is None
    ):
        raise ParticipantDivergedError("Session identity is invalid")
    return StartupParticipantOutcome.VERIFIED_CONSISTENT


class SessionTurnParticipant:
    """A no-stage participant whose authority resets with the process epoch."""

    participant_id = SESSION_TURN_PARTICIPANT_ID
    prepare_is_read_only = True
    capabilities = (
        ParticipantCapability.IDEMPOTENT_FINALIZE,
        ParticipantCapability.INSPECT_RECONCILE,
        ParticipantCapability.PREPARE,
    )

    def __init__(
        self, session_state: SessionState, operation: SessionTurnOperation
    ) -> None:
        self.session_state = session_state
        self.operation = operation
        self.operation_digest = session_turn_operation_digest(operation)

    def prepare(self, binding: TransactionBinding) -> None:
        """Validate only; SessionState must not advance before internal commit."""

        self._validate_binding(binding)

    def finalize(self, binding: TransactionBinding) -> ParticipantOutcome:
        self._validate_binding(binding)
        try:
            added = self.session_state.apply_coordinated_turn(
                (
                    binding.transaction_id,
                    binding.participant_id,
                    binding.operation_digest,
                ),
                self.operation.user_input,
                self.operation.response,
            )
        except ValueError:
            raise ParticipantDivergedError("Session operation conflicts") from None
        return (
            ParticipantOutcome.FINALIZED
            if added
            else ParticipantOutcome.ALREADY_CONSISTENT
        )

    def inspect_reconciliation(
        self, binding: TransactionBinding
    ) -> StartupParticipantOutcome:
        """A fresh process has no stale ephemeral Session effect to recover."""

        self._validate_binding(binding)
        return inspect_reset_session_operation(
            binding.transaction_id, binding.participant_id, binding.operation_digest
        )

    def reconcile(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        self._validate_binding(binding)
        return inspect_reset_session_operation(
            binding.transaction_id, binding.participant_id, binding.operation_digest
        )

    def _validate_binding(self, binding: TransactionBinding) -> None:
        if (
            not validate_transaction_binding(binding)
            or binding.participant_id != self.participant_id
            or re.fullmatch(r"[0-9a-f]{64}", binding.operation_digest) is None
            or binding.operation_digest != self.operation_digest
        ):
            raise ParticipantDivergedError("Session binding is invalid")
