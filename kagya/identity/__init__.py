"""Dependency-light identity provenance and Value domain primitives."""

from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    recompute_origin_id,
    validate_origin_id,
)
from kagya.identity.value_system import (
    ValueConflictDefinition,
    ValueEvidence,
    ValueProposal,
    ValueReason,
    ValueScope,
    ValueState,
    canonical_seed_payload,
    recompute_seed_contract_digest,
    validate_seed_contract_digest,
)

__all__ = [
    "IdentityOrigin",
    "OriginActor",
    "OriginInputKind",
    "ValueAdmissionStatus",
    "ValueConflictDefinition",
    "ValueEvidence",
    "ValueProposal",
    "ValueReason",
    "ValueScope",
    "ValueState",
    "canonical_seed_payload",
    "recompute_origin_id",
    "recompute_seed_contract_digest",
    "validate_origin_id",
    "validate_seed_contract_digest",
]
