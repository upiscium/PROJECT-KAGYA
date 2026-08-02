"""The operator-facing, fail-closed point-in-time restore contract."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import hashlib
import json
import re
from threading import Lock
from typing import Any, Callable, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kagya.external_transaction import (
    ExternalTransactionCoordinator,
    ExternalTransactionStatus,
)
from kagya.runtime.agent_runtime import current_agent_event, register_event_rollback
from kagya.runtime.agent_state import AgentStateSnapshot, AgentStateStore
from kagya.runtime.event_journal import (
    EventJournal,
    JournalIntegrityError,
    JournalLifecycle,
    JournalRecord,
    hash_snapshot,
)
from kagya.runtime.state_wal import StateWAL, StateWalIntegrityError, StateWalRecord


class RestoreErrorCode(StrEnum):
    TARGET_NOT_RETAINED = "restore_target_not_retained"
    TARGET_UNVERIFIED = "restore_target_unverified"
    WAL_INTEGRITY_INVALID = "restore_wal_integrity_invalid"
    JOURNAL_INTEGRITY_INVALID = "restore_journal_integrity_invalid"
    CHECKPOINT_MISMATCH = "restore_checkpoint_mismatch"
    PREVIEW_STALE = "restore_preview_stale"
    PREVIEW_EXPIRED = "restore_preview_expired"
    CONFIRMATION_REQUIRED = "restore_confirmation_required"
    EXTERNAL_STATE_INCONSISTENT = "restore_external_state_inconsistent"
    UNSUPPORTED_DOMAIN = "restore_unsupported_domain"
    OPERATION_IN_PROGRESS = "restore_operation_in_progress"
    NOT_AUTHORITATIVE = "restore_not_authoritative"
    COMMIT_INDETERMINATE = "commit_indeterminate"


class RestoreContractError(RuntimeError):
    """A deliberately non-disclosive contract error."""

    def __init__(self, code: RestoreErrorCode | str, message: str = "") -> None:
        self.code = str(code)
        super().__init__("restore request rejected")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RestoreCommitRequest(_Strict):
    target_sequence: int = Field(ge=0)
    expected_target_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_semantic_revision: int = Field(ge=0)
    expected_current_logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_preview_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_external_effect_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_phrase: str = Field(min_length=1, max_length=200)


class RestoreRef(_Strict):
    kind: Literal[
        "goal",
        "commitment",
        "decision",
        "plan",
        "action",
        "outbox",
        "journal",
        "experience",
        "memory",
        "belief",
    ]
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

    @model_validator(mode="after")
    def validate_public_id(self) -> "RestoreRef":
        if not public_reference(self.id, self.kind):
            raise ValueError("reference ID is not public-safe")
        return self


class RestoreDomain(_Strict):
    domain: Literal[
        "emotion_state",
        "working_memory",
        "motivation",
        "identity",
        "context_state",
        "appraisal",
        "experience",
        "belief",
        "attention",
        "decision",
        "decision_explanation",
        "agency_attribution",
        "counterfactual",
        "feedback",
        "metacognition",
        "action_execution",
        "proactive_outbox",
        "subject_scheduler",
        "extensions",
    ]
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    added_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    changed_count: int = Field(ge=0)
    changed_revision_count: int = Field(ge=0)
    newer_state_loss_count: int = Field(ge=0)
    refs: list[RestoreRef] = Field(max_length=256)
    truncated: bool
    reason_code: Literal["restore_unsupported_domain"] | None


class RestoreArtifact(_Strict):
    artifact_type: Literal["memory", "dataset", "adapter", "outbox", "unknown"]
    count: int = Field(ge=0)
    refs: list[str] = Field(max_length=256)
    truncated: bool

    @field_validator("refs")
    @classmethod
    def validate_public_refs(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
            raise ValueError("artifact reference is not an opaque handle")
        return values


class RestoreExternal(_Strict):
    consistency_status: Literal["consistent", "inconsistent"]
    artifacts: list[RestoreArtifact]
    retained_not_replayed_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    orphaned_count: int = Field(ge=0)
    retryable_count: int = Field(ge=0)
    effect_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_side_effects_replayed: Literal[False] = False


class RestoreTarget(_Strict):
    target_sequence: int = Field(ge=0)
    target_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_kind: Literal[
        "bootstrap", "journal_completed", "journal_recovered", "checkpoint"
    ]
    timestamp: str
    event_type: str | None
    eligible: bool
    reason_codes: list[str]


class RestoreOperation(_Strict):
    operation_id: str = Field(min_length=1, max_length=128)
    target_sequence: int = Field(ge=0)
    target_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preview_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_at: str
    started_at: str | None
    completed_at: str | None
    event_id: str = Field(min_length=1, max_length=128)
    processing_sequence: int | None = Field(default=None, ge=0)
    state: Literal[
        "previewed", "finalizing", "completed", "failed", "commit_indeterminate"
    ]
    error_code: str | None
    external_side_effects_replayed: Literal[False] = False

    @model_validator(mode="after")
    def validate_operation_binding(self) -> "RestoreOperation":
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            self.operation_id,
        ) or self.event_id != event_id_for_operation(self.operation_id):
            raise ValueError("restore operation identity binding is invalid")
        return self

    @field_validator("requested_at", "started_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("restore operation timestamp must include a timezone")
        return value


class RestoreSummary(_Strict):
    schema_version: Literal[1] = 1
    current_sequence: int
    current_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_revision: int = Field(ge=0)
    retained_min_sequence: int = Field(ge=0)
    retained_max_sequence: int = Field(ge=0)
    targets: list[RestoreTarget]
    latest_operation: RestoreOperation | None
    external_side_effects_replayed: Literal[False] = False


class RestorePreview(_Strict):
    schema_version: Literal[1] = 1
    operation_id: str = Field(min_length=1, max_length=128)
    preview_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    expires_at: str
    current_logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_revision: int = Field(ge=0)
    display_sequence: int = Field(ge=0)
    target_sequence: int = Field(ge=0)
    target_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    newer_authoritative_event_count: int = Field(ge=0)
    domains: list[RestoreDomain]
    external_effects: RestoreExternal
    restoreable: bool
    reason_codes: list[str]
    external_side_effects_replayed: Literal[False] = False
    confirmation_phrase: str

    @field_validator("operation_id")
    @classmethod
    def validate_public_operation_id(cls, value: str) -> str:
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            value,
        ):
            raise ValueError("operation ID is not canonical")
        return value


class RestoreCommitResponse(_Strict):
    command: Literal["restore"] = "restore"
    disposition: Literal["completed", "commit_indeterminate"]
    operation_id: str
    event_id: str
    processing_sequence: int = Field(ge=0)
    restored_target_sequence: int
    restored_target_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_restore_sequence: int = Field(ge=0)
    post_restore_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation_status: Literal[
        "finalizing", "completed", "failed", "commit_indeterminate"
    ]
    error_code: str | None
    external_side_effects_replayed: Literal[False] = False

    @model_validator(mode="after")
    def validate_operation_binding(self) -> "RestoreCommitResponse":
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            self.operation_id,
        ) or self.event_id != event_id_for_operation(self.operation_id):
            raise ValueError("restore operation identity binding is invalid")
        return self


_RESERVED = "operator_restore"
_EXTENSIONS = {
    "appraisal_calibration": "appraisal",
    "experiences": "experience",
    "beliefs": "belief",
    "attention": "attention",
    "decision_records": "decision",
    "decision_explanations": "decision_explanation",
    "agency_attribution": "agency_attribution",
    "counterfactual_simulation": "counterfactual",
    "feedback": "feedback",
    "metacognition": "metacognition",
    "action_execution": "action_execution",
    "proactive_outbox": "proactive_outbox",
    "subject_scheduler": "subject_scheduler",
}
_NESTED_EXTENSIONS = {
    "working_memory": set(),
    "motivation": {
        "goal_decisions",
        "intrinsic_goal_deliberations",
        "dynamics",
        "plans",
        "homeostatic_signal",
    },
    "identity": {"identity_boundary", "narrative_self"},
}
_EXTERNAL_REARM_DOMAINS = {
    "action_execution",
    "proactive_outbox",
    "subject_scheduler",
}
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PRIVATE_REFERENCE = re.compile(
    r"private[_-]*(?:sentinel|state|session|context)|"
    r"hidden[_-]*thought|raw[_-]*prompt|chain[_-]*of[_-]*thought|"
    r"api[_-]*(?:key|token|secret)|access[_-]*token|bearer|"
    r"credential|token|secret|password|prompt|attachment[_-]*body",
    re.IGNORECASE,
)
_PATH_REFERENCE = re.compile(r"^(?:[A-Za-z]:|~[\\/]|[\\/])")
_UUID_REFERENCE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_OPAQUE_REFERENCE = re.compile(
    r"[A-Za-z][A-Za-z0-9_.:-]{0,96}[-:](?:\d+|[a-f0-9]{16,}|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_OPAQUE_TAIL = re.compile(
    r"(?:\d+|[a-f0-9]{16,}|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_REFERENCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "goal": ("goal-",),
    "commitment": ("commitment-",),
    "decision": ("decision-", "goal-decision-"),
    "plan": ("plan-",),
    "action": ("action-", "intent-"),
    "outbox": ("outbox-", "message-"),
    "journal": ("event-", "operator-restore-", "journal-"),
    "experience": ("experience-",),
    "memory": ("memory-", "episode-", "semantic-"),
    "belief": ("belief-",),
}
_REF_KEYS: dict[str, str] = {
    "goal_id": "goal",
    "commitment_id": "commitment",
    "decision_id": "decision",
    "plan_id": "plan",
    "intent_id": "action",
    "message_id": "outbox",
    "event_id": "journal",
    "experience_id": "experience",
    "memory_id": "memory",
    "belief_id": "belief",
}
_ENTITY_ID_KEYS = tuple(_REF_KEYS) + (
    "item_id",
    "schedule_id",
    "approval_id",
    "receipt_id",
    "observation_id",
    "verification_id",
    "validation_id",
    "rejection_id",
    "notification_id",
)


@dataclass
class _SequenceEvidence:
    eligible: list[tuple[JournalRecord, str]] = field(default_factory=list)
    non_target: list[JournalRecord] = field(default_factory=list)


@dataclass(frozen=True)
class _VerifiedTarget:
    wal: StateWalRecord
    journal: StateWalRecord | JournalRecord
    kind: str
    semantic_count: int


@dataclass(frozen=True)
class _EvidenceIndex:
    floor: int
    maximum: int
    semantic_revision: int
    semantic_transition_count: int
    restorable_by_sequence: dict[int, _VerifiedTarget]
    semantic_targets: tuple[_VerifiedTarget, ...]


@dataclass(frozen=True)
class _PreviewCapability:
    preview: RestorePreview
    canonical_external_digest: str
    reserved: bool = False


def public_reference(value: str, kind: str | None = None) -> bool:
    """Return whether an opaque identifier is safe for public projections."""

    if (
        not _SAFE.fullmatch(value)
        or _PRIVATE_REFERENCE.search(value)
        or _PATH_REFERENCE.search(value)
    ):
        return False
    opaque = bool(
        _UUID_REFERENCE.fullmatch(value)
        or re.fullmatch(r"[a-f0-9]{32,}", value, re.IGNORECASE)
        or _OPAQUE_REFERENCE.fullmatch(value)
    )
    if not opaque or kind is None or _UUID_REFERENCE.fullmatch(value):
        return opaque
    lowered = value.casefold()
    return any(
        lowered.startswith(prefix)
        and bool(_OPAQUE_TAIL.fullmatch(value[len(prefix) :]))
        for prefix in _REFERENCE_PREFIXES[kind]
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def logical_state_digest(snapshot: AgentStateSnapshot) -> str:
    value = snapshot.model_dump(mode="json")
    value.pop("saved_at", None)
    value.pop("last_processed_event_sequence", None)
    value.get("extensions", {}).pop(_RESERVED, None)
    return _hash(value)


def event_id_for_operation(operation_id: str) -> str:
    return f"operator-restore-{operation_id}"


def _counts(before: Any, after: Any) -> tuple[int, int, int, int, int]:
    if isinstance(before, dict) and isinstance(after, dict):
        keys = set(before) | set(after)
        return (
            len(before),
            len(after),
            sum(k not in before for k in keys),
            sum(k not in after for k in keys),
            sum(k in before and k in after and before[k] != after[k] for k in keys),
        )
    return (
        1 if before is not None else 0,
        1 if after is not None else 0,
        0,
        0,
        int(before != after),
    )


def _record_map(items: Any, prefix: str, id_keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    records: dict[str, Any] = {}
    for index, item in enumerate(items):
        identifier = next(
            (
                item.get(key)
                for key in id_keys
                if isinstance(item, dict)
                and isinstance(item.get(key), str)
                and public_reference(item[key])
            ),
            None,
        )
        key = f"{prefix}:{identifier or index}"
        if key in records:
            key = f"{key}:{index}"
        records[key] = item
    return records


def _core_domain_records(name: str, value: dict[str, Any]) -> dict[str, Any]:
    if name == "emotion_state":
        return {"emotion": value}
    if name == "working_memory":
        return _record_map(value.get("items"), "working-memory", ("item_id",))
    if name == "motivation":
        return {
            **_record_map(value.get("active_goals"), "goal", ("goal_id",)),
            **_record_map(value.get("commitments"), "commitment", ("commitment_id",)),
        }
    if name == "context_state":
        return {
            **_record_map(value.get("frames"), "context", ("context_id",)),
            **_record_map(
                value.get("interlocutors"), "interlocutor", ("interlocutor_id",)
            ),
            **(
                {
                    f"relationship:{key}": item
                    for key, item in value.get("relationships", {}).items()
                }
                if isinstance(value.get("relationships"), dict)
                else {}
            ),
        }
    if name == "identity":
        return {"values": value.get("values"), "self_model": value.get("self_model")}
    return value


def _extension_records(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    list_records: dict[str, Any] = {}
    for key, items in value.items():
        if isinstance(items, list):
            list_records.update(
                _record_map(
                    items,
                    str(key),
                    (
                        "id",
                        "intent_id",
                        "message_id",
                        "decision_id",
                        "experience_id",
                        "belief_id",
                        "schedule_id",
                    ),
                )
            )
    return list_records if list_records else value


def _changed_record_values(before: Any, after: Any) -> list[Any]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return [before, after] if before != after else []
    changed: list[Any] = []
    for key in set(before) | set(after):
        if before.get(key) != after.get(key):
            if key in before:
                changed.append(before[key])
            if key in after:
                changed.append(after[key])
    return changed


def _safe_refs(value: Any, *, limit: int = 20) -> tuple[list[RestoreRef], bool]:
    refs: dict[tuple[str, str], RestoreRef] = {}
    omitted = False

    def visit(item: Any) -> None:
        nonlocal omitted
        if len(refs) > limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                kind = _REF_KEYS.get(str(key))
                if kind is not None and isinstance(child, str):
                    if public_reference(child, kind):
                        refs[(kind, child)] = RestoreRef(kind=cast(Any, kind), id=child)
                    else:
                        omitted = True
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    ordered = [refs[key] for key in sorted(refs)]
    return ordered[:limit], omitted or len(ordered) > limit


def _public_artifact_type(value: str) -> str | None:
    lowered = value.casefold()
    if lowered == "episodic_chroma" or lowered.startswith("chroma_"):
        return "memory"
    if lowered.startswith("dataset"):
        return "dataset"
    if lowered.startswith("adapter"):
        return "adapter"
    if lowered.startswith("outbox"):
        return "outbox"
    return None


def _protected_domain_has_active_work(key: str, value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, dict):
        raise ValueError("protected domain state must be an object")
    if key == "action_execution":
        from kagya.actions.execution import (
            ActionState,
            IntentStatus,
            receipt_matches_intent,
        )

        if value.get("schema_version") != 4:
            raise ValueError("protected action state version is invalid")
        action_state = ActionState.model_validate(value)
        intents = {item.intent_id: item for item in action_state.intents}
        validations = {
            item.validation_id: item for item in action_state.validation_records
        }
        approvals = {item.approval_id: item for item in action_state.approvals}
        receipts = {item.receipt_id: item for item in action_state.receipts}
        observations = {item.observation_id: item for item in action_state.observations}
        verifications = {
            item.verification_id: item for item in action_state.verifications
        }
        collections = (
            (intents, action_state.intents),
            (validations, action_state.validation_records),
            (approvals, action_state.approvals),
            (receipts, action_state.receipts),
            (observations, action_state.observations),
            (verifications, action_state.verifications),
        )
        if any(len(index) != len(records) for index, records in collections):
            raise ValueError("protected action identifiers must be unique")
        if any(
            item.intent_id is not None and item.intent_id not in intents
            for item in action_state.validation_records
        ) or any(
            item.validation_id not in validations
            for item in action_state.policy_rejections
        ):
            raise ValueError("protected action validation binding is invalid")
        for intent in action_state.intents:
            if (
                intent.validation_record_id is not None
                and intent.validation_record_id not in validations
            ):
                raise ValueError("protected action intent validation is invalid")
            if intent.approval_id is not None and intent.approval_id not in approvals:
                raise ValueError("protected action approval binding is invalid")
            if intent.receipt_id is not None and intent.receipt_id not in receipts:
                raise ValueError("protected action receipt binding is invalid")
            if (
                intent.receipt_id is not None
                and receipts[intent.receipt_id].intent_id != intent.intent_id
            ):
                raise ValueError("protected action receipt ownership is invalid")
        for validation in action_state.validation_records:
            if validation.intent_id is not None and (
                intents[validation.intent_id].validation_record_id
                != validation.validation_id
            ):
                raise ValueError("protected action validation ownership is invalid")
        approvals_by_intent: dict[str, Any] = {}
        for approval_record in action_state.approvals:
            if (
                approval_record.intent_id not in intents
                or approval_record.intent_id in approvals_by_intent
            ):
                raise ValueError("protected approval intent binding is invalid")
            approvals_by_intent[approval_record.intent_id] = approval_record
        for intent in action_state.intents:
            bound_approval = approvals_by_intent.get(intent.intent_id)
            if (intent.approval_id is None) != (bound_approval is None) or (
                bound_approval is not None
                and bound_approval.approval_id != intent.approval_id
            ):
                raise ValueError("protected approval binding is inconsistent")
        for receipt_record in action_state.receipts:
            bound_intent = intents.get(receipt_record.intent_id)
            if (
                bound_intent is None
                or not receipt_matches_intent(receipt_record, bound_intent)
                or (
                    receipt_record.compensation_of is not None
                    and receipt_record.compensation_of not in receipts
                )
            ):
                raise ValueError("protected receipt binding is invalid")
            if receipt_record.observation_id is not None and (
                receipt_record.observation_id not in observations
                or observations[receipt_record.observation_id].receipt_id
                != receipt_record.receipt_id
                or observations[receipt_record.observation_id].intent_id
                != receipt_record.intent_id
            ):
                raise ValueError("protected receipt observation binding is invalid")
            if receipt_record.verification_id is not None and (
                receipt_record.verification_id not in verifications
                or verifications[receipt_record.verification_id].intent_id
                != receipt_record.intent_id
            ):
                raise ValueError("protected receipt verification binding is invalid")
        for observation_record in action_state.observations:
            bound_receipt = receipts.get(observation_record.receipt_id)
            if (
                observation_record.intent_id not in intents
                or bound_receipt is None
                or bound_receipt.intent_id != observation_record.intent_id
            ):
                raise ValueError("protected observation binding is invalid")
        for verification_record in action_state.verifications:
            bound_observation = (
                None
                if verification_record.observation_id is None
                else observations.get(verification_record.observation_id)
            )
            if verification_record.intent_id not in intents or (
                verification_record.observation_id is not None
                and (
                    bound_observation is None
                    or bound_observation.intent_id != verification_record.intent_id
                )
            ):
                raise ValueError("protected verification binding is invalid")
        active_statuses = {
            IntentStatus.AWAITING_APPROVAL,
            IntentStatus.APPROVED,
            IntentStatus.EXECUTING,
            IntentStatus.RETRY_PENDING,
        }
        notification_statuses = {
            "queued",
            "pending",
            "delivering",
            "retry_pending",
            "executing",
            "delivered",
            "cancelled",
            "expired",
            "failed",
        }
        notification_ids: set[Any] = set()
        notification_keys: set[Any] = set()
        for notification in action_state.notifications:
            if (
                set(notification)
                != {
                    "notification_id",
                    "idempotency_key",
                    "channel",
                    "title",
                    "body",
                    "status",
                    "created_at",
                }
                or notification.get("status") not in notification_statuses
                or notification.get("channel") != "local"
                or not isinstance(notification.get("notification_id"), str)
                or not isinstance(notification.get("idempotency_key"), str)
            ):
                raise ValueError("protected action notification is invalid")
            notification_id = notification["notification_id"]
            idempotency_key = notification["idempotency_key"]
            if (
                notification_id in notification_ids
                or idempotency_key in notification_keys
                or sum(
                    intent.idempotency_key == idempotency_key
                    and intent.tool_name == "local_notification_enqueue"
                    for intent in action_state.intents
                )
                != 1
            ):
                raise ValueError("protected action notification binding is invalid")
            notification_ids.add(notification_id)
            notification_keys.add(idempotency_key)
        return (
            any(item.status in active_statuses for item in action_state.intents)
            or any(item.status == "pending" for item in action_state.approvals)
            or any(
                item["status"]
                in {"queued", "pending", "delivering", "retry_pending", "executing"}
                for item in action_state.notifications
            )
        )
    if key == "proactive_outbox":
        from kagya.outbox import (
            AcknowledgmentStatus,
            DeliveryStatus,
            OutboxState,
        )

        if value.get("schema_version") != 1:
            raise ValueError("protected outbox state version is invalid")
        outbox_state = OutboxState.model_validate(value)
        identifiers = {item.message_id for item in outbox_state.messages}
        if len(identifiers) != len(outbox_state.messages):
            raise ValueError("protected outbox identifiers must be unique")
        return any(
            item.delivery_status in {DeliveryStatus.PENDING, DeliveryStatus.FAILED}
            and item.acknowledgment_status == AcknowledgmentStatus.UNACKNOWLEDGED
            for item in outbox_state.messages
        )
    if key == "subject_scheduler":
        from kagya.runtime.autonomy import ScheduleStatus, WakeUpSchedule

        if value.get("schema_version") != 1 or set(value) != {
            "schema_version",
            "schedules",
        }:
            raise ValueError("protected scheduler state is invalid")
        raw_schedules = value.get("schedules")
        if not isinstance(raw_schedules, list):
            raise ValueError("protected scheduler state is invalid")
        schedules = [WakeUpSchedule.model_validate(item) for item in raw_schedules]
        identifiers = {item.schedule_id for item in schedules}
        if len(identifiers) != len(schedules):
            raise ValueError("protected scheduler identifiers must be unique")
        return any(item.status == ScheduleStatus.PENDING for item in schedules)
    raise ValueError("unknown protected domain")


def _has_duplicate_entity_ids(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_duplicate_entity_ids(item) for item in value.values())
    if not isinstance(value, list):
        return False
    for key in _ENTITY_ID_KEYS:
        identifiers = [
            item.get(key)
            for item in value
            if isinstance(item, dict) and isinstance(item.get(key), str)
        ]
        if len(identifiers) != len(set(identifiers)):
            return True
    return any(_has_duplicate_entity_ids(item) for item in value)


class OperatorRestoreService:
    def __init__(
        self,
        state_store: AgentStateStore,
        wal: StateWAL,
        journal: EventJournal,
        external: ExternalTransactionCoordinator,
        *,
        ttl_seconds: int = 300,
        authority: Callable[[str], bool] | None = None,
        max_targets: int = 100,
        clock: Callable[[], datetime] | None = None,
        external_heads: Callable[[], Any] | None = None,
    ) -> None:
        self.state_store, self.wal, self.journal, self.external = (
            state_store,
            wal,
            journal,
            external,
        )
        self.ttl_seconds, self.authority, self.max_targets, self.clock = (
            ttl_seconds,
            authority,
            max_targets,
            clock or (lambda: datetime.now(UTC)),
        )
        self.external_heads = external_heads or (lambda: {})
        self._lock = Lock()
        self._capabilities: dict[str, _PreviewCapability] = {}
        self._consumed: set[str] = set()
        self._operations: list[RestoreOperation] = []
        self._semantic_cache: tuple[int, str, int] | None = None

    def _fail(self, code: RestoreErrorCode) -> None:
        raise RestoreContractError(code)

    def _assert_authoritative(self, actor_id: str) -> None:
        if self.authority and not self.authority(actor_id):
            self._fail(RestoreErrorCode.NOT_AUTHORITATIVE)

    def _evict_capabilities_locked(self) -> None:
        now = self.clock()
        expired = [
            digest
            for digest, capability in self._capabilities.items()
            if not capability.reserved
            and now >= datetime.fromisoformat(capability.preview.expires_at)
        ]
        for digest in expired:
            self._capabilities.pop(digest, None)
        if len(self._capabilities) > 512:
            ordered = sorted(
                (
                    digest
                    for digest, capability in self._capabilities.items()
                    if not capability.reserved
                ),
                key=lambda digest: self._capabilities[digest].preview.created_at,
            )
            for digest in ordered[: max(0, len(self._capabilities) - 512)]:
                self._capabilities.pop(digest, None)
        while len(self._consumed) > 512:
            self._consumed.pop()

    def _evidence(self) -> tuple[list[Any], list[Any]]:
        try:
            wal, journal = self.wal.verify(), self.journal.verify()
        except (StateWalIntegrityError, OSError, ValueError) as exc:
            raise RestoreContractError(RestoreErrorCode.WAL_INTEGRITY_INVALID) from exc
        except JournalIntegrityError as exc:
            raise RestoreContractError(
                RestoreErrorCode.JOURNAL_INTEGRITY_INVALID
            ) from exc
        return wal, journal

    def _current(self) -> AgentStateSnapshot:
        current = self.state_store.last_snapshot
        if current is None:
            self._fail(RestoreErrorCode.WAL_INTEGRITY_INVALID)
            raise AssertionError
        return current

    def _retention_floor(
        self, wal: list[StateWalRecord], journal: list[JournalRecord]
    ) -> int:
        if not wal:
            return 0
        oldest = journal[0] if journal else None
        if oldest is None or oldest.lifecycle != JournalLifecycle.CHECKPOINT:
            return wal[0].processing_sequence
        if oldest.snapshot_sequence is None or oldest.snapshot_hash is None:
            self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
        assert oldest.snapshot_sequence is not None
        return max(wal[0].processing_sequence, oldest.snapshot_sequence)

    def _semantic_anchor(self, wal: list[StateWalRecord], start: int) -> int:
        with self._lock:
            cached = self._semantic_cache
        if cached is not None:
            sequence, record_hash, revision = cached
            cached_index = bisect_left(
                wal, sequence, key=lambda record: record.processing_sequence
            )
            if (
                cached_index < len(wal)
                and wal[cached_index].processing_sequence == sequence
                and wal[cached_index].record_hash == record_hash
            ):
                return revision
        read_only = {
            "state_snapshot",
            "state_export",
            "backup_create",
            "backup_verify",
            "backup_rotate",
            "behavioral_evaluate",
        }
        for index in range(start, 0, -1):
            record = wal[index]
            if record.event_type == "state_point_in_time_restore":
                return record.processing_sequence
            if record.event_type.endswith("_read") or record.event_type in read_only:
                continue
            if logical_state_digest(record.patch.value) != logical_state_digest(
                wal[index - 1].patch.value
            ):
                return record.processing_sequence
        return 0

    def _analyze_evidence(
        self,
        wal: list[StateWalRecord],
        journal: list[JournalRecord],
    ) -> _EvidenceIndex:
        if not wal:
            self._fail(RestoreErrorCode.WAL_INTEGRITY_INVALID)
        floor = self._retention_floor(wal, journal)
        evidence: dict[int, _SequenceEvidence] = {}
        for item in journal:
            sequence = item.snapshot_sequence
            if sequence is None or sequence < floor or item.snapshot_hash is None:
                continue
            entry = evidence.setdefault(sequence, _SequenceEvidence())
            if item.lifecycle == JournalLifecycle.COMPLETED:
                entry.eligible.append((item, "journal_completed"))
            elif (
                item.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and item.failure_category == "committed_before_crash"
            ):
                entry.eligible.append((item, "journal_recovered"))
            elif item.lifecycle == JournalLifecycle.CHECKPOINT:
                entry.eligible.append((item, "checkpoint"))
            elif item.lifecycle == JournalLifecycle.FAILED or (
                item.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and item.failure_category == "uncommitted_after_crash"
                and item.processing_sequence == item.snapshot_sequence
            ):
                entry.non_target.append(item)

        start = bisect_left(wal, floor, key=lambda record: record.processing_sequence)
        if start >= len(wal) or wal[start].processing_sequence != floor:
            self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
        restorable_by_sequence: dict[int, _VerifiedTarget] = {}
        semantic_targets: list[_VerifiedTarget] = []
        projected_semantic_counts: set[int] = set()
        latest_restorable: _VerifiedTarget | None = None
        previous_digest: str | None = None
        semantic_revision = self._semantic_anchor(wal, start)
        semantic_count = 0
        priority = {"journal_completed": 0, "journal_recovered": 1, "checkpoint": 2}
        for record in wal[start:]:
            digest = logical_state_digest(record.patch.value)
            semantic_boundary = previous_digest is not None and (
                digest != previous_digest
                or record.event_type == "state_point_in_time_restore"
            )
            if semantic_boundary:
                semantic_count += 1
                semantic_revision = record.processing_sequence
            previous_digest = digest

            if (
                record.processing_sequence == 0
                and record.event_type == "state_snapshot"
            ):
                bootstrap_evidence = evidence.pop(0) if 0 in evidence else None
                journal_record: StateWalRecord | JournalRecord = record
                if bootstrap_evidence is not None:
                    hashes = {
                        item.snapshot_hash
                        for item, _kind in bootstrap_evidence.eligible
                    } | {item.snapshot_hash for item in bootstrap_evidence.non_target}
                    if (
                        len(hashes) != 1
                        or record.state_hash_after not in hashes
                        or bootstrap_evidence.non_target
                        or not bootstrap_evidence.eligible
                    ):
                        self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
                    journal_record, _checkpoint_kind = min(
                        bootstrap_evidence.eligible,
                        key=lambda candidate: priority[candidate[1]],
                    )
                target = _VerifiedTarget(
                    wal=record,
                    journal=journal_record,
                    kind="bootstrap",
                    semantic_count=semantic_count,
                )
                restorable_by_sequence[record.processing_sequence] = target
                semantic_targets.append(target)
                projected_semantic_counts.add(semantic_count)
                latest_restorable = target
                continue
            sequence_evidence = (
                evidence.pop(record.processing_sequence)
                if record.processing_sequence in evidence
                else None
            )
            if sequence_evidence is None:
                self._fail(RestoreErrorCode.TARGET_UNVERIFIED)
            assert sequence_evidence is not None
            hashes = {
                item.snapshot_hash for item, _kind in sequence_evidence.eligible
            } | {item.snapshot_hash for item in sequence_evidence.non_target}
            if len(hashes) != 1 or record.state_hash_after not in hashes:
                self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
            event_evidence = [
                item
                for item, kind in sequence_evidence.eligible
                if kind != "checkpoint"
            ] + sequence_evidence.non_target
            if any(
                item.event_id != record.event_id
                or item.processing_sequence != record.processing_sequence
                for item in event_evidence
            ):
                self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
            restorable_evidence = [
                item for item in sequence_evidence.eligible if item[1] != "checkpoint"
            ]
            if restorable_evidence and sequence_evidence.non_target:
                self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
            if sequence_evidence.non_target:
                continue
            if not sequence_evidence.eligible:
                self._fail(RestoreErrorCode.TARGET_UNVERIFIED)
            journal_record, kind = min(
                sequence_evidence.eligible,
                key=lambda candidate: priority[candidate[1]],
            )
            target = _VerifiedTarget(
                wal=record,
                journal=journal_record,
                kind=kind,
                semantic_count=semantic_count,
            )
            restorable_by_sequence[record.processing_sequence] = target
            if semantic_count not in projected_semantic_counts:
                semantic_targets.append(target)
                projected_semantic_counts.add(semantic_count)
            latest_restorable = target
        if evidence:
            self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
        if latest_restorable is not None and (
            not semantic_targets
            or semantic_targets[-1].wal.processing_sequence
            != latest_restorable.wal.processing_sequence
        ):
            semantic_targets.append(latest_restorable)
        with self._lock:
            self._semantic_cache = (
                wal[-1].processing_sequence,
                wal[-1].record_hash,
                semantic_revision,
            )
        return _EvidenceIndex(
            floor=floor,
            maximum=wal[-1].processing_sequence,
            semantic_revision=semantic_revision,
            semantic_transition_count=semantic_count,
            restorable_by_sequence=restorable_by_sequence,
            semantic_targets=tuple(reversed(semantic_targets)),
        )

    def _domains(
        self, before: AgentStateSnapshot, after: AgentStateSnapshot
    ) -> tuple[list[RestoreDomain], bool]:
        supported = True
        domains: list[RestoreDomain] = []
        for name in (
            "emotion_state",
            "working_memory",
            "motivation",
            "identity",
            "context_state",
        ):
            left, right = (
                getattr(before, name).model_dump(mode="json"),
                getattr(after, name).model_dump(mode="json"),
            )
            left_records = _core_domain_records(name, left)
            right_records = _core_domain_records(name, right)
            b, a, added, removed, changed = _counts(
                left_records,
                right_records,
            )
            if changed or added or removed:
                refs, truncated = _safe_refs(
                    _changed_record_values(left_records, right_records)
                )
                domains.append(
                    RestoreDomain(
                        domain=cast(Any, name),
                        before_count=b,
                        after_count=a,
                        added_count=added,
                        removed_count=removed,
                        changed_count=changed,
                        changed_revision_count=changed,
                        newer_state_loss_count=removed + changed,
                        refs=refs,
                        truncated=truncated,
                        reason_code=None,
                    )
                )
            nested_before = left.get("extensions", {}) or {}
            nested_after = right.get("extensions", {}) or {}
            nested = set(nested_before) | set(nested_after)
            allowed_nested = _NESTED_EXTENSIONS.get(name)
            if allowed_nested is not None and any(
                key not in allowed_nested for key in nested
            ):
                supported = False
            if _has_duplicate_entity_ids([left, right]):
                supported = False
        keys = set(before.extensions) | set(after.extensions)
        if any(key not in _EXTENSIONS and key != _RESERVED for key in keys):
            supported = False
            domains.append(
                RestoreDomain(
                    domain="extensions",
                    before_count=0,
                    after_count=0,
                    added_count=0,
                    removed_count=0,
                    changed_count=0,
                    changed_revision_count=0,
                    newer_state_loss_count=0,
                    refs=[],
                    truncated=False,
                    reason_code=RestoreErrorCode.UNSUPPORTED_DOMAIN.value,
                )
            )
        for key in sorted(keys):
            if key == _RESERVED or key not in _EXTENSIONS:
                continue
            if any(
                value is not None and not isinstance(value, (dict, list))
                for value in (before.extensions.get(key), after.extensions.get(key))
            ):
                supported = False
            if _has_duplicate_entity_ids(
                [before.extensions.get(key), after.extensions.get(key)]
            ):
                supported = False
            protected_domain_blocked = False
            if key in _EXTERNAL_REARM_DOMAINS:
                try:
                    protected_domain_blocked = any(
                        _protected_domain_has_active_work(key, value)
                        for value in (
                            before.extensions.get(key),
                            after.extensions.get(key),
                        )
                    )
                except (TypeError, ValueError):
                    protected_domain_blocked = True
                if protected_domain_blocked:
                    supported = False
            before_records = _extension_records(before.extensions.get(key))
            after_records = _extension_records(after.extensions.get(key))
            b, a, added, removed, changed = _counts(before_records, after_records)
            if changed or added or removed:
                refs, truncated = _safe_refs(
                    _changed_record_values(before_records, after_records)
                )
                domains.append(
                    RestoreDomain(
                        domain=cast(Any, _EXTENSIONS[key]),
                        before_count=b,
                        after_count=a,
                        added_count=added,
                        removed_count=removed,
                        changed_count=changed,
                        changed_revision_count=changed,
                        newer_state_loss_count=removed + changed,
                        refs=refs,
                        truncated=truncated,
                        reason_code=None,
                    )
                )
                if protected_domain_blocked:
                    supported = False
        return domains, supported

    def _external(self, target: int, *, nonce: str) -> tuple[RestoreExternal, str]:
        try:
            records, diff, reconciliation = self.external.restore_view(target)
            effects = diff.effects
            pending = sum(
                r.status == ExternalTransactionStatus.PENDING for r in records
            )
            orphaned = sum(
                r.status == ExternalTransactionStatus.ORPHANED for r in records
            )
            retryable = reconciliation.retryable
            invalid = any(
                _public_artifact_type(r.artifact_type) is None
                or not isinstance(r.status, ExternalTransactionStatus)
                or (
                    r.status
                    in {
                        ExternalTransactionStatus.PENDING,
                        ExternalTransactionStatus.ORPHANED,
                    }
                    and r.processing_sequence is None
                )
                for r in records
            )
            bad = pending > 0 or orphaned > 0 or retryable > 0 or invalid
            canonical = {
                "records": [r.model_dump(mode="json") for r in records],
                "reconciliation": reconciliation.model_dump(mode="json"),
                "authoritative_heads": self.external_heads(),
            }
            canonical_digest = _hash(canonical)
            artifacts: list[RestoreArtifact] = []
            if not bad:
                public_types = {
                    _public_artifact_type(effect.artifact_type) for effect in effects
                }
                for kind in sorted(item for item in public_types if item is not None):
                    refs = [
                        _hash([nonce, kind, effect.artifact_id])
                        for effect in effects
                        if _public_artifact_type(effect.artifact_type) == kind
                    ]
                    artifacts.append(
                        RestoreArtifact(
                            artifact_type=cast(Any, kind),
                            count=len(refs),
                            refs=refs[:256],
                            truncated=len(refs) > 256,
                        )
                    )
            return (
                RestoreExternal(
                    consistency_status="inconsistent" if bad else "consistent",
                    artifacts=artifacts,
                    retained_not_replayed_count=len(effects),
                    pending_count=pending,
                    orphaned_count=orphaned,
                    retryable_count=retryable,
                    effect_digest=_hash([nonce, canonical_digest, target]),
                ),
                canonical_digest,
            )
        except Exception as exc:
            raise RestoreContractError(
                RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT
            ) from exc

    def preview(
        self, target_sequence: int, actor_id: str = "operator"
    ) -> RestorePreview:
        self._assert_authoritative(actor_id)
        wal, journal = self._evidence()
        analysis = self._analyze_evidence(wal, journal)
        target = analysis.restorable_by_sequence.get(target_sequence)
        if target is None:
            if target_sequence < analysis.floor:
                self._fail(RestoreErrorCode.TARGET_NOT_RETAINED)
            if any(r.processing_sequence == target_sequence for r in wal):
                self._fail(RestoreErrorCode.TARGET_UNVERIFIED)
            self._fail(RestoreErrorCode.TARGET_NOT_RETAINED)
        assert target is not None
        record = target.wal
        current = self._current()
        domains, supported = self._domains(current, record.patch.value)
        nonce = str(uuid4())
        external, canonical_external_digest = self._external(
            target_sequence, nonce=nonce
        )
        reasons = ([] if supported else [RestoreErrorCode.UNSUPPORTED_DOMAIN.value]) + (
            []
            if external.consistency_status == "consistent"
            else [RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT.value]
        )
        now = self.clock()
        operation_id = str(uuid4())
        phrase = f"RESTORE {target_sequence} {record.state_hash_after[:16]} {_hash([operation_id, logical_state_digest(current)])[:16]}"
        raw: dict[str, Any] = {
            "schema_version": 1,
            "operation_id": operation_id,
            "preview_digest": "0" * 64,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.ttl_seconds)).isoformat(),
            "current_logical_digest": logical_state_digest(current),
            "semantic_revision": analysis.semantic_revision,
            "display_sequence": current.last_processed_event_sequence,
            "target_sequence": target_sequence,
            "target_snapshot_hash": record.state_hash_after,
            "newer_authoritative_event_count": (
                analysis.semantic_transition_count - target.semantic_count
            ),
            "domains": domains,
            "external_effects": external,
            "restoreable": not reasons,
            "reason_codes": reasons,
            "confirmation_phrase": phrase,
        }
        digest = _hash({k: v for k, v in raw.items() if k != "preview_digest"})
        preview = RestorePreview.model_validate({**raw, "preview_digest": digest})
        with self._lock:
            self._evict_capabilities_locked()
            if len(self._capabilities) >= 512 and all(
                capability.reserved for capability in self._capabilities.values()
            ):
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            self._capabilities[digest] = _PreviewCapability(
                preview=preview,
                canonical_external_digest=canonical_external_digest,
            )
            self._evict_capabilities_locked()
        return preview

    def _check_request(
        self,
        preview: RestorePreview,
        request: RestoreCommitRequest,
        *,
        stale: bool = False,
        allow_expired: bool = False,
    ) -> None:
        if RestoreErrorCode.UNSUPPORTED_DOMAIN.value in preview.reason_codes:
            self._fail(RestoreErrorCode.UNSUPPORTED_DOMAIN)
        if RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT.value in preview.reason_codes:
            self._fail(RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT)
        if not allow_expired and self.clock() >= datetime.fromisoformat(
            preview.expires_at
        ):
            self._fail(RestoreErrorCode.PREVIEW_EXPIRED)
        if request.confirmation_phrase != preview.confirmation_phrase:
            self._fail(RestoreErrorCode.CONFIRMATION_REQUIRED)
        if (
            stale
            or request.target_sequence != preview.target_sequence
            or request.expected_target_hash != preview.target_snapshot_hash
            or request.expected_semantic_revision != preview.semantic_revision
            or request.expected_current_logical_digest != preview.current_logical_digest
            or request.expected_preview_digest != preview.preview_digest
            or request.expected_external_effect_digest
            != preview.external_effects.effect_digest
        ):
            self._fail(RestoreErrorCode.PREVIEW_STALE)

    def preflight(self, request: RestoreCommitRequest, actor_id: str) -> RestorePreview:
        self._assert_authoritative(actor_id)
        with self._lock:
            if request.expected_preview_digest in self._consumed:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            entry = self._capabilities.get(request.expected_preview_digest)
            if entry is None:
                self._fail(RestoreErrorCode.PREVIEW_STALE)
            assert entry is not None
            preview = entry.preview
            if entry.reserved:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
        self._check_request(preview, request)
        wal, journal = self._evidence()
        analysis = self._analyze_evidence(wal, journal)
        verified_target = analysis.restorable_by_sequence.get(request.target_sequence)
        target = None if verified_target is None else verified_target.wal
        if target is None or target.state_hash_after != request.expected_target_hash:
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        current = self._current()
        current_external, canonical_external_digest = self._external(
            request.target_sequence, nonce=str(uuid4())
        )
        if current_external.consistency_status != "consistent":
            self._fail(RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT)
        if (
            logical_state_digest(current) != request.expected_current_logical_digest
            or analysis.semantic_revision != request.expected_semantic_revision
            or canonical_external_digest != entry.canonical_external_digest
        ):
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        with self._lock:
            entry = self._capabilities.get(request.expected_preview_digest)
            if entry is None:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            assert entry is not None
            if entry.reserved:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            self._capabilities[request.expected_preview_digest] = _PreviewCapability(
                preview=entry.preview,
                canonical_external_digest=entry.canonical_external_digest,
                reserved=True,
            )
        return preview

    def reserve(self, preview: RestorePreview, actor_id: str) -> RestorePreview:
        # Kept for callers that split preflight and reservation; reservation itself is atomic.
        request = RestoreCommitRequest(
            target_sequence=preview.target_sequence,
            expected_target_hash=preview.target_snapshot_hash,
            expected_semantic_revision=preview.semantic_revision,
            expected_current_logical_digest=preview.current_logical_digest,
            expected_preview_digest=preview.preview_digest,
            expected_external_effect_digest=preview.external_effects.effect_digest,
            confirmation_phrase=preview.confirmation_phrase,
        )
        return self.preflight(request, actor_id)

    def release(self, preview_digest: str) -> None:
        with self._lock:
            self._capabilities.pop(preview_digest, None)

    def apply_commit(
        self, main_loop: Any, request: RestoreCommitRequest, actor_id: str
    ) -> RestoreOperation:
        self._assert_authoritative(actor_id)
        with self._lock:
            if request.expected_preview_digest in self._consumed:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            entry = self._capabilities.get(request.expected_preview_digest)
            if entry is None:
                self._fail(RestoreErrorCode.PREVIEW_STALE)
            assert entry is not None
            preview = entry.preview
            if not entry.reserved:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
        self._check_request(preview, request, allow_expired=True)
        wal, journal = self._evidence()
        analysis = self._analyze_evidence(wal, journal)
        verified_target = analysis.restorable_by_sequence.get(request.target_sequence)
        target_record = None if verified_target is None else verified_target.wal
        if (
            target_record is None
            or target_record.state_hash_after != request.expected_target_hash
        ):
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        current = self._current()
        current_external, canonical_external_digest = self._external(
            request.target_sequence, nonce=str(uuid4())
        )
        if current_external.consistency_status != "consistent":
            self._fail(RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT)
        if (
            logical_state_digest(current) != request.expected_current_logical_digest
            or analysis.semantic_revision != request.expected_semantic_revision
            or canonical_external_digest != entry.canonical_external_digest
        ):
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        event = current_agent_event()
        if (
            event is None
            or event.processing_sequence is None
            or event.event_id != event_id_for_operation(preview.operation_id)
        ):
            self._fail(RestoreErrorCode.NOT_AUTHORITATIVE)
        assert event is not None and event.processing_sequence is not None
        with self._lock:
            self._consumed.add(request.expected_preview_digest)
            self._capabilities.pop(request.expected_preview_digest, None)
            self._evict_capabilities_locked()
        wal, _ = self._evidence()
        current = self._current()
        target = self.wal.reconstruct(request.target_sequence).snapshot
        operation = RestoreOperation(
            operation_id=preview.operation_id,
            target_sequence=request.target_sequence,
            target_snapshot_hash=request.expected_target_hash,
            preview_digest=request.expected_preview_digest,
            requested_at=preview.created_at,
            started_at=self.clock().isoformat(),
            completed_at=None,
            event_id=event.event_id,
            processing_sequence=event.processing_sequence,
            state="finalizing",
            error_code=None,
        )
        with self._lock:
            self._operations.append(operation)
            self._operations = self._operations[-50:]
        prior = current
        register_event_rollback(lambda: self.state_store.restore_into(main_loop, prior))
        old = (
            target.extensions.get(_RESERVED, [])
            if isinstance(target.extensions.get(_RESERVED, []), list)
            else []
        )
        current_history = (
            current.extensions.get(_RESERVED, [])
            if isinstance(current.extensions.get(_RESERVED, []), list)
            else []
        )
        merged: dict[str, dict[str, Any]] = {}
        for item in [*old, *current_history, operation.model_dump(mode="json")]:
            if isinstance(item, dict) and isinstance(item.get("operation_id"), str):
                merged[item["operation_id"]] = item
        target = target.model_copy(
            update={
                "extensions": {
                    **target.extensions,
                    _RESERVED: list(merged.values())[-50:],
                }
            }
        )
        self.state_store.restore_into(main_loop, target)
        return operation

    def build_completion_response(
        self, operation_id: str
    ) -> RestoreCommitResponse | None:
        operation = next(
            (o for o in reversed(self._operations) if o.operation_id == operation_id),
            None,
        )
        if operation is None or operation.processing_sequence is None:
            return None
        current = self._current()
        return RestoreCommitResponse(
            disposition="completed",
            operation_id=operation_id,
            event_id=operation.event_id,
            processing_sequence=operation.processing_sequence,
            restored_target_sequence=operation.target_sequence,
            restored_target_hash=operation.target_snapshot_hash,
            post_restore_sequence=current.last_processed_event_sequence,
            post_restore_hash=hash_snapshot(current),
            operation_status="completed",
            error_code=None,
        )

    def indeterminate_projection(self, operation_id: str) -> RestoreOperation | None:
        operation = next(
            (o for o in reversed(self._operations) if o.operation_id == operation_id),
            None,
        )
        return (
            None
            if operation is None
            else operation.model_copy(
                update={
                    "state": "commit_indeterminate",
                    "error_code": RestoreErrorCode.COMMIT_INDETERMINATE.value,
                }
            )
        )

    def _journal_operations(self, journal: list[Any]) -> list[RestoreOperation]:
        grouped: dict[str, list[Any]] = {}
        for item in journal:
            if item.event_type == "state_point_in_time_restore" and re.fullmatch(
                r"operator-restore-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                item.event_id,
            ):
                grouped.setdefault(item.event_id, []).append(item)
        operations: list[RestoreOperation] = []
        for event_id, records in grouped.items():
            if not any(
                item.source == "api.state.operator_restore.commit" for item in records
            ):
                continue
            if any(
                item.source
                not in {"api.state.operator_restore.commit", "journal.recovery"}
                for item in records
            ):
                self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
            operation_id = event_id.removeprefix("operator-restore-")
            target_values = {item.target for item in records if item.target is not None}
            preview_values = {
                item.correlation_id
                for item in records
                if item.correlation_id is not None
            }
            if len(target_values) != 1 or len(preview_values) != 1:
                self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
            target_match = re.fullmatch(
                r"restore:(\d+):([0-9a-f]{64})", next(iter(target_values))
            )
            preview_digest = next(iter(preview_values))
            processing_sequences = {
                item.processing_sequence
                for item in records
                if item.processing_sequence is not None
            }
            if target_match is None or not re.fullmatch(
                r"[0-9a-f]{64}", preview_digest
            ):
                self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
            assert target_match is not None
            if len(processing_sequences) > 1:
                self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
            accepted = next(
                (
                    item
                    for item in records
                    if item.lifecycle == JournalLifecycle.ACCEPTED
                ),
                records[0],
            )
            started = next(
                (
                    item
                    for item in records
                    if item.lifecycle == JournalLifecycle.STARTED
                ),
                None,
            )
            operations.append(
                RestoreOperation(
                    operation_id=operation_id,
                    target_sequence=int(target_match.group(1)),
                    target_snapshot_hash=target_match.group(2),
                    preview_digest=preview_digest,
                    requested_at=accepted.timestamp.isoformat(),
                    started_at=(
                        None if started is None else started.timestamp.isoformat()
                    ),
                    completed_at=None,
                    event_id=event_id,
                    processing_sequence=(
                        next(iter(processing_sequences))
                        if processing_sequences
                        else None
                    ),
                    state="finalizing",
                    error_code=None,
                )
            )
        return operations[-50:]

    def summary(
        self, limit: int | None = None, actor_id: str = "operator"
    ) -> RestoreSummary:
        self._assert_authoritative(actor_id)
        wal, journal = self._evidence()
        analysis = self._analyze_evidence(wal, journal)
        current = self._current()
        display_limit = min(limit or self.max_targets, self.max_targets)
        targets = [
            RestoreTarget(
                target_sequence=item.wal.processing_sequence,
                target_snapshot_hash=item.wal.state_hash_after,
                checkpoint_kind=cast(Any, item.kind),
                timestamp=item.journal.timestamp.isoformat(),
                event_type=item.journal.event_type,
                eligible=True,
                reason_codes=[],
            )
            for item in analysis.semantic_targets[:display_limit]
        ]
        operations_by_id = {item.operation_id: item for item in self._operations}
        operations_by_id.update(
            {item.operation_id: item for item in self._journal_operations(journal)}
        )
        raw = current.extensions.get(_RESERVED, [])
        persisted_operation_ids: set[str] = set()
        if raw is not None and not isinstance(raw, list):
            self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
        if isinstance(raw, list):
            for item in raw:
                try:
                    operation = RestoreOperation.model_validate(item)
                    operations_by_id[operation.operation_id] = operation
                    persisted_operation_ids.add(operation.operation_id)
                except Exception as exc:
                    raise RestoreContractError(
                        RestoreErrorCode.JOURNAL_INTEGRITY_INVALID
                    ) from exc
        operations = list(operations_by_id.values())
        by_event = {j.event_id: j for j in journal}
        journal_anchor: dict[str, tuple[int, int]] = {}
        latest_journal_sequence = -1
        for ordinal, item in enumerate(journal):
            if item.processing_sequence is not None:
                latest_journal_sequence = max(
                    latest_journal_sequence, item.processing_sequence
                )
            anchor = (latest_journal_sequence, ordinal)
            if item.lifecycle == JournalLifecycle.ACCEPTED:
                journal_anchor[item.event_id] = anchor
            else:
                journal_anchor.setdefault(item.event_id, anchor)
        projected: list[RestoreOperation] = []
        for operation in operations:
            evidence = by_event.get(operation.event_id)
            if evidence and evidence.lifecycle == JournalLifecycle.COMPLETED:
                projected.append(
                    operation.model_copy(
                        update={
                            "state": "completed",
                            "completed_at": evidence.timestamp.isoformat(),
                        }
                    )
                )
            elif (
                evidence
                and evidence.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and evidence.failure_category == "committed_before_crash"
            ):
                projected.append(
                    operation.model_copy(
                        update={
                            "state": "completed",
                            "completed_at": evidence.timestamp.isoformat(),
                        }
                    )
                )
            elif evidence and evidence.lifecycle == JournalLifecycle.FAILED:
                projected.append(
                    operation.model_copy(
                        update={
                            "state": "failed",
                            "error_code": "restore_commit_failed",
                        }
                    )
                )
            elif (
                evidence
                and evidence.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and evidence.failure_category
                in {
                    "uncommitted_after_crash",
                    "started_without_sequence",
                    "accepted_not_started",
                }
            ):
                projected.append(
                    operation.model_copy(
                        update={
                            "state": "failed",
                            "completed_at": evidence.timestamp.isoformat(),
                            "error_code": "restore_commit_failed",
                        }
                    )
                )
            elif (
                operation.operation_id in persisted_operation_ids
                and operation.processing_sequence is not None
                and operation.processing_sequence < analysis.floor
            ):
                projected.append(
                    operation.model_copy(
                        update={"state": "completed", "error_code": None}
                    )
                    if operation.state == "finalizing"
                    else operation
                )
            else:
                projected.append(
                    operation.model_copy(
                        update={
                            "state": "commit_indeterminate",
                            "error_code": RestoreErrorCode.COMMIT_INDETERMINATE.value,
                        }
                    )
                )

        def operation_timestamp(operation: RestoreOperation) -> datetime:
            return max(
                datetime.fromisoformat(timestamp).astimezone(UTC)
                for timestamp in (
                    operation.requested_at,
                    operation.started_at,
                    operation.completed_at,
                )
                if timestamp is not None
            )

        def operation_chronology(
            operation: RestoreOperation,
        ) -> tuple[int, int, datetime, str]:
            anchor_sequence, ordinal = journal_anchor.get(operation.event_id, (-1, -1))
            return (
                (
                    anchor_sequence
                    if ordinal >= 0
                    else operation.processing_sequence or -1
                ),
                ordinal,
                operation_timestamp(operation),
                operation.operation_id,
            )

        latest_operation = max(projected, key=operation_chronology, default=None)
        return RestoreSummary(
            current_sequence=current.last_processed_event_sequence,
            current_snapshot_hash=hash_snapshot(current),
            current_logical_digest=logical_state_digest(current),
            semantic_revision=analysis.semantic_revision,
            retained_min_sequence=analysis.floor,
            retained_max_sequence=analysis.maximum,
            targets=targets,
            latest_operation=latest_operation,
        )
