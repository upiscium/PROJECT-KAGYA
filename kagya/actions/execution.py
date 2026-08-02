"""Governed action intents and bounded local execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any, Literal, NoReturn, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from kagya.decision import DecisionStatus
from kagya.outbox import (
    OutboxMessageKind,
    OutboxReferences,
    OutboxUrgency,
    PrivacyClass,
)
from kagya.runtime.agent_runtime import current_agent_event


ACTION_STATE_KEY = "action_execution"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskClass(StrEnum):
    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"
    HIGH_IMPACT = "high_impact"


class OperatorCommand(StrEnum):
    """Commands that an operator may request for an action intent."""

    APPROVE = "approve"
    REJECT = "reject"
    CANCEL = "cancel"
    RETRY_NOW = "retry_now"
    COMPENSATE = "compensate"


class _PublicSummary(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str


class MetadataPublicSummary(_PublicSummary):
    kind: Literal["metadata"] = "metadata"
    namespace: str
    key: str


class DocumentPublicSummary(_PublicSummary):
    kind: Literal["document_search"] = "document_search"
    max_results: int
    query_length: int
    path_scoped: bool


class CalendarPublicSummary(_PublicSummary):
    kind: Literal["calendar_read"] = "calendar_read"
    starts_at: datetime
    ends_at: datetime
    max_results: int


class NotificationPublicSummary(_PublicSummary):
    kind: Literal["local_notification_enqueue"] = "local_notification_enqueue"
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def public_text_only(self) -> "NotificationPublicSummary":
        for value in (self.title, self.body):
            lowered = value.lower()
            if (
                any(ord(character) < 32 or ord(character) == 127 for character in value)
                or any(
                    marker in lowered
                    for marker in (
                        "private_sentinel",
                        "hidden_thought",
                        "raw_prompt",
                        "<think",
                        "</think",
                        "credential",
                        "password",
                        "api_key",
                        "api key",
                        "access_token",
                        "secret",
                    )
                )
                or re.search(r"(?:^|\s)/(?:etc|home|root|tmp|var)/", value)
            ):
                raise ValueError("notification public preview contains private text")
        return self


PublicArgumentSummary = (
    MetadataPublicSummary
    | DocumentPublicSummary
    | CalendarPublicSummary
    | NotificationPublicSummary
)


class ActionToolDescriptor(_StrictModel):
    """The complete, safe-to-render description of an allowlisted tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    tool_name: str
    risk_class: RiskClass
    effect: str
    effect_code: str
    approval_required: bool
    reversible: bool
    validation_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_preview_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    enabled: bool
    executable: bool


class ActionOperatorPreview(_StrictModel):
    """Safe operator projection; it intentionally contains no raw arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    intent_id: str
    revision: int
    tool: ActionToolDescriptor
    arguments: PublicArgumentSummary
    preview_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect: str
    effect_code: str
    available_commands: tuple[OperatorCommand, ...]
    approval_id: str | None = None

    @property
    def argument_digest(self) -> str:
        """Compatibility spelling for consumers that call summaries arguments."""
        return self.preview_digest


class OperatorCommandRequest(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intent_id: str
    command: OperatorCommand
    expected_revision: int = Field(ge=1)
    expected_preview_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_approval_id: str | None = None
    confirmation: str | None = Field(default=None, max_length=200)


class OperatorCommandValidation(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    valid: bool
    reason: str | None = None
    command: OperatorCommand
    intent_id: str


class IntentStatus(StrEnum):
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    DRY_RUN = "dry_run"
    EXECUTING = "executing"
    RETRY_PENDING = "retry_pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    COMPENSATED = "compensated"


class ReceiptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    COMPENSATED = "compensated"


class MetadataReadArguments(_StrictModel):
    namespace: Literal["project", "runtime"]
    key: Literal["name", "environment", "node_id", "model_provider"]

    @model_validator(mode="after")
    def valid_namespace_key(self) -> "MetadataReadArguments":
        allowed = {
            "project": {"name", "environment"},
            "runtime": {"node_id", "model_provider"},
        }
        if self.key not in allowed[self.namespace]:
            raise ValueError("metadata key is not valid for namespace")
        return self


class DocumentSearchArguments(_StrictModel):
    query: str = Field(min_length=1, max_length=256)
    relative_path: str | None = Field(default=None, max_length=256)
    max_results: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def safe_path(self) -> "DocumentSearchArguments":
        if self.relative_path is not None:
            path = Path(self.relative_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "document relative_path must remain under the document root"
                )
        return self


class CalendarReadArguments(_StrictModel):
    starts_at: datetime
    ends_at: datetime
    max_results: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def valid_window(self) -> "CalendarReadArguments":
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("calendar range must include timezones")
        if self.ends_at <= self.starts_at:
            raise ValueError("calendar ends_at must be after starts_at")
        if self.ends_at - self.starts_at > timedelta(days=366):
            raise ValueError("calendar range exceeds the one-year bound")
        return self


class NotificationArguments(_StrictModel):
    channel: Literal["local"] = "local"
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=1000)
    # The explicit public preview certifies the canonical executable text as
    # operator-visible; the validator below prevents a different hidden payload.
    public_preview: NotificationPublicSummary | None = None

    @model_validator(mode="after")
    def preview_is_bounded(self) -> "NotificationArguments":
        if self.public_preview is None:
            return self
        if self.public_preview.kind != "local_notification_enqueue":
            raise ValueError("notification public_preview has the wrong kind")
        if (
            self.public_preview.title != self.title
            or self.public_preview.body != self.body
        ):
            raise ValueError(
                "notification public_preview must exactly bind executable content"
            )
        return self


class ActionBudget(_StrictModel):
    max_attempts: int = Field(default=2, ge=1, le=3)
    timeout_seconds: float = Field(default=2.0, gt=0.0, le=10.0)
    max_result_bytes: int = Field(default=32768, ge=256, le=131072)
    max_cost_units: int = Field(default=1, ge=1, le=10)
    max_monetary_cost: Literal[0] = 0
    max_risk_class: Literal["read_only", "reversible_write"] = "reversible_write"


class ActionProvenance(_StrictModel):
    decision_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    triggering_event_id: str | None = None
    triggering_event_sequence: int | None = None
    plan_id: str | None = None
    plan_revision: int | None = Field(default=None, ge=1)
    step_id: str | None = None
    boundary_assessment_id: str | None = None
    boundary_assessment_revision: int | None = Field(default=None, ge=1)
    boundary_assessment_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ActionValidationErrorCode(StrEnum):
    ACTION_CONTRACT_INVALID = "action_contract_invalid"
    TOOL_NOT_ALLOWLISTED = "tool_not_allowlisted"
    RISK_CLASS_DISABLED = "risk_class_disabled"
    ARGUMENTS_SCHEMA_INVALID = "arguments_schema_invalid"
    ARGUMENT_PATH_OUT_OF_SCOPE = "argument_path_out_of_scope"
    ARGUMENT_SCOPE_INVALID = "argument_scope_invalid"


class ActionValidationRecord(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    validation_id: str
    idempotency_key: str = Field(min_length=1, max_length=128)
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_id: str | None = None
    intent_id: str | None = None
    tool_name: str
    risk_class: RiskClass | None = None
    arguments_valid: bool
    validation_schema_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_error_codes: tuple[ActionValidationErrorCode, ...] = Field(
        default=(), max_length=4
    )
    validated_event_id: str
    validated_event_sequence: int = Field(ge=1)
    canonical_arguments_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    validated_at: datetime

    @model_validator(mode="after")
    def valid_result(self) -> "ActionValidationRecord":
        if self.arguments_valid == bool(self.validation_error_codes):
            raise ValueError("Action validation result and error codes disagree")
        if len(self.validation_error_codes) != len(set(self.validation_error_codes)):
            raise ValueError("Action validation error codes must be unique")
        if (self.intent_id is not None) != self.arguments_valid:
            raise ValueError("Action validation intent binding disagrees with result")
        return self


class ActionPolicyRejectionRecord(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    rejection_id: str
    idempotency_key: str = Field(min_length=1, max_length=128)
    decision_id: str
    candidate_id: str
    validation_id: str
    risk_class: RiskClass
    policy_code: Literal["risk_budget_denied"]
    reason_code: Literal["risk_class_exceeds_budget"]
    event_id: str
    event_sequence: int = Field(ge=1)
    rejected_at: datetime


class PolicyEvaluation(_StrictModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    tool_name: str
    risk_class: RiskClass
    allowed: bool
    approval_required: bool
    reasons: tuple[str, ...]
    argument_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated_at: datetime


class ActionPreview(_StrictModel):
    tool_name: str
    risk_class: RiskClass
    arguments: dict[str, JsonValue]
    effect: str
    bounded_by: ActionBudget
    compensation_available: bool


class ApprovalRecord(_StrictModel):
    approval_id: str
    intent_id: str
    status: Literal["pending", "approved", "rejected"]
    requested_at: datetime
    resolved_at: datetime | None = None
    actor_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class ActionIntent(_StrictModel):
    schema_version: Literal[1] = 1
    intent_id: str
    revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=128)
    tool_name: str
    arguments: dict[str, JsonValue]
    risk_class: RiskClass
    status: IntentStatus
    dry_run: bool
    policy: PolicyEvaluation
    preview: ActionPreview
    provenance: ActionProvenance
    budget: ActionBudget
    attempts: int = Field(ge=0)
    cost_units_used: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime
    retry_at: datetime | None = None
    approval_id: str | None = None
    receipt_id: str | None = None
    failure_code: str | None = None
    validation_record_id: str | None = None
    explanation_refs: tuple[str, ...] = ()


class Observation(_StrictModel):
    schema_version: Literal[1] = 1
    observation_id: str
    intent_id: str
    receipt_id: str
    observed_at: datetime
    data: JsonValue
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid: bool
    validation_errors: tuple[str, ...] = ()


class OutcomeVerification(_StrictModel):
    schema_version: Literal[1] = 1
    verification_id: str
    intent_id: str
    observation_id: str | None = None
    success: bool
    reason: str
    verified_at: datetime


class ExecutionReceipt(_StrictModel):
    schema_version: Literal[1] = 1
    receipt_id: str
    intent_id: str
    idempotency_key: str
    attempt: int = Field(ge=0)
    status: ReceiptStatus
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0.0)
    observation_id: str | None = None
    verification_id: str | None = None
    event_id: str | None = None
    event_sequence: int | None = None
    decision_id: str
    plan_id: str | None = None
    plan_revision: int | None = None
    step_id: str | None = None
    compensation_of: str | None = None
    error_code: str | None = None
    explanation_refs: tuple[str, ...] = ()


class ActionState(_StrictModel):
    schema_version: Literal[4] = 4
    intents: tuple[ActionIntent, ...] = ()
    validation_records: tuple[ActionValidationRecord, ...] = ()
    policy_rejections: tuple[ActionPolicyRejectionRecord, ...] = ()
    approvals: tuple[ApprovalRecord, ...] = ()
    receipts: tuple[ExecutionReceipt, ...] = ()
    observations: tuple[Observation, ...] = ()
    verifications: tuple[OutcomeVerification, ...] = ()
    notifications: tuple[dict[str, JsonValue], ...] = ()


@dataclass(frozen=True)
class ActionOperatorSnapshot:
    """One validated state read and immutable indexes for an operator request."""

    intents: Mapping[str, ActionIntent]
    approvals: Mapping[str, ApprovalRecord]
    approvals_by_intent: Mapping[str, ApprovalRecord]
    receipts: Mapping[str, ExecutionReceipt]
    receipts_by_intent: Mapping[str, tuple[ExecutionReceipt, ...]]
    compensations_by_target: Mapping[str, tuple[ExecutionReceipt, ...]]
    observations: Mapping[str, Observation]
    observations_by_intent: Mapping[str, tuple[Observation, ...]]
    verifications: Mapping[str, OutcomeVerification]
    verifications_by_intent: Mapping[str, tuple[OutcomeVerification, ...]]
    validations: Mapping[str, ActionValidationRecord]

    @classmethod
    def build(cls, execution: "ActionExecutionLayer") -> "ActionOperatorSnapshot":
        state = execution._state_checked()
        return _snapshot_from_state(state)


def _snapshot_from_state(state: ActionState) -> ActionOperatorSnapshot:

    def index(
        values: tuple[Any, ...],
        identifier: Callable[[Any], str],
        label: str,
    ) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for value in values:
            key = identifier(value)
            if key in result:
                raise ActionPolicyError(f"Duplicate or corrupt {label} records")
            result[key] = value
        return MappingProxyType(result)

    approvals_by_intent_data: dict[str, ApprovalRecord] = {}
    for approval in state.approvals:
        if approval.intent_id in approvals_by_intent_data:
            raise ActionPolicyError("Approval binding is not unique")
        approvals_by_intent_data[approval.intent_id] = approval
    receipts_by_intent_lists: dict[str, list[ExecutionReceipt]] = {}
    for receipt in state.receipts:
        receipts_by_intent_lists.setdefault(receipt.intent_id, []).append(receipt)
    receipts_by_intent_data = {
        intent_id: tuple(receipts)
        for intent_id, receipts in receipts_by_intent_lists.items()
    }
    compensations_by_target_lists: dict[str, list[ExecutionReceipt]] = {}
    for receipt in state.receipts:
        if receipt.compensation_of is not None:
            compensations_by_target_lists.setdefault(
                receipt.compensation_of, []
            ).append(receipt)
    compensations_by_target_data = {
        receipt_id: tuple(receipts)
        for receipt_id, receipts in compensations_by_target_lists.items()
    }
    observations_by_intent_lists: dict[str, list[Observation]] = {}
    for observation in state.observations:
        observations_by_intent_lists.setdefault(observation.intent_id, []).append(
            observation
        )
    verifications_by_intent_lists: dict[str, list[OutcomeVerification]] = {}
    for verification in state.verifications:
        verifications_by_intent_lists.setdefault(verification.intent_id, []).append(
            verification
        )
    snapshot = ActionOperatorSnapshot(
        intents=index(state.intents, lambda item: item.intent_id, "intent"),
        approvals=index(state.approvals, lambda item: item.approval_id, "approval"),
        approvals_by_intent=MappingProxyType(approvals_by_intent_data),
        receipts=index(state.receipts, lambda item: item.receipt_id, "receipt"),
        receipts_by_intent=MappingProxyType(receipts_by_intent_data),
        compensations_by_target=MappingProxyType(compensations_by_target_data),
        observations=index(
            state.observations, lambda item: item.observation_id, "observation"
        ),
        observations_by_intent=MappingProxyType(
            {
                intent_id: tuple(observations)
                for intent_id, observations in observations_by_intent_lists.items()
            }
        ),
        verifications=index(
            state.verifications, lambda item: item.verification_id, "verification"
        ),
        verifications_by_intent=MappingProxyType(
            {
                intent_id: tuple(verifications)
                for intent_id, verifications in verifications_by_intent_lists.items()
            }
        ),
        validations=index(
            state.validation_records, lambda item: item.validation_id, "validation"
        ),
    )
    for intent in snapshot.intents.values():
        bound_approval = snapshot.approvals_by_intent.get(intent.intent_id)
        if (intent.approval_id is None) != (bound_approval is None) or (
            bound_approval is not None
            and bound_approval.approval_id != intent.approval_id
        ):
            raise ActionPolicyError("Approval binding is inconsistent")
    return snapshot


class ActionPolicyError(ValueError):
    """An action was rejected before execution."""


def receipt_matches_intent(receipt: ExecutionReceipt, intent: ActionIntent) -> bool:
    """Return whether a receipt is immutably bound to an action intent."""
    provenance = intent.provenance
    return (
        receipt.intent_id == intent.intent_id
        and receipt.idempotency_key == intent.idempotency_key
        and receipt.decision_id == provenance.decision_id
        and receipt.plan_id == provenance.plan_id
        and receipt.plan_revision == provenance.plan_revision
        and receipt.step_id == provenance.step_id
    )


def _semantic_binding(
    receipt: ExecutionReceipt,
    intent: ActionIntent,
    observation: Observation | None = None,
    verification: OutcomeVerification | None = None,
    *,
    successful: bool = False,
) -> bool:
    """Validate the complete, non-rebindable receipt evidence contract."""
    if not receipt_matches_intent(receipt, intent):
        return False
    if not successful:
        return True
    if (
        receipt.status != ReceiptStatus.SUCCEEDED
        or receipt.observation_id is None
        or receipt.verification_id is None
        or observation is None
        or verification is None
        or not observation.valid
        or not verification.success
    ):
        return False
    return bool(
        observation.intent_id == intent.intent_id
        and observation.receipt_id == receipt.receipt_id
        and verification.intent_id == intent.intent_id
        and verification.observation_id == observation.observation_id
    )


def _intent_receipts_semantically_valid(
    intent: ActionIntent,
    receipts: tuple[ExecutionReceipt, ...],
    observations: Mapping[str, Observation],
    verifications: Mapping[str, OutcomeVerification],
    compensations: tuple[ExecutionReceipt, ...] = (),
) -> bool:
    """Validate only one intent's receipt/evidence graph; never raise globally."""
    for receipt in receipts:
        if not receipt_matches_intent(receipt, intent):
            return False
        if receipt.status == ReceiptStatus.SUCCEEDED and not _semantic_binding(
            receipt,
            intent,
            observations.get(receipt.observation_id or ""),
            verifications.get(receipt.verification_id or ""),
            successful=True,
        ):
            return False
    associated_ids = {item.receipt_id for item in receipts}
    receipts_by_id = {item.receipt_id: item for item in receipts}
    compensation_targets: set[str] = set()
    for receipt in compensations:
        target_id = receipt.compensation_of
        if target_id is None or target_id not in associated_ids:
            continue
        target = receipts_by_id.get(target_id)
        if (
            receipt.status != ReceiptStatus.COMPENSATED
            or not receipt_matches_intent(receipt, intent)
            or target_id == receipt.receipt_id
            or target is None
            or target_id in compensation_targets
            or not _semantic_binding(
                target,
                intent,
                observations.get(target.observation_id or ""),
                verifications.get(target.verification_id or ""),
                successful=True,
            )
        ):
            return False
        compensation_targets.add(target_id)
    return True


ArgumentModel = type[_StrictModel]


class _ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    risk: RiskClass
    arguments: ArgumentModel
    approval_required: bool
    reversible: bool
    effect: str


_TOOLS = {
    "restricted_metadata_read": _ToolSpec(
        name="restricted_metadata_read",
        risk=RiskClass.READ_ONLY,
        arguments=MetadataReadArguments,
        approval_required=False,
        reversible=False,
        effect="Read one non-secret allowlisted runtime metadata value",
    ),
    "document_search": _ToolSpec(
        name="document_search",
        risk=RiskClass.READ_ONLY,
        arguments=DocumentSearchArguments,
        approval_required=False,
        reversible=False,
        effect="Search bounded UTF-8 text documents under the configured local root",
    ),
    "calendar_read": _ToolSpec(
        name="calendar_read",
        risk=RiskClass.READ_ONLY,
        arguments=CalendarReadArguments,
        approval_required=False,
        reversible=False,
        effect="Read bounded events from the configured local calendar file",
    ),
    "local_notification_enqueue": _ToolSpec(
        name="local_notification_enqueue",
        risk=RiskClass.REVERSIBLE_WRITE,
        arguments=NotificationArguments,
        approval_required=True,
        reversible=True,
        effect="Enqueue one local-only notification in authoritative state",
    ),
}


def _intent_records_consistent_indexed(
    snapshot: ActionOperatorSnapshot, intent: ActionIntent
) -> bool:
    """Pure semantic validation for one indexed persisted intent graph."""
    try:
        if intent.approval_id is not None:
            approval = snapshot.approvals.get(intent.approval_id)
            if approval is None or approval.intent_id != intent.intent_id:
                return False
        elif intent.intent_id in snapshot.approvals_by_intent:
            return False
        spec = _TOOLS.get(intent.tool_name)
        if spec is None or intent.risk_class != spec.risk:
            return False
        if (
            intent.policy.tool_name != intent.tool_name
            or intent.policy.risk_class != intent.risk_class
            or intent.policy.allowed is not True
            or intent.policy.approval_required != spec.approval_required
            or intent.preview.tool_name != intent.tool_name
            or intent.preview.risk_class != intent.risk_class
            or intent.preview.effect != spec.effect
            or intent.preview.compensation_available != spec.reversible
            or intent.preview.bounded_by != intent.budget
            or (
                intent.budget.max_risk_class == "read_only"
                and intent.risk_class != RiskClass.READ_ONLY
            )
        ):
            return False
        validated = spec.arguments.model_validate(intent.arguments).model_dump(
            mode="json"
        )
        legacy_notification = (
            intent.tool_name == "local_notification_enqueue"
            and intent.arguments.get("public_preview") is None
        )
        if legacy_notification:
            validated.pop("public_preview", None)
        if validated != intent.arguments or validated != intent.preview.arguments:
            return False
        associated_receipts = snapshot.receipts_by_intent.get(intent.intent_id, ())
        if not _intent_receipts_semantically_valid(
            intent,
            associated_receipts,
            snapshot.observations,
            snapshot.verifications,
            tuple(
                compensation
                for receipt in associated_receipts
                for compensation in snapshot.compensations_by_target.get(
                    receipt.receipt_id, ()
                )
            ),
        ):
            return False
        if intent.validation_record_id is None:
            return False
        record = snapshot.validations.get(intent.validation_record_id)
        return bool(
            record is not None
            and record.intent_id == intent.intent_id
            and record.decision_id == intent.provenance.decision_id
            and record.tool_name == intent.tool_name
            and record.risk_class == intent.risk_class
            and record.arguments_valid
            and not record.validation_error_codes
            and record.idempotency_key == intent.idempotency_key
            and record.canonical_arguments_digest == _digest(validated)
            and (
                legacy_notification
                or record.validation_schema_revision
                == _validation_schema_revision(intent.tool_name, spec)
            )
            and intent.policy.argument_digest == _digest(validated)
        )
    except (ActionPolicyError, ValidationError, ValueError, TypeError):
        return False


def public_tool_name(value: str) -> str | None:
    """Return only stable, allowlisted tool identifiers."""
    return value if value in _TOOLS else None


_VALIDATION_SCHEMA_REVISIONS = {
    "restricted_metadata_read": 1,
    "document_search": 1,
    "calendar_read": 1,
    "local_notification_enqueue": 1,
}


def _effect_code(tool_name: str) -> str:
    return {
        "restricted_metadata_read": "metadata.read",
        "document_search": "documents.search",
        "calendar_read": "calendar.read",
        "local_notification_enqueue": "notification.enqueue",
    }.get(tool_name, "action.unknown")


def _public_summary(
    tool_name: str, arguments: dict[str, JsonValue]
) -> PublicArgumentSummary:
    if tool_name == "restricted_metadata_read":
        return MetadataPublicSummary(
            namespace=str(arguments["namespace"]), key=str(arguments["key"])
        )
    if tool_name == "document_search":
        query = str(arguments["query"])
        return DocumentPublicSummary(
            max_results=int(cast(int, arguments["max_results"])),
            query_length=len(query),
            path_scoped=arguments.get("relative_path") is not None,
        )
    if tool_name == "calendar_read":
        return CalendarPublicSummary(
            starts_at=datetime.fromisoformat(str(arguments["starts_at"])),
            ends_at=datetime.fromisoformat(str(arguments["ends_at"])),
            max_results=int(cast(int, arguments["max_results"])),
        )
    if tool_name == "local_notification_enqueue":
        preview = arguments.get("public_preview")
        if isinstance(preview, dict):
            return NotificationPublicSummary.model_validate(preview)
        # Legacy intents are still executable, but never expose their raw text.
        return NotificationPublicSummary(
            title="Local notification", body="A local notification is queued."
        )
    raise ActionPolicyError("Tool is not allowlisted")


def _public_preview_revision(tool_name: str) -> str:
    return _digest({"tool_name": tool_name, "preview_revision": 1})


def _operator_preview_digest(
    intent: ActionIntent,
    descriptor: ActionToolDescriptor,
    summary: PublicArgumentSummary,
) -> str:
    """Create an opaque exact binding without publishing an argument oracle."""
    return _digest(
        cast(
            JsonValue,
            {
                "intent_id": intent.intent_id,
                "revision": intent.revision,
                "tool": descriptor.model_dump(mode="json"),
                "arguments": summary.model_dump(mode="json"),
                "effect": descriptor.effect,
                "effect_code": descriptor.effect_code,
                "policy": {
                    "allowed": intent.policy.allowed,
                    "approval_required": intent.policy.approval_required,
                    "reasons": list(intent.policy.reasons),
                },
                "budget": intent.budget.model_dump(mode="json"),
                # The strictly bound validation UUID remains private and
                # provides deterministic entropy without mutating state.
                "private_binding": _digest(
                    {
                        "domain": "action_operator_preview_v1",
                        "validation_record_id": intent.validation_record_id,
                        "canonical_arguments_digest": intent.policy.argument_digest,
                    }
                ),
            },
        )
    )


class ActionExecutionLayer:
    """Fail-closed action state machine owned by the subject runtime."""

    def __init__(
        self,
        main_loop: Any,
        *,
        document_root: Path,
        calendar_path: Path,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.main_loop = main_loop
        self.document_root = document_root.resolve()
        self.calendar_path = calendar_path.resolve()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.monotonic = monotonic or time.monotonic
        self._state()

    def list_tool_descriptors(self) -> tuple[ActionToolDescriptor, ...]:
        """Return stable descriptors, never the internal tool registry."""
        return tuple(self.get_tool_descriptor(name) for name in _TOOLS)

    def get_tool_descriptor(self, tool_name: str) -> ActionToolDescriptor:
        spec = _TOOLS.get(tool_name)
        if spec is None:
            raise ValueError("Unknown action tool")
        enabled = spec.risk not in {
            RiskClass.EXTERNAL_WRITE,
            RiskClass.DESTRUCTIVE,
            RiskClass.HIGH_IMPACT,
        }
        return ActionToolDescriptor(
            tool_name=spec.name,
            risk_class=spec.risk,
            effect=spec.effect,
            effect_code=_effect_code(spec.name),
            approval_required=spec.approval_required,
            reversible=spec.reversible,
            validation_revision=_validation_schema_revision(spec.name, spec),
            public_preview_revision=_public_preview_revision(spec.name),
            enabled=enabled,
            executable=enabled,
        )

    def operator_preview(self, intent_id: str) -> ActionOperatorPreview:
        snapshot = self.operator_snapshot()
        intent = snapshot.intents.get(intent_id)
        if intent is None:
            raise ValueError(f"Unknown action intent: {intent_id}")
        return self.operator_preview_from_snapshot(snapshot, intent)

    def operator_snapshot(self) -> ActionOperatorSnapshot:
        return ActionOperatorSnapshot.build(self)

    def operator_preview_from_snapshot(
        self, snapshot: ActionOperatorSnapshot, intent: ActionIntent
    ) -> ActionOperatorPreview:
        if snapshot.intents.get(intent.intent_id) != intent:
            raise ActionPolicyError("Action operator snapshot binding is invalid")
        if not self._intent_records_consistent_indexed(snapshot, intent):
            raise ActionPolicyError("Action operator binding is invalid")
        descriptor = self.get_tool_descriptor(intent.tool_name)
        summary = _public_summary(intent.tool_name, intent.arguments)
        digest = _operator_preview_digest(intent, descriptor, summary)
        return ActionOperatorPreview(
            intent_id=intent.intent_id,
            revision=intent.revision,
            tool=descriptor,
            arguments=summary,
            preview_digest=digest,
            effect=descriptor.effect,
            effect_code=descriptor.effect_code,
            available_commands=self.available_commands_from_snapshot(snapshot, intent),
            approval_id=intent.approval_id,
        )

    # Alias used by API adapters which call this a projection.
    operator_projection = operator_preview

    def available_commands(self, intent_id: str) -> tuple[OperatorCommand, ...]:
        snapshot = self.operator_snapshot()
        intent = snapshot.intents.get(intent_id)
        if intent is None:
            raise ValueError(f"Unknown action intent: {intent_id}")
        return self.available_commands_from_snapshot(snapshot, intent)

    def available_commands_from_snapshot(
        self, snapshot: ActionOperatorSnapshot, intent: ActionIntent
    ) -> tuple[OperatorCommand, ...]:
        descriptor = self.get_tool_descriptor(intent.tool_name)
        if (
            snapshot.intents.get(intent.intent_id) != intent
            or not self._snapshot_ownership_valid(snapshot, intent)
            or not descriptor.enabled
            or not self._intent_records_consistent_indexed(snapshot, intent)
        ):
            return ()
        commands: list[OperatorCommand] = []
        approval = snapshot.approvals_by_intent.get(intent.intent_id)
        has_public_preview = self._has_public_preview(intent)
        if (
            intent.status == IntentStatus.AWAITING_APPROVAL
            and approval is not None
            and approval.approval_id == intent.approval_id
            and approval.status == "pending"
        ):
            if has_public_preview:
                commands.append(OperatorCommand.APPROVE)
            commands.append(OperatorCommand.REJECT)
        if intent.status in {
            IntentStatus.AWAITING_APPROVAL,
            IntentStatus.APPROVED,
            IntentStatus.RETRY_PENDING,
        }:
            commands.append(OperatorCommand.CANCEL)
        if intent.status == IntentStatus.RETRY_PENDING and self._retry_executable(
            intent
        ):
            commands.append(OperatorCommand.RETRY_NOW)
        if (
            intent.status == IntentStatus.SUCCEEDED
            and descriptor.reversible
            and self._compensation_eligible_indexed(snapshot, intent)
        ):
            commands.append(OperatorCommand.COMPENSATE)
        return tuple(commands)

    def confirmation_phrase(self, intent_id: str) -> str | None:
        snapshot = self.operator_snapshot()
        intent = snapshot.intents.get(intent_id)
        if intent is None:
            raise ValueError(f"Unknown action intent: {intent_id}")
        if intent.risk_class not in {RiskClass.DESTRUCTIVE, RiskClass.HIGH_IMPACT}:
            return None
        preview = self.operator_preview_from_snapshot(snapshot, intent)
        return f"CONFIRM {intent.intent_id} {intent.revision} {preview.preview_digest}"

    def validate_command_request(
        self, request: OperatorCommandRequest
    ) -> OperatorCommandValidation:
        snapshot = self.operator_snapshot()
        intent = snapshot.intents.get(request.intent_id)
        if intent is None:
            raise ValueError(f"Unknown action intent: {request.intent_id}")
        preview = self.operator_preview_from_snapshot(snapshot, intent)
        reason: str | None = None
        if not self._intent_records_consistent_indexed(snapshot, intent):
            reason = "action semantic binding is invalid"
        elif request.expected_revision != intent.revision:
            reason = "stale intent revision"
        elif request.expected_preview_digest != preview.argument_digest:
            reason = "stale preview digest"
        elif request.command not in preview.available_commands:
            reason = "command is not currently available"
        elif (
            request.command in {OperatorCommand.APPROVE, OperatorCommand.REJECT}
            and request.expected_approval_id != intent.approval_id
        ):
            reason = "approval binding is stale"
        elif intent.risk_class in {RiskClass.DESTRUCTIVE, RiskClass.HIGH_IMPACT}:
            expected = (
                f"CONFIRM {intent.intent_id} {intent.revision} {preview.preview_digest}"
            )
            if request.confirmation != expected:
                reason = "confirmation phrase is invalid"
        return OperatorCommandValidation(
            valid=reason is None,
            reason=reason,
            command=request.command,
            intent_id=intent.intent_id,
        )

    def execute_command(
        self, request: OperatorCommandRequest, *, actor_id: str = "operator"
    ) -> ActionIntent:
        checked = self.validate_command_request(request)
        if not checked.valid:
            raise ActionPolicyError(checked.reason or "invalid operator command")
        if request.command == OperatorCommand.APPROVE:
            return self.resolve_approval(
                request.intent_id, approved=True, actor_id=actor_id
            )
        if request.command == OperatorCommand.REJECT:
            return self.resolve_approval(
                request.intent_id, approved=False, actor_id=actor_id
            )
        if request.command == OperatorCommand.CANCEL:
            return self.cancel(request.intent_id)
        if request.command == OperatorCommand.RETRY_NOW:
            return self.execute(request.intent_id)
        return self.compensate(request.intent_id)

    def list_intents(self) -> tuple[ActionIntent, ...]:
        return self._state().intents

    def list_validation_records(self) -> tuple[ActionValidationRecord, ...]:
        return self._state().validation_records

    def list_policy_rejections(self) -> tuple[ActionPolicyRejectionRecord, ...]:
        return self._state().policy_rejections

    def list_approvals(
        self, *, pending_only: bool = False
    ) -> tuple[ApprovalRecord, ...]:
        values = self._state().approvals
        return tuple(
            item for item in values if not pending_only or item.status == "pending"
        )

    def list_receipts(self) -> tuple[ExecutionReceipt, ...]:
        return self._state().receipts

    def list_observations(self) -> tuple[Observation, ...]:
        return self._state().observations

    def list_verifications(self) -> tuple[OutcomeVerification, ...]:
        return self._state().verifications

    def get_receipt(self, receipt_id: str) -> ExecutionReceipt:
        receipt = next(
            (item for item in self._state().receipts if item.receipt_id == receipt_id),
            None,
        )
        if receipt is None:
            raise ValueError(f"Unknown execution receipt: {receipt_id}")
        return receipt

    def get_observation(self, observation_id: str) -> Observation:
        observation = next(
            (
                item
                for item in self._state().observations
                if item.observation_id == observation_id
            ),
            None,
        )
        if observation is None:
            raise ValueError(f"Unknown action observation: {observation_id}")
        return observation

    def get_intent(self, intent_id: str) -> ActionIntent:
        intent = next(
            (item for item in self._state().intents if item.intent_id == intent_id),
            None,
        )
        if intent is None:
            raise ValueError(f"Unknown action intent: {intent_id}")
        return intent

    def get_validation_record(self, validation_id: str) -> ActionValidationRecord:
        record = next(
            (
                item
                for item in self._state().validation_records
                if item.validation_id == validation_id
            ),
            None,
        )
        if record is None:
            raise ValueError(f"Unknown action validation record: {validation_id}")
        return record

    def validate_decision_outcome(self, decision_id: str, success: bool) -> bool:
        state = self._state()
        decision = self.main_loop.decision_store.get(decision_id)
        intents = [
            item
            for item in state.intents
            if item.provenance.decision_id == decision_id
            and item.provenance.candidate_id == decision.selected_candidate_id
        ]
        if not intents:
            linked_validation = any(
                item.decision_id == decision_id for item in state.validation_records
            )
            linked_rejection = any(
                item.decision_id == decision_id
                and item.candidate_id == decision.selected_candidate_id
                for item in state.policy_rejections
            )
            if linked_validation or linked_rejection:
                raise ValueError("Linked action has no verified terminal outcome")
            return False
        intent = max(intents, key=lambda item: (item.revision, item.updated_at))
        terminal = {
            IntentStatus.SUCCEEDED,
            IntentStatus.FAILED,
            IntentStatus.CANCELLED,
            IntentStatus.REJECTED,
            IntentStatus.COMPENSATED,
        }
        if intent.status not in terminal:
            raise ValueError("Linked action is still awaiting approval or execution")
        if intent.receipt_id is None:
            raise ValueError("Action-backed outcome requires a linked receipt")
        receipt = next(
            (
                item
                for item in state.receipts
                if item.receipt_id == intent.receipt_id
                and item.intent_id == intent.intent_id
                and item.decision_id == decision_id
            ),
            None,
        )
        observation = next(
            (
                item
                for item in state.observations
                if receipt is not None
                and item.observation_id == receipt.observation_id
                and item.receipt_id == receipt.receipt_id
                and item.intent_id == intent.intent_id
            ),
            None,
        )
        verification = next(
            (
                item
                for item in state.verifications
                if receipt is not None
                and observation is not None
                and item.verification_id == receipt.verification_id
                and item.observation_id == observation.observation_id
                and item.intent_id == intent.intent_id
            ),
            None,
        )
        if receipt is None or observation is None or verification is None:
            raise ValueError(
                "Action-backed outcome requires matching receipt, observation, and verification"
            )
        if success != verification.success or success != (
            receipt.status == ReceiptStatus.SUCCEEDED
        ):
            raise ValueError("Decision outcome contradicts action verification")
        return True

    def link_explanation(self, decision_id: str, explanation_ref: str) -> None:
        if (
            not explanation_ref
            or len(explanation_ref) > 128
            or any(
                character in explanation_ref for character in ("/", "\\", "\n", "\r")
            )
        ):
            raise ValueError("Explanation reference must be a safe opaque identifier")
        state = self._state()
        intents = tuple(
            item.model_copy(
                update={
                    "revision": item.revision + 1,
                    "explanation_refs": tuple(
                        dict.fromkeys((*item.explanation_refs, explanation_ref))
                    ),
                }
            )
            if item.provenance.decision_id == decision_id
            and explanation_ref not in item.explanation_refs
            else item
            for item in state.intents
        )
        receipts = tuple(
            item.model_copy(
                update={
                    "explanation_refs": tuple(
                        dict.fromkeys((*item.explanation_refs, explanation_ref))
                    )
                }
            )
            if item.decision_id == decision_id
            and explanation_ref not in item.explanation_refs
            else item
            for item in state.receipts
        )
        if intents != state.intents or receipts != state.receipts:
            self._save(
                state.model_copy(update={"intents": intents, "receipts": receipts})
            )

    def create_from_decision(
        self,
        decision_id: str,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        budget: ActionBudget | None = None,
    ) -> ActionIntent | ActionValidationRecord | ActionPolicyRejectionRecord:
        event = current_agent_event()
        if event is None or event.processing_sequence is None:
            raise RuntimeError(
                "Action validation requires an authoritative AgentRuntime event"
            )
        if not idempotency_key or len(idempotency_key) > 128:
            raise ActionPolicyError("Idempotency key must contain 1 to 128 characters")
        state = self._state()
        prior_rejection = next(
            (
                item
                for item in state.policy_rejections
                if item.idempotency_key == idempotency_key
            ),
            None,
        )
        if prior_rejection is not None:
            if prior_rejection.decision_id != decision_id:
                raise ActionPolicyError(
                    "Idempotency key is already bound to another decision"
                )
            return prior_rejection
        duplicate = next(
            (item for item in state.intents if item.idempotency_key == idempotency_key),
            None,
        )
        if duplicate is not None:
            if duplicate.provenance.decision_id != decision_id:
                raise ActionPolicyError(
                    "Idempotency key is already bound to another decision"
                )
            return duplicate
        decision = self.main_loop.decision_store.get(decision_id)
        selected = next(
            item.candidate
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        malformed_parameters = (
            not set(selected.parameters) <= {"action", "value_effects"}
            or "action" not in selected.parameters
            or not isinstance(selected.parameters["action"], dict)
        )
        contract = selected.parameters["action"] if not malformed_parameters else {}
        tool_name = contract.get("tool_name")
        arguments = contract.get("arguments")
        request_digest = _digest(
            cast(
                JsonValue,
                {
                    "decision_id": decision_id,
                    "candidate_id": selected.candidate_id,
                    "action": selected.parameters.get("action"),
                    "dry_run": dry_run,
                    "budget": (budget or ActionBudget()).model_dump(mode="json"),
                },
            )
        )
        prior_validation = next(
            (
                item
                for item in state.validation_records
                if not item.arguments_valid and item.idempotency_key == idempotency_key
            ),
            None,
        )
        if prior_validation is not None:
            if prior_validation.request_digest != request_digest:
                raise ActionPolicyError(
                    "Idempotency key is already bound to a different action request"
                )
            return prior_validation
        self._validate_boundary_assessment(decision)
        if decision.status != DecisionStatus.AWAITING_OUTCOME:
            raise ActionPolicyError(
                "Action requires a selected decision awaiting outcome"
            )
        now = self.clock()
        intent_id = str(uuid4())
        if (
            malformed_parameters
            or set(contract) != {"tool_name", "arguments"}
            or not isinstance(tool_name, str)
            or not isinstance(arguments, dict)
        ):
            bounded_tool_name = (
                tool_name[:128] if isinstance(tool_name, str) else "invalid_action"
            )
            validation = ActionValidationRecord(
                validation_id=str(uuid4()),
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                decision_id=decision_id,
                tool_name=bounded_tool_name,
                arguments_valid=False,
                validation_schema_revision=_validation_schema_revision(
                    bounded_tool_name, _TOOLS.get(bounded_tool_name)
                ),
                validation_error_codes=(
                    ActionValidationErrorCode.ACTION_CONTRACT_INVALID,
                ),
                validated_event_id=event.event_id,
                validated_event_sequence=event.processing_sequence,
                canonical_arguments_digest=_digest(
                    cast(dict[str, JsonValue], arguments)
                    if isinstance(arguments, dict)
                    else {"invalid_contract": True}
                ),
                validated_at=now,
            )
            self._save(
                state.model_copy(
                    update={
                        "validation_records": (*state.validation_records, validation)
                    }
                )
            )
            return validation
        assert isinstance(tool_name, str) and isinstance(arguments, dict)
        spec, validated, error_codes = self._validate_arguments(tool_name, arguments)
        validation = ActionValidationRecord(
            validation_id=str(uuid4()),
            idempotency_key=idempotency_key,
            request_digest=request_digest,
            decision_id=decision_id,
            intent_id=intent_id if validated is not None else None,
            tool_name=tool_name,
            risk_class=None if spec is None else spec.risk,
            arguments_valid=validated is not None,
            validation_schema_revision=_validation_schema_revision(tool_name, spec),
            validation_error_codes=error_codes,
            validated_event_id=event.event_id,
            validated_event_sequence=event.processing_sequence,
            canonical_arguments_digest=_digest(
                validated
                if validated is not None
                else cast(dict[str, JsonValue], arguments)
            ),
            validated_at=now,
        )
        if validated is None or spec is None:
            self._save(
                state.model_copy(
                    update={
                        "validation_records": (*state.validation_records, validation)
                    }
                )
            )
            return validation
        bounded = budget or ActionBudget()
        if bounded.max_risk_class == "read_only" and spec.risk != RiskClass.READ_ONLY:
            rejection = ActionPolicyRejectionRecord(
                rejection_id=str(uuid4()),
                idempotency_key=idempotency_key,
                decision_id=decision_id,
                candidate_id=selected.candidate_id,
                validation_id=validation.validation_id,
                risk_class=spec.risk,
                policy_code="risk_budget_denied",
                reason_code="risk_class_exceeds_budget",
                event_id=event.event_id,
                event_sequence=event.processing_sequence,
                rejected_at=now,
            )
            self._save(
                state.model_copy(
                    update={
                        "validation_records": (*state.validation_records, validation),
                        "policy_rejections": (*state.policy_rejections, rejection),
                    }
                )
            )
            return rejection
        digest = _digest(validated)
        policy = PolicyEvaluation(
            evaluation_id=str(uuid4()),
            tool_name=tool_name,
            risk_class=spec.risk,
            allowed=True,
            approval_required=spec.approval_required,
            reasons=(
                "tool_allowlisted",
                "arguments_schema_valid",
                "risk_budget_valid",
                "human_approval_required"
                if spec.approval_required
                else "read_only_auto_approved",
            ),
            argument_digest=digest,
            evaluated_at=now,
        )
        approval_id = str(uuid4()) if spec.approval_required and not dry_run else None
        intent = ActionIntent(
            intent_id=intent_id,
            revision=1,
            idempotency_key=idempotency_key,
            tool_name=tool_name,
            arguments=validated,
            risk_class=spec.risk,
            status=IntentStatus.DRY_RUN
            if dry_run
            else IntentStatus.AWAITING_APPROVAL
            if spec.approval_required
            else IntentStatus.APPROVED,
            dry_run=dry_run,
            policy=policy,
            preview=ActionPreview(
                tool_name=tool_name,
                risk_class=spec.risk,
                arguments=validated,
                effect=spec.effect,
                bounded_by=bounded,
                compensation_available=spec.reversible,
            ),
            provenance=ActionProvenance(
                decision_id=decision_id,
                candidate_id=selected.candidate_id,
                triggering_event_id=decision.triggering_event_id,
                triggering_event_sequence=decision.triggering_event_sequence,
                plan_id=selected.plan_id,
                plan_revision=selected.plan_revision,
                step_id=selected.step_id,
                boundary_assessment_id=decision.boundary_assessment_id,
                boundary_assessment_revision=decision.boundary_assessment_revision,
                boundary_assessment_digest=decision.boundary_assessment_digest,
            ),
            budget=bounded,
            attempts=0,
            cost_units_used=0,
            created_at=now,
            updated_at=now,
            deadline_at=now + timedelta(seconds=bounded.timeout_seconds),
            approval_id=approval_id,
            validation_record_id=validation.validation_id,
        )
        approvals = state.approvals
        if approval_id is not None:
            approvals = (
                *approvals,
                ApprovalRecord(
                    approval_id=approval_id,
                    intent_id=intent_id,
                    status="pending",
                    requested_at=now,
                ),
            )
        self._save(
            state.model_copy(
                update={
                    "intents": (*state.intents, intent),
                    "validation_records": (*state.validation_records, validation),
                    "approvals": approvals,
                }
            )
        )
        if approval_id is not None:
            self._enqueue_outbox(
                OutboxMessageKind.APPROVAL_REQUEST,
                title="Action approval required",
                body=f"Review the bounded {tool_name} action before execution.",
                deduplication_key=f"action-approval:{intent_id}",
                intent=intent,
                urgency=OutboxUrgency.HIGH,
            )
        return intent

    def resolve_approval(
        self,
        intent_id: str,
        *,
        approved: bool,
        actor_id: str,
        reason: str | None = None,
    ) -> ActionIntent:
        state = self._state()
        intent = self.get_intent(intent_id)
        decision = self.main_loop.decision_store.get(intent.provenance.decision_id)
        if decision.status == DecisionStatus.RESOLVED:
            raise ActionPolicyError("Decision already has a terminal outcome")
        self._validate_boundary_assessment(decision)
        if (
            intent.status != IntentStatus.AWAITING_APPROVAL
            or intent.approval_id is None
        ):
            raise ValueError("Action intent is not awaiting approval")
        now = self.clock()
        approvals = tuple(
            item.model_copy(
                update={
                    "status": "approved" if approved else "rejected",
                    "resolved_at": now,
                    "actor_id": actor_id,
                    "reason": reason,
                }
            )
            if item.approval_id == intent.approval_id
            else item
            for item in state.approvals
        )
        updated = intent.model_copy(
            update={
                "revision": intent.revision + 1,
                "status": IntentStatus.APPROVED if approved else IntentStatus.REJECTED,
                "updated_at": now,
                "failure_code": None if approved else "operator_rejected",
            }
        )
        if approved:
            self._replace(state, updated, approvals=approvals)
        else:
            receipt_id = str(uuid4())
            observation, verification = self._failure_evidence(
                intent,
                receipt_id,
                ReceiptStatus.CANCELLED,
                "operator_rejected",
                now,
            )
            receipt = self._receipt(
                intent,
                receipt_id,
                intent.attempts,
                ReceiptStatus.CANCELLED,
                now,
                now,
                0.0,
                observation_id=observation.observation_id,
                verification_id=verification.verification_id,
                error_code="operator_rejected",
            )
            updated = updated.model_copy(update={"receipt_id": receipt_id})
            self._replace(
                state,
                updated,
                approvals=approvals,
                receipts=(*state.receipts, receipt),
                observations=(*state.observations, observation),
                verifications=(*state.verifications, verification),
            )
            self._resolve_decision(updated, False, "action_rejected")
        outbox = getattr(self.main_loop, "outbox", None)
        if outbox is not None:
            event = current_agent_event()
            outbox.respond_to_action_approval(
                intent_id,
                approved=approved,
                actor_id=actor_id,
                text=reason,
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
        return updated

    def _validate_boundary_assessment(self, decision: Any) -> None:
        assessment_id = decision.boundary_assessment_id
        if assessment_id is None:
            raise ActionPolicyError("Decision has no boundary assessment")
        store = self.main_loop.identity_boundary_store
        assessment = store.get_assessment(assessment_id)
        if (
            assessment.action_ref != f"decision:{decision.decision_id}"
            or assessment.context_id != decision.context_id
            or assessment.adapter_id != decision.adapter_id
            or assessment.adapter_hash != decision.adapter_hash
            or assessment.activation_sequence != decision.activation_sequence
            or decision.boundary_assessment_revision != assessment.revision
            or decision.boundary_assessment_digest
            != store.assessment_digest(assessment_id)
            or decision.boundary_recommendation != assessment.recommendation.value
        ):
            raise ActionPolicyError("Boundary assessment binding is invalid")
        if (
            not store.assessments
            or store.assessments[-1].assessment_id != assessment_id
        ):
            raise ActionPolicyError(
                "Boundary assessment is stale; reassessment required"
            )
        if assessment.recommendation.value in {"refuse", "defer"}:
            raise ActionPolicyError(
                "Boundary disposition blocks action pending reviewed reassessment"
            )
        validator = getattr(
            self.main_loop, "validate_identity_boundary_assessment", None
        )
        if validator is not None:
            try:
                validator(
                    assessment,
                    action_ref=f"decision:{decision.decision_id}",
                    context_id=decision.context_id,
                )
            except ValueError as exc:
                raise ActionPolicyError(str(exc)) from exc

    def execute(self, intent_id: str) -> ActionIntent:
        state = self._state()
        intent = self.get_intent(intent_id)
        if intent.status == IntentStatus.SUCCEEDED and self._succeeded_receipt_valid(
            state, intent
        ):
            return intent
        arguments = self._validated_arguments_for_execution(intent)
        if intent.dry_run:
            raise ActionPolicyError("Dry-run intents cannot execute")
        if intent.status not in {IntentStatus.APPROVED, IntentStatus.RETRY_PENDING}:
            raise ActionPolicyError(
                f"Action intent is not executable: {intent.status.value}"
            )
        if intent.cost_units_used >= intent.budget.max_cost_units:
            return self._fail(intent, "cost_budget_exhausted", ReceiptStatus.FAILED)
        if intent.attempts >= intent.budget.max_attempts:
            return self._fail(intent, "retry_budget_exhausted", ReceiptStatus.FAILED)
        spec = _TOOLS.get(intent.tool_name)
        if spec is None:
            raise ActionPolicyError("Action validation schema is no longer available")
        if spec.approval_required:
            approval = next(
                (
                    item
                    for item in state.approvals
                    if item.approval_id == intent.approval_id
                ),
                None,
            )
            if approval is None or approval.status != "approved":
                raise ActionPolicyError("Approved operator record is required")
        decision = self.main_loop.decision_store.get(intent.provenance.decision_id)
        if decision.status == DecisionStatus.RESOLVED:
            raise ActionPolicyError("Decision already has a terminal outcome")
        self._validate_boundary_assessment(decision)
        if (
            intent.provenance.boundary_assessment_id != decision.boundary_assessment_id
            or intent.provenance.boundary_assessment_revision
            != decision.boundary_assessment_revision
            or intent.provenance.boundary_assessment_digest
            != decision.boundary_assessment_digest
        ):
            raise ActionPolicyError("Action intent boundary binding is stale")
        if (
            intent.deadline_at < self.clock()
            and intent.status != IntentStatus.RETRY_PENDING
        ):
            return self._fail(
                intent, "execution_deadline_expired", ReceiptStatus.TIMED_OUT
            )

        started_at = self.clock()
        started_clock = self.monotonic()
        receipt_id = str(uuid4())
        attempt = intent.attempts + 1
        if (
            intent.provenance.plan_id is not None
            and intent.provenance.step_id is not None
        ):
            self.main_loop.start_action_plan_step(
                intent.provenance.plan_id, intent.provenance.step_id
            )
        try:
            result = self._invoke(intent, arguments)
            duration = self.monotonic() - started_clock
            if duration > intent.budget.timeout_seconds:
                raise TimeoutError("tool execution exceeded timeout")
            serialized = json.dumps(result, sort_keys=True, ensure_ascii=True).encode()
            if len(serialized) > intent.budget.max_result_bytes:
                raise ValueError("tool result exceeded byte budget")
            valid, errors = self._validate_result(intent.tool_name, result)
        except Exception as exc:
            self._rollback_partial_effect(intent)
            retryable = (
                isinstance(exc, (OSError, TimeoutError))
                and attempt < intent.budget.max_attempts
            )
            return self._execution_failure(
                intent,
                receipt_id,
                attempt,
                started_at,
                started_clock,
                "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__,
                retryable,
            )

        finished_at = self.clock()
        observation_id = str(uuid4())
        verification_id = str(uuid4())
        observation = Observation(
            observation_id=observation_id,
            intent_id=intent.intent_id,
            receipt_id=receipt_id,
            observed_at=finished_at,
            data=result,
            result_digest=_digest(result),
            valid=valid,
            validation_errors=errors,
        )
        verification = OutcomeVerification(
            verification_id=verification_id,
            intent_id=intent.intent_id,
            observation_id=observation_id,
            success=valid,
            reason="observation_schema_valid"
            if valid
            else "observation_schema_invalid",
            verified_at=finished_at,
        )
        receipt = self._receipt(
            intent,
            receipt_id,
            attempt,
            ReceiptStatus.SUCCEEDED if valid else ReceiptStatus.FAILED,
            started_at,
            finished_at,
            (self.monotonic() - started_clock) * 1000,
            observation_id=observation_id,
            verification_id=verification_id,
            error_code=None if valid else "result_validation_failed",
        )
        updated = intent.model_copy(
            update={
                "revision": intent.revision + 1,
                "status": IntentStatus.SUCCEEDED if valid else IntentStatus.FAILED,
                "attempts": attempt,
                "cost_units_used": intent.cost_units_used + 1,
                "updated_at": finished_at,
                "receipt_id": receipt_id,
                "retry_at": None,
                "failure_code": None if valid else "result_validation_failed",
            }
        )
        state = self._state()
        self._replace(
            state,
            updated,
            receipts=(*state.receipts, receipt),
            observations=(*state.observations, observation),
            verifications=(*state.verifications, verification),
        )
        if (
            valid
            and intent.provenance.plan_id is not None
            and intent.provenance.step_id is not None
        ):
            self.main_loop.record_action_plan_observation(
                intent.provenance.plan_id,
                intent.provenance.step_id,
                observation_id,
                intent.tool_name,
            )
        self._resolve_decision(updated, valid, verification.reason)
        if intent.tool_name != "local_notification_enqueue":
            self._enqueue_outbox(
                OutboxMessageKind.ACTION_RESULT,
                title="Action completed" if valid else "Action verification failed",
                body=f"The {intent.tool_name} action {'completed successfully' if valid else 'failed result verification'}.",
                deduplication_key=f"action-result:{intent.intent_id}:{updated.revision}",
                intent=updated,
                urgency=OutboxUrgency.NORMAL if valid else OutboxUrgency.HIGH,
            )
        return updated

    def cancel(self, intent_id: str) -> ActionIntent:
        intent = self.get_intent(intent_id)
        if intent.status not in {
            IntentStatus.AWAITING_APPROVAL,
            IntentStatus.APPROVED,
            IntentStatus.RETRY_PENDING,
        }:
            raise ValueError("Action intent cannot be cancelled")
        now = self.clock()
        updated = intent.model_copy(
            update={
                "revision": intent.revision + 1,
                "status": IntentStatus.CANCELLED,
                "updated_at": now,
                "retry_at": None,
                "failure_code": "cancelled",
            }
        )
        state = self._state()
        receipt_id = str(uuid4())
        observation, verification = self._failure_evidence(
            intent, receipt_id, ReceiptStatus.CANCELLED, "cancelled", now
        )
        receipt = self._receipt(
            intent,
            receipt_id,
            intent.attempts,
            ReceiptStatus.CANCELLED,
            now,
            now,
            0.0,
            observation_id=observation.observation_id,
            verification_id=verification.verification_id,
            error_code="cancelled",
        )
        self._replace(
            state,
            updated,
            receipts=(*state.receipts, receipt),
            observations=(*state.observations, observation),
            verifications=(*state.verifications, verification),
        )
        self._resolve_decision(updated, False, "action_cancelled")
        return updated

    def timeout(self, intent_id: str) -> ActionIntent:
        intent = self.get_intent(intent_id)
        if intent.status not in {IntentStatus.APPROVED, IntentStatus.RETRY_PENDING}:
            return intent
        if self.clock() < intent.deadline_at and (
            intent.retry_at is None or self.clock() < intent.retry_at
        ):
            return intent
        if intent.status == IntentStatus.RETRY_PENDING and intent.retry_at is not None:
            return self.execute(intent_id)
        return self._fail(intent, "execution_deadline_expired", ReceiptStatus.TIMED_OUT)

    def compensate(self, intent_id: str) -> ActionIntent:
        intent = self.get_intent(intent_id)
        state = self._state_checked()
        descriptor = self.get_tool_descriptor(intent.tool_name)
        if (
            intent.status != IntentStatus.SUCCEEDED
            or not descriptor.reversible
            or not self._compensation_eligible(state, intent)
        ):
            raise ActionPolicyError("Action has no available compensation")
        notifications = tuple(
            {**item, "status": "cancelled"}
            if item.get("idempotency_key") == intent.idempotency_key
            else item
            for item in state.notifications
        )
        now = self.clock()
        receipt_id = str(uuid4())
        receipt = self._receipt(
            intent,
            receipt_id,
            intent.attempts,
            ReceiptStatus.COMPENSATED,
            now,
            now,
            0.0,
            compensation_of=intent.receipt_id,
        )
        updated = intent.model_copy(
            update={
                "revision": intent.revision + 1,
                "status": IntentStatus.COMPENSATED,
                "updated_at": now,
                "receipt_id": receipt_id,
            }
        )
        self._replace(
            state,
            updated,
            receipts=(*state.receipts, receipt),
            notifications=notifications,
        )
        outbox = getattr(self.main_loop, "outbox", None)
        if outbox is not None:
            outbox.cancel(f"local-notification:{intent.idempotency_key}")
        record_compensation = getattr(
            self.main_loop, "record_decision_compensation", None
        )
        if callable(record_compensation):
            record_compensation(
                intent.provenance.decision_id,
                receipt_id=receipt_id,
            )
        return updated

    def _invoke(
        self, intent: ActionIntent, arguments: dict[str, JsonValue]
    ) -> JsonValue:
        if intent.tool_name == "restricted_metadata_read":
            values = {
                "project": {
                    "name": self.main_loop.settings.project.name,
                    "environment": self.main_loop.settings.project.environment,
                },
                "runtime": {
                    "node_id": self.main_loop.settings.deployment.node.id,
                    "model_provider": self.main_loop.settings.model.provider,
                },
            }
            namespace = str(arguments["namespace"])
            key = str(arguments["key"])
            if key not in values[namespace]:
                raise ValueError("metadata key is not valid for namespace")
            return {"namespace": namespace, "key": key, "value": values[namespace][key]}
        if intent.tool_name == "document_search":
            return self._document_search(arguments)
        if intent.tool_name == "calendar_read":
            return self._calendar_read(arguments)
        if intent.tool_name == "local_notification_enqueue":
            state = self._state()
            existing = next(
                (
                    item
                    for item in state.notifications
                    if item.get("idempotency_key") == intent.idempotency_key
                ),
                None,
            )
            if existing is not None:
                return {
                    "notification_id": existing["notification_id"],
                    "status": existing["status"],
                }
            notification: dict[str, JsonValue] = {
                "notification_id": str(uuid4()),
                "idempotency_key": intent.idempotency_key,
                "channel": arguments["channel"],
                "title": arguments["title"],
                "body": arguments["body"],
                "status": "queued",
                "created_at": self.clock().isoformat(),
            }
            self._save(
                state.model_copy(
                    update={"notifications": (*state.notifications, notification)}
                )
            )
            self._enqueue_outbox(
                OutboxMessageKind.ACTION_RESULT,
                title=str(arguments["title"]),
                body=str(arguments["body"]),
                deduplication_key=f"local-notification:{intent.idempotency_key}",
                intent=intent,
            )
            return {
                "notification_id": notification["notification_id"],
                "status": "queued",
            }
        raise ActionPolicyError("Tool is not allowlisted")

    def _state_checked(self) -> ActionState:
        state = self._state()
        ids = (
            ([item.intent_id for item in state.intents], "intent"),
            ([item.approval_id for item in state.approvals], "approval"),
            ([item.receipt_id for item in state.receipts], "receipt"),
            ([item.observation_id for item in state.observations], "observation"),
            ([item.verification_id for item in state.verifications], "verification"),
            ([item.validation_id for item in state.validation_records], "validation"),
        )
        for values, label in ids:
            if any(value is None for value in values) or len(values) != len(
                set(values)
            ):
                raise ActionPolicyError(f"Duplicate or corrupt {label} records")
        intents_by_id = {item.intent_id: item for item in state.intents}
        intent_ids = set(intents_by_id)
        if any(item.intent_id not in intent_ids for item in state.approvals):
            raise ActionPolicyError("Approval record has an invalid intent binding")
        if any(item.intent_id not in intent_ids for item in state.receipts):
            raise ActionPolicyError("Receipt record has an invalid intent binding")
        if any(item.intent_id not in intent_ids for item in state.observations):
            raise ActionPolicyError("Observation record has an invalid intent binding")
        if any(item.intent_id not in intent_ids for item in state.verifications):
            raise ActionPolicyError("Verification record has an invalid intent binding")
        receipt_ids = {item.receipt_id for item in state.receipts}
        if any(item.receipt_id not in receipt_ids for item in state.observations):
            raise ActionPolicyError("Observation record has an invalid receipt binding")
        approvals_by_intent: dict[str, ApprovalRecord] = {}
        for approval in state.approvals:
            if approval.intent_id in approvals_by_intent:
                raise ActionPolicyError("Approval binding is not unique")
            approvals_by_intent[approval.intent_id] = approval
        for intent in state.intents:
            bound_approval = approvals_by_intent.get(intent.intent_id)
            if (intent.approval_id is None) != (bound_approval is None) or (
                bound_approval is not None
                and bound_approval.approval_id != intent.approval_id
            ):
                raise ActionPolicyError("Approval binding is inconsistent")
        return state

    @staticmethod
    def _unique_approval(
        state: ActionState, intent: ActionIntent
    ) -> ApprovalRecord | None:
        values = [
            item for item in state.approvals if item.intent_id == intent.intent_id
        ]
        if intent.approval_id is None:
            return (
                None if not values else (_raise_corrupt("Unexpected approval record"))
            )
        matches = [item for item in values if item.approval_id == intent.approval_id]
        if len(values) != 1 or len(matches) != 1:
            return _raise_corrupt("Approval binding is not unique")
        return matches[0]

    @staticmethod
    def _snapshot_ownership_valid(
        snapshot: ActionOperatorSnapshot, intent: ActionIntent
    ) -> bool:
        """Check ownership for one projection without rejecting other intents."""
        for observation in snapshot.observations_by_intent.get(intent.intent_id, ()):
            receipt = snapshot.receipts.get(observation.receipt_id)
            if receipt is None or receipt.intent_id != intent.intent_id:
                return False
        for verification in snapshot.verifications_by_intent.get(intent.intent_id, ()):
            bound_observation = snapshot.observations.get(
                verification.observation_id or ""
            )
            if (
                bound_observation is None
                or bound_observation.intent_id != intent.intent_id
            ):
                return False
        return True

    def _intent_records_consistent_indexed(
        self, snapshot: ActionOperatorSnapshot, intent: ActionIntent
    ) -> bool:
        return _intent_records_consistent_indexed(snapshot, intent)

    @staticmethod
    def _has_public_preview(intent: ActionIntent) -> bool:
        if intent.tool_name != "local_notification_enqueue":
            return True
        preview = intent.arguments.get("public_preview")
        if not isinstance(preview, dict):
            return False
        try:
            NotificationPublicSummary.model_validate(preview)
        except ValidationError:
            return False
        return True

    def _retry_executable(self, intent: ActionIntent) -> bool:
        now = self.clock()
        return bool(
            now <= intent.deadline_at
            and intent.attempts < intent.budget.max_attempts
            and intent.cost_units_used < intent.budget.max_cost_units
        )

    def _compensation_eligible(self, state: ActionState, intent: ActionIntent) -> bool:
        if intent.receipt_id is None:
            return False
        associated = tuple(
            item for item in state.receipts if item.intent_id == intent.intent_id
        )
        targets = tuple(
            item for item in associated if item.receipt_id == intent.receipt_id
        )
        if len(targets) != 1 or not _intent_receipts_semantically_valid(
            intent,
            associated,
            {item.observation_id: item for item in state.observations},
            {item.verification_id: item for item in state.verifications},
            tuple(
                item
                for item in state.receipts
                if item.compensation_of in {target.receipt_id for target in associated}
            ),
        ):
            return False
        successful = targets[0]
        if not _semantic_binding(
            successful,
            intent,
            next(
                (
                    item
                    for item in state.observations
                    if item.observation_id == successful.observation_id
                ),
                None,
            ),
            next(
                (
                    item
                    for item in state.verifications
                    if item.verification_id == successful.verification_id
                ),
                None,
            ),
            successful=True,
        ):
            return False
        return not any(
            item.compensation_of == intent.receipt_id
            and item.intent_id == intent.intent_id
            and item.status == ReceiptStatus.COMPENSATED
            for item in state.receipts
        )

    def _compensation_eligible_indexed(
        self, snapshot: ActionOperatorSnapshot, intent: ActionIntent
    ) -> bool:
        if intent.receipt_id is None:
            return False
        receipt = snapshot.receipts.get(intent.receipt_id)
        associated = snapshot.receipts_by_intent.get(intent.intent_id, ())
        if (
            receipt is None
            or sum(item.receipt_id == intent.receipt_id for item in associated) != 1
            or not _intent_receipts_semantically_valid(
                intent,
                associated,
                snapshot.observations,
                snapshot.verifications,
                tuple(
                    compensation
                    for target in associated
                    for compensation in snapshot.compensations_by_target.get(
                        target.receipt_id, ()
                    )
                ),
            )
            or not _semantic_binding(
                receipt,
                intent,
                snapshot.observations.get(receipt.observation_id or ""),
                snapshot.verifications.get(receipt.verification_id or ""),
                successful=True,
            )
        ):
            return False
        return not snapshot.compensations_by_target.get(intent.receipt_id, ())

    @staticmethod
    def _succeeded_receipt_valid_indexed(
        snapshot: ActionOperatorSnapshot, intent: ActionIntent
    ) -> bool:
        if intent.receipt_id is None:
            return False
        receipt = snapshot.receipts.get(intent.receipt_id)
        return bool(
            receipt is not None
            and _semantic_binding(
                receipt,
                intent,
                snapshot.observations.get(receipt.observation_id or ""),
                snapshot.verifications.get(receipt.verification_id or ""),
                successful=True,
            )
        )

    def _document_search(self, arguments: dict[str, JsonValue]) -> JsonValue:
        root = self.document_root
        relative = arguments.get("relative_path")
        target = root if relative is None else (root / str(relative)).resolve()
        if target != root and root not in target.parents:
            raise ValueError("document path escaped configured root")
        files = (
            [target]
            if target.is_file()
            else sorted(target.rglob("*"))
            if target.exists()
            else []
        )
        query = str(arguments["query"]).casefold()
        limit = cast(int, arguments["max_results"])
        matches: list[dict[str, JsonValue]] = []
        for path in files[:500]:
            resolved = path.resolve()
            if (
                path.is_symlink()
                or (resolved != root and root not in resolved.parents)
                or not path.is_file()
                or path.suffix.lower() not in {".txt", ".md", ".json"}
            ):
                continue
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), 1):
                if query in line.casefold():
                    matches.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "line": line_number,
                            "excerpt": line[:500],
                        }
                    )
                    if len(matches) >= limit:
                        return cast(JsonValue, {"matches": matches, "truncated": True})
        return cast(JsonValue, {"matches": matches, "truncated": False})

    def _calendar_read(self, arguments: dict[str, JsonValue]) -> JsonValue:
        if not self.calendar_path.exists():
            return {"events": []}
        raw = json.loads(self.calendar_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or set(raw) != {"schema_version", "events"}
            or raw["schema_version"] != 1
        ):
            raise ValueError("calendar file has invalid schema")
        if not isinstance(raw["events"], list):
            raise ValueError("calendar events must be a list")
        starts = datetime.fromisoformat(str(arguments["starts_at"]))
        ends = datetime.fromisoformat(str(arguments["ends_at"]))
        selected: list[dict[str, JsonValue]] = []
        for item in raw["events"]:
            if not isinstance(item, dict) or set(item) != {
                "event_id",
                "title",
                "starts_at",
                "ends_at",
            }:
                raise ValueError("calendar event has invalid schema")
            event_start = datetime.fromisoformat(str(item["starts_at"]))
            event_end = datetime.fromisoformat(str(item["ends_at"]))
            if event_start.tzinfo is None or event_end.tzinfo is None:
                raise ValueError("calendar event timestamps require timezones")
            if event_end > starts and event_start < ends:
                selected.append(item)
        selected.sort(key=lambda item: (str(item["starts_at"]), str(item["event_id"])))
        limit = cast(int, arguments["max_results"])
        return cast(JsonValue, {"events": selected[:limit]})

    def _validate_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[
        _ToolSpec | None,
        dict[str, JsonValue] | None,
        tuple[ActionValidationErrorCode, ...],
    ]:
        spec = _TOOLS.get(tool_name)
        if spec is None:
            return None, None, (ActionValidationErrorCode.TOOL_NOT_ALLOWLISTED,)
        if spec.risk in {
            RiskClass.EXTERNAL_WRITE,
            RiskClass.DESTRUCTIVE,
            RiskClass.HIGH_IMPACT,
        }:
            return spec, None, (ActionValidationErrorCode.RISK_CLASS_DISABLED,)
        try:
            validated = spec.arguments.model_validate(arguments)
        except ValidationError as exc:
            return spec, None, _validation_error_codes(tool_name, exc)
        if (
            tool_name == "local_notification_enqueue"
            and isinstance(validated, NotificationArguments)
            and validated.public_preview is None
        ):
            return (
                spec,
                None,
                (ActionValidationErrorCode.ARGUMENTS_SCHEMA_INVALID,),
            )
        return spec, validated.model_dump(mode="json"), ()

    def _validated_arguments_for_execution(
        self, intent: ActionIntent
    ) -> dict[str, JsonValue]:
        event = current_agent_event()
        if event is None or event.processing_sequence is None:
            raise ActionPolicyError("Action execution requires an authoritative event")
        if intent.validation_record_id is None:
            raise ActionPolicyError("Action intent has no validation record")
        try:
            record = self.get_validation_record(intent.validation_record_id)
        except ValueError as exc:
            raise ActionPolicyError(
                "Action intent validation record is missing"
            ) from exc
        if not record.arguments_valid or record.validation_error_codes:
            raise ActionPolicyError("Action intent validation is invalid")
        if record.intent_id != intent.intent_id or record.tool_name != intent.tool_name:
            raise ActionPolicyError("Action intent validation binding is inconsistent")
        if (
            event.event_id == record.validated_event_id
            or event.processing_sequence <= record.validated_event_sequence
        ):
            raise ActionPolicyError(
                "Action execution event is inconsistent with validation"
            )
        spec = _TOOLS.get(intent.tool_name)
        current_revision = _validation_schema_revision(intent.tool_name, spec)
        if current_revision != record.validation_schema_revision:
            raise ActionPolicyError("Action intent validation schema revision is stale")
        digest = _digest(intent.arguments)
        if (
            digest != record.canonical_arguments_digest
            or digest != intent.policy.argument_digest
            or _digest(intent.preview.arguments) != digest
        ):
            raise ActionPolicyError("Action intent arguments changed after validation")
        return intent.arguments

    def _validate_result(
        self, tool_name: str, result: JsonValue
    ) -> tuple[bool, tuple[str, ...]]:
        if not isinstance(result, dict):
            return False, ("result_not_object",)
        required = {
            "restricted_metadata_read": {"namespace", "key", "value"},
            "document_search": {"matches", "truncated"},
            "calendar_read": {"events"},
            "local_notification_enqueue": {"notification_id", "status"},
        }[tool_name]
        return (
            (True, ())
            if set(result) == required
            else (False, ("result_fields_invalid",))
        )

    def _rollback_partial_effect(self, intent: ActionIntent) -> None:
        if intent.tool_name != "local_notification_enqueue":
            return
        state = self._state()
        notifications = tuple(
            {**item, "status": "cancelled"}
            if item.get("idempotency_key") == intent.idempotency_key
            else item
            for item in state.notifications
        )
        self._save(state.model_copy(update={"notifications": notifications}))

    def _execution_failure(
        self,
        intent: ActionIntent,
        receipt_id: str,
        attempt: int,
        started_at: datetime,
        started_clock: float,
        code: str,
        retryable: bool,
    ) -> ActionIntent:
        now = self.clock()
        status = ReceiptStatus.TIMED_OUT if code == "timeout" else ReceiptStatus.FAILED
        observation, verification = self._failure_evidence(
            intent, receipt_id, status, code, now
        )
        receipt = self._receipt(
            intent,
            receipt_id,
            attempt,
            status,
            started_at,
            now,
            (self.monotonic() - started_clock) * 1000,
            observation_id=observation.observation_id,
            verification_id=verification.verification_id,
            error_code=code,
        )
        updated = intent.model_copy(
            update={
                "revision": intent.revision + 1,
                "status": IntentStatus.RETRY_PENDING
                if retryable
                else IntentStatus.FAILED,
                "attempts": attempt,
                "cost_units_used": intent.cost_units_used + 1,
                "updated_at": now,
                "deadline_at": now + timedelta(seconds=intent.budget.timeout_seconds),
                "retry_at": now + timedelta(seconds=1) if retryable else None,
                "receipt_id": receipt_id,
                "failure_code": code,
            }
        )
        state = self._state()
        self._replace(
            state,
            updated,
            receipts=(*state.receipts, receipt),
            observations=(*state.observations, observation),
            verifications=(*state.verifications, verification),
        )
        if not retryable:
            self._resolve_decision(updated, False, f"action_{code}")
        self._enqueue_outbox(
            OutboxMessageKind.ANOMALY,
            title="Action retry scheduled" if retryable else "Action failed",
            body=f"The {intent.tool_name} action failed with code {code}.",
            deduplication_key=f"action-failure:{intent.intent_id}:{updated.revision}",
            intent=updated,
            urgency=OutboxUrgency.HIGH,
        )
        return updated

    def _enqueue_outbox(
        self,
        kind: OutboxMessageKind,
        *,
        title: str,
        body: str,
        deduplication_key: str,
        intent: ActionIntent,
        urgency: OutboxUrgency = OutboxUrgency.NORMAL,
    ) -> None:
        outbox = getattr(self.main_loop, "outbox", None)
        if outbox is None:
            return
        outbox.enqueue(
            kind,
            title=title,
            body=body,
            deduplication_key=deduplication_key,
            references=OutboxReferences(
                event_id=intent.provenance.triggering_event_id,
                plan_id=intent.provenance.plan_id,
                decision_id=intent.provenance.decision_id,
                action_id=intent.intent_id,
            ),
            urgency=urgency,
            privacy_class=PrivacyClass.OPERATOR,
        )

    def _fail(
        self, intent: ActionIntent, code: str, status: ReceiptStatus
    ) -> ActionIntent:
        now = self.clock()
        receipt_id = str(uuid4())
        observation, verification = self._failure_evidence(
            intent, receipt_id, status, code, now
        )
        receipt = self._receipt(
            intent,
            receipt_id,
            intent.attempts,
            status,
            now,
            now,
            0.0,
            observation_id=observation.observation_id,
            verification_id=verification.verification_id,
            error_code=code,
        )
        updated = intent.model_copy(
            update={
                "revision": intent.revision + 1,
                "status": IntentStatus.FAILED,
                "updated_at": now,
                "retry_at": None,
                "receipt_id": receipt_id,
                "failure_code": code,
            }
        )
        state = self._state()
        self._replace(
            state,
            updated,
            receipts=(*state.receipts, receipt),
            observations=(*state.observations, observation),
            verifications=(*state.verifications, verification),
        )
        self._resolve_decision(updated, False, f"action_{code}")
        self._enqueue_outbox(
            OutboxMessageKind.ANOMALY,
            title="Action failed",
            body=f"The {intent.tool_name} action failed with code {code}.",
            deduplication_key=f"action-failure:{intent.intent_id}:{updated.revision}",
            intent=updated,
            urgency=OutboxUrgency.HIGH,
        )
        return updated

    def _receipt(
        self,
        intent: ActionIntent,
        receipt_id: str,
        attempt: int,
        status: ReceiptStatus,
        started_at: datetime,
        finished_at: datetime,
        duration_ms: float,
        **values: Any,
    ) -> ExecutionReceipt:
        event = current_agent_event()
        return ExecutionReceipt(
            receipt_id=receipt_id,
            intent_id=intent.intent_id,
            idempotency_key=intent.idempotency_key,
            attempt=attempt,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            decision_id=intent.provenance.decision_id,
            plan_id=intent.provenance.plan_id,
            plan_revision=intent.provenance.plan_revision,
            step_id=intent.provenance.step_id,
            **values,
        )

    def _failure_evidence(
        self,
        intent: ActionIntent,
        receipt_id: str,
        status: ReceiptStatus,
        code: str,
        observed_at: datetime,
    ) -> tuple[Observation, OutcomeVerification]:
        data: dict[str, JsonValue] = {
            "status": status.value,
            "error_code": code,
        }
        observation = Observation(
            observation_id=str(uuid4()),
            intent_id=intent.intent_id,
            receipt_id=receipt_id,
            observed_at=observed_at,
            data=data,
            result_digest=_digest(data),
            valid=True,
        )
        verification = OutcomeVerification(
            verification_id=str(uuid4()),
            intent_id=intent.intent_id,
            observation_id=observation.observation_id,
            success=False,
            reason=f"execution_{code}",
            verified_at=observed_at,
        )
        return observation, verification

    def _resolve_decision(
        self, intent: ActionIntent, success: bool, description: str
    ) -> None:
        decision = self.main_loop.decision_store.get(intent.provenance.decision_id)
        self.validate_decision_outcome(decision.decision_id, success)
        if decision.status == DecisionStatus.RESOLVED:
            if (
                decision.actual_outcome is None
                or decision.actual_outcome.success != success
            ):
                raise ValueError(
                    "Terminal decision outcome contradicts action verification"
                )
            return
        record_experience = getattr(
            self.main_loop, "record_verified_action_experience", None
        )
        if callable(record_experience) and current_agent_event() is not None:
            record_experience(intent.intent_id)
        self.main_loop.record_decision_outcome(
            decision.decision_id,
            description=description,
            utility=1.0 if success else -1.0,
            success=success,
        )

    def _state(self) -> ActionState:
        raw = self.main_loop.persistent_state.extensions.get(ACTION_STATE_KEY)
        if raw is None:
            state = ActionState()
            self._save(state)
            return state
        try:
            version = raw.get("schema_version") if isinstance(raw, dict) else None
            legacy_v1 = version == 1
            migrated = version in {1, 2, 3, 5}
            if legacy_v1:
                raw = {**raw, "validation_records": []}
            if version in {1, 2}:
                raw = {**raw, "policy_rejections": []}
            if version in {1, 2, 3}:
                intent_keys = {
                    item.get("validation_record_id"): item.get("idempotency_key")
                    for item in raw.get("intents", [])
                    if isinstance(item, dict)
                }
                validations = []
                for item in raw.get("validation_records", []):
                    if not isinstance(item, dict):
                        validations.append(item)
                        continue
                    validation_id = str(item.get("validation_id", "unknown"))
                    key = intent_keys.get(validation_id) or f"legacy:{validation_id}"
                    validations.append(
                        {
                            **item,
                            "idempotency_key": str(key)[:128],
                            "request_digest": _digest(
                                {"legacy_validation_id": validation_id}
                            ),
                        }
                    )
                raw = {
                    **raw,
                    "schema_version": 4,
                    "validation_records": validations,
                }
            elif version == 5:
                raw = {
                    **raw,
                    "schema_version": 4,
                    "intents": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "operator_binding_nonce"
                        }
                        if isinstance(item, dict)
                        else item
                        for item in raw.get("intents", [])
                    ],
                }
            state = ActionState.model_validate(raw)
            if legacy_v1:
                pending = {
                    IntentStatus.AWAITING_APPROVAL,
                    IntentStatus.APPROVED,
                    IntentStatus.DRY_RUN,
                    IntentStatus.EXECUTING,
                    IntentStatus.RETRY_PENDING,
                }
                intents = []
                for intent in state.intents:
                    safe_success = (
                        intent.status == IntentStatus.SUCCEEDED
                        and self._succeeded_receipt_valid(state, intent)
                    )
                    if intent.status in pending or (
                        intent.status == IntentStatus.SUCCEEDED and not safe_success
                    ):
                        intent = intent.model_copy(
                            update={
                                "revision": intent.revision + 1,
                                "status": IntentStatus.REJECTED,
                                "failure_code": "legacy_unvalidated_intent",
                                "retry_at": None,
                                # Preserve a deterministic timestamp from the
                                # authoritative legacy record during migration.
                                "updated_at": intent.updated_at,
                            }
                        )
                    intents.append(intent)
                rejected_ids = {
                    item.intent_id
                    for item in intents
                    if item.failure_code == "legacy_unvalidated_intent"
                }
                rejected_at_by_id = {
                    item.intent_id: item.updated_at
                    for item in intents
                    if item.intent_id in rejected_ids
                }
                approvals = tuple(
                    item.model_copy(
                        update={
                            "status": "rejected",
                            "resolved_at": rejected_at_by_id[item.intent_id],
                            "reason": "legacy_unvalidated_intent",
                        }
                    )
                    if item.intent_id in rejected_ids and item.status == "pending"
                    else item
                    for item in state.approvals
                )
                state = state.model_copy(
                    update={"intents": tuple(intents), "approvals": approvals}
                )
                self._save(state)
            elif migrated:
                self._save(state)
            return state
        except ValidationError as exc:
            raise ValueError("Invalid action execution state") from exc

    def _save(self, state: ActionState) -> None:
        self.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = state.model_dump(
            mode="json"
        )

    def _replace(self, state: ActionState, intent: ActionIntent, **values: Any) -> None:
        intents = tuple(
            intent if item.intent_id == intent.intent_id else item
            for item in state.intents
        )
        self._save(state.model_copy(update={"intents": intents, **values}))

    def _succeeded_receipt_valid(
        self, state: ActionState, intent: ActionIntent
    ) -> bool:
        if intent.receipt_id is None:
            return False
        receipt = next(
            (item for item in state.receipts if item.receipt_id == intent.receipt_id),
            None,
        )
        if receipt is None:
            return False
        observation = next(
            (
                item
                for item in state.observations
                if item.observation_id == receipt.observation_id
            ),
            None,
        )
        verification = next(
            (
                item
                for item in state.verifications
                if item.verification_id == receipt.verification_id
            ),
            None,
        )
        return bool(
            _semantic_binding(
                receipt,
                intent,
                observation,
                verification,
                successful=True,
            )
        )


def _digest(value: JsonValue | dict[str, JsonValue]) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _raise_corrupt(message: str) -> Any:
    raise ActionPolicyError(message)


def _validation_schema_revision(tool_name: str, spec: _ToolSpec | None) -> str:
    schema: JsonValue = (
        {"tool_name": tool_name, "available": False}
        if spec is None
        else cast(
            JsonValue,
            {
                "tool_name": tool_name,
                "arguments_schema": spec.arguments.model_json_schema(),
                "validator_revision": _VALIDATION_SCHEMA_REVISIONS.get(tool_name, 0),
            },
        )
    )
    return _digest(schema)


ValidatedActionState = ActionOperatorSnapshot


def validate_action_state_semantics(state: ActionState) -> ValidatedActionState:
    """Validate a persisted action graph without accessing runtime services.

    The returned snapshot is an immutable, indexed view of *state*.  A
    ``ValueError`` (normally ``ActionPolicyError``) means that no part of the
    graph is trusted; callers must not use a partially validated projection.
    """
    snapshot = _snapshot_from_state(state)
    intents = snapshot.intents

    def reject(message: str) -> NoReturn:
        raise ActionPolicyError(message)

    rejection_ids = [item.rejection_id for item in state.policy_rejections]
    if len(rejection_ids) != len(set(rejection_ids)):
        reject("Policy rejection identifiers are not unique")
    intent_keys = [item.idempotency_key for item in state.intents]
    if len(intent_keys) != len(set(intent_keys)):
        reject("Action idempotency keys are not unique")
    validation_owners: dict[str, int] = {
        item.validation_id: 0 for item in state.validation_records
    }
    for intent in state.intents:
        if intent.validation_record_id is not None:
            validation_owners[intent.validation_record_id] = (
                validation_owners.get(intent.validation_record_id, 0) + 1
            )
    for rejection in state.policy_rejections:
        validation_owners[rejection.validation_id] = (
            validation_owners.get(rejection.validation_id, 0) + 1
        )
    for record in state.validation_records:
        expected_owners = 1 if record.arguments_valid else 0
        if validation_owners.get(record.validation_id, 0) != expected_owners:
            reject("Action validation ownership is invalid")
        if not record.arguments_valid and record.intent_id is not None:
            reject("Invalid validation cannot bind an intent")

    # Every cross-record reference is owned by the intent named by the record.
    for approval in state.approvals:
        if approval.intent_id not in intents:
            reject("Approval record has an invalid intent binding")
    for receipt in state.receipts:
        bound_intent = intents.get(receipt.intent_id)
        if bound_intent is None or not receipt_matches_intent(receipt, bound_intent):
            reject("Receipt record has an invalid intent binding")
        if (receipt.status == ReceiptStatus.COMPENSATED) != (
            receipt.compensation_of is not None
        ):
            reject("Compensation receipt status and target disagree")
        if receipt.compensation_of is not None:
            target = snapshot.receipts.get(receipt.compensation_of)
            if target is None or target.intent_id != receipt.intent_id:
                reject("Compensation receipt has an invalid target binding")
            if target.receipt_id == receipt.receipt_id:
                reject("Compensation receipt cannot target itself")
        else:
            observation = snapshot.observations.get(receipt.observation_id or "")
            verification = snapshot.verifications.get(receipt.verification_id or "")
            if (
                observation is None
                or verification is None
                or observation.receipt_id != receipt.receipt_id
                or verification.observation_id != observation.observation_id
                or observation.valid is not True
                or verification.success != (receipt.status == ReceiptStatus.SUCCEEDED)
            ):
                reject("Receipt evidence is missing or has an invalid status binding")
    for observation in state.observations:
        bound_receipt = snapshot.receipts.get(observation.receipt_id)
        if observation.intent_id not in intents or bound_receipt is None:
            reject("Observation record has an invalid binding")
        if bound_receipt.intent_id != observation.intent_id:
            reject("Observation record has an invalid receipt binding")
    for verification in state.verifications:
        if verification.intent_id not in intents:
            reject("Verification record has an invalid intent binding")
        if verification.observation_id is not None:
            observation = snapshot.observations.get(verification.observation_id)
            if observation is None or observation.intent_id != verification.intent_id:
                reject("Verification record has an invalid observation binding")

    for intent in intents.values():
        if not _intent_records_consistent_indexed(snapshot, intent):
            reject("Action intent semantic binding is invalid")
        intent_receipts = snapshot.receipts_by_intent.get(intent.intent_id, ())
        if (intent.status == IntentStatus.DRY_RUN) != intent.dry_run:
            reject("Action dry-run status is inconsistent")
        if intent.status in {
            IntentStatus.AWAITING_APPROVAL,
            IntentStatus.APPROVED,
            IntentStatus.EXECUTING,
            IntentStatus.DRY_RUN,
        } and (intent.receipt_id is not None or intent_receipts):
            reject("Unexecuted action cannot contain receipt evidence")
        terminal = {
            IntentStatus.SUCCEEDED,
            IntentStatus.FAILED,
            IntentStatus.CANCELLED,
            IntentStatus.REJECTED,
            IntentStatus.COMPENSATED,
        }
        if intent.status in terminal and intent.receipt_id is None:
            reject("Terminal action requires a linked receipt")
        if intent.status == IntentStatus.RETRY_PENDING and intent.receipt_id is None:
            reject("Retry-pending action requires failed receipt evidence")
        if intent.receipt_id is not None:
            linked_receipt = snapshot.receipts.get(intent.receipt_id)
            if linked_receipt is None or linked_receipt.intent_id != intent.intent_id:
                reject("Action intent receipt binding is invalid")
            expected = {
                IntentStatus.SUCCEEDED: ReceiptStatus.SUCCEEDED,
                IntentStatus.FAILED: {ReceiptStatus.FAILED, ReceiptStatus.TIMED_OUT},
                IntentStatus.CANCELLED: ReceiptStatus.CANCELLED,
                IntentStatus.REJECTED: ReceiptStatus.CANCELLED,
                IntentStatus.COMPENSATED: ReceiptStatus.COMPENSATED,
            }.get(intent.status)
            if expected is not None and linked_receipt.status not in (
                expected if isinstance(expected, set) else {expected}
            ):
                reject("Action status does not match receipt status")
            if (
                intent.status == IntentStatus.RETRY_PENDING
                and linked_receipt.status
                not in {
                    ReceiptStatus.FAILED,
                    ReceiptStatus.TIMED_OUT,
                }
            ):
                reject("Retry-pending action must link failed receipt evidence")

    for rejection in state.policy_rejections:
        bound_record = snapshot.validations.get(rejection.validation_id)
        if (
            bound_record is None
            or bound_record.arguments_valid is not True
            or bound_record.decision_id != rejection.decision_id
            or bound_record.idempotency_key != rejection.idempotency_key
            or bound_record.risk_class != rejection.risk_class
            or bound_record.intent_id in intents
            or any(
                item.idempotency_key == rejection.idempotency_key
                for item in state.intents
            )
        ):
            reject("Policy rejection binding is invalid")

    notification_keys: set[str] = set()
    notification_ids: set[str] = set()
    for notification in state.notifications:
        key = notification.get("idempotency_key")
        notification_id = notification.get("notification_id")
        notification_intent = next(
            (item for item in state.intents if item.idempotency_key == key), None
        )
        if (
            not isinstance(key, str)
            or key in notification_keys
            or not isinstance(notification_id, str)
            or notification_id in notification_ids
            or set(notification)
            != {
                "notification_id",
                "idempotency_key",
                "channel",
                "title",
                "body",
                "status",
                "created_at",
            }
            or notification_intent is None
            or notification_intent.tool_name != "local_notification_enqueue"
            or notification.get("channel") != "local"
            or notification.get("status") not in {"queued", "cancelled"}
            or notification.get("title") != notification_intent.arguments.get("title")
            or notification.get("body") != notification_intent.arguments.get("body")
        ):
            reject("Notification idempotency binding is invalid")
        try:
            created_at = datetime.fromisoformat(str(notification.get("created_at")))
        except ValueError as exc:
            raise ActionPolicyError("Notification timestamp is invalid") from exc
        if created_at.tzinfo is None:
            reject("Notification timestamp is invalid")
        if isinstance(key, str):
            notification_keys.add(key)
        if isinstance(notification_id, str):
            notification_ids.add(notification_id)
    return snapshot


def _validation_error_codes(
    tool_name: str, error: ValidationError
) -> tuple[ActionValidationErrorCode, ...]:
    messages = " ".join(str(item.get("msg", "")) for item in error.errors())
    locations = {str(part) for item in error.errors() for part in item.get("loc", ())}
    codes: list[ActionValidationErrorCode] = []
    if tool_name == "document_search" and (
        "relative_path" in locations or "document relative_path" in messages
    ):
        codes.append(ActionValidationErrorCode.ARGUMENT_PATH_OUT_OF_SCOPE)
    if tool_name == "restricted_metadata_read" and (
        {"namespace", "key"} & locations or "namespace" in messages
    ):
        codes.append(ActionValidationErrorCode.ARGUMENT_SCOPE_INVALID)
    if not codes:
        codes.append(ActionValidationErrorCode.ARGUMENTS_SCHEMA_INVALID)
    return tuple(codes)
