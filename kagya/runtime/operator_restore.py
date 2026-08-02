"""The operator-facing, fail-closed point-in-time restore contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
import hashlib
import json
import re
from threading import Lock
from typing import Any, Callable, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

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
    hash_snapshot,
)
from kagya.runtime.state_wal import StateWAL, StateWalIntegrityError


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
                and _SAFE.fullmatch(item[key])
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


def _safe_refs(value: Any, *, limit: int = 20) -> tuple[list[RestoreRef], bool]:
    refs: dict[tuple[str, str], RestoreRef] = {}

    def visit(item: Any) -> None:
        if len(refs) > limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                kind = _REF_KEYS.get(str(key))
                if (
                    kind is not None
                    and isinstance(child, str)
                    and _SAFE.fullmatch(child)
                ):
                    refs[(kind, child)] = RestoreRef(kind=cast(Any, kind), id=child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    ordered = [refs[key] for key in sorted(refs)]
    return ordered[:limit], len(ordered) > limit


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


def _contains_rearmable_records(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, dict):
        return True
    return any(isinstance(item, list) and bool(item) for item in value.values())


def _valid_protected_domain(key: str, value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    requirements = {
        "action_execution": (
            4,
            {
                "intents",
                "validation_records",
                "policy_rejections",
                "approvals",
                "receipts",
                "observations",
                "verifications",
                "notifications",
            },
        ),
        "proactive_outbox": (1, {"messages"}),
        "subject_scheduler": (1, {"schedules"}),
    }
    schema_version, list_keys = requirements[key]
    return value.get("schema_version") == schema_version and all(
        isinstance(value.get(item), list)
        and all(isinstance(record, dict) for record in value[item])
        for item in list_keys
    )


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
        self._capabilities: dict[str, tuple[RestorePreview, bool]] = {}
        self._consumed: set[str] = set()
        self._operations: list[RestoreOperation] = []

    def _fail(self, code: RestoreErrorCode) -> None:
        raise RestoreContractError(code)

    def _assert_authoritative(self, actor_id: str) -> None:
        if self.authority and not self.authority(actor_id):
            self._fail(RestoreErrorCode.NOT_AUTHORITATIVE)

    def _evict_capabilities_locked(self) -> None:
        now = self.clock()
        expired = [
            digest
            for digest, (preview, _) in self._capabilities.items()
            if now >= datetime.fromisoformat(preview.expires_at)
        ]
        for digest in expired:
            self._capabilities.pop(digest, None)
        if len(self._capabilities) > 512:
            ordered = sorted(
                self._capabilities,
                key=lambda digest: self._capabilities[digest][0].created_at,
            )
            for digest in ordered[: len(self._capabilities) - 512]:
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

    def _targets(
        self, wal: list[Any], journal: list[Any]
    ) -> list[tuple[Any, Any, str]]:
        by_seq: dict[int, list[tuple[Any, str]]] = {}
        for j in journal:
            kind: str | None = None
            if j.lifecycle == JournalLifecycle.COMPLETED:
                kind = "journal_completed"
            elif (
                j.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and j.failure_category == "committed_before_crash"
            ):
                kind = "journal_recovered"
            elif j.lifecycle == JournalLifecycle.CHECKPOINT:
                kind = "checkpoint"
            if kind and j.snapshot_sequence is not None and j.snapshot_hash:
                by_seq.setdefault(j.snapshot_sequence, []).append((j, kind))
        result: list[tuple[Any, Any, str]] = []
        for record in wal:
            if (
                record.processing_sequence == 0
                and record.event_type == "state_snapshot"
            ):
                result.append((record, record, "bootstrap"))
                continue
            candidates = by_seq.get(record.processing_sequence, [])
            if not candidates:
                failed_evidence = [
                    item
                    for item in journal
                    if (
                        item.lifecycle == JournalLifecycle.FAILED
                        or (
                            item.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                            and item.failure_category == "uncommitted_after_crash"
                        )
                    )
                    and item.snapshot_sequence == record.processing_sequence
                    and item.snapshot_hash == record.state_hash_after
                ]
                if failed_evidence:
                    continue
                self._fail(RestoreErrorCode.TARGET_UNVERIFIED)
            if len({j.snapshot_hash for j, _ in candidates}) != 1:
                self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
            matches = [
                (j, k)
                for j, k in candidates
                if j.snapshot_hash == record.state_hash_after
            ]
            if not matches:
                self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
            priority = {"journal_completed": 0, "journal_recovered": 1, "checkpoint": 2}
            j, kind = min(matches, key=lambda item: priority[item[1]])
            result.append((record, j, kind))
        if any(
            seq not in {r.processing_sequence for r, _, _ in result} for seq in by_seq
        ):
            self._fail(RestoreErrorCode.CHECKPOINT_MISMATCH)
        evidenced_minimum = min(by_seq, default=None)
        terminal_sequences = {
            item.snapshot_sequence
            for item in journal
            if item.snapshot_sequence is not None
            and item.lifecycle
            in {
                JournalLifecycle.COMPLETED,
                JournalLifecycle.FAILED,
                JournalLifecycle.RECOVERY_CLASSIFIED,
                JournalLifecycle.CHECKPOINT,
            }
        }
        if evidenced_minimum is not None and any(
            record.processing_sequence >= evidenced_minimum
            and record.processing_sequence not in terminal_sequences
            for record in wal
        ):
            self._fail(RestoreErrorCode.TARGET_UNVERIFIED)
        return list(reversed(result))[: self.max_targets]

    def _revision(self, wal: list[Any]) -> int:
        previous: str | None = None
        revision = 0
        for record in wal:
            digest = logical_state_digest(record.patch.value)
            if previous is not None and (
                digest != previous or record.event_type == "state_point_in_time_restore"
            ):
                revision += 1
            previous = digest
        return revision

    def _semantic_events_after(self, wal: list[Any], target_sequence: int) -> int:
        previous: str | None = None
        count = 0
        for record in wal:
            digest = logical_state_digest(record.patch.value)
            if (
                record.processing_sequence > target_sequence
                and previous is not None
                and (
                    digest != previous
                    or record.event_type == "state_point_in_time_restore"
                )
            ):
                count += 1
            previous = digest
        return count

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
            b, a, added, removed, changed = _counts(
                _core_domain_records(name, left),
                _core_domain_records(name, right),
            )
            if changed or added or removed:
                refs, truncated = _safe_refs([left, right])
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
            if key in _EXTERNAL_REARM_DOMAINS and any(
                not _valid_protected_domain(key, value)
                for value in (before.extensions.get(key), after.extensions.get(key))
            ):
                supported = False
            b, a, added, removed, changed = _counts(
                _extension_records(before.extensions.get(key)),
                _extension_records(after.extensions.get(key)),
            )
            if changed or added or removed:
                refs, truncated = _safe_refs(
                    [before.extensions.get(key), after.extensions.get(key)]
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
                if key in _EXTERNAL_REARM_DOMAINS and (
                    _contains_rearmable_records(before.extensions.get(key))
                    or _contains_rearmable_records(after.extensions.get(key))
                ):
                    supported = False
        return domains, supported

    def _external(self, target: int) -> RestoreExternal:
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
                or not _SAFE.fullmatch(r.artifact_id)
                or not _SAFE.fullmatch(r.transaction_id)
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
            digest = _hash(canonical)
            artifacts: list[RestoreArtifact] = []
            if not bad:
                public_types = {
                    _public_artifact_type(effect.artifact_type) for effect in effects
                }
                for kind in sorted(item for item in public_types if item is not None):
                    refs = [
                        effect.artifact_id
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
            return RestoreExternal(
                consistency_status="inconsistent" if bad else "consistent",
                artifacts=artifacts,
                retained_not_replayed_count=len(effects),
                pending_count=pending,
                orphaned_count=orphaned,
                retryable_count=retryable,
                effect_digest=digest,
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
        target = next(
            (
                x
                for x in self._targets(wal, journal)
                if x[0].processing_sequence == target_sequence
            ),
            None,
        )
        if target is None:
            if any(r.processing_sequence == target_sequence for r in wal):
                self._fail(RestoreErrorCode.TARGET_UNVERIFIED)
            self._fail(RestoreErrorCode.TARGET_NOT_RETAINED)
        assert target is not None
        record, _, _ = target
        current = self._current()
        domains, supported = self._domains(current, record.patch.value)
        external = self._external(target_sequence)
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
            "semantic_revision": self._revision(wal),
            "display_sequence": current.last_processed_event_sequence,
            "target_sequence": target_sequence,
            "target_snapshot_hash": record.state_hash_after,
            "newer_authoritative_event_count": self._semantic_events_after(
                wal, target_sequence
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
            self._capabilities[digest] = (preview, False)
        return preview

    def _check_request(
        self,
        preview: RestorePreview,
        request: RestoreCommitRequest,
        *,
        stale: bool = False,
    ) -> None:
        if RestoreErrorCode.UNSUPPORTED_DOMAIN.value in preview.reason_codes:
            self._fail(RestoreErrorCode.UNSUPPORTED_DOMAIN)
        if RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT.value in preview.reason_codes:
            self._fail(RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT)
        if self.clock() >= datetime.fromisoformat(preview.expires_at):
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
            preview, reserved = entry
            if reserved:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
        self._check_request(preview, request)
        wal, journal = self._evidence()
        target = next(
            (
                record
                for record, _, _ in self._targets(wal, journal)
                if record.processing_sequence == request.target_sequence
            ),
            None,
        )
        if target is None or target.state_hash_after != request.expected_target_hash:
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        current = self._current()
        current_external = self._external(request.target_sequence)
        if current_external.consistency_status != "consistent":
            self._fail(RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT)
        if (
            logical_state_digest(current) != request.expected_current_logical_digest
            or self._revision(wal) != request.expected_semantic_revision
            or current_external.effect_digest != request.expected_external_effect_digest
        ):
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        with self._lock:
            entry = self._capabilities.get(request.expected_preview_digest)
            if entry is None:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            assert entry is not None
            if entry[1]:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
            self._capabilities[request.expected_preview_digest] = (entry[0], True)
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
            self._consumed.discard(preview_digest)

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
            preview, reserved = entry
            if not reserved:
                self._fail(RestoreErrorCode.OPERATION_IN_PROGRESS)
        self._check_request(preview, request)
        wal, journal = self._evidence()
        target_record = next(
            (
                record
                for record, _, _ in self._targets(wal, journal)
                if record.processing_sequence == request.target_sequence
            ),
            None,
        )
        if (
            target_record is None
            or target_record.state_hash_after != request.expected_target_hash
        ):
            self._fail(RestoreErrorCode.PREVIEW_STALE)
        current = self._current()
        current_external = self._external(request.target_sequence)
        if current_external.consistency_status != "consistent":
            self._fail(RestoreErrorCode.EXTERNAL_STATE_INCONSISTENT)
        if (
            logical_state_digest(current) != request.expected_current_logical_digest
            or self._revision(wal) != request.expected_semantic_revision
            or current_external.effect_digest != request.expected_external_effect_digest
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
            if item.event_type == "state_point_in_time_restore":
                grouped.setdefault(item.event_id, []).append(item)
        operations: list[RestoreOperation] = []
        for event_id, records in grouped.items():
            if not event_id.startswith("operator-restore-"):
                continue
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
        current = self._current()
        targets = [
            RestoreTarget(
                target_sequence=r.processing_sequence,
                target_snapshot_hash=r.state_hash_after,
                checkpoint_kind=cast(Any, k),
                timestamp=j.timestamp.isoformat(),
                event_type=j.event_type,
                eligible=True,
                reason_codes=[],
            )
            for r, j, k in self._targets(wal, journal)[
                : min(limit or self.max_targets, self.max_targets)
            ]
        ]
        operations_by_id = {item.operation_id: item for item in self._operations}
        operations_by_id.update(
            {item.operation_id: item for item in self._journal_operations(journal)}
        )
        raw = current.extensions.get(_RESERVED, [])
        if raw is not None and not isinstance(raw, list):
            self._fail(RestoreErrorCode.JOURNAL_INTEGRITY_INVALID)
        if isinstance(raw, list):
            for item in raw:
                try:
                    operation = RestoreOperation.model_validate(item)
                    operations_by_id[operation.operation_id] = operation
                except Exception as exc:
                    raise RestoreContractError(
                        RestoreErrorCode.JOURNAL_INTEGRITY_INVALID
                    ) from exc
        operations = list(operations_by_id.values())
        by_event = {j.event_id: j for j in journal}
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
            else:
                projected.append(
                    operation.model_copy(
                        update={
                            "state": "commit_indeterminate",
                            "error_code": RestoreErrorCode.COMMIT_INDETERMINATE.value,
                        }
                    )
                )
        return RestoreSummary(
            current_sequence=current.last_processed_event_sequence,
            current_snapshot_hash=hash_snapshot(current),
            current_logical_digest=logical_state_digest(current),
            semantic_revision=self._revision(wal),
            retained_min_sequence=min((r.processing_sequence for r in wal), default=0),
            retained_max_sequence=max((r.processing_sequence for r in wal), default=0),
            targets=targets,
            latest_operation=projected[-1] if projected else None,
        )
