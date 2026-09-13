"""Deterministic, bounded, reference-only Working Memory authority."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from threading import RLock


MAX_ITEM_CAPACITY = 4_096
MAX_PROJECTION_BYTES = 16 * 1024 * 1024
MAX_SOURCE_ID_BYTES = 128
REACTIVATION_BONUS = 0.2
_ITEM_ID_DOMAIN = b"kagya-working-memory-item-v1\0"
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class WorkingMemorySourceKind(str, Enum):
    """The only source authorities R08 U1 may reference."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class WorkingMemoryRetentionReason(str, Enum):
    """Small R08-only membership retention vocabulary."""

    RECENT = "recent"
    REACTIVATED = "reactivated"


class WorkingMemoryDecisionReason(str, Enum):
    """Bounded reasons emitted by pure projection selection."""

    SELECTED = "selected"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    RESOLVER_FAILURE = "resolver_failure"
    PROJECTION_BUDGET = "projection_budget"


class WorkingMemoryAdmissionReason(str, Enum):
    """Bounded result of one explicit admission mutation."""

    ADMITTED = "admitted"
    REACTIVATED = "reactivated"
    CAPACITY_EVICTED = "capacity_evicted"


@dataclass(frozen=True, slots=True)
class WorkingMemoryItem:
    """Authoritative reference membership and R08-owned ranking metadata."""

    item_id: str
    source_kind: WorkingMemorySourceKind
    source_id: str
    activation: float
    salience: float
    retention_reason: WorkingMemoryRetentionReason
    created_revision: int
    last_activated_revision: int


@dataclass(frozen=True, slots=True)
class WorkingMemoryAdmission:
    """Explicitly distinguishes retained admission from immediate eviction."""

    item: WorkingMemoryItem
    retained: bool
    reason: WorkingMemoryAdmissionReason
    evicted_item_id: str | None


@dataclass(frozen=True, slots=True)
class WorkingMemorySelection:
    """One ephemeral resolved source selected for projection."""

    item_id: str
    source_kind: WorkingMemorySourceKind
    source_id: str
    rendered_content: str
    score: float
    reason: WorkingMemoryDecisionReason


@dataclass(frozen=True, slots=True)
class WorkingMemoryDecision:
    """One bounded selection decision, including unresolved/rejected items."""

    item_id: str
    source_kind: WorkingMemorySourceKind
    source_id: str
    selected: bool
    score: float
    reason: WorkingMemoryDecisionReason


@dataclass(frozen=True, slots=True)
class WorkingMemoryView:
    """Immutable process-local projection; resolved content is not authority."""

    selected: tuple[WorkingMemorySelection, ...]
    decisions: tuple[WorkingMemoryDecision, ...]
    projected_bytes: int
    item_capacity: int
    projection_max_bytes: int
    revision: int


WorkingMemoryResolver = Callable[[WorkingMemoryItem], str | None]


def working_memory_item_id(
    source_kind: WorkingMemorySourceKind, source_id: str
) -> str:
    """Derive stable identity as SHA-256(domain || kind || NUL || source ID)."""

    _validate_source(source_kind, source_id)
    payload = (
        _ITEM_ID_DOMAIN
        + source_kind.value.encode("ascii")
        + b"\0"
        + source_id.encode("ascii")
    )
    return "wm-" + hashlib.sha256(payload).hexdigest()


class WorkingMemory:
    """Finite authoritative membership with an explicit monotonic revision."""

    def __init__(self, item_capacity: int, projection_max_bytes: int) -> None:
        self._item_capacity = _bounded_int(
            item_capacity, "item_capacity", MAX_ITEM_CAPACITY
        )
        self._projection_max_bytes = _bounded_int(
            projection_max_bytes, "projection_max_bytes", MAX_PROJECTION_BYTES
        )
        self._revision = 0
        self._items: dict[str, WorkingMemoryItem] = {}
        self._lock = RLock()
        self._selecting = False

    @property
    def item_capacity(self) -> int:
        return self._item_capacity

    @property
    def projection_max_bytes(self) -> int:
        return self._projection_max_bytes

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def items(self) -> tuple[WorkingMemoryItem, ...]:
        """Return deterministic immutable authoritative state."""

        with self._lock:
            return tuple(self._items[item_id] for item_id in sorted(self._items))

    def restore_exact(
        self, revision: int, items: Iterable[WorkingMemoryItem]
    ) -> None:
        """Restore canonical membership exactly without replaying mutations."""

        with self._lock:
            self._require_mutation_available()
            # Validate and publish under one lock: a concurrent admission must
            # not be able to slip between validation and exact replacement.
            revision_value = _nonnegative_revision(revision, "revision")
            restored: dict[str, WorkingMemoryItem] = {}
            source_references: set[tuple[WorkingMemorySourceKind, str]] = set()
            for item in items:
                if not isinstance(item, WorkingMemoryItem):
                    raise ValueError("Working Memory item is invalid")
                _validate_source(item.source_kind, item.source_id)
                if item.item_id != working_memory_item_id(
                    item.source_kind, item.source_id
                ):
                    raise ValueError("Working Memory item identity is invalid")
                _strict_unit_float(item.activation, "activation")
                _strict_unit_float(item.salience, "salience")
                if type(item.retention_reason) is not WorkingMemoryRetentionReason:
                    raise ValueError("Working Memory retention reason is invalid")
                created_revision = _nonnegative_revision(
                    item.created_revision, "created_revision"
                )
                last_activated_revision = _nonnegative_revision(
                    item.last_activated_revision, "last_activated_revision"
                )
                if (
                    created_revision > revision_value
                    or last_activated_revision > revision_value
                ):
                    raise ValueError("Working Memory item revision is invalid")
                source_reference = (item.source_kind, item.source_id)
                if item.item_id in restored or source_reference in source_references:
                    raise ValueError("Working Memory item is duplicated")
                restored[item.item_id] = item
                source_references.add(source_reference)
            if len(restored) > self._item_capacity:
                raise ValueError("Working Memory state exceeds configured capacity")
            self._items = restored
            self._revision = revision_value

    @staticmethod
    def score(item: WorkingMemoryItem) -> float:
        """Return the minimal R08 score: 0.6A + 0.4S + retention bonus."""

        retention_bonus = (
            0.1
            if item.retention_reason is WorkingMemoryRetentionReason.REACTIVATED
            else 0.0
        )
        return 0.6 * item.activation + 0.4 * item.salience + retention_bonus

    def admit(
        self,
        source_kind: WorkingMemorySourceKind,
        source_id: str,
        activation: float,
        salience: float,
    ) -> WorkingMemoryAdmission:
        """Admit or reactivate one source reference and enforce hard capacity."""

        activation_value = _unit_float(activation, "activation")
        salience_value = _unit_float(salience, "salience")
        item_id = working_memory_item_id(source_kind, source_id)
        with self._lock:
            self._require_mutation_available()
            self._revision += 1
            existing = self._items.get(item_id)
            if existing is None:
                item = WorkingMemoryItem(
                    item_id=item_id,
                    source_kind=source_kind,
                    source_id=source_id,
                    activation=activation_value,
                    salience=salience_value,
                    retention_reason=WorkingMemoryRetentionReason.RECENT,
                    created_revision=self._revision,
                    last_activated_revision=self._revision,
                )
                reason = WorkingMemoryAdmissionReason.ADMITTED
            else:
                item = WorkingMemoryItem(
                    item_id=existing.item_id,
                    source_kind=existing.source_kind,
                    source_id=existing.source_id,
                    activation=min(
                        1.0,
                        max(existing.activation, activation_value)
                        + REACTIVATION_BONUS,
                    ),
                    salience=max(existing.salience, salience_value),
                    retention_reason=WorkingMemoryRetentionReason.REACTIVATED,
                    created_revision=existing.created_revision,
                    last_activated_revision=self._revision,
                )
                reason = WorkingMemoryAdmissionReason.REACTIVATED
            self._items[item_id] = item
            evicted_item_id: str | None = None
            while len(self._items) > self._item_capacity:
                evicted = min(self._items.values(), key=self._rank_key)
                evicted_item_id = evicted.item_id
                del self._items[evicted.item_id]
            retained = item_id in self._items
            if not retained:
                reason = WorkingMemoryAdmissionReason.CAPACITY_EVICTED
            return WorkingMemoryAdmission(
                item, retained, reason, evicted_item_id
            )

    def advance(
        self, decay: float = 0.15, forget_below: float = 0.1
    ) -> tuple[WorkingMemoryItem, ...]:
        """Apply explicit activation decay and forget weak membership."""

        decay_value = _unit_float(decay, "decay")
        threshold = _unit_float(forget_below, "forget_below")
        with self._lock:
            self._require_mutation_available()
            updated: dict[str, WorkingMemoryItem] = {}
            for item_id, item in self._items.items():
                activation = max(0.0, item.activation - decay_value)
                if activation < threshold:
                    continue
                updated[item_id] = (
                    item
                    if activation == item.activation
                    else WorkingMemoryItem(
                        item_id=item.item_id,
                        source_kind=item.source_kind,
                        source_id=item.source_id,
                        activation=activation,
                        salience=item.salience,
                        retention_reason=item.retention_reason,
                        created_revision=item.created_revision,
                        last_activated_revision=item.last_activated_revision,
                    )
                )
            if updated != self._items:
                self._items = updated
                self._revision += 1
            return tuple(self._items[item_id] for item_id in sorted(self._items))

    def forget(self, item_id: str) -> bool:
        """Remove membership idempotently; missing identity is a no-op."""

        if not isinstance(item_id, str):
            raise TypeError("item_id must be a string")
        with self._lock:
            self._require_mutation_available()
            if item_id not in self._items:
                return False
            del self._items[item_id]
            self._revision += 1
            return True

    def select(self, resolver: WorkingMemoryResolver) -> WorkingMemoryView:
        """Resolve and pack an immutable byte-bounded view without mutation."""

        if not callable(resolver):
            raise TypeError("resolver must be callable")
        with self._lock:
            if self._selecting:
                raise RuntimeError("Working Memory selection is already active")
            self._selecting = True
            ranked = tuple(
                sorted(self._items.values(), key=self._rank_key, reverse=True)
            )
            revision = self._revision
            item_capacity = self._item_capacity
            projection_max_bytes = self._projection_max_bytes
        try:
            selected: list[WorkingMemorySelection] = []
            decisions: list[WorkingMemoryDecision] = []
            projected_bytes = 0
            for item in ranked:
                score = self.score(item)
                reason: WorkingMemoryDecisionReason
                rendered: str | None = None
                try:
                    resolved = resolver(item)
                    if resolved is None:
                        reason = WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
                    elif not isinstance(resolved, str):
                        reason = WorkingMemoryDecisionReason.RESOLVER_FAILURE
                    else:
                        resolved_bytes = resolved.encode("utf-8")
                        if (
                            projected_bytes + len(resolved_bytes)
                            > projection_max_bytes
                        ):
                            reason = WorkingMemoryDecisionReason.PROJECTION_BUDGET
                        else:
                            reason = WorkingMemoryDecisionReason.SELECTED
                            rendered = resolved
                            projected_bytes += len(resolved_bytes)
                except Exception:
                    reason = WorkingMemoryDecisionReason.RESOLVER_FAILURE
                is_selected = reason is WorkingMemoryDecisionReason.SELECTED
                decisions.append(
                    WorkingMemoryDecision(
                        item.item_id,
                        item.source_kind,
                        item.source_id,
                        is_selected,
                        score,
                        reason,
                    )
                )
                if is_selected:
                    assert rendered is not None
                    selected.append(
                        WorkingMemorySelection(
                            item.item_id,
                            item.source_kind,
                            item.source_id,
                            rendered,
                            score,
                            reason,
                        )
                    )
            return WorkingMemoryView(
                tuple(selected),
                tuple(decisions),
                projected_bytes,
                item_capacity,
                projection_max_bytes,
                revision,
            )
        finally:
            with self._lock:
                self._selecting = False

    def _require_mutation_available(self) -> None:
        if self._selecting:
            raise RuntimeError("Working Memory cannot mutate during selection")

    def _rank_key(
        self, item: WorkingMemoryItem
    ) -> tuple[float, float, float, int, int, str]:
        return (
            self.score(item),
            item.activation,
            item.salience,
            item.last_activated_revision,
            item.created_revision,
            item.item_id,
        )


def _bounded_int(value: int, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _unit_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _strict_unit_float(value: object, name: str) -> float:
    """Validate the already-decoded float representation of durable state."""

    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _nonnegative_revision(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_source(
    source_kind: WorkingMemorySourceKind, source_id: str
) -> None:
    if type(source_kind) is not WorkingMemorySourceKind:
        raise TypeError("source_kind must be WorkingMemorySourceKind")
    if not isinstance(source_id, str):
        raise TypeError("source_id must be a string")
    if (
        not source_id
        or len(source_id.encode("utf-8")) > MAX_SOURCE_ID_BYTES
        or _SOURCE_ID.fullmatch(source_id) is None
        or source_id in {".", ".."}
    ):
        raise ValueError("source_id is not a valid bounded opaque identifier")
