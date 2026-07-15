"""Finite attention-based working memory for the active subject."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Callable, Iterable


class WorkingMemoryKind(StrEnum):
    CONVERSATION = "conversation"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    GOAL = "goal"
    UNRESOLVED = "unresolved"
    COMMITMENT = "commitment"
    EMOTION = "emotion"


class RetentionReason(StrEnum):
    RECENT_CONTEXT = "recent_context"
    ONGOING_GOAL = "ongoing_goal"
    UNRESOLVED = "unresolved"
    ACTIVE_COMMITMENT = "active_commitment"
    CURRENT_EMOTION = "current_emotion"
    REACTIVATED = "reactivated"


@dataclass(frozen=True)
class WorkingMemoryItem:
    item_id: str
    kind: WorkingMemoryKind
    content: str | None = None
    reference: str | None = None
    source_event_id: str | None = None
    source_event_sequence: int | None = None
    context_id: str | None = None
    source: str = "unknown"
    source_channel: str = "unknown"
    source_session_id: str | None = None
    activation: float = 0.5
    salience: float = 0.5
    created_at: datetime = datetime.min.replace(tzinfo=UTC)
    last_accessed_at: datetime = datetime.min.replace(tzinfo=UTC)
    retention_reason: RetentionReason = RetentionReason.RECENT_CONTEXT

    def __post_init__(self) -> None:
        if not self.content and not self.reference:
            raise ValueError("working-memory item requires content or reference")
        if not 0.0 <= self.activation <= 1.0:
            raise ValueError("activation must be between zero and one")
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError("salience must be between zero and one")
        if self.created_at.tzinfo is None or self.last_accessed_at.tzinfo is None:
            raise ValueError("working-memory timestamps must include a timezone")


@dataclass(frozen=True)
class WorkingMemoryDecision:
    item_id: str
    kind: WorkingMemoryKind
    selected: bool
    score: float
    reasons: tuple[str, ...]
    activation: float
    salience: float
    retention_reason: RetentionReason
    reference: str | None
    context_id: str | None
    context_compatibility: float
    context_relation: str
    cross_context: bool


@dataclass(frozen=True)
class WorkingMemorySelection:
    item: WorkingMemoryItem
    rendered_content: str
    score: float
    reasons: tuple[str, ...]
    context_compatibility: float
    context_relation: str
    cross_context: bool


@dataclass(frozen=True)
class WorkingMemoryView:
    selected: tuple[WorkingMemorySelection, ...]
    decisions: tuple[WorkingMemoryDecision, ...]
    token_count: int
    item_capacity: int
    token_capacity: int

    def context_text(self) -> str:
        return "\n".join(selection.rendered_content for selection in self.selected)


Resolver = Callable[[WorkingMemoryItem], str | None]
TokenCounter = Callable[[str], int]
ContextCompatibility = Callable[[str | None], tuple[float, str]]


class WorkingMemory:
    """Bounded deterministic working-memory store."""

    def __init__(
        self,
        *,
        item_capacity: int,
        token_capacity: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if item_capacity <= 0 or token_capacity <= 0:
            raise ValueError("working-memory capacities must be positive")
        self.item_capacity = item_capacity
        self.token_capacity = token_capacity
        self._token_counter = token_counter or _conservative_token_count
        self._items: dict[str, WorkingMemoryItem] = {}

    @property
    def items(self) -> tuple[WorkingMemoryItem, ...]:
        return tuple(self._items.values())

    def admit(self, item: WorkingMemoryItem) -> None:
        now = _now()
        protected = {
            RetentionReason.ONGOING_GOAL,
            RetentionReason.UNRESOLVED,
            RetentionReason.ACTIVE_COMMITMENT,
            RetentionReason.CURRENT_EMOTION,
        }
        if self._score(item) < 0.15 and item.retention_reason not in protected:
            return
        existing = self._items.get(item.item_id)
        if existing is None and item.reference is not None:
            existing = next(
                (candidate for candidate in self._items.values() if candidate.reference == item.reference),
                None,
            )
        if existing is not None:
            self._items.pop(existing.item_id, None)
            item = replace(
                item,
                item_id=existing.item_id,
                created_at=existing.created_at,
                last_accessed_at=now,
                activation=min(1.0, max(existing.activation, item.activation) + 0.2),
                salience=max(existing.salience, item.salience),
                retention_reason=RetentionReason.REACTIVATED,
            )
        self._items[item.item_id] = item
        self._trim_items()

    def advance(self, *, decay: float = 0.15, forget_below: float = 0.1) -> None:
        protected = {
            RetentionReason.ONGOING_GOAL,
            RetentionReason.UNRESOLVED,
            RetentionReason.ACTIVE_COMMITMENT,
            RetentionReason.CURRENT_EMOTION,
        }
        updated: dict[str, WorkingMemoryItem] = {}
        for item in self._items.values():
            activation = max(0.0, item.activation - decay)
            if activation < forget_below and item.retention_reason not in protected:
                continue
            updated[item.item_id] = replace(item, activation=activation)
        self._items = updated

    def select(
        self,
        *,
        resolver: Resolver | None = None,
        context_compatibility: ContextCompatibility | None = None,
    ) -> WorkingMemoryView:
        def context_score(item: WorkingMemoryItem) -> tuple[float, str]:
            if context_compatibility is None:
                return 1.0, "global"
            return context_compatibility(item.context_id)

        ranked = sorted(
            self._items.values(),
            key=lambda item: (
                self._score(item) + 0.3 * context_score(item)[0],
                item.last_accessed_at,
                item.created_at,
                item.item_id,
            ),
            reverse=True,
        )
        selected: list[WorkingMemorySelection] = []
        decisions: list[WorkingMemoryDecision] = []
        token_count = 0
        for item in ranked:
            rendered = item.content or (resolver(item) if resolver is not None else None)
            compatibility, relation = context_score(item)
            score = self._score(item) + 0.3 * compatibility
            cross_context = item.context_id is not None and relation != "same_context"
            reasons = [item.retention_reason.value, relation]
            selected_item = False
            if not rendered:
                reasons.append("unresolved_reference")
            else:
                item_tokens = self._token_counter(rendered)
                if token_count + item_tokens <= self.token_capacity:
                    selected_item = True
                    token_count += item_tokens
                    selected.append(
                        WorkingMemorySelection(
                            item=item,
                            rendered_content=rendered,
                            score=score,
                            reasons=tuple(reasons),
                            context_compatibility=compatibility,
                            context_relation=relation,
                            cross_context=cross_context,
                        )
                    )
                    self._items[item.item_id] = replace(
                        item, last_accessed_at=_now()
                    )
                else:
                    reasons.append("token_capacity")
            decisions.append(
                WorkingMemoryDecision(
                    item_id=item.item_id,
                    kind=item.kind,
                    selected=selected_item,
                    score=score,
                    reasons=tuple(reasons),
                    activation=item.activation,
                    salience=item.salience,
                    retention_reason=item.retention_reason,
                    reference=item.reference,
                    context_id=item.context_id,
                    context_compatibility=compatibility,
                    context_relation=relation,
                    cross_context=cross_context,
                )
            )
        return WorkingMemoryView(
            selected=tuple(selected),
            decisions=tuple(decisions),
            token_count=token_count,
            item_capacity=self.item_capacity,
            token_capacity=self.token_capacity,
        )

    def forget(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def release_retention(self, item_id: str) -> None:
        item = self._items.get(item_id)
        if item is not None:
            self._items[item_id] = replace(
                item, retention_reason=RetentionReason.RECENT_CONTEXT
            )

    def restore(self, items: Iterable[WorkingMemoryItem]) -> None:
        self._items = {item.item_id: item for item in items}
        self._trim_items()

    def clear(self) -> None:
        self._items.clear()

    def _trim_items(self) -> None:
        while len(self._items) > self.item_capacity:
            lowest = min(self._items.values(), key=self._rank_key)
            del self._items[lowest.item_id]

    def _rank_key(self, item: WorkingMemoryItem) -> tuple[float, datetime, datetime, str]:
        return (
            self._score(item),
            item.last_accessed_at,
            item.created_at,
            item.item_id,
        )

    @staticmethod
    def _score(item: WorkingMemoryItem) -> float:
        bonus = {
            RetentionReason.ONGOING_GOAL: 0.35,
            RetentionReason.UNRESOLVED: 0.3,
            RetentionReason.ACTIVE_COMMITMENT: 0.35,
            RetentionReason.CURRENT_EMOTION: 0.25,
            RetentionReason.REACTIVATED: 0.1,
        }.get(item.retention_reason, 0.0)
        return 0.6 * item.activation + 0.4 * item.salience + bonus


def working_memory_item(
    *,
    item_id: str,
    kind: WorkingMemoryKind,
    content: str | None = None,
    reference: str | None = None,
    source_event_id: str | None = None,
    source_event_sequence: int | None = None,
    context_id: str | None = None,
    source: str = "unknown",
    source_channel: str = "unknown",
    source_session_id: str | None = None,
    activation: float = 0.5,
    salience: float = 0.5,
    retention_reason: RetentionReason = RetentionReason.RECENT_CONTEXT,
) -> WorkingMemoryItem:
    now = _now()
    return WorkingMemoryItem(
        item_id=item_id,
        kind=kind,
        content=content,
        reference=reference,
        source_event_id=source_event_id,
        source_event_sequence=source_event_sequence,
        context_id=context_id,
        source=source,
        source_channel=source_channel,
        source_session_id=source_session_id,
        activation=activation,
        salience=salience,
        created_at=now,
        last_accessed_at=now,
        retention_reason=retention_reason,
    )


def _conservative_token_count(text: str) -> int:
    return max(1, len(text.encode("utf-8")))


def _now() -> datetime:
    return datetime.now(UTC)
