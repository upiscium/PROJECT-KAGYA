"""Situation and interlocutor models for the single subject."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ContextStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass(frozen=True)
class InferredAttribute:
    value: Any
    confidence: float
    evidence_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attribute confidence must be between zero and one")


@dataclass(frozen=True)
class InterlocutorModel:
    identity_key: str
    relationship_metadata: dict[str, Any] = field(default_factory=dict)
    shared_history_references: tuple[str, ...] = ()
    inferred_preferences: dict[str, InferredAttribute] = field(default_factory=dict)
    uncertainties: dict[str, InferredAttribute] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextFrame:
    context_id: str
    context_type: str
    source_channel: str
    source_session_id: str | None
    participant_ids: tuple[str, ...]
    active_topic: str | None
    active_task: str | None
    started_at: datetime
    last_active_at: datetime
    parent_context_id: str | None
    related_context_ids: tuple[str, ...]
    status: ContextStatus


class ContextRegistry:
    def __init__(self) -> None:
        self._frames: dict[str, ContextFrame] = {}
        self._interlocutors: dict[str, InterlocutorModel] = {}

    @property
    def frames(self) -> tuple[ContextFrame, ...]:
        return tuple(self._frames.values())

    @property
    def interlocutors(self) -> tuple[InterlocutorModel, ...]:
        return tuple(self._interlocutors.values())

    def create(
        self,
        *,
        context_id: str | None = None,
        context_type: str = "conversation",
        source_channel: str = "api.chat",
        source_session_id: str | None = None,
        participant_ids: tuple[str, ...] = (),
        active_topic: str | None = None,
        active_task: str | None = None,
        parent_context_id: str | None = None,
    ) -> ContextFrame:
        identifier = context_id or f"ctx-{uuid4()}"
        if identifier in self._frames:
            raise ValueError(f"Context already exists: {identifier}")
        if parent_context_id == identifier:
            raise ValueError("Context cannot be its own parent")
        now = datetime.now(UTC)
        frame = ContextFrame(
            context_id=identifier,
            context_type=context_type,
            source_channel=source_channel,
            source_session_id=source_session_id,
            participant_ids=participant_ids,
            active_topic=active_topic,
            active_task=active_task,
            started_at=now,
            last_active_at=now,
            parent_context_id=parent_context_id,
            related_context_ids=(),
            status=ContextStatus.ACTIVE,
        )
        self._frames[identifier] = frame
        return frame

    def get(self, context_id: str) -> ContextFrame | None:
        return self._frames.get(context_id)

    def resume(self, context_id: str) -> ContextFrame:
        frame = self._require(context_id)
        if frame.status == ContextStatus.CLOSED:
            raise ValueError("Closed context cannot be resumed")
        updated = replace(
            frame, status=ContextStatus.ACTIVE, last_active_at=datetime.now(UTC)
        )
        self._frames[context_id] = updated
        return updated

    def suspend(self, context_id: str) -> ContextFrame:
        return self._set_status(context_id, ContextStatus.SUSPENDED)

    def end(self, context_id: str) -> ContextFrame:
        return self._set_status(context_id, ContextStatus.CLOSED)

    def relate(self, first_id: str, second_id: str) -> None:
        if first_id == second_id:
            raise ValueError("Context cannot relate to itself")
        first = self._require(first_id)
        second = self._require(second_id)
        self._frames[first_id] = replace(
            first,
            related_context_ids=tuple(sorted(set(first.related_context_ids) | {second_id})),
        )
        self._frames[second_id] = replace(
            second,
            related_context_ids=tuple(sorted(set(second.related_context_ids) | {first_id})),
        )

    def register_interlocutor(self, model: InterlocutorModel) -> None:
        self._interlocutors[model.identity_key] = model

    def get_interlocutor(self, identity_key: str) -> InterlocutorModel | None:
        return self._interlocutors.get(identity_key)

    def record_shared_history(
        self, identity_keys: tuple[str, ...], reference: str
    ) -> None:
        for identity_key in identity_keys:
            model = self._interlocutors.get(identity_key)
            if model is None:
                continue
            self._interlocutors[identity_key] = replace(
                model,
                shared_history_references=tuple(
                    dict.fromkeys((*model.shared_history_references, reference))
                ),
            )

    def restore(
        self,
        frames: tuple[ContextFrame, ...],
        interlocutors: tuple[InterlocutorModel, ...],
    ) -> None:
        self._frames = {frame.context_id: frame for frame in frames}
        self._interlocutors = {model.identity_key: model for model in interlocutors}

    def clear(self) -> None:
        self._frames.clear()
        self._interlocutors.clear()

    def compatibility(self, source_context_id: str | None, current: ContextFrame) -> tuple[float, str]:
        current = self._frames.get(current.context_id, current)
        if source_context_id is None:
            return 0.45, "legacy_unknown"
        if source_context_id == current.context_id:
            return 1.0, "same_context"
        source = self._frames.get(source_context_id)
        if source is None:
            return 0.35, "unknown_context"
        if (
            source.context_id == current.parent_context_id
            or source.parent_context_id == current.context_id
        ):
            return 0.8, "parent_child"
        if source.context_id in current.related_context_ids:
            return 0.75, "related"
        if set(source.participant_ids) & set(current.participant_ids):
            return 0.65, "shared_interlocutor"
        return 0.2, "unrelated"

    def _set_status(self, context_id: str, status: ContextStatus) -> ContextFrame:
        frame = self._require(context_id)
        updated = replace(frame, status=status, last_active_at=datetime.now(UTC))
        self._frames[context_id] = updated
        return updated

    def _require(self, context_id: str) -> ContextFrame:
        frame = self._frames.get(context_id)
        if frame is None:
            raise KeyError(context_id)
        return frame
