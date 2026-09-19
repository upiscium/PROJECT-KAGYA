"""Finite, deterministic process-local context registry."""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import math
from threading import RLock
from typing import cast

from kagya.identifiers import validate_identifier


MAX_CONTEXTS = 1024
MAX_PARTICIPANTS_PER_CONTEXT = 32
MAX_RELATIONS_PER_CONTEXT = 64
MAX_INTERLOCUTOR_BINDINGS = 1024
MAX_EVIDENCE_REFERENCES = 32
class ContextError(ValueError):
    """Base class for context domain failures."""


class ContextNotFound(ContextError):
    """A requested Context does not exist."""


class ContextConflict(ContextError):
    """A Context identity conflicts with existing state."""


class ContextCapacityExceeded(ContextError):
    """A fixed Context-domain capacity would be exceeded."""


class ContextStateInvalid(ContextError):
    """Context state or a requested transition is invalid."""


class ContextType(StrEnum):
    """Explicitly supported Context kinds."""

    CONVERSATION = "conversation"


class ContextStatus(StrEnum):
    """Finite Context lifecycle states."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class ContextRelation(StrEnum):
    """Pure compatibility classifications in precedence order."""

    SAME_CONTEXT = "same_context"
    PARENT_CHILD = "parent_child"
    RELATED = "related"
    SHARED_INTERLOCUTOR = "shared_interlocutor"
    LEGACY_UNKNOWN = "legacy_unknown"
    UNKNOWN_CONTEXT = "unknown_context"
    UNRELATED = "unrelated"


@dataclass(frozen=True, slots=True)
class ContextFrame:
    """Immutable authoritative Context identity and lifecycle state."""

    context_id: str
    context_type: ContextType
    source_channel: str
    source_session_id: str | None
    participant_refs: tuple[str, ...]
    parent_context_id: str | None
    related_context_ids: tuple[str, ...]
    status: ContextStatus
    created_revision: int
    last_modified_revision: int
    started_at: datetime
    last_active_at: datetime


@dataclass(frozen=True, slots=True)
class InterlocutorBinding:
    """Minimal evidence-bound identity uncertainty for one opaque reference."""

    reference_key: str
    identity_key: str | None
    confidence: float
    evidence_references: tuple[str, ...]
    created_revision: int
    last_modified_revision: int


@dataclass(frozen=True, slots=True)
class ContextCompatibility:
    """Immutable result of a pure Context compatibility read."""

    score: float
    relation: ContextRelation
    source_context_id: str | None
    current_context_id: str


@dataclass(frozen=True, slots=True)
class ContextRegistryState:
    """Deterministically ordered exact Context state projection."""

    revision: int
    current_context_id: str | None
    frames: tuple[ContextFrame, ...]
    interlocutor_bindings: tuple[InterlocutorBinding, ...]


def _id(value: object, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    try:
        return validate_identifier(value)
    except Exception:
        raise ContextStateInvalid("invalid identifier") from None


def _refs(values: Iterable[object], limit: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ContextStateInvalid("invalid references")
    try:
        iterator = iter(values)
    except Exception:
        raise ContextStateInvalid("invalid references") from None
    checked: set[str] = set()
    for index in range(limit + 1):
        try:
            value = next(iterator)
        except StopIteration:
            return tuple(sorted(checked))
        except Exception:
            raise ContextStateInvalid("invalid references") from None
        if index >= limit:
            raise ContextCapacityExceeded("reference capacity exceeded")
        item = _id(value)
        if item is None:
            raise ContextStateInvalid("invalid references")
        checked.add(item)
    raise ContextCapacityExceeded("reference capacity exceeded")


def _revision(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ContextStateInvalid("invalid revision")
    return value


def _time(value: object, *, require_utc: bool = False) -> datetime:
    if type(value) is not datetime or value.tzinfo is None:
        raise ContextStateInvalid("invalid timestamp")
    try:
        offset = value.utcoffset()
        if offset is None or (require_utc and offset != timedelta(0)):
            raise ValueError
        result = value if require_utc else value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        raise ContextStateInvalid("invalid timestamp") from None
    return result


def _confidence(value: object, *, canonical: bool = False) -> float:
    if type(value) not in (float, int):
        raise ContextStateInvalid("invalid confidence")
    if canonical and type(value) is not float:
        raise ContextStateInvalid("invalid confidence")
    try:
        result = float(cast(float | int, value))
    except (OverflowError, TypeError, ValueError):
        raise ContextStateInvalid("invalid confidence") from None
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ContextStateInvalid("invalid confidence")
    return result


class ContextRegistry:
    """Finite authority for Context lifecycle, relations, and identity bindings."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock if clock is not None else lambda: datetime.now(timezone.utc)
        self._revision = 0
        self._current: str | None = None
        self._frames: dict[str, ContextFrame] = {}
        self._bindings: dict[str, InterlocutorBinding] = {}
        self._lock = RLock()
        self._mutating = False

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        with self._lock:
            if self._mutating:
                raise ContextConflict("context mutation already in progress")
            self._mutating = True
            try:
                yield
            finally:
                self._mutating = False

    @property
    def state(self) -> ContextRegistryState:
        with self._lock:
            return ContextRegistryState(
                self._revision,
                self._current,
                tuple(self._frames[k] for k in sorted(self._frames)),
                tuple(self._bindings[k] for k in sorted(self._bindings)),
            )

    @property
    def current_context_id(self) -> str | None:
        with self._lock:
            return self._current

    @property
    def current_context(self) -> ContextFrame | None:
        """Return the exact active current frame without changing registry state."""

        with self._lock:
            if self._current is None:
                return None
            frame = self._frames.get(self._current)
            if frame is None or frame.status is not ContextStatus.ACTIVE:
                raise ContextStateInvalid("current context is invalid")
            return frame

    def _frame(self, context_id: str) -> ContextFrame:
        frame = self._frames.get(context_id)
        if frame is None:
            raise ContextNotFound("context not found")
        return frame

    def _tick(self) -> int:
        self._revision += 1
        return self._revision

    def _now(self) -> datetime:
        try:
            value = self._clock()
        except Exception:
            raise ContextStateInvalid("context clock failed") from None
        return _time(value)

    def create(
        self,
        context_id: str,
        context_type: ContextType,
        source_channel: str,
        source_session_id: str | None = None,
        participant_refs: Iterable[str] = (),
        parent_context_id: str | None = None,
    ) -> ContextFrame:
        with self._mutation():
            _id(context_id)
            _id(source_channel)
            _id(source_session_id, True)
            if type(context_type) is not ContextType:
                raise ContextStateInvalid("invalid context type")
            if context_id in self._frames:
                raise ContextConflict("context already exists")
            if len(self._frames) >= MAX_CONTEXTS:
                raise ContextCapacityExceeded("context capacity exceeded")
            parent = _id(parent_context_id, True)
            if parent is not None and (
                parent == context_id or parent not in self._frames
            ):
                raise ContextStateInvalid("invalid parent context")
            participants = _refs(participant_refs, MAX_PARTICIPANTS_PER_CONTEXT)
            now = self._now()
            revision = self._tick()
            frame = ContextFrame(
                context_id,
                context_type,
                source_channel,
                source_session_id,
                participants,
                parent,
                (),
                ContextStatus.ACTIVE,
                revision,
                revision,
                now,
                now,
            )
            self._frames[context_id] = frame
            return frame

    def set_current(self, context_id: str | None) -> None:
        with self._mutation():
            if context_id is not None:
                _id(context_id)
                frame = self._frame(context_id)
                if frame.status is not ContextStatus.ACTIVE:
                    raise ContextStateInvalid("current context must be active")
            if self._current == context_id:
                return
            self._current = context_id
            self._tick()

    def _transition(self, context_id: str, status: ContextStatus) -> ContextFrame:
        with self._mutation():
            _id(context_id)
            frame = self._frame(context_id)
            if frame.status is status:
                return frame
            valid = (
                frame.status is ContextStatus.ACTIVE
                and status in (ContextStatus.SUSPENDED, ContextStatus.CLOSED)
            ) or (
                frame.status is ContextStatus.SUSPENDED
                and status in (ContextStatus.ACTIVE, ContextStatus.CLOSED)
            )
            if not valid:
                raise ContextStateInvalid("invalid context transition")
            now = self._now()
            if status is ContextStatus.ACTIVE and now < frame.last_active_at:
                raise ContextStateInvalid("context clock regressed")
            revision = self._tick()
            active = now if status is ContextStatus.ACTIVE else frame.last_active_at
            updated = ContextFrame(
                frame.context_id,
                frame.context_type,
                frame.source_channel,
                frame.source_session_id,
                frame.participant_refs,
                frame.parent_context_id,
                frame.related_context_ids,
                status,
                frame.created_revision,
                revision,
                frame.started_at,
                active,
            )
            self._frames[context_id] = updated
            if self._current == context_id and status is not ContextStatus.ACTIVE:
                self._current = None
            return updated

    def suspend(self, context_id: str) -> ContextFrame:
        return self._transition(context_id, ContextStatus.SUSPENDED)

    def resume(self, context_id: str) -> ContextFrame:
        return self._transition(context_id, ContextStatus.ACTIVE)

    def close(self, context_id: str) -> ContextFrame:
        return self._transition(context_id, ContextStatus.CLOSED)

    def relate(self, context_id: str, related_context_id: str) -> None:
        with self._mutation():
            _id(context_id)
            _id(related_context_id)
            left = self._frame(context_id)
            right = self._frame(related_context_id)
            if context_id == related_context_id:
                raise ContextStateInvalid("context cannot relate to itself")
            if (
                related_context_id in left.related_context_ids
                and context_id in right.related_context_ids
            ):
                return
            new_left = tuple(
                sorted(set(left.related_context_ids) | {related_context_id})
            )
            new_right = tuple(sorted(set(right.related_context_ids) | {context_id}))
            if (
                len(new_left) > MAX_RELATIONS_PER_CONTEXT
                or len(new_right) > MAX_RELATIONS_PER_CONTEXT
            ):
                raise ContextCapacityExceeded("relation capacity exceeded")
            revision = self._tick()
            self._frames[context_id] = ContextFrame(
                left.context_id,
                left.context_type,
                left.source_channel,
                left.source_session_id,
                left.participant_refs,
                left.parent_context_id,
                new_left,
                left.status,
                left.created_revision,
                revision,
                left.started_at,
                left.last_active_at,
            )
            self._frames[related_context_id] = ContextFrame(
                right.context_id,
                right.context_type,
                right.source_channel,
                right.source_session_id,
                right.participant_refs,
                right.parent_context_id,
                new_right,
                right.status,
                right.created_revision,
                revision,
                right.started_at,
                right.last_active_at,
            )

    def upsert_interlocutor_binding(
        self,
        reference_key: str,
        identity_key: str | None = None,
        confidence: float | int = 1.0,
        evidence_references: Iterable[str] = (),
    ) -> InterlocutorBinding:
        with self._mutation():
            _id(reference_key)
            _id(identity_key, True)
            confidence_value = _confidence(confidence)
            evidence = _refs(evidence_references, MAX_EVIDENCE_REFERENCES)
            old = self._bindings.get(reference_key)
            if old is not None and (
                old.identity_key,
                old.confidence,
                old.evidence_references,
            ) == (identity_key, confidence_value, evidence):
                return old
            if old is None and len(self._bindings) >= MAX_INTERLOCUTOR_BINDINGS:
                raise ContextCapacityExceeded("binding capacity exceeded")
            revision = self._tick()
            binding = InterlocutorBinding(
                reference_key,
                identity_key,
                confidence_value,
                evidence,
                old.created_revision if old else revision,
                revision,
            )
            self._bindings[reference_key] = binding
            return binding

    def compatibility(
        self, source_context_id: str | None, current_context_id: str
    ) -> ContextCompatibility:
        with self._lock:
            _id(source_context_id, True)
            _id(current_context_id)
            current = self._frame(current_context_id)
            if current.status is not ContextStatus.ACTIVE:
                raise ContextStateInvalid("current context must be active")
            source = (
                None
                if source_context_id is None
                else self._frames.get(source_context_id)
            )
            if source_context_id is None:
                relation = ContextRelation.LEGACY_UNKNOWN
            elif source is None:
                relation = ContextRelation.UNKNOWN_CONTEXT
            elif source.context_id == current.context_id:
                relation = ContextRelation.SAME_CONTEXT
            elif (
                source.parent_context_id == current.context_id
                or current.parent_context_id == source.context_id
            ):
                relation = ContextRelation.PARENT_CHILD
            elif (
                current.context_id in source.related_context_ids
                or source.context_id in current.related_context_ids
            ):
                relation = ContextRelation.RELATED
            else:
                overlap = set(source.participant_refs) & set(current.participant_refs)
                relation = (
                    ContextRelation.SHARED_INTERLOCUTOR
                    if overlap
                    else ContextRelation.UNRELATED
                )
            scores = {
                ContextRelation.SAME_CONTEXT: 1.0,
                ContextRelation.PARENT_CHILD: 0.8,
                ContextRelation.RELATED: 0.75,
                ContextRelation.SHARED_INTERLOCUTOR: 0.65,
                ContextRelation.LEGACY_UNKNOWN: 0.45,
                ContextRelation.UNKNOWN_CONTEXT: 0.35,
                ContextRelation.UNRELATED: 0.2,
            }
            return ContextCompatibility(
                scores[relation], relation, source_context_id, current_context_id
            )

    def restore_exact(self, state: ContextRegistryState) -> None:
        with self._mutation():
            try:
                validate_context_registry_state(state)
            except ContextError:
                raise
            except Exception:
                raise ContextStateInvalid("invalid registry state") from None
            self._revision = state.revision
            self._current = state.current_context_id
            self._frames = {frame.context_id: frame for frame in state.frames}
            self._bindings = {
                binding.reference_key: binding
                for binding in state.interlocutor_bindings
            }

    @staticmethod
    def _validate_state(state: ContextRegistryState) -> None:
        if type(state) is not ContextRegistryState:
            raise ContextStateInvalid("invalid registry state")
        revision = _revision(state.revision)
        _id(state.current_context_id, True)
        if (
            type(state.frames) is not tuple
            or type(state.interlocutor_bindings) is not tuple
        ):
            raise ContextStateInvalid("invalid registry state")
        if (
            len(state.frames) > MAX_CONTEXTS
            or len(state.interlocutor_bindings) > MAX_INTERLOCUTOR_BINDINGS
        ):
            raise ContextCapacityExceeded("registry capacity exceeded")
        frames: dict[str, ContextFrame] = {}
        creation_revisions: set[int] = set()
        for frame in state.frames:
            if type(frame) is not ContextFrame or frame.context_id in frames:
                raise ContextStateInvalid("invalid frame")
            _id(frame.context_id)
            _id(frame.source_channel)
            _id(frame.source_session_id, True)
            if (
                type(frame.context_type) is not ContextType
                or type(frame.status) is not ContextStatus
            ):
                raise ContextStateInvalid("invalid frame")
            if (
                type(frame.participant_refs) is not tuple
                or type(frame.related_context_ids) is not tuple
            ):
                raise ContextStateInvalid("invalid frame references")
            if frame.participant_refs != _refs(
                frame.participant_refs, MAX_PARTICIPANTS_PER_CONTEXT
            ) or frame.related_context_ids != _refs(
                frame.related_context_ids, MAX_RELATIONS_PER_CONTEXT
            ):
                raise ContextStateInvalid("noncanonical references")
            _id(frame.parent_context_id, True)
            _revision(frame.created_revision)
            _revision(frame.last_modified_revision)
            if (
                frame.created_revision > frame.last_modified_revision
                or frame.last_modified_revision > revision
                or frame.created_revision == 0
            ):
                raise ContextStateInvalid("invalid frame revisions")
            if frame.created_revision in creation_revisions:
                raise ContextStateInvalid("duplicate creation revision")
            creation_revisions.add(frame.created_revision)
            started = _time(frame.started_at, require_utc=True)
            active = _time(frame.last_active_at, require_utc=True)
            if (
                started > active
                or started != frame.started_at
                or active != frame.last_active_at
            ):
                raise ContextStateInvalid("noncanonical timestamps")
            frames[frame.context_id] = frame
        if tuple(frame.context_id for frame in state.frames) != tuple(sorted(frames)):
            raise ContextStateInvalid("noncanonical registry ordering")
        for frame in frames.values():
            if frame.parent_context_id is not None and (
                frame.parent_context_id not in frames
                or frame.parent_context_id == frame.context_id
            ):
                raise ContextStateInvalid("invalid parent context")
            if (
                frame.parent_context_id is not None
                and frames[frame.parent_context_id].created_revision
                >= frame.created_revision
            ):
                raise ContextStateInvalid("invalid parent creation order")
            if any(
                x not in frames or x == frame.context_id
                for x in frame.related_context_ids
            ):
                raise ContextStateInvalid("invalid related context")
        for frame in frames.values():
            for related in frame.related_context_ids:
                if frame.context_id not in frames[related].related_context_ids:
                    raise ContextStateInvalid("asymmetric relation")
        bindings: dict[str, InterlocutorBinding] = {}
        for binding in state.interlocutor_bindings:
            if (
                type(binding) is not InterlocutorBinding
                or binding.reference_key in bindings
            ):
                raise ContextStateInvalid("invalid binding")
            _id(binding.reference_key)
            _id(binding.identity_key, True)
            _confidence(binding.confidence, canonical=True)
            if type(binding.evidence_references) is not tuple:
                raise ContextStateInvalid("invalid binding evidence")
            if binding.evidence_references != _refs(
                binding.evidence_references, MAX_EVIDENCE_REFERENCES
            ):
                raise ContextStateInvalid("noncanonical evidence")
            _revision(binding.created_revision)
            _revision(binding.last_modified_revision)
            if (
                binding.created_revision > binding.last_modified_revision
                or binding.last_modified_revision > revision
                or binding.created_revision == 0
            ):
                raise ContextStateInvalid("invalid binding revisions")
            if binding.created_revision in creation_revisions:
                raise ContextStateInvalid("duplicate creation revision")
            creation_revisions.add(binding.created_revision)
            bindings[binding.reference_key] = binding
        if tuple(
            binding.reference_key for binding in state.interlocutor_bindings
        ) != tuple(sorted(bindings)):
            raise ContextStateInvalid("noncanonical registry ordering")
        if state.current_context_id is not None and (
            state.current_context_id not in frames
            or frames[state.current_context_id].status is not ContextStatus.ACTIVE
        ):
            raise ContextStateInvalid("invalid current context")


def validate_context_registry_state(state: ContextRegistryState) -> None:
    """Validate an exact registry projection without reading or mutating a registry."""

    try:
        ContextRegistry._validate_state(state)
    except ContextError:
        raise
    except Exception:
        raise ContextStateInvalid("invalid registry state") from None
