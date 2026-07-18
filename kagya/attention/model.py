"""Structured attention candidates, focus state, and selection history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
import re
from typing import Any, Iterable
from uuid import uuid4


class AttentionSource(StrEnum):
    EXPERIENCE = "experience"
    MOTIVATION = "motivation"
    GOAL = "goal"
    COMMITMENT = "commitment"
    EXTERNAL = "external"


class AttentionCandidateStatus(StrEnum):
    AVAILABLE = "available"
    FOCUSED = "focused"
    DEFERRED = "deferred"
    IGNORED = "ignored"
    INACTIVE = "inactive"


class AttentionAction(StrEnum):
    COMPETE = "compete"
    REFOCUS = "refocus"
    DEFER = "defer"
    IGNORE = "ignore"
    IDLE = "idle"


@dataclass(frozen=True)
class AttentionCandidate:
    candidate_id: str
    target_ref: str
    source: AttentionSource
    source_refs: tuple[str, ...]
    working_memory_ref: str | None
    salience: float
    drive: float
    urgency: float
    novelty: float
    value_relevance: float
    commitment_cost: float
    arousal: float
    persistence: float
    habituation: float
    inhibition: float
    status: AttentionCandidateStatus
    first_observed_at: str
    last_observed_at: str
    focused_cycles: int = 0
    unattended_cycles: int = 0
    revision: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported attention candidate schema version: {self.schema_version}"
            )
        _safe_ref(self.candidate_id, "candidate ID")
        _safe_ref(self.target_ref, "attention target")
        _safe_refs(self.source_refs, "attention source reference")
        if not self.source_refs:
            raise ValueError("attention candidates require source provenance")
        if self.working_memory_ref is not None:
            _safe_ref(self.working_memory_ref, "working-memory reference")
        for name in (
            "salience",
            "drive",
            "urgency",
            "novelty",
            "value_relevance",
            "commitment_cost",
            "arousal",
            "persistence",
            "habituation",
            "inhibition",
        ):
            _unit(getattr(self, name), name)
        if self.focused_cycles < 0 or self.unattended_cycles < 0 or self.revision < 0:
            raise ValueError("attention counters must not be negative")
        _timestamp(self.first_observed_at)
        _timestamp(self.last_observed_at)

    def to_json(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class AttentionFocus:
    candidate_ids: tuple[str, ...]
    idle: bool
    unfinished_candidate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    provenance_refs: tuple[str, ...]
    selected_at: str
    event_id: str | None
    event_sequence: int | None
    revision: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported attention focus schema version")
        if self.idle == bool(self.candidate_ids):
            raise ValueError(
                "idle focus must be empty and active focus must not be empty"
            )
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("focused candidate identifiers must be unique")
        _safe_refs(self.candidate_ids, "focused candidate ID")
        _safe_refs(self.unfinished_candidate_ids, "unfinished candidate ID")
        _safe_refs(self.reason_codes, "attention reason code")
        _safe_refs(self.provenance_refs, "attention provenance reference")
        _timestamp(self.selected_at)
        if self.revision < 0:
            raise ValueError("attention focus revision must not be negative")


@dataclass(frozen=True)
class AttentionHistoryEntry:
    history_id: str
    action: AttentionAction
    from_candidate_ids: tuple[str, ...]
    to_candidate_ids: tuple[str, ...]
    unfinished_candidate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    provenance_refs: tuple[str, ...]
    switch_cost: float
    event_id: str | None
    event_sequence: int | None
    created_at: str

    def __post_init__(self) -> None:
        _safe_ref(self.history_id, "attention history ID")
        for values, name in (
            (self.from_candidate_ids, "previous focus ID"),
            (self.to_candidate_ids, "new focus ID"),
            (self.unfinished_candidate_ids, "unfinished focus ID"),
            (self.reason_codes, "attention reason code"),
            (self.provenance_refs, "attention provenance reference"),
        ):
            _safe_refs(values, name)
        _unit(self.switch_cost, "switch cost")
        _timestamp(self.created_at)


class AttentionSystem:
    """Select a bounded focus independently of prompt token relevance."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        capacity: int,
        high_arousal_cap: int = 1,
        switch_cost: float = 0.18,
        idle_threshold: float = 0.2,
    ) -> None:
        if capacity <= 0:
            raise ValueError("attention capacity must be positive")
        if high_arousal_cap < 0 or high_arousal_cap > capacity:
            raise ValueError("high-arousal cap must fit attention capacity")
        _unit(switch_cost, "switch cost")
        _unit(idle_threshold, "idle threshold")
        self.capacity = capacity
        self.high_arousal_cap = high_arousal_cap
        self.switch_cost = switch_cost
        self.idle_threshold = idle_threshold
        self.candidates: dict[str, AttentionCandidate] = {}
        self.focus = _idle_focus()
        self.history: list[AttentionHistoryEntry] = []

    def observe(
        self,
        *,
        candidate_id: str,
        target_ref: str,
        source: AttentionSource,
        source_refs: tuple[str, ...],
        working_memory_ref: str | None = None,
        salience: float = 0.0,
        drive: float = 0.0,
        urgency: float = 0.0,
        novelty: float = 0.0,
        value_relevance: float = 0.0,
        commitment_cost: float = 0.0,
        arousal: float = 0.0,
        persistence: float = 0.0,
    ) -> AttentionCandidate:
        now = _now()
        current = self.candidates.get(candidate_id)
        if current is None:
            candidate = AttentionCandidate(
                candidate_id=candidate_id,
                target_ref=target_ref,
                source=source,
                source_refs=source_refs,
                working_memory_ref=working_memory_ref,
                salience=salience,
                drive=drive,
                urgency=urgency,
                novelty=novelty,
                value_relevance=value_relevance,
                commitment_cost=commitment_cost,
                arousal=arousal,
                persistence=persistence,
                status=AttentionCandidateStatus.AVAILABLE,
                first_observed_at=now,
                last_observed_at=now,
                habituation=0.0,
                inhibition=0.0,
            )
        else:
            candidate = replace(
                current,
                target_ref=target_ref,
                source=source,
                source_refs=source_refs,
                working_memory_ref=working_memory_ref,
                salience=salience,
                drive=drive,
                urgency=urgency,
                novelty=novelty,
                value_relevance=value_relevance,
                commitment_cost=commitment_cost,
                arousal=arousal,
                persistence=persistence,
                last_observed_at=now,
                habituation=min(1.0, current.habituation + 0.08),
                inhibition=max(0.0, current.inhibition - 0.05),
                revision=current.revision + 1,
            )
        self.candidates[candidate_id] = candidate
        return candidate

    def synchronize_source(
        self, source: AttentionSource, active_candidate_ids: set[str]
    ) -> None:
        for identifier, candidate in list(self.candidates.items()):
            if (
                candidate.source != source
                or identifier in active_candidate_ids
                or candidate.status
                in {AttentionCandidateStatus.IGNORED, AttentionCandidateStatus.INACTIVE}
            ):
                continue
            self.candidates[identifier] = replace(
                candidate,
                status=AttentionCandidateStatus.INACTIVE,
                inhibition=1.0,
                revision=candidate.revision + 1,
            )

    def compete(
        self,
        *,
        event_id: str | None = None,
        event_sequence: int | None = None,
        reason_code: str = "automatic_competition",
    ) -> AttentionFocus:
        previous = self.focus.candidate_ids
        ranked = sorted(
            (
                candidate
                for candidate in self.candidates.values()
                if candidate.status
                in {
                    AttentionCandidateStatus.AVAILABLE,
                    AttentionCandidateStatus.FOCUSED,
                }
            ),
            key=lambda item: (self._score(item, previous), item.candidate_id),
            reverse=True,
        )
        selected: list[AttentionCandidate] = []
        high_arousal = 0
        cap_applied = False
        for candidate in ranked:
            score = self._score(candidate, previous)
            if score < self.idle_threshold:
                continue
            is_high_arousal = candidate.arousal >= 0.75
            if is_high_arousal and high_arousal >= self.high_arousal_cap:
                cap_applied = True
                continue
            selected.append(candidate)
            high_arousal += int(is_high_arousal)
            if len(selected) == self.capacity:
                break
        reasons = [reason_code]
        if cap_applied:
            reasons.append("high_arousal_cap")
        if not selected:
            reasons.append("no_candidate_above_idle_threshold")
        elif any(item.candidate_id in previous for item in selected):
            reasons.append("focus_persistence")
        if tuple(item.candidate_id for item in selected) != previous:
            reasons.append("focus_switched")
        return self._set_focus(
            tuple(item.candidate_id for item in selected),
            action=AttentionAction.COMPETE if selected else AttentionAction.IDLE,
            reason_codes=tuple(reasons),
            provenance_refs=tuple(
                dict.fromkeys(ref for item in selected for ref in item.source_refs)
            ),
            event_id=event_id,
            event_sequence=event_sequence,
        )

    def refocus(
        self,
        candidate_ids: Iterable[str],
        *,
        reason_code: str,
        provenance_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> AttentionFocus:
        selected = tuple(dict.fromkeys(candidate_ids))
        if not selected or len(selected) > self.capacity:
            raise ValueError("deliberate refocus requires candidates within capacity")
        candidates = [self.get(identifier) for identifier in selected]
        if any(item.status == AttentionCandidateStatus.IGNORED for item in candidates):
            raise ValueError("ignored candidates cannot be refocused")
        if sum(item.arousal >= 0.75 for item in candidates) > self.high_arousal_cap:
            raise ValueError("deliberate refocus exceeds the high-arousal cap")
        return self._set_focus(
            selected,
            action=AttentionAction.REFOCUS,
            reason_codes=(reason_code, "deliberate_refocus"),
            provenance_refs=provenance_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )

    def defer(
        self,
        candidate_id: str,
        *,
        reason_code: str,
        provenance_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> AttentionFocus:
        current = self.get(candidate_id)
        self.candidates[candidate_id] = replace(
            current,
            status=AttentionCandidateStatus.DEFERRED,
            inhibition=min(1.0, current.inhibition + 0.35),
            revision=current.revision + 1,
        )
        remaining = tuple(
            item for item in self.focus.candidate_ids if item != candidate_id
        )
        return self._set_focus(
            remaining,
            action=AttentionAction.DEFER,
            reason_codes=(reason_code, "deliberately_deferred"),
            provenance_refs=provenance_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )

    def ignore(
        self,
        candidate_id: str,
        *,
        reason_code: str,
        provenance_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> AttentionFocus:
        current = self.get(candidate_id)
        self.candidates[candidate_id] = replace(
            current,
            status=AttentionCandidateStatus.IGNORED,
            inhibition=1.0,
            revision=current.revision + 1,
        )
        remaining = tuple(
            item for item in self.focus.candidate_ids if item != candidate_id
        )
        return self._set_focus(
            remaining,
            action=AttentionAction.IGNORE,
            reason_codes=(reason_code, "deliberately_ignored"),
            provenance_refs=provenance_refs,
            event_id=event_id,
            event_sequence=event_sequence,
        )

    def resume(self, candidate_id: str) -> AttentionCandidate:
        current = self.get(candidate_id)
        if current.status != AttentionCandidateStatus.DEFERRED:
            raise ValueError("only deferred attention candidates can be resumed")
        resumed = replace(
            current,
            status=AttentionCandidateStatus.AVAILABLE,
            inhibition=max(0.0, current.inhibition - 0.25),
            revision=current.revision + 1,
        )
        self.candidates[candidate_id] = resumed
        return resumed

    def get(self, candidate_id: str) -> AttentionCandidate:
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Unknown attention candidate: {candidate_id}")
        return candidate

    def list_candidates(self) -> list[AttentionCandidate]:
        return sorted(self.candidates.values(), key=lambda item: item.candidate_id)

    def focused_working_memory_refs(self) -> frozenset[str]:
        return frozenset(
            reference
            for identifier in self.focus.candidate_ids
            if (reference := self.candidates[identifier].working_memory_ref) is not None
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "high_arousal_cap": self.high_arousal_cap,
            "switch_cost": self.switch_cost,
            "idle_threshold": self.idle_threshold,
            "candidates": [item.to_json() for item in self.list_candidates()],
            "focus": _json_value(asdict(self.focus)),
            "history": [_json_value(asdict(item)) for item in self.history],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.candidates = {}
            self.focus = _idle_focus()
            self.history = []
            return
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported attention system schema version")
        candidates = [
            _candidate_from_json(item) for item in payload.get("candidates", [])
        ]
        if len(candidates) != len({item.candidate_id for item in candidates}):
            raise ValueError("Attention candidate identifiers must be unique")
        focus = _focus_from_json(payload.get("focus", {}))
        candidate_ids = {item.candidate_id for item in candidates}
        if not set(focus.candidate_ids).issubset(candidate_ids):
            raise ValueError("Attention focus references an unknown candidate")
        history = [_history_from_json(item) for item in payload.get("history", [])]
        self.candidates = {item.candidate_id: item for item in candidates}
        self.focus = focus
        self.history = history

    def _score(self, candidate: AttentionCandidate, focused: tuple[str, ...]) -> float:
        score = (
            0.24 * candidate.salience
            + 0.18 * candidate.drive
            + 0.18 * candidate.urgency
            + 0.10 * candidate.novelty
            + 0.10 * candidate.value_relevance
            + 0.10 * candidate.commitment_cost
            + 0.10 * candidate.persistence
            - 0.22 * candidate.habituation
            - 0.30 * candidate.inhibition
        )
        if candidate.candidate_id in focused:
            score += 0.16 + 0.10 * candidate.persistence
        elif focused:
            score -= self.switch_cost
        return score

    def _set_focus(
        self,
        candidate_ids: tuple[str, ...],
        *,
        action: AttentionAction,
        reason_codes: tuple[str, ...],
        provenance_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> AttentionFocus:
        if not reason_codes or not provenance_refs and candidate_ids:
            raise ValueError("attention changes require reasons and provenance")
        previous = self.focus.candidate_ids
        unfinished = tuple(
            identifier for identifier in previous if identifier not in candidate_ids
        )
        now = _now()
        for identifier, candidate in list(self.candidates.items()):
            if identifier in candidate_ids:
                self.candidates[identifier] = replace(
                    candidate,
                    status=AttentionCandidateStatus.FOCUSED,
                    focused_cycles=candidate.focused_cycles + 1,
                    habituation=min(1.0, candidate.habituation + 0.04),
                    inhibition=max(0.0, candidate.inhibition - 0.1),
                    revision=candidate.revision + 1,
                )
            elif candidate.status == AttentionCandidateStatus.FOCUSED:
                self.candidates[identifier] = replace(
                    candidate,
                    status=AttentionCandidateStatus.AVAILABLE,
                    unattended_cycles=candidate.unattended_cycles + 1,
                    inhibition=min(1.0, candidate.inhibition + 0.12),
                    revision=candidate.revision + 1,
                )
        switched = previous != candidate_ids
        applied_switch_cost = (
            self.switch_cost if switched and previous and candidate_ids else 0.0
        )
        focus = AttentionFocus(
            candidate_ids=candidate_ids,
            idle=not candidate_ids,
            unfinished_candidate_ids=unfinished,
            reason_codes=reason_codes,
            provenance_refs=provenance_refs,
            selected_at=now,
            event_id=event_id,
            event_sequence=event_sequence,
            revision=self.focus.revision + 1,
        )
        self.focus = focus
        self.history.append(
            AttentionHistoryEntry(
                history_id=f"attention-history-{uuid4()}",
                action=action,
                from_candidate_ids=previous,
                to_candidate_ids=candidate_ids,
                unfinished_candidate_ids=unfinished,
                reason_codes=reason_codes,
                provenance_refs=provenance_refs,
                switch_cost=applied_switch_cost,
                event_id=event_id,
                event_sequence=event_sequence,
                created_at=now,
            )
        )
        return focus


def _idle_focus() -> AttentionFocus:
    return AttentionFocus(
        candidate_ids=(),
        idle=True,
        unfinished_candidate_ids=(),
        reason_codes=("initial_idle",),
        provenance_refs=(),
        selected_at=_now(),
        event_id=None,
        event_sequence=None,
        revision=0,
    )


def _candidate_from_json(payload: dict[str, Any]) -> AttentionCandidate:
    data = dict(payload)
    data["source"] = AttentionSource(data["source"])
    data["status"] = AttentionCandidateStatus(data["status"])
    data["source_refs"] = tuple(data.get("source_refs", ()))
    return AttentionCandidate(**data)


def _focus_from_json(payload: dict[str, Any]) -> AttentionFocus:
    data = dict(payload)
    for name in (
        "candidate_ids",
        "unfinished_candidate_ids",
        "reason_codes",
        "provenance_refs",
    ):
        data[name] = tuple(data.get(name, ()))
    return AttentionFocus(**data)


def _history_from_json(payload: dict[str, Any]) -> AttentionHistoryEntry:
    data = dict(payload)
    data["action"] = AttentionAction(data["action"])
    for name in (
        "from_candidate_ids",
        "to_candidate_ids",
        "unfinished_candidate_ids",
        "reason_codes",
        "provenance_refs",
    ):
        data[name] = tuple(data.get(name, ()))
    return AttentionHistoryEntry(**data)


def _safe_refs(values: Iterable[str], name: str) -> None:
    for value in values:
        _safe_ref(value, name)


def _safe_ref(value: str, name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", value) is None:
        raise ValueError(f"{name} must be an opaque safe reference")


def _unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")


def _timestamp(value: str) -> None:
    if datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("attention timestamps must include a timezone")


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()
