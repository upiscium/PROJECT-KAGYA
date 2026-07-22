"""Versioned, evidence-bound subjective relationship state."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any, Iterable
from uuid import uuid4

from kagya.experience import AgencyAttribution, ExperienceRecord


class RelationshipEvidenceKind(StrEnum):
    INTERACTION = "interaction"
    RELIABLE = "reliable"
    RECIPROCAL = "reciprocal"
    BOUNDARY_RESPECTED = "boundary_respected"
    BOUNDARY_VIOLATION = "boundary_violation"
    COMMITMENT_KEPT = "commitment_kept"
    COMMITMENT_BREACHED = "commitment_breached"
    CONFLICT = "conflict"
    REPAIR = "repair"


@dataclass(frozen=True)
class PerceivedAttribute:
    value: str
    confidence: float
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("perceived attribute value must not be empty")
        _bounded(self.confidence, "perceived attribute confidence")


@dataclass(frozen=True)
class RelationshipAxes:
    trust: float = 0.5
    familiarity: float = 0.0
    closeness: float = 0.1
    caution: float = 0.25

    def __post_init__(self) -> None:
        for name in ("trust", "familiarity", "closeness", "caution"):
            _bounded(getattr(self, name), f"relationship {name}")


@dataclass(frozen=True)
class RelationshipEvidence:
    evidence_id: str
    kind: RelationshipEvidenceKind
    experience_id: str
    axis_signals: dict[str, float]
    reciprocity_signal: float
    uncertainty_signal: float
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported relationship evidence schema version")
        if not self.evidence_id or not self.experience_id:
            raise ValueError("relationship evidence identifiers must not be empty")
        if set(self.axis_signals) - {"trust", "familiarity", "closeness", "caution"}:
            raise ValueError("unsupported relationship evidence axis")
        for value in (
            *self.axis_signals.values(),
            self.reciprocity_signal,
            self.uncertainty_signal,
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(
                    "relationship evidence signals must be between -1 and 1"
                )


@dataclass(frozen=True)
class RelationshipRevision:
    revision_id: str
    from_revision: int
    to_revision: int
    reason_code: str
    evidence_refs: tuple[str, ...]
    changed_fields: tuple[str, ...]
    event_id: str | None
    event_sequence: int | None
    created_at: str

    def __post_init__(self) -> None:
        if self.to_revision != self.from_revision + 1:
            raise ValueError("relationship revisions must be sequential")
        if not self.reason_code or not self.evidence_refs:
            raise ValueError("relationship revisions require reason and evidence")


@dataclass(frozen=True)
class RelationshipState:
    relationship_id: str
    interlocutor_keys: tuple[str, ...]
    perceived_identity: PerceivedAttribute | None
    perceived_role: PerceivedAttribute | None
    axes: RelationshipAxes
    expectations: dict[str, PerceivedAttribute]
    boundaries: dict[str, PerceivedAttribute]
    other_values: dict[str, PerceivedAttribute]
    other_beliefs: dict[str, PerceivedAttribute]
    shared_experience_refs: tuple[str, ...]
    goal_refs: tuple[str, ...]
    commitment_refs: tuple[str, ...]
    unresolved_matter_refs: tuple[str, ...]
    reciprocity: float
    conflict_refs: tuple[str, ...]
    repair_refs: tuple[str, ...]
    confidence: float
    uncertainty: float
    evidence: tuple[RelationshipEvidence, ...]
    revision: int
    revisions: tuple[RelationshipRevision, ...]
    created_at: str
    updated_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported relationship schema version")
        if not self.relationship_id or not self.interlocutor_keys:
            raise ValueError("relationship requires an ID and interlocutor key")
        if len(self.interlocutor_keys) != len(set(self.interlocutor_keys)):
            raise ValueError("relationship interlocutor keys must be unique")
        for name in ("reciprocity", "confidence", "uncertainty"):
            _bounded(getattr(self, name), f"relationship {name}")

    def to_json(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class RelationshipInfluence:
    relationship_refs: tuple[str, ...] = ()
    threat: float = 0.0
    certainty: float = 0.5
    social_relevance: float = 0.0
    expected_reciprocity: float = 0.5
    unresolved_refs: tuple[str, ...] = ()
    goal_refs: tuple[str, ...] = ()
    commitment_refs: tuple[str, ...] = ()


class RelationshipStore:
    """Own stable interlocutor mappings and subjective relationship revisions."""

    SCHEMA_VERSION = 1
    MAX_AXIS_UPDATE = 0.08
    REQUIRED_CONSISTENT_EVIDENCE = 2

    def __init__(self) -> None:
        self.relationships: dict[str, RelationshipState] = {}
        self._interlocutor_index: dict[str, str] = {}
        self._experience_index: set[str] = set()

    def ensure_interlocutor(self, interlocutor_key: str) -> RelationshipState:
        if not interlocutor_key:
            raise ValueError("interlocutor key must not be empty")
        existing = self.for_interlocutor(interlocutor_key)
        if existing is not None:
            return existing
        now = _now()
        state = RelationshipState(
            relationship_id=f"relationship-{uuid4()}",
            interlocutor_keys=(interlocutor_key,),
            perceived_identity=None,
            perceived_role=None,
            axes=RelationshipAxes(),
            expectations={},
            boundaries={},
            other_values={},
            other_beliefs={},
            shared_experience_refs=(),
            goal_refs=(),
            commitment_refs=(),
            unresolved_matter_refs=(),
            reciprocity=0.5,
            conflict_refs=(),
            repair_refs=(),
            confidence=0.1,
            uncertainty=0.9,
            evidence=(),
            revision=0,
            revisions=(),
            created_at=now,
            updated_at=now,
        )
        self.relationships[state.relationship_id] = state
        self._interlocutor_index[interlocutor_key] = state.relationship_id
        return state

    def get(self, relationship_id: str) -> RelationshipState:
        state = self.relationships.get(relationship_id)
        if state is None:
            raise ValueError(f"Unknown relationship: {relationship_id}")
        return state

    def for_interlocutor(self, interlocutor_key: str) -> RelationshipState | None:
        relationship_id = self._interlocutor_index.get(interlocutor_key)
        return None if relationship_id is None else self.relationships[relationship_id]

    def list_relationships(self) -> list[RelationshipState]:
        return sorted(
            self.relationships.values(), key=lambda item: item.relationship_id
        )

    def observe_experience(
        self,
        experience: ExperienceRecord,
        *,
        active_commitment_refs: tuple[str, ...] = (),
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> tuple[RelationshipState, ...]:
        if experience.experience_id in self._experience_index:
            return tuple(
                state
                for key in experience.interlocutor_ids
                if (state := self.for_interlocutor(key)) is not None
            )
        updated: list[RelationshipState] = []
        for interlocutor_key in experience.interlocutor_ids:
            current = self.ensure_interlocutor(interlocutor_key)
            evidence = evidence_from_experience(experience)
            updated.append(
                self._apply_evidence(
                    current,
                    evidence,
                    active_goal_refs=experience.active_goal_refs,
                    active_commitment_refs=active_commitment_refs,
                    unresolved=experience.unresolved_tension >= 0.6,
                    event_id=event_id,
                    event_sequence=event_sequence,
                )
            )
        self._experience_index.add(experience.experience_id)
        return tuple(updated)

    def influence(self, interlocutor_keys: Iterable[str]) -> RelationshipInfluence:
        states = [
            state
            for key in interlocutor_keys
            if (state := self.for_interlocutor(key)) is not None
        ]
        if not states:
            return RelationshipInfluence()
        count = len(states)
        caution = sum(item.axes.caution for item in states) / count
        trust = sum(item.axes.trust for item in states) / count
        return RelationshipInfluence(
            relationship_refs=tuple(
                f"relationship:{item.relationship_id}@{item.revision}"
                for item in states
            ),
            threat=_clamp(max(0.0, caution - 0.5 * trust)),
            certainty=_clamp(sum((1.0 - item.uncertainty) for item in states) / count),
            social_relevance=_clamp(
                sum(max(item.axes.familiarity, item.axes.closeness) for item in states)
                / count
            ),
            expected_reciprocity=sum(item.reciprocity for item in states) / count,
            unresolved_refs=_unique(
                ref for item in states for ref in item.unresolved_matter_refs
            ),
            goal_refs=_unique(ref for item in states for ref in item.goal_refs),
            commitment_refs=_unique(
                ref for item in states for ref in item.commitment_refs
            ),
        )

    def attach_alias(
        self,
        relationship_id: str,
        interlocutor_key: str,
        *,
        evidence_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> RelationshipState:
        if len(set(evidence_refs)) < self.REQUIRED_CONSISTENT_EVIDENCE:
            raise ValueError(
                "alias merge requires at least two independent evidence references"
            )
        mapped = self._interlocutor_index.get(interlocutor_key)
        if mapped is not None and mapped != relationship_id:
            raise ValueError("interlocutor key already belongs to another relationship")
        current = self.get(relationship_id)
        if interlocutor_key in current.interlocutor_keys:
            return current
        return self._revise(
            current,
            reason_code="corroborated_alias_merge",
            evidence_refs=evidence_refs,
            changed_fields=("interlocutor_keys",),
            event_id=event_id,
            event_sequence=event_sequence,
            interlocutor_keys=(*current.interlocutor_keys, interlocutor_key),
        )

    def split_alias(
        self,
        relationship_id: str,
        interlocutor_key: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> RelationshipState:
        current = self.get(relationship_id)
        if interlocutor_key not in current.interlocutor_keys:
            raise ValueError("interlocutor key is not mapped to relationship")
        if len(current.interlocutor_keys) < 2:
            raise ValueError("cannot split the only interlocutor key")
        remaining = tuple(
            key for key in current.interlocutor_keys if key != interlocutor_key
        )
        self._revise(
            current,
            reason_code=reason,
            evidence_refs=evidence_refs,
            changed_fields=("interlocutor_keys",),
            event_id=event_id,
            event_sequence=event_sequence,
            interlocutor_keys=remaining,
        )
        self._interlocutor_index.pop(interlocutor_key, None)
        return self.ensure_interlocutor(interlocutor_key)

    def correct(
        self,
        relationship_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        axes: dict[str, float] | None = None,
        perceived_identity: PerceivedAttribute | None = None,
        perceived_role: PerceivedAttribute | None = None,
        expectations: dict[str, PerceivedAttribute] | None = None,
        boundaries: dict[str, PerceivedAttribute] | None = None,
        other_values: dict[str, PerceivedAttribute] | None = None,
        other_beliefs: dict[str, PerceivedAttribute] | None = None,
        reciprocity: float | None = None,
        uncertainty: float | None = None,
        commitment_refs: tuple[str, ...] | None = None,
        unresolved_matter_refs: tuple[str, ...] | None = None,
        conflict_refs: tuple[str, ...] | None = None,
        repair_refs: tuple[str, ...] | None = None,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> RelationshipState:
        current = self.get(relationship_id)
        updates: dict[str, Any] = {}
        if axes is not None:
            unknown = set(axes) - {"trust", "familiarity", "closeness", "caution"}
            if unknown:
                raise ValueError(f"Unsupported relationship axis: {sorted(unknown)[0]}")
            updates["axes"] = replace(current.axes, **axes)
        for name, value in (
            ("perceived_identity", perceived_identity),
            ("perceived_role", perceived_role),
            ("expectations", expectations),
            ("boundaries", boundaries),
            ("other_values", other_values),
            ("other_beliefs", other_beliefs),
            ("reciprocity", reciprocity),
            ("uncertainty", uncertainty),
            ("commitment_refs", commitment_refs),
            ("unresolved_matter_refs", unresolved_matter_refs),
            ("conflict_refs", conflict_refs),
            ("repair_refs", repair_refs),
        ):
            if value is not None:
                updates[name] = value
        if not updates:
            raise ValueError("relationship correction did not specify changes")
        return self._revise(
            current,
            reason_code=reason,
            evidence_refs=evidence_refs,
            changed_fields=tuple(updates),
            event_id=event_id,
            event_sequence=event_sequence,
            **updates,
        )

    def link_commitment(
        self, interlocutor_key: str, commitment_ref: str, *, unresolved: bool = True
    ) -> RelationshipState:
        current = self.ensure_interlocutor(interlocutor_key)
        evidence_refs = (commitment_ref,)
        updates: dict[str, Any] = {
            "commitment_refs": _unique((*current.commitment_refs, commitment_ref))
        }
        if unresolved:
            updates["unresolved_matter_refs"] = _unique(
                (*current.unresolved_matter_refs, commitment_ref)
            )
        return self._revise(
            current,
            reason_code="relationship_commitment_linked",
            evidence_refs=evidence_refs,
            changed_fields=tuple(updates),
            event_id=None,
            event_sequence=None,
            **updates,
        )

    def transition_commitment(
        self, commitment_ref: str, *, status: str, evidence_ref: str
    ) -> tuple[RelationshipState, ...]:
        updated: list[RelationshipState] = []
        for current in tuple(self.relationships.values()):
            if commitment_ref not in current.commitment_refs:
                continue
            unresolved = tuple(
                ref
                for ref in current.unresolved_matter_refs
                if ref != commitment_ref and not ref.startswith(f"{commitment_ref}:")
            )
            changes: dict[str, Any] = {"unresolved_matter_refs": unresolved}
            if status == "breached":
                changes["unresolved_matter_refs"] = _unique(
                    (*unresolved, evidence_ref)
                )
                changes["conflict_refs"] = _unique(
                    (*current.conflict_refs, evidence_ref)
                )
            elif status == "fulfilled":
                changes["repair_refs"] = _unique((*current.repair_refs, evidence_ref))
            elif status == "repaired":
                changes["unresolved_matter_refs"] = unresolved
                changes["repair_refs"] = _unique((*current.repair_refs, evidence_ref))
            updated.append(
                self._revise(
                    current,
                    reason_code=f"relationship_commitment_{status}",
                    evidence_refs=(evidence_ref,),
                    changed_fields=tuple(changes),
                    event_id=None,
                    event_sequence=None,
                    **changes,
                )
            )
        return tuple(updated)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "relationships": [item.to_json() for item in self.list_relationships()],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.relationships = {}
            self._interlocutor_index = {}
            self._experience_index = set()
            return
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported relationship store schema version")
        states = [
            _state_from_json(item)
            for item in payload.get("relationships", [])
            if isinstance(item, dict)
        ]
        relationships = {item.relationship_id: item for item in states}
        if len(relationships) != len(states):
            raise ValueError("relationship identifiers must be unique")
        index: dict[str, str] = {}
        for state in states:
            for key in state.interlocutor_keys:
                if key in index:
                    raise ValueError("interlocutor key maps to multiple relationships")
                index[key] = state.relationship_id
        self.relationships = relationships
        self._interlocutor_index = index
        self._experience_index = {
            item.experience_id for state in states for item in state.evidence
        }

    def _apply_evidence(
        self,
        current: RelationshipState,
        evidence: RelationshipEvidence,
        *,
        active_goal_refs: tuple[str, ...],
        active_commitment_refs: tuple[str, ...],
        unresolved: bool,
        event_id: str | None,
        event_sequence: int | None,
    ) -> RelationshipState:
        axis_updates: dict[str, float] = {}
        for axis, signal in evidence.axis_signals.items():
            if signal == 0.0 or not _corroborated(current.evidence, axis, signal):
                continue
            bounded = max(-self.MAX_AXIS_UPDATE, min(self.MAX_AXIS_UPDATE, signal))
            axis_updates[axis] = _clamp(getattr(current.axes, axis) + bounded)
        reciprocity = current.reciprocity
        if evidence.reciprocity_signal and _corroborated(
            current.evidence, "reciprocity", evidence.reciprocity_signal
        ):
            reciprocity = _clamp(
                reciprocity
                + max(
                    -self.MAX_AXIS_UPDATE,
                    min(self.MAX_AXIS_UPDATE, evidence.reciprocity_signal),
                )
            )
        uncertainty = current.uncertainty
        if evidence.uncertainty_signal and _corroborated(
            current.evidence, "uncertainty", evidence.uncertainty_signal
        ):
            uncertainty = _clamp(
                uncertainty
                + max(
                    -self.MAX_AXIS_UPDATE,
                    min(self.MAX_AXIS_UPDATE, evidence.uncertainty_signal),
                )
            )
        experience_ref = f"experience:{evidence.experience_id}"
        conflict_refs = current.conflict_refs
        repair_refs = current.repair_refs
        if evidence.kind == RelationshipEvidenceKind.CONFLICT:
            conflict_refs = _unique((*conflict_refs, experience_ref))
        elif evidence.kind == RelationshipEvidenceKind.REPAIR:
            repair_refs = _unique((*repair_refs, experience_ref))
        unresolved_refs = current.unresolved_matter_refs
        if unresolved:
            unresolved_refs = _unique((*unresolved_refs, experience_ref))
        elif evidence.kind == RelationshipEvidenceKind.REPAIR and unresolved_refs:
            unresolved_refs = unresolved_refs[:-1]
        updates: dict[str, Any] = {
            "axes": replace(current.axes, **axis_updates),
            "shared_experience_refs": _unique(
                (*current.shared_experience_refs, experience_ref)
            ),
            "goal_refs": _unique((*current.goal_refs, *active_goal_refs)),
            "commitment_refs": _unique(
                (*current.commitment_refs, *active_commitment_refs)
            ),
            "unresolved_matter_refs": unresolved_refs,
            "reciprocity": reciprocity,
            "conflict_refs": conflict_refs,
            "repair_refs": repair_refs,
            "confidence": _clamp(current.confidence + (0.03 if axis_updates else 0.01)),
            "uncertainty": uncertainty,
            "evidence": (*current.evidence, evidence),
        }
        return self._revise(
            current,
            reason_code="experience_evidence_integrated",
            evidence_refs=(experience_ref,),
            changed_fields=tuple(updates),
            event_id=event_id,
            event_sequence=event_sequence,
            **updates,
        )

    def _revise(
        self,
        current: RelationshipState,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        changed_fields: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
        **updates: Any,
    ) -> RelationshipState:
        if not reason_code or not evidence_refs:
            raise ValueError("relationship revision requires reason and evidence")
        now = _now()
        revision = RelationshipRevision(
            revision_id=f"relationship-revision-{uuid4()}",
            from_revision=current.revision,
            to_revision=current.revision + 1,
            reason_code=reason_code,
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
            changed_fields=changed_fields,
            event_id=event_id,
            event_sequence=event_sequence,
            created_at=now,
        )
        updated = replace(
            current,
            **updates,
            revision=revision.to_revision,
            revisions=(*current.revisions, revision),
            updated_at=now,
        )
        self.relationships[current.relationship_id] = updated
        for key in updated.interlocutor_keys:
            self._interlocutor_index[key] = updated.relationship_id
        return updated


def evidence_from_experience(experience: ExperienceRecord) -> RelationshipEvidence:
    appraisal = experience.appraisal
    kind = RelationshipEvidenceKind.INTERACTION
    if appraisal.threat >= 0.6 or appraisal.goal_progress <= -0.5:
        kind = RelationshipEvidenceKind.CONFLICT
    elif appraisal.goal_progress >= 0.4 and appraisal.valence >= 0.2:
        kind = RelationshipEvidenceKind.REPAIR
    trust = 0.0
    if appraisal.threat >= 0.5 or appraisal.valence <= -0.3:
        trust = -0.06
    elif appraisal.valence >= 0.2 and appraisal.threat <= 0.25:
        trust = 0.04
    closeness = (
        0.04 if appraisal.social_relevance >= 0.5 and appraisal.valence >= 0.2 else 0.0
    )
    caution = (
        0.08
        if appraisal.threat >= 0.5
        else (-0.03 if appraisal.certainty >= 0.7 else 0.0)
    )
    reciprocity = (
        0.04
        if experience.agency_attribution == AgencyAttribution.SHARED
        and appraisal.goal_progress >= 0.0
        else 0.0
    )
    return RelationshipEvidence(
        evidence_id=f"relationship-evidence-{uuid4()}",
        kind=kind,
        experience_id=experience.experience_id,
        axis_signals={
            "trust": trust,
            "familiarity": 0.05,
            "closeness": closeness,
            "caution": caution,
        },
        reciprocity_signal=reciprocity,
        uncertainty_signal=-0.04 if appraisal.certainty >= 0.65 else 0.03,
        created_at=_now(),
    )


def _corroborated(
    evidence: tuple[RelationshipEvidence, ...], axis: str, signal: float
) -> bool:
    prior = []
    for item in evidence:
        if axis == "reciprocity":
            value = item.reciprocity_signal
        elif axis == "uncertainty":
            value = item.uncertainty_signal
        else:
            value = item.axis_signals.get(axis, 0.0)
        if value and (value > 0) == (signal > 0):
            prior.append(value)
    return len(prior) >= RelationshipStore.REQUIRED_CONSISTENT_EVIDENCE - 1


def _state_from_json(payload: dict[str, Any]) -> RelationshipState:
    data = dict(payload)
    data["interlocutor_keys"] = tuple(data.get("interlocutor_keys", ()))
    data["axes"] = RelationshipAxes(**data.get("axes", {}))
    for name in (
        "expectations",
        "boundaries",
        "other_values",
        "other_beliefs",
    ):
        data[name] = {
            str(key): _attribute_from_json(value)
            for key, value in data.get(name, {}).items()
        }
    for name in ("perceived_identity", "perceived_role"):
        if data.get(name) is not None:
            data[name] = _attribute_from_json(data[name])
    for name in (
        "shared_experience_refs",
        "goal_refs",
        "commitment_refs",
        "unresolved_matter_refs",
        "conflict_refs",
        "repair_refs",
    ):
        data[name] = tuple(data.get(name, ()))
    data["evidence"] = tuple(
        RelationshipEvidence(
            **{
                **item,
                "kind": RelationshipEvidenceKind(item["kind"]),
                "axis_signals": {
                    str(key): float(value)
                    for key, value in item.get("axis_signals", {}).items()
                },
            }
        )
        for item in data.get("evidence", ())
    )
    data["revisions"] = tuple(
        RelationshipRevision(
            **{
                **item,
                "evidence_refs": tuple(item.get("evidence_refs", ())),
                "changed_fields": tuple(item.get("changed_fields", ())),
            }
        )
        for item in data.get("revisions", ())
    )
    return RelationshipState(**data)


def _attribute_from_json(value: dict[str, Any]) -> PerceivedAttribute:
    return PerceivedAttribute(
        value=str(value["value"]),
        confidence=float(value["confidence"]),
        evidence_refs=tuple(value.get("evidence_refs", ())),
    )


def _bounded(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
