"""Versioned first-person experience records without private free-form thought."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
import re
from typing import Any, Iterable
from uuid import uuid4

from kagya.cognition import AppraisalResult
from kagya.identity import IdentityOrigin, identity_origin_from_json


class AgencyAttribution(StrEnum):
    SELF = "self"
    OTHER = "other"
    SHARED = "shared"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExperienceAppraisal:
    valence: float
    arousal: float
    novelty: float | None
    novelty_valid: bool
    goal_progress: float
    threat: float
    controllability: float
    certainty: float
    social_relevance: float
    effort_cost: float
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _bounded(self.valence, "appraisal valence", -1.0, 1.0)
        for name in (
            "arousal",
            "threat",
            "controllability",
            "certainty",
            "social_relevance",
            "effort_cost",
        ):
            _bounded(getattr(self, name), name, 0.0, 1.0)
        _bounded(self.goal_progress, "goal progress", -1.0, 1.0)
        if self.novelty is not None:
            _bounded(self.novelty, "novelty", 0.0, 1.0)
        _safe_codes(self.reason_codes, "appraisal reason")


@dataclass(frozen=True)
class ExperienceRevision:
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
        _safe_ref(self.revision_id, "revision ID")
        _safe_ref(self.reason_code, "revision reason")
        _safe_codes(self.evidence_refs, "revision evidence reference")
        _safe_codes(self.changed_fields, "revision changed field")
        if self.from_revision < 0 or self.to_revision != self.from_revision + 1:
            raise ValueError("experience revisions must be sequential")


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    source_event_id: str | None
    source_event_sequence: int | None
    external_observation_refs: tuple[str, ...]
    subject_action_refs: tuple[str, ...]
    identity_origin: IdentityOrigin
    context_id: str
    interlocutor_ids: tuple[str, ...]
    situation_codes: tuple[str, ...]
    interpretation_codes: tuple[str, ...]
    self_relevance: float
    appraisal: ExperienceAppraisal
    subjective_salience: float
    familiarity: float
    agency_attribution: AgencyAttribution
    prediction_error: float | None
    value_revision_refs: dict[str, int]
    active_goal_refs: tuple[str, ...]
    self_model_revision: int
    unresolved_tension: float
    autobiographical_importance: float
    result_refs: dict[str, tuple[str, ...]]
    created_at: str
    updated_at: str
    revision: int = 0
    revisions: tuple[ExperienceRevision, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported experience schema version: {self.schema_version}"
            )
        _safe_ref(self.experience_id, "experience ID")
        _safe_ref(self.context_id, "experience context ID")
        for values, name in (
            (self.external_observation_refs, "observation reference"),
            (self.subject_action_refs, "subject action reference"),
            (self.interlocutor_ids, "interlocutor ID"),
            (self.situation_codes, "situation code"),
            (self.interpretation_codes, "interpretation code"),
            (self.active_goal_refs, "active goal reference"),
        ):
            _safe_codes(values, name)
        for name in (
            "self_relevance",
            "subjective_salience",
            "familiarity",
            "unresolved_tension",
            "autobiographical_importance",
        ):
            _bounded(getattr(self, name), name, 0.0, 1.0)
        if self.prediction_error is not None and (
            not math.isfinite(self.prediction_error) or self.prediction_error < 0.0
        ):
            raise ValueError("prediction error must be finite and non-negative")
        if self.self_model_revision < 0 or self.revision < 0:
            raise ValueError("experience revisions must not be negative")
        if any(not key or value < 0 for key, value in self.value_revision_refs.items()):
            raise ValueError("value revision references must be non-negative")
        for kind, refs in self.result_refs.items():
            if kind not in {
                "memory",
                "belief",
                "value",
                "goal",
                "self_model",
                "decision",
                "narrative",
                "relationship",
            }:
                raise ValueError(f"Unsupported experience result kind: {kind}")
            _safe_codes(refs, "experience result reference")
        for timestamp in (self.created_at, self.updated_at):
            if datetime.fromisoformat(timestamp).tzinfo is None:
                raise ValueError("experience timestamps must include a timezone")

    def to_json(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class ExperienceStore:
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self.records: dict[str, ExperienceRecord] = {}
        self._event_index: dict[str, str] = {}
        self._observation_index: dict[str, str] = {}

    def integrate(self, record: ExperienceRecord) -> ExperienceRecord:
        existing_id = (
            self._event_index.get(record.source_event_id)
            if record.source_event_id is not None
            else None
        )
        if existing_id is None:
            existing_id = next(
                (
                    self._observation_index[reference]
                    for reference in record.external_observation_refs
                    if reference in self._observation_index
                ),
                None,
            )
        if existing_id is not None:
            return self.records[existing_id]
        if record.experience_id in self.records:
            raise ValueError(f"Experience already exists: {record.experience_id}")
        self.records[record.experience_id] = record
        self._index(record)
        return record

    def get(self, experience_id: str) -> ExperienceRecord:
        record = self.records.get(experience_id)
        if record is None:
            raise ValueError(f"Unknown experience: {experience_id}")
        return record

    def list_records(self) -> list[ExperienceRecord]:
        return sorted(
            self.records.values(),
            key=lambda item: (item.created_at, item.experience_id),
        )

    def latest_for_context(self, context_id: str) -> ExperienceRecord | None:
        candidates = [
            record
            for record in self.records.values()
            if record.context_id == context_id
        ]
        return max(
            candidates,
            key=lambda item: (item.created_at, item.experience_id),
            default=None,
        )

    def revise_appraisal(
        self,
        experience_id: str,
        *,
        appraisal: ExperienceAppraisal,
        reason_code: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> ExperienceRecord:
        if not evidence_refs:
            raise ValueError("experience reassessment requires evidence references")
        current = self.get(experience_id)
        metrics = _experience_metrics(appraisal, current.prediction_error)
        revision = _revision(
            current,
            reason_code,
            evidence_refs,
            (
                "appraisal",
                "self_relevance",
                "subjective_salience",
                "familiarity",
                "agency_attribution",
                "unresolved_tension",
                "autobiographical_importance",
            ),
            event_id,
            event_sequence,
        )
        updated = replace(
            current,
            appraisal=appraisal,
            **metrics,
            revision=revision.to_revision,
            revisions=(*current.revisions, revision),
            updated_at=revision.created_at,
        )
        self.records[experience_id] = updated
        return updated

    def link_result(
        self,
        experience_id: str,
        *,
        kind: str,
        reference: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> ExperienceRecord:
        current = self.get(experience_id)
        if not evidence_refs:
            raise ValueError("experience result links require evidence references")
        if kind not in {
            "memory",
            "belief",
            "value",
            "goal",
            "self_model",
            "decision",
            "narrative",
            "relationship",
        }:
            raise ValueError(f"Unsupported experience result kind: {kind}")
        _safe_ref(reference, "experience result reference")
        existing = current.result_refs.get(kind, ())
        if reference in existing:
            return current
        revision = _revision(
            current,
            f"link_{kind}",
            evidence_refs,
            ("result_refs",),
            event_id,
            event_sequence,
        )
        result_refs = dict(current.result_refs)
        result_refs[kind] = (*existing, reference)
        updated = replace(
            current,
            result_refs=result_refs,
            revision=revision.to_revision,
            revisions=(*current.revisions, revision),
            updated_at=revision.created_at,
        )
        self.records[experience_id] = updated
        return updated

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "records": [record.to_json() for record in self.list_records()],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.records = {}
            self._event_index = {}
            self._observation_index = {}
            return
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported experience store schema version: {payload.get('schema_version')}"
            )
        records = [
            _experience_from_json(item)
            for item in payload.get("records", [])
            if isinstance(item, dict)
        ]
        if len(records) != len({record.experience_id for record in records}):
            raise ValueError("Experience identifiers must be unique")
        self.records = {record.experience_id: record for record in records}
        self._event_index = {}
        self._observation_index = {}
        for record in records:
            self._index(record)

    def _index(self, record: ExperienceRecord) -> None:
        if record.source_event_id is not None:
            existing = self._event_index.get(record.source_event_id)
            if existing is not None and existing != record.experience_id:
                raise ValueError("Multiple experiences reference the same source event")
            self._event_index[record.source_event_id] = record.experience_id
        for reference in record.external_observation_refs:
            existing = self._observation_index.get(reference)
            if existing is not None and existing != record.experience_id:
                raise ValueError("Multiple experiences reference the same observation")
            self._observation_index[reference] = record.experience_id


def build_chat_experience(
    *,
    source_event_id: str | None,
    source_event_sequence: int | None,
    episode_id: str,
    identity_origin: IdentityOrigin,
    context_id: str,
    interlocutor_ids: tuple[str, ...],
    appraisal: AppraisalResult,
    valence: float,
    arousal: float,
    prediction_error: float | None,
    value_revision_refs: dict[str, int],
    active_goal_refs: tuple[str, ...],
    self_model_revision: int,
) -> ExperienceRecord:
    structured_appraisal = ExperienceAppraisal(
        valence=valence,
        arousal=arousal,
        novelty=appraisal.novelty,
        novelty_valid=appraisal.novelty_valid,
        goal_progress=appraisal.goal_progress,
        threat=appraisal.threat,
        controllability=appraisal.controllability,
        certainty=appraisal.certainty,
        social_relevance=appraisal.social_relevance,
        effort_cost=appraisal.effort_cost,
        reason_codes=appraisal.reasons,
    )
    metrics = _experience_metrics(structured_appraisal, prediction_error)
    now = _now()
    situation_codes = ["interaction_observed"]
    if interlocutor_ids:
        situation_codes.append("interpersonal_context")
    if appraisal.novelty_valid and appraisal.novelty is not None:
        situation_codes.append(
            "novel_situation" if appraisal.novelty >= 0.6 else "familiar_situation"
        )
    else:
        situation_codes.append("novelty_uncertain")
    interpretation_codes = list(appraisal.reasons)
    if active_goal_refs:
        interpretation_codes.append("active_goal_context")
    return ExperienceRecord(
        experience_id=f"experience-{uuid4()}",
        source_event_id=source_event_id,
        source_event_sequence=source_event_sequence,
        external_observation_refs=(f"episode:{episode_id}:input",),
        subject_action_refs=(f"episode:{episode_id}:response",),
        identity_origin=identity_origin,
        context_id=context_id,
        interlocutor_ids=interlocutor_ids,
        situation_codes=tuple(situation_codes),
        interpretation_codes=tuple(dict.fromkeys(interpretation_codes)),
        appraisal=structured_appraisal,
        prediction_error=prediction_error,
        value_revision_refs=dict(value_revision_refs),
        active_goal_refs=active_goal_refs,
        self_model_revision=self_model_revision,
        result_refs={"memory": (f"episode:{episode_id}",)},
        created_at=now,
        updated_at=now,
        **metrics,
    )


def _experience_metrics(
    appraisal: ExperienceAppraisal, prediction_error: float | None
) -> dict[str, Any]:
    novelty = (
        appraisal.novelty
        if appraisal.novelty_valid and appraisal.novelty is not None
        else 0.0
    )
    affective_intensity = 0.6 * appraisal.arousal + 0.4 * abs(appraisal.valence)
    self_relevance = max(
        appraisal.social_relevance,
        appraisal.threat,
        abs(appraisal.goal_progress),
    )
    unresolved_tension = max(
        appraisal.threat,
        max(0.0, -appraisal.goal_progress),
        1.0 - appraisal.certainty,
    )
    normalized_prediction_error = (
        0.0 if prediction_error is None else 1.0 - math.exp(-max(0.0, prediction_error))
    )
    subjective_salience = _clamp(
        0.3 * novelty
        + 0.25 * affective_intensity
        + 0.2 * self_relevance
        + 0.15 * unresolved_tension
        + 0.1 * normalized_prediction_error
    )
    autobiographical_importance = _clamp(
        0.45 * self_relevance
        + 0.35 * subjective_salience
        + 0.2 * max(abs(appraisal.goal_progress), appraisal.social_relevance)
    )
    if appraisal.controllability >= 0.7:
        agency = AgencyAttribution.SELF
    elif appraisal.controllability >= 0.35:
        agency = AgencyAttribution.SHARED
    else:
        agency = AgencyAttribution.OTHER
    return {
        "self_relevance": self_relevance,
        "subjective_salience": subjective_salience,
        "familiarity": 1.0 - novelty if appraisal.novelty_valid else 0.5,
        "agency_attribution": agency,
        "unresolved_tension": unresolved_tension,
        "autobiographical_importance": autobiographical_importance,
    }


def _revision(
    current: ExperienceRecord,
    reason_code: str,
    evidence_refs: tuple[str, ...],
    changed_fields: tuple[str, ...],
    event_id: str | None,
    event_sequence: int | None,
) -> ExperienceRevision:
    return ExperienceRevision(
        revision_id=f"experience-revision-{uuid4()}",
        from_revision=current.revision,
        to_revision=current.revision + 1,
        reason_code=reason_code,
        evidence_refs=evidence_refs,
        changed_fields=changed_fields,
        event_id=event_id,
        event_sequence=event_sequence,
        created_at=_now(),
    )


def _experience_from_json(payload: dict[str, Any]) -> ExperienceRecord:
    data = dict(payload)
    data["identity_origin"] = identity_origin_from_json(data.get("identity_origin"))
    appraisal = dict(data["appraisal"])
    appraisal["reason_codes"] = tuple(appraisal.get("reason_codes", ()))
    data["appraisal"] = ExperienceAppraisal(**appraisal)
    for name in (
        "external_observation_refs",
        "subject_action_refs",
        "interlocutor_ids",
        "situation_codes",
        "interpretation_codes",
        "active_goal_refs",
    ):
        data[name] = tuple(data.get(name, ()))
    data["value_revision_refs"] = {
        str(key): int(value)
        for key, value in data.get("value_revision_refs", {}).items()
    }
    data["result_refs"] = {
        str(key): tuple(value) for key, value in data.get("result_refs", {}).items()
    }
    data["agency_attribution"] = AgencyAttribution(data["agency_attribution"])
    data["revisions"] = tuple(
        ExperienceRevision(
            **{
                **item,
                "evidence_refs": tuple(item.get("evidence_refs", ())),
                "changed_fields": tuple(item.get("changed_fields", ())),
            }
        )
        for item in data.get("revisions", ())
    )
    return ExperienceRecord(**data)


def _safe_codes(values: Iterable[str], name: str) -> None:
    for value in values:
        _safe_ref(value, name)


def _safe_ref(value: str, name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9._:@/-]{1,200}", value) is None:
        raise ValueError(f"{name} must be an opaque safe reference")


def _bounded(value: float, name: str, minimum: float, maximum: float) -> None:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


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
